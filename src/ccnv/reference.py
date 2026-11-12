"""Cohort validation and vectorized Schema v2.0.0 reference construction."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Final

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from ccnv.features.cohort import (
    Cohort,
    SexInference,
    SexLabel,
    build_fisher_weighted_cohort,
    cohort_from_fasta,
    compute_bin_exposure,
    estimate_size_factors,
    infer_germline_sex_2d,
    load_and_assemble_cohort,
    mask_low_coverage_per_contig,
)
from ccnv.features.covariates import (
    build_exogenous_covariate_basis,
    covariate_features,
    fit_poisson_covariate_trends,
    mask_covariates,
    validate_min_mappability,
)
from ccnv.features.prenatal import fit_prenatal_models
from ccnv.io.parquet import write_reference_parquet
from ccnv.io.samplesheet import load_samplesheet
from ccnv.io.tracks import (
    load_gc_track,
    load_mappability_track,
    read_fasta_contigs,
)
from ccnv.models import (
    BRANCH_AUTOSOMAL,
    BRANCH_FEMALE_SEX,
    BRANCH_MALE_SEX,
    NUM_PROTOTYPES_PER_BIN,
    AnalysisMode,
    BranchFitArrays,
    CovariateError,
    ReferenceConfig,
    ReferenceError,
    ReferenceMetadata,
    SamplesheetEntry,
    assemble_reference_table,
    parquet_path,
)
from ccnv.qc.reference import run_reference_qc_on_table
from ccnv.stats.neighbors import (
    build_whitened_subspace_features,
    calibrate_prototype_dispersion,
    fit_prototype_graph,
)
from ccnv.stats.pca import fit_whitened_pca, reconstruct_whitened_pca_expected

DEFAULT_SEX_CONTIGS: Final[tuple[str, ...]] = ("X", "Y", "chrX", "chrY")
DEFAULT_SEX_SIGNAL_CONTIGS: Final[tuple[str, ...]] = ("chrY", "Y")

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_SEX_CONTIGS",
    "DEFAULT_SEX_SIGNAL_CONTIGS",
    "ReferenceError",
    "build_reference_table",
    "run",
    "run_reference",
]


def run(config: ReferenceConfig) -> Path:
    """Validate a cohort and write a versioned Schema v2.0.0 reference Parquet file."""
    _validate_config(config)
    if config.samplesheet_path is not None:
        config = replace(config, entries=load_samplesheet(config.samplesheet_path))
    if config.fasta is not None:
        if config.bin_size is None:
            raise ReferenceError("--fasta requires --binsize")
        contigs = read_fasta_contigs(config.fasta, all_contigs=config.all_contigs)
        cohort = cohort_from_fasta(
            contigs,
            config.bin_size,
            fasta_path=config.fasta,
        )
        config = (
            replace(config, sex_contigs=())
            if (config.sex_contigs is None or config.sex_contigs == DEFAULT_SEX_CONTIGS)
            else config
        )
    else:
        cohort = load_and_assemble_cohort(
            config.entries,
            target_bin_size=config.bin_size,
            threads=config.threads,
            max_batch_memory_mb=config.max_batch_memory_mb,
        )
    table, path = build_reference_table(cohort, config), parquet_path(config.prefix)
    logger.debug("writing reference parquet %s (%d rows)", path, table.num_rows)
    write_reference_parquet(table, path)
    logger.debug(
        "wrote reference parquet %s (%.1f KB, %d rows)",
        path,
        path.stat().st_size / 1024,
        table.num_rows,
    )
    if config.qc_report:
        run_reference_qc_on_table(
            table,
            prefix=config.prefix,
            reference_path=path,
            write_json=True,
            emit_stderr=True,
        )
    return path


def build_reference_table(
    cohort: Cohort,
    config: ReferenceConfig,
) -> pa.Table:
    """Fit autosomal and optional sex-contig branches into one Schema v2.0.0 table."""
    is_flat = config.fasta is not None
    if not is_flat and cohort.counts.shape[1] < config.min_cohort_size:
        raise ReferenceError(
            f"provide at least {config.min_cohort_size} samples to build a reference"
        )
    try:
        validate_min_mappability(config.min_mappability)
    except CovariateError as err:
        raise ReferenceError(str(err)) from err

    cohort_gc: NDArray[np.float64] | None = None
    cohort_n_mask: NDArray[np.bool_] | None = None
    cohort_map: NDArray[np.float64] | None = None
    cohort_gc_q: NDArray[np.float32] | None = None
    try:
        gc_src = config.gc if config.gc is not None else config.fasta
        if gc_src is not None:
            cohort_gc, cohort_n_mask, cohort_gc_q = load_gc_track(
                gc_src, cohort.contigs, cohort.starts, cohort.ends
            )
            if config.gc is not None and config.fasta is not None:
                _, fasta_n_mask, fasta_gc_q = load_gc_track(
                    config.fasta, cohort.contigs, cohort.starts, cohort.ends
                )
                cohort_n_mask = cohort_n_mask | fasta_n_mask
                if cohort_gc_q is None:
                    cohort_gc_q = fasta_gc_q
        if config.mappability is not None:
            cohort_map = load_mappability_track(
                config.mappability, cohort.contigs, cohort.starts, cohort.ends
            )
    except CovariateError as err:
        raise ReferenceError(str(err)) from err

    autosomes, sex_contigs = _resolve_contig_roles(cohort, config)
    prenatal_meta = None
    prenatal_auto_fit: BranchFitArrays | None = None
    prenatal_ffy_fit: BranchFitArrays | None = None

    if config.mode == "prenatal":
        eff_bs = config.bin_size if config.bin_size is not None else cohort.bin_size
        entries = config.entries or tuple(
            SamplesheetEntry(filepath=Path(f"sample_{i}.ccnv"))
            for i in range(cohort.counts.shape[1])
        )
        prenatal_auto_fit, prenatal_ffy_fit, prenatal_meta, is_m = fit_prenatal_models(
            cohort,
            entries,
            bin_size=eff_bs,
            prenatal_bin_size=config.prenatal_bin_size,
            seed=config.seed,
            cohort_gc=cohort_gc,
            cohort_mappability=cohort_map,
        )
        sexes: tuple[SexLabel, ...] | None = tuple(
            "male" if m else "female" for m in np.array(is_m, dtype=np.bool_)
        )
        sex_source, signal = "prenatal_model", _resolve_sex_signal(cohort, config)
        quarantined = np.zeros(len(is_m), dtype=np.bool_)
        if prenatal_meta.quarantined_samples is not None:
            quarantined[prenatal_meta.quarantined_samples] = True
        inference: SexInference | None = SexInference(
            labels=sexes,
            cutoff=float(prenatal_meta.yfrac_threshold),
            fractions=(),
            passed=True,
            reason="prenatal_model",
            quarantined_mask=quarantined,
        )
    elif is_flat:
        sexes, sex_source, inference, signal = (
            None,
            "none",
            None,
            _resolve_sex_signal(cohort, config),
        )
    else:
        _, cohort_c_ids = np.unique(
            np.asarray(cohort.contigs, dtype=str), return_inverse=True
        )
        cohort_exp = compute_bin_exposure(cohort.starts, cohort.ends, cohort.bin_size)
        cohort_usable = mask_covariates(
            mask_low_coverage_per_contig(
                cohort.counts, cohort_c_ids.astype(np.int16), cohort_exp
            ),
            gc=cohort_gc,
            mappability=cohort_map,
            n_mask=cohort_n_mask,
            min_mappability=config.min_mappability,
        )
        sexes, sex_source, inference, signal = _resolve_sexes(
            cohort, config, sex_contigs, cohort_usable
        )

    cnv_cohort = (
        build_fisher_weighted_cohort(
            cohort,
            prenatal_meta.band_fisher_weights,
            max_batch_memory_mb=config.max_batch_memory_mb,
            threads=config.threads,
        )
        if (
            config.mode == "prenatal"
            and prenatal_meta is not None
            and prenatal_meta.band_fisher_weights is not None
        )
        else cohort
    )

    branch_fits: list[BranchFitArrays] = []

    def _add_branch(
        branch_name: str,
        sample_indices: NDArray[np.integer],
        contig_names: tuple[str, ...],
    ) -> None:
        branch_fits.append(
            _fit_branch(
                branch=branch_name,
                cohort=cnv_cohort,
                sample_index=sample_indices,
                contig_names=contig_names,
                config=config,
                cohort_gc=cohort_gc,
                cohort_mappability=cohort_map,
                cohort_n_mask=cohort_n_mask,
                cohort_gc_quantiles=cohort_gc_q,
            )
        )

    _add_branch(BRANCH_AUTOSOMAL, np.arange(cohort.counts.shape[1]), autosomes)
    if not is_flat and sex_contigs and sexes is not None:
        labels = np.asarray(sexes)
        clean = (
            ~inference.quarantined_mask
            if (
                inference is not None
                and inference.quarantined_mask.shape == (cohort.counts.shape[1],)
            )
            else np.ones(cohort.counts.shape[1], dtype=np.bool_)
        )
        if (
            female_indices := np.flatnonzero((labels == "female") & clean)
        ).size >= config.min_sex_group_size:
            _add_branch(BRANCH_FEMALE_SEX, female_indices, sex_contigs)
        if (
            male_indices := np.flatnonzero((labels == "male") & clean)
        ).size >= config.min_sex_group_size:
            _add_branch(BRANCH_MALE_SEX, male_indices, sex_contigs)

    if config.mode == "prenatal":
        if prenatal_auto_fit is None or prenatal_ffy_fit is None:
            raise ReferenceError("prenatal branch models were not initialized")
        for fit in (prenatal_auto_fit, prenatal_ffy_fit):
            if fit.num_rows > 0:
                branch_fits.append(fit)

    p_max = max((b.pca_rank for b in branch_fits), default=0)
    built_branches = tuple(b.branch for b in branch_fits)
    branch_pca_ranks = {b.branch: b.pca_rank for b in branch_fits}

    first_filters: dict[str, object] | None = None
    if cohort.sample_metadata:
        raw_f = cohort.sample_metadata[0].get("filters")
        if isinstance(raw_f, Mapping):
            first_filters = {str(k): v for k, v in raw_f.items()}

    resolved_sex_cutoff = (
        config.sex_cutoff
        if config.sex_cutoff is not None
        else (inference.cutoff if inference is not None and inference.passed else None)
    )
    ref_meta = ReferenceMetadata(
        bin_size=cohort.bin_size,
        contig_lengths=cohort.contig_lengths,
        branches=built_branches,
        autosomal_contigs=autosomes,
        sex_contigs=sex_contigs,
        pca_components=(
            config.pca_components if config.pca_components is not None else p_max
        ),
        mode=AnalysisMode(config.mode),
        prenatal_bin_size=(
            config.prenatal_bin_size if config.mode == "prenatal" else None
        ),
        sample_identities=cohort.identities,
        sex_signal_contig=signal,
        sex_cutoff=resolved_sex_cutoff,
        sex_source=sex_source,
        sample_sexes=sexes,
        sex_inference=(
            {
                "passed": inference.passed,
                "cutoff": inference.cutoff,
                "fractions": list(inference.fractions),
                "reason": inference.reason,
                "quarantined_indices": [
                    int(i) for i in np.flatnonzero(inference.quarantined_mask)
                ],
            }
            if inference is not None
            else None
        ),
        max_batch_memory_mb=config.max_batch_memory_mb,
        flat=is_flat,
        gc_corrected=(config.gc is not None) or is_flat,
        mappability_corrected=config.mappability is not None,
        fragmentomics_corrected=(cohort.ref_frag_len is not None)
        or (cohort.ref_ep_rate is not None),
        min_mappability=config.min_mappability,
        branch_pca_ranks=branch_pca_ranks,
        filters=first_filters,
        prenatal=prenatal_meta if config.mode == "prenatal" else None,
    )
    return assemble_reference_table(
        branch_fits,
        contig_order=tuple(cohort.contig_lengths.keys()),
        metadata=ref_meta.build_arrow_metadata(),
    )


def _validate_config(config: ReferenceConfig) -> None:
    """Validate reference build configuration parameters and input paths."""
    if config.mode == "prenatal" and config.fasta is not None:
        raise ReferenceError(
            "Flat FASTA references are only supported in --mode germline; "
            "--mode prenatal requires a cohort --samplesheet."
        )
    has_samples, has_fasta = (
        bool(config.entries or config.samplesheet_path),
        config.fasta is not None,
    )
    if not has_samples and not has_fasta:
        raise ReferenceError(
            "provide sample Parquet entries or a FASTA to build a flat reference"
        )
    if has_samples and has_fasta:
        raise ReferenceError(
            "provide either sample Parquet entries or a FASTA, not both"
        )
    if has_fasta and (config.bin_size is None or config.bin_size <= 0):
        raise ReferenceError("bin_size must be a positive integer when using a FASTA")
    if config.pca_components is not None and config.pca_components < 1:
        raise ReferenceError("pca_components must be a positive integer")
    for val, name in [
        (config.min_cohort_size, "min_cohort_size"),
        (config.min_sex_group_size, "min_sex_group_size"),
    ]:
        if val < 1:
            raise ReferenceError(f"{name} must be a positive integer")
    if config.max_batch_memory_mb < 0:
        raise ReferenceError("max_batch_memory_mb must be a non-negative integer")
    try:
        validate_min_mappability(config.min_mappability)
    except CovariateError as err:
        raise ReferenceError(str(err)) from err
    if config.sex_cutoff is not None and (
        not math.isfinite(config.sex_cutoff) or not (0.0 <= config.sex_cutoff <= 1.0)
    ):
        raise ReferenceError("sex_cutoff must be a finite number in [0, 1]")


def _resolve_contig_roles(
    cohort: Cohort, config: ReferenceConfig
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Partition cohort contigs into autosomal and sex-chromosome sets."""
    ordered, avail = tuple(dict.fromkeys(cohort.contigs)), set(cohort.contigs)
    req_sex = (
        None if config.sex_contigs is None else tuple(dict.fromkeys(config.sex_contigs))
    )
    sex_contigs = (
        tuple(n for n in DEFAULT_SEX_CONTIGS if n in avail)
        if (req_sex is None or req_sex == DEFAULT_SEX_CONTIGS)
        else req_sex
    )
    if (
        req_sex is not None
        and req_sex != DEFAULT_SEX_CONTIGS
        and any(n not in avail for n in sex_contigs)
    ):
        unknown = ", ".join(n for n in sex_contigs if n not in avail)
        raise ReferenceError(f"unknown sex contig(s): {unknown}")
    autosomes = (
        tuple(n for n in ordered if n not in set(sex_contigs))
        if config.autosomal_contigs is None
        else tuple(dict.fromkeys(config.autosomal_contigs))
    )
    if any(n not in avail for n in autosomes):
        unknown_auto = ", ".join(n for n in autosomes if n not in avail)
        raise ReferenceError(f"unknown autosomal contig(s): {unknown_auto}")
    if set(autosomes) & set(sex_contigs):
        overlap = ", ".join(sorted(set(autosomes) & set(sex_contigs)))
        raise ReferenceError(
            f"autosomal and sex contig sets must be disjoint: {overlap}"
        )
    if not autosomes:
        raise ReferenceError("reference requires at least one autosomal contig")
    return autosomes, sex_contigs


def _resolve_sexes(
    cohort: Cohort,
    config: ReferenceConfig,
    sex_contigs: tuple[str, ...],
    usable: NDArray[np.bool_],
) -> tuple[tuple[SexLabel, ...] | None, str, SexInference | None, str | None]:
    """Resolve sample sex assignments, 2D outlier quarantine, and signal cutoff."""
    signal = _resolve_sex_signal(cohort, config)
    if not sex_contigs or signal is None:
        return None, "none", None, signal
    explicit: tuple[SexLabel | None, ...] | None = (
        tuple(
            "male" if e.sex == "M" else ("female" if e.sex == "F" else None)
            for e in config.entries
        )
        if config.entries and len(config.entries) == cohort.counts.shape[1]
        else None
    )
    all_labeled = explicit is not None and all(x is not None for x in explicit)
    inference = infer_germline_sex_2d(
        cohort,
        usable,
        min_group_size=config.min_sex_group_size,
        signal_contig=signal,
        cutoff=config.sex_cutoff,
        explicit_labels=explicit,
    )
    if all_labeled and explicit is not None:
        lbls: tuple[SexLabel, ...] = tuple(
            "male" if x == "male" else "female" for x in explicit
        )
        return lbls, "samplesheet", inference, signal
    if not inference.passed or inference.labels is None:
        return None, "none", inference, signal
    return inference.labels, "inferred", inference, signal


def _resolve_sex_signal(cohort: Cohort, config: ReferenceConfig) -> str | None:
    """Identify the sex-signal contig available in the cohort."""
    avail, req = set(cohort.contigs), config.sex_signal_contig
    if req is None or req == DEFAULT_SEX_SIGNAL_CONTIGS[0]:
        return next((n for n in DEFAULT_SEX_SIGNAL_CONTIGS if n in avail), None)
    if req not in avail:
        raise ReferenceError(f"unknown sex-signal contig {req!r}")
    return req


def _fit_branch(
    *,
    branch: str,
    cohort: Cohort,
    sample_index: NDArray[np.integer],
    contig_names: tuple[str, ...],
    config: ReferenceConfig,
    cohort_gc: NDArray[np.float64] | None = None,
    cohort_mappability: NDArray[np.float64] | None = None,
    cohort_n_mask: NDArray[np.bool_] | None = None,
    cohort_gc_quantiles: NDArray[np.float32] | None = None,
) -> BranchFitArrays:
    """Fit PCA, Prototype Graph, and Empirical Bayes overdispersion for one branch."""
    selected = np.isin(cohort.contigs, contig_names)
    if not np.any(selected):
        raise ReferenceError(f"branch {branch!r} has no bins")
    n_bins = int(np.count_nonzero(selected))
    is_flat = config.fasta is not None

    contig_subset = tuple(
        str(c) for c in np.array(cohort.contigs, dtype=object)[selected]
    )
    uniq_c = list(dict.fromkeys(contig_subset))
    c_map = {name: idx for idx, name in enumerate(uniq_c)}
    c_codes = np.array([c_map[c] for c in contig_subset], dtype=np.int16)
    branch_starts = cohort.starts[selected]
    branch_ends = cohort.ends[selected]
    branch_exp = compute_bin_exposure(branch_starts, branch_ends, cohort.bin_size)
    branch_gc = cohort_gc[selected] if cohort_gc is not None else None
    branch_gc_q = (
        cohort_gc_quantiles[selected] if cohort_gc_quantiles is not None else None
    )
    branch_map = (
        cohort_mappability[selected] if cohort_mappability is not None else None
    )
    branch_n = cohort_n_mask[selected] if cohort_n_mask is not None else None
    branch_frag_len = (
        cohort.ref_frag_len[selected] if cohort.ref_frag_len is not None else None
    )
    branch_ep_ratio = (
        cohort.ref_ep_rate[selected] if cohort.ref_ep_rate is not None else None
    )

    if is_flat:
        counts = np.empty((n_bins, 0), dtype=np.float32)
        base_usable = np.ones(n_bins, dtype=np.bool_)
    else:
        n_sub = int(sample_index.size)
        if n_sub == 0:
            raise ReferenceError(f"branch {branch!r} has no samples")
        full_samples = n_sub == cohort.counts.shape[1] and bool(
            np.array_equal(sample_index, np.arange(cohort.counts.shape[1]))
        )
        if bool(np.all(selected)) and full_samples:
            counts = (
                cohort.counts
                if cohort.counts.dtype == np.float32
                else np.asarray(cohort.counts, dtype=np.float32)
            )
        else:
            sel_idx = np.flatnonzero(selected)
            counts = np.empty((n_bins, n_sub), dtype=np.float32)
            for r0 in range(0, n_bins, 32_768):
                r1 = min(n_bins, r0 + 32_768)
                sub_rows = cohort.counts[sel_idx[r0:r1]]
                counts[r0:r1] = (
                    sub_rows
                    if full_samples
                    else np.asarray(sub_rows[:, sample_index], dtype=np.float32)
                )

        col_totals = np.zeros(n_sub, dtype=np.float64)
        for r0 in range(0, n_bins, 32_768):
            col_totals += np.sum(
                counts[r0 : min(n_bins, r0 + 32_768)], axis=0, dtype=np.float64
            )
        if np.any(col_totals <= 0.0):
            raise ReferenceError(f"branch {branch!r} has a zero-coverage sample")
        base_usable = mask_low_coverage_per_contig(counts, c_codes, branch_exp)

    usable = mask_covariates(
        base_usable,
        gc=branch_gc,
        mappability=branch_map,
        n_mask=branch_n,
        min_mappability=config.min_mappability,
    )
    if not np.any(usable):
        raise ReferenceError(f"branch {branch!r} has no unmasked bins")
    u_count = int(np.count_nonzero(usable))
    logger.debug("branch %s: mask kept %d/%d bins", branch, u_count, int(usable.size))

    cov_mat = covariate_features(
        usable,
        gc=branch_gc,
        mappability=branch_map,
        ref_frag_len=branch_frag_len,
        ref_ep_rate=branch_ep_ratio,
    )

    if is_flat:
        branch_pca_mean = np.zeros(n_bins, dtype=np.float32)
        branch_pca_mean[usable] = 1.0
        pca_loadings = np.empty((0, u_count), dtype=np.float32)
        profiles = build_whitened_subspace_features(
            pca_loadings,
            np.empty(0, dtype=np.float64),
            0.0,
            covariate_mat=cov_mat,
        )
        _, pids_u, pweights_u, mdist_u = fit_prototype_graph(
            profiles,
            max_prototypes=2048,
            top_r=NUM_PROTOTYPES_PER_BIN,
        )
        branch_null_mean = np.zeros(n_bins, dtype=np.float32)
        branch_bio_disp = np.zeros(n_bins, dtype=np.float32)
    else:
        sf = estimate_size_factors(
            counts,
            mask=usable,
            contigs=c_codes,
            starts=branch_starts,
            ends=branch_ends,
        )
        design_u = build_exogenous_covariate_basis(
            usable,
            bin_size=cohort.bin_size,
            gc=branch_gc,
            gc_quantiles=branch_gc_q,
            mappability=branch_map,
            ref_frag_len=branch_frag_len,
            ref_ep_rate=branch_ep_ratio,
        )
        expected = fit_poisson_covariate_trends(
            counts,
            sf,
            branch_exp,
            usable,
            design_u,
            threads=config.threads,
        )
        expected *= sf[np.newaxis, :].astype(np.float32)

        pca_res = fit_whitened_pca(
            counts, expected, usable, max_components=config.pca_components
        )
        branch_pca_mean = pca_res.mean_rate
        pca_loadings = pca_res.loadings
        profiles = build_whitened_subspace_features(
            pca_res.loadings,
            pca_res.singular_values,
            pca_res.mp_sigma,
            covariate_mat=cov_mat,
            fallback_rate=pca_res.mean_rate[usable],
        )
        centers, pids_u, pweights_u, mdist_u = fit_prototype_graph(
            profiles,
            max_prototypes=2048,
            top_r=NUM_PROTOTYPES_PER_BIN,
        )
        recon_expected = reconstruct_whitened_pca_expected(expected, usable, pca_res)
        branch_null_mean, branch_bio_disp = calibrate_prototype_dispersion(
            counts,
            recon_expected,
            c_codes,
            usable,
            pids_u,
            pweights_u,
            pca_rank=int(pca_loadings.shape[0]),
            n_prototypes=int(centers.shape[0]),
        )

    raw_rank = int(pca_loadings.shape[0])
    full_ld: NDArray[np.float32] | None = None
    if raw_rank > 0:
        full_ld = np.zeros((n_bins, raw_rank), dtype=np.float32)
        full_ld[usable] = pca_loadings.T.astype(np.float32)

    full_pids = np.zeros((n_bins, NUM_PROTOTYPES_PER_BIN), dtype=np.uint16)
    full_pids[usable] = pids_u
    full_pweights = np.zeros((n_bins, NUM_PROTOTYPES_PER_BIN), dtype=np.float16)
    full_pweights[usable] = pweights_u
    branch_mean_dist = np.zeros(n_bins, dtype=np.float32)
    branch_mean_dist[usable] = mdist_u

    return BranchFitArrays(
        branch=branch,
        contigs=contig_subset,
        starts=branch_starts,
        ends=branch_ends,
        masked=~usable,
        gc=branch_gc,
        mappability=branch_map,
        gc_quantiles=branch_gc_q,
        ref_frag_len=branch_frag_len,
        ref_ep_rate=branch_ep_ratio,
        pca_mean=branch_pca_mean,
        pca_loadings=full_ld,
        proto_ids=full_pids,
        proto_weights=full_pweights,
        mean_distance=branch_mean_dist,
        null_mean=branch_null_mean,
        bio_dispersion=branch_bio_disp,
    )


run_reference: Final = run
