"""Cohort validation and vectorized reference construction."""

from __future__ import annotations

import logging
import math
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
    cohort_from_fasta,
    infer_sample_sexes,
    mask_low_coverage,
    normalize_depth,
    validate_cohort,
)
from ccnv.features.covariates import (
    correct_covariate_bias,
    covariate_features,
    mask_covariates,
    validate_min_mappability,
)
from ccnv.features.prenatal import fit_prenatal_models
from ccnv.io.parquet import read_sample_tables, write_reference_parquet
from ccnv.io.samplesheet import load_samplesheet
from ccnv.io.tracks import load_gc_track, load_mappability_track, read_fasta_contigs
from ccnv.models import (
    BRANCH_AUTOSOMAL,
    BRANCH_FEMALE_SEX,
    BRANCH_MALE_SEX,
    BRANCH_PRENATAL_AUTOSOMAL,
    BRANCH_PRENATAL_FFY,
    REFERENCE_SCHEMA,
    ArtifactKind,
    CovariateError,
    ReferenceConfig,
    ReferenceError,
    SamplesheetEntry,
    metadata_for,
    parquet_path,
)
from ccnv.qc.reference import run_reference_qc_on_table
from ccnv.stats.neighbors import (
    build_signal_space_features,
    compute_null_models,
    distance_cutoff,
    select_nearest_bins,
)
from ccnv.stats.pca import fit_pca

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


def run(config: ReferenceConfig) -> None:
    """Validate a cohort and write a versioned reference Parquet file."""
    _validate_config(config)
    if config.samplesheet_path is not None:
        entries = load_samplesheet(config.samplesheet_path)
        config = replace(
            config, entries=entries, samples=tuple(e.filepath for e in entries)
        )
    elif config.entries:
        config = replace(config, samples=tuple(e.filepath for e in config.entries))
    elif config.samples:
        config = replace(
            config, entries=tuple(SamplesheetEntry(filepath=p) for p in config.samples)
        )
    if config.fasta is not None:
        assert config.bin_size is not None
        contigs = read_fasta_contigs(config.fasta, all_contigs=config.all_contigs)
        cohort = cohort_from_fasta(
            contigs,
            config.bin_size,
            n_samples=max(config.min_cohort_size, 2),
            fasta_path=config.fasta,
        )
        config = (
            replace(config, sex_contigs=())
            if (config.sex_contigs is None or config.sex_contigs == DEFAULT_SEX_CONTIGS)
            else config
        )
    else:
        sample_tables = read_sample_tables(config.samples, threads=config.threads)
        cohort = validate_cohort(
            sample_tables,
            paths=config.samples,
            target_bin_size=config.bin_size,
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


def build_reference_table(
    cohort: Cohort,
    config: ReferenceConfig,
) -> pa.Table:
    """Fit autosomal and optional sex-contig branches into one reference table."""

    if cohort.counts.shape[1] < config.min_cohort_size:
        raise ReferenceError(
            f"provide at least {config.min_cohort_size} samples to build a reference"
        )
    try:
        validate_min_mappability(config.min_mappability)
    except CovariateError as err:
        raise ReferenceError(str(err)) from err
    cohort_gc, cohort_n_mask, cohort_map = None, None, None
    try:
        if config.gc is not None or config.fasta is not None:
            cohort_gc, cohort_n_mask = load_gc_track(
                config.gc or config.fasta, cohort.contigs, cohort.starts, cohort.ends
            )
            if config.gc is not None and config.fasta is not None:
                cohort_n_mask = (
                    cohort_n_mask
                    | load_gc_track(
                        config.fasta, cohort.contigs, cohort.starts, cohort.ends
                    )[1]
                )
        if config.mappability is not None:
            cohort_map = load_mappability_track(
                config.mappability, cohort.contigs, cohort.starts, cohort.ends
            )
    except CovariateError as err:
        raise ReferenceError(str(err)) from err
    autosomes, sex_contigs = _resolve_contig_roles(cohort, config)
    prenatal_meta, prenatal_auto, prenatal_ffy = None, None, None
    if config.mode == "prenatal":
        eff_bs = config.bin_size if config.bin_size is not None else cohort.bin_size
        entries = config.entries or tuple(
            SamplesheetEntry(filepath=Path(f"sample_{i}.ccnv"))
            for i in range(cohort.counts.shape[1])
        )
        prenatal_auto, prenatal_ffy, prenatal_meta, is_m = fit_prenatal_models(
            cohort,
            entries,
            bin_size=eff_bs,
            prenatal_bin_size=config.prenatal_bin_size,
            seed=config.seed,
        )

        sexes = tuple("male" if m else "female" for m in np.array(is_m, dtype=np.bool_))
        sex_source, signal = "prenatal_model", _resolve_sex_signal(cohort, config)
        inference = SexInference(
            labels=sexes,
            cutoff=float(prenatal_meta.yfrac_threshold),
            fractions=(),
            passed=True,
            reason="prenatal_model",
        )
    else:
        sexes, sex_source, inference, signal = _resolve_sexes(
            cohort, config, sex_contigs
        )

    def _add_branch(
        branch_name: str,
        sample_indices: NDArray[np.integer],
        contig_names: tuple[str, ...],
    ) -> None:
        """Construct and append a reference branch table for the given cohort subset."""
        branch_tables.append(
            _build_branch_table(
                branch=branch_name,
                cohort=cohort,
                sample_index=sample_indices,
                contig_names=contig_names,
                config=config,
                cohort_gc=cohort_gc,
                cohort_mappability=cohort_map,
                cohort_n_mask=cohort_n_mask,
            )
        )
        built_branches.append(branch_name)

    branch_tables, built_branches = [], []
    _add_branch(BRANCH_AUTOSOMAL, np.arange(cohort.counts.shape[1]), autosomes)
    if sex_contigs and sexes is not None:
        labels = np.asarray(sexes)
        if (
            female_indices := np.flatnonzero(labels == "female")
        ).size >= config.min_sex_group_size:
            _add_branch(BRANCH_FEMALE_SEX, female_indices, sex_contigs)
        if (
            male_indices := np.flatnonzero(labels == "male")
        ).size >= config.min_sex_group_size:
            _add_branch(BRANCH_MALE_SEX, male_indices, sex_contigs)
    if config.mode == "prenatal":
        assert prenatal_auto is not None and prenatal_ffy is not None
        for branch_name, branch_table in [
            (BRANCH_PRENATAL_AUTOSOMAL, prenatal_auto),
            (BRANCH_PRENATAL_FFY, prenatal_ffy),
        ]:
            if branch_table.num_rows > 0:
                built_branches.append(branch_name)
                branch_tables.append(branch_table)
    model_metadata = {
        "bin_size": cohort.bin_size,
        "branches": built_branches,
        "autosomal_contigs": list(autosomes),
        "sex_contigs": list(sex_contigs),
        "sex_signal_contig": signal,
        "sex_cutoff": config.sex_cutoff,
        "sex_source": sex_source,
        "sample_sexes": list(sexes) if sexes is not None else None,
        "sex_inference": {
            "passed": inference.passed,
            "cutoff": inference.cutoff,
            "fractions": list(inference.fractions),
            "reason": inference.reason,
        }
        if inference
        else None,
        "pca_components": config.pca_components,
        "nearest_bins": config.nearest_bins,
        "max_batch_memory_mb": config.max_batch_memory_mb,
        "flat": config.fasta is not None,
        "gc_corrected": (config.gc is not None) or (config.fasta is not None),
        "mappability_corrected": config.mappability is not None,
        "min_mappability": config.min_mappability,
        "mode": config.mode,
        **(
            {"prenatal_bin_size": config.prenatal_bin_size}
            if config.mode == "prenatal"
            else {}
        ),
    }
    values: dict[str, object] = {
        "sample_identities": list(cohort.identities),
        "model": model_metadata,
        **(
            {"prenatal": prenatal_meta.to_dict()}
            if (config.mode == "prenatal" and prenatal_meta is not None)
            else {}
        ),
    }
    return pa.concat_tables(branch_tables).replace_schema_metadata(
        metadata_for(
            ArtifactKind.REFERENCE, contig_lengths=cohort.contig_lengths, values=values
        )
    )


def _validate_config(config: ReferenceConfig) -> None:
    """Validate reference build configuration parameters and input paths."""
    if config.mode == "prenatal" and config.fasta is not None:
        raise ReferenceError(
            "Flat FASTA references are only supported in --mode germline; "
            "--mode prenatal requires a cohort --samplesheet."
        )
    has_samples, has_fasta = (
        bool(config.samples or config.entries or config.samplesheet_path),
        config.fasta is not None,
    )
    if not has_samples and not has_fasta:
        raise ReferenceError(
            "provide sample Parquet paths or a FASTA to build a flat reference"
        )
    if has_samples and has_fasta:
        raise ReferenceError("provide either sample Parquet paths or a FASTA, not both")
    if has_fasta and (config.bin_size is None or config.bin_size <= 0):
        raise ReferenceError("bin_size must be a positive integer when using a FASTA")
    for val, name in [
        (config.pca_components, "pca_components"),
        (config.nearest_bins, "nearest_bins"),
        (config.max_batch_memory_mb, "max_batch_memory_mb"),
        (config.min_cohort_size, "min_cohort_size"),
        (config.min_sex_group_size, "min_sex_group_size"),
    ]:
        if val < 1:
            raise ReferenceError(f"{name} must be a positive integer")
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
        msg = f"autosomal and sex contig sets must be disjoint: {overlap}"
        raise ReferenceError(msg)
    if not autosomes:
        raise ReferenceError("reference requires at least one autosomal contig")
    return autosomes, sex_contigs


def _resolve_sexes(
    cohort: Cohort, config: ReferenceConfig, sex_contigs: tuple[str, ...]
) -> tuple[tuple[SexLabel, ...] | None, str, SexInference | None, str | None]:
    """Resolve sample sex assignments from samplesheet or coverage inference."""
    signal = _resolve_sex_signal(cohort, config)
    if not sex_contigs or signal is None:
        return None, "none", None, signal
    if (
        config.entries
        and len(config.entries) == cohort.counts.shape[1]
        and all(e.sex in ("M", "F") for e in config.entries)
    ):
        return (
            tuple("male" if e.sex == "M" else "female" for e in config.entries),
            "samplesheet",
            None,
            signal,
        )
    inference = infer_sample_sexes(
        _contig_fractions(cohort, signal),
        config.min_sex_group_size,
        cutoff=config.sex_cutoff,
    )
    if not inference.passed:
        return None, "none", inference, signal
    labels = (
        [
            "male" if e.sex == "M" else "female" if e.sex == "F" else lbl
            for lbl, e in zip(inference.labels, config.entries, strict=False)
        ]
        if config.entries and len(config.entries) == len(inference.labels)
        else list(inference.labels)
    )
    return tuple(labels), "inferred", inference, signal


def _resolve_sex_signal(cohort: Cohort, config: ReferenceConfig) -> str | None:
    """Identify the sex-signal contig available in the cohort."""
    avail, req = set(cohort.contigs), config.sex_signal_contig
    if req is None or req == DEFAULT_SEX_SIGNAL_CONTIGS[0]:
        return next((n for n in DEFAULT_SEX_SIGNAL_CONTIGS if n in avail), None)
    if req not in avail:
        raise ReferenceError(f"unknown sex-signal contig {req!r}")
    return req


def _contig_fractions(cohort: Cohort, contig: str) -> NDArray[np.float64]:
    """Calculate per-sample fractional depth for a specific contig."""
    totals = cohort.counts.sum(axis=0)
    if np.any(totals <= 0):
        raise ReferenceError("every sample must have a positive total count")
    return cohort.counts[np.isin(cohort.contigs, [contig])].sum(axis=0) / totals


def _build_branch_table(
    *,
    branch: str,
    cohort: Cohort,
    sample_index: NDArray[np.integer],
    contig_names: tuple[str, ...],
    config: ReferenceConfig,
    cohort_gc: NDArray[np.float64] | None = None,
    cohort_mappability: NDArray[np.float64] | None = None,
    cohort_n_mask: NDArray[np.bool_] | None = None,
) -> pa.Table:
    """Build a reference branch Parquet table including PCA, KDTree, and null models."""
    selected = np.isin(cohort.contigs, contig_names)
    if not np.any(selected):
        raise ReferenceError(f"branch {branch!r} has no bins")
    counts = cohort.counts[np.ix_(selected, sample_index)]
    branch_gc = cohort_gc[selected] if cohort_gc is not None else None
    branch_map = (
        cohort_mappability[selected] if cohort_mappability is not None else None
    )
    branch_n = cohort_n_mask[selected] if cohort_n_mask is not None else None
    usable = mask_covariates(
        mask_low_coverage(normalize_depth(counts)),
        gc=branch_gc,
        mappability=branch_map,
        n_mask=branch_n,
        min_mappability=config.min_mappability,
    )
    if not np.any(usable):
        raise ReferenceError(f"branch {branch!r} has no unmasked bins")
    logger.debug(
        "branch %s: mask kept %d/%d bins (%.1f%%)",
        branch,
        np.count_nonzero(usable),
        usable.size,
        np.count_nonzero(usable) / usable.size * 100 if usable.size else 0.0,
    )
    branch_frag_len = (
        cohort.mean_frag_len[selected] if cohort.mean_frag_len is not None else None
    )
    branch_ep_ratio = (
        cohort.endpoint_ratio[selected] if cohort.endpoint_ratio is not None else None
    )
    normalized = correct_covariate_bias(
        normalize_depth(counts, mask=usable),
        usable,
        gc=branch_gc,
        mappability=branch_map,
        mean_frag_len=branch_frag_len,
        endpoint_ratio=branch_ep_ratio,
    )
    corrected, pca_mean, pca_loadings = fit_pca(
        normalized[usable], config.pca_components
    )
    contig_subset = np.array(cohort.contigs, dtype=object)[selected]
    _, contig_ids = np.unique(
        np.asarray(contig_subset[usable], dtype=str), return_inverse=True
    )
    cov_mat = covariate_features(
        usable,
        gc=branch_gc,
        mappability=branch_map,
        mean_frag_len=branch_frag_len,
        endpoint_ratio=branch_ep_ratio,
    )
    scaled_loadings = (
        pca_loadings * np.sqrt(pca_loadings.shape[1])
        if pca_loadings.size > 0
        else pca_loadings
    )
    profiles = build_signal_space_features(scaled_loadings, covariate_mat=cov_mat)
    if profiles.size == 0 or np.allclose(profiles, 0.0):
        profiles = corrected
    logger.debug(
        "branch %s: selecting %d nearest off-contig bins (batch memory limit=%d MB)",
        branch,
        config.nearest_bins,
        config.max_batch_memory_mb,
    )
    nearest_idx, nearest_dist = select_nearest_bins(
        profiles,
        contig_ids,
        k=config.nearest_bins,
        max_batch_memory_mb=config.max_batch_memory_mb,
        threads=config.threads,
    )
    null_mean, null_std = compute_null_models(
        corrected,
        nearest_idx,
        threads=config.threads,
        n_pca_components=0 if config.fasta is not None else pca_loadings.shape[0],
    )
    logger.debug(
        "branch %s: null models computed for %d bins",
        branch,
        np.count_nonzero(usable),
    )
    cutoff = distance_cutoff(nearest_dist, repeats=3)
    filtered_nearest, mean_distances = [], []
    for distance_list, index_list in zip(nearest_dist, nearest_idx, strict=True):
        distance_array = np.asarray(distance_list, dtype=np.float64)
        index_array = np.asarray(index_list, dtype=np.int32)
        valid = (distance_array <= cutoff) & np.isfinite(distance_array)
        mean_distances.append(
            float(np.mean(np.sqrt(np.maximum(distance_array[valid], 0.0))))
            if np.any(valid)
            else 0.0
        )
        filtered_nearest.append(
            np.sort(index_array[valid]).tolist() if np.any(valid) else []
        )
    n_bins = int(np.count_nonzero(selected))
    output_gc = (
        np.where(np.isfinite(branch_gc), branch_gc, -1.0).astype(np.float32)
        if branch_gc is not None
        else np.full(n_bins, -1.0, dtype=np.float32)
    )
    output_map = (
        np.where(np.isfinite(branch_map), branch_map, -1.0).astype(np.float32)
        if branch_map is not None
        else np.full(n_bins, -1.0, dtype=np.float32)
    )
    branch_pca_mean = np.zeros(n_bins, dtype=np.float64)
    branch_mean_dist = np.zeros(n_bins, dtype=np.float32)
    branch_null_mean = np.zeros(n_bins, dtype=np.float64)
    branch_null_std = np.zeros(n_bins, dtype=np.float64)
    (
        branch_pca_mean[usable],
        branch_mean_dist[usable],
        branch_null_mean[usable],
        branch_null_std[usable],
    ) = (
        pca_mean,
        mean_distances,
        null_mean,
        null_std,
    )
    full_pca_loadings = [
        pca_loadings[:, u_idx].tolist() if usable[bin_idx] else []
        for bin_idx, u_idx in zip(range(n_bins), np.cumsum(usable) - 1, strict=True)
    ]
    full_nearest_indices = [
        filtered_nearest[u_idx] if usable[bin_idx] else []
        for bin_idx, u_idx in zip(range(n_bins), np.cumsum(usable) - 1, strict=True)
    ]
    return pa.table(
        {
            "branch": pa.array([branch] * n_bins, type=pa.string()),
            "contig": pa.array(contig_subset.tolist(), type=pa.string()),
            "start": pa.array(cohort.starts[selected], type=pa.int64()),
            "end": pa.array(cohort.ends[selected], type=pa.int64()),
            "masked": pa.array(~usable, type=pa.bool_()),
            "gc": pa.array(output_gc, type=pa.float32()),
            "mappability": pa.array(output_map, type=pa.float32()),
            "pca_mean": pa.array(branch_pca_mean, type=pa.float64()),
            "pca_loadings": pa.array(full_pca_loadings, type=pa.list_(pa.float64())),
            "nearest_ref_indices": pa.array(
                full_nearest_indices, type=pa.list_(pa.int32())
            ),
            "mean_distance": pa.array(branch_mean_dist, type=pa.float32()),
            "null_mean": pa.array(branch_null_mean, type=pa.float64()),
            "null_std": pa.array(branch_null_std, type=pa.float64()),
        },
        schema=REFERENCE_SCHEMA,
    )


run_reference: Final = run
