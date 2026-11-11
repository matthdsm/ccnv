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
    cohort_from_fasta,
    infer_sample_sexes,
    load_and_assemble_cohort,
    mask_low_coverage,
    normalize_depth,
)
from ccnv.features.covariates import (
    correct_covariate_bias,
    covariate_features,
    mask_covariates,
    validate_min_mappability,
)
from ccnv.features.prenatal import fit_prenatal_models
from ccnv.io.parquet import write_reference_parquet
from ccnv.io.samplesheet import load_samplesheet
from ccnv.io.tracks import (
    load_fasta_gc_quantiles,
    load_gc_track,
    load_mappability_track,
    read_fasta_contigs,
)
from ccnv.models import (
    BRANCH_AUTOSOMAL,
    BRANCH_FEMALE_SEX,
    BRANCH_MALE_SEX,
    BRANCH_PRENATAL_AUTOSOMAL,
    BRANCH_PRENATAL_FFY,
    NUM_GC_QUANTILES,
    NUM_PROTOTYPES_PER_BIN,
    ArtifactKind,
    CovariateError,
    ReferenceConfig,
    ReferenceError,
    SamplesheetEntry,
    SchemaError,
    build_reference_schema,
    fixed_size_list_to_numpy,
    metadata_for,
    parquet_path,
)
from ccnv.qc.reference import run_reference_qc_on_table
from ccnv.stats.neighbors import (
    build_signal_space_features,
    compute_prototype_null_models,
    fit_prototype_graph,
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


def run(config: ReferenceConfig) -> Path:
    """Validate a cohort and write a versioned Schema v2.0.0 reference Parquet file."""
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
        if config.bin_size is None:
            raise ReferenceError("--fasta requires --binsize")
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
        cohort = load_and_assemble_cohort(
            config.entries,
            target_bin_size=config.bin_size,
            threads=config.threads,
            max_batch_memory_mb=config.max_batch_memory_mb,
            collect_prenatal_bands=(config.mode == "prenatal"),
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


def _unify_branch_table(
    table: pa.Table,
    *,
    branch_code: int,
    branch_vocab: pa.Array,
    contig_vocab: pa.Array,
    contig_to_code: dict[str, int],
    p_max: int,
) -> tuple[pa.Table, int]:
    """Normalize branch dictionary, contig dictionary, and right-padded PCA rank."""
    n_rows = table.num_rows
    c_col = table.column("contig").combine_chunks()
    if isinstance(c_col, pa.DictionaryArray):
        local_vocab = [str(v) for v in c_col.dictionary.to_pylist()]
        local_codes = c_col.indices.to_numpy(zero_copy_only=False)
        mapped_c = np.array(
            [contig_to_code[local_vocab[int(c)]] for c in local_codes],
            dtype=np.int16,
        )
    else:
        mapped_c = np.array(
            [contig_to_code[str(v)] for v in c_col.to_numpy(zero_copy_only=False)],
            dtype=np.int16,
        )

    ld_col = table.column("pca_loadings").combine_chunks()
    ld_type = ld_col.type
    if not isinstance(ld_type, pa.FixedSizeListType):
        raise SchemaError(
            f"expected FixedSizeListType for 'pca_loadings', found {ld_type}"
        )
    p_branch = ld_type.list_size
    if p_max == 0:
        padded_ld: pa.Array = pa.nulls(n_rows, type=pa.list_(pa.float32(), 0))
    elif p_branch == p_max:
        padded_ld = ld_col
    else:
        padded_arr = np.zeros((n_rows, p_max), dtype=np.float32)
        if p_branch > 0 and n_rows > 0:
            raw_ld = fixed_size_list_to_numpy(ld_col, p_branch, np.float32)
            padded_arr[:, :p_branch] = np.nan_to_num(raw_ld, nan=0.0)
        padded_ld = pa.FixedSizeListArray.from_arrays(
            pa.array(padded_arr.ravel(), type=pa.float32()),
            p_max,
        )

    unified_branch = pa.DictionaryArray.from_arrays(
        pa.array(np.full(n_rows, branch_code, dtype=np.int8), type=pa.int8()),
        branch_vocab,
    )
    unified_contig = pa.DictionaryArray.from_arrays(
        pa.array(mapped_c, type=pa.int16()),
        contig_vocab,
    )

    schema = build_reference_schema(p_max)
    cols: dict[str, pa.Array | pa.ChunkedArray] = {
        name: table.column(name) for name in schema.names
    }
    cols["branch"] = unified_branch
    cols["contig"] = unified_contig
    cols["pca_loadings"] = padded_ld
    return pa.table(cols, schema=schema), p_branch


def build_reference_table(
    cohort: Cohort,
    config: ReferenceConfig,
) -> pa.Table:
    """Fit autosomal and optional sex-contig branches into one Schema v2.0.0 table."""
    if cohort.counts.shape[1] < config.min_cohort_size:
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
            cohort_gc, cohort_n_mask = load_gc_track(
                gc_src, cohort.contigs, cohort.starts, cohort.ends
            )
            cohort_gc_q = load_fasta_gc_quantiles(
                gc_src, cohort.contigs, cohort.starts, cohort.ends
            )
            if config.gc is not None and config.fasta is not None:
                cohort_n_mask = (
                    cohort_n_mask
                    | load_gc_track(
                        config.fasta, cohort.contigs, cohort.starts, cohort.ends
                    )[1]
                )
                if cohort_gc_q is None:
                    cohort_gc_q = load_fasta_gc_quantiles(
                        config.fasta, cohort.contigs, cohort.starts, cohort.ends
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

        sexes: tuple[SexLabel, ...] | None = tuple(
            "male" if m else "female" for m in np.array(is_m, dtype=np.bool_)
        )
        sex_source, signal = "prenatal_model", _resolve_sex_signal(cohort, config)
        inference: SexInference | None = SexInference(
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

    branch_tables: list[pa.Table] = []
    built_branches: list[str] = []

    def _add_branch(
        branch_name: str,
        sample_indices: NDArray[np.integer],
        contig_names: tuple[str, ...],
    ) -> None:
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
                cohort_gc_quantiles=cohort_gc_q,
            )
        )
        built_branches.append(branch_name)

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
        if prenatal_auto is None or prenatal_ffy is None:
            raise ReferenceError("prenatal branch models were not initialized")
        for branch_name, branch_table in [
            (BRANCH_PRENATAL_AUTOSOMAL, prenatal_auto),
            (BRANCH_PRENATAL_FFY, prenatal_ffy),
        ]:
            if branch_table.num_rows > 0:
                built_branches.append(branch_name)
                branch_tables.append(branch_table)

    p_max = 0
    for bt in branch_tables:
        lt = bt.schema.field("pca_loadings").type
        if not isinstance(lt, pa.FixedSizeListType):
            raise SchemaError(
                f"expected FixedSizeListType for 'pca_loadings', found {lt}"
            )
        p_max = max(p_max, lt.list_size)

    all_contigs_vocab = list(cohort.contig_lengths.keys())
    for c in cohort.contigs:
        if c not in cohort.contig_lengths:
            all_contigs_vocab.append(c)
    contig_to_code = {name: idx for idx, name in enumerate(all_contigs_vocab)}
    branch_vocab_arr = pa.array(built_branches, type=pa.string())
    contig_vocab_arr = pa.array(all_contigs_vocab, type=pa.string())

    unified_tables: list[pa.Table] = []
    branch_pca_ranks: dict[str, int] = {}
    for b_idx, (b_name, bt) in enumerate(
        zip(built_branches, branch_tables, strict=True)
    ):
        u_tbl, p_b = _unify_branch_table(
            bt,
            branch_code=b_idx,
            branch_vocab=branch_vocab_arr,
            contig_vocab=contig_vocab_arr,
            contig_to_code=contig_to_code,
            p_max=p_max,
        )
        unified_tables.append(u_tbl)
        branch_pca_ranks[b_name] = 0 if b_name == BRANCH_PRENATAL_FFY else p_b

    first_filters: dict[str, object] | None = None
    if cohort.sample_metadata:
        raw_f = cohort.sample_metadata[0].get("filters")
        if isinstance(raw_f, Mapping):
            first_filters = {str(k): v for k, v in raw_f.items()}

    model_metadata: dict[str, object] = {
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
        "fragmentomics_corrected": (cohort.ref_frag_len is not None)
        or (cohort.ref_ep_rate is not None),
        "min_mappability": config.min_mappability,
        "branch_pca_ranks": branch_pca_ranks,
        "filters": first_filters,
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
    return pa.concat_tables(unified_tables).replace_schema_metadata(
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
    if not inference.passed or inference.labels is None:
        return None, "none", inference, signal
    labels: list[SexLabel] = (
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
    totals = np.asarray(cohort.counts, dtype=np.float64).sum(axis=0)
    if np.any(totals <= 0):
        raise ReferenceError("every sample must have a positive total count")
    sub = np.asarray(cohort.counts[np.isin(cohort.contigs, [contig])], dtype=np.float64)
    return sub.sum(axis=0) / totals


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
    cohort_gc_quantiles: NDArray[np.float32] | None = None,
) -> pa.Table:
    """Build a Schema v2.0.0 reference branch table with PCA, Prototype Graph, and null dispersion."""
    selected = np.isin(cohort.contigs, contig_names)
    if not np.any(selected):
        raise ReferenceError(f"branch {branch!r} has no bins")
    counts = np.asarray(cohort.counts[np.ix_(selected, sample_index)], dtype=np.float64)
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
        "branch %s: mask kept %d/%d bins",
        branch,
        int(np.count_nonzero(usable)),
        int(usable.size),
    )

    branch_frag_len = (
        cohort.ref_frag_len[selected] if cohort.ref_frag_len is not None else None
    )
    branch_ep_ratio = (
        cohort.ref_ep_rate[selected] if cohort.ref_ep_rate is not None else None
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
        "branch %s: fitting prototype graph on %d usable bins",
        branch,
        int(np.count_nonzero(usable)),
    )
    _, pids_u, pweights_u, mdist_u = fit_prototype_graph(
        profiles.astype(np.float32),
        max_prototypes=2048,
        top_r=NUM_PROTOTYPES_PER_BIN,
    )
    pca_rank = 0 if config.fasta is not None else int(pca_loadings.shape[0])
    null_mean_u, bio_disp_u = compute_prototype_null_models(
        corrected,
        contig_ids,
        pids_u,
        pweights_u,
        n_pca_components=pca_rank,
    )
    logger.debug(
        "branch %s: null models computed for %d bins",
        branch,
        int(np.count_nonzero(usable)),
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
    gc_q_col: pa.Array = (
        pa.FixedSizeListArray.from_arrays(
            pa.array(cohort_gc_quantiles[selected].ravel(), type=pa.float32()),
            NUM_GC_QUANTILES,
        )
        if cohort_gc_quantiles is not None
        else pa.nulls(n_bins, type=pa.list_(pa.float32(), NUM_GC_QUANTILES))
    )
    output_frag_len = (
        np.where(np.isfinite(branch_frag_len), branch_frag_len, -1.0).astype(np.float32)
        if branch_frag_len is not None
        else np.full(n_bins, -1.0, dtype=np.float32)
    )
    output_ep_rate = (
        np.where(np.isfinite(branch_ep_ratio), branch_ep_ratio, -1.0).astype(np.float32)
        if branch_ep_ratio is not None
        else np.full(n_bins, -1.0, dtype=np.float32)
    )

    branch_pca_mean = np.zeros(n_bins, dtype=np.float32)
    branch_pca_mean[usable] = pca_mean.astype(np.float32)

    raw_rank = int(pca_loadings.shape[0])
    if raw_rank == 0:
        loadings_col: pa.Array = pa.nulls(n_bins, type=pa.list_(pa.float32(), 0))
    else:
        full_ld = np.zeros((n_bins, raw_rank), dtype=np.float32)
        full_ld[usable] = pca_loadings.T.astype(np.float32)
        loadings_col = pa.FixedSizeListArray.from_arrays(
            pa.array(full_ld.ravel(), type=pa.float32()),
            raw_rank,
        )

    full_pids = np.zeros((n_bins, NUM_PROTOTYPES_PER_BIN), dtype=np.uint16)
    full_pids[usable] = pids_u
    full_pweights = np.zeros((n_bins, NUM_PROTOTYPES_PER_BIN), dtype=np.float16)
    full_pweights[usable] = pweights_u

    branch_mean_dist = np.zeros(n_bins, dtype=np.float32)
    branch_mean_dist[usable] = mdist_u
    branch_null_mean = np.zeros(n_bins, dtype=np.float32)
    branch_null_mean[usable] = null_mean_u
    branch_bio_disp = np.zeros(n_bins, dtype=np.float32)
    branch_bio_disp[usable] = bio_disp_u

    uniq_c = list(dict.fromkeys(str(c) for c in contig_subset))
    c_map = {name: idx for idx, name in enumerate(uniq_c)}
    c_codes = np.array([c_map[str(c)] for c in contig_subset], dtype=np.int16)

    schema = build_reference_schema(raw_rank)
    return pa.table(
        {
            "branch": pa.DictionaryArray.from_arrays(
                pa.array(np.zeros(n_bins, dtype=np.int8), type=pa.int8()),
                pa.array([branch], type=pa.string()),
            ),
            "contig": pa.DictionaryArray.from_arrays(
                pa.array(c_codes, type=pa.int16()),
                pa.array(uniq_c, type=pa.string()),
            ),
            "start": cohort.starts[selected].astype(np.uint32),
            "end": cohort.ends[selected].astype(np.uint32),
            "masked": ~usable,
            "gc": output_gc,
            "mappability": output_map,
            "gc_quantiles": gc_q_col,
            "ref_frag_len": output_frag_len,
            "ref_ep_rate": output_ep_rate,
            "pca_mean": branch_pca_mean,
            "pca_loadings": loadings_col,
            "proto_ids": pa.FixedSizeListArray.from_arrays(
                pa.array(full_pids.ravel(), type=pa.uint16()),
                NUM_PROTOTYPES_PER_BIN,
            ),
            "proto_weights": pa.FixedSizeListArray.from_arrays(
                pa.array(full_pweights.ravel(), type=pa.float16()),
                NUM_PROTOTYPES_PER_BIN,
            ),
            "mean_distance": branch_mean_dist,
            "null_mean": branch_null_mean,
            "bio_dispersion": branch_bio_disp,
        },
        schema=schema,
    )


run_reference: Final = run
