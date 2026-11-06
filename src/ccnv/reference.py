"""Cohort validation and vectorized reference construction."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Final

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from ccnv.cohort import (
    Cohort as Cohort,
)
from ccnv.cohort import (
    ReferenceError as ReferenceError,
)
from ccnv.cohort import (
    cohort_from_fasta as cohort_from_fasta,
)
from ccnv.cohort import (
    load_cohort as load_cohort,
)
from ccnv.cohort import (
    validate_cohort as validate_cohort,
)
from ccnv.covariates import (
    CovariateError,
    correct_covariate_bias,
    covariate_features,
    load_gc_track,
    load_mappability_track,
    mask_covariates,
    validate_min_mappability,
)
from ccnv.features.cohort import (
    SexInference as SexInference,
)
from ccnv.features.cohort import (
    SexLabel as SexLabel,
)
from ccnv.features.cohort import (
    infer_sample_sexes as infer_sample_sexes,
)
from ccnv.features.cohort import (
    mask_low_coverage as mask_low_coverage,
)
from ccnv.features.cohort import (
    normalize_depth as normalize_depth,
)
from ccnv.io.parquet import write_reference_parquet
from ccnv.models import ReferenceConfig, SamplesheetEntry, parquet_path
from ccnv.prenatal import fit_prenatal_models
from ccnv.reference_compute import (
    compute_null_models,
)
from ccnv.reference_compute import (
    select_nearest_bins as select_nearest_bins,
)
from ccnv.reference_qc import run_reference_qc_on_table
from ccnv.samplesheet import load_samplesheet
from ccnv.schemas import (
    BRANCH_AUTOSOMAL,
    BRANCH_FEMALE_SEX,
    BRANCH_MALE_SEX,
    BRANCH_PRENATAL_AUTOSOMAL,
    BRANCH_PRENATAL_FFY,
    REFERENCE_SCHEMA,
    ArtifactKind,
    metadata_for,
)
from ccnv.stats.neighbors import distance_cutoff as distance_cutoff
from ccnv.stats.pca import fit_pca as fit_pca

DEFAULT_SEX_CONTIGS: Final[tuple[str, ...]] = ("X", "Y", "chrX", "chrY")
DEFAULT_SEX_SIGNAL_CONTIGS: Final[tuple[str, ...]] = ("chrY", "Y")

logger = logging.getLogger(__name__)


def run(config: ReferenceConfig) -> None:
    """Validate a cohort and write a versioned reference Parquet file."""

    _validate_config(config)
    if config.samplesheet_path is not None:
        entries = load_samplesheet(config.samplesheet_path)
        sample_paths = tuple(entry.filepath for entry in entries)
        config = replace(
            config,
            entries=entries,
            samples=sample_paths,
        )
    elif config.entries:
        entries = config.entries
        sample_paths = tuple(entry.filepath for entry in entries)
        config = replace(
            config,
            samples=sample_paths,
        )
    elif config.samples:
        entries = tuple(SamplesheetEntry(filepath=p) for p in config.samples)
        config = replace(
            config,
            entries=entries,
        )
    else:
        entries = ()

    if config.fasta is not None:
        assert config.bin_size is not None
        logger.debug("building flat reference from FASTA %s", config.fasta)
        n_samples = max(config.min_cohort_size, 2)
        cohort = cohort_from_fasta(
            config.fasta,
            config.bin_size,
            n_samples=n_samples,
            all_contigs=config.all_contigs,
        )
        if config.sex_contigs is None or config.sex_contigs == DEFAULT_SEX_CONTIGS:
            config = replace(config, sex_contigs=())
    else:
        logger.debug("loading %d sample parquet files", len(config.samples))
        cohort = load_cohort(
            config.samples,
            target_bin_size=config.bin_size,
            threads=config.threads,
        )
    logger.debug(
        "loaded cohort: %d bins, %d samples, bin_size=%d",
        cohort.counts.shape[0],
        cohort.counts.shape[1],
        cohort.bin_size,
    )

    table = build_reference_table(cohort, config)
    path = parquet_path(config.prefix)
    logger.debug("writing reference parquet %s (%d rows)", path, table.num_rows)
    write_reference_parquet(table, path)
    logger.debug(
        "wrote reference parquet %s (%.1f KB)",
        path,
        path.stat().st_size / 1024,
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
    sample_tables: Sequence[pa.Table] | None = None,
) -> pa.Table:
    """Fit autosomal and optional sex-contig branches into one reference table."""

    if cohort.counts.shape[1] < config.min_cohort_size:
        raise ReferenceError(
            f"provide at least {config.min_cohort_size} samples to build a reference"
        )
    try:
        validate_min_mappability(config.min_mappability)
    except CovariateError as error:
        raise ReferenceError(str(error)) from error

    cohort_gc: NDArray[np.float64] | None = None
    cohort_n_mask: NDArray[np.bool_] | None = None
    cohort_mappability: NDArray[np.float64] | None = None
    try:
        if config.gc is not None:
            cohort_gc, cohort_n_mask = load_gc_track(
                config.gc, cohort.contigs, cohort.starts, cohort.ends
            )
            if config.fasta is not None:
                _, fasta_n_mask = load_gc_track(
                    config.fasta, cohort.contigs, cohort.starts, cohort.ends
                )
                cohort_n_mask = cohort_n_mask | fasta_n_mask
        elif config.fasta is not None:
            cohort_gc, cohort_n_mask = load_gc_track(
                config.fasta, cohort.contigs, cohort.starts, cohort.ends
            )
        if config.mappability is not None:
            cohort_mappability = load_mappability_track(
                config.mappability, cohort.contigs, cohort.starts, cohort.ends
            )
    except CovariateError as error:
        raise ReferenceError(str(error)) from error

    autosomes, sex_contigs = _resolve_contig_roles(cohort, config)
    logger.debug(
        "resolved contig roles: autosomal=%s sex=%s",
        autosomes,
        sex_contigs,
    )

    prenatal_meta = None
    prenatal_auto_table = None
    prenatal_ffy_table = None
    if config.mode == "prenatal":
        effective_bin_size = (
            config.bin_size if config.bin_size is not None else cohort.bin_size
        )
        entries = config.entries
        if len(entries) != cohort.counts.shape[1]:
            entries = tuple(
                SamplesheetEntry(filepath=Path(f"sample_{i}.ccnv"))
                for i in range(cohort.counts.shape[1])
            )
            config = replace(config, entries=entries)

        seed = getattr(config, "seed", 42)
        prenatal_input = sample_tables if sample_tables is not None else cohort
        (
            prenatal_auto_table,
            prenatal_ffy_table,
            prenatal_meta,
            is_male_tuple,
        ) = fit_prenatal_models(
            prenatal_input,
            entries,
            bin_size=effective_bin_size,
            prenatal_bin_size=config.prenatal_bin_size,
            seed=seed,
        )
        is_male_arr = np.array(is_male_tuple, dtype=np.bool_)
        sexes: tuple[SexLabel, ...] | None = tuple(
            "male" if m else "female" for m in is_male_arr
        )
        sex_source = "prenatal_model"
        signal = _resolve_sex_signal(cohort, config)
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

    if inference is not None:
        cutoff_str = (
            f"{inference.cutoff:.4f}" if inference.cutoff is not None else "None"
        )
        logger.debug(
            "sex inference on signal %s: cutoff=%s (passed=%s, reason=%s)",
            signal,
            cutoff_str,
            inference.passed,
            inference.reason,
        )
    logger.debug(
        "resolved sample sexes via %s (signal=%s): %s",
        sex_source,
        signal,
        sexes,
    )

    branch_tables: list[pa.Table] = []
    built_branches: list[str] = []

    branch_tables.append(
        _build_branch_table(
            branch=BRANCH_AUTOSOMAL,
            cohort=cohort,
            sample_index=np.arange(cohort.counts.shape[1]),
            contig_names=autosomes,
            config=config,
            cohort_gc=cohort_gc,
            cohort_mappability=cohort_mappability,
            cohort_n_mask=cohort_n_mask,
        )
    )
    built_branches.append(BRANCH_AUTOSOMAL)

    if sex_contigs and sexes is not None:
        labels = np.asarray(sexes)
        female_index = np.flatnonzero(labels == "female")
        male_index = np.flatnonzero(labels == "male")
        if female_index.size >= config.min_sex_group_size:
            branch_tables.append(
                _build_branch_table(
                    branch=BRANCH_FEMALE_SEX,
                    cohort=cohort,
                    sample_index=female_index,
                    contig_names=sex_contigs,
                    config=config,
                    cohort_gc=cohort_gc,
                    cohort_mappability=cohort_mappability,
                    cohort_n_mask=cohort_n_mask,
                )
            )
            built_branches.append(BRANCH_FEMALE_SEX)
        if male_index.size >= config.min_sex_group_size:
            branch_tables.append(
                _build_branch_table(
                    branch=BRANCH_MALE_SEX,
                    cohort=cohort,
                    sample_index=male_index,
                    contig_names=sex_contigs,
                    config=config,
                    cohort_gc=cohort_gc,
                    cohort_mappability=cohort_mappability,
                    cohort_n_mask=cohort_n_mask,
                )
            )
            built_branches.append(BRANCH_MALE_SEX)

    if config.mode == "prenatal":
        assert prenatal_auto_table is not None and prenatal_ffy_table is not None
        if prenatal_auto_table.num_rows > 0:
            built_branches.append(BRANCH_PRENATAL_AUTOSOMAL)
            branch_tables.append(prenatal_auto_table)
        if prenatal_ffy_table.num_rows > 0:
            built_branches.append(BRANCH_PRENATAL_FFY)
            branch_tables.append(prenatal_ffy_table)

    full_table = pa.concat_tables(branch_tables)

    inference_meta: dict[str, object] | None = None
    if inference is not None:
        inference_meta = {
            "passed": inference.passed,
            "cutoff": inference.cutoff,
            "fractions": list(inference.fractions),
            "reason": inference.reason,
        }

    model_metadata: dict[str, object] = {
        "bin_size": cohort.bin_size,
        "branches": built_branches,
        "autosomal_contigs": list(autosomes),
        "sex_contigs": list(sex_contigs),
        "sex_signal_contig": signal,
        "sex_cutoff": config.sex_cutoff,
        "sex_source": sex_source,
        "sample_sexes": list(sexes) if sexes is not None else None,
        "sex_inference": inference_meta,
        "pca_components": config.pca_components,
        "nearest_bins": config.nearest_bins,
        "max_batch_memory_mb": config.max_batch_memory_mb,
        "flat": config.fasta is not None,
        "gc_corrected": (config.gc is not None) or (config.fasta is not None),
        "mappability_corrected": config.mappability is not None,
        "min_mappability": config.min_mappability,
        "mode": config.mode,
    }
    values: dict[str, object] = {
        "sample_identities": list(cohort.identities),
        "model": model_metadata,
    }
    if config.mode == "prenatal":
        assert prenatal_meta is not None
        model_metadata["prenatal_bin_size"] = config.prenatal_bin_size
        values["prenatal"] = prenatal_meta.to_dict()

    metadata = metadata_for(
        ArtifactKind.REFERENCE,
        contig_lengths=cohort.contig_lengths,
        values=values,
    )
    return full_table.replace_schema_metadata(metadata)


def _validate_config(config: ReferenceConfig) -> None:
    if config.mode == "prenatal" and config.fasta is not None:
        raise ReferenceError(
            "Flat FASTA references are only supported in --mode germline; "
            "--mode prenatal requires a cohort --samplesheet."
        )
    has_samples = bool(
        config.samples
        or config.entries
        or config.input_paths
        or config.samplesheet_path
    )
    has_fasta = config.fasta is not None
    if not has_samples and not has_fasta:
        raise ReferenceError(
            "provide sample Parquet paths or a FASTA to build a flat reference"
        )
    if has_samples and has_fasta:
        raise ReferenceError("provide either sample Parquet paths or a FASTA, not both")
    if has_fasta and (config.bin_size is None or config.bin_size <= 0):
        raise ReferenceError("bin_size must be a positive integer when using a FASTA")
    if config.pca_components < 1:
        raise ReferenceError("pca_components must be a positive integer")
    if config.nearest_bins < 1:
        raise ReferenceError("nearest_bins must be a positive integer")
    if config.max_batch_memory_mb < 1:
        raise ReferenceError("max_batch_memory_mb must be a positive integer")
    if config.min_cohort_size < 1:
        raise ReferenceError("min_cohort_size must be a positive integer")
    if config.min_sex_group_size < 1:
        raise ReferenceError("min_sex_group_size must be a positive integer")
    try:
        validate_min_mappability(config.min_mappability)
    except CovariateError as error:
        raise ReferenceError(str(error)) from error
    _validate_sex_cutoff(config.sex_cutoff)


def _resolve_contig_roles(
    cohort: Cohort,
    config: ReferenceConfig,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    ordered = tuple(dict.fromkeys(cohort.contigs))
    available = set(ordered)
    requested_sex = (
        None if config.sex_contigs is None else tuple(dict.fromkeys(config.sex_contigs))
    )
    if requested_sex is None or requested_sex == DEFAULT_SEX_CONTIGS:
        sex_contigs = tuple(name for name in DEFAULT_SEX_CONTIGS if name in available)
    else:
        sex_contigs = requested_sex
        unknown_sex = [name for name in sex_contigs if name not in available]
        if unknown_sex:
            raise ReferenceError(f"unknown sex contig(s): {', '.join(unknown_sex)}")
    if config.autosomal_contigs is None:
        autosomes = tuple(name for name in ordered if name not in set(sex_contigs))
    else:
        autosomes = tuple(dict.fromkeys(config.autosomal_contigs))
        unknown_auto = [name for name in autosomes if name not in available]
        if unknown_auto:
            raise ReferenceError(
                f"unknown autosomal contig(s): {', '.join(unknown_auto)}"
            )
        overlap = set(autosomes) & set(sex_contigs)
        if overlap:
            raise ReferenceError(
                "autosomal and sex contig sets must be disjoint: "
                + ", ".join(sorted(overlap))
            )
    if not autosomes:
        raise ReferenceError("reference requires at least one autosomal contig")
    return autosomes, sex_contigs


def _resolve_sexes(
    cohort: Cohort,
    config: ReferenceConfig,
    sex_contigs: tuple[str, ...],
) -> tuple[tuple[SexLabel, ...] | None, str, SexInference | None, str | None]:
    signal = _resolve_sex_signal(cohort, config)
    if not sex_contigs or signal is None:
        return None, "none", None, signal
    if (
        config.entries
        and len(config.entries) == cohort.counts.shape[1]
        and all(entry.sex in ("M", "F") for entry in config.entries)
    ):
        labels: tuple[SexLabel, ...] = tuple(
            "male" if entry.sex == "M" else "female" for entry in config.entries
        )
        return labels, "samplesheet", None, signal
    fractions = _contig_fractions(cohort, signal)
    inference = infer_sample_sexes(
        fractions,
        config.min_sex_group_size,
        cutoff=config.sex_cutoff,
    )
    if not inference.passed:
        return None, "none", inference, signal
    labels = list(inference.labels)
    if config.entries and len(config.entries) == len(labels):
        for i, entry in enumerate(config.entries):
            if entry.sex == "M":
                labels[i] = "male"
            elif entry.sex == "F":
                labels[i] = "female"
    return tuple(labels), "inferred", inference, signal


def _resolve_sex_signal(cohort: Cohort, config: ReferenceConfig) -> str | None:
    available = set(cohort.contigs)
    requested = config.sex_signal_contig
    if requested is None or requested == DEFAULT_SEX_SIGNAL_CONTIGS[0]:
        for name in DEFAULT_SEX_SIGNAL_CONTIGS:
            if name in available:
                return name
        return None
    if requested not in available:
        raise ReferenceError(f"unknown sex-signal contig {requested!r}")
    return requested


def _validate_sex_cutoff(cutoff: float | None) -> None:
    if cutoff is None:
        return
    if not math.isfinite(cutoff) or cutoff < 0.0 or cutoff > 1.0:
        raise ReferenceError("sex_cutoff must be a finite number in [0, 1]")


def _contig_fractions(cohort: Cohort, contig: str) -> NDArray[np.float64]:
    totals = cohort.counts.sum(axis=0)
    if np.any(totals <= 0):
        raise ReferenceError("every sample must have a positive total count")
    mask = np.array([name == contig for name in cohort.contigs], dtype=np.bool_)
    return cohort.counts[mask].sum(axis=0) / totals


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
    selected = np.array(
        [name in set(contig_names) for name in cohort.contigs],
        dtype=np.bool_,
    )
    if not np.any(selected):
        raise ReferenceError(f"branch {branch!r} has no bins")
    counts = cohort.counts[np.ix_(selected, sample_index)]
    branch_gc = cohort_gc[selected] if cohort_gc is not None else None
    branch_map = (
        cohort_mappability[selected] if cohort_mappability is not None else None
    )
    branch_n_mask = cohort_n_mask[selected] if cohort_n_mask is not None else None
    logger.debug(
        "building %s branch: %d bins, %d samples",
        branch,
        counts.shape[0],
        counts.shape[1],
    )
    initial_normalized = normalize_depth(counts)
    usable = mask_low_coverage(initial_normalized)
    usable = mask_covariates(
        usable,
        gc=branch_gc,
        mappability=branch_map,
        n_mask=branch_n_mask,
        min_mappability=config.min_mappability,
    )
    if not np.any(usable):
        raise ReferenceError(f"branch {branch!r} has no unmasked bins")
    normalized = normalize_depth(counts, mask=usable)
    n_unmasked = int(np.count_nonzero(usable))
    logger.debug(
        "%s mask kept %d/%d bins (%.1f%% unmasked, %d masked)",
        branch,
        n_unmasked,
        usable.size,
        100.0 * n_unmasked / usable.size,
        usable.size - n_unmasked,
    )

    normalized = correct_covariate_bias(
        normalized, usable, gc=branch_gc, mappability=branch_map
    )

    logger.debug("fitting PCA for %s (%d components)", branch, config.pca_components)
    corrected, pca_mean, pca_loadings = fit_pca(
        normalized[usable], config.pca_components
    )
    contig_subset = np.array(cohort.contigs, dtype=object)[selected]
    unmasked_contigs = np.asarray(contig_subset[usable], dtype=str)
    _, contig_ids = np.unique(unmasked_contigs, return_inverse=True)
    logger.debug(
        "selecting %d nearest off-contig bins for %s (%d unmasked bins)",
        config.nearest_bins,
        branch,
        n_unmasked,
    )
    search_profiles = corrected
    if config.fasta is not None:
        cov_mat = covariate_features(usable, gc=branch_gc, mappability=branch_map)
        if cov_mat is not None:
            search_profiles = cov_mat
    nearest_indices, nearest_distances = select_nearest_bins(
        search_profiles,
        contig_ids,
        k=config.nearest_bins,
        max_batch_memory_mb=config.max_batch_memory_mb,
        threads=config.threads,
    )
    null_mean, null_std = compute_null_models(
        corrected,
        nearest_indices,
        threads=config.threads,
    )
    valid_nulls = null_mean[np.isfinite(null_mean)]
    logger.debug(
        "%s null models computed for %d bins (mean=%.4f, std=%.4f)",
        branch,
        len(null_mean),
        float(np.mean(valid_nulls)) if valid_nulls.size > 0 else 0.0,
        float(np.std(valid_nulls)) if valid_nulls.size > 0 else 0.0,
    )

    flat_distances = [d for sub in nearest_distances for d in sub if math.isfinite(d)]
    dist_arr = np.asarray(flat_distances, dtype=np.float64)
    cutoff = distance_cutoff(dist_arr, repeats=3)
    logger.debug("%s distance cutoff computed: %.4f", branch, cutoff)

    filtered_nearest: list[list[int]] = []
    mean_distances: list[float] = []
    for d_list, idx_list in zip(nearest_distances, nearest_indices, strict=True):
        d_arr = np.asarray(d_list, dtype=np.float64)
        idx_arr = np.asarray(idx_list, dtype=np.int32)
        valid = (d_arr <= cutoff) & np.isfinite(d_arr)
        if np.any(valid):
            mean_dist = float(np.mean(np.sqrt(np.maximum(d_arr[valid], 0.0))))
            mean_distances.append(mean_dist)
            filtered_idx = np.sort(idx_arr[valid])
            filtered_nearest.append(filtered_idx.tolist())
        else:
            mean_distances.append(0.0)
            filtered_nearest.append([])

    n_bins = int(np.count_nonzero(selected))
    starts = cohort.starts[selected]
    ends = cohort.ends[selected]

    out_gc = np.full(n_bins, -1.0, dtype=np.float32)
    if branch_gc is not None:
        gc_finite = np.isfinite(branch_gc)
        out_gc[gc_finite] = branch_gc[gc_finite]

    out_map = np.full(n_bins, -1.0, dtype=np.float32)
    if branch_map is not None:
        map_finite = np.isfinite(branch_map)
        out_map[map_finite] = branch_map[map_finite]

    branch_pca_mean = np.zeros(n_bins, dtype=np.float64)
    branch_pca_mean[usable] = pca_mean

    branch_mean_dist = np.zeros(n_bins, dtype=np.float32)
    branch_mean_dist[usable] = mean_distances

    branch_null_mean = np.zeros(n_bins, dtype=np.float64)
    branch_null_mean[usable] = null_mean

    branch_null_std = np.zeros(n_bins, dtype=np.float64)
    branch_null_std[usable] = null_std

    full_pca_loadings: list[list[float]] = [[] for _ in range(n_bins)]
    full_nearest: list[list[int]] = [[] for _ in range(n_bins)]
    unmasked_indices = np.flatnonzero(usable)
    for u_idx, orig_idx in enumerate(unmasked_indices):
        full_pca_loadings[orig_idx] = pca_loadings[:, u_idx].tolist()
        full_nearest[orig_idx] = filtered_nearest[u_idx]

    return pa.table(
        {
            "branch": pa.array([branch] * n_bins, type=pa.string()),
            "contig": pa.array(contig_subset.tolist(), type=pa.string()),
            "start": pa.array(starts, type=pa.int64()),
            "end": pa.array(ends, type=pa.int64()),
            "masked": pa.array(~usable, type=pa.bool_()),
            "gc": pa.array(out_gc, type=pa.float32()),
            "mappability": pa.array(out_map, type=pa.float32()),
            "pca_mean": pa.array(branch_pca_mean, type=pa.float64()),
            "pca_loadings": pa.array(full_pca_loadings, type=pa.list_(pa.float64())),
            "nearest_ref_indices": pa.array(full_nearest, type=pa.list_(pa.int32())),
            "mean_distance": pa.array(branch_mean_dist, type=pa.float32()),
            "null_mean": pa.array(branch_null_mean, type=pa.float64()),
            "null_std": pa.array(branch_null_std, type=pa.float64()),
        },
        schema=REFERENCE_SCHEMA,
    )
