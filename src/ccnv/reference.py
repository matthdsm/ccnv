"""Cohort validation and vectorized reference construction."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray
from sklearn.decomposition import PCA

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
from ccnv.convert import rebin_sample_table
from ccnv.covariates import (
    CovariateError,
    correct_covariate_bias,
    covariate_features,
    load_gc_track,
    load_mappability_track,
    mask_covariates,
    validate_min_mappability,
)
from ccnv.models import ReadFilters, ReferenceConfig, SamplesheetEntry, parquet_path
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
    BRANCH_PRENATAL_AUTOSOMAL,
    BRANCH_PRENATAL_FFY,
    REFERENCE_SCHEMA,
    SAMPLE_SCHEMA,
    ArtifactKind,
    decode_metadata,
    metadata_for,
    validate_table,
)

BRANCH_AUTOSOMAL: Final[str] = "autosomal"
BRANCH_FEMALE_SEX: Final[str] = "female_sex"
BRANCH_MALE_SEX: Final[str] = "male_sex"
DEFAULT_SEX_CONTIGS: Final[tuple[str, ...]] = ("X", "Y", "chrX", "chrY")
DEFAULT_SEX_SIGNAL_CONTIGS: Final[tuple[str, ...]] = ("chrY", "Y")

SexLabel = Literal["female", "male"]
_MASK_FRACTION: Final[float] = 0.05
_MAX_NULL_SAMPLES: Final[int] = 100
_SEPARATION_MAD_MULTIPLIER: Final[float] = 4.0

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SexInference:
    """Deterministic sex assignments, or a rejected inference."""

    labels: tuple[SexLabel, ...] | None
    cutoff: float | None
    fractions: tuple[float, ...]
    passed: bool
    reason: str


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
            input_paths=sample_paths,
        )
    elif config.entries:
        entries = config.entries
        sample_paths = tuple(entry.filepath for entry in entries)
        config = replace(
            config,
            samples=sample_paths,
            input_paths=sample_paths,
        )
    elif config.input_paths:
        entries = tuple(SamplesheetEntry(filepath=p) for p in config.input_paths)
        config = replace(
            config,
            entries=entries,
            samples=config.input_paths,
        )
    elif config.samples:
        entries = tuple(SamplesheetEntry(filepath=p) for p in config.samples)
        config = replace(
            config,
            entries=entries,
            input_paths=config.samples,
        )
    else:
        entries = ()

    sample_tables: list[pa.Table] | None = None
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
        if config.mode == "prenatal":
            sample_tables = _load_sample_tables(
                entries,
                target_bin_size=config.bin_size,
            )
    logger.debug(
        "loaded cohort: %d bins, %d samples, bin_size=%d",
        cohort.counts.shape[0],
        cohort.counts.shape[1],
        cohort.bin_size,
    )

    table = build_reference_table(cohort, config, sample_tables=sample_tables)
    validate_table(table, ArtifactKind.REFERENCE)
    path = parquet_path(config.prefix)
    logger.debug("writing reference parquet %s (%d rows)", path, table.num_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=3,
        use_dictionary=["branch", "contig"],
        column_encoding={
            "start": "DELTA_BINARY_PACKED",
            "end": "DELTA_BINARY_PACKED",
            "nearest_ref_indices.list.element": "DELTA_BINARY_PACKED",
        },
    )
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


def normalize_depth(
    counts: NDArray[np.floating],
    mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Scale each sample to unit total depth, optionally on usable bins."""

    matrix = np.asarray(counts, dtype=np.float64)
    sub = matrix[mask] if mask is not None else matrix
    totals = sub.sum(axis=0)
    if np.any(totals <= 0):
        suffix = " on usable bins" if mask is not None else ""
        raise ReferenceError(f"every sample must have a positive total count{suffix}")
    return matrix / totals


def mask_low_coverage(normalized: NDArray[np.floating]) -> NDArray[np.bool_]:
    """Mask bins with no coverage or coverage below 5% of the positive median."""

    per_bin = np.asarray(normalized, dtype=np.float64).sum(axis=1)
    positive = per_bin[per_bin > 0]
    if positive.size == 0:
        return np.zeros(per_bin.shape, dtype=np.bool_)
    threshold = _MASK_FRACTION * float(np.median(positive))
    return per_bin > threshold


def fit_pca(
    masked_normalized: NDArray[np.floating],
    n_components: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Fit a deterministic PCA model on samples-by-bins and correct coverage."""

    data = np.asarray(masked_normalized, dtype=np.float64)
    if data.ndim != 2 or data.size == 0:
        raise ReferenceError("PCA requires a non-empty bins-by-samples matrix")
    samples = data.T
    n_components = min(n_components, samples.shape[0], samples.shape[1])
    if n_components < 1:
        raise ReferenceError("PCA requires at least one sample and one unmasked bin")
    mean = np.mean(samples, axis=0)
    if np.allclose(samples, mean):
        return (
            np.ones(data.shape, dtype=np.float64),
            mean.astype(np.float64, copy=False),
            np.zeros((n_components, samples.shape[1]), dtype=np.float64),
        )
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(samples)
    if (
        hasattr(pca, "explained_variance_ratio_")
        and pca.explained_variance_ratio_ is not None
    ):
        finite_vars = np.nan_to_num(pca.explained_variance_ratio_ * 100.0)
        logger.debug(
            "PCA fitted %d components: explained variance [%s] (total=%.2f%%)",
            n_components,
            ", ".join(f"{v:.2f}%" for v in finite_vars),
            float(np.sum(finite_vars)),
        )
    reconstructed = pca.inverse_transform(pca.transform(samples))
    with np.errstate(divide="ignore", invalid="ignore"):
        corrected = samples / reconstructed
    corrected = np.nan_to_num(corrected, nan=1.0, posinf=1.0, neginf=1.0)
    return (
        corrected.T.astype(np.float64, copy=False),
        pca.mean_.astype(np.float64, copy=False),
        pca.components_.astype(np.float64, copy=False),
    )


def infer_sample_sexes(
    fractions: NDArray[np.floating],
    min_group_size: int,
    min_separation: float = 0.0,
    cutoff: float | None = None,
) -> SexInference:
    """Split Y-like fractions into female/male groups when separation is clear."""

    values = np.asarray(fractions, dtype=np.float64)
    recorded = tuple(float(value) for value in values)
    if min_group_size < 1:
        raise ReferenceError("min_sex_group_size must be a positive integer")
    if cutoff is not None:
        return _labels_from_cutoff(values, recorded, cutoff)
    if values.size < 2 * min_group_size:
        return SexInference(
            labels=None,
            cutoff=None,
            fractions=recorded,
            passed=False,
            reason="cohort is too small for sex inference",
        )

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    best_gap = -np.inf
    best_split = -1
    for split in range(min_group_size, values.size - min_group_size + 1):
        gap = float(sorted_values[split] - sorted_values[split - 1])
        if gap > best_gap:
            best_gap = gap
            best_split = split

    if best_split < 0 or best_gap <= 0:
        return SexInference(
            labels=None,
            cutoff=None,
            fractions=recorded,
            passed=False,
            reason="sex-signal fractions are not separable",
        )

    low = sorted_values[:best_split]
    high = sorted_values[best_split:]
    spread = max(
        _median_abs_deviation(low),
        _median_abs_deviation(high),
        1e-12,
    )
    cutoff = float((low[-1] + high[0]) / 2.0)
    if best_gap < max(min_separation, _SEPARATION_MAD_MULTIPLIER * spread):
        return SexInference(
            labels=None,
            cutoff=cutoff,
            fractions=recorded,
            passed=False,
            reason="sex-signal groups are not sufficiently separated",
        )

    labels = ["female"] * values.size
    for index in order[best_split:]:
        labels[int(index)] = "male"
    return SexInference(
        labels=tuple(labels),
        cutoff=cutoff,
        fractions=recorded,
        passed=True,
        reason="separated",
    )


def _sample_tables_from_cohort(cohort: Cohort) -> list[pa.Table]:
    tables: list[pa.Table] = []
    for i in range(cohort.counts.shape[1]):
        metadata = metadata_for(
            ArtifactKind.SAMPLE,
            contig_lengths=cohort.contig_lengths,
            values={
                "bin_size": cohort.bin_size,
                "source_identity": (
                    cohort.identities[i]
                    if i < len(cohort.identities)
                    else {"id": f"sample_{i}"}
                ),
                "reference_identity": {"name": "test"},
                "qc": {},
                "filters": asdict(ReadFilters()),
            },
        )
        table = pa.table(
            {
                "contig": pa.array(cohort.contigs, type=pa.string()),
                "start": pa.array(cohort.starts, type=pa.int64()),
                "end": pa.array(cohort.ends, type=pa.int64()),
                "count": pa.array(
                    cohort.counts[:, i].astype(np.uint64), type=pa.uint64()
                ),
            },
            schema=SAMPLE_SCHEMA.with_metadata(metadata),
        )
        tables.append(table)
    return tables


def _load_sample_tables(
    entries: Sequence[SamplesheetEntry],
    target_bin_size: int | None = None,
) -> list[pa.Table]:
    tables: list[pa.Table] = []
    for entry in entries:
        table = pq.read_table(entry.filepath)
        if target_bin_size is not None:
            metadata = decode_metadata(table.schema.metadata)
            if metadata.get("bin_size") != target_bin_size:
                table = rebin_sample_table(table, target_bin_size)
        tables.append(table)
    return tables


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
        if sample_tables is None:
            if config.entries and all(e.filepath.is_file() for e in config.entries):
                sample_tables = _load_sample_tables(
                    config.entries, target_bin_size=effective_bin_size
                )
            else:
                sample_tables = _sample_tables_from_cohort(cohort)

        entries = config.entries
        if len(entries) != len(sample_tables):
            entries = tuple(
                SamplesheetEntry(filepath=Path(f"sample_{i}.ccnv"))
                for i in range(len(sample_tables))
            )
            config = replace(config, entries=entries)

        seed = getattr(config, "seed", 42)
        (
            prenatal_auto_table,
            prenatal_ffy_table,
            prenatal_meta,
            is_male_tuple,
        ) = fit_prenatal_models(
            sample_tables,
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
    rows = _empty_branch_columns()
    built_branches: list[str] = []

    _append_branch(
        rows,
        branch=BRANCH_AUTOSOMAL,
        cohort=cohort,
        sample_index=np.arange(cohort.counts.shape[1]),
        contig_names=autosomes,
        config=config,
        cohort_gc=cohort_gc,
        cohort_mappability=cohort_mappability,
        cohort_n_mask=cohort_n_mask,
    )
    built_branches.append(BRANCH_AUTOSOMAL)

    if sex_contigs and sexes is not None:
        labels = np.asarray(sexes)
        female_index = np.flatnonzero(labels == "female")
        male_index = np.flatnonzero(labels == "male")
        if female_index.size >= config.min_sex_group_size:
            _append_branch(
                rows,
                branch=BRANCH_FEMALE_SEX,
                cohort=cohort,
                sample_index=female_index,
                contig_names=sex_contigs,
                config=config,
                cohort_gc=cohort_gc,
                cohort_mappability=cohort_mappability,
                cohort_n_mask=cohort_n_mask,
            )
            built_branches.append(BRANCH_FEMALE_SEX)
        if male_index.size >= config.min_sex_group_size:
            _append_branch(
                rows,
                branch=BRANCH_MALE_SEX,
                cohort=cohort,
                sample_index=male_index,
                contig_names=sex_contigs,
                config=config,
                cohort_gc=cohort_gc,
                cohort_mappability=cohort_mappability,
                cohort_n_mask=cohort_n_mask,
            )
            built_branches.append(BRANCH_MALE_SEX)

    cnv_table = pa.table(
        {
            "branch": pa.array(rows["branch"], type=pa.string()),
            "contig": pa.array(rows["contig"], type=pa.string()),
            "start": pa.array(rows["start"], type=pa.int64()),
            "end": pa.array(rows["end"], type=pa.int64()),
            "masked": pa.array(rows["masked"], type=pa.bool_()),
            "gc": pa.array(rows["gc"], type=pa.float32()),
            "mappability": pa.array(rows["mappability"], type=pa.float32()),
            "pca_mean": pa.array(rows["pca_mean"], type=pa.float64()),
            "pca_loadings": pa.array(rows["pca_loadings"], type=pa.list_(pa.float64())),
            "nearest_ref_indices": pa.array(
                rows["nearest_ref_indices"], type=pa.list_(pa.int32())
            ),
            "mean_distance": pa.array(rows["mean_distance"], type=pa.float32()),
            "null_mean": pa.array(rows["null_mean"], type=pa.float64()),
            "null_std": pa.array(rows["null_std"], type=pa.float64()),
        },
        schema=REFERENCE_SCHEMA,
    )

    if config.mode == "prenatal":
        assert prenatal_auto_table is not None and prenatal_ffy_table is not None
        if prenatal_auto_table.num_rows > 0:
            built_branches.append(BRANCH_PRENATAL_AUTOSOMAL)
        if prenatal_ffy_table.num_rows > 0:
            built_branches.append(BRANCH_PRENATAL_FFY)
        full_table = pa.concat_tables(
            [cnv_table, prenatal_auto_table, prenatal_ffy_table]
        )
    else:
        full_table = cnv_table

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
    fractions = _contig_fractions(cohort, signal)
    inference = infer_sample_sexes(
        fractions,
        config.min_sex_group_size,
        cutoff=config.sex_cutoff,
    )
    if not inference.passed:
        return None, "none", inference, signal
    return inference.labels, "inferred", inference, signal


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


def _labels_from_cutoff(
    values: NDArray[np.float64],
    recorded: tuple[float, ...],
    cutoff: float,
) -> SexInference:
    _validate_sex_cutoff(cutoff)
    labels = tuple("male" if float(value) > cutoff else "female" for value in values)
    return SexInference(
        labels=labels,
        cutoff=float(cutoff),
        fractions=recorded,
        passed=True,
        reason="cutoff",
    )


def _contig_fractions(cohort: Cohort, contig: str) -> NDArray[np.float64]:
    totals = cohort.counts.sum(axis=0)
    if np.any(totals <= 0):
        raise ReferenceError("every sample must have a positive total count")
    mask = np.array([name == contig for name in cohort.contigs], dtype=np.bool_)
    return cohort.counts[mask].sum(axis=0) / totals


def distance_cutoff(
    distances: (
        tuple[NDArray[np.float64], ...]
        | NDArray[np.floating]
        | Sequence[Sequence[float]]
    ),
    repeats: int = 3,
) -> float:
    """Return an iterative three-sigma cutoff for nearest-bin distances."""

    if isinstance(distances, np.ndarray):
        values = distances.ravel()
    elif (
        isinstance(distances, tuple)
        and distances
        and isinstance(distances[0], np.ndarray)
    ):
        values = np.concatenate(distances)
    else:
        values = np.asarray([d for sub in distances for d in sub], dtype=np.float64)

    values = values[np.isfinite(values) & (values >= 0)]
    if values.size == 0:
        return float("inf")
    cutoff = float("inf")
    for _ in range(repeats):
        selected = values[values < cutoff]
        if selected.size == 0:
            break
        cutoff = float(np.mean(selected) + 3.0 * np.std(selected))
    return cutoff


def _empty_branch_columns() -> dict[str, list[object]]:
    return {
        "branch": [],
        "contig": [],
        "start": [],
        "end": [],
        "masked": [],
        "gc": [],
        "mappability": [],
        "pca_mean": [],
        "pca_loadings": [],
        "nearest_ref_indices": [],
        "mean_distance": [],
        "null_mean": [],
        "null_std": [],
    }


def _append_branch(
    rows: dict[str, list[object]],
    *,
    branch: str,
    cohort: Cohort,
    sample_index: NDArray[np.integer],
    contig_names: tuple[str, ...],
    config: ReferenceConfig,
    cohort_gc: NDArray[np.float64] | None = None,
    cohort_mappability: NDArray[np.float64] | None = None,
    cohort_n_mask: NDArray[np.bool_] | None = None,
) -> None:
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

    unmasked_iter = 0
    starts = cohort.starts[selected]
    ends = cohort.ends[selected]
    for local, contig in enumerate(contig_subset.tolist()):
        rows["branch"].append(branch)
        rows["contig"].append(contig)
        rows["start"].append(int(starts[local]))
        rows["end"].append(int(ends[local]))
        rows["gc"].append(
            float(branch_gc[local])
            if branch_gc is not None and math.isfinite(float(branch_gc[local]))
            else -1.0
        )
        rows["mappability"].append(
            float(branch_map[local])
            if branch_map is not None and math.isfinite(float(branch_map[local]))
            else -1.0
        )
        if not usable[local]:
            rows["masked"].append(True)
            rows["pca_mean"].append(0.0)
            rows["pca_loadings"].append([])
            rows["nearest_ref_indices"].append([])
            rows["mean_distance"].append(0.0)
            rows["null_mean"].append(0.0)
            rows["null_std"].append(0.0)
            continue
        rows["masked"].append(False)
        rows["pca_mean"].append(float(pca_mean[unmasked_iter]))
        rows["pca_loadings"].append(pca_loadings[:, unmasked_iter].tolist())
        rows["nearest_ref_indices"].append(filtered_nearest[unmasked_iter])
        rows["mean_distance"].append(mean_distances[unmasked_iter])
        rows["null_mean"].append(float(null_mean[unmasked_iter]))
        rows["null_std"].append(float(null_std[unmasked_iter]))
        unmasked_iter += 1


def _median_abs_deviation(values: NDArray[np.floating]) -> float:
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))
