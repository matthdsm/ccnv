"""Cohort validation and vectorized reference construction."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pysam
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from ccnv.convert import bin_intervals, rebin_sample_table, select_layout
from ccnv.models import ReferenceConfig, parquet_path
from ccnv.reference_qc import run_reference_qc_on_table
from ccnv.schemas import (
    REFERENCE_SCHEMA,
    ArtifactKind,
    SchemaError,
    bin_coordinates,
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


class ReferenceError(ValueError):
    """Raised when a cohort cannot be turned into a reference."""


@dataclass(frozen=True, slots=True)
class Cohort:
    """Aligned binned counts from compatible sample Parquet files."""

    counts: NDArray[np.float64]
    contigs: tuple[str, ...]
    starts: NDArray[np.int64]
    ends: NDArray[np.int64]
    contig_lengths: dict[str, int]
    bin_size: int
    identities: tuple[object, ...]
    paths: tuple[Path, ...]


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
        cohort = load_cohort(config.samples, target_bin_size=config.bin_size)
    logger.debug(
        "loaded cohort: %d bins, %d samples, bin_size=%d",
        cohort.counts.shape[0],
        cohort.counts.shape[1],
        cohort.bin_size,
    )

    table = build_reference_table(cohort, config)
    validate_table(table, ArtifactKind.REFERENCE)
    path = parquet_path(config.prefix)
    logger.debug("writing reference parquet %s (%d rows)", path, table.num_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
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


def load_cohort(
    paths: tuple[Path, ...],
    target_bin_size: int | None = None,
) -> Cohort:
    """Load sample Parquet files and reject incompatible cohorts."""

    if not paths:
        raise ReferenceError("at least one sample Parquet path is required")
    tables = tuple(_read_sample(path) for path in paths)
    return validate_cohort(tables, paths, target_bin_size=target_bin_size)


def cohort_from_fasta(
    fasta: Path,
    bin_size: int,
    *,
    n_samples: int,
    all_contigs: bool = False,
) -> Cohort:
    """Build identical uniform-depth samples from FASTA contig lengths."""

    if n_samples < 1:
        raise ReferenceError("n_samples must be a positive integer")
    names, lengths = _fasta_contigs(fasta, all_contigs=all_contigs)
    contigs: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for name, length in zip(names, lengths, strict=True):
        contig_starts, contig_ends = bin_intervals(length, bin_size)
        contigs.extend([name] * contig_starts.size)
        starts.extend(int(value) for value in contig_starts)
        ends.extend(int(value) for value in contig_ends)
    start_array = np.asarray(starts, dtype=np.int64)
    end_array = np.asarray(ends, dtype=np.int64)
    expected = np.maximum(end_array - start_array, 1).astype(np.float64)
    counts = np.broadcast_to(expected[:, np.newaxis], (expected.size, n_samples)).copy()
    resolved = fasta.expanduser().resolve()
    identities = tuple(
        {
            "path": str(resolved),
            "format": "flat-fasta",
            "replica": index,
        }
        for index in range(n_samples)
    )
    return Cohort(
        counts=counts,
        contigs=tuple(contigs),
        starts=start_array,
        ends=end_array,
        contig_lengths=dict(zip(names, lengths, strict=True)),
        bin_size=bin_size,
        identities=identities,
        paths=tuple(Path(f"{resolved}#flat-{index}") for index in range(n_samples)),
    )


def _fasta_contigs(path: Path, *, all_contigs: bool) -> tuple[list[str], list[int]]:
    fasta = path.expanduser()
    if not fasta.is_file():
        raise ReferenceError(f"FASTA not found: {fasta}")
    if not Path(f"{fasta}.fai").is_file():
        logger.debug("creating FASTA index for %s", fasta)
        try:
            pysam.faidx(str(fasta))
        except Exception as error:
            raise ReferenceError(f"failed to index FASTA {fasta}: {error}") from error
    handle = pysam.FastaFile(str(fasta))
    try:
        header_names = list(handle.references)
        header_lengths = [int(length) for length in handle.lengths]
    finally:
        handle.close()
    if any(length <= 0 for length in header_lengths):
        raise ReferenceError(f"{fasta} contains non-positive contig lengths")
    layout = select_layout(header_names, header_lengths, all_contigs=all_contigs)
    if not layout:
        raise ReferenceError(
            f"{fasta} declares no canonical chromosomes; "
            "pass --all-contigs to include every contig"
        )
    return [contig.name for contig in layout], [contig.length for contig in layout]


def validate_cohort(
    tables: tuple[pa.Table, ...],
    paths: tuple[Path, ...] | None = None,
    target_bin_size: int | None = None,
) -> Cohort:
    """Require matching contig lengths across samples, rebinning counts if set."""

    if len(tables) < 1:
        raise ReferenceError("cohort must contain at least one sample")
    resolved_paths = paths or tuple(
        Path(f"sample_{index}") for index in range(len(tables))
    )
    if len(resolved_paths) != len(tables):
        raise ReferenceError("paths must align with sample tables")

    for path, table in zip(resolved_paths, tables, strict=True):
        try:
            validate_table(table, ArtifactKind.SAMPLE)
        except SchemaError as error:
            raise ReferenceError(f"{path}: {error}") from error

    expected_lengths: dict[str, int] | None = None
    expected_bin_size: int | None = None

    for path, table in zip(resolved_paths, tables, strict=True):
        metadata = decode_metadata(table.schema.metadata)
        contig_lengths = metadata["contig_lengths"]
        bin_size = metadata["bin_size"]
        if not isinstance(contig_lengths, dict) or not isinstance(bin_size, int):
            raise ReferenceError(f"{path}: sample metadata is incomplete")
        current_lengths = {
            str(name): int(length) for name, length in contig_lengths.items()
        }
        if expected_lengths is None:
            expected_lengths = current_lengths
            expected_bin_size = bin_size
        elif current_lengths != expected_lengths:
            raise ReferenceError(f"{path}: contig_lengths do not match the cohort")
        elif target_bin_size is None and bin_size != expected_bin_size:
            raise ReferenceError(
                f"{path}: bin_size {bin_size} does not match cohort {expected_bin_size}"
            )

    assert expected_lengths is not None
    effective_bin_size = (
        target_bin_size if target_bin_size is not None else expected_bin_size
    )
    assert effective_bin_size is not None

    processed_tables: list[pa.Table] = []
    for path, table in zip(resolved_paths, tables, strict=True):
        metadata = decode_metadata(table.schema.metadata)
        orig_bin_size = metadata.get("bin_size")
        if orig_bin_size != effective_bin_size:
            logger.debug(
                "rebinning cohort sample %s from %s bp to %s bp",
                path.name,
                orig_bin_size,
                effective_bin_size,
            )
            table = rebin_sample_table(table, effective_bin_size)
        processed_tables.append(table)

    identities: list[object] = []
    counts: list[NDArray[np.float64]] = []
    expected_bins: tuple[tuple[str, int, int], ...] | None = None

    for path, table in zip(resolved_paths, processed_tables, strict=True):
        metadata = decode_metadata(table.schema.metadata)
        bins = bin_coordinates(table)
        if expected_bins is None:
            expected_bins = bins
        elif bins != expected_bins:
            raise ReferenceError(f"{path}: bin coordinates do not match the cohort")
        identities.append(metadata.get("source_identity", {"path": str(path)}))
        col_counts = np.asarray(table.column("count").to_pylist(), dtype=np.float64)
        logger.debug(
            "loaded cohort sample %s: %d bins, %d total depth",
            path.name,
            table.num_rows,
            int(np.sum(col_counts)),
        )
        counts.append(col_counts)

    assert expected_bins is not None
    return Cohort(
        counts=np.column_stack(counts),
        contigs=tuple(contig for contig, _, _ in expected_bins),
        starts=np.array([start for _, start, _ in expected_bins], dtype=np.int64),
        ends=np.array([end for _, _, end in expected_bins], dtype=np.int64),
        contig_lengths=expected_lengths,
        bin_size=effective_bin_size,
        identities=tuple(identities),
        paths=resolved_paths,
    )


def normalize_depth(counts: NDArray[np.floating]) -> NDArray[np.float64]:
    """Scale each sample to unit total depth."""

    matrix = np.asarray(counts, dtype=np.float64)
    totals = matrix.sum(axis=0)
    if np.any(totals <= 0):
        raise ReferenceError("every sample must have a positive total count")
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


def select_nearest_bins(
    profiles: NDArray[np.floating],
    contig_ids: NDArray[np.integer],
    k: int,
    max_batch_memory_mb: int,
) -> tuple[list[list[int]], list[list[float]]]:
    """Pick k nearest off-contig bins with batched distances and argpartition."""

    data = np.asarray(profiles, dtype=np.float64)
    contig_ids = np.asarray(contig_ids)
    n_bins = data.shape[0]
    if k < 1:
        raise ReferenceError("nearest_bins must be a positive integer")
    if n_bins == 0:
        return [], []

    max_bytes = max(int(max_batch_memory_mb * 1024 * 1024), 1)
    bytes_per_query = max(n_bins, 1) * data.dtype.itemsize
    batch_size = max(1, max_bytes // bytes_per_query)
    candidate_norm = np.sum(data * data, axis=1)

    indices: list[list[int]] = [[] for _ in range(n_bins)]
    distances: list[list[float]] = [[] for _ in range(n_bins)]
    for start in range(0, n_bins, batch_size):
        stop = min(start + batch_size, n_bins)
        batch = data[start:stop]
        query_norm = np.sum(batch * batch, axis=1, keepdims=True)
        dist = query_norm + candidate_norm - 2.0 * (batch @ data.T)
        np.maximum(dist, 0.0, out=dist)
        for offset, query in enumerate(range(start, stop)):
            row = dist[offset]
            row[contig_ids == contig_ids[query]] = np.inf
            available = int(np.count_nonzero(np.isfinite(row)))
            if available == 0:
                continue
            take = min(k, available)
            partitioned = np.argpartition(row, take - 1)[:take]
            ordered = partitioned[np.argsort(row[partitioned], kind="mergesort")]
            indices[query] = [int(index) for index in ordered]
            distances[query] = [float(row[index]) for index in ordered]
        del dist, batch, query_norm
    return indices, distances


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


def build_reference_table(cohort: Cohort, config: ReferenceConfig) -> pa.Table:
    """Fit autosomal and optional sex-contig branches into one reference table."""

    if cohort.counts.shape[1] < config.min_cohort_size:
        raise ReferenceError(
            f"provide at least {config.min_cohort_size} samples to build a reference"
        )

    autosomes, sex_contigs = _resolve_contig_roles(cohort, config)
    logger.debug(
        "resolved contig roles: autosomal=%s sex=%s",
        autosomes,
        sex_contigs,
    )
    sexes, sex_source, inference, signal = _resolve_sexes(cohort, config, sex_contigs)
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
            )
            built_branches.append(BRANCH_MALE_SEX)

    inference_meta: dict[str, object] | None = None
    if inference is not None:
        inference_meta = {
            "passed": inference.passed,
            "cutoff": inference.cutoff,
            "fractions": list(inference.fractions),
            "reason": inference.reason,
        }
    metadata = metadata_for(
        ArtifactKind.REFERENCE,
        contig_lengths=cohort.contig_lengths,
        values={
            "sample_identities": list(cohort.identities),
            "model": {
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
            },
        },
    )
    return pa.table(
        {
            "branch": pa.array(rows["branch"], type=pa.string()),
            "contig": pa.array(rows["contig"], type=pa.string()),
            "start": pa.array(rows["start"], type=pa.int64()),
            "end": pa.array(rows["end"], type=pa.int64()),
            "masked": pa.array(rows["masked"], type=pa.bool_()),
            "pca_mean": pa.array(rows["pca_mean"], type=pa.float64()),
            "pca_loadings": pa.array(rows["pca_loadings"], type=pa.list_(pa.float64())),
            "nearest_ref_indices": pa.array(
                rows["nearest_ref_indices"], type=pa.list_(pa.int32())
            ),
            "nearest_ref_distances": pa.array(
                rows["nearest_ref_distances"], type=pa.list_(pa.float64())
            ),
            "null_ratios": pa.array(rows["null_ratios"], type=pa.list_(pa.float64())),
        },
        schema=REFERENCE_SCHEMA.with_metadata(metadata),
    )


def _validate_config(config: ReferenceConfig) -> None:
    has_samples = bool(config.samples)
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
    _validate_sex_cutoff(config.sex_cutoff)


def _read_sample(path: Path) -> pa.Table:
    try:
        return pq.read_table(path)
    except (FileNotFoundError, OSError, pa.ArrowInvalid) as error:
        raise ReferenceError(f"{path}: cannot read sample Parquet ({error})") from error


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


def _empty_branch_columns() -> dict[str, list[object]]:
    return {
        "branch": [],
        "contig": [],
        "start": [],
        "end": [],
        "masked": [],
        "pca_mean": [],
        "pca_loadings": [],
        "nearest_ref_indices": [],
        "nearest_ref_distances": [],
        "null_ratios": [],
    }


def _append_branch(
    rows: dict[str, list[object]],
    *,
    branch: str,
    cohort: Cohort,
    sample_index: NDArray[np.integer],
    contig_names: tuple[str, ...],
    config: ReferenceConfig,
) -> None:
    selected = np.array(
        [name in set(contig_names) for name in cohort.contigs],
        dtype=np.bool_,
    )
    if not np.any(selected):
        raise ReferenceError(f"branch {branch!r} has no bins")
    counts = cohort.counts[np.ix_(selected, sample_index)]
    logger.debug(
        "building %s branch: %d bins, %d samples",
        branch,
        counts.shape[0],
        counts.shape[1],
    )
    normalized = normalize_depth(counts)
    usable = mask_low_coverage(normalized)
    if not np.any(usable):
        raise ReferenceError(f"branch {branch!r} has no unmasked bins")
    n_unmasked = int(np.count_nonzero(usable))
    logger.debug(
        "%s mask kept %d/%d bins (%.1f%% unmasked, %d masked)",
        branch,
        n_unmasked,
        usable.size,
        100.0 * n_unmasked / usable.size,
        usable.size - n_unmasked,
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
    nearest_indices, nearest_distances = select_nearest_bins(
        corrected,
        contig_ids,
        k=config.nearest_bins,
        max_batch_memory_mb=config.max_batch_memory_mb,
    )
    null_ratios = _null_ratios(corrected, nearest_indices)
    valid_nulls = [r for sub in null_ratios for r in sub if np.isfinite(r)]
    logger.debug(
        "%s null ratios computed for %d bins (mean=%.4f, std=%.4f)",
        branch,
        len(null_ratios),
        float(np.mean(valid_nulls)) if valid_nulls else 0.0,
        float(np.std(valid_nulls)) if valid_nulls else 0.0,
    )

    unmasked_iter = 0
    starts = cohort.starts[selected]
    ends = cohort.ends[selected]
    for local, contig in enumerate(contig_subset.tolist()):
        rows["branch"].append(branch)
        rows["contig"].append(contig)
        rows["start"].append(int(starts[local]))
        rows["end"].append(int(ends[local]))
        if not usable[local]:
            rows["masked"].append(True)
            rows["pca_mean"].append(0.0)
            rows["pca_loadings"].append([])
            rows["nearest_ref_indices"].append([])
            rows["nearest_ref_distances"].append([])
            rows["null_ratios"].append([])
            continue
        rows["masked"].append(False)
        rows["pca_mean"].append(float(pca_mean[unmasked_iter]))
        rows["pca_loadings"].append(pca_loadings[:, unmasked_iter].tolist())
        rows["nearest_ref_indices"].append(nearest_indices[unmasked_iter])
        rows["nearest_ref_distances"].append(nearest_distances[unmasked_iter])
        rows["null_ratios"].append(null_ratios[unmasked_iter])
        unmasked_iter += 1


def _null_ratios(
    corrected: NDArray[np.float64],
    nearest_indices: list[list[int]],
) -> list[list[float]]:
    n_null = min(_MAX_NULL_SAMPLES, corrected.shape[1])
    ratios: list[list[float]] = []
    for bin_index, neighbors in enumerate(nearest_indices):
        if not neighbors:
            ratios.append([0.0] * n_null)
            continue
        reference = np.median(corrected[neighbors, :n_null], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.log2(corrected[bin_index, :n_null] / reference)
        ratios.append(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).tolist())
    return ratios


def _median_abs_deviation(values: NDArray[np.floating]) -> float:
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))
