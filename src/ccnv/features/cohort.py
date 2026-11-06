"""Cohort sample aggregation, rebinning, depth normalization, and sex inference."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from ccnv.features.binning import bin_intervals, rebin_counts
from ccnv.models import (
    ArtifactKind,
    ReferenceError,
    SchemaError,
    decode_metadata,
    metadata_for,
    select_layout,
    validate_table,
)

logger = logging.getLogger(__name__)

SexLabel = Literal["female", "male"]
_MASK_FRACTION: Final[float] = 0.05
_SEPARATION_MAD_MULTIPLIER: Final[float] = 4.0

_ParsedSample = tuple[
    dict[str, object],
    dict[str, int],
    tuple[str, ...],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
]


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


def _median_abs_deviation(values: NDArray[np.floating]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


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
    cutoff_val = float((low[-1] + high[0]) / 2.0)
    if best_gap < max(min_separation, _SEPARATION_MAD_MULTIPLIER * spread):
        return SexInference(
            labels=None,
            cutoff=cutoff_val,
            fractions=recorded,
            passed=False,
            reason="sex-signal groups are not sufficiently separated",
        )

    labels = ["female"] * values.size
    for index in order[best_split:]:
        labels[int(index)] = "male"
    return SexInference(
        labels=tuple(labels),
        cutoff=cutoff_val,
        fractions=recorded,
        passed=True,
        reason="separated",
    )


def rebin_sample_table(table: pa.Table, target_bin_size: int) -> pa.Table:
    """Rescale sample Parquet counts to target_bin_size without changing length."""
    if target_bin_size <= 0:
        raise ValueError("target_bin_size must be a positive integer")
    metadata = decode_metadata(table.schema.metadata)
    current_bin_size = metadata.get("bin_size")
    if current_bin_size == target_bin_size:
        return table

    contig_lengths = metadata.get("contig_lengths")
    if not isinstance(contig_lengths, dict) or not contig_lengths:
        raise ValueError("table metadata lacks contig_lengths")

    contig_arr = np.asarray(table.column("contig"))
    start_col = np.asarray(
        table.column("start").to_numpy(zero_copy_only=False), dtype=np.int64
    )
    end_col = np.asarray(
        table.column("end").to_numpy(zero_copy_only=False), dtype=np.int64
    )
    count_col = np.asarray(
        table.column("count").to_numpy(zero_copy_only=False), dtype=np.float64
    )

    new_contigs: list[np.ndarray] = []
    new_starts: list[np.ndarray] = []
    new_ends: list[np.ndarray] = []
    new_counts: list[np.ndarray] = []

    if len(contig_arr) == 0:
        contig_slices: dict[str, tuple[int, int]] = {}
    else:
        changes = np.flatnonzero(contig_arr[1:] != contig_arr[:-1]) + 1
        starts = np.concatenate(([0], changes))
        ends = np.concatenate((changes, [len(contig_arr)]))
        contig_slices = {
            str(contig_arr[s]): (int(s), int(e))
            for s, e in zip(starts, ends, strict=True)
        }

    for contig_name, length in contig_lengths.items():
        target_starts, target_ends = bin_intervals(int(length), target_bin_size)
        span = contig_slices.get(contig_name)
        if span is None:
            target_counts = np.zeros(target_starts.size, dtype=np.float64)
        else:
            s, e = span
            s_starts = start_col[s:e]
            s_ends = end_col[s:e]
            s_counts = count_col[s:e]
            target_counts = rebin_counts(
                s_starts, s_ends, s_counts, target_starts, target_ends
            )
        new_contigs.append(np.full(target_starts.size, contig_name, dtype=object))
        new_starts.append(target_starts)
        new_ends.append(target_ends)
        new_counts.append(target_counts)

    meta_values = {
        key: val
        for key, val in metadata.items()
        if key
        not in (
            "artifact_kind",
            "schema_version",
            "coordinate_system",
            "contig_lengths",
        )
    }
    meta_values["bin_size"] = target_bin_size
    new_metadata = metadata_for(
        ArtifactKind.SAMPLE,
        contig_lengths={str(k): int(v) for k, v in contig_lengths.items()},
        values=meta_values,
    )
    new_schema = pa.schema(
        [
            pa.field("contig", pa.string(), nullable=False),
            pa.field("start", pa.int64(), nullable=False),
            pa.field("end", pa.int64(), nullable=False),
            pa.field("count", pa.float64(), nullable=False),
        ]
    ).with_metadata(new_metadata)

    empty_str = np.array([], dtype=object)
    empty_i64 = np.empty(0, dtype=np.int64)
    empty_f64 = np.empty(0, dtype=np.float64)
    return pa.table(
        {
            "contig": pa.array(
                np.concatenate(new_contigs) if new_contigs else empty_str,
                type=pa.string(),
            ),
            "start": pa.array(
                np.concatenate(new_starts) if new_starts else empty_i64,
                type=pa.int64(),
            ),
            "end": pa.array(
                np.concatenate(new_ends) if new_ends else empty_i64,
                type=pa.int64(),
            ),
            "count": pa.array(
                np.concatenate(new_counts) if new_counts else empty_f64,
                type=pa.float64(),
            ),
        },
        schema=new_schema,
    )


def _read_sample(path: Path) -> pa.Table:
    try:
        return pq.read_table(path)
    except (FileNotFoundError, OSError, pa.ArrowInvalid) as error:
        raise ReferenceError(f"{path}: cannot read sample Parquet ({error})") from error


def _parse_sample_table(
    table: pa.Table,
    label: str,
    target_bin_size: int | None = None,
) -> _ParsedSample:
    try:
        validate_table(table, ArtifactKind.SAMPLE)
    except SchemaError as error:
        raise ReferenceError(f"{label}: {error}") from error

    metadata = decode_metadata(table.schema.metadata)
    contig_lengths = metadata.get("contig_lengths")
    orig_bin_size = metadata.get("bin_size")
    if not isinstance(contig_lengths, dict) or not isinstance(orig_bin_size, int):
        raise ReferenceError(f"{label}: sample metadata is incomplete")

    effective_bin_size = (
        target_bin_size if target_bin_size is not None else orig_bin_size
    )
    if orig_bin_size != effective_bin_size:
        logger.debug(
            "rebinning cohort sample %s from %s bp to %s bp",
            Path(label).name,
            orig_bin_size,
            effective_bin_size,
        )
        table = rebin_sample_table(table, effective_bin_size)
        metadata = decode_metadata(table.schema.metadata)

    contigs = tuple(str(c) for c in table.column("contig").to_pylist())
    starts = np.asarray(
        table.column("start").to_numpy(zero_copy_only=False), dtype=np.int64
    )
    ends = np.asarray(
        table.column("end").to_numpy(zero_copy_only=False), dtype=np.int64
    )
    counts = np.asarray(
        table.column("count").to_numpy(zero_copy_only=False), dtype=np.float64
    )
    lengths = {str(k): int(v) for k, v in contig_lengths.items()}
    return metadata, lengths, contigs, starts, ends, counts


def _assemble_cohort(
    parsed: Sequence[_ParsedSample],
    paths: Sequence[Path],
    target_bin_size: int | None = None,
) -> Cohort:
    (
        first_meta,
        expected_lengths,
        expected_contigs,
        expected_starts,
        expected_ends,
        _,
    ) = parsed[0]
    expected_bin_size = int(first_meta["bin_size"])
    counts_list: list[NDArray[np.float64]] = []
    identities: list[object] = []

    for path, (metadata, lengths, contigs, starts, ends, counts) in zip(
        paths, parsed, strict=True
    ):
        bin_size = int(metadata["bin_size"])
        if lengths != expected_lengths:
            raise ReferenceError(f"{path}: contig_lengths do not match the cohort")
        if target_bin_size is None and bin_size != expected_bin_size:
            raise ReferenceError(
                f"{path}: bin_size {bin_size} does not match cohort {expected_bin_size}"
            )
        if (
            contigs != expected_contigs
            or not np.array_equal(starts, expected_starts)
            or not np.array_equal(ends, expected_ends)
        ):
            raise ReferenceError(f"{path}: bin coordinates do not match the cohort")

        logger.debug(
            "loaded cohort sample %s: %d bins, %d total depth",
            path.name,
            counts.size,
            int(np.sum(counts)),
        )
        counts_list.append(counts)
        identities.append(metadata.get("source_identity", {"path": str(path)}))

    return Cohort(
        counts=np.column_stack(counts_list),
        contigs=expected_contigs,
        starts=expected_starts,
        ends=expected_ends,
        contig_lengths=expected_lengths,
        bin_size=expected_bin_size,
        identities=tuple(identities),
        paths=tuple(paths),
    )


def _load_sample_worker(args: tuple[Path, int | None]) -> _ParsedSample:
    path, target_bin_size = args
    return _parse_sample_table(_read_sample(path), str(path), target_bin_size)


def load_cohort(
    paths: tuple[Path, ...],
    target_bin_size: int | None = None,
    threads: int = 1,
) -> Cohort:
    """Load sample Parquet files and reject incompatible cohorts."""
    if not paths:
        raise ReferenceError("at least one sample Parquet path is required")

    args = [(path, target_bin_size) for path in paths]
    if threads > 1 and len(paths) > 1:
        max_workers = min(threads, len(paths))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_load_sample_worker, args))
    else:
        results = [_load_sample_worker(arg) for arg in args]

    return _assemble_cohort(results, paths, target_bin_size)


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

    parsed = [
        _parse_sample_table(table, str(path), target_bin_size)
        for path, table in zip(resolved_paths, tables, strict=True)
    ]
    return _assemble_cohort(parsed, resolved_paths, target_bin_size)


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
    import pysam

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
