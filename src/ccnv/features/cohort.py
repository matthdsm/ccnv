"""Cohort sample aggregation, rebinning, depth normalization, and sex inference."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import polars as pl
import pyarrow as pa
from numpy.typing import NDArray

from ccnv.features.binning import bin_intervals, rebin_counts
from ccnv.models import (
    FRAGMENTOMICS_METADATA_KEY,
    ArtifactKind,
    ReferenceError,
    SchemaError,
    decode_metadata,
    metadata_for,
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
    sample_metadata: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class SexInference:
    """Deterministic sex assignments, or a rejected inference."""

    labels: tuple[SexLabel, ...] | None
    cutoff: float | None
    fractions: tuple[float, ...]
    passed: bool
    reason: str


class InvalidInputError(ReferenceError, ValueError):
    """Raised when count matrix input validation fails."""


def estimate_size_factors(
    counts: NDArray[np.floating],
    mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Estimate DESeq2 median-of-ratios size factors for each sample."""
    matrix = np.asarray(counts, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise InvalidInputError("Counts matrix must be a non-empty 2D array.")
    if np.any(matrix < 0):
        raise InvalidInputError("Counts matrix cannot contain negative values.")

    if mask is not None:
        mask_arr = np.asarray(mask, dtype=np.bool_)
        if mask_arr.ndim != 1 or mask_arr.shape[0] != matrix.shape[0]:
            raise InvalidInputError("mask length must match counts row count")
        sub = matrix[mask_arr]
        if sub.shape[0] == 0:
            raise InvalidInputError("No usable bins remaining after mask.")
    else:
        sub = matrix

    totals = sub.sum(axis=0)
    if np.any(totals <= 0):
        suffix = " on usable bins" if mask is not None else ""
        raise InvalidInputError(
            f"Total read count is zero for at least one sample{suffix}."
        )

    pos_rows = np.all(sub > 0, axis=1)
    if np.count_nonzero(pos_rows) >= 1:
        log_geomean = np.mean(np.log(sub[pos_rows]), axis=1, keepdims=True)
        raw_sf = np.exp(np.median(np.log(sub[pos_rows]) - log_geomean, axis=0))
    else:
        pos_medians = [
            float(np.median(col[col > 0])) if np.any(col > 0) else 0.0 for col in sub.T
        ]
        if all(m > 0 for m in pos_medians):
            raw_sf = np.asarray(pos_medians, dtype=np.float64)
        else:
            raw_sf = totals.astype(np.float64)

    sf = raw_sf / np.exp(np.mean(np.log(raw_sf)))
    return sf


def normalize_depth(
    counts: NDArray[np.floating],
    mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Normalize sample count matrix using DESeq2 median-of-ratios depth scaling."""
    matrix = np.asarray(counts, dtype=np.float64)
    sf = estimate_size_factors(matrix, mask=mask)
    sub = matrix[mask] if mask is not None else matrix
    col_sums = sub.sum(axis=0)
    median_scale = float(np.median(col_sums / sf))
    s = sf * median_scale
    return matrix / s[np.newaxis, :]


def mask_low_coverage(normalized: NDArray[np.floating]) -> NDArray[np.bool_]:
    """Mask bins with no coverage or coverage below 5% of the positive median."""
    per_bin = np.asarray(normalized, dtype=np.float64).sum(axis=1)
    positive = per_bin[per_bin > 0]
    if positive.size == 0:
        return np.zeros(per_bin.shape, dtype=np.bool_)
    threshold = _MASK_FRACTION * float(np.median(positive))
    return per_bin > threshold


def _median_abs_deviation(values: NDArray[np.floating]) -> float:
    """Compute the unscaled Median Absolute Deviation (MAD) of an array."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


def _validate_sex_cutoff(cutoff: float | None) -> None:
    """Validate that sex separation cutoff is in range [0, 1]."""
    if cutoff is None:
        return
    if not math.isfinite(cutoff) or cutoff < 0.0 or cutoff > 1.0:
        raise ReferenceError("sex_cutoff must be a finite number in [0, 1]")


def _labels_from_cutoff(
    values: NDArray[np.float64],
    recorded: tuple[float, ...],
    cutoff: float,
) -> SexInference:
    """Assign sex labels by thresholding contig fractions against an explicit cutoff."""
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


def _build_target_grid(
    contig_lengths: Mapping[str, int], target_bin_size: int
) -> pl.DataFrame:
    """Construct a canonical DataFrame grid of genomic bin intervals for given contigs."""
    grid_contigs: list[str] = []
    grid_starts: list[np.ndarray] = []
    grid_ends: list[np.ndarray] = []
    for c_name, c_len in contig_lengths.items():
        s, e = bin_intervals(int(c_len), target_bin_size)
        grid_contigs.extend([str(c_name)] * s.size)
        grid_starts.append(s)
        grid_ends.append(e)

    return pl.DataFrame(
        {
            "contig": grid_contigs,
            "start": (
                np.concatenate(grid_starts)
                if grid_starts
                else np.empty(0, dtype=np.int64)
            ),
            "end": (
                np.concatenate(grid_ends) if grid_ends else np.empty(0, dtype=np.int64)
            ),
        },
        schema={"contig": pl.String, "start": pl.Int64, "end": pl.Int64},
    )


def _redistribute_by_overlap(
    src_starts: NDArray[np.int64],
    src_ends: NDArray[np.int64],
    src_counts: NDArray[np.floating],
    tgt_starts: NDArray[np.int64],
    tgt_ends: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Redistribute count mass from arbitrary source intervals to target bins by overlap."""
    n_targets = tgt_starts.size
    if n_targets == 0:
        return (
            np.empty((0, src_counts.shape[1]), dtype=np.float64)
            if src_counts.ndim == 2
            else np.empty(0, dtype=np.float64)
        )
    if src_starts.size == 0:
        return (
            np.zeros((n_targets, src_counts.shape[1]), dtype=np.float64)
            if src_counts.ndim == 2
            else np.zeros(n_targets, dtype=np.float64)
        )

    is_2d = src_counts.ndim == 2
    out = (
        np.zeros((n_targets, src_counts.shape[1]), dtype=np.float64)
        if is_2d
        else np.zeros(n_targets, dtype=np.float64)
    )

    t_bin_size = max(int(tgt_ends[0] - tgt_starts[0]), 1)
    lens = src_ends - src_starts
    valid = lens > 0
    if not np.any(valid):
        return out

    s_s = src_starts[valid]
    s_e = src_ends[valid]
    s_c = src_counts[valid]
    s_len = lens[valid].astype(np.float64)

    first_k = np.clip(s_s // t_bin_size, 0, n_targets - 1)
    last_k = np.clip((s_e - 1) // t_bin_size, 0, n_targets - 1)

    for i in range(len(s_s)):
        for k in range(int(first_k[i]), int(last_k[i]) + 1):
            ov = max(
                0,
                min(int(s_e[i]), int(tgt_ends[k]))
                - max(int(s_s[i]), int(tgt_starts[k])),
            )
            if ov > 0:
                frac = ov / s_len[i]
                if is_2d:
                    out[k] += frac * s_c[i]
                else:
                    out[k] += frac * s_c[i]
    return out


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
    has_frag = "fragment_lengths" in table.column_names
    has_ep = "endpoint_count" in table.column_names
    has_gc = "frag_gc_sum" in table.column_names

    schema_fields = [
        pa.field("contig", pa.string(), nullable=False),
        pa.field("start", pa.int64(), nullable=False),
        pa.field("end", pa.int64(), nullable=False),
        pa.field("count", pa.float64(), nullable=False),
    ]
    if has_frag:
        schema_fields.append(
            pa.field("fragment_lengths", pa.list_(pa.uint16()), nullable=False)
        )
    if has_ep:
        schema_fields.append(pa.field("endpoint_count", pa.uint32(), nullable=False))
    if has_gc:
        schema_fields.append(pa.field("frag_gc_sum", pa.float32(), nullable=False))

    new_schema = pa.schema(schema_fields).with_metadata(new_metadata)

    intervals_contained = False
    if (
        isinstance(current_bin_size, int)
        and current_bin_size > 0
        and target_bin_size % current_bin_size == 0
        and target_bin_size > current_bin_size
    ):
        df = pl.from_arrow(table)
        assert isinstance(df, pl.DataFrame)
        intervals_contained = (
            df.height == 0
            or df.select(
                (
                    (pl.col("start") // target_bin_size)
                    == ((pl.col("end") - 1) // target_bin_size)
                ).all()
            ).item()
        )

    if intervals_contained:
        target_grid = _build_target_grid(contig_lengths, target_bin_size)
        df = pl.from_arrow(table)
        assert isinstance(df, pl.DataFrame)
        if df.height > 0:
            agg_exprs = [pl.col("count").sum().alias("count")]
            select_cols = [
                pl.col("contig").cast(pl.String),
                ((pl.col("start") // target_bin_size) * target_bin_size)
                .cast(pl.Int64)
                .alias("start"),
                pl.col("count").cast(pl.Float64),
            ]
            if has_frag:
                agg_exprs.append(
                    pl.col("fragment_lengths").list.explode().alias("fragment_lengths")
                )
                select_cols.append(pl.col("fragment_lengths"))
            if has_ep:
                agg_exprs.append(
                    pl.col("endpoint_count")
                    .sum()
                    .cast(pl.UInt32)
                    .alias("endpoint_count")
                )
                select_cols.append(pl.col("endpoint_count"))
            if has_gc:
                agg_exprs.append(
                    pl.col("frag_gc_sum").sum().cast(pl.Float32).alias("frag_gc_sum")
                )
                select_cols.append(pl.col("frag_gc_sum"))

            agg_df = (
                df.select(select_cols)
                .group_by(["contig", "start"], maintain_order=True)
                .agg(agg_exprs)
            )

            with_cols = [pl.col("count").fill_null(0.0).cast(pl.Float64)]
            if has_frag:
                with_cols.append(
                    pl.col("fragment_lengths").fill_null(
                        pl.lit([], dtype=pl.List(pl.UInt16))
                    )
                )
            if has_ep:
                with_cols.append(pl.col("endpoint_count").fill_null(0).cast(pl.UInt32))
            if has_gc:
                with_cols.append(pl.col("frag_gc_sum").fill_null(0.0).cast(pl.Float32))

            result_df = target_grid.join(
                agg_df, on=["contig", "start"], how="left"
            ).with_columns(with_cols)
        else:
            with_cols = [pl.lit(0.0, dtype=pl.Float64).alias("count")]
            if has_frag:
                with_cols.append(
                    pl.lit([], dtype=pl.List(pl.UInt16)).alias("fragment_lengths")
                )
            if has_ep:
                with_cols.append(pl.lit(0, dtype=pl.UInt32).alias("endpoint_count"))
            if has_gc:
                with_cols.append(pl.lit(0.0, dtype=pl.Float32).alias("frag_gc_sum"))
            result_df = target_grid.with_columns(with_cols)

        arrow_table = result_df.to_arrow()
        table_cols = {
            "contig": pa.array(
                arrow_table.column("contig").to_pylist(), type=pa.string()
            ),
            "start": pa.array(arrow_table.column("start").to_numpy(), type=pa.int64()),
            "end": pa.array(arrow_table.column("end").to_numpy(), type=pa.int64()),
            "count": pa.array(
                arrow_table.column("count").to_numpy(), type=pa.float64()
            ),
        }
        if has_frag:
            table_cols["fragment_lengths"] = pa.array(
                arrow_table.column("fragment_lengths").to_pylist(),
                type=pa.list_(pa.uint16()),
            )
        if has_ep:
            table_cols["endpoint_count"] = pa.array(
                arrow_table.column("endpoint_count").to_numpy(),
                type=pa.uint32(),
            )
        if has_gc:
            table_cols["frag_gc_sum"] = pa.array(
                arrow_table.column("frag_gc_sum").to_numpy(),
                type=pa.float32(),
            )

        return pa.table(table_cols, schema=new_schema)

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
            is_part = (
                s_starts.size > 0
                and s_starts[0] == 0
                and s_ends[-1] == target_ends[-1]
                and np.array_equal(s_starts[1:], s_ends[:-1])
            )
            if is_part:
                target_counts = rebin_counts(
                    s_starts, s_ends, s_counts, target_starts, target_ends
                )
            else:
                target_counts = _redistribute_by_overlap(
                    s_starts, s_ends, s_counts, target_starts, target_ends
                )
        new_contigs.append(np.full(target_starts.size, contig_name, dtype=object))
        new_starts.append(target_starts)
        new_ends.append(target_ends)
        new_counts.append(target_counts)

    empty_str = np.array([], dtype=object)
    empty_i64 = np.empty(0, dtype=np.int64)
    empty_f64 = np.empty(0, dtype=np.float64)

    out_cols = {
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
    }
    n_rows = len(out_cols["contig"])
    if has_frag:
        out_cols["fragment_lengths"] = pa.array(
            [[]] * n_rows, type=pa.list_(pa.uint16())
        )
    if has_ep:
        out_cols["endpoint_count"] = pa.array(
            np.zeros(n_rows, dtype=np.uint32), type=pa.uint32()
        )
    if has_gc:
        out_cols["frag_gc_sum"] = pa.array(
            np.zeros(n_rows, dtype=np.float32), type=pa.float32()
        )

    return pa.table(out_cols, schema=new_schema)


def _parse_sample_table(
    table: pa.Table,
    label: str,
    target_bin_size: int | None = None,
) -> _ParsedSample:
    """Validate and unpack a sample table, rebinning if needed."""
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
    """Combine parsed sample tables into an aligned Cohort structure."""
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

    def _is_paired(meta: dict[str, object]) -> bool:
        for k in ("fragmentomics", "ccnv.fragmentomics", FRAGMENTOMICS_METADATA_KEY):
            frag = meta.get(k)
            if isinstance(frag, dict):
                return bool(frag.get("paired_end", False))
            if isinstance(frag, (bytes, str)):
                try:
                    loaded = json.loads(frag)
                    if isinstance(loaded, dict):
                        return bool(loaded.get("paired_end", False))
                except json.JSONDecodeError, UnicodeDecodeError, TypeError:
                    continue
        return False

    first_paired = _is_paired(first_meta)

    for path, (metadata, lengths, contigs, starts, ends, counts) in zip(
        paths, parsed, strict=True
    ):
        if _is_paired(metadata) != first_paired:
            raise ReferenceError(
                "mixed single-end and paired-end cohort controls are not supported"
            )
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
        sample_metadata=tuple(p[0] for p in parsed),
    )


def rebin_cohort(cohort: Cohort, target_bin_size: int) -> Cohort:
    """Rescale 2D cohort count matrix to target_bin_size."""
    if target_bin_size <= 0:
        raise ValueError("target_bin_size must be a positive integer")
    if cohort.bin_size == target_bin_size:
        return cohort

    current_bin_size = cohort.bin_size
    n_samples = cohort.counts.shape[1]

    intervals_contained = cohort.starts.size == 0 or bool(
        np.all(
            (cohort.starts // target_bin_size) == ((cohort.ends - 1) // target_bin_size)
        )
    )

    if (
        current_bin_size > 0
        and target_bin_size % current_bin_size == 0
        and target_bin_size > current_bin_size
        and intervals_contained
    ):
        target_grid = _build_target_grid(cohort.contig_lengths, target_bin_size)
        if cohort.counts.shape[0] > 0 and n_samples > 0:
            sample_col_names = [f"c_{j}" for j in range(n_samples)]
            df_dict: dict[str, object] = {
                "contig": [str(c) for c in cohort.contigs],
                "start": cohort.starts,
            }
            for j, name in enumerate(sample_col_names):
                df_dict[name] = cohort.counts[:, j]
            df = pl.DataFrame(df_dict)

            agg_df = (
                df.select(
                    [
                        pl.col("contig"),
                        ((pl.col("start") // target_bin_size) * target_bin_size)
                        .cast(pl.Int64)
                        .alias("start"),
                        pl.col("^c_.*$"),
                    ]
                )
                .group_by(["contig", "start"], maintain_order=True)
                .agg(pl.col("^c_.*$").sum())
            )
            result_df = target_grid.join(
                agg_df, on=["contig", "start"], how="left"
            ).with_columns(pl.col("^c_.*$").fill_null(0.0).cast(pl.Float64))
            new_counts = (
                result_df.select(sample_col_names).to_numpy()
                if target_grid.height > 0
                else np.empty((0, n_samples), dtype=np.float64)
            )
        else:
            result_df = target_grid
            new_counts = (
                np.zeros((target_grid.height, n_samples), dtype=np.float64)
                if n_samples > 0
                else np.empty((target_grid.height, 0), dtype=np.float64)
            )

        return Cohort(
            contigs=tuple(result_df["contig"].to_list()),
            starts=result_df["start"].to_numpy(),
            ends=result_df["end"].to_numpy(),
            counts=new_counts,
            contig_lengths=cohort.contig_lengths,
            identities=cohort.identities,
            bin_size=target_bin_size,
            paths=cohort.paths,
            sample_metadata=cohort.sample_metadata,
        )

    contig_arr = np.asarray(cohort.contigs)
    n_samples = cohort.counts.shape[1]
    if len(contig_arr) == 0:
        contig_slices: dict[str, tuple[int, int]] = {}
    else:
        changes = np.flatnonzero(contig_arr[1:] != contig_arr[:-1]) + 1
        starts_idx = np.concatenate(([0], changes))
        ends_idx = np.concatenate((changes, [len(contig_arr)]))
        contig_slices = {
            str(contig_arr[s]): (int(s), int(e))
            for s, e in zip(starts_idx, ends_idx, strict=True)
        }

    new_c: list[str] = []
    new_s: list[np.ndarray] = []
    new_e: list[np.ndarray] = []
    new_counts: list[np.ndarray] = []

    for c_name, c_len in cohort.contig_lengths.items():
        t_starts, t_ends = bin_intervals(int(c_len), target_bin_size)
        span = contig_slices.get(c_name)
        if span is None:
            t_counts = np.zeros((t_starts.size, n_samples), dtype=np.float64)
        else:
            s_i, e_i = span
            src_s = cohort.starts[s_i:e_i]
            src_e = cohort.ends[s_i:e_i]
            is_part = (
                src_s.size > 0
                and src_s[0] == 0
                and src_e[-1] == t_ends[-1]
                and np.array_equal(src_s[1:], src_e[:-1])
            )
            if is_part:
                t_counts = np.column_stack(
                    [
                        rebin_counts(
                            src_s,
                            src_e,
                            cohort.counts[s_i:e_i, j],
                            t_starts,
                            t_ends,
                        )
                        for j in range(n_samples)
                    ]
                )
            else:
                t_counts = _redistribute_by_overlap(
                    src_s,
                    src_e,
                    cohort.counts[s_i:e_i, :],
                    t_starts,
                    t_ends,
                )
        new_c.extend([c_name] * t_starts.size)
        new_s.append(t_starts)
        new_e.append(t_ends)
        new_counts.append(t_counts)

    return Cohort(
        contigs=tuple(new_c),
        starts=np.concatenate(new_s) if new_s else np.empty(0, dtype=np.int64),
        ends=np.concatenate(new_e) if new_e else np.empty(0, dtype=np.int64),
        counts=(
            np.vstack(new_counts)
            if new_counts
            else np.empty((0, n_samples), dtype=np.float64)
        ),
        contig_lengths=cohort.contig_lengths,
        identities=cohort.identities,
        bin_size=target_bin_size,
        paths=cohort.paths,
        sample_metadata=cohort.sample_metadata,
    )


def validate_cohort(
    tables: Sequence[pa.Table],
    paths: Sequence[Path] | None = None,
    target_bin_size: int | None = None,
) -> Cohort:
    """Require matching contig lengths across samples, rebinning counts if set."""
    if len(tables) < 1:
        raise ReferenceError("cohort must contain at least one sample")
    resolved_paths = (
        tuple(paths)
        if paths is not None
        else tuple(Path(f"sample_{index}") for index in range(len(tables)))
    )
    if len(resolved_paths) != len(tables):
        raise ReferenceError("paths must align with sample tables")

    parsed = [
        _parse_sample_table(table, str(path), target_bin_size)
        for path, table in zip(resolved_paths, tables, strict=True)
    ]
    return _assemble_cohort(parsed, resolved_paths, target_bin_size)


def cohort_from_fasta(
    contigs_dict: Mapping[str, int],
    bin_size: int,
    *,
    n_samples: int,
    fasta_path: Path | None = None,
) -> Cohort:
    """Build identical uniform-depth samples from FASTA contig lengths."""
    if n_samples < 1:
        raise ReferenceError("n_samples must be a positive integer")
    names = list(contigs_dict.keys())
    lengths = [int(v) for v in contigs_dict.values()]
    source_path = fasta_path or Path("flat_fasta")

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
    resolved = source_path.expanduser().resolve()
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
