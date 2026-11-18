"""Shared coordinate sorting, contig chunking, DataFrame builders, and atomic BGZF writing."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby
from operator import attrgetter
from pathlib import Path
from typing import Final, Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from ccnv._core import write_bgzf_indexed
from ccnv.models import BinResult, SegmentResult, natural_sort_key

type BinResults = Sequence[BinResult]

STREAM_CHUNK_ROWS: Final[int] = 65_536

_FINITE_BASE_FLOAT_COLS: Final[tuple[str, ...]] = ("ratio", "zscore", "weight")
_OPTIONAL_PRENATAL_FLOAT_COLS: Final[tuple[str, ...]] = (
    "expected_ratio",
    "implied_ff",
    "frag_short_ratio",
    "frag_llr_size",
    "mosaic_fraction",
)
_GERMLINE_SEGMENT_SCHEMA: Final[dict[str, pl.DataType]] = {
    "contig": pl.String(),
    "start": pl.Int64(),
    "end": pl.Int64(),
    "ratio": pl.Float64(),
    "zscore": pl.Float64(),
    "weight": pl.Float64(),
    "bin_count": pl.Int64(),
    "call": pl.String(),
}
_PRENATAL_SEGMENT_SCHEMA: Final[dict[str, pl.DataType]] = {
    **_GERMLINE_SEGMENT_SCHEMA,
    "expected_ratio": pl.Float64(),
    "implied_ff": pl.Float64(),
    "discordant": pl.Boolean(),
    "discordance_reason": pl.String(),
    "frag_short_ratio": pl.Float64(),
    "frag_llr_size": pl.Float64(),
    "origin_call": pl.String(),
    "mosaic_fraction": pl.Float64(),
}
_GERMLINE_SEGMENT_GETTER: Final = attrgetter(*_GERMLINE_SEGMENT_SCHEMA.keys())
_PRENATAL_SEGMENT_GETTER: Final = attrgetter(*_PRENATAL_SEGMENT_SCHEMA.keys())


@dataclass(frozen=True, slots=True)
class ContigBinChunk:
    """Coordinate-sorted slice of bin evidence on a single contig."""

    contig: str
    starts: NDArray[np.int64]
    ends: NDArray[np.int64]
    ratios: NDArray[np.float64]
    zscores: NDArray[np.float64]
    weights: NDArray[np.float64]
    reference_sizes: NDArray[np.int64]
    masked: NDArray[np.bool_]
    branch: str | tuple[str, ...]


__all__ = [
    "STREAM_CHUNK_ROWS",
    "BinResults",
    "ContigBinChunk",
    "bin_chunk_to_dataframe",
    "iter_contig_bin_chunks",
    "max_bin_end",
    "max_df_end",
    "ordered_contig_names",
    "partition_contig_runs",
    "resolve_natural_contig_bounds",
    "resolve_segments_dataframe",
    "segments_to_dataframe",
    "sort_bin_result",
    "sort_bin_results",
    "sort_segments",
    "sort_segments_dataframe",
    "validate_threads",
    "write_bgzf_atomic",
]


def validate_threads(threads: int) -> None:
    """Validate that threads is a positive integer."""
    if isinstance(threads, bool) or threads < 1:
        raise ValueError("threads must be a positive integer")


def write_bgzf_atomic(
    gz_path: Path,
    preset: Literal["bed", "vcf"],
    contig_names: Sequence[str],
    max_contig_len: int,
    chunks: Iterable[bytes],
    threads: int,
) -> Path:
    """Atomically write BGZF archive and its Tabix/CSI index via sibling temp files."""
    gz_path.parent.mkdir(parents=True, exist_ok=True)
    token = f".{gz_path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    tmp_gz = gz_path.parent / token
    tmp_idx: Path | None = None
    try:
        tmp_idx_str = write_bgzf_indexed(
            str(tmp_gz),
            preset,
            list(contig_names),
            int(max_contig_len),
            chunks,
            threads,
        )
        tmp_idx = Path(tmp_idx_str)
        idx_ext = ".csi" if tmp_idx_str.endswith(".csi") else ".tbi"
        stale_ext = ".tbi" if idx_ext == ".csi" else ".csi"
        final_idx = Path(f"{gz_path}{idx_ext}")
        os.replace(tmp_gz, gz_path)
        os.replace(tmp_idx, final_idx)
        Path(f"{gz_path}{stale_ext}").unlink(missing_ok=True)
    except BaseException:
        tmp_gz.unlink(missing_ok=True)
        if tmp_idx is not None:
            tmp_idx.unlink(missing_ok=True)
        else:
            Path(f"{tmp_gz}.tbi").unlink(missing_ok=True)
            Path(f"{tmp_gz}.csi").unlink(missing_ok=True)
        raise
    return gz_path


def partition_contig_runs(contigs: Sequence[str]) -> list[tuple[str, int, int]]:
    """Identify contiguous ``(contig, start_idx, end_idx)`` runs in a contig sequence."""
    if not contigs:
        return []
    runs: list[tuple[str, int, int]] = []
    pos = 0
    for contig_name, group in groupby(contigs):
        run_len = len(list(group))
        end_pos = pos + run_len
        runs.append((contig_name, pos, end_pos))
        pos = end_pos
    return runs


def ordered_contig_names(
    contigs: Iterable[str],
    preferred_order: Sequence[str] = (),
) -> list[str]:
    """Order unique contig names preserving ``preferred_order`` then natural sort."""
    ordered = list(dict.fromkeys(preferred_order))
    seen = set(ordered)
    ordered.extend(sorted({c for c in contigs if c not in seen}, key=natural_sort_key))
    return ordered


def resolve_natural_contig_bounds(
    record_contigs: Iterable[str],
    record_max_end: int,
    contig_lengths: Mapping[str, int] | None,
) -> tuple[list[str], int]:
    """Resolve naturally ordered contig vocabulary and maximum end coordinate."""
    all_contigs = set(record_contigs)
    max_end = record_max_end
    if contig_lengths:
        all_contigs.update(contig_lengths.keys())
        max_end = max(max_end, max(contig_lengths.values(), default=0))
    return ordered_contig_names(all_contigs), max_end


def iter_contig_bin_chunks(
    bin_results: BinResults,
    contig_order: Sequence[str] = (),
    *,
    chunk_rows: int = STREAM_CHUNK_ROWS,
) -> Iterator[ContigBinChunk]:
    """Yield coordinate-sorted per-contig bin slices in exact ``contig_order``."""
    slices_by_contig: dict[str, list[tuple[BinResult, int, int]]] = {}
    for res in bin_results:
        for contig_name, lo, hi in partition_contig_runs(res.contigs):
            slices_by_contig.setdefault(contig_name, []).append((res, lo, hi))

    for contig_name in ordered_contig_names(slices_by_contig.keys(), contig_order):
        slices = slices_by_contig.get(contig_name)
        if not slices:
            continue
        if len(slices) == 1:
            res, lo, hi = slices[0]
            s_view = np.asarray(res.starts[lo:hi], dtype=np.int64)
            e_view = np.asarray(res.ends[lo:hi], dtype=np.int64)
            coords_sorted = s_view.size <= 1 or bool(
                np.all(
                    (s_view[1:] > s_view[:-1])
                    | ((s_view[1:] == s_view[:-1]) & (e_view[1:] >= e_view[:-1]))
                )
            )
            if coords_sorted:
                n_contig = hi - lo
                for offset in range(0, n_contig, chunk_rows):
                    stop = min(offset + chunk_rows, n_contig)
                    a = lo + offset
                    b = lo + stop
                    yield ContigBinChunk(
                        contig=contig_name,
                        starts=s_view[offset:stop],
                        ends=e_view[offset:stop],
                        ratios=np.asarray(res.ratios[a:b], dtype=np.float64),
                        zscores=np.asarray(res.zscores[a:b], dtype=np.float64),
                        weights=np.asarray(res.weights[a:b], dtype=np.float64),
                        reference_sizes=np.asarray(
                            res.reference_sizes[a:b], dtype=np.int64
                        ),
                        masked=np.asarray(res.masked[a:b], dtype=np.bool_),
                        branch=res.branch,
                    )
                continue

        s_cat = np.concatenate(
            [np.asarray(r.starts[lo:hi], dtype=np.int64) for r, lo, hi in slices]
        )
        e_cat = np.concatenate(
            [np.asarray(r.ends[lo:hi], dtype=np.int64) for r, lo, hi in slices]
        )
        r_cat = np.concatenate(
            [np.asarray(r.ratios[lo:hi], dtype=np.float64) for r, lo, hi in slices]
        )
        z_cat = np.concatenate(
            [np.asarray(r.zscores[lo:hi], dtype=np.float64) for r, lo, hi in slices]
        )
        w_cat = np.concatenate(
            [np.asarray(r.weights[lo:hi], dtype=np.float64) for r, lo, hi in slices]
        )
        ref_cat = np.concatenate(
            [
                np.asarray(r.reference_sizes[lo:hi], dtype=np.int64)
                for r, lo, hi in slices
            ]
        )
        m_cat = np.concatenate(
            [np.asarray(r.masked[lo:hi], dtype=np.bool_) for r, lo, hi in slices]
        )
        perm = np.lexsort((e_cat, s_cat))
        s_cat = s_cat[perm]
        e_cat = e_cat[perm]
        r_cat = r_cat[perm]
        z_cat = z_cat[perm]
        w_cat = w_cat[perm]
        ref_cat = ref_cat[perm]
        m_cat = m_cat[perm]

        if len({r.branch for r, _, _ in slices}) == 1:
            single_branch: str | None = slices[0][0].branch
            branch_list: list[str] = []
        else:
            single_branch = None
            raw_branches: list[str] = []
            for r, lo, hi in slices:
                raw_branches.extend([r.branch] * (hi - lo))
            branch_list = [raw_branches[int(i)] for i in perm.tolist()]

        for offset in range(0, s_cat.size, chunk_rows):
            stop = min(offset + chunk_rows, s_cat.size)
            chunk_branch: str | tuple[str, ...] = (
                single_branch
                if single_branch is not None
                else tuple(branch_list[offset:stop])
            )
            yield ContigBinChunk(
                contig=contig_name,
                starts=s_cat[offset:stop],
                ends=e_cat[offset:stop],
                ratios=r_cat[offset:stop],
                zscores=z_cat[offset:stop],
                weights=w_cat[offset:stop],
                reference_sizes=ref_cat[offset:stop],
                masked=m_cat[offset:stop],
                branch=chunk_branch,
            )


def bin_chunk_to_dataframe(chunk: ContigBinChunk) -> pl.DataFrame:
    """Build a Polars DataFrame from a ContigBinChunk with non-finite floats mapped to null."""
    cols: dict[str, object] = {
        "start": chunk.starts,
        "end": chunk.ends,
        "ratio": chunk.ratios,
        "zscore": chunk.zscores,
        "weight": chunk.weights,
        "reference_size": chunk.reference_sizes,
        "masked": chunk.masked,
    }
    if not isinstance(chunk.branch, str):
        cols["branch"] = pl.Series("branch", chunk.branch, dtype=pl.String)
    return pl.DataFrame(cols).with_columns(
        pl.when(pl.col(col).is_not_null() & pl.col(col).is_finite())
        .then(pl.col(col))
        .otherwise(None)
        .alias(col)
        for col in _FINITE_BASE_FLOAT_COLS
    )


def sort_bin_result(result: BinResult) -> BinResult:
    """Sort BinResult bins in natural chromosome order, preserving coords."""
    if len(result.contigs) <= 1:
        return result
    runs = partition_contig_runs(result.contigs)
    if len(runs) <= 1:
        return result
    unique_contigs = list(dict.fromkeys(name for name, _, _ in runs))
    sorted_contigs = sorted(unique_contigs, key=natural_sort_key)
    if len(runs) == len(unique_contigs) and unique_contigs == sorted_contigs:
        return result

    runs_by_contig: dict[str, list[tuple[int, int]]] = {}
    for name, lo, hi in runs:
        runs_by_contig.setdefault(name, []).append((lo, hi))

    idx_parts: list[NDArray[np.intp]] = []
    new_contigs: list[str] = []
    for name in sorted_contigs:
        contig_count = 0
        for lo, hi in runs_by_contig[name]:
            idx_parts.append(np.arange(lo, hi, dtype=np.intp))
            contig_count += hi - lo
        new_contigs.extend([name] * contig_count)

    idx = np.concatenate(idx_parts)
    return BinResult(
        branch=result.branch,
        contigs=tuple(new_contigs),
        starts=result.starts[idx],
        ends=result.ends[idx],
        ratios=result.ratios[idx],
        zscores=result.zscores[idx],
        weights=result.weights[idx],
        reference_sizes=result.reference_sizes[idx],
        masked=result.masked[idx],
        counts=result.counts[idx] if result.counts is not None else None,
        expected_counts=(
            result.expected_counts[idx] if result.expected_counts is not None else None
        ),
        background_counts=(
            result.background_counts[idx]
            if result.background_counts is not None
            else None
        ),
        baseline_log2_var=result.baseline_log2_var,
    )


def sort_bin_results(results: tuple[BinResult, ...]) -> tuple[BinResult, ...]:
    """Sort a collection of BinResult objects by natural chromosome order."""
    sorted_branches = tuple(sort_bin_result(res) for res in results)
    return tuple(
        sorted(
            sorted_branches,
            key=lambda res: (
                natural_sort_key(res.contigs[0]) if res.contigs else (),
                res.branch,
            ),
        )
    )


def sort_segments(segments: Iterable[SegmentResult]) -> tuple[SegmentResult, ...]:
    """Sort SegmentResult records naturally by chromosome and start position."""
    return tuple(
        sorted(
            segments,
            key=lambda s: (natural_sort_key(s.contig), s.start, s.end),
        )
    )


def segments_to_dataframe(
    segments: Sequence[SegmentResult],
    *,
    is_prenatal: bool = False,
) -> pl.DataFrame:
    """Unpack SegmentResult records once into a typed Polars DataFrame with null non-finite floats."""
    schema = _PRENATAL_SEGMENT_SCHEMA if is_prenatal else _GERMLINE_SEGMENT_SCHEMA
    if not segments:
        return pl.DataFrame(schema=schema)

    getter = _PRENATAL_SEGMENT_GETTER if is_prenatal else _GERMLINE_SEGMENT_GETTER
    columns = list(zip(*map(getter, segments), strict=True))
    df = pl.DataFrame(columns, schema=schema, orient="col")

    float_cols = (
        (*_FINITE_BASE_FLOAT_COLS, *_OPTIONAL_PRENATAL_FLOAT_COLS)
        if is_prenatal
        else _FINITE_BASE_FLOAT_COLS
    )
    return df.with_columns(
        pl.when(pl.col(col).is_not_null() & pl.col(col).is_finite())
        .then(pl.col(col))
        .otherwise(None)
        .alias(col)
        for col in float_cols
    )


def max_bin_end(results: Sequence[BinResult]) -> int:
    """Return the maximum bin end coordinate across a sequence of BinResult objects."""
    return max(
        (int(np.max(res.ends)) for res in results if res.ends.size > 0),
        default=0,
    )


def max_df_end(df: pl.DataFrame) -> int:
    """Return the maximum integer ``end`` coordinate in a segment Polars DataFrame."""
    if df.height == 0:
        return 0
    end_max = df.get_column("end").max()
    return (
        int(end_max)
        if isinstance(end_max, int) and not isinstance(end_max, bool)
        else 0
    )


def sort_segments_dataframe(
    df: pl.DataFrame,
    contig_names: Sequence[str],
) -> pl.DataFrame:
    """Sort a segment Polars DataFrame by explicit contig rank, start, and end."""
    if df.height <= 1:
        return df
    contig_rank = {c: idx for idx, c in enumerate(contig_names)}
    return (
        df.with_columns(
            pl.col("contig")
            .replace_strict(contig_rank, return_dtype=pl.Int32)
            .alias("_contig_rank")
        )
        .sort("_contig_rank", "start", "end", maintain_order=True)
        .drop("_contig_rank")
    )


def resolve_segments_dataframe(
    segments: Sequence[SegmentResult],
    segments_df: pl.DataFrame | None,
    *,
    calls_only: bool,
    is_prenatal: bool,
    sort_natural: bool = False,
) -> pl.DataFrame:
    """Reuse or build a segment Polars DataFrame with optional call filtering and sorting."""
    if segments_df is not None and (
        not is_prenatal or "expected_ratio" in segments_df.columns
    ):
        filtered = (
            segments_df.filter(pl.col("call").is_not_null())
            if calls_only and segments_df.get_column("call").null_count() > 0
            else segments_df
        )
        if sort_natural and filtered.height > 1:
            c_names = ordered_contig_names(
                filtered.get_column("contig").unique().to_list()
            )
            return sort_segments_dataframe(filtered, c_names)
        return filtered
    ordered: Iterable[SegmentResult] = (
        sort_segments(segments) if sort_natural else segments
    )
    items = [s for s in ordered if not calls_only or s.call is not None]
    return segments_to_dataframe(items, is_prenatal=is_prenatal)
