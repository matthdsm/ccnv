"""Formatted BED generation with single-pass Rust BGZF compression and Tabix/CSI indexing."""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import pyarrow as pa
from numpy.typing import NDArray

from ccnv._core import write_bgzf_indexed
from ccnv.models import BinResult, SegmentResult, natural_sort_key

type BinResults = Sequence[BinResult]

BED_CHUNK_ROWS: Final[int] = 65_536


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
    "BED_CHUNK_ROWS",
    "BINS_BED_HEADER",
    "CALLS_BED_HEADER",
    "PRENATAL_CALLS_BED_HEADER",
    "PRENATAL_SEGMENTS_BED_HEADER",
    "SEGMENTS_BED_HEADER",
    "BinResults",
    "ContigBinChunk",
    "bin_bed_rows",
    "calls_bed_rows",
    "format_bed_float",
    "iter_contig_bin_chunks",
    "natural_sort_key",
    "prenatal_bed_fields",
    "segment_bed_rows",
    "sort_bin_result",
    "sort_bin_results",
    "sort_segments",
    "write_bins_bed_streaming",
    "write_compressed_bed",
    "write_segments_bed_streaming",
]

BINS_BED_HEADER: Final[tuple[str, ...]] = (
    "#contig",
    "start",
    "end",
    "id",
    "ratio",
    "zscore",
    "weight",
    "reference_size",
    "masked",
    "branch",
)
SEGMENTS_BED_HEADER: Final[tuple[str, ...]] = (
    "#contig",
    "start",
    "end",
    "ratio",
    "zscore",
    "weight",
    "bin_count",
    "call",
)
CALLS_BED_HEADER: Final[tuple[str, ...]] = (
    "#contig",
    "start",
    "end",
    "ratio",
    "zscore",
    "type",
)
PRENATAL_SEGMENTS_BED_HEADER: Final[tuple[str, ...]] = (
    "#contig",
    "start",
    "end",
    "ratio",
    "zscore",
    "weight",
    "bin_count",
    "call",
    "expected_ratio",
    "implied_ff",
    "discordant",
    "discordance_reason",
    "frag_short_ratio",
    "frag_llr_size",
    "origin_call",
)
PRENATAL_CALLS_BED_HEADER: Final[tuple[str, ...]] = (
    "#contig",
    "start",
    "end",
    "ratio",
    "zscore",
    "type",
    "expected_ratio",
    "implied_ff",
    "discordant",
    "discordance_reason",
    "frag_short_ratio",
    "frag_llr_size",
    "origin_call",
)


def _validate_threads(threads: int) -> None:
    if isinstance(threads, bool) or threads < 1:
        raise ValueError("threads must be a positive integer")


def _finite_or_nan(value: float) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def _optional_bed_float(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) if np.isfinite(value) else float("nan")


def _merge_contig_bounds(
    seen_contigs: dict[str, None],
    max_end: int,
    contig_lengths: Mapping[str, int] | None,
) -> tuple[list[str], int]:
    if contig_lengths:
        for contig_name in sorted(contig_lengths, key=natural_sort_key):
            seen_contigs.setdefault(contig_name, None)
        cl_max = max(contig_lengths.values(), default=0)
        max_end = max(max_end, cl_max)
    return list(seen_contigs.keys()), max_end


def _resolve_natural_contig_bounds(
    record_contigs: Iterable[str],
    record_max_end: int,
    contig_lengths: Mapping[str, int] | None,
) -> tuple[list[str], int]:
    all_contigs = set(record_contigs)
    max_end = record_max_end
    if contig_lengths:
        all_contigs.update(contig_lengths.keys())
        max_end = max(max_end, max(contig_lengths.values(), default=0))
    return sorted(all_contigs, key=natural_sort_key), max_end


def _partition_bin_result_runs(res: BinResult) -> list[tuple[str, int, int]]:
    if not res.contigs:
        return []
    dict_arr = pa.array(res.contigs, type=pa.string()).dictionary_encode()
    vocab: list[str] = dict_arr.dictionary.to_pylist()
    codes = dict_arr.indices.to_numpy(zero_copy_only=False)
    change = np.empty(codes.size, dtype=np.bool_)
    change[0] = True
    change[1:] = codes[1:] != codes[:-1]
    starts = np.flatnonzero(change)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = codes.size
    return [
        (vocab[int(codes[s])], int(s), int(e))
        for s, e in zip(starts.tolist(), ends.tolist(), strict=True)
    ]


def iter_contig_bin_chunks(
    bin_results: BinResults,
    contig_order: Sequence[str] = (),
    *,
    chunk_rows: int = BED_CHUNK_ROWS,
) -> Iterator[ContigBinChunk]:
    """Yield coordinate-sorted per-contig bin slices in exact contig_order."""
    slices_by_contig: dict[str, list[tuple[BinResult, int, int]]] = {}
    for res in bin_results:
        for contig_name, lo, hi in _partition_bin_result_runs(res):
            slices_by_contig.setdefault(contig_name, []).append((res, lo, hi))

    ordered_names = list(dict.fromkeys(contig_order))
    seen = set(ordered_names)
    ordered_names.extend(
        sorted((c for c in slices_by_contig if c not in seen), key=natural_sort_key)
    )

    for contig_name in ordered_names:
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


def format_bed_float(value: float) -> str:
    """Format floating point numbers for BED output."""
    if not np.isfinite(value):
        return "NaN"
    return f"{value:.8g}"


def sort_bin_result(result: BinResult) -> BinResult:
    """Sort BinResult bins in natural chromosome order, preserving coords."""
    if len(result.contigs) <= 1:
        return result
    unique_contigs = list(dict.fromkeys(result.contigs))
    sorted_contigs = sorted(unique_contigs, key=natural_sort_key)
    if unique_contigs == sorted_contigs:
        return result

    contig_indices: dict[str, list[int]] = {}
    for i, c in enumerate(result.contigs):
        contig_indices.setdefault(c, []).append(i)

    idx_list: list[int] = []
    new_contigs: list[str] = []
    for c in sorted_contigs:
        indices = contig_indices[c]
        idx_list.extend(indices)
        new_contigs.extend([c] * len(indices))

    idx = np.array(idx_list, dtype=np.intp)
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


def bin_bed_rows(results: tuple[BinResult, ...]) -> Iterator[tuple[object, ...]]:
    """Format bin records into BED tuples."""
    for chunk in iter_contig_bin_chunks(results):
        for index in range(chunk.starts.size):
            start = int(chunk.starts[index])
            end = int(chunk.ends[index])
            branch_val = (
                chunk.branch if isinstance(chunk.branch, str) else chunk.branch[index]
            )
            yield (
                chunk.contig,
                start,
                end,
                f"{chunk.contig}:{start}-{end}",
                format_bed_float(float(chunk.ratios[index])),
                format_bed_float(float(chunk.zscores[index])),
                format_bed_float(float(chunk.weights[index])),
                int(chunk.reference_sizes[index]),
                "true" if bool(chunk.masked[index]) else "false",
                branch_val,
            )


def prenatal_bed_fields(item: SegmentResult) -> tuple[object, ...]:
    """Return prenatal discordance and fragmentomic origin annotations for a segment."""
    ratio_str = (
        format_bed_float(item.expected_ratio) if item.expected_ratio is not None else ""
    )
    ff_str = format_bed_float(item.implied_ff) if item.implied_ff is not None else ""
    sr_str = (
        format_bed_float(item.frag_short_ratio)
        if item.frag_short_ratio is not None
        else ""
    )
    llr_str = (
        format_bed_float(item.frag_llr_size) if item.frag_llr_size is not None else ""
    )
    return (
        ratio_str,
        ff_str,
        "true" if item.discordant else "false",
        item.discordance_reason or "",
        sr_str,
        llr_str,
        item.origin_call or "",
    )


def segment_bed_rows(
    results: tuple[SegmentResult, ...],
    *,
    is_prenatal: bool = False,
) -> tuple[tuple[object, ...], ...]:
    """Format segment records into BED tuples."""
    return tuple(
        (
            item.contig,
            item.start,
            item.end,
            format_bed_float(item.ratio),
            format_bed_float(item.zscore),
            format_bed_float(item.weight),
            item.bin_count,
            item.call or "",
            *(prenatal_bed_fields(item) if is_prenatal else ()),
        )
        for item in results
    )


def calls_bed_rows(
    calls: Sequence[SegmentResult],
    *,
    is_prenatal: bool = False,
) -> tuple[tuple[object, ...], ...]:
    """Format call records into BED tuples."""
    return tuple(
        (
            item.contig,
            item.start,
            item.end,
            format_bed_float(item.ratio),
            format_bed_float(item.zscore),
            item.call,
            *(prenatal_bed_fields(item) if is_prenatal else ()),
        )
        for item in calls
    )


def write_bins_bed_streaming(
    gz_path: Path,
    bin_results: BinResults,
    *,
    contig_lengths: Mapping[str, int] | None = None,
    threads: int = 1,
) -> Path:
    """Stream bin evidence in 65,536-row Polars TSV slices to Rust BGZF+Tabix/CSI."""
    _validate_threads(threads)
    record_max = max(
        (int(np.max(res.ends)) for res in bin_results if res.ends.size > 0),
        default=0,
    )
    contig_names, max_end = _resolve_natural_contig_bounds(
        (c for res in bin_results for c in dict.fromkeys(res.contigs)),
        record_max,
        contig_lengths,
    )

    def _chunks() -> Iterator[bytes]:
        yield ("\t".join(BINS_BED_HEADER) + "\n").encode("utf-8")
        for chunk in iter_contig_bin_chunks(
            bin_results, contig_names, chunk_rows=BED_CHUNK_ROWS
        ):
            cols: dict[str, object] = {
                "start": chunk.starts,
                "end": chunk.ends,
                "ratio": np.where(
                    np.isfinite(chunk.ratios), chunk.ratios, np.nan
                ).astype(np.float64, copy=False),
                "zscore": np.where(
                    np.isfinite(chunk.zscores), chunk.zscores, np.nan
                ).astype(np.float64, copy=False),
                "weight": np.where(
                    np.isfinite(chunk.weights), chunk.weights, np.nan
                ).astype(np.float64, copy=False),
                "reference_size": chunk.reference_sizes,
                "masked": chunk.masked,
            }
            if isinstance(chunk.branch, str):
                branch_expr = pl.lit(chunk.branch, dtype=pl.String)
            else:
                cols["branch"] = pl.Series("branch", chunk.branch, dtype=pl.String)
                branch_expr = pl.col("branch")
            df = pl.DataFrame(cols).select(
                pl.lit(chunk.contig, dtype=pl.String).alias("contig"),
                pl.col("start"),
                pl.col("end"),
                pl.concat_str(
                    [
                        pl.lit(f"{chunk.contig}:"),
                        pl.col("start").cast(pl.String),
                        pl.lit("-"),
                        pl.col("end").cast(pl.String),
                    ]
                ).alias("id"),
                pl.col("ratio"),
                pl.col("zscore"),
                pl.col("weight"),
                pl.col("reference_size"),
                pl.col("masked"),
                branch_expr.alias("branch"),
            )
            csv_str = df.write_csv(
                None,
                separator="\t",
                include_header=False,
                float_precision=6,
                null_value=".",
                quote_style="never",
            )
            yield csv_str.encode("utf-8")

    write_bgzf_indexed(
        str(gz_path),
        "bed",
        contig_names,
        max_end,
        _chunks(),
        threads,
    )
    return gz_path


def write_segments_bed_streaming(
    gz_path: Path,
    segments: Sequence[SegmentResult],
    *,
    calls_only: bool = False,
    is_prenatal: bool = False,
    contig_lengths: Mapping[str, int] | None = None,
    threads: int = 1,
) -> Path:
    """Stream segment or call records in 65,536-row Polars TSV slices to Rust BGZF+Tabix/CSI."""
    _validate_threads(threads)
    items = [s for s in sort_segments(segments) if not calls_only or s.call is not None]
    if calls_only:
        header = PRENATAL_CALLS_BED_HEADER if is_prenatal else CALLS_BED_HEADER
    else:
        header = PRENATAL_SEGMENTS_BED_HEADER if is_prenatal else SEGMENTS_BED_HEADER

    contig_names, max_end = _resolve_natural_contig_bounds(
        (item.contig for item in items),
        max((int(item.end) for item in items), default=0),
        contig_lengths,
    )

    def _chunks() -> Iterator[bytes]:
        yield ("\t".join(header) + "\n").encode("utf-8")
        for offset in range(0, len(items), BED_CHUNK_ROWS):
            batch = items[offset : offset + BED_CHUNK_ROWS]
            cols: dict[str, pl.Series] = {
                "contig": pl.Series(
                    "contig", [s.contig for s in batch], dtype=pl.String
                ),
                "start": pl.Series(
                    "start", [int(s.start) for s in batch], dtype=pl.Int64
                ),
                "end": pl.Series("end", [int(s.end) for s in batch], dtype=pl.Int64),
                "ratio": pl.Series(
                    "ratio", [_finite_or_nan(s.ratio) for s in batch], dtype=pl.Float64
                ),
                "zscore": pl.Series(
                    "zscore",
                    [_finite_or_nan(s.zscore) for s in batch],
                    dtype=pl.Float64,
                ),
            }
            if calls_only:
                cols["type"] = pl.Series(
                    "type", [s.call or "" for s in batch], dtype=pl.String
                )
            else:
                cols["weight"] = pl.Series(
                    "weight",
                    [_finite_or_nan(s.weight) for s in batch],
                    dtype=pl.Float64,
                )
                cols["bin_count"] = pl.Series(
                    "bin_count", [int(s.bin_count) for s in batch], dtype=pl.Int64
                )
                cols["call"] = pl.Series(
                    "call", [s.call or "" for s in batch], dtype=pl.String
                )
            if is_prenatal:
                cols["expected_ratio"] = pl.Series(
                    "expected_ratio",
                    [_optional_bed_float(s.expected_ratio) for s in batch],
                    dtype=pl.Float64,
                )
                cols["implied_ff"] = pl.Series(
                    "implied_ff",
                    [_optional_bed_float(s.implied_ff) for s in batch],
                    dtype=pl.Float64,
                )
                cols["discordant"] = pl.Series(
                    "discordant", [bool(s.discordant) for s in batch], dtype=pl.Boolean
                )
                cols["discordance_reason"] = pl.Series(
                    "discordance_reason",
                    [s.discordance_reason or "" for s in batch],
                    dtype=pl.String,
                )
                cols["frag_short_ratio"] = pl.Series(
                    "frag_short_ratio",
                    [_optional_bed_float(s.frag_short_ratio) for s in batch],
                    dtype=pl.Float64,
                )
                cols["frag_llr_size"] = pl.Series(
                    "frag_llr_size",
                    [_optional_bed_float(s.frag_llr_size) for s in batch],
                    dtype=pl.Float64,
                )
                cols["origin_call"] = pl.Series(
                    "origin_call",
                    [s.origin_call or "" for s in batch],
                    dtype=pl.String,
                )
            csv_str = pl.DataFrame(cols).write_csv(
                None,
                separator="\t",
                include_header=False,
                float_precision=6,
                null_value="",
                quote_style="never",
            )
            yield csv_str.encode("utf-8")

    write_bgzf_indexed(
        str(gz_path),
        "bed",
        contig_names,
        max_end,
        _chunks(),
        threads,
    )
    return gz_path


def write_compressed_bed(
    gz_path: Path,
    header: Sequence[str],
    rows: Iterable[Sequence[object]],
    *,
    contig_lengths: Mapping[str, int] | None = None,
    threads: int = 1,
) -> Path:
    """Write tab-delimited records to a BGZF-compressed file with Tabix/CSI index."""
    _validate_threads(threads)
    row_list = rows if isinstance(rows, Sequence) else list(rows)
    seen_contigs: dict[str, None] = {}
    max_end = 0
    for row in row_list:
        seen_contigs[str(row[0])] = None
        end_val = int(str(row[2]))
        max_end = max(max_end, end_val)
    contig_names, max_end = _merge_contig_bounds(seen_contigs, max_end, contig_lengths)

    def _chunks() -> Iterator[bytes]:
        yield ("\t".join(header) + "\n").encode("utf-8")
        for offset in range(0, len(row_list), BED_CHUNK_ROWS):
            batch = row_list[offset : offset + BED_CHUNK_ROWS]
            block = "".join(
                "\t".join(str(value) for value in row) + "\n" for row in batch
            )
            yield block.encode("utf-8")

    write_bgzf_indexed(
        str(gz_path),
        "bed",
        contig_names,
        max_end,
        _chunks(),
        threads,
    )
    return gz_path
