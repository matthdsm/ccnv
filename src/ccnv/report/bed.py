"""Formatted BED generation with single-pass Rust BGZF compression and Tabix/CSI indexing."""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

from ccnv._core import write_bgzf_indexed
from ccnv.models import BinResult, SegmentResult, natural_sort_key

type BinResults = Sequence[BinResult]

BED_CHUNK_ROWS: Final[int] = 65_536

__all__ = [
    "BED_CHUNK_ROWS",
    "BINS_BED_HEADER",
    "CALLS_BED_HEADER",
    "PRENATAL_CALLS_BED_HEADER",
    "PRENATAL_SEGMENTS_BED_HEADER",
    "SEGMENTS_BED_HEADER",
    "BinResults",
    "bin_bed_rows",
    "calls_bed_rows",
    "format_bed_float",
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
    for result in results:
        for index in range(len(result.contigs)):
            contig = result.contigs[index]
            start = int(result.starts[index])
            end = int(result.ends[index])
            yield (
                contig,
                start,
                end,
                f"{contig}:{start}-{end}",
                format_bed_float(float(result.ratios[index])),
                format_bed_float(float(result.zscores[index])),
                format_bed_float(float(result.weights[index])),
                int(result.reference_sizes[index]),
                "true" if bool(result.masked[index]) else "false",
                result.branch,
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
    ordered_results = sort_bin_results(tuple(bin_results))

    seen_contigs: dict[str, None] = {}
    max_end = 0
    for res in ordered_results:
        for contig_name in dict.fromkeys(res.contigs):
            seen_contigs[contig_name] = None
        if res.ends.size > 0:
            res_max = int(np.max(res.ends))
            max_end = max(max_end, res_max)
    contig_names, max_end = _merge_contig_bounds(seen_contigs, max_end, contig_lengths)

    def _chunks() -> Iterator[bytes]:
        yield ("\t".join(BINS_BED_HEADER) + "\n").encode("utf-8")
        for res in ordered_results:
            n_rows = len(res.contigs)
            for offset in range(0, n_rows, BED_CHUNK_ROWS):
                stop = min(offset + BED_CHUNK_ROWS, n_rows)
                r_raw = res.ratios[offset:stop]
                z_raw = res.zscores[offset:stop]
                w_raw = res.weights[offset:stop]
                df = pl.DataFrame(
                    {
                        "contig": pl.Series(
                            "contig", res.contigs[offset:stop], dtype=pl.String
                        ),
                        "start": np.asarray(res.starts[offset:stop], dtype=np.int64),
                        "end": np.asarray(res.ends[offset:stop], dtype=np.int64),
                        "ratio": np.where(np.isfinite(r_raw), r_raw, np.nan).astype(
                            np.float64, copy=False
                        ),
                        "zscore": np.where(np.isfinite(z_raw), z_raw, np.nan).astype(
                            np.float64, copy=False
                        ),
                        "weight": np.where(np.isfinite(w_raw), w_raw, np.nan).astype(
                            np.float64, copy=False
                        ),
                        "reference_size": np.asarray(
                            res.reference_sizes[offset:stop], dtype=np.int64
                        ),
                        "masked": np.asarray(res.masked[offset:stop], dtype=np.bool_),
                    }
                ).select(
                    pl.col("contig"),
                    pl.col("start"),
                    pl.col("end"),
                    pl.concat_str(
                        [
                            pl.col("contig"),
                            pl.lit(":"),
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
                    pl.lit(res.branch, dtype=pl.String).alias("branch"),
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

    seen_contigs: dict[str, None] = {}
    max_end = 0
    for item in items:
        seen_contigs[item.contig] = None
        if item.end > max_end:
            max_end = int(item.end)
    contig_names, max_end = _merge_contig_bounds(seen_contigs, max_end, contig_lengths)

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
