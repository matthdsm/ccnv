"""Formatted BED generation with single-pass Rust BGZF compression and Tabix/CSI indexing."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Final

import polars as pl

from ccnv.models import SegmentResult, natural_sort_key
from ccnv.report._common import (
    STREAM_CHUNK_ROWS,
    BinResults,
    ContigBinChunk,
    bin_chunk_to_dataframe,
    iter_contig_bin_chunks,
    max_bin_end,
    max_df_end,
    resolve_natural_contig_bounds,
    resolve_segments_dataframe,
    sort_bin_result,
    sort_bin_results,
    sort_segments,
    validate_threads,
    write_bgzf_atomic,
)

BED_CHUNK_ROWS: Final[int] = STREAM_CHUNK_ROWS

__all__ = [
    "BED_CHUNK_ROWS",
    "BINS_BED_HEADER",
    "CALLS_BED_HEADER",
    "PRENATAL_CALLS_BED_HEADER",
    "PRENATAL_SEGMENTS_BED_HEADER",
    "SEGMENTS_BED_HEADER",
    "BinResults",
    "ContigBinChunk",
    "iter_contig_bin_chunks",
    "natural_sort_key",
    "sort_bin_result",
    "sort_bin_results",
    "sort_segments",
    "write_bins_bed_streaming",
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


def write_bins_bed_streaming(
    gz_path: Path,
    bin_results: BinResults,
    *,
    contig_lengths: Mapping[str, int] | None = None,
    threads: int = 1,
) -> Path:
    """Stream bin evidence in 65,536-row Polars TSV slices to Rust BGZF+Tabix/CSI."""
    validate_threads(threads)
    record_max = max_bin_end(bin_results)
    contig_names, max_end = resolve_natural_contig_bounds(
        (c for res in bin_results for c in dict.fromkeys(res.contigs)),
        record_max,
        contig_lengths,
    )

    def _chunks() -> Iterator[bytes]:
        yield ("\t".join(BINS_BED_HEADER) + "\n").encode("utf-8")
        for chunk in iter_contig_bin_chunks(
            bin_results, contig_names, chunk_rows=BED_CHUNK_ROWS
        ):
            branch_expr = (
                pl.lit(chunk.branch, dtype=pl.String)
                if isinstance(chunk.branch, str)
                else pl.col("branch")
            )
            df = bin_chunk_to_dataframe(chunk).select(
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

    return write_bgzf_atomic(
        gz_path,
        "bed",
        contig_names,
        max_end,
        _chunks(),
        threads,
    )


def write_segments_bed_streaming(
    gz_path: Path,
    segments: Sequence[SegmentResult] = (),
    *,
    calls_only: bool = False,
    is_prenatal: bool = False,
    contig_lengths: Mapping[str, int] | None = None,
    threads: int = 1,
    segments_df: pl.DataFrame | None = None,
) -> Path:
    """Stream segment or call records in 65,536-row Polars TSV slices to Rust BGZF+Tabix/CSI."""
    validate_threads(threads)
    df = resolve_segments_dataframe(
        segments,
        segments_df,
        calls_only=calls_only,
        is_prenatal=is_prenatal,
        sort_natural=True,
    )

    if calls_only:
        header = PRENATAL_CALLS_BED_HEADER if is_prenatal else CALLS_BED_HEADER
        select_exprs: list[pl.Expr] = [
            pl.col("contig"),
            pl.col("start"),
            pl.col("end"),
            pl.col("ratio"),
            pl.col("zscore"),
            pl.col("call").alias("type"),
        ]
    else:
        header = PRENATAL_SEGMENTS_BED_HEADER if is_prenatal else SEGMENTS_BED_HEADER
        select_exprs = [
            pl.col("contig"),
            pl.col("start"),
            pl.col("end"),
            pl.col("ratio"),
            pl.col("zscore"),
            pl.col("weight"),
            pl.col("bin_count"),
            pl.col("call"),
        ]

    if is_prenatal:
        select_exprs.extend(
            [
                pl.col("expected_ratio"),
                pl.col("implied_ff"),
                pl.col("discordant"),
                pl.col("discordance_reason"),
                pl.col("frag_short_ratio"),
                pl.col("frag_llr_size"),
                pl.col("origin_call"),
            ]
        )

    out_df = df.select(select_exprs)
    record_max = max_df_end(out_df)
    contig_names, max_end = resolve_natural_contig_bounds(
        out_df.get_column("contig").unique(maintain_order=True).to_list(),
        record_max,
        contig_lengths,
    )

    def _chunks() -> Iterator[bytes]:
        yield ("\t".join(header) + "\n").encode("utf-8")
        for offset in range(0, out_df.height, BED_CHUNK_ROWS):
            csv_str = out_df.slice(offset, BED_CHUNK_ROWS).write_csv(
                None,
                separator="\t",
                include_header=False,
                float_precision=6,
                null_value="",
                quote_style="never",
            )
            yield csv_str.encode("utf-8")

    return write_bgzf_atomic(
        gz_path,
        "bed",
        contig_names,
        max_end,
        _chunks(),
        threads,
    )
