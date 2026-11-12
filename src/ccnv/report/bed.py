"""Formatted BED generation with single-pass Rust BGZF compression and Tabix/CSI indexing."""

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

from ccnv._core import write_bgzf_indexed
from ccnv.models import SegmentResult, natural_sort_key
from ccnv.report._common import (
    STREAM_CHUNK_ROWS,
    BinResults,
    ContigBinChunk,
    iter_contig_bin_chunks,
    resolve_natural_contig_bounds,
    segments_to_dataframe,
    sort_bin_result,
    sort_bin_results,
    sort_segments,
    validate_threads,
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
    record_max = max(
        (int(np.max(res.ends)) for res in bin_results if res.ends.size > 0),
        default=0,
    )
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
    if segments_df is not None and (
        not is_prenatal or "expected_ratio" in segments_df.columns
    ):
        df = (
            segments_df.filter(pl.col("call").is_not_null())
            if calls_only and segments_df.get_column("call").null_count() > 0
            else segments_df
        )
    else:
        items = [
            s for s in sort_segments(segments) if not calls_only or s.call is not None
        ]
        df = segments_to_dataframe(items, is_prenatal=is_prenatal)

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
    end_max = out_df.get_column("end").max()
    record_max = (
        int(end_max)
        if isinstance(end_max, int) and not isinstance(end_max, bool)
        else 0
    )
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

    write_bgzf_indexed(
        str(gz_path),
        "bed",
        contig_names,
        max_end,
        _chunks(),
        threads,
    )
    return gz_path
