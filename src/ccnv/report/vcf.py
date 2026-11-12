"""Structural variant VCFv4.2 streaming writer via Rust BGZF and Tabix/CSI indexing."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Final, Literal

import polars as pl

from ccnv.models import (
    BinResult,
    FetalFractionResult,
    SegmentResult,
    natural_sort_key,
)
from ccnv.report._common import (
    STREAM_CHUNK_ROWS,
    bin_chunk_to_dataframe,
    iter_contig_bin_chunks,
    max_bin_end,
    max_df_end,
    resolve_segments_dataframe,
    sort_segments_dataframe,
    validate_threads,
    write_bgzf_atomic,
)

VCF_CHUNK_ROWS: Final[int] = STREAM_CHUNK_ROWS
_CONTIG_HEADER_RE: Final[re.Pattern[str]] = re.compile(
    r"^##contig=<ID=([^,>]+)(?:,.*?length=(\d+))?.*>$"
)

__all__ = [
    "VCF_CHUNK_ROWS",
    "build_vcf_header",
    "write_bins_vcf",
    "write_segments_vcf",
    "write_vcf_streaming",
]


def _fmt_vcf_float_expr(col_name: str) -> pl.Expr:
    col = pl.col(col_name)
    return (
        pl.when(col.is_not_null() & col.is_finite())
        .then(col.round(6).cast(pl.String))
        .otherwise(pl.lit("."))
    )


def _extract_header_contigs(header_lines: Sequence[str]) -> tuple[list[str], int]:
    names: list[str] = []
    max_len = 0
    for line in header_lines:
        match = _CONTIG_HEADER_RE.match(line)
        if match is not None:
            names.append(match.group(1))
            len_token = match.group(2)
            if len_token is not None:
                max_len = max(max_len, int(len_token))
    return names, max_len


def _merge_vcf_contig_bounds(
    header_lines: Sequence[str],
    record_contigs: Iterable[str],
    record_max_end: int,
    contig_lengths: Mapping[str, int] | None,
) -> tuple[list[str], int]:
    hdr_contigs, max_len = _extract_header_contigs(header_lines)
    max_len = max(max_len, record_max_end)
    seen_contigs: dict[str, None] = dict.fromkeys(hdr_contigs)
    extra_contigs = set(record_contigs)
    if contig_lengths:
        extra_contigs.update(contig_lengths.keys())
        max_len = max(max_len, max(contig_lengths.values(), default=0))
    for contig_name in sorted(extra_contigs, key=natural_sort_key):
        seen_contigs.setdefault(contig_name, None)
    return list(seen_contigs.keys()), max_len


def build_vcf_header(
    contig_lengths: Mapping[str, int],
    sample_name: str,
    record_type: Literal["bins", "segments"],
    extra_contigs: Iterable[str] = (),
    prenatal_result: FetalFractionResult | None = None,
) -> tuple[str, ...]:
    """Construct VCFv4.2 header lines declared with contigs, INFO, and FORMAT fields."""
    if not sample_name:
        raise ValueError("sample_name must be non-empty")

    lines: list[str] = [
        "##fileformat=VCFv4.2",
        '##FILTER=<ID=PASS,Description="All filters passed">',
        "##source=ccnv_predict",
    ]
    all_contigs = set(contig_lengths.keys()) | set(extra_contigs)
    for contig in sorted(all_contigs, key=natural_sort_key):
        length = contig_lengths.get(contig)
        if length is not None:
            lines.append(f"##contig=<ID={contig},length={length}>")
        else:
            lines.append(f"##contig=<ID={contig}>")

    lines.extend(
        [
            '##INFO=<ID=END,Number=1,Type=Integer,Description="End coordinate of structural variant">',
            '##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">',
            '##FORMAT=<ID=LOG2R,Number=1,Type=Float,Description="Log2 copy ratio against within-sample reference">',
            '##FORMAT=<ID=ZSCORE,Number=1,Type=Float,Description="Z-score relative to reference cohort null model">',
            '##FORMAT=<ID=WEIGHT,Number=1,Type=Float,Description="Precision weight for segmentation">',
        ]
    )

    if record_type == "bins":
        lines.extend(
            [
                '##ALT=<ID=CNV,Description="Copy number variation">',
                '##INFO=<ID=BRANCH,Number=1,Type=String,Description="Model branch">',
                '##FILTER=<ID=MASKED,Description="Bin masked due to low coverage or quality filters">',
                '##FORMAT=<ID=REF_SIZE,Number=1,Type=Integer,Description="Number of within-sample reference bins used">',
            ]
        )
    else:
        lines.extend(
            [
                '##ALT=<ID=DUP,Description="Duplication / copy number gain">',
                '##ALT=<ID=DEL,Description="Deletion / copy number loss">',
                '##ALT=<ID=CNV,Description="Copy number variation">',
                '##INFO=<ID=IMPRECISE,Number=0,Type=Flag,Description="Imprecise structural variation">',
                '##FORMAT=<ID=BINS,Number=1,Type=Integer,Description="Number of bins in segment">',
            ]
        )
        if prenatal_result is not None:
            lines.extend(
                [
                    f"##fetalSex={prenatal_result.fetal_sex}",
                    f"##fetalFraction={prenatal_result.active_ff:.4f}",
                    f"##scaCall={prenatal_result.sca.call}",
                    (
                        "##INFO=<ID=DISCORDANT,Number=0,Type=Flag,"
                        'Description="Segment ratio is discordant with fetal '
                        'fraction or karyotype">'
                    ),
                    '##INFO=<ID=DISC_REASON,Number=1,Type=String,Description="Reason for discordance flag">',
                    '##FORMAT=<ID=EXP_LOG2R,Number=1,Type=Float,Description="Expected log2 ratio based on active fetal fraction">',
                    '##FORMAT=<ID=IMPLIED_FF,Number=1,Type=Float,Description="Implied fetal fraction based on segment ratio">',
                    '##INFO=<ID=FRAG_SHORT_R,Number=1,Type=Float,Description="Fraction of short cell-free DNA fragments < 150 bp">',
                    '##INFO=<ID=FRAG_LLR,Number=1,Type=Float,Description="Fragment length log-likelihood ratio for fetal vs maternal origin">',
                    (
                        "##INFO=<ID=ORIGIN,Number=1,Type=String,"
                        'Description="Fragmentomic origin call: FETAL_CONSTITUTIONAL, '
                        "PLACENTAL_MOSAIC, MATERNAL_UNINHERITED, "
                        'MATERNAL_AND_FETAL_INHERITED, MATERNAL_MOSAIC, or INDETERMINATE">'
                    ),
                    (
                        "##INFO=<ID=MOSAIC_FRAC,Number=1,Type=Float,"
                        'Description="Inferred mosaic fraction for mosaic origin calls">'
                    ),
                ]
            )

    lines.append(
        f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample_name}"
    )
    return tuple(lines)


def write_vcf_streaming(
    gz_path: Path,
    header_lines: Sequence[str],
    body_chunks: Iterable[bytes],
    contig_names: Sequence[str],
    max_contig_len: int,
    *,
    threads: int = 1,
) -> Path:
    """Stream VCFv4.2 header and UTF-8 body byte chunks into Rust BGZF+Tabix/CSI."""
    validate_threads(threads)
    if isinstance(max_contig_len, bool) or max_contig_len < 0:
        raise ValueError("max_contig_len must be a non-negative integer")

    def _chunks() -> Iterator[bytes]:
        if header_lines:
            header_text = "\n".join(line.rstrip("\r\n") for line in header_lines) + "\n"
            yield header_text.encode("utf-8")
        yield from body_chunks

    return write_bgzf_atomic(
        gz_path,
        "vcf",
        contig_names,
        max_contig_len,
        _chunks(),
        threads,
    )


def write_bins_vcf(
    gz_path: Path,
    header: Sequence[str],
    results: Sequence[BinResult],
    sample_name: str,
    *,
    contig_lengths: Mapping[str, int] | None = None,
    threads: int = 1,
) -> Path:
    """Write bin-level copy ratios to a compressed, indexed VCFv4.2 in 65,536-row chunks."""
    if not sample_name:
        raise ValueError("sample_name must be non-empty")
    record_max = max_bin_end(results)
    contig_names, max_len = _merge_vcf_contig_bounds(
        header,
        (c for res in results for c in dict.fromkeys(res.contigs)),
        record_max,
        contig_lengths,
    )

    def _body_chunks() -> Iterator[bytes]:
        for chunk in iter_contig_bin_chunks(
            results, contig_names, chunk_rows=VCF_CHUNK_ROWS
        ):
            if isinstance(chunk.branch, str):
                info_expr = pl.concat_str(
                    [
                        pl.lit("END="),
                        pl.col("end").cast(pl.String),
                        pl.lit(f";SVTYPE=CNV;BRANCH={chunk.branch}"),
                    ]
                )
            else:
                info_expr = pl.concat_str(
                    [
                        pl.lit("END="),
                        pl.col("end").cast(pl.String),
                        pl.lit(";SVTYPE=CNV;BRANCH="),
                        pl.col("branch"),
                    ]
                )
            df = bin_chunk_to_dataframe(chunk).select(
                pl.lit(chunk.contig, dtype=pl.String).alias("chrom"),
                (pl.col("start") + 1).alias("pos"),
                pl.concat_str(
                    [
                        pl.lit(f"{chunk.contig}:"),
                        pl.col("start").cast(pl.String),
                        pl.lit("-"),
                        pl.col("end").cast(pl.String),
                    ]
                ).alias("id"),
                pl.lit("N", dtype=pl.String).alias("ref"),
                pl.lit("<CNV>", dtype=pl.String).alias("alt"),
                pl.lit(None, dtype=pl.String).alias("qual"),
                pl.when(pl.col("masked"))
                .then(pl.lit("MASKED"))
                .otherwise(pl.lit("PASS"))
                .alias("filter"),
                info_expr.alias("info"),
                pl.lit("LOG2R:ZSCORE:WEIGHT:REF_SIZE", dtype=pl.String).alias("format"),
                pl.concat_str(
                    [
                        _fmt_vcf_float_expr("ratio"),
                        _fmt_vcf_float_expr("zscore"),
                        _fmt_vcf_float_expr("weight"),
                        pl.col("reference_size").cast(pl.String),
                    ],
                    separator=":",
                ).alias("sample"),
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

    return write_vcf_streaming(
        gz_path,
        header,
        _body_chunks(),
        contig_names,
        max_len,
        threads=threads,
    )


def write_segments_vcf(
    gz_path: Path,
    header: Sequence[str],
    results: Sequence[SegmentResult] = (),
    sample_name: str = "SAMPLE",
    *,
    calls_only: bool = False,
    is_prenatal: bool = False,
    contig_lengths: Mapping[str, int] | None = None,
    threads: int = 1,
    segments_df: pl.DataFrame | None = None,
) -> Path:
    """Write segment-level calls and metrics to a compressed, indexed VCFv4.2."""
    if not sample_name:
        raise ValueError("sample_name must be non-empty")

    df_in = resolve_segments_dataframe(
        results,
        segments_df,
        calls_only=calls_only,
        is_prenatal=is_prenatal,
        sort_natural=False,
    )

    record_max = max_df_end(df_in)
    contig_names, max_len = _merge_vcf_contig_bounds(
        header,
        df_in.get_column("contig").unique(maintain_order=True).to_list(),
        record_max,
        contig_lengths,
    )
    sorted_df = sort_segments_dataframe(df_in, contig_names)

    svtype_expr = (
        pl.when(pl.col("call") == "gain")
        .then(pl.lit("DUP"))
        .when(pl.col("call") == "loss")
        .then(pl.lit("DEL"))
        .otherwise(pl.lit("CNV"))
    )
    info_parts: list[pl.Expr] = [
        pl.lit("END="),
        pl.col("end").cast(pl.String),
        pl.lit(";SVTYPE="),
        svtype_expr,
        pl.lit(";IMPRECISE"),
    ]
    if is_prenatal:
        info_parts.extend(
            [
                pl.when(pl.col("discordant"))
                .then(pl.lit(";DISCORDANT"))
                .otherwise(pl.lit("")),
                pl.when(
                    pl.col("discordant")
                    & pl.col("discordance_reason").is_not_null()
                    & (pl.col("discordance_reason") != "")
                )
                .then(
                    pl.concat_str(
                        [pl.lit(";DISC_REASON="), pl.col("discordance_reason")]
                    )
                )
                .otherwise(pl.lit("")),
                pl.when(
                    pl.col("frag_short_ratio").is_not_null()
                    & pl.col("frag_short_ratio").is_finite()
                )
                .then(
                    pl.concat_str(
                        [
                            pl.lit(";FRAG_SHORT_R="),
                            pl.col("frag_short_ratio").round(6).cast(pl.String),
                        ]
                    )
                )
                .otherwise(pl.lit("")),
                pl.when(
                    pl.col("frag_llr_size").is_not_null()
                    & pl.col("frag_llr_size").is_finite()
                )
                .then(
                    pl.concat_str(
                        [
                            pl.lit(";FRAG_LLR="),
                            pl.col("frag_llr_size").round(6).cast(pl.String),
                        ]
                    )
                )
                .otherwise(pl.lit("")),
                pl.when(
                    pl.col("origin_call").is_not_null() & (pl.col("origin_call") != "")
                )
                .then(pl.concat_str([pl.lit(";ORIGIN="), pl.col("origin_call")]))
                .otherwise(pl.lit("")),
                pl.when(
                    pl.col("mosaic_fraction").is_not_null()
                    & pl.col("mosaic_fraction").is_finite()
                )
                .then(
                    pl.concat_str(
                        [
                            pl.lit(";MOSAIC_FRAC="),
                            pl.col("mosaic_fraction").round(4).cast(pl.String),
                        ]
                    )
                )
                .otherwise(pl.lit("")),
            ]
        )
        fmt_str = "LOG2R:ZSCORE:WEIGHT:BINS:EXP_LOG2R:IMPLIED_FF"
        sample_parts = [
            _fmt_vcf_float_expr("ratio"),
            _fmt_vcf_float_expr("zscore"),
            _fmt_vcf_float_expr("weight"),
            pl.col("bin_count").cast(pl.String),
            _fmt_vcf_float_expr("expected_ratio"),
            _fmt_vcf_float_expr("implied_ff"),
        ]
    else:
        fmt_str = "LOG2R:ZSCORE:WEIGHT:BINS"
        sample_parts = [
            _fmt_vcf_float_expr("ratio"),
            _fmt_vcf_float_expr("zscore"),
            _fmt_vcf_float_expr("weight"),
            pl.col("bin_count").cast(pl.String),
        ]

    def _body_chunks() -> Iterator[bytes]:
        for offset in range(0, sorted_df.height, VCF_CHUNK_ROWS):
            batch_df = sorted_df.slice(offset, VCF_CHUNK_ROWS).select(
                pl.col("contig").alias("chrom"),
                (pl.col("start") + 1).alias("pos"),
                pl.lit(None, dtype=pl.String).alias("id"),
                pl.lit("N", dtype=pl.String).alias("ref"),
                pl.when(pl.col("call") == "gain")
                .then(pl.lit("<DUP>"))
                .when(pl.col("call") == "loss")
                .then(pl.lit("<DEL>"))
                .otherwise(pl.lit("<CNV>"))
                .alias("alt"),
                pl.lit(None, dtype=pl.String).alias("qual"),
                pl.when(pl.col("call").is_in(["gain", "loss"]))
                .then(pl.lit("PASS"))
                .otherwise(pl.lit(None, dtype=pl.String))
                .alias("filter"),
                pl.concat_str(info_parts).alias("info"),
                pl.lit(fmt_str, dtype=pl.String).alias("format"),
                pl.concat_str(sample_parts, separator=":").alias("sample"),
            )
            csv_str = batch_df.write_csv(
                None,
                separator="\t",
                include_header=False,
                float_precision=6,
                null_value=".",
                quote_style="never",
            )
            yield csv_str.encode("utf-8")

    return write_vcf_streaming(
        gz_path,
        header,
        _body_chunks(),
        contig_names,
        max_len,
        threads=threads,
    )
