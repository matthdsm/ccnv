"""Structural variant VCFv4.2 streaming writer via Rust BGZF and Tabix/CSI indexing."""

import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Final, Literal

import numpy as np
import polars as pl

from ccnv._core import write_bgzf_indexed
from ccnv.models import (
    BinResult,
    FetalFractionResult,
    SegmentResult,
    natural_sort_key,
)
from ccnv.report.bed import iter_contig_bin_chunks

VCF_CHUNK_ROWS: Final[int] = 65_536
_CONTIG_HEADER_RE: Final[re.Pattern[str]] = re.compile(
    r"^##contig=<ID=([^,>]+)(?:,.*?length=(\d+))?.*>$"
)

__all__ = [
    "VCF_CHUNK_ROWS",
    "build_vcf_header",
    "vcf_float",
    "write_bins_vcf",
    "write_segments_vcf",
    "write_vcf_streaming",
]


def _validate_threads(threads: int) -> None:
    if isinstance(threads, bool) or threads < 1:
        raise ValueError("threads must be a positive integer")


def _fmt_vcf_float_expr(col_name: str) -> pl.Expr:
    col = pl.col(col_name)
    return (
        pl.when(col.is_finite())
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
                parsed = int(len_token)
                max_len = max(max_len, parsed)
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


def vcf_float(value: float) -> float | None:
    """Return finite float or None for missing VCF values."""
    return float(value) if np.isfinite(value) else None


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
                    '##INFO=<ID=FRAG_SHORT_R,Number=1,Type=Float,Description="Fraction of short cell-free DNA fragments <= 150 bp">',
                    '##INFO=<ID=FRAG_LLR,Number=1,Type=Float,Description="Fragment length log-likelihood ratio for fetal vs maternal origin">',
                    (
                        "##INFO=<ID=ORIGIN,Number=1,Type=String,"
                        'Description="Fragmentomic origin call: FETAL_CONFIRMED, MATERNAL_CONFIRMED, or INDETERMINATE">'
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
    _validate_threads(threads)
    if isinstance(max_contig_len, bool) or max_contig_len < 0:
        raise ValueError("max_contig_len must be a non-negative integer")

    def _chunks() -> Iterator[bytes]:
        if header_lines:
            header_text = "\n".join(line.rstrip("\r\n") for line in header_lines) + "\n"
            yield header_text.encode("utf-8")
        yield from body_chunks

    write_bgzf_indexed(
        str(gz_path),
        "vcf",
        list(contig_names),
        int(max_contig_len),
        _chunks(),
        threads,
    )
    return gz_path


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
    record_max = max(
        (int(np.max(res.ends)) for res in results if res.ends.size > 0),
        default=0,
    )
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
            cols: dict[str, object] = {
                "start0": chunk.starts,
                "end0": chunk.ends,
                "ratio": chunk.ratios,
                "zscore": chunk.zscores,
                "weight": chunk.weights,
                "ref_size": chunk.reference_sizes,
                "masked": chunk.masked,
            }
            if isinstance(chunk.branch, str):
                info_expr = pl.concat_str(
                    [
                        pl.lit("END="),
                        pl.col("end0").cast(pl.String),
                        pl.lit(f";SVTYPE=CNV;BRANCH={chunk.branch}"),
                    ]
                )
            else:
                cols["branch"] = pl.Series("branch", chunk.branch, dtype=pl.String)
                info_expr = pl.concat_str(
                    [
                        pl.lit("END="),
                        pl.col("end0").cast(pl.String),
                        pl.lit(";SVTYPE=CNV;BRANCH="),
                        pl.col("branch"),
                    ]
                )
            df = pl.DataFrame(cols).select(
                pl.lit(chunk.contig, dtype=pl.String).alias("chrom"),
                (pl.col("start0") + 1).alias("pos"),
                pl.concat_str(
                    [
                        pl.lit(f"{chunk.contig}:"),
                        pl.col("start0").cast(pl.String),
                        pl.lit("-"),
                        pl.col("end0").cast(pl.String),
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
                        pl.col("ref_size").cast(pl.String),
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
    results: Sequence[SegmentResult],
    sample_name: str,
    *,
    calls_only: bool = False,
    is_prenatal: bool = False,
    contig_lengths: Mapping[str, int] | None = None,
    threads: int = 1,
) -> Path:
    """Write segment-level calls and metrics to a compressed, indexed VCFv4.2."""
    if not sample_name:
        raise ValueError("sample_name must be non-empty")
    filtered = [s for s in results if not calls_only or s.call is not None]
    contig_names, max_len = _merge_vcf_contig_bounds(
        header,
        (item.contig for item in filtered),
        max((int(item.end) for item in filtered), default=0),
        contig_lengths,
    )
    contig_rank = {c: idx for idx, c in enumerate(contig_names)}
    items = sorted(filtered, key=lambda s: (contig_rank[s.contig], s.start, s.end))

    def _body_chunks() -> Iterator[bytes]:
        for offset in range(0, len(items), VCF_CHUNK_ROWS):
            batch = items[offset : offset + VCF_CHUNK_ROWS]
            cols: dict[str, pl.Series] = {
                "chrom": pl.Series("chrom", [s.contig for s in batch], dtype=pl.String),
                "pos": pl.Series(
                    "pos", [int(s.start) + 1 for s in batch], dtype=pl.Int64
                ),
                "end": pl.Series("end", [int(s.end) for s in batch], dtype=pl.Int64),
                "call": pl.Series("call", [s.call for s in batch], dtype=pl.String),
                "ratio": pl.Series(
                    "ratio", [float(s.ratio) for s in batch], dtype=pl.Float64
                ),
                "zscore": pl.Series(
                    "zscore", [float(s.zscore) for s in batch], dtype=pl.Float64
                ),
                "weight": pl.Series(
                    "weight", [float(s.weight) for s in batch], dtype=pl.Float64
                ),
                "bin_count": pl.Series(
                    "bin_count", [int(s.bin_count) for s in batch], dtype=pl.Int64
                ),
            }
            if is_prenatal:
                cols["discordant"] = pl.Series(
                    "discordant", [bool(s.discordant) for s in batch], dtype=pl.Boolean
                )
                cols["disc_reason"] = pl.Series(
                    "disc_reason",
                    [s.discordance_reason for s in batch],
                    dtype=pl.String,
                )
                cols["frag_short_ratio"] = pl.Series(
                    "frag_short_ratio",
                    [
                        vcf_float(s.frag_short_ratio)
                        if s.frag_short_ratio is not None
                        else None
                        for s in batch
                    ],
                    dtype=pl.Float64,
                )
                cols["frag_llr_size"] = pl.Series(
                    "frag_llr_size",
                    [
                        vcf_float(s.frag_llr_size)
                        if s.frag_llr_size is not None
                        else None
                        for s in batch
                    ],
                    dtype=pl.Float64,
                )
                cols["origin_call"] = pl.Series(
                    "origin_call", [s.origin_call for s in batch], dtype=pl.String
                )
                cols["expected_ratio"] = pl.Series(
                    "expected_ratio",
                    [
                        vcf_float(s.expected_ratio)
                        if s.expected_ratio is not None
                        else None
                        for s in batch
                    ],
                    dtype=pl.Float64,
                )
                cols["implied_ff"] = pl.Series(
                    "implied_ff",
                    [
                        vcf_float(s.implied_ff) if s.implied_ff is not None else None
                        for s in batch
                    ],
                    dtype=pl.Float64,
                )

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
                            & pl.col("disc_reason").is_not_null()
                            & (pl.col("disc_reason") != "")
                        )
                        .then(
                            pl.concat_str(
                                [pl.lit(";DISC_REASON="), pl.col("disc_reason")]
                            )
                        )
                        .otherwise(pl.lit("")),
                        pl.when(pl.col("frag_short_ratio").is_finite())
                        .then(
                            pl.concat_str(
                                [
                                    pl.lit(";FRAG_SHORT_R="),
                                    pl.col("frag_short_ratio").round(6).cast(pl.String),
                                ]
                            )
                        )
                        .otherwise(pl.lit("")),
                        pl.when(pl.col("frag_llr_size").is_finite())
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
                            pl.col("origin_call").is_not_null()
                            & (pl.col("origin_call") != "")
                        )
                        .then(
                            pl.concat_str([pl.lit(";ORIGIN="), pl.col("origin_call")])
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

            df = pl.DataFrame(cols).select(
                pl.col("chrom"),
                pl.col("pos"),
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
