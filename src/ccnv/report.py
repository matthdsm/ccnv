"""BED, Parquet, and SVG writers for prediction artifacts."""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ccnv.results import BinResult, PredictionOutputs, SegmentResult

BINS_BED_HEADER = (
    "contig",
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
SEGMENTS_BED_HEADER = (
    "contig",
    "start",
    "end",
    "ratio",
    "zscore",
    "weight",
    "bin_count",
    "call",
)
CALLS_BED_HEADER = ("contig", "start", "end", "ratio", "zscore", "type")


def write_outputs(
    prefix: Path,
    bins: tuple[BinResult, ...],
    segments: tuple[SegmentResult, ...],
    contig_lengths: dict[str, int],
    plot: bool,
) -> PredictionOutputs:
    """Write 0-based half-open BED evidence, matching Parquet, and an SVG plot."""

    prefix.parent.mkdir(parents=True, exist_ok=True)
    bins_bed = Path(f"{prefix}_bins.bed")
    segments_bed = Path(f"{prefix}_segments.bed")
    calls_bed = Path(f"{prefix}_calls.bed")
    bins_parquet = Path(f"{prefix}_bins.parquet")
    segments_parquet = Path(f"{prefix}_segments.parquet")
    plot_path = Path(f"{prefix}_plot.svg") if plot else None
    _write_bed(bins_bed, BINS_BED_HEADER, _bin_bed_rows(bins))
    _write_bed(segments_bed, SEGMENTS_BED_HEADER, _segment_bed_rows(segments))
    _write_bed(
        calls_bed,
        CALLS_BED_HEADER,
        (
            (
                item.contig,
                item.start,
                item.end,
                _format_bed_float(item.ratio),
                _format_bed_float(item.zscore),
                item.call,
            )
            for item in segments
            if item.call is not None
        ),
    )
    pq.write_table(_bins_table(bins), bins_parquet, compression="zstd")
    pq.write_table(_segments_table(segments), segments_parquet, compression="zstd")
    if plot_path is not None:
        write_genome_plot(plot_path, bins, segments, contig_lengths)
    return PredictionOutputs(
        bins_bed,
        segments_bed,
        calls_bed,
        plot_path,
        bins_parquet,
        segments_parquet,
        bins,
        segments,
    )


def write_genome_plot(
    path: Path,
    bins: tuple[BinResult, ...],
    segments: tuple[SegmentResult, ...],
    contig_lengths: dict[str, int],
) -> None:
    """Write a dependency-free SVG with true contig-length genome offsets."""

    ordered = tuple(
        dict.fromkeys(contig for result in bins for contig in result.contigs)
    )
    offsets: dict[str, int] = {}
    total = 0
    for contig in ordered:
        offsets[contig] = total
        total += contig_lengths[contig]
    width, height = 1200, 360
    left, right, top, bottom = 55, 20, 20, 45
    plot_width, plot_height = width - left - right, height - top - bottom

    def x(contig: str, position: float) -> float:
        return left + plot_width * (offsets[contig] + position) / max(total, 1)

    def y(value: float) -> float:
        clipped = min(max(value, -2.0), 2.0)
        return top + plot_height * (2.0 - clipped) / 4.0

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{y(0):.2f}" x2="{width - right}" '
        f'y2="{y(0):.2f}" stroke="#999"/>',
    ]
    for result in bins:
        for contig, start, end, ratio, masked in zip(
            result.contigs,
            result.starts,
            result.ends,
            result.ratios,
            result.masked,
            strict=True,
        ):
            if masked or not np.isfinite(ratio):
                continue
            lines.append(
                f'<circle cx="{x(contig, (start + end) / 2):.2f}" '
                f'cy="{y(float(ratio)):.2f}" r="1.2" fill="#4776b4"/>'
            )
    for item in segments:
        color = (
            "#c43c39"
            if item.call == "gain"
            else ("#3c8c5a" if item.call == "loss" else "#222")
        )
        lines.append(
            f'<line x1="{x(item.contig, item.start):.2f}" '
            f'y1="{y(item.ratio):.2f}" x2="{x(item.contig, item.end):.2f}" '
            f'y2="{y(item.ratio):.2f}" stroke="{color}" stroke-width="2"/>'
        )
    for contig in ordered:
        midpoint = contig_lengths[contig] / 2
        lines.append(
            f'<text x="{x(contig, midpoint):.2f}" y="{height - 15}" '
            f'text-anchor="middle" font-size="10">{_xml_escape(contig)}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_bed(
    path: Path, header: tuple[str, ...], rows: Iterable[tuple[object, ...]]
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write("\t".join(str(value) for value in row) + "\n")


def _format_bed_float(value: float) -> str:
    if not np.isfinite(value):
        return "NaN"
    return f"{value:.8g}"


def _bin_bed_rows(results: tuple[BinResult, ...]) -> tuple[tuple[object, ...], ...]:
    rows: list[tuple[object, ...]] = []
    for result in results:
        for index in range(len(result.contigs)):
            contig = result.contigs[index]
            start = int(result.starts[index])
            end = int(result.ends[index])
            rows.append(
                (
                    contig,
                    start,
                    end,
                    f"{contig}:{start}-{end}",
                    _format_bed_float(float(result.ratios[index])),
                    _format_bed_float(float(result.zscores[index])),
                    _format_bed_float(float(result.weights[index])),
                    int(result.reference_sizes[index]),
                    "true" if bool(result.masked[index]) else "false",
                    result.branch,
                )
            )
    return tuple(rows)


def _segment_bed_rows(
    results: tuple[SegmentResult, ...],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            item.contig,
            item.start,
            item.end,
            _format_bed_float(item.ratio),
            _format_bed_float(item.zscore),
            _format_bed_float(item.weight),
            item.bin_count,
            item.call or "",
        )
        for item in results
    )


def _bins_table(results: tuple[BinResult, ...]) -> pa.Table:
    rows = [
        (result, index) for result in results for index in range(len(result.contigs))
    ]
    return pa.table(
        {
            "branch": [result.branch for result, _ in rows],
            "contig": [result.contigs[index] for result, index in rows],
            "start": pa.array(
                [int(result.starts[index]) for result, index in rows], type=pa.int64()
            ),
            "end": pa.array(
                [int(result.ends[index]) for result, index in rows], type=pa.int64()
            ),
            "ratio": pa.array(
                [float(result.ratios[index]) for result, index in rows],
                type=pa.float64(),
                from_pandas=True,
            ),
            "zscore": pa.array(
                [float(result.zscores[index]) for result, index in rows],
                type=pa.float64(),
                from_pandas=True,
            ),
            "weight": pa.array(
                [float(result.weights[index]) for result, index in rows],
                type=pa.float64(),
            ),
            "reference_size": pa.array(
                [int(result.reference_sizes[index]) for result, index in rows],
                type=pa.int64(),
            ),
            "masked": [bool(result.masked[index]) for result, index in rows],
        }
    )


def _segments_table(results: tuple[SegmentResult, ...]) -> pa.Table:
    return pa.table(
        {
            "contig": [item.contig for item in results],
            "start": pa.array([item.start for item in results], type=pa.int64()),
            "end": pa.array([item.end for item in results], type=pa.int64()),
            "ratio": pa.array([item.ratio for item in results], type=pa.float64()),
            "zscore": pa.array(
                [item.zscore for item in results],
                type=pa.float64(),
                from_pandas=True,
            ),
            "weight": pa.array([item.weight for item in results], type=pa.float64()),
            "bin_count": pa.array(
                [item.bin_count for item in results], type=pa.int64()
            ),
            "call": pa.array([item.call for item in results], type=pa.string()),
        }
    )


def _xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
