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
    bins_parquet = Path(f"{prefix}_bins.ccnv")
    segments_parquet = Path(f"{prefix}_segments.ccnv")
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
    width, height = 1200, 380
    left, right, top, bottom = 55, 40, 20, 75
    plot_width, plot_height = width - left - right, height - top - bottom

    def x(contig: str, position: float) -> float:
        return left + plot_width * (offsets[contig] + position) / max(total, 1)

    def y(value: float) -> float:
        clipped = min(max(value, -2.0), 2.0)
        return top + plot_height * (2.0 - clipped) / 4.0

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        "font-family=\"system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, "
        'Arial, sans-serif">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    for i, contig in enumerate(ordered):
        x_start = x(contig, 0)
        x_end = x(contig, contig_lengths[contig])
        w = x_end - x_start
        if i % 2 == 1:
            lines.append(
                f'<rect x="{x_start:g}" y="{top}" width="{w:g}" '
                f'height="{plot_height}" fill="#f5f5f5"/>'
            )
        if i > 0:
            lines.append(
                f'<line x1="{x_start:g}" y1="{top}" x2="{x_start:g}" '
                f'y2="{top + plot_height}" stroke="#e0e0e0" stroke-width="1"/>'
            )

    for tick in (-2, -1, 0, 1, 2):
        y_pos = y(float(tick))
        if tick == 0:
            lines.append(
                f'<line x1="{left}" y1="{y_pos:g}" x2="{width - right}" '
                f'y2="{y_pos:g}" stroke="#888" stroke-width="1"/>'
            )
        elif tick in (-1, 1):
            lines.append(
                f'<line x1="{left}" y1="{y_pos:g}" x2="{width - right}" '
                f'y2="{y_pos:g}" stroke="#e5e5e5" stroke-width="1" '
                'stroke-dasharray="4,4"/>'
            )
        else:
            lines.append(
                f'<line x1="{left}" y1="{y_pos:g}" x2="{width - right}" '
                f'y2="{y_pos:g}" stroke="#ccc" stroke-width="1"/>'
            )
        lines.append(
            f'<line x1="{left - 4}" y1="{y_pos:g}" x2="{left}" '
            f'y2="{y_pos:g}" stroke="#333" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{left - 7}" y="{y_pos:g}" text-anchor="end" '
            f'dominant-baseline="central" font-size="10" fill="#555">{tick}</text>'
        )

    seen_points: set[tuple[float, float]] = set()
    bin_path_cmds: list[str] = []
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
            cx = round(x(contig, (start + end) / 2) * 2) / 2
            cy = round(y(float(ratio)) * 2) / 2
            point = (cx, cy)
            if point in seen_points:
                continue
            seen_points.add(point)
            bin_path_cmds.append(f"M{cx:g},{cy:g}h.1")

    if bin_path_cmds:
        lines.append(
            f'<path d="{"".join(bin_path_cmds)}" fill="none" stroke="#4776b4" '
            'stroke-width="2.4" stroke-linecap="round"/>'
        )

    segment_colors = {"gain": "#c43c39", "loss": "#3c8c5a"}
    segment_paths: dict[str, list[str]] = {"gain": [], "loss": [], "neutral": []}
    for item in segments:
        cat = item.call if item.call in segment_colors else "neutral"
        x1 = x(item.contig, item.start)
        x2 = x(item.contig, item.end)
        y_val = y(item.ratio)
        segment_paths[cat].append(f"M{x1:g},{y_val:g}H{x2:g}")

    for cat, cmds in segment_paths.items():
        if cmds:
            color = segment_colors.get(cat, "#222")
            lines.append(
                f'<path d="{"".join(cmds)}" stroke="{color}" stroke-width="2" '
                'fill="none"/>'
            )

    lines.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" '
        'stroke="#333" stroke-width="1"/>'
    )
    lines.append(
        f'<line x1="{left}" y1="{top + plot_height}" x2="{width - right}" '
        f'y2="{top + plot_height}" stroke="#333" stroke-width="1"/>'
    )

    y_mid = top + plot_height / 2
    lines.append(
        f'<text transform="rotate(-90)" x="{-y_mid:g}" y="16" '
        'text-anchor="middle" font-size="11" font-weight="600" fill="#333">'
        "Log2 Ratio</text>"
    )

    label_y = top + plot_height + 8
    for contig in ordered:
        x_start = x(contig, 0)
        x_end = x(contig, contig_lengths[contig])
        if x_end - x_start < 10:
            continue
        cx = (x_start + x_end) / 2
        lines.append(
            f'<text transform="translate({cx:g},{label_y:g}) rotate(45)" '
            f'text-anchor="start" font-size="9" fill="#555">'
            f"{_xml_escape(contig)}</text>"
        )

    x_mid = left + plot_width / 2
    lines.append(
        f'<text x="{x_mid:g}" y="{height - 10}" text-anchor="middle" '
        'font-size="11" font-weight="600" fill="#333">Chromosome</text>'
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
