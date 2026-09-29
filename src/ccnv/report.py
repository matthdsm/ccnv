import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pysam

from ccnv.results import BinResult, PredictionOutputs, SegmentResult

logger = logging.getLogger(__name__)

BINS_BED_HEADER = (
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
SEGMENTS_BED_HEADER = (
    "#contig",
    "start",
    "end",
    "ratio",
    "zscore",
    "weight",
    "bin_count",
    "call",
)
CALLS_BED_HEADER = ("#contig", "start", "end", "ratio", "zscore", "type")


def natural_sort_key(name: str) -> tuple[tuple[int, int | str], ...]:
    """Return key for natural sort order treating embedded numbers as integers."""

    return tuple(
        (0, int(chunk)) if chunk.isdigit() else (1, chunk.casefold())
        for chunk in re.split(r"(\d+)", name)
        if chunk
    )


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


def write_outputs(
    prefix: Path,
    bins: tuple[BinResult, ...],
    segments: tuple[SegmentResult, ...],
    contig_lengths: dict[str, int],
    plot: bool,
    vcf: bool = True,
) -> PredictionOutputs:
    """Write compressed BED evidence, matching Parquet, VCFs, and an SVG plot."""

    prefix.parent.mkdir(parents=True, exist_ok=True)
    bins = sort_bin_results(bins)
    segments = sort_segments(segments)
    bins_bed = Path(f"{prefix}_bins.bed.gz")
    segments_bed = Path(f"{prefix}_segments.bed.gz")
    calls_bed = Path(f"{prefix}_calls.bed.gz")
    bins_parquet = Path(f"{prefix}_bins.ccnv")
    segments_parquet = Path(f"{prefix}_segments.ccnv")
    plot_path = Path(f"{prefix}_plot.svg") if plot else None
    sample_name = prefix.name if prefix.name else "SAMPLE"
    calls = [item for item in segments if item.call is not None]
    total_bins = sum(b.starts.size for b in bins)
    logger.debug(
        "writing prediction outputs to prefix %s (%d bins, %d segments, %d calls)",
        prefix,
        total_bins,
        len(segments),
        len(calls),
    )
    _write_compressed_bed(bins_bed, BINS_BED_HEADER, _bin_bed_rows(bins))
    _write_compressed_bed(
        segments_bed, SEGMENTS_BED_HEADER, _segment_bed_rows(segments)
    )
    _write_compressed_bed(
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
            for item in calls
        ),
    )
    bins_vcf: Path | None = None
    segments_vcf: Path | None = None
    calls_vcf: Path | None = None
    if vcf:
        bins_vcf = Path(f"{prefix}_bins.vcf.gz")
        segments_vcf = Path(f"{prefix}_segments.vcf.gz")
        calls_vcf = Path(f"{prefix}_calls.vcf.gz")
        _write_bins_vcf(
            bins_vcf,
            _build_vcf_header(
                contig_lengths,
                sample_name,
                "bins",
                extra_contigs=(c for r in bins for c in r.contigs),
            ),
            bins,
            sample_name,
        )
        _write_segments_vcf(
            segments_vcf,
            _build_vcf_header(
                contig_lengths,
                sample_name,
                "segments",
                extra_contigs=(s.contig for s in segments),
            ),
            segments,
            sample_name,
        )
        _write_segments_vcf(
            calls_vcf,
            _build_vcf_header(
                contig_lengths,
                sample_name,
                "segments",
                extra_contigs=(s.contig for s in segments),
            ),
            segments,
            sample_name,
            calls_only=True,
        )
    pq.write_table(_bins_table(bins), bins_parquet, compression="zstd")
    pq.write_table(_segments_table(segments), segments_parquet, compression="zstd")
    logger.debug(
        "wrote %s, %s, %s, %s, %s",
        bins_bed.name,
        segments_bed.name,
        calls_bed.name,
        bins_parquet.name,
        segments_parquet.name,
    )
    if plot_path is not None:
        write_genome_plot(plot_path, bins, segments, contig_lengths)
    return PredictionOutputs(
        bins_bed,
        segments_bed,
        calls_bed,
        plot_path,
        bins_parquet,
        segments_parquet,
        bins_vcf,
        segments_vcf,
        calls_vcf,
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

    total_raw_bins = sum(b.starts.size for b in bins)
    raw_contigs = dict.fromkeys(
        [contig for result in bins for contig in result.contigs]
        + [item.contig for item in segments]
    )
    ordered = tuple(
        sorted(
            (c for c in raw_contigs if c in contig_lengths and contig_lengths[c] > 0),
            key=natural_sort_key,
        )
    )
    logger.debug(
        "generating genome SVG plot %s (%d contigs, %d raw bins)",
        path,
        len(ordered),
        total_raw_bins,
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
    reduction = (
        100.0 * (1.0 - len(seen_points) / total_raw_bins) if total_raw_bins > 0 else 0.0
    )
    logger.debug(
        "wrote genome SVG plot %s (%.1f KB, %d points, %.1f%% sub-pixel reduction)",
        path,
        path.stat().st_size / 1024,
        len(seen_points),
        reduction,
    )


def _format_bed_float(value: float) -> str:
    if not np.isfinite(value):
        return "NaN"
    return f"{value:.8g}"


def _vcf_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _write_compressed_bed(
    gz_path: Path, header: tuple[str, ...], rows: Iterable[tuple[object, ...]]
) -> Path:
    with pysam.BGZFile(str(gz_path), "wb") as bgz:
        bgz.write(("\t".join(header) + "\n").encode("utf-8"))
        for row in rows:
            bgz.write(("\t".join(str(value) for value in row) + "\n").encode("utf-8"))
    pysam.tabix_index(str(gz_path), preset="bed", force=True)
    return gz_path


def _build_vcf_header(
    contig_lengths: dict[str, int],
    sample_name: str,
    record_type: Literal["bins", "segments"],
    extra_contigs: Iterable[str] = (),
) -> pysam.VariantHeader:
    header = pysam.VariantHeader()
    header.add_line("##source=ccnv_predict")
    all_contigs = set(contig_lengths.keys()) | set(extra_contigs)
    for contig in sorted(all_contigs, key=natural_sort_key):
        length = contig_lengths.get(contig)
        if length is not None:
            header.contigs.add(contig, length=length)
        else:
            header.contigs.add(contig)

    if record_type == "bins":
        header.add_line('##ALT=<ID=CNV,Description="Copy number variation">')
        header.add_line(
            "##INFO=<ID=END,Number=1,Type=Integer,"
            'Description="End coordinate of structural variant">'
        )
        header.add_line(
            "##INFO=<ID=SVTYPE,Number=1,Type=String,"
            'Description="Type of structural variant">'
        )
        header.add_line(
            '##INFO=<ID=BRANCH,Number=1,Type=String,Description="Model branch">'
        )
        header.add_line(
            "##FILTER=<ID=MASKED,"
            'Description="Bin masked due to low coverage or quality filters">'
        )
        header.add_line(
            "##FORMAT=<ID=LOG2R,Number=1,Type=Float,"
            'Description="Log2 copy ratio against within-sample reference">'
        )
        header.add_line(
            "##FORMAT=<ID=ZSCORE,Number=1,Type=Float,"
            'Description="Z-score relative to reference cohort null model">'
        )
        header.add_line(
            "##FORMAT=<ID=WEIGHT,Number=1,Type=Float,"
            'Description="Precision weight for segmentation">'
        )
        header.add_line(
            "##FORMAT=<ID=REF_SIZE,Number=1,Type=Integer,"
            'Description="Number of within-sample reference bins used">'
        )
    else:
        header.add_line('##ALT=<ID=DUP,Description="Duplication / copy number gain">')
        header.add_line('##ALT=<ID=DEL,Description="Deletion / copy number loss">')
        header.add_line('##ALT=<ID=CNV,Description="Copy number variation">')
        header.add_line(
            "##INFO=<ID=END,Number=1,Type=Integer,"
            'Description="End coordinate of structural variant">'
        )
        header.add_line(
            "##INFO=<ID=SVTYPE,Number=1,Type=String,"
            'Description="Type of structural variant">'
        )
        header.add_line(
            "##INFO=<ID=IMPRECISE,Number=0,Type=Flag,"
            'Description="Imprecise structural variation">'
        )
        header.add_line(
            "##FORMAT=<ID=LOG2R,Number=1,Type=Float,"
            'Description="Log2 copy ratio against within-sample reference">'
        )
        header.add_line(
            "##FORMAT=<ID=ZSCORE,Number=1,Type=Float,"
            'Description="Z-score relative to reference cohort null model">'
        )
        header.add_line(
            "##FORMAT=<ID=WEIGHT,Number=1,Type=Float,"
            'Description="Precision weight for segmentation">'
        )
        header.add_line(
            "##FORMAT=<ID=BINS,Number=1,Type=Integer,"
            'Description="Number of bins in segment">'
        )

    header.add_sample(sample_name)
    return header


def _write_bins_vcf(
    gz_path: Path,
    header: pysam.VariantHeader,
    results: tuple[BinResult, ...],
    sample_name: str,
) -> Path:
    with pysam.VariantFile(str(gz_path), "wz", header=header) as vf:
        for result in results:
            branch = result.branch
            for index in range(len(result.contigs)):
                contig = result.contigs[index]
                start = int(result.starts[index])
                end = int(result.ends[index])
                is_masked = bool(result.masked[index])
                filt = "MASKED" if is_masked else "PASS"
                rec = vf.new_record(
                    contig=contig,
                    start=start,
                    stop=end,
                    alleles=("N", "<CNV>"),
                    id=f"{contig}:{start}-{end}",
                    filter=filt,
                )
                rec.info["SVTYPE"] = "CNV"
                rec.info["BRANCH"] = branch
                sample = rec.samples[sample_name]
                sample["LOG2R"] = _vcf_float(float(result.ratios[index]))
                sample["ZSCORE"] = _vcf_float(float(result.zscores[index]))
                sample["WEIGHT"] = _vcf_float(float(result.weights[index]))
                sample["REF_SIZE"] = int(result.reference_sizes[index])
                vf.write(rec)
    pysam.tabix_index(str(gz_path), preset="vcf", force=True)
    return gz_path


def _write_segments_vcf(
    gz_path: Path,
    header: pysam.VariantHeader,
    results: tuple[SegmentResult, ...],
    sample_name: str,
    *,
    calls_only: bool = False,
) -> Path:
    with pysam.VariantFile(str(gz_path), "wz", header=header) as vf:
        for item in results:
            if calls_only and item.call is None:
                continue
            contig = item.contig
            start = item.start
            end = item.end
            if item.call == "gain":
                alt = "<DUP>"
                svtype = "DUP"
                filt = "PASS"
            elif item.call == "loss":
                alt = "<DEL>"
                svtype = "DEL"
                filt = "PASS"
            else:
                alt = "<CNV>"
                svtype = "CNV"
                filt = None
            rec = vf.new_record(
                contig=contig,
                start=start,
                stop=end,
                alleles=("N", alt),
                id=".",
                filter=filt,
            )
            rec.info["SVTYPE"] = svtype
            rec.info["IMPRECISE"] = True
            sample = rec.samples[sample_name]
            sample["LOG2R"] = _vcf_float(item.ratio)
            sample["ZSCORE"] = _vcf_float(item.zscore)
            sample["WEIGHT"] = _vcf_float(item.weight)
            sample["BINS"] = int(item.bin_count)
            vf.write(rec)
    pysam.tabix_index(str(gz_path), preset="vcf", force=True)
    return gz_path


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
