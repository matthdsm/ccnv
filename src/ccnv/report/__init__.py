"""Downstream deliverables and visualization for ccnv."""

import logging
from collections.abc import Sequence
from pathlib import Path

from ccnv.models import (
    AnalysisMode,
    BinResult,
    Cytoband,
    FetalFractionResult,
    PredictionOutputs,
    SegmentResult,
    natural_sort_key,
)
from ccnv.report.bed import (
    BINS_BED_HEADER,
    CALLS_BED_HEADER,
    PRENATAL_CALLS_BED_HEADER,
    PRENATAL_SEGMENTS_BED_HEADER,
    SEGMENTS_BED_HEADER,
    bin_bed_rows,
    calls_bed_rows,
    segment_bed_rows,
    sort_bin_result,
    sort_bin_results,
    sort_segments,
    write_compressed_bed,
)
from ccnv.report.html import (
    PLOTLY_CDN_URL,
    _load_cytobands_path,
    encode_b64,
    render_genome_plot,
    write_genome_plot,
)
from ccnv.report.vcf import (
    build_vcf_header,
    write_bins_vcf,
    write_segments_vcf,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BINS_BED_HEADER",
    "CALLS_BED_HEADER",
    "PLOTLY_CDN_URL",
    "PRENATAL_CALLS_BED_HEADER",
    "PRENATAL_SEGMENTS_BED_HEADER",
    "SEGMENTS_BED_HEADER",
    "PredictionOutputs",
    "build_vcf_header",
    "encode_b64",
    "natural_sort_key",
    "read_cytobands",
    "render_genome_plot",
    "sort_bin_result",
    "sort_bin_results",
    "sort_segments",
    "write_bins_vcf",
    "write_compressed_bed",
    "write_genome_plot",
    "write_outputs",
    "write_segments_vcf",
]


def read_cytobands(path: Path, contig_lengths: dict[str, int]) -> tuple[Cytoband, ...]:
    """Transitional compatibility wrapper reading cytobands."""
    return _load_cytobands_path(path, contig_lengths)


def write_outputs(
    prefix: Path,
    bins: tuple[BinResult, ...],
    segments: tuple[SegmentResult, ...],
    contig_lengths: dict[str, int],
    plot: bool,
    vcf: bool = True,
    *,
    cytobands: Path | Sequence[Cytoband] | None = None,
    z_threshold: float = 3.0,
    ratio_threshold: float | None = None,
    prenatal_result: FetalFractionResult | None = None,
    mode: AnalysisMode = "germline",
    qc_json_path: Path | None = None,
) -> PredictionOutputs:
    """Write compressed BED evidence, VCFs, and an interactive HTML plot."""
    prefix.parent.mkdir(parents=True, exist_ok=True)
    bins = sort_bin_results(bins)
    segments = sort_segments(segments)
    bins_bed = Path(f"{prefix}_bins.bed.gz")
    segments_bed = Path(f"{prefix}_segments.bed.gz")
    calls_bed = Path(f"{prefix}_calls.bed.gz")
    plot_path = Path(f"{prefix}_plot.html") if plot else None
    sample_name = prefix.name if prefix.name else "SAMPLE"
    calls = [item for item in segments if item.call is not None]
    total_bins = sum(b.starts.size for b in bins)
    is_prenatal = mode == "prenatal"
    seg_header = PRENATAL_SEGMENTS_BED_HEADER if is_prenatal else SEGMENTS_BED_HEADER
    call_header = PRENATAL_CALLS_BED_HEADER if is_prenatal else CALLS_BED_HEADER

    logger.debug(
        "writing prediction outputs to prefix %s "
        "(%d bins, %d segments, %d calls, mode=%s)",
        prefix,
        total_bins,
        len(segments),
        len(calls),
        mode,
    )
    write_compressed_bed(bins_bed, BINS_BED_HEADER, bin_bed_rows(bins))
    write_compressed_bed(
        segments_bed, seg_header, segment_bed_rows(segments, is_prenatal=is_prenatal)
    )
    write_compressed_bed(
        calls_bed,
        call_header,
        calls_bed_rows(calls, is_prenatal=is_prenatal),
    )
    bins_vcf: Path | None = None
    segments_vcf: Path | None = None
    calls_vcf: Path | None = None
    if vcf:
        bins_vcf = Path(f"{prefix}_bins.vcf.gz")
        segments_vcf = Path(f"{prefix}_segments.vcf.gz")
        calls_vcf = Path(f"{prefix}_calls.vcf.gz")
        write_bins_vcf(
            bins_vcf,
            build_vcf_header(
                contig_lengths,
                sample_name,
                "bins",
                extra_contigs=(c for r in bins for c in r.contigs),
            ),
            bins,
            sample_name,
        )
        seg_header_vcf = build_vcf_header(
            contig_lengths,
            sample_name,
            "segments",
            extra_contigs=(s.contig for s in segments),
            prenatal_result=prenatal_result if is_prenatal else None,
        )
        write_segments_vcf(
            segments_vcf,
            seg_header_vcf,
            segments,
            sample_name,
            is_prenatal=is_prenatal,
        )
        write_segments_vcf(
            calls_vcf,
            seg_header_vcf,
            segments,
            sample_name,
            calls_only=True,
            is_prenatal=is_prenatal,
        )
    logger.debug(
        "wrote %s, %s, %s",
        bins_bed.name,
        segments_bed.name,
        calls_bed.name,
    )
    if plot_path is not None:
        render_genome_plot(
            plot_path,
            bins,
            segments,
            contig_lengths,
            cytobands=cytobands,
            z_threshold=z_threshold,
            ratio_threshold=ratio_threshold,
            sample_name=sample_name,
            prenatal_result=prenatal_result if is_prenatal else None,
        )
    return PredictionOutputs(
        bins_bed=bins_bed,
        segments_bed=segments_bed,
        calls_bed=calls_bed,
        plot_html=plot_path,
        bins_vcf=bins_vcf,
        segments_vcf=segments_vcf,
        calls_vcf=calls_vcf,
        bins=bins,
        segments=segments,
        qc_json_path=qc_json_path,
        prenatal_result=prenatal_result,
    )
