"""Stream BAM/CRAM alignments into compact Schema v2.0.0 sample sketch Parquet."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
import pyarrow as pa

from ccnv.io.alignments import (
    ScanAlignmentResult,
    read_alignment_header,
    scan_alignment_bins,
    validate_alignment_inputs,
)
from ccnv.io.parquet import write_sample_parquet
from ccnv.models import (
    BinCompatibilityError,
    ContigLookup,
    ExtractConfig,
    ExtractError,
    FragmentomicsMetadata,
    PredictError,
    ReadFilters,
    ReferenceMetadata,
    SampleBuffers,
    SampleMetadata,
    SchemaError,
    parquet_path,
    select_layout,
)
from ccnv.qc.sample import build_sample_qc
from ccnv.stats.fragmentomics import estimate_sample_landmarks

logger = logging.getLogger(__name__)

__all__ = [
    "ExtractError",
    "extract_for_predict",
    "extract_sample_table",
    "run",
    "run_extract",
]


def _scan_to_buffers(
    scan: ScanAlignmentResult,
    selected: Sequence[tuple[str, int]],
    *,
    bin_size: int,
    fmt: str,
    resolved_alignment: Path,
    resolved_fasta: Path | None,
    filters: ReadFilters,
) -> SampleBuffers:
    paired_end = bool(scan["paired_end"])
    raw_bands = scan["frag_bands"]
    bands_arr = (
        np.asarray(raw_bands, dtype=np.uint32)
        if paired_end and raw_bands is not None
        else None
    )
    qc = build_sample_qc(
        {str(k): int(v) for k, v in scan["qc_counts"].items() if isinstance(v, int)}
    )
    hist_auto = np.asarray(scan["hist_autosomal"], dtype=np.int64)
    hist_chry = np.asarray(scan["hist_chry"], dtype=np.int64)
    hist_chrx = np.asarray(scan["hist_chrx"], dtype=np.int64)
    motifs = np.asarray(scan["motif_counts"], dtype=np.int64)
    landmarks = estimate_sample_landmarks(
        hist_autosomal=hist_auto,
        end_motif_counts=motifs,
        paired_end=paired_end,
    )
    meta_frag = FragmentomicsMetadata(
        paired_end=landmarks.paired_end,
        total_paired_fragments=landmarks.total_paired_fragments,
        chromatosome_mode_bp=landmarks.chromatosome_mode_bp,
        core_shoulder_bp=landmarks.core_shoulder_bp,
        sizing_shift_bp=landmarks.sizing_shift_bp,
        self_referenced_peak_ratio=landmarks.self_referenced_peak_ratio,
        dinucleosome_fraction=landmarks.dinucleosome_fraction,
        high_mw_fraction=landmarks.high_mw_fraction,
        helical_10bp_amplitude=landmarks.helical_10bp_amplitude,
        helical_10bp_phase=landmarks.helical_10bp_phase,
        motif_diversity_score=landmarks.motif_diversity_score,
        hist_autosomal=hist_auto.tolist(),
        hist_chry=hist_chry.tolist(),
        hist_chrx=hist_chrx.tolist(),
        end_motif_counts=motifs.tolist(),
    )
    contig_order = [name for name, _ in selected]
    contig_lengths = dict(selected)
    source_id: dict[str, object] = {
        "path": str(resolved_alignment.resolve()),
        "format": fmt,
    }
    ref_id: dict[str, object] = {
        "n_contigs": len(contig_order),
        "contig_order": contig_order,
        **(
            {"fasta": str(resolved_fasta.resolve())}
            if resolved_fasta is not None
            else {}
        ),
    }
    sample_meta = SampleMetadata(
        bin_size=bin_size,
        contig_order=tuple(contig_order),
        contig_lengths=contig_lengths,
        source_identity=source_id,
        reference_identity=ref_id,
        qc=dict(qc),
        filters=filters.to_dict(),
        fragmentomics=meta_frag,
    )
    return SampleBuffers(
        meta=sample_meta,
        count=np.asarray(scan["count"], dtype=np.uint32),
        paired_count=np.asarray(scan["paired_count"], dtype=np.uint32),
        frag_len_sum=np.asarray(scan["frag_len_sum"], dtype=np.uint64),
        short_frag_count=np.asarray(scan["short_frag_count"], dtype=np.uint32),
        endpoint_count=np.asarray(scan["endpoint_count"], dtype=np.uint32),
        frag_gc_sum=np.asarray(scan["frag_gc_sum"], dtype=np.float32),
        frag_bands=bands_arr,
    )


def extract_for_predict(
    alignment: Path,
    ref_meta: ReferenceMetadata,
    *,
    threads: int = 1,
    fasta: Path | None = None,
    collect_bands: bool = False,
) -> SampleBuffers:
    """Stream BAM/CRAM directly into reference-aligned SampleBuffers via ccnv._core."""
    try:
        fmt, resolved_alignment, resolved_fasta = validate_alignment_inputs(
            alignment, fasta
        )
        header = read_alignment_header(resolved_alignment, resolved_fasta)
    except (ExtractError, ValueError, OSError) as err:
        raise PredictError(f"failed to read sample {alignment}: {err}") from err

    if not header or any(length <= 0 for _, length in header):
        raise PredictError(f"{resolved_alignment} header declares invalid contigs")
    if len(dict(header)) != len(header):
        raise PredictError(f"{resolved_alignment} header declares duplicate contigs")

    lookup = ContigLookup.from_names(ref_meta.contig_lengths.keys())
    seen_ref_contigs: set[str] = set()
    selected: list[tuple[str, int]] = []
    for header_name, actual_len in header:
        ref_name = lookup.resolve(header_name)
        if ref_name is None:
            continue
        if ref_name in seen_ref_contigs:
            raise PredictError(
                f"{resolved_alignment} header maps multiple contigs to {ref_name!r}"
            )
        seen_ref_contigs.add(ref_name)
        expected_len = ref_meta.contig_lengths[ref_name]
        if actual_len != expected_len:
            raise BinCompatibilityError(f"contig length mismatch for {ref_name!r}")
        selected.append((ref_name, expected_len))
    if not selected:
        raise PredictError(
            f"{resolved_alignment} declares none of the reference contigs"
        )

    filters = ReadFilters.from_dict(ref_meta.filters or {}, strict=False)
    try:
        scan = scan_alignment_bins(
            resolved_alignment,
            selected,
            ref_meta.bin_size,
            filters,
            threads=threads,
            fasta=resolved_fasta,
            collect_bands=collect_bands,
        )
    except (ExtractError, ValueError, OSError) as err:
        raise PredictError(f"failed to stream alignment {alignment}: {err}") from err

    return _scan_to_buffers(
        scan,
        selected,
        bin_size=ref_meta.bin_size,
        fmt=fmt,
        resolved_alignment=resolved_alignment,
        resolved_fasta=resolved_fasta,
        filters=filters,
    )


def extract_sample_table(config: ExtractConfig) -> pa.Table:
    """Extract a Schema v2.0.0 sample sketch table in memory from one BAM or CRAM."""
    fmt, resolved_alignment, resolved_fasta = validate_alignment_inputs(
        config.alignment, config.fasta
    )
    logger.debug("opening alignment %s as %s", resolved_alignment, fmt)

    header = read_alignment_header(resolved_alignment, resolved_fasta)
    if not header or any(length <= 0 for _, length in header):
        raise ExtractError(f"{resolved_alignment} header declares invalid contigs")

    selected = select_layout(header, all_contigs=config.all_contigs)
    if not selected:
        raise ExtractError(
            f"{resolved_alignment} header declares no canonical chromosomes; "
            "pass --all-contigs to bin every contig"
        )
    if len(dict(selected)) != len(selected):
        raise ExtractError(f"{resolved_alignment} header declares duplicate contigs")

    scan = scan_alignment_bins(
        resolved_alignment,
        selected,
        config.bin_size,
        config.filters,
        threads=config.threads,
        fasta=resolved_fasta,
        collect_bands=True,
    )
    buffers = _scan_to_buffers(
        scan,
        selected,
        bin_size=config.bin_size,
        fmt=fmt,
        resolved_alignment=resolved_alignment,
        resolved_fasta=resolved_fasta,
        filters=config.filters,
    )
    qc = buffers.meta.qc
    logger.debug(
        "extract QC summary: %s reads seen, %s kept, %s mate_dedup_skipped, %s filtered",
        qc.get("reads_seen"),
        qc.get("reads_kept"),
        qc.get("mate_dedup_skipped"),
        qc.get("reads_filtered"),
    )
    return buffers.to_table()


def run(config: ExtractConfig) -> None:
    """Extract binned counts from one BAM/CRAM into canonical sample Parquet."""
    table = extract_sample_table(config)
    output = parquet_path(config.prefix)
    logger.debug("writing sample parquet %s (%d bins)", output, table.num_rows)
    try:
        write_sample_parquet(table, output)
        size_kb = output.stat().st_size / 1024
    except (SchemaError, OSError, pa.ArrowException) as err:
        raise ExtractError(f"failed to write sample parquet {output}: {err}") from err
    logger.debug(
        "wrote sample parquet %s (%.1f KB)",
        output,
        size_kb,
    )


run_extract: Final = run
