"""Stream BAM/CRAM alignments into compact Schema v2.0.0 sample sketch Parquet."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict
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
    AlignmentError,
    BinCompatibilityError,
    ContigLookup,
    ExtractConfig,
    ExtractError,
    PredictError,
    ReadFilters,
    ReferenceMetadata,
    SampleBuffers,
    SampleMetadata,
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
    meta_frag = estimate_sample_landmarks(
        hist_autosomal=np.asarray(scan["hist_autosomal"], dtype=np.int64),
        hist_chry=np.asarray(scan["hist_chry"], dtype=np.int64),
        hist_chrx=np.asarray(scan["hist_chrx"], dtype=np.int64),
        end_motif_counts=np.asarray(scan["motif_counts"], dtype=np.int64),
        paired_end=paired_end,
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
        filters=dict(asdict(filters)),
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


def _filters_from_ref_meta(ref_meta: ReferenceMetadata) -> ReadFilters:
    defaults = ReadFilters()
    raw = ref_meta.filters
    if not isinstance(raw, Mapping):
        return defaults

    def _bool(key: str, default: bool) -> bool:
        v = raw.get(key)
        return v if isinstance(v, bool) else default

    mq = raw.get("min_mapping_quality")
    min_mq = (
        int(mq)
        if isinstance(mq, int) and not isinstance(mq, bool) and mq >= 0
        else defaults.min_mapping_quality
    )
    return ReadFilters(
        min_mapping_quality=min_mq,
        exclude_duplicates=_bool("exclude_duplicates", defaults.exclude_duplicates),
        exclude_secondary=_bool("exclude_secondary", defaults.exclude_secondary),
        exclude_supplementary=_bool(
            "exclude_supplementary", defaults.exclude_supplementary
        ),
        exclude_unmapped=_bool("exclude_unmapped", defaults.exclude_unmapped),
        require_proper_pair=_bool("require_proper_pair", defaults.require_proper_pair),
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
    except (AlignmentError, ExtractError, ValueError, OSError) as err:
        raise PredictError(f"failed to read sample {alignment}: {err}") from err

    if not header or any(length <= 0 for _, length in header):
        raise PredictError(f"{resolved_alignment} header declares invalid contigs")
    header_map = dict(header)
    if len(header_map) != len(header):
        raise PredictError(f"{resolved_alignment} header declares duplicate contigs")

    lookup = ContigLookup.from_names(header_map.keys())
    selected: list[tuple[str, int]] = []
    for name, expected_len in ref_meta.contig_lengths.items():
        resolved = lookup.resolve(name)
        if resolved is None:
            continue
        actual_len = header_map[resolved]
        if actual_len != expected_len:
            raise BinCompatibilityError(f"contig length mismatch for {name!r}")
        selected.append((name, expected_len))
    if not selected:
        raise PredictError(
            f"{resolved_alignment} declares none of the reference contigs"
        )

    filters = _filters_from_ref_meta(ref_meta)
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
    except (AlignmentError, ExtractError, ValueError, OSError) as err:
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
    write_sample_parquet(table, output)
    logger.debug(
        "wrote sample parquet %s (%.1f KB)",
        output,
        output.stat().st_size / 1024,
    )


run_extract: Final = run
