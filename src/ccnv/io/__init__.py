"""External format ingestion and pipeline table storage for ccnv."""

from ccnv.io.alignments import (
    ScanAlignmentResult,
    alignment_format,
    read_alignment_header,
    scan_alignment_bins,
    validate_alignment_inputs,
)
from ccnv.io.parquet import (
    is_sample_parquet,
    read_reference_parquet,
    read_sample_buffers,
    read_sample_parquet,
    write_reference_parquet,
    write_sample_parquet,
)
from ccnv.io.samplesheet import (
    load_samplesheet,
)
from ccnv.io.tracks import (
    load_cytobands_track,
    load_gc_track,
    load_mappability_track,
    read_fasta_contigs,
)

__all__ = [
    "ScanAlignmentResult",
    "alignment_format",
    "is_sample_parquet",
    "load_cytobands_track",
    "load_gc_track",
    "load_mappability_track",
    "load_samplesheet",
    "read_alignment_header",
    "read_fasta_contigs",
    "read_reference_parquet",
    "read_sample_buffers",
    "read_sample_parquet",
    "scan_alignment_bins",
    "validate_alignment_inputs",
    "write_reference_parquet",
    "write_sample_parquet",
]
