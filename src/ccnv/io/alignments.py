"""Alignment file streaming, index verification, and read filtering."""

import logging
from collections.abc import Iterator
from pathlib import Path

import pysam

from ccnv.models import ExtractError, ReadFilters

logger = logging.getLogger(__name__)

__all__ = [
    "alignment_format",
    "ensure_index",
    "open_alignment",
    "read_alignment_header",
    "rejection_reason",
    "stream_contig_reads",
]


def alignment_format(path: Path) -> str:
    """Detect BAM or CRAM format from filename extension."""
    suffix = path.suffix.lower()
    if suffix == ".bam":
        return "bam"
    if suffix == ".cram":
        return "cram"
    raise ExtractError(f"{path} must be BAM or CRAM (found suffix {path.suffix!r})")


def open_alignment(
    path: Path,
    fmt: str | None = None,
    fasta: Path | None = None,
    threads: int = 1,
) -> pysam.AlignmentFile:
    """Open an indexed BAM/CRAM file with configured decode threads."""
    if fmt is None:
        fmt = alignment_format(path)
    decode_threads = max(1, threads)
    if fmt == "cram":
        if fasta is None:
            raise ExtractError(
                f"{path} is CRAM and requires a reference FASTA. "
                "Pass --fasta /path/to/reference.fa (indexed with samtools faidx)."
            )
        if not fasta.is_file():
            raise ExtractError(f"CRAM reference FASTA not found: {fasta}")
        return pysam.AlignmentFile(
            str(path),
            "rc",
            reference_filename=str(fasta),
            threads=decode_threads,
        )
    return pysam.AlignmentFile(str(path), "rb", threads=decode_threads)


def ensure_index(
    path: Path,
    fmt: str | None = None,
    handle: pysam.AlignmentFile | None = None,
    fasta: Path | None = None,
    threads: int = 1,
) -> pysam.AlignmentFile:
    """Verify an alignment index exists, generating one on-the-fly if missing."""
    if fmt is None:
        fmt = alignment_format(path)
    if handle is None:
        handle = open_alignment(path, fmt, fasta, threads=threads)
    if handle.has_index():
        logger.debug("%s already has an index", path)
        return handle
    handle.close()
    kind = "CRAM" if fmt == "cram" else "BAM"

    logger.debug("creating %s index for %s", kind, path)
    try:
        pysam.index(str(path))
    except Exception as error:
        raise ExtractError(
            f"failed to create {kind} index for {path}: {error}"
        ) from error
    new_handle = open_alignment(path, fmt, fasta, threads=threads)
    if new_handle.has_index():
        return new_handle
    new_handle.close()
    raise ExtractError(f"failed to create {kind} index for {path}")


def read_alignment_header(
    handle: pysam.AlignmentFile,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Return contig names and lengths declared in the alignment header."""
    names = tuple(str(r) for r in handle.references)
    lengths = tuple(int(length) for length in handle.lengths)
    return names, lengths


def rejection_reason(read: pysam.AlignedSegment, filters: ReadFilters) -> str | None:
    """Return the first matching QC counter name, or None if kept."""
    if filters.exclude_unmapped and read.is_unmapped:
        return "filter_unmapped"
    start = read.reference_start
    if start is None or int(start) < 0:
        return "filter_no_start"
    if filters.exclude_secondary and read.is_secondary:
        return "filter_secondary"
    if filters.exclude_supplementary and read.is_supplementary:
        return "filter_supplementary"
    if filters.exclude_duplicates and read.is_duplicate:
        return "filter_duplicate"
    if filters.require_proper_pair and read.is_paired and not read.is_proper_pair:
        return "filter_improper_pair"
    if int(read.mapping_quality) < filters.min_mapping_quality:
        return "filter_mapq"
    return None


def stream_contig_reads(
    handle: pysam.AlignmentFile,
    contig: str,
    filters: ReadFilters,
    *,
    contig_length: int | None = None,
    qc: dict[str, int] | None = None,
) -> Iterator[int]:
    """Iterate 0-based reference starts of reads passing filters on a contig."""
    for read in handle.fetch(contig):
        if qc is not None:
            qc["reads_seen"] += 1
        reason = rejection_reason(read, filters)
        if reason is not None:
            if qc is not None:
                qc[reason] += 1
            continue
        start = int(read.reference_start)
        if contig_length is not None and start >= contig_length:
            if qc is not None:
                qc["filter_no_start"] += 1
            continue
        if qc is not None:
            qc["reads_kept"] += 1
        yield start
