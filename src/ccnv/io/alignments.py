"""Alignment format validation, header inspection, and binned Rust streaming."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ccnv._core import (
    read_alignment_header as _read_alignment_header_core,
)
from ccnv._core import (
    scan_alignment_bins as _scan_alignment_bins_core,
)
from ccnv.models import AlignmentError, ReadFilters, ScanAlignmentResult

__all__ = [
    "ScanAlignmentResult",
    "alignment_format",
    "read_alignment_header",
    "scan_alignment_bins",
    "validate_alignment_inputs",
]


def alignment_format(path: Path) -> str:
    """Detect BAM or CRAM format from filename extension."""
    suffix = path.suffix.lower()
    if suffix == ".bam":
        return "bam"
    if suffix == ".cram":
        return "cram"
    raise AlignmentError(f"{path} must be BAM or CRAM (found suffix {path.suffix!r})")


def validate_alignment_inputs(
    path: Path | str,
    fasta: Path | str | None = None,
) -> tuple[str, Path, Path | None]:
    """Validate alignment path and mandatory reference FASTA for CRAM inputs."""
    resolved = Path(path).expanduser()
    fmt = alignment_format(resolved)
    if fmt == "cram":
        if fasta is None:
            raise AlignmentError(
                f"{resolved} is CRAM and requires a reference FASTA. "
                "Pass --fasta /path/to/reference.fa."
            )
        resolved_fasta = Path(fasta).expanduser()
        if not resolved_fasta.is_file():
            raise AlignmentError(f"CRAM reference FASTA not found: {resolved_fasta}")
    else:
        resolved_fasta = Path(fasta).expanduser() if fasta is not None else None
        if resolved_fasta is not None and not resolved_fasta.is_file():
            raise AlignmentError(f"reference FASTA not found: {resolved_fasta}")

    if not resolved.is_file():
        raise AlignmentError(f"alignment not found: {resolved}")
    return fmt, resolved, resolved_fasta


def read_alignment_header(
    path: Path | str,
    fasta: Path | str | None = None,
) -> list[tuple[str, int]]:
    """Return ordered `(contig, length)` pairs declared in a BAM/CRAM header."""
    resolved = Path(path).expanduser()
    resolved_fasta = Path(fasta).expanduser() if fasta is not None else None
    return _read_alignment_header_core(
        str(resolved),
        str(resolved_fasta) if resolved_fasta is not None else None,
    )


def scan_alignment_bins(
    path: Path | str,
    contigs: Sequence[tuple[str, int]],
    bin_size: int,
    filters: ReadFilters | None = None,
    *,
    threads: int = 1,
    fasta: Path | str | None = None,
    collect_bands: bool = True,
    se_fragment_length: int = 0,
) -> ScanAlignmentResult:
    """Scan BAM or CRAM alignments into per-bin summary arrays via `ccnv._core`."""
    resolved = Path(path).expanduser()
    resolved_fasta = Path(fasta).expanduser() if fasta is not None else None
    active_filters = filters if filters is not None else ReadFilters()
    return _scan_alignment_bins_core(
        str(resolved),
        contigs,
        bin_size,
        active_filters.min_mapping_quality,
        active_filters.require_proper_pair,
        threads,
        str(resolved_fasta) if resolved_fasta is not None else None,
        collect_bands,
        se_fragment_length,
        exclude_duplicates=active_filters.exclude_duplicates,
        exclude_secondary=active_filters.exclude_secondary,
        exclude_supplementary=active_filters.exclude_supplementary,
        exclude_unmapped=active_filters.exclude_unmapped,
    )
