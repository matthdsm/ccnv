"""Cohort sample loading and layout validation (transitional stub)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ccnv.features.binning import bin_intervals
from ccnv.features.cohort import (
    Cohort,
    SexInference,
    SexLabel,
    infer_sample_sexes,
    load_cohort,
    mask_low_coverage,
    normalize_depth,
    rebin_sample_table,
    validate_cohort,
)
from ccnv.models import ReferenceError, select_layout

logger = logging.getLogger(__name__)

__all__ = [
    "Cohort",
    "ReferenceError",
    "SexInference",
    "SexLabel",
    "cohort_from_fasta",
    "infer_sample_sexes",
    "load_cohort",
    "mask_low_coverage",
    "normalize_depth",
    "rebin_sample_table",
    "validate_cohort",
]


def cohort_from_fasta(
    fasta: Path,
    bin_size: int,
    *,
    n_samples: int,
    all_contigs: bool = False,
) -> Cohort:
    """Build identical uniform-depth samples from FASTA contig lengths."""
    if n_samples < 1:
        raise ReferenceError("n_samples must be a positive integer")
    names, lengths = _fasta_contigs(fasta, all_contigs=all_contigs)
    contigs: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for name, length in zip(names, lengths, strict=True):
        contig_starts, contig_ends = bin_intervals(length, bin_size)
        contigs.extend([name] * contig_starts.size)
        starts.extend(int(value) for value in contig_starts)
        ends.extend(int(value) for value in contig_ends)
    start_array = np.asarray(starts, dtype=np.int64)
    end_array = np.asarray(ends, dtype=np.int64)
    expected = np.maximum(end_array - start_array, 1).astype(np.float64)
    counts = np.broadcast_to(expected[:, np.newaxis], (expected.size, n_samples)).copy()
    resolved = fasta.expanduser().resolve()
    identities = tuple(
        {
            "path": str(resolved),
            "format": "flat-fasta",
            "replica": index,
        }
        for index in range(n_samples)
    )
    return Cohort(
        counts=counts,
        contigs=tuple(contigs),
        starts=start_array,
        ends=end_array,
        contig_lengths=dict(zip(names, lengths, strict=True)),
        bin_size=bin_size,
        identities=identities,
        paths=tuple(Path(f"{resolved}#flat-{index}") for index in range(n_samples)),
    )


def _fasta_contigs(path: Path, *, all_contigs: bool) -> tuple[list[str], list[int]]:
    import pysam

    fasta = path.expanduser()
    if not fasta.is_file():
        raise ReferenceError(f"FASTA not found: {fasta}")
    if not Path(f"{fasta}.fai").is_file():
        logger.debug("creating FASTA index for %s", fasta)
        try:
            pysam.faidx(str(fasta))
        except Exception as error:
            raise ReferenceError(f"failed to index FASTA {fasta}: {error}") from error
    handle = pysam.FastaFile(str(fasta))
    try:
        header_names = list(handle.references)
        header_lengths = [int(length) for length in handle.lengths]
    finally:
        handle.close()
    if any(length <= 0 for length in header_lengths):
        raise ReferenceError(f"{fasta} contains non-positive contig lengths")
    layout = select_layout(header_names, header_lengths, all_contigs=all_contigs)
    if not layout:
        raise ReferenceError(
            f"{fasta} declares no canonical chromosomes; "
            "pass --all-contigs to include every contig"
        )
    return [contig.name for contig in layout], [contig.length for contig in layout]
