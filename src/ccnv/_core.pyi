"""Type stubs for the compiled ccnv._core Rust extension module."""

from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ccnv.models.results import ScanAlignmentResult as _ScanAlignmentResult

FRAG_BAND_EDGES: tuple[int, ...]
GC_QUANTILE_EDGES: tuple[float, ...]

__all__ = [
    "FRAG_BAND_EDGES",
    "GC_QUANTILE_EDGES",
    "read_alignment_header",
    "read_fasta_contigs_core",
    "scan_alignment_bins",
    "scan_bigwig_means",
    "scan_fasta_gc_quantiles",
    "write_bgzf_indexed",
]

def read_alignment_header(
    path: str,
    fasta_path: str | None = None,
) -> list[tuple[str, int]]: ...
def scan_alignment_bins(
    path: str,
    contigs: Sequence[tuple[str, int]],
    bin_size: int,
    min_mapq: int = 1,
    require_proper_pair: bool = True,
    threads: int = 1,
    fasta_path: str | None = None,
    collect_bands: bool = True,
    se_fragment_length: int = 0,
    *,
    exclude_duplicates: bool = True,
    exclude_secondary: bool = True,
    exclude_supplementary: bool = True,
    exclude_unmapped: bool = True,
) -> _ScanAlignmentResult: ...
def read_fasta_contigs_core(fasta_path: str) -> list[tuple[str, int]]: ...
def scan_fasta_gc_quantiles(
    fasta_path: str,
    contigs: Sequence[tuple[str, int]],
    bin_size: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]: ...
def scan_bigwig_means(
    bw_path: str,
    contigs: Sequence[tuple[str, int]],
    bin_size: int,
    full_bin_denominator: bool = False,
) -> NDArray[np.float32]: ...
def write_bgzf_indexed(
    gz_path: str,
    preset: Literal["bed", "vcf"],
    contig_names: Sequence[str],
    max_contig_len: int,
    chunks: Iterable[bytes],
    threads: int = 1,
) -> str: ...
