"""Genomic track readers for mappability, GC content, and cytobands."""

from __future__ import annotations

import gzip
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np
import pyBigWig
import pysam
from numpy.typing import NDArray

from ccnv.models import (
    CovariateError,
    Cytoband,
    ReferenceError,
    natural_sort_key,
    resolve_contig_name,
    select_layout,
)

logger = logging.getLogger(__name__)

__all__ = [
    "load_cytobands_track",
    "load_gc_track",
    "load_mappability_track",
    "read_fasta_contigs",
]

_MAX_N_FRACTION: Final[float] = 0.5

_FASTA_SUFFIXES: Final[frozenset[str]] = frozenset({".fa", ".fasta", ".fna"})
_BIGWIG_SUFFIXES: Final[frozenset[str]] = frozenset({".bw", ".bigwig"})
_BED_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".bed", ".bedgraph", ".tsv", ".txt", ".wig"}
)

_ASCII_BASE_LUT: Final[NDArray[np.uint8]] = np.zeros(256, dtype=np.uint8)
for _ch in b"GgCc":
    _ASCII_BASE_LUT[_ch] = 1
for _ch in b"AaTtUu":
    _ASCII_BASE_LUT[_ch] = 2


def _bin_weighted_mean(
    bin_starts: NDArray[np.int64],
    bin_ends: NDArray[np.int64],
    iv_starts: NDArray[np.int64],
    iv_ends: NDArray[np.int64],
    iv_values: NDArray[np.float64],
    *,
    full_bin_denominator: bool = False,
) -> NDArray[np.float64]:
    """Compute overlap-weighted mean track value for sorted non-overlapping bins."""
    n_bins = bin_starts.size
    default_val = 0.0 if full_bin_denominator else np.nan
    result = np.full(n_bins, default_val, dtype=np.float64)
    if n_bins == 0 or iv_starts.size == 0:
        return result

    valid = np.isfinite(iv_values) & (iv_ends > iv_starts)
    if not np.any(valid):
        return result
    iv_s = iv_starts[valid]
    iv_e = iv_ends[valid]
    iv_v = iv_values[valid]

    first_bin = np.searchsorted(bin_ends, iv_s, side="right")
    last_bin = np.searchsorted(bin_starts, iv_e, side="left")
    has_overlap = first_bin < last_bin
    if not np.any(has_overlap):
        return result

    iv_s = iv_s[has_overlap]
    iv_e = iv_e[has_overlap]
    iv_v = iv_v[has_overlap]
    first_bin = first_bin[has_overlap]
    last_bin = last_bin[has_overlap]

    weighted_sum = np.zeros(n_bins, dtype=np.float64)
    covered_len = np.zeros(n_bins, dtype=np.float64)

    single = (last_bin - first_bin) == 1
    if np.any(single):
        b_idx = first_bin[single]
        ov = np.minimum(iv_e[single], bin_ends[b_idx]) - np.maximum(
            iv_s[single], bin_starts[b_idx]
        )
        pos = ov > 0
        if np.any(pos):
            b_pos = b_idx[pos]
            w_pos = ov[pos].astype(np.float64)
            weighted_sum += np.bincount(
                b_pos, weights=w_pos * iv_v[single][pos], minlength=n_bins
            )
            covered_len += np.bincount(b_pos, weights=w_pos, minlength=n_bins)

    multi_indices = np.flatnonzero(~single)
    for idx in multi_indices:
        b0 = int(first_bin[idx])
        b1 = int(last_bin[idx])
        s = int(iv_s[idx])
        e = int(iv_e[idx])
        v = float(iv_v[idx])
        b_slice = slice(b0, b1)
        ov = np.minimum(e, bin_ends[b_slice]) - np.maximum(s, bin_starts[b_slice])
        pos = ov > 0
        if np.any(pos):
            w = ov[pos].astype(np.float64)
            weighted_sum[b_slice][pos] += w * v
            covered_len[b_slice][pos] += w

    has_cov = covered_len > 0
    if full_bin_denominator:
        bin_lengths = np.maximum(bin_ends - bin_starts, 1).astype(np.float64)
        result[has_cov] = weighted_sum[has_cov] / bin_lengths[has_cov]
    else:
        result[has_cov] = weighted_sum[has_cov] / covered_len[has_cov]
    return result


def _load_bigwig_intervals(
    path: Path,
    contigs: Sequence[str],
    starts: NDArray[np.int64],
    ends: NDArray[np.int64],
    *,
    label: str,
    full_bin_denominator: bool = False,
) -> NDArray[np.float64]:
    """Read per-bin overlap-weighted means from a BigWig file via pyBigWig."""
    resolved = path.expanduser()
    if not resolved.is_file():
        raise CovariateError(f"{label} file not found: {resolved}")
    try:
        bw = pyBigWig.open(str(resolved))
    except Exception as error:
        raise CovariateError(
            f"failed to open {label} BigWig {resolved}: {error}"
        ) from error
    if bw is None or not bw.isBigWig():
        if bw is not None:
            bw.close()
        raise CovariateError(f"{resolved} is not a valid BigWig file")

    try:
        bw_chroms = bw.chroms() or {}
        avail = set(bw_chroms.keys())
        contig_arr = np.asarray(contigs, dtype=object)
        out = np.full(
            starts.size,
            0.0 if full_bin_denominator else np.nan,
            dtype=np.float64,
        )
        matched_any = False

        for contig in dict.fromkeys(contigs):
            bw_contig = resolve_contig_name(str(contig), avail)
            if bw_contig is None:
                continue
            matched_any = True
            mask = contig_arr == contig
            c_starts = starts[mask]
            c_ends = ends[mask]
            chrom_len = int(bw_chroms[bw_contig])
            raw_intervals = bw.intervals(bw_contig, 0, chrom_len)
            if raw_intervals:
                iv_arr = np.array(raw_intervals, dtype=np.float64)
                out[mask] = _bin_weighted_mean(
                    c_starts,
                    c_ends,
                    iv_arr[:, 0].astype(np.int64),
                    iv_arr[:, 1].astype(np.int64),
                    iv_arr[:, 2],
                    full_bin_denominator=full_bin_denominator,
                )
    finally:
        bw.close()

    if not matched_any and starts.size > 0:
        raise CovariateError(
            f"{label} BigWig {resolved} contains none of the target contigs"
        )
    return out


def load_mappability_track(
    path: Path,
    contigs: Sequence[str],
    starts: NDArray[np.int64],
    ends: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Load per-bin mean mappability from a BigWig file."""
    values = _load_bigwig_intervals(
        path,
        contigs,
        starts,
        ends,
        label="mappability",
        full_bin_denominator=True,
    )
    values = np.nan_to_num(values, nan=0.0)
    values = np.clip(values, 0.0, 1.0)
    logger.debug(
        "loaded mappability track %s: %d/%d finite bins",
        path,
        int(np.count_nonzero(np.isfinite(values))),
        values.size,
    )
    return values


def _compute_fasta_gc(
    path: Path,
    contigs: Sequence[str],
    starts: NDArray[np.int64],
    ends: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Compute per-bin GC fraction and >50% N-base mask from an indexed FASTA."""
    fasta = path.expanduser()
    if not fasta.is_file():
        raise CovariateError(f"FASTA not found: {fasta}")
    if not Path(f"{fasta}.fai").is_file():
        try:
            pysam.faidx(str(fasta))
        except Exception as error:
            raise CovariateError(f"failed to index FASTA {fasta}: {error}") from error

    try:
        handle = pysam.FastaFile(str(fasta))
    except Exception as error:
        raise CovariateError(f"failed to open FASTA {fasta}: {error}") from error

    try:
        avail = set(handle.references)
        contig_arr = np.asarray(contigs, dtype=object)
        gc_out = np.full(starts.size, np.nan, dtype=np.float64)
        n_mask = np.ones(starts.size, dtype=np.bool_)
        matched_any = False

        for contig in dict.fromkeys(contigs):
            fa_contig = resolve_contig_name(str(contig), avail)
            if fa_contig is None:
                continue
            matched_any = True
            mask = contig_arr == contig
            c_starts = starts[mask]
            c_ends = ends[mask]
            seq_bytes = np.frombuffer(
                handle.fetch(fa_contig).encode("ascii"), dtype=np.uint8
            )
            seq_len = seq_bytes.size
            if seq_len == 0:
                continue
            codes = _ASCII_BASE_LUT[seq_bytes]
            del seq_bytes
            gc_prefix = np.empty(seq_len + 1, dtype=np.int32)
            at_prefix = np.empty(seq_len + 1, dtype=np.int32)
            gc_prefix[0] = 0
            at_prefix[0] = 0
            np.cumsum(codes == 1, dtype=np.int32, out=gc_prefix[1:])
            np.cumsum(codes == 2, dtype=np.int32, out=at_prefix[1:])
            del codes

            clamped_s = np.clip(c_starts, 0, seq_len)
            clamped_e = np.clip(c_ends, clamped_s, seq_len)
            gc_counts = gc_prefix[clamped_e] - gc_prefix[clamped_s]
            at_counts = at_prefix[clamped_e] - at_prefix[clamped_s]
            valid_bases = gc_counts + at_counts
            bin_lengths = np.maximum(c_ends - c_starts, 1)
            n_counts = np.maximum(bin_lengths - valid_bases, 0)
            n_fraction = n_counts / bin_lengths

            bad = (valid_bases <= 0) | (n_fraction > _MAX_N_FRACTION)
            c_gc = np.full(c_starts.size, np.nan, dtype=np.float64)
            good = ~bad
            c_gc[good] = gc_counts[good] / valid_bases[good]

            gc_out[mask] = c_gc
            n_mask[mask] = bad
    finally:
        handle.close()

    if not matched_any and starts.size > 0:
        raise CovariateError(f"FASTA {fasta} contains none of the target contigs")
    return gc_out, n_mask


def _load_bed_gc(
    path: Path,
    contigs: Sequence[str],
    starts: NDArray[np.int64],
    ends: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Load per-bin GC fraction from a 0-based half-open BED/bedGraph/TSV file."""
    resolved = path.expanduser()
    if not resolved.is_file():
        raise CovariateError(f"GC track file not found: {resolved}")

    by_contig: dict[str, tuple[list[int], list[int], list[float]]] = {}
    opener = gzip.open if resolved.suffix.lower() == ".gz" else open
    try:
        with opener(resolved, "rt", encoding="utf-8") as handle:
            for line_num, raw in enumerate(handle, start=1):
                line = raw.strip()
                if (
                    not line
                    or line.startswith("#")
                    or line.startswith("track")
                    or line.startswith("browser")
                ):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    raise CovariateError(
                        f"{resolved}:{line_num}: expected at least 4 columns"
                    )
                if line_num == 1 and not parts[1].lstrip("-").isdigit():
                    continue
                try:
                    s = int(parts[1])
                    e = int(parts[2])
                    val = float(parts[3])
                except ValueError as error:
                    raise CovariateError(
                        f"{resolved}:{line_num}: invalid numeric coordinate or GC value"
                    ) from error
                bucket = by_contig.setdefault(parts[0], ([], [], []))
                bucket[0].append(s)
                bucket[1].append(e)
                bucket[2].append(val)
    except OSError as error:
        raise CovariateError(f"failed to read GC track {resolved}: {error}") from error

    avail = set(by_contig.keys())
    contig_arr = np.asarray(contigs, dtype=object)
    out = np.full(starts.size, np.nan, dtype=np.float64)
    matched_any = False

    for contig in dict.fromkeys(contigs):
        bed_contig = resolve_contig_name(str(contig), avail)
        if bed_contig is None:
            continue
        matched_any = True
        mask = contig_arr == contig
        iv_s, iv_e, iv_v = by_contig[bed_contig]
        out[mask] = _bin_weighted_mean(
            starts[mask],
            ends[mask],
            np.asarray(iv_s, dtype=np.int64),
            np.asarray(iv_e, dtype=np.int64),
            np.asarray(iv_v, dtype=np.float64),
        )

    if not matched_any and starts.size > 0:
        raise CovariateError(f"GC track {resolved} contains none of the target contigs")
    return out


def _classify_gc_source(path: Path) -> str:
    """Identify whether a GC track path is FASTA, BigWig, or BED/TSV."""
    suffixes = [s.lower() for s in path.suffixes]
    if not suffixes:
        return "bed"
    effective = (
        suffixes[-2] if suffixes[-1] == ".gz" and len(suffixes) >= 2 else suffixes[-1]
    )
    if effective in _FASTA_SUFFIXES:
        return "fasta"
    if effective in _BIGWIG_SUFFIXES:
        return "bigwig"
    if effective in _BED_SUFFIXES:
        return "bed"
    raise CovariateError(
        f"unsupported GC track extension for {path}; "
        "expected BigWig (.bw/.bigWig), BED (.bed/.bed.gz), or FASTA (.fa/.fasta/.fna)"
    )


def load_gc_track(
    path: Path,
    contigs: Sequence[str],
    starts: NDArray[np.int64],
    ends: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Load per-bin GC content in [0, 1] and N-base mask from FASTA, BigWig, or BED."""
    kind = _classify_gc_source(path)
    if kind == "fasta":
        gc, n_mask = _compute_fasta_gc(path, contigs, starts, ends)
    elif kind == "bigwig":
        gc = _load_bigwig_intervals(path, contigs, starts, ends, label="GC")
        n_mask = ~np.isfinite(gc)
    else:
        gc = _load_bed_gc(path, contigs, starts, ends)
        n_mask = ~np.isfinite(gc)

    finite = np.isfinite(gc)
    if np.any(finite):
        max_val = float(np.nanmax(gc[finite]))
        if 1.0001 < max_val <= 100.0:
            gc[finite] /= 100.0
        invalid_range = (gc < 0.0) | (gc > 1.0)
        gc[invalid_range] = np.nan
        n_mask |= invalid_range

    logger.debug(
        "loaded GC track %s (%s): %d/%d valid bins",
        path,
        kind,
        int(np.count_nonzero(np.isfinite(gc) & ~n_mask)),
        gc.size,
    )
    return gc, n_mask


def load_cytobands_track(
    path: Path, contig_lengths: Mapping[str, int]
) -> tuple[Cytoband, ...]:
    """Parse a UCSC cytoband BED/TSV file (.txt, .bed, or .gz) aligned to contigs."""
    contig_lookup: dict[str, str] = {}
    for c in contig_lengths:
        contig_lookup[c.casefold()] = c
        stripped = c[3:] if c.lower().startswith("chr") else c
        contig_lookup[stripped.casefold()] = c

    open_fn = gzip.open if path.suffix.lower() == ".gz" else open
    bands: list[Cytoband] = []
    with open_fn(path, "rt", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("track"):
                continue
            fields = line.split("\t") if "\t" in line else line.split()
            if len(fields) < 4:
                continue
            raw_contig = fields[0]
            stripped_raw = (
                raw_contig[3:] if raw_contig.lower().startswith("chr") else raw_contig
            )
            canonical = contig_lookup.get(raw_contig.casefold()) or contig_lookup.get(
                stripped_raw.casefold()
            )
            if canonical is None:
                continue
            try:
                start = int(fields[1])
                end = int(fields[2])
            except ValueError:
                continue
            max_len = contig_lengths[canonical]
            start = max(0, min(start, max_len))
            end = max(0, min(end, max_len))
            if end <= start:
                continue
            name = fields[3]
            stain = fields[4] if len(fields) >= 5 else "gneg"
            bands.append(
                Cytoband(
                    contig=canonical,
                    start=start,
                    end=end,
                    name=name,
                    stain=stain,
                )
            )
    return tuple(
        sorted(bands, key=lambda b: (natural_sort_key(b.contig), b.start, b.end))
    )


def read_fasta_contigs(
    fasta_path: Path | str,
    *,
    all_contigs: bool = False,
) -> dict[str, int]:
    """Read contig names and lengths from a FASTA file and select canonical layout."""
    fasta = Path(fasta_path).expanduser().resolve()
    if not fasta.is_file():
        raise ReferenceError(f"FASTA not found: {fasta}")
    if not Path(f"{fasta}.fai").is_file():
        logger.debug("creating FASTA index for %s", fasta)
        try:
            pysam.faidx(str(fasta))
        except Exception as error:
            raise ReferenceError(f"failed to index FASTA {fasta}: {error}") from error
    try:
        handle = pysam.FastaFile(str(fasta))
    except Exception as error:
        raise ReferenceError(f"failed to open FASTA {fasta}: {error}") from error
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
    return {contig.name: contig.length for contig in layout}
