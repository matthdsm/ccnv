"""Genomic track readers for mappability, GC content, and cytobands."""

from __future__ import annotations

import functools
import gzip
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from numpy.typing import NDArray

from ccnv._core import (
    read_fasta_contigs_core,
    scan_bigwig_means,
    scan_fasta_gc_quantiles,
)
from ccnv.models import (
    ContigLookup,
    CovariateError,
    Cytoband,
    ReferenceError,
    natural_sort_key,
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


def _extract_contig_grid(
    contigs: Sequence[str],
    starts: NDArray[np.int64],
    ends: NDArray[np.int64],
) -> tuple[
    list[tuple[str, int]], int, list[tuple[slice, slice, NDArray[np.intp] | None]]
]:
    """Map caller bin arrays to `(contig, length)` specs and output slices."""
    n = starts.size
    if len(contigs) != n or ends.size != n:
        raise CovariateError("contig, start, and end arrays must have equal length")
    if n == 0:
        return [], 1, []

    runs: list[tuple[str, int, int]] = []
    seen_contigs: set[str] = set()
    run_start = 0
    prev = str(contigs[0])
    for idx in range(1, n):
        cur = str(contigs[idx])
        if cur != prev:
            if prev in seen_contigs:
                raise CovariateError(f"non-contiguous contig run on contig {prev}")
            seen_contigs.add(prev)
            runs.append((prev, run_start, idx))
            prev = cur
            run_start = idx
    if prev in seen_contigs:
        raise CovariateError(f"non-contiguous contig run on contig {prev}")
    runs.append((prev, run_start, n))

    for cname, s_idx, e_idx in runs:
        c_starts = starts[s_idx:e_idx]
        c_ends = ends[s_idx:e_idx]
        if bool(np.any(c_starts < 0)) or bool(np.any(c_ends <= c_starts)):
            raise CovariateError(
                f"non-positive or inverted bin coordinates on contig {cname}"
            )
        if c_starts.size >= 2 and bool(np.any(c_starts[1:] < c_ends[:-1])):
            raise CovariateError(
                f"unsorted or overlapping bin coordinates on contig {cname}"
            )

    bin_size = 0
    for _, s_idx, e_idx in runs:
        if e_idx - s_idx >= 2:
            bin_size = int(ends[s_idx] - starts[s_idx])
            break

    if bin_size <= 0:
        max_width = max(int(np.max(ends - starts)), 1)
        pos_starts = [int(s) for s in starts if int(s) > 0]
        if pos_starts:
            step = functools.reduce(math.gcd, pos_starts)
            if step < max_width:
                raise CovariateError(
                    f"non-uniform bin coordinates on contig {runs[0][0]} are unsupported"
                )
            bin_size = step
        else:
            bin_size = max_width

    specs: list[tuple[str, int]] = []
    placements: list[tuple[slice, slice, NDArray[np.intp] | None]] = []
    core_offset = 0
    for cname, s_idx, e_idx in runs:
        c_starts = starts[s_idx:e_idx]
        c_ends = ends[s_idx:e_idx]
        contig_len = int(c_ends[-1])
        n_core = (contig_len + bin_size - 1) // bin_size
        is_full_grid = (
            c_starts.size == n_core
            and int(c_starts[0]) == 0
            and (
                n_core == 1
                or (
                    bool(np.all(c_starts[1:] == c_ends[:-1]))
                    and bool(np.all(c_ends[:-1] - c_starts[:-1] == bin_size))
                )
            )
        )
        if is_full_grid:
            bin_idx = None
        elif bool(np.all(c_starts % bin_size == 0)) and bool(
            np.all(c_ends == np.minimum(c_starts + bin_size, contig_len))
        ):
            bin_idx = (c_starts // bin_size).astype(np.intp)
        else:
            raise CovariateError(
                f"non-uniform bin coordinates on contig {cname} are unsupported"
            )
        specs.append((cname, contig_len))
        placements.append(
            (slice(s_idx, e_idx), slice(core_offset, core_offset + n_core), bin_idx)
        )
        core_offset += n_core

    return specs, bin_size, placements


def _load_bigwig_intervals(
    path: Path,
    contigs: Sequence[str],
    starts: NDArray[np.int64],
    ends: NDArray[np.int64],
    *,
    label: str,
    full_bin_denominator: bool = False,
) -> NDArray[np.float64]:
    """Read per-bin overlap-weighted means from a BigWig file via `ccnv._core`."""
    resolved = path.expanduser()
    if not resolved.is_file():
        raise CovariateError(f"{label} file not found: {resolved}")

    specs, bin_size, placements = _extract_contig_grid(contigs, starts, ends)
    core_vals = scan_bigwig_means(str(resolved), specs, bin_size, full_bin_denominator)
    if starts.size == 0:
        return np.empty(0, dtype=np.float64)

    if len(placements) == 1 and placements[0][2] is None:
        return core_vals.astype(np.float64)

    out = np.empty(starts.size, dtype=np.float64)
    for dst_slice, src_slice, bin_idx in placements:
        chunk = core_vals[src_slice]
        out[dst_slice] = chunk if bin_idx is None else chunk[bin_idx]
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
    if np.any(np.isinf(values)):
        raise CovariateError(f"mappability track {path} contains infinite values")
    finite_vals = values[np.isfinite(values)]
    if finite_vals.size > 0 and (
        bool(np.any(finite_vals < -1e-5)) or bool(np.any(finite_vals > 1.0 + 1e-5))
    ):
        raise CovariateError(
            f"mappability track {path} contains values outside [0.0, 1.0]: "
            f"min={float(np.min(finite_vals)):.6g}, max={float(np.max(finite_vals)):.6g}"
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
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float32]]:
    """Compute per-bin GC fraction, >50% N-base mask, and 8-bin GC quantiles in one FASTA pass."""
    fasta = path.expanduser()
    if not fasta.is_file():
        raise CovariateError(f"FASTA not found: {fasta}")

    specs, bin_size, placements = _extract_contig_grid(contigs, starts, ends)
    gc_core, n_frac_core, gc_q_core = scan_fasta_gc_quantiles(
        str(fasta), specs, bin_size
    )
    if starts.size == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.bool_),
            np.empty((0, 8), dtype=np.float32),
        )

    if len(placements) == 1 and placements[0][2] is None:
        gc_raw = gc_core.astype(np.float64)
        n_frac = n_frac_core
        gc_q_out = np.nan_to_num(gc_q_core, nan=0.0).astype(np.float32, copy=False)
    else:
        gc_raw = np.empty(starts.size, dtype=np.float64)
        n_frac = np.empty(starts.size, dtype=np.float32)
        gc_q_out = np.empty((starts.size, gc_q_core.shape[1]), dtype=np.float32)
        for dst_slice, src_slice, bin_idx in placements:
            c_gc = gc_core[src_slice]
            c_nf = n_frac_core[src_slice]
            c_gq = gc_q_core[src_slice]
            gc_raw[dst_slice] = c_gc if bin_idx is None else c_gc[bin_idx]
            n_frac[dst_slice] = c_nf if bin_idx is None else c_nf[bin_idx]
            gc_q_out[dst_slice] = c_gq if bin_idx is None else c_gq[bin_idx]
        gc_q_out = np.nan_to_num(gc_q_out, nan=0.0).astype(np.float32, copy=False)

    n_mask: NDArray[np.bool_] = (~np.isfinite(gc_raw)) | (n_frac > _MAX_N_FRACTION)
    gc_raw[n_mask] = np.nan
    return gc_raw, n_mask, gc_q_out


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

    _extract_contig_grid(contigs, starts, ends)
    if starts.size == 0:
        return np.empty(0, dtype=np.float64)

    try:
        df = pl.read_csv(
            resolved,
            has_header=False,
            separator="\x00",
            quote_char=None,
            comment_prefix="#",
            infer_schema=False,
        )
    except pl.exceptions.NoDataError:
        df = pl.DataFrame()
    except (OSError, pl.exceptions.PolarsError) as error:
        raise CovariateError(f"failed to read GC track {resolved}: {error}") from error

    if df.height > 0:
        col0 = df.columns[0]
        df = df.select(pl.col(col0).str.strip_chars().alias("line")).filter(
            pl.col("line").is_not_null()
            & (pl.col("line").str.len_bytes() > 0)
            & ~pl.col("line").str.starts_with("#")
            & ~pl.col("line").str.starts_with("track")
            & ~pl.col("line").str.starts_with("browser")
        )
        if df.height > 0:
            df = df.select(
                pl.col("line")
                .str.replace_all(r"\s+", "\t")
                .str.split_exact("\t", 3)
                .struct.rename_fields(["contig", "start_s", "end_s", "val_s"])
                .alias("_parts")
            ).unnest("_parts")

            first_s = df.item(0, "start_s")
            first_v = df.item(0, "val_s")
            if (
                isinstance(first_s, str)
                and isinstance(first_v, str)
                and not first_s.lstrip("-").isdigit()
            ):
                df = df.slice(1)

        if df.height > 0:
            if (
                df.select(
                    pl.any_horizontal(pl.col("start_s", "end_s", "val_s").is_null())
                )
                .to_series()
                .any()
            ):
                raise CovariateError(f"{resolved}: expected at least 4 columns")
            try:
                df = df.select(
                    pl.col("contig"),
                    pl.col("start_s").cast(pl.Int64, strict=True).alias("start"),
                    pl.col("end_s").cast(pl.Int64, strict=True).alias("end"),
                    pl.col("val_s").cast(pl.Float64, strict=True).alias("val"),
                )
            except pl.exceptions.PolarsError as error:
                raise CovariateError(
                    f"{resolved}: invalid numeric coordinate or GC value"
                ) from error

    by_contig: dict[str, pl.DataFrame] = {}
    if df.height > 0 and "start" in df.columns:
        for key, part in df.partition_by("contig", as_dict=True).items():
            contig_key = str(key[0]) if isinstance(key, tuple) else str(key)
            by_contig[contig_key] = part

    lookup = ContigLookup.from_names(by_contig.keys())
    contig_arr = np.asarray(contigs, dtype=object)
    out = np.full(starts.size, np.nan, dtype=np.float64)
    matched_any = False

    for contig in dict.fromkeys(contigs):
        bed_contig = lookup.resolve(str(contig))
        if bed_contig is None:
            continue
        matched_any = True
        mask = contig_arr == contig
        part = by_contig[bed_contig]
        out[mask] = _bin_weighted_mean(
            starts[mask],
            ends[mask],
            part["start"].to_numpy(),
            part["end"].to_numpy(),
            part["val"].to_numpy(),
        )

    if not matched_any:
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
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float32] | None]:
    """Load per-bin GC content in [0, 1], N-base mask, and optional FASTA GC quantiles."""
    kind = _classify_gc_source(path)
    gc_quantiles: NDArray[np.float32] | None = None
    if kind == "fasta":
        gc, n_mask, gc_quantiles = _compute_fasta_gc(path, contigs, starts, ends)
    elif kind == "bigwig":
        gc = _load_bigwig_intervals(path, contigs, starts, ends, label="GC")
        n_mask = ~np.isfinite(gc)
    else:
        gc = _load_bed_gc(path, contigs, starts, ends)
        n_mask = ~np.isfinite(gc)

    finite = np.isfinite(gc)
    if np.any(finite):
        finite_vals = gc[finite]
        if 1.0 < float(np.median(finite_vals)) and float(np.max(finite_vals)) <= 100.0:
            gc[finite] = finite_vals / 100.0
            finite_vals = gc[finite]
        invalid_range = np.zeros(gc.shape, dtype=np.bool_)
        invalid_range[finite] = (finite_vals < 0.0) | (finite_vals > 1.0)
        gc[invalid_range] = np.nan
        n_mask |= invalid_range

    logger.debug(
        "loaded GC track %s (%s): %d/%d valid bins",
        path,
        kind,
        int(np.count_nonzero(np.isfinite(gc) & ~n_mask)),
        gc.size,
    )
    return gc, n_mask, gc_quantiles


def load_cytobands_track(
    path: Path, contig_lengths: Mapping[str, int]
) -> tuple[Cytoband, ...]:
    """Parse a UCSC cytoband BED/TSV file (.txt, .bed, or .gz) aligned to contigs."""
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise CovariateError(f"cytobands file not found: {resolved}")

    lookup = ContigLookup.from_names(contig_lengths.keys())
    open_fn = gzip.open if resolved.suffix.lower() == ".gz" else open
    bands: list[Cytoband] = []
    try:
        with open_fn(resolved, "rt", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith(("#", "track", "browser")):
                    continue
                fields = line.split("\t") if "\t" in line else line.split()
                if len(fields) < 4:
                    continue
                canonical = lookup.resolve(fields[0])
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
    except (OSError, EOFError, UnicodeDecodeError) as error:
        raise CovariateError(f"failed to read cytobands {resolved}: {error}") from error

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
    try:
        pairs = read_fasta_contigs_core(str(fasta))
    except (CovariateError, OSError) as error:
        raise ReferenceError(f"failed to read FASTA {fasta}: {error}") from error
    if not pairs:
        raise ReferenceError(f"{fasta} declares no contigs")
    if any(length <= 0 for _, length in pairs):
        raise ReferenceError(f"{fasta} contains non-positive contig lengths")
    if len({name for name, _ in pairs}) != len(pairs):
        raise ReferenceError(f"{fasta} declares duplicate contig names")
    selected = select_layout(pairs, all_contigs=all_contigs)
    if not selected:
        raise ReferenceError(
            f"{fasta} declares no canonical chromosomes; "
            "pass --all-contigs to include every contig"
        )
    return dict(selected)
