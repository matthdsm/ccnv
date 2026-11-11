"""Genomic track readers for mappability, GC content, and cytobands."""

from __future__ import annotations

import gzip
import logging
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
    if n == 0:
        return [], 1, []

    runs: list[tuple[str, int, int]] = []
    run_start = 0
    prev = str(contigs[0])
    for idx in range(1, n):
        cur = str(contigs[idx])
        if cur != prev:
            runs.append((prev, run_start, idx))
            prev = cur
            run_start = idx
    runs.append((prev, run_start, n))

    bin_size = 0
    for _, s_idx, e_idx in runs:
        if e_idx - s_idx >= 2:
            bin_size = int(ends[s_idx] - starts[s_idx])
            break
    if bin_size <= 0:
        bin_size = max(int(np.max(ends - starts)), 1)

    specs: list[tuple[str, int]] = []
    placements: list[tuple[slice, slice, NDArray[np.intp] | None]] = []
    core_offset = 0
    for cname, s_idx, e_idx in runs:
        c_starts = starts[s_idx:e_idx]
        c_ends = ends[s_idx:e_idx]
        contig_len = int(np.max(c_ends))
        if contig_len <= 0:
            raise CovariateError(f"non-positive bin end coordinate on contig {cname}")
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
        elif bool(np.all(c_starts % bin_size == 0)):
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
    """Compute per-bin GC fraction and >50% N-base mask from a FASTA via `ccnv._core`."""
    fasta = path.expanduser()
    if not fasta.is_file():
        raise CovariateError(f"FASTA not found: {fasta}")

    specs, bin_size, placements = _extract_contig_grid(contigs, starts, ends)
    gc_core, n_frac_core, _ = scan_fasta_gc_quantiles(str(fasta), specs, bin_size)
    if starts.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.bool_)

    if len(placements) == 1 and placements[0][2] is None:
        gc_raw = gc_core.astype(np.float64)
        n_frac = n_frac_core
    else:
        gc_raw = np.empty(starts.size, dtype=np.float64)
        n_frac = np.empty(starts.size, dtype=np.float32)
        for dst_slice, src_slice, bin_idx in placements:
            c_gc = gc_core[src_slice]
            c_nf = n_frac_core[src_slice]
            gc_raw[dst_slice] = c_gc if bin_idx is None else c_gc[bin_idx]
            n_frac[dst_slice] = c_nf if bin_idx is None else c_nf[bin_idx]

    n_mask: NDArray[np.bool_] = (~np.isfinite(gc_raw)) | (n_frac > _MAX_N_FRACTION)
    gc_raw[n_mask] = np.nan
    return gc_raw, n_mask


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

    try:
        df = pl.read_csv(
            resolved,
            has_header=False,
            separator="\t",
            comment_prefix="#",
            infer_schema=False,
            truncate_ragged_lines=True,
        )
    except pl.exceptions.NoDataError:
        df = pl.DataFrame()
    except (OSError, pl.exceptions.PolarsError) as error:
        raise CovariateError(f"failed to read GC track {resolved}: {error}") from error

    if df.height > 0:
        if df.width == 1:
            col0 = df.columns[0]
            df = df.filter(pl.col(col0).str.strip_chars().str.len_bytes() > 0)
            if df.height > 0:
                df = df.select(
                    pl.col(col0)
                    .str.strip_chars()
                    .str.replace_all(r"\s+", "\t")
                    .str.split_exact("\t", 3)
                    .struct.rename_fields(
                        ["column_1", "column_2", "column_3", "column_4"]
                    )
                    .alias("_parts")
                ).unnest("_parts")
        elif df.width < 4:
            raise CovariateError(f"{resolved}: expected at least 4 columns")

        c1, c2, c3, c4 = df.columns[0], df.columns[1], df.columns[2], df.columns[3]
        df = df.select(
            pl.col(c1).str.strip_chars().alias("contig"),
            pl.col(c2).str.strip_chars().alias("start_s"),
            pl.col(c3).str.strip_chars().alias("end_s"),
            pl.col(c4).str.strip_chars().alias("val_s"),
        ).filter(
            pl.col("contig").is_not_null()
            & (pl.col("contig").str.len_bytes() > 0)
            & ~pl.col("contig").str.starts_with("track")
            & ~pl.col("contig").str.starts_with("browser")
        )

        if df.height > 0:
            first_s = df.item(0, "start_s")
            if isinstance(first_s, str) and not first_s.lstrip("-").isdigit():
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
        part = by_contig[bed_contig]
        out[mask] = _bin_weighted_mean(
            starts[mask],
            ends[mask],
            part["start"].to_numpy(),
            part["end"].to_numpy(),
            part["val"].to_numpy(),
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
        if (
            1.0 < float(np.nanmedian(gc[finite]))
            and float(np.nanmax(gc[finite])) <= 100.0
        ):
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
            if not line or line.startswith(("#", "track")):
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
    try:
        pairs = read_fasta_contigs_core(str(fasta))
    except (CovariateError, OSError) as error:
        raise ReferenceError(f"failed to read FASTA {fasta}: {error}") from error
    if not pairs:
        raise ReferenceError(f"{fasta} declares no contigs")
    header_names = [name for name, _ in pairs]
    header_lengths = [int(length) for _, length in pairs]
    if any(length <= 0 for length in header_lengths):
        raise ReferenceError(f"{fasta} contains non-positive contig lengths")
    layout = select_layout(header_names, header_lengths, all_contigs=all_contigs)
    if not layout:
        raise ReferenceError(
            f"{fasta} declares no canonical chromosomes; "
            "pass --all-contigs to include every contig"
        )
    return {contig.name: contig.length for contig in layout}
