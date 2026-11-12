"""Grid alignment, covariate normalization, and 100 kb profile helpers for prenatal pipelines."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from ccnv.features.binning import (
    align_sample_to_grid_indices,
    bin_intervals,
    compute_bin_exposure,
    rebin_counts,
)
from ccnv.models import (
    PredictError,
    SampleBuffers,
    SampleMetadata,
    decode_dictionary_contigs,
    decode_metadata,
    is_autosome_name,
    normalize_chr,
    reconstruct_sample_grid,
    unique_dictionary_contigs,
)
from ccnv.stats.prenatal_math import compute_autosomal_density

__all__ = [
    "col_f64",
    "col_i64",
    "compute_100k_profiles",
    "extract_ref_contig_roles",
    "prepare_covariate_multipliers",
    "rebin_track_1d",
    "resolve_5k_covariates",
    "resolve_contig_masks",
    "sample_grid",
]

_EXCLUDED_PRENATAL_AUTOSOMES: Final[frozenset[str]] = frozenset(("13", "18", "21"))


def col_f64(table: pa.Table, name: str) -> NDArray[np.float64]:
    """Extract a PyArrow table column as a float64 NumPy array."""
    return np.asarray(
        table.column(name).to_numpy(zero_copy_only=False), dtype=np.float64
    )


def col_i64(table: pa.Table, name: str) -> NDArray[np.int64]:
    """Extract a PyArrow table column as an int64 NumPy array."""
    return np.asarray(table.column(name).to_numpy(zero_copy_only=False), dtype=np.int64)


def sample_grid(
    meta: SampleMetadata,
) -> tuple[list[str], NDArray[np.int64], NDArray[np.int64]]:
    """Reconstruct ordered contig names and int64 interval coordinates from sample metadata."""
    codes, names, starts_u32, ends_u32 = reconstruct_sample_grid(meta)
    contigs = [names[int(c)] for c in codes]
    return contigs, starts_u32.astype(np.int64), ends_u32.astype(np.int64)


def resolve_contig_masks(
    contigs: Sequence[str],
    *,
    autosomes: Sequence[str] | None = None,
    sex_contigs: Sequence[str] | None = None,
    sex_signal_contig: str | None = None,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    """Resolve boolean (autosomal, chrx, chry) bin masks honoring custom contig roles."""
    if not contigs:
        empty = np.empty(0, dtype=np.bool_)
        return empty, empty, empty
    raw_c = np.asarray(contigs, dtype=object)
    norm_c = np.array([normalize_chr(str(c)) for c in contigs], dtype=object)

    def _matches(names: Sequence[str]) -> NDArray[np.bool_]:
        norm_names = [normalize_chr(c) for c in names]
        return np.isin(raw_c, list(set(names))) | np.isin(norm_c, norm_names)

    chry = (
        _matches((sex_signal_contig,))
        if sex_signal_contig is not None
        else (norm_c == "Y")
    )
    if sex_contigs is not None:
        in_sex = _matches(sex_contigs) & ~chry
        chrx = (
            (in_sex & (norm_c == "X")) if np.any(in_sex & (norm_c == "X")) else in_sex
        )
    else:
        chrx = (norm_c == "X") & ~chry
    auto_all = (
        _matches(autosomes)
        if autosomes is not None
        else np.array([is_autosome_name(str(nc)) for nc in norm_c], dtype=np.bool_)
    ) & ~(chry | chrx)
    filtered = auto_all & ~np.isin(norm_c, list(_EXCLUDED_PRENATAL_AUTOSOMES))
    return (filtered if np.any(filtered) else auto_all), chrx, chry


def prepare_covariate_multipliers(
    gc: NDArray[np.floating] | None,
    mappability: NDArray[np.floating] | None,
    n_mask: NDArray[np.bool_] | None,
    min_mappability: float,
    length: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Normalize GC around median 1.0 and threshold mappability/N-mask bins."""
    for label, arr in (("gc", gc), ("mappability", mappability), ("n_mask", n_mask)):
        if arr is not None and np.asarray(arr).shape != (length,):
            raise PredictError(f"{label} length {np.asarray(arr).size} != {length}")

    map_out = np.ones(length, dtype=np.float32)
    if mappability is not None:
        raw_m = np.asarray(mappability, dtype=np.float32)
        if np.any(np.isfinite(raw_m) & (raw_m >= 0.0)):
            keep_m = np.isfinite(raw_m) & (raw_m >= float(min_mappability))
            map_out = np.where(keep_m, raw_m, np.float32(0.0))
    if n_mask is not None:
        map_out = np.where(np.asarray(n_mask, dtype=np.bool_), np.float32(0.0), map_out)

    if gc is not None:
        raw_gc = np.asarray(gc, dtype=np.float64)
        in_range = np.isfinite(raw_gc) & (raw_gc > 0.0) & (raw_gc <= 1.0)
        if np.any(in_range):
            map_out = np.where(
                np.isfinite(raw_gc) & (raw_gc < 0.0), np.float32(0.0), map_out
            )
        usable_gc = in_range & (map_out > 0.0)
        usable_gc = usable_gc if np.any(usable_gc) else in_range
        if np.any(usable_gc) and (med := float(np.median(raw_gc[usable_gc]))) > 0.0:
            return np.where(in_range, raw_gc / med, 1.0).astype(np.float32), map_out
    return np.ones(length, dtype=np.float32), map_out


def rebin_track_1d(
    values: NDArray[np.floating] | NDArray[np.bool_] | None,
    src_meta: SampleMetadata,
    dst_meta: SampleMetadata,
) -> NDArray[np.float32] | None:
    """Project a sample-grid covariate or mask array onto a rebinned sample grid."""
    if values is None:
        return None
    arr = np.asarray(values)
    if arr.shape != (src_meta.expected_num_bins,):
        raise PredictError(
            f"override track length {arr.size} != {src_meta.expected_num_bins}"
        )
    if src_meta.bin_size == dst_meta.bin_size:
        return arr.astype(np.float32, copy=False)

    is_b = arr.dtype == np.bool_
    val_f64 = np.where(np.isfinite(arr), arr.astype(np.float64), 0.0)
    valid = (
        np.ones(arr.size)
        if is_b
        else (np.isfinite(arr) & (val_f64 >= 0.0)).astype(float)
    )
    fill = 0.0 if is_b else -1.0
    parts: list[NDArray[np.float64]] = []
    pos = 0
    for cname in src_meta.contig_order:
        s_src, e_src = bin_intervals(src_meta.contig_lengths[cname], src_meta.bin_size)
        s_dst, e_dst = bin_intervals(src_meta.contig_lengths[cname], dst_meta.bin_size)
        w = (e_src - s_src).astype(np.float64)
        sl = slice(pos, pos + s_src.size)
        pos += s_src.size
        num = rebin_counts(s_src, e_src, val_f64[sl] * valid[sl] * w, s_dst, e_dst)
        den = rebin_counts(s_src, e_src, valid[sl] * w, s_dst, e_dst)
        parts.append(np.where(den > 0.0, num / np.maximum(den, 1e-9), fill))
    return (np.concatenate(parts) if parts else np.empty(0)).astype(np.float32)


def compute_100k_profiles(
    counts_matrix: NDArray[np.floating],
    contigs: Sequence[str],
    starts: NDArray[np.int64],
    ends: NDArray[np.int64],
    bin_size: int,
    auto_mask: NDArray[np.bool_],
    chrx_mask: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute coarse per-bin log2 ratio profiles and chrX dosage ratios."""
    n_bins, n_samples = counts_matrix.shape
    _, c_ids = np.unique([normalize_chr(str(c)) for c in contigs], return_inverse=True)
    c_ids_i16 = c_ids.astype(np.int16)
    exp = (
        compute_bin_exposure(starts, ends, max(1, bin_size))
        if n_bins
        else np.empty(0, np.float32)
    )
    ones = np.ones(n_bins, dtype=np.float32)
    full_log2 = np.full((n_samples, n_bins), np.nan, dtype=np.float64)
    chrx_ratios = np.ones(n_samples, dtype=np.float64)

    for i in range(n_samples):
        col = np.asarray(counts_matrix[:, i], dtype=np.float64)
        r_i = compute_autosomal_density(col, c_ids_i16, exp, ones, ones, auto_mask)
        denom = r_i * exp.astype(np.float64)
        pos = (denom > 0.0) & (col > 0.0)
        full_log2[i, pos] = np.log2(col[pos] / denom[pos])
        x_vals = full_log2[i, chrx_mask & pos]
        if x_vals.size > 0:
            chrx_ratios[i] = float(2.0 ** float(np.mean(x_vals)))
    return full_log2, chrx_ratios


def extract_ref_contig_roles(
    reference_table: pa.Table, ffy_ref: pa.Table
) -> tuple[tuple[str, ...] | None, tuple[str, ...] | None, str | None]:
    """Read optional custom contig roles from reference metadata or prenatal_ffy branch."""
    model = decode_metadata(reference_table.schema.metadata).get("model")
    autosomes = sex_contigs = None
    sig_contig: str | None = None
    if isinstance(model, dict):
        if isinstance(raw_a := model.get("autosomal_contigs"), list) and raw_a:
            autosomes = tuple(str(c) for c in raw_a)
        if isinstance(raw_s := model.get("sex_contigs"), list) and raw_s:
            sex_contigs = tuple(str(c) for c in raw_s)
        if isinstance(raw_sig := model.get("sex_signal_contig"), str) and raw_sig:
            sig_contig = raw_sig
    if sig_contig is None and ffy_ref.num_rows > 0:
        ffy_c = unique_dictionary_contigs(ffy_ref.column("contig"))
        if len(ffy_c) == 1:
            sig_contig = ffy_c[0]
    return autosomes, sex_contigs, sig_contig


def resolve_5k_covariates(
    bufs: SampleBuffers,
    bufs_5k: SampleBuffers,
    auto_5k: pa.Table | None,
    ffy_ref: pa.Table,
    ffy_align: NDArray[np.int64],
    gc_override: NDArray[np.floating] | None,
    map_override: NDArray[np.floating] | None,
    n_mask_override: NDArray[np.bool_] | None,
    min_mappability: float,
) -> tuple[
    NDArray[np.float32], NDArray[np.float32], NDArray[np.float64], NDArray[np.float64]
]:
    """Resolve 5 kb autosomal and MSY GC/mappability multipliers honoring CLI overrides."""
    n_5k = bufs_5k.num_rows
    gc_ovr = rebin_track_1d(gc_override, bufs.meta, bufs_5k.meta)
    map_ovr = rebin_track_1d(map_override, bufs.meta, bufs_5k.meta)
    nmask_ovr = rebin_track_1d(n_mask_override, bufs.meta, bufs_5k.meta)
    nmask_b = (nmask_ovr >= 0.5) if nmask_ovr is not None else None

    ref_gc = np.full(n_5k, np.nan, dtype=np.float32)
    ref_map = np.ones(n_5k, dtype=np.float32)
    if gc_ovr is None or map_ovr is None:
        ref_tables = [auto_5k, ffy_ref] if auto_5k is not None else [ffy_ref]
        for tbl in ref_tables:
            align = (
                ffy_align
                if tbl is ffy_ref
                else align_sample_to_grid_indices(
                    bufs_5k.meta,
                    decode_dictionary_contigs(tbl.column("contig")),
                    col_i64(tbl, "start"),
                    col_i64(tbl, "end"),
                    allow_alias=True,
                )
            )
            hit = align >= 0
            if np.any(hit):
                col_g, col_m = col_f64(tbl, "gc"), col_f64(tbl, "mappability")
                if np.any(np.isfinite(col_g) & (col_g > 0.0)):
                    ref_gc[align[hit]] = col_g[hit].astype(np.float32)
                if np.any(np.isfinite(col_m) & (col_m >= 0.0)):
                    ref_map[align[hit]] = col_m[hit].astype(np.float32)

    gc_5k, map_5k = prepare_covariate_multipliers(
        gc_ovr if gc_ovr is not None else ref_gc,
        map_ovr if map_ovr is not None else ref_map,
        nmask_b,
        min_mappability,
        n_5k,
    )
    hit_y = ffy_align >= 0
    ffy_gc, ffy_map = np.ones(ffy_ref.num_rows), np.zeros(ffy_ref.num_rows)
    ffy_gc[hit_y] = gc_5k[ffy_align[hit_y]].astype(np.float64)
    ffy_map[hit_y] = map_5k[ffy_align[hit_y]].astype(np.float64)
    return gc_5k, map_5k, ffy_gc, ffy_map
