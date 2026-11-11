"""Pure numeric array binning and coordinate redistribution routines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

__all__ = [
    "bin_contig_fragments",
    "bin_counts",
    "bin_intervals",
    "build_uniform_fasta_grid",
    "rebin_counts",
    "rebin_counts_matrix",
]


def bin_intervals(
    length: int, bin_size: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return 0-based half-open bin edges, keeping a partial last bin."""
    if length <= 0:
        raise ValueError("contig length must be positive")
    if bin_size <= 0:
        raise ValueError("bin_size must be a positive integer")
    n_bins = (length + bin_size - 1) // bin_size
    starts = np.arange(n_bins, dtype=np.int64) * np.int64(bin_size)
    ends = np.minimum(starts + np.int64(bin_size), np.int64(length))
    return starts, ends


def bin_counts(
    starts: NDArray[np.int64], bin_size: int, n_bins: int
) -> NDArray[np.uint64]:
    """Count start positions into bins with a single numpy.bincount."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if starts.size == 0:
        return np.zeros(n_bins, dtype=np.uint64)
    indices = starts // bin_size
    valid = (indices >= 0) & (indices < n_bins)
    return np.bincount(indices[valid], minlength=n_bins).astype(np.uint64)


def rebin_counts(
    source_starts: NDArray[np.int64],
    source_ends: NDArray[np.int64],
    source_counts: NDArray[np.floating],
    target_starts: NDArray[np.int64],
    target_ends: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Redistribute 1D binned counts from source intervals to target intervals.

    Both intervals must partition the same contig length [0, L].
    Coarsening sums nested bins; refining splits counts uniformly by overlap length.
    """
    if target_starts.size == 0:
        return np.empty(0, dtype=np.float64)
    if source_starts.size == 0:
        return np.zeros(target_starts.size, dtype=np.float64)
    if source_ends[-1] != target_ends[-1]:
        raise ValueError(
            f"contig length mismatch: source ends at {source_ends[-1]}, "
            f"target ends at {target_ends[-1]}"
        )
    if (
        source_starts.size == target_starts.size
        and np.array_equal(source_starts, target_starts)
        and np.array_equal(source_ends, target_ends)
    ):
        return np.asarray(source_counts, dtype=np.float64).copy()

    source_knots = np.concatenate(([source_starts[0]], source_ends))
    source_cum = np.concatenate(([0.0], np.cumsum(source_counts, dtype=np.float64)))
    target_knots = np.concatenate(([target_starts[0]], target_ends))
    target_cum = np.interp(target_knots, source_knots, source_cum)
    target_counts = np.diff(target_cum)
    return np.maximum(target_counts, 0.0)


def _redistribute_2d_overlap(
    src_starts: NDArray[np.int64],
    src_ends: NDArray[np.int64],
    src_counts: NDArray[np.floating],
    tgt_starts: NDArray[np.int64],
    tgt_ends: NDArray[np.int64],
) -> NDArray[np.float64]:
    n_samples = src_counts.shape[1]
    out = np.zeros((tgt_starts.size, n_samples), dtype=np.float64)
    if src_starts.size == 0 or tgt_starts.size == 0:
        return out
    widths = np.maximum(src_ends - src_starts, 1).astype(np.float64)
    left_idx = np.searchsorted(tgt_ends, src_starts, side="right")
    right_idx = np.searchsorted(tgt_starts, src_ends, side="left")
    for s_idx in range(src_starts.size):
        lo = int(left_idx[s_idx])
        hi = int(right_idx[s_idx])
        if lo >= hi:
            continue
        ov = np.minimum(src_ends[s_idx], tgt_ends[lo:hi]) - np.maximum(
            src_starts[s_idx], tgt_starts[lo:hi]
        )
        pos = ov > 0
        if np.any(pos):
            frac = (ov[pos] / widths[s_idx])[:, np.newaxis]
            out[lo:hi][pos] += frac * src_counts[s_idx, np.newaxis, :]
    return out


def rebin_counts_matrix(
    counts: NDArray[np.floating],
    contigs: Sequence[str],
    starts: NDArray[np.int64],
    ends: NDArray[np.int64],
    contig_lengths: Mapping[str, int],
    source_bin_size: int,
    target_bin_size: int,
) -> tuple[
    NDArray[np.float64],
    tuple[str, ...],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    """Rescale a 2D binned count matrix across contigs to target_bin_size."""
    if target_bin_size <= 0:
        raise ValueError("target_bin_size must be a positive integer")
    n_samples = counts.shape[1]
    contig_arr = np.asarray(contigs, dtype=object)
    if contig_arr.size == 0:
        contig_slices: dict[str, tuple[int, int]] = {}
    else:
        changes = np.flatnonzero(contig_arr[1:] != contig_arr[:-1]) + 1
        s_idx = np.concatenate(([0], changes))
        e_idx = np.concatenate((changes, [contig_arr.size]))
        contig_slices = {
            str(contig_arr[int(s)]): (int(s), int(e))
            for s, e in zip(s_idx, e_idx, strict=True)
        }

    is_nested = (
        source_bin_size > 0
        and target_bin_size > source_bin_size
        and (target_bin_size % source_bin_size == 0)
    )
    k = target_bin_size // source_bin_size if is_nested else 0
    new_c: list[str] = []
    new_s: list[NDArray[np.int64]] = []
    new_e: list[NDArray[np.int64]] = []
    new_blocks: list[NDArray[np.float64]] = []

    for c_name, c_len in contig_lengths.items():
        t_starts, t_ends = bin_intervals(int(c_len), target_bin_size)
        span = contig_slices.get(c_name)
        if span is None or t_starts.size == 0:
            t_counts = np.zeros((t_starts.size, n_samples), dtype=np.float64)
        else:
            lo, hi = span
            src_s = starts[lo:hi]
            src_e = ends[lo:hi]
            block = np.asarray(counts[lo:hi, :], dtype=np.float64)
            is_part = (
                src_s.size > 0
                and src_s[0] == 0
                and src_e[-1] == t_ends[-1]
                and np.array_equal(src_s[1:], src_e[:-1])
            )
            if is_nested and is_part and ((src_s.size + k - 1) // k == t_starts.size):
                cut = np.arange(0, src_s.size, k, dtype=np.int64)
                t_counts = np.add.reduceat(block, cut, axis=0)
            elif is_part:
                t_counts = np.column_stack(
                    [
                        rebin_counts(src_s, src_e, block[:, j], t_starts, t_ends)
                        for j in range(n_samples)
                    ]
                )
            else:
                t_counts = _redistribute_2d_overlap(
                    src_s, src_e, block, t_starts, t_ends
                )
        new_c.extend([c_name] * t_starts.size)
        new_s.append(t_starts)
        new_e.append(t_ends)
        new_blocks.append(t_counts)

    out_counts = (
        np.vstack(new_blocks)
        if new_blocks
        else np.empty((0, n_samples), dtype=np.float64)
    )
    out_starts = np.concatenate(new_s) if new_s else np.empty(0, dtype=np.int64)
    out_ends = np.concatenate(new_e) if new_e else np.empty(0, dtype=np.int64)
    return out_counts, tuple(new_c), out_starts, out_ends


def bin_contig_fragments(
    midpoints: NDArray[np.int64],
    starts: NDArray[np.int64],
    ends: NDArray[np.int64],
    lengths: NDArray[np.uint16 | np.int64],
    gc_fracs: NDArray[np.floating],
    *,
    is_paired: bool,
    n_bins: int,
    bin_size: int,
) -> tuple[NDArray[np.uint64], pa.Array, NDArray[np.uint32], NDArray[np.float32]]:
    """Assemble fragment count, ListArray fragment lengths, endpoint count, and GC sums for a contig."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if bin_size <= 0:
        raise ValueError("bin_size must be a positive integer")

    if midpoints.size == 0:
        counts = np.zeros(n_bins, dtype=np.uint64)
        endpoint_counts = np.zeros(n_bins, dtype=np.uint32)
        frag_gc_sums = np.zeros(n_bins, dtype=np.float32)
        offsets = np.zeros(n_bins + 1, dtype=np.int32)
        empty_vals = np.empty(0, dtype=np.uint16)
        frag_arr = pa.ListArray.from_arrays(offsets, empty_vals)
        return counts, frag_arr, endpoint_counts, frag_gc_sums

    bin_indices = np.clip(midpoints // bin_size, 0, n_bins - 1).astype(np.int64)

    # 1. Fragment counts
    counts = np.bincount(bin_indices, minlength=n_bins).astype(np.uint64)

    # 2. Endpoints (both start and end)
    ep_s = np.clip(starts // bin_size, 0, n_bins - 1).astype(np.int64)
    ep_e = np.clip(ends // bin_size, 0, n_bins - 1).astype(np.int64)
    ep_combined = np.concatenate([ep_s, ep_e])
    endpoint_counts = np.bincount(ep_combined, minlength=n_bins).astype(np.uint32)

    # 3. Fragment GC sums
    frag_gc_sums = np.bincount(bin_indices, weights=gc_fracs, minlength=n_bins).astype(
        np.float32
    )

    # 4. Vectorized PyArrow ListArray
    if not is_paired:
        offsets = np.zeros(n_bins + 1, dtype=np.int32)
        empty_vals = np.empty(0, dtype=np.uint16)
        frag_arr = pa.ListArray.from_arrays(offsets, empty_vals)
    else:
        sort_order = np.argsort(bin_indices, kind="stable")
        sorted_lens = lengths[sort_order].astype(np.uint16)
        offsets = np.zeros(n_bins + 1, dtype=np.int32)
        np.cumsum(counts, out=offsets[1:])
        frag_arr = pa.ListArray.from_arrays(offsets, sorted_lens)

    return counts, frag_arr, endpoint_counts, frag_gc_sums


def build_uniform_fasta_grid(
    contigs_dict: Mapping[str, int],
    bin_size: int,
    n_samples: int,
) -> tuple[
    NDArray[np.float32],
    tuple[str, ...],
    NDArray[np.int64],
    NDArray[np.int64],
    dict[str, int],
]:
    """Construct uniform-depth Fortran-contiguous count matrix and coordinate grid."""
    names = list(contigs_dict.keys())
    lengths = [int(v) for v in contigs_dict.values()]
    contigs: list[str] = []
    starts: list[NDArray[np.int64]] = []
    ends: list[NDArray[np.int64]] = []
    for name, length in zip(names, lengths, strict=True):
        c_s, c_e = bin_intervals(length, bin_size)
        contigs.extend([name] * c_s.size)
        starts.append(c_s)
        ends.append(c_e)
    start_arr = np.concatenate(starts) if starts else np.empty(0, dtype=np.int64)
    end_arr = np.concatenate(ends) if ends else np.empty(0, dtype=np.int64)
    expected = np.maximum(end_arr - start_arr, 1).astype(np.float32)
    counts = np.asfortranarray(
        np.broadcast_to(expected[:, np.newaxis], (expected.size, n_samples)).copy()
    )
    return (
        counts,
        tuple(contigs),
        start_arr,
        end_arr,
        dict(zip(names, lengths, strict=True)),
    )
