"""Pure numeric array binning and coordinate redistribution routines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ccnv.models.exceptions import InvalidInputError

__all__ = [
    "bin_intervals",
    "build_coordinate_grid",
    "compute_bin_exposure",
    "rebin_counts",
    "rebin_counts_matrix",
]


def compute_bin_exposure(
    starts: NDArray[np.integer],
    ends: NDArray[np.integer],
    bin_size: int,
) -> NDArray[np.float32]:
    """Compute geometric bin-width exposure a_i = (end_i - start_i) / bin_size."""
    if not isinstance(bin_size, int) or isinstance(bin_size, bool) or bin_size <= 0:
        raise InvalidInputError("bin_size must be a positive integer")
    s_arr = np.asarray(starts, dtype=np.int64)
    e_arr = np.asarray(ends, dtype=np.int64)
    if s_arr.ndim != 1 or s_arr.shape != e_arr.shape:
        raise InvalidInputError("starts and ends must be aligned 1D arrays")
    widths = e_arr - s_arr
    if np.any(widths < 0):
        raise InvalidInputError("bin widths cannot be negative")
    return (widths.astype(np.float64) / float(bin_size)).astype(np.float32)


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


def build_coordinate_grid(
    contigs_dict: Mapping[str, int],
    bin_size: int,
) -> tuple[tuple[str, ...], NDArray[np.int64], NDArray[np.int64], dict[str, int]]:
    """Construct genomic bin coordinate grid across ordered contigs."""
    contigs: list[str] = []
    starts: list[NDArray[np.int64]] = []
    ends: list[NDArray[np.int64]] = []
    contig_lengths: dict[str, int] = {}
    for name, raw_len in contigs_dict.items():
        length = int(raw_len)
        c_s, c_e = bin_intervals(length, bin_size)
        contigs.extend([name] * c_s.size)
        starts.append(c_s)
        ends.append(c_e)
        contig_lengths[name] = length
    start_arr = np.concatenate(starts) if starts else np.empty(0, dtype=np.int64)
    end_arr = np.concatenate(ends) if ends else np.empty(0, dtype=np.int64)
    return tuple(contigs), start_arr, end_arr, contig_lengths


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
        if span is None or t_starts.size == 0 or n_samples == 0:
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
            if not is_part:
                raise ValueError(
                    f"source intervals on {c_name!r} do not partition [0, {c_len}]"
                )
            if is_nested and ((src_s.size + k - 1) // k == t_starts.size):
                cut = np.arange(0, src_s.size, k, dtype=np.int64)
                t_counts = np.add.reduceat(block, cut, axis=0)
            else:
                t_counts = np.column_stack(
                    [
                        rebin_counts(src_s, src_e, block[:, j], t_starts, t_ends)
                        for j in range(n_samples)
                    ]
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
