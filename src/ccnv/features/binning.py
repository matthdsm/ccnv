"""Pure numeric array binning and coordinate redistribution routines."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "bin_counts",
    "bin_intervals",
    "rebin_counts",
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
