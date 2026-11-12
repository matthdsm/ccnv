"""Contiguous 1 Mb super-block partitioning and DESeq2 median-of-ratios depth scaling."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ccnv.models.exceptions import InvalidInputError

__all__ = [
    "estimate_superblock_scales",
    "partition_contiguous_superblocks",
]


def _partition_contig_runs(
    c_ids: NDArray[np.int64],
    bins_per_block: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Partition contiguous contig runs into [start, end) index slices."""
    n = int(c_ids.size)
    if n == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    k = max(1, int(bins_per_block))
    half_k = k // 2
    run_cuts = np.concatenate(([0], np.flatnonzero(c_ids[1:] != c_ids[:-1]) + 1, [n]))
    starts_list: list[int] = []
    for r_idx in range(run_cuts.size - 1):
        r0 = int(run_cuts[r_idx])
        r1 = int(run_cuts[r_idx + 1])
        cuts = [r0] if r1 - r0 <= k else list(range(r0, r1, k))
        if len(cuts) > 1 and (r1 - cuts[-1]) < half_k:
            cuts.pop()
        starts_list.extend(cuts)

    b_starts = np.asarray(starts_list, dtype=np.int64)
    b_ends = np.concatenate((b_starts[1:], np.asarray([n], dtype=np.int64)))
    return b_starts, b_ends


def _sum_superblocks(
    counts_2d: NDArray[np.floating] | NDArray[np.integer],
    u_idx: NDArray[np.int64] | None,
    b_starts: NDArray[np.int64],
    b_ends: NDArray[np.int64],
    chunk_rows: int,
) -> NDArray[np.float64]:
    """Accumulate per-block sample count sums in bounded memory chunks."""
    block_sums = np.empty((b_starts.size, counts_2d.shape[1]), dtype=np.float64)
    b0 = 0
    while b0 < b_starts.size:
        b1 = b0 + 1
        while b1 < b_starts.size and int(b_ends[b1 - 1] - b_starts[b0]) < chunk_rows:
            b1 += 1
        u0, u1 = int(b_starts[b0]), int(b_ends[b1 - 1])
        rows = slice(u0, u1) if u_idx is None else u_idx[u0:u1]
        sub = np.asarray(counts_2d[rows], dtype=np.float64)
        block_sums[b0:b1] = np.add.reduceat(sub, b_starts[b0:b1] - u0, axis=0)
        b0 = b1
    return block_sums


def _align_usable_1d(
    raw: ArrayLike, n_bins: int, n_u: int, u_idx: NDArray[np.int64] | None, name: str
) -> NDArray[np.float64]:
    """Slice a 1D array matching either full genome bins or pre-filtered usable bins."""
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 1:
        raise InvalidInputError(f"{name} must be a 1D array")
    if arr.size == n_u:
        return arr
    if u_idx is not None and arr.size == n_bins:
        return arr[u_idx]
    raise InvalidInputError(f"{name} length must match counts or usable bins")


def partition_contiguous_superblocks(
    contig_ids: ArrayLike,
    *,
    bin_size: int = 5_000,
    block_bp: int = 1_000_000,
    block_bins: int | None = None,
    min_blocks: int = 1,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Partition ordered contig codes into contiguous [start, end) super-block offsets."""
    c_ids = np.asarray(contig_ids, dtype=np.int64)
    if c_ids.ndim != 1:
        raise InvalidInputError("contig_ids must be a 1D integer array")

    if block_bins is not None:
        if (
            not isinstance(block_bins, int)
            or isinstance(block_bins, bool)
            or block_bins <= 0
        ):
            raise InvalidInputError("block_bins must be a positive integer")
        k = block_bins
    else:
        if not isinstance(bin_size, int) or isinstance(bin_size, bool) or bin_size <= 0:
            raise InvalidInputError("bin_size must be a positive integer")
        if not isinstance(block_bp, int) or isinstance(block_bp, bool) or block_bp <= 0:
            raise InvalidInputError("block_bp must be a positive integer")
        k = max(10, round(max(block_bp, 10 * bin_size) / bin_size))

    b_starts, b_ends = _partition_contig_runs(c_ids, k)
    if min_blocks > 1 and 1 < c_ids.size and b_starts.size < min_blocks:
        fallback_k = max(1, int(c_ids.size) // max(2 * min_blocks, 8))
        b_starts, b_ends = _partition_contig_runs(c_ids, fallback_k)
    return b_starts, b_ends


def estimate_superblock_scales(
    counts: NDArray[np.floating] | NDArray[np.integer],
    contig_ids: ArrayLike,
    *,
    usable: NDArray[np.bool_] | None = None,
    bin_size: int = 5_000,
    block_bp: int = 1_000_000,
    reference: ArrayLike | None = None,
    min_blocks: int | None = None,
) -> NDArray[np.float64]:
    """Estimate DESeq2 median-of-ratios scales over contiguous 1 Mb super-blocks.

    When ``reference`` is ``None``, ``counts`` must be a non-empty 2D cohort matrix
    and returned size factors are median-normalized to 1.0 across samples. When
    ``reference`` is provided as a 1D baseline vector, ``counts`` may be 1D or 2D
    and unnormalized sample-to-reference depth scales are returned.
    """
    counts_arr = counts if isinstance(counts, np.ndarray) else np.asarray(counts)
    if reference is None:
        if counts_arr.ndim != 2 or 0 in counts_arr.shape:
            raise InvalidInputError("Counts matrix must be a non-empty 2D array.")
        counts_2d = counts_arr
    elif counts_arr.ndim == 1:
        counts_2d = counts_arr[:, np.newaxis]
    elif counts_arr.ndim == 2:
        counts_2d = counts_arr
    else:
        raise InvalidInputError("counts must be a 1D or 2D array")

    n_bins, n_samples = counts_2d.shape
    chunk_rows = max(1024, (64 * 1024 * 1024) // max(n_samples, 1))
    if reference is None:
        for r0 in range(0, n_bins, chunk_rows):
            if np.any(counts_2d[r0 : r0 + chunk_rows] < 0):
                raise InvalidInputError("Counts matrix cannot contain negative values.")

    if usable is not None:
        mask = np.asarray(usable, dtype=np.bool_)
        if mask.ndim != 1 or mask.shape[0] != n_bins:
            raise InvalidInputError("usable length must match counts row count")
        u_idx: NDArray[np.int64] | None = np.flatnonzero(mask)
        n_u = int(u_idx.size)
        if reference is None and n_u == 0:
            raise InvalidInputError("No usable bins remaining after mask.")
    else:
        u_idx = None
        n_u = n_bins

    c_ids_u = _align_usable_1d(contig_ids, n_bins, n_u, u_idx, "contig_ids").astype(
        np.int64
    )
    default_min_b = 4 if reference is not None else 1
    ref_u: NDArray[np.float64] | None = None
    if reference is not None:
        ref_u = _align_usable_1d(reference, n_bins, n_u, u_idx, "reference")
        if n_u == 0 or n_samples == 0:
            return np.ones(n_samples, dtype=np.float64)

    b_starts, b_ends = partition_contiguous_superblocks(
        c_ids_u,
        bin_size=bin_size,
        block_bp=block_bp,
        min_blocks=default_min_b if min_blocks is None else max(1, min_blocks),
    )
    block_sums = _sum_superblocks(counts_2d, u_idx, b_starts, b_ends, chunk_rows)

    if ref_u is not None:
        denom_blocks = np.add.reduceat(ref_u, b_starts)
        tot_denom = float(np.sum(ref_u))
        scales = np.ones(n_samples, dtype=np.float64)
        denom_ok = np.isfinite(denom_blocks) & (denom_blocks > 0.0)
        for j in range(n_samples):
            num_j = block_sums[:, j]
            valid_j = denom_ok & np.isfinite(num_j) & (num_j > 0.0)
            med_j = (
                float(np.median(num_j[valid_j] / denom_blocks[valid_j]))
                if int(np.count_nonzero(valid_j)) >= 2
                else 0.0
            )
            if math.isfinite(med_j) and med_j > 0.0:
                scales[j] = med_j
                continue
            tot_num = float(np.sum(num_j))
            if (
                math.isfinite(tot_denom)
                and tot_denom > 0.0
                and math.isfinite(tot_num)
                and tot_num > 0.0
            ):
                scales[j] = tot_num / tot_denom
        return scales

    totals = block_sums.sum(axis=0)
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0.0):
        suffix = " on usable bins" if usable is not None else ""
        raise InvalidInputError(
            f"Total read count is zero for at least one sample{suffix}."
        )
    ref_block = block_sums.mean(axis=1, keepdims=True)
    valid = ref_block[:, 0] > 0.0
    raw_sf = np.median(block_sums[valid] / ref_block[valid], axis=0)
    total_ref = float(np.sum(ref_block[valid]))
    fallback = totals / total_ref if total_ref > 0.0 else np.ones_like(totals)
    sparse_two = int(np.count_nonzero(valid)) <= 2 and bool(
        np.any(block_sums[valid] == 0.0)
    )
    raw_sf = (
        fallback
        if sparse_two
        else np.where(~np.isfinite(raw_sf) | (raw_sf <= 0.0), fallback, raw_sf)
    )
    med = float(np.median(raw_sf))
    if not math.isfinite(med) or med <= 0.0:
        raw_sf = fallback
        med = float(np.median(raw_sf))
    return raw_sf / med
