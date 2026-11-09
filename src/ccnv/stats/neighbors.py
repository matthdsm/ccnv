"""Batched nearest-neighbor selection and null model computation."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics.pairwise import euclidean_distances

logger = logging.getLogger(__name__)

_MAX_NULL_SAMPLES: Final[int] = 100

__all__ = [
    "compute_null_models",
    "distance_cutoff",
    "select_nearest_bins",
]


@dataclass(frozen=True, slots=True)
class _NearestContext:
    data: NDArray[np.float64]
    norm: NDArray[np.float64]
    is_contiguous: bool
    ranges: dict[int, tuple[int, int]]
    cids: NDArray[np.integer]
    k: int
    batch_size: int


@dataclass(frozen=True, slots=True)
class _NullContext:
    corrected: NDArray[np.float64]
    nearest_indices: list[list[int]]
    n_null: int


_worker_nearest_ctx: _NearestContext | None = None
_worker_null_ctx: _NullContext | None = None


def _init_nearest_worker(ctx: _NearestContext) -> None:
    global _worker_nearest_ctx
    _worker_nearest_ctx = ctx


def _run_nearest_chunk(
    ctx: _NearestContext,
    chunk: tuple[int, int],
) -> tuple[int, list[list[int]], list[list[float]]]:
    start, stop = chunk
    n_bins = ctx.data.shape[0]
    out_indices: list[list[int]] = []
    out_distances: list[list[float]] = []

    for b_start in range(start, stop, ctx.batch_size):
        b_stop = min(b_start + ctx.batch_size, stop)
        batch = ctx.data[b_start:b_stop]
        dist = euclidean_distances(
            batch, ctx.data, squared=True, Y_norm_squared=ctx.norm
        )

        for offset, query in enumerate(range(b_start, b_stop)):
            row = dist[offset]
            cid = int(ctx.cids[query])
            if ctx.is_contiguous:
                c_start, c_end = ctx.ranges[cid]
                row[c_start:c_end] = np.inf
                available = n_bins - (c_end - c_start)
            else:
                row[ctx.cids == cid] = np.inf
                available = int(np.count_nonzero(np.isfinite(row)))

            if available == 0:
                out_indices.append([])
                out_distances.append([])
                continue

            take = min(ctx.k, available)
            partitioned = np.argpartition(row, take - 1)[:take]
            ordered = partitioned[np.argsort(row[partitioned], kind="mergesort")]
            out_indices.append([int(idx) for idx in ordered])
            out_distances.append([float(row[idx]) for idx in ordered])

    return start, out_indices, out_distances


def _chunk_nearest_worker(
    chunk: tuple[int, int],
) -> tuple[int, list[list[int]], list[list[float]]]:
    assert _worker_nearest_ctx is not None
    return _run_nearest_chunk(_worker_nearest_ctx, chunk)


def select_nearest_bins(
    profiles: NDArray[np.floating],
    contig_ids: NDArray[np.integer],
    k: int,
    max_batch_memory_mb: int,
    threads: int = 1,
) -> tuple[list[list[int]], list[list[float]]]:
    """Pick k nearest off-contig bins with batched distances and argpartition."""
    data = np.asarray(profiles, dtype=np.float64)
    contig_ids = np.asarray(contig_ids)
    n_bins = data.shape[0]
    if k < 1:
        raise ValueError("nearest_bins must be a positive integer")
    if n_bins == 0:
        return [], []

    max_bytes = max(int(max_batch_memory_mb * 1024 * 1024), 1)
    bytes_per_query = max(n_bins, 1) * data.dtype.itemsize
    batch_size = max(1, max_bytes // bytes_per_query)
    candidate_norm = np.sum(data * data, axis=1)

    # Check if contig_ids form contiguous blocks for fast range masking
    changes = np.flatnonzero(contig_ids[1:] != contig_ids[:-1]) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(contig_ids)]))
    seen_cids: set[int] = set()
    is_contiguous = True
    contig_ranges: dict[int, tuple[int, int]] = {}
    for s, e in zip(starts, ends, strict=True):
        cid = int(contig_ids[s])
        if cid in seen_cids:
            is_contiguous = False
            break
        seen_cids.add(cid)
        contig_ranges[cid] = (int(s), int(e))

    ctx = _NearestContext(
        data=data,
        norm=candidate_norm,
        is_contiguous=is_contiguous,
        ranges=contig_ranges,
        cids=contig_ids,
        k=k,
        batch_size=batch_size,
    )

    if threads > 1 and n_bins > 2000:
        n_workers = min(threads, max(1, n_bins // 1000))
        chunk_step = (n_bins + n_workers - 1) // n_workers
        chunks = [
            (i * chunk_step, min((i + 1) * chunk_step, n_bins))
            for i in range(n_workers)
            if i * chunk_step < n_bins
        ]
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_nearest_worker,
            initargs=(ctx,),
        ) as executor:
            chunk_results = list(executor.map(_chunk_nearest_worker, chunks))

        chunk_results.sort(key=lambda r: r[0])
        indices: list[list[int]] = []
        distances: list[list[float]] = []
        for _, chunk_idx, chunk_dst in chunk_results:
            indices.extend(chunk_idx)
            distances.extend(chunk_dst)
        return indices, distances

    # Sequential execution: direct call, no globals mutated
    _, indices, distances = _run_nearest_chunk(ctx, (0, n_bins))
    return indices, distances


def _init_null_worker(ctx: _NullContext) -> None:
    global _worker_null_ctx
    _worker_null_ctx = ctx


def _run_null_chunk(
    ctx: _NullContext,
    chunk: tuple[int, int],
) -> tuple[int, NDArray[np.float64], NDArray[np.float64]]:
    start, stop = chunk
    out_mean = np.zeros(stop - start, dtype=np.float64)
    out_std = np.zeros(stop - start, dtype=np.float64)

    for offset, bin_index in enumerate(range(start, stop)):
        neighbors = ctx.nearest_indices[bin_index]
        if not neighbors:
            continue
        reference = np.median(ctx.corrected[neighbors, : ctx.n_null], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.log2(ctx.corrected[bin_index, : ctx.n_null] / reference)
        values_clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        out_mean[offset] = float(np.mean(values_clean))
        s = float(np.std(values_clean, ddof=1)) if values_clean.size > 1 else 0.0
        out_std[offset] = s if np.isfinite(s) and s > 1e-8 else 0.0

    return start, out_mean, out_std


def _chunk_null_worker(
    chunk: tuple[int, int],
) -> tuple[int, NDArray[np.float64], NDArray[np.float64]]:
    assert _worker_null_ctx is not None
    return _run_null_chunk(_worker_null_ctx, chunk)


def compute_null_models(
    corrected: NDArray[np.float64],
    nearest_indices: list[list[int]],
    threads: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute per-bin reference medians and log2-ratio null statistics."""
    n_bins, n_samples = corrected.shape
    n_null = min(n_samples, _MAX_NULL_SAMPLES)
    if n_null == 0 or n_bins == 0:
        return np.zeros(n_bins, dtype=np.float64), np.zeros(n_bins, dtype=np.float64)

    ctx = _NullContext(
        corrected=corrected,
        nearest_indices=nearest_indices,
        n_null=n_null,
    )

    if threads > 1 and n_bins > 5000:
        n_workers = min(threads, max(1, n_bins // 2000))
        chunk_step = (n_bins + n_workers - 1) // n_workers
        chunks = [
            (i * chunk_step, min((i + 1) * chunk_step, n_bins))
            for i in range(n_workers)
            if i * chunk_step < n_bins
        ]
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_null_worker,
            initargs=(ctx,),
        ) as executor:
            chunk_results = list(executor.map(_chunk_null_worker, chunks))

        chunk_results.sort(key=lambda r: r[0])
        null_mean = np.concatenate([r[1] for r in chunk_results])
        null_std = np.concatenate([r[2] for r in chunk_results])
        return null_mean, null_std

    # Sequential execution: direct call, no globals mutated
    _, null_mean, null_std = _run_null_chunk(ctx, (0, n_bins))
    return null_mean, null_std


def distance_cutoff(
    distances: (
        tuple[NDArray[np.float64], ...]
        | NDArray[np.floating]
        | Sequence[Sequence[float]]
    ),
    repeats: int = 3,
) -> float:
    """Return an iterative three-sigma cutoff for nearest-bin distances."""
    if isinstance(distances, np.ndarray):
        values = distances.ravel()
    elif (
        isinstance(distances, tuple)
        and distances
        and isinstance(distances[0], np.ndarray)
    ):
        values = np.concatenate(distances)
    else:
        values = np.asarray([d for sub in distances for d in sub], dtype=np.float64)

    values = values[np.isfinite(values) & (values >= 0)]
    if values.size == 0:
        return float("inf")
    cutoff = float("inf")
    for _ in range(repeats):
        selected = values[values < cutoff]
        if selected.size == 0:
            break
        cutoff = float(np.mean(selected) + 3.0 * np.std(selected))
    return cutoff
