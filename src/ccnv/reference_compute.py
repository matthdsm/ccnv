"""High-performance compute routines for reference model construction."""

import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Final

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_MAX_NULL_SAMPLES: Final[int] = 100

# Worker-local global state for nearest-bin search to avoid IPC serialization
_w_data: NDArray[np.float64] | None = None
_w_norm: NDArray[np.float64] | None = None
_w_is_contiguous: bool = False
_w_ranges: dict[int, tuple[int, int]] | None = None
_w_cids: NDArray[np.integer] | None = None
_w_k: int = 0
_w_batch_size: int = 0

# Worker-local global state for null model computation
_w_null_corrected: NDArray[np.float64] | None = None
_w_null_indices: list[list[int]] | None = None
_w_null_samples: int = 0


def _init_nearest_worker(
    data: NDArray[np.float64],
    norm: NDArray[np.float64],
    is_contiguous: bool,
    ranges: dict[int, tuple[int, int]],
    cids: NDArray[np.integer],
    k: int,
    batch_size: int,
) -> None:
    global _w_data, _w_norm, _w_is_contiguous, _w_ranges, _w_cids, _w_k, _w_batch_size
    _w_data = data
    _w_norm = norm
    _w_is_contiguous = is_contiguous
    _w_ranges = ranges
    _w_cids = cids
    _w_k = k
    _w_batch_size = batch_size


def _chunk_nearest_worker(
    chunk: tuple[int, int],
) -> tuple[int, list[list[int]], list[list[float]]]:
    start, stop = chunk
    assert _w_data is not None
    assert _w_norm is not None
    assert _w_cids is not None
    assert _w_ranges is not None

    n_bins = _w_data.shape[0]
    out_indices: list[list[int]] = []
    out_distances: list[list[float]] = []

    for b_start in range(start, stop, _w_batch_size):
        b_stop = min(b_start + _w_batch_size, stop)
        batch = _w_data[b_start:b_stop]
        query_norm = np.sum(batch * batch, axis=1, keepdims=True)
        dist = query_norm + _w_norm - 2.0 * (batch @ _w_data.T)
        np.maximum(dist, 0.0, out=dist)

        for offset, query in enumerate(range(b_start, b_stop)):
            row = dist[offset]
            cid = int(_w_cids[query])
            if _w_is_contiguous:
                c_start, c_end = _w_ranges[cid]
                row[c_start:c_end] = np.inf
                available = n_bins - (c_end - c_start)
            else:
                row[_w_cids == cid] = np.inf
                available = int(np.count_nonzero(np.isfinite(row)))

            if available == 0:
                out_indices.append([])
                out_distances.append([])
                continue

            take = min(_w_k, available)
            partitioned = np.argpartition(row, take - 1)[:take]
            ordered = partitioned[np.argsort(row[partitioned], kind="mergesort")]
            out_indices.append([int(idx) for idx in ordered])
            out_distances.append([float(row[idx]) for idx in ordered])

    return start, out_indices, out_distances


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
            initargs=(
                data,
                candidate_norm,
                is_contiguous,
                contig_ranges,
                contig_ids,
                k,
                batch_size,
            ),
        ) as executor:
            chunk_results = list(executor.map(_chunk_nearest_worker, chunks))

        chunk_results.sort(key=lambda r: r[0])
        indices: list[list[int]] = []
        distances: list[list[float]] = []
        for _, chunk_idx, chunk_dst in chunk_results:
            indices.extend(chunk_idx)
            distances.extend(chunk_dst)
        return indices, distances

    # Sequential execution
    _init_nearest_worker(
        data,
        candidate_norm,
        is_contiguous,
        contig_ranges,
        contig_ids,
        k,
        batch_size,
    )
    _, indices, distances = _chunk_nearest_worker((0, n_bins))
    return indices, distances


def _init_null_worker(
    corrected: NDArray[np.float64],
    nearest_indices: list[list[int]],
    n_null: int,
) -> None:
    global _w_null_corrected, _w_null_indices, _w_null_samples
    _w_null_corrected = corrected
    _w_null_indices = nearest_indices
    _w_null_samples = n_null


def _chunk_null_worker(
    chunk: tuple[int, int],
) -> tuple[int, NDArray[np.float64], NDArray[np.float64]]:
    start, stop = chunk
    assert _w_null_corrected is not None
    assert _w_null_indices is not None

    out_mean = np.zeros(stop - start, dtype=np.float64)
    out_std = np.zeros(stop - start, dtype=np.float64)

    for offset, bin_index in enumerate(range(start, stop)):
        neighbors = _w_null_indices[bin_index]
        if not neighbors:
            continue
        reference = np.median(_w_null_corrected[neighbors, :_w_null_samples], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.log2(_w_null_corrected[bin_index, :_w_null_samples] / reference)
        values_clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        out_mean[offset] = float(np.mean(values_clean))
        s = float(np.std(values_clean, ddof=1)) if values_clean.size > 1 else 0.0
        out_std[offset] = s if np.isfinite(s) and s > 0 else 0.0

    return start, out_mean, out_std


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
            initargs=(corrected, nearest_indices, n_null),
        ) as executor:
            chunk_results = list(executor.map(_chunk_null_worker, chunks))

        chunk_results.sort(key=lambda r: r[0])
        null_mean = np.concatenate([r[1] for r in chunk_results])
        null_std = np.concatenate([r[2] for r in chunk_results])
        return null_mean, null_std

    _init_null_worker(corrected, nearest_indices, n_null)
    _, null_mean, null_std = _chunk_null_worker((0, n_bins))
    return null_mean, null_std
