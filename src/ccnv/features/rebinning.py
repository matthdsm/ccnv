"""Vectorized sample sketch coarsening across nested and non-nested bin grids."""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from ccnv.models import (
    BinCompatibilityError,
    InvalidInputError,
    SampleBuffers,
)

logger = logging.getLogger(__name__)

__all__ = [
    "rebin_sample_buffers",
]


def _validate_target_bin_size(target_bin_size: int) -> None:
    if (
        not isinstance(target_bin_size, int)
        or isinstance(target_bin_size, bool)
        or target_bin_size <= 0
    ):
        raise InvalidInputError(
            f"target_bin_size must be a positive integer, got {target_bin_size!r}"
        )


def rebin_sample_buffers(bufs: SampleBuffers, target_bin_size: int) -> SampleBuffers:
    """Coarsen in-memory NumPy SampleBuffers without building a PyArrow Table."""
    _validate_target_bin_size(target_bin_size)
    meta = bufs.meta
    if target_bin_size < meta.bin_size:
        raise BinCompatibilityError(
            f"cannot upsample sample sketch from {meta.bin_size} bp "
            f"to finer {target_bin_size} bp"
        )
    if target_bin_size == meta.bin_size:
        return bufs

    is_float_count = bool(np.issubdtype(bufs.count.dtype, np.floating))
    if target_bin_size % meta.bin_size == 0:
        factor = target_bin_size // meta.bin_size
        parts: list[NDArray[np.int64]] = []
        off = 0
        for name in meta.contig_order:
            src_n = (meta.contig_lengths[name] + meta.bin_size - 1) // meta.bin_size
            parts.append(off + np.arange(0, src_n, factor, dtype=np.int64))
            off += src_n
        r_idx = np.concatenate(parts)
        dst_bands = (
            np.add.reduceat(
                bufs.frag_bands.astype(np.uint64, copy=False), r_idx, axis=0
            ).astype(np.uint32)
            if bufs.frag_bands is not None
            else None
        )
        dst_count: NDArray[np.uint32] | NDArray[np.float64] = (
            np.add.reduceat(np.asarray(bufs.count, dtype=np.float64), r_idx)
            if is_float_count
            else np.add.reduceat(np.asarray(bufs.count, dtype=np.uint64), r_idx).astype(
                np.uint32
            )
        )
        return SampleBuffers(
            meta=replace(meta, bin_size=target_bin_size),
            count=dst_count,
            paired_count=np.add.reduceat(
                bufs.paired_count.astype(np.uint64, copy=False), r_idx
            ).astype(np.uint32),
            frag_len_sum=np.add.reduceat(bufs.frag_len_sum, r_idx).astype(
                np.uint64, copy=False
            ),
            short_frag_count=np.add.reduceat(
                bufs.short_frag_count.astype(np.uint64, copy=False), r_idx
            ).astype(np.uint32),
            endpoint_count=np.add.reduceat(
                bufs.endpoint_count.astype(np.uint64, copy=False), r_idx
            ).astype(np.uint32),
            frag_gc_sum=np.add.reduceat(
                bufs.frag_gc_sum.astype(np.float64), r_idx
            ).astype(np.float32),
            frag_bands=dst_bands,
        )

    if (
        meta.paired_end
        or np.any(bufs.paired_count > 0)
        or np.any(bufs.endpoint_count > 0)
    ):
        logger.warning(
            "Non-nested sample rebinning (%d bp -> %d bp) clears fragmentomic "
            "channels and ccnv.fragmentomics metadata.",
            meta.bin_size,
            target_bin_size,
        )
    dst_parts: list[NDArray[np.uint32] | NDArray[np.float64]] = []
    off = 0
    for name in meta.contig_order:
        length = meta.contig_lengths[name]
        src_n = (length + meta.bin_size - 1) // meta.bin_size
        dst_n = (length + target_bin_size - 1) // target_bin_size
        c_slice = bufs.count[off : off + src_n]
        off += src_n
        src_ends = np.minimum(
            np.arange(1, src_n + 1, dtype=np.int64) * meta.bin_size, length
        ).astype(np.float64)
        dst_ends = np.minimum(
            np.arange(1, dst_n + 1, dtype=np.int64) * target_bin_size, length
        ).astype(np.float64)
        knots_x = np.concatenate((np.array([0.0]), src_ends))
        knots_y = np.concatenate(
            (np.array([0.0]), np.cumsum(c_slice, dtype=np.float64))
        )
        interp_cum = np.interp(dst_ends, knots_x, knots_y)
        if is_float_count:
            dst_c_f = np.maximum(
                0.0, np.diff(np.concatenate((np.array([0.0]), interp_cum)))
            )
            dst_parts.append(dst_c_f)
        else:
            rounded_cum = np.rint(interp_cum).astype(np.int64)
            dst_c = np.diff(
                np.concatenate((np.array([0], dtype=np.int64), rounded_cum))
            )
            dst_parts.append(dst_c.astype(np.uint32))

    dst_counts: NDArray[np.uint32] | NDArray[np.float64] = np.concatenate(dst_parts)
    n_dst = int(dst_counts.size)
    return SampleBuffers(
        meta=replace(meta, bin_size=target_bin_size, fragmentomics=None),
        count=dst_counts,
        paired_count=np.zeros(n_dst, dtype=np.uint32),
        frag_len_sum=np.zeros(n_dst, dtype=np.uint64),
        short_frag_count=np.zeros(n_dst, dtype=np.uint32),
        endpoint_count=np.zeros(n_dst, dtype=np.uint32),
        frag_gc_sum=np.zeros(n_dst, dtype=np.float32),
        frag_bands=None,
    )
