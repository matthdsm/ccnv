"""Vectorized sample sketch coarsening across nested and non-nested bin grids."""

from __future__ import annotations

import json
import logging
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from numpy.typing import NDArray

from ccnv.models import (
    FRAGMENTOMICS_METADATA_KEY,
    METADATA_PREFIX,
    NUM_FRAG_BANDS,
    SAMPLE_SCHEMA,
    BinCompatibilityError,
    SampleBuffers,
    SampleMetadata,
    fixed_size_list_to_numpy,
    validate_sample_table,
)

logger = logging.getLogger(__name__)

_BIN_SIZE_KEY = f"{METADATA_PREFIX}bin_size".encode()

__all__ = [
    "rebin_sample_buffers",
    "rebin_sample_sketch",
]


def _col_numpy(table: pa.Table, name: str) -> np.ndarray:
    return table.column(name).combine_chunks().to_numpy(zero_copy_only=False)


def _coarsen_nested_sketch(
    table: pa.Table,
    meta: SampleMetadata,
    target_bin_size: int,
) -> pa.Table:
    """Coarsen integer-multiple bin grids across all 7 channels via np.add.reduceat."""
    factor = target_bin_size // meta.bin_size
    reduce_parts: list[NDArray[np.int64]] = []
    src_offset = 0
    for name in meta.contig_order:
        length = meta.contig_lengths[name]
        src_n = (length + meta.bin_size - 1) // meta.bin_size
        reduce_parts.append(src_offset + np.arange(0, src_n, factor, dtype=np.int64))
        src_offset += src_n

    reduce_idx = np.concatenate(reduce_parts)
    n_dst = int(reduce_idx.size)

    counts = np.add.reduceat(
        _col_numpy(table, "count").astype(np.uint64, copy=False), reduce_idx
    ).astype(np.uint32)
    paired = np.add.reduceat(
        _col_numpy(table, "paired_count").astype(np.uint64, copy=False), reduce_idx
    ).astype(np.uint32)
    flen_sum = np.add.reduceat(
        _col_numpy(table, "frag_len_sum").astype(np.uint64, copy=False), reduce_idx
    ).astype(np.uint64, copy=False)
    short_cnt = np.add.reduceat(
        _col_numpy(table, "short_frag_count").astype(np.uint64, copy=False),
        reduce_idx,
    ).astype(np.uint32)
    ep_cnt = np.add.reduceat(
        _col_numpy(table, "endpoint_count").astype(np.uint64, copy=False),
        reduce_idx,
    ).astype(np.uint32)
    gc_sum = np.add.reduceat(
        _col_numpy(table, "frag_gc_sum").astype(np.float64), reduce_idx
    ).astype(np.float32)

    raw_bands_col = table.column("frag_bands")
    if raw_bands_col.null_count == table.num_rows:
        bands_col: pa.Array = pa.nulls(
            n_dst, type=pa.list_(pa.uint32(), NUM_FRAG_BANDS)
        )
    else:
        bands_2d = fixed_size_list_to_numpy(
            raw_bands_col, NUM_FRAG_BANDS, np.uint32
        ).astype(np.uint64)
        dst_bands = np.add.reduceat(bands_2d, reduce_idx, axis=0).astype(np.uint32)
        bands_col = pa.FixedSizeListArray.from_arrays(
            pa.array(dst_bands.ravel(), type=pa.uint32()),
            NUM_FRAG_BANDS,
        )

    new_meta = dict(table.schema.metadata or {})
    new_meta[_BIN_SIZE_KEY] = json.dumps(target_bin_size).encode("utf-8")
    return pa.table(
        {
            "count": counts,
            "paired_count": paired,
            "frag_len_sum": flen_sum,
            "short_frag_count": short_cnt,
            "endpoint_count": ep_cnt,
            "frag_gc_sum": gc_sum,
            "frag_bands": bands_col,
        },
        schema=SAMPLE_SCHEMA.with_metadata(new_meta),
    )


def _coarsen_non_nested_sketch(
    table: pa.Table,
    meta: SampleMetadata,
    target_bin_size: int,
) -> pa.Table:
    """Redistribute counts across non-nested coarse bins and clear fragmentomic channels."""
    ep_sum = int(pc.sum(table.column("endpoint_count")).as_py() or 0)
    paired_sum = int(pc.sum(table.column("paired_count")).as_py() or 0)
    if meta.paired_end or paired_sum > 0 or ep_sum > 0:
        logger.warning(
            "Non-nested sample rebinning (%d bp -> %d bp) clears fragmentomic "
            "channels and ccnv.fragmentomics metadata.",
            meta.bin_size,
            target_bin_size,
        )

    src_counts = _col_numpy(table, "count")
    dst_parts: list[NDArray[np.uint32]] = []
    src_offset = 0
    for name in meta.contig_order:
        length = meta.contig_lengths[name]
        src_n = (length + meta.bin_size - 1) // meta.bin_size
        dst_n = (length + target_bin_size - 1) // target_bin_size
        c_slice = src_counts[src_offset : src_offset + src_n]
        src_offset += src_n

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
        rounded_cum = np.rint(np.interp(dst_ends, knots_x, knots_y)).astype(np.int64)
        dst_c = np.diff(np.concatenate((np.array([0], dtype=np.int64), rounded_cum)))
        dst_parts.append(dst_c.astype(np.uint32))

    dst_counts = np.concatenate(dst_parts)
    n_dst = int(dst_counts.size)

    new_meta = dict(table.schema.metadata or {})
    new_meta[_BIN_SIZE_KEY] = json.dumps(target_bin_size).encode("utf-8")
    new_meta.pop(FRAGMENTOMICS_METADATA_KEY, None)
    return pa.table(
        {
            "count": dst_counts,
            "paired_count": np.zeros(n_dst, dtype=np.uint32),
            "frag_len_sum": np.zeros(n_dst, dtype=np.uint64),
            "short_frag_count": np.zeros(n_dst, dtype=np.uint32),
            "endpoint_count": np.zeros(n_dst, dtype=np.uint32),
            "frag_gc_sum": np.zeros(n_dst, dtype=np.float32),
            "frag_bands": pa.nulls(n_dst, type=pa.list_(pa.uint32(), NUM_FRAG_BANDS)),
        },
        schema=SAMPLE_SCHEMA.with_metadata(new_meta),
    )


def rebin_sample_sketch(table: pa.Table, target_bin_size: int) -> pa.Table:
    """Coarsen a Schema v2.0.0 sample sketch table to ``target_bin_size``.

    Raises ``BinCompatibilityError`` when ``target_bin_size < source_bin_size``.
    """
    if (
        not isinstance(target_bin_size, int)
        or isinstance(target_bin_size, bool)
        or target_bin_size <= 0
    ):
        raise ValueError(
            f"target_bin_size must be a positive integer, got {target_bin_size!r}"
        )

    meta = validate_sample_table(table)
    if target_bin_size < meta.bin_size:
        raise BinCompatibilityError(
            f"cannot upsample sample sketch from {meta.bin_size} bp "
            f"to finer {target_bin_size} bp"
        )
    if target_bin_size == meta.bin_size:
        return table
    if target_bin_size % meta.bin_size == 0:
        return _coarsen_nested_sketch(table, meta, target_bin_size)
    return _coarsen_non_nested_sketch(table, meta, target_bin_size)


def rebin_sample_buffers(bufs: SampleBuffers, target_bin_size: int) -> SampleBuffers:
    """Coarsen in-memory NumPy SampleBuffers without building a PyArrow Table."""
    if (
        not isinstance(target_bin_size, int)
        or isinstance(target_bin_size, bool)
        or target_bin_size <= 0
    ):
        raise ValueError(
            f"target_bin_size must be a positive integer, got {target_bin_size!r}"
        )
    meta = bufs.meta
    if target_bin_size < meta.bin_size:
        raise BinCompatibilityError(
            f"cannot upsample sample sketch from {meta.bin_size} bp "
            f"to finer {target_bin_size} bp"
        )
    if target_bin_size == meta.bin_size:
        return bufs

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
        return SampleBuffers(
            meta=replace(meta, bin_size=target_bin_size),
            count=np.add.reduceat(
                np.asarray(bufs.count, dtype=np.uint64), r_idx
            ).astype(np.uint32),
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
    dst_parts: list[NDArray[np.uint32]] = []
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
        rounded_cum = np.rint(np.interp(dst_ends, knots_x, knots_y)).astype(np.int64)
        dst_c = np.diff(np.concatenate((np.array([0], dtype=np.int64), rounded_cum)))
        dst_parts.append(dst_c.astype(np.uint32))

    dst_counts = np.concatenate(dst_parts)
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
