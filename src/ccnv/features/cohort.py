"""Cohort sample aggregation, streaming assembly, depth normalization, and sex inference."""

from __future__ import annotations

import logging
import math
import os
import tempfile
import weakref
from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from ccnv.features.binning import (
    build_coordinate_grid,
    compute_bin_exposure,
    rebin_counts_matrix,
)
from ccnv.features.rebinning import rebin_sample_buffers
from ccnv.features.sex import (
    SexInference,
    SexLabel,
    infer_germline_sex_2d,
    infer_sample_sexes,
    nonpar_contig_mask,
)
from ccnv.models import (
    NUM_FRAG_BANDS,
    BinCompatibilityError,
    InvalidInputError,
    ReferenceError,
    SampleBuffers,
    SampleMetadata,
    SamplesheetEntry,
    SchemaError,
    decode_metadata,
    reconstruct_sample_grid,
)
from ccnv.stats.fragmentomics import evaluate_band_weighted_counts

logger = logging.getLogger(__name__)
_MASK_FRACTION: Final[float] = 0.05
_DEFAULT_FRAG_LEN_BP: Final[float] = 167.0

__all__ = [
    "Cohort",
    "InvalidInputError",
    "SexInference",
    "SexLabel",
    "build_fisher_weighted_cohort",
    "cohort_from_fasta",
    "compute_bin_exposure",
    "estimate_size_factors",
    "estimate_superblock_size_factors",
    "infer_germline_sex_2d",
    "infer_sample_sexes",
    "load_and_assemble_cohort",
    "mask_low_coverage_per_contig",
    "nonpar_contig_mask",
    "rebin_cohort",
    "validate_cohort",
]


@dataclass(frozen=True, slots=True)
class Cohort:
    """Aligned binned counts and fragmentomic summaries across cohort samples."""

    counts: NDArray[np.float32] | NDArray[np.float64]
    contigs: tuple[str, ...]
    starts: NDArray[np.int64]
    ends: NDArray[np.int64]
    contig_lengths: dict[str, int]
    bin_size: int
    identities: tuple[object, ...]
    paths: tuple[Path, ...]
    sample_metadata: tuple[dict[str, object], ...] = ()
    ref_frag_len: NDArray[np.float32] | None = None
    ref_ep_rate: NDArray[np.float32] | None = None
    in_memory_tables: tuple[pa.Table, ...] = ()


@dataclass(frozen=True, slots=True)
class _StreamedSample:
    index: int
    path: Path
    raw_meta: dict[str, object]
    sample_meta: SampleMetadata
    is_paired: bool
    count: NDArray[np.uint32]
    paired_count: NDArray[np.uint32]
    frag_len_sum: NDArray[np.uint64]
    endpoint_count: NDArray[np.uint32]


def _encode_contig_ids(
    contigs: NDArray[np.integer]
    | Sequence[str]
    | NDArray[np.str_]
    | NDArray[np.object_],
    expected_len: int,
) -> NDArray[np.int64]:
    if isinstance(contigs, np.ndarray) and np.issubdtype(contigs.dtype, np.integer):
        c_ids = np.asarray(contigs, dtype=np.int64)
    else:
        c_arr = np.asarray(contigs, dtype=str)
        if c_arr.shape != (expected_len,):
            raise InvalidInputError("contigs length must match counts row count")
        _, c_ids = np.unique(c_arr, return_inverse=True)
        c_ids = c_ids.astype(np.int64, copy=False)
    if c_ids.shape != (expected_len,):
        raise InvalidInputError("contigs length must match counts row count")
    return c_ids


def _size_factors_from_blocks(
    block_sums: NDArray[np.float64], *, has_mask: bool
) -> NDArray[np.float64]:
    totals = block_sums.sum(axis=0)
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0):
        suffix = " on usable bins" if has_mask else ""
        raise InvalidInputError(
            f"Total read count is zero for at least one sample{suffix}."
        )
    ref_block = block_sums.mean(axis=1, keepdims=True)
    valid = ref_block[:, 0] > 0
    raw_sf = np.median(block_sums[valid] / ref_block[valid], axis=0)
    total_ref = float(np.sum(ref_block[valid]))
    fallback = totals / total_ref if total_ref > 0 else np.ones_like(totals)
    sparse_two = int(np.count_nonzero(valid)) <= 2 and bool(
        np.any(block_sums[valid] == 0)
    )
    raw_sf = (
        fallback
        if sparse_two
        else np.where(~np.isfinite(raw_sf) | (raw_sf <= 0), fallback, raw_sf)
    )
    med = float(np.median(raw_sf))
    if not math.isfinite(med) or med <= 0.0:
        raw_sf = fallback
        med = float(np.median(raw_sf))
    return raw_sf / med


def estimate_superblock_size_factors(
    counts: NDArray[np.floating] | NDArray[np.integer],
    contigs: NDArray[np.integer]
    | Sequence[str]
    | NDArray[np.str_]
    | NDArray[np.object_],
    starts: NDArray[np.integer],
    ends: NDArray[np.integer],
    usable: NDArray[np.bool_],
    *,
    block_bp: int = 1_000_000,
) -> NDArray[np.float64]:
    """Estimate DESeq2 median-of-ratios size factors over per-contig 1 Mb super-blocks."""
    if counts.ndim != 2 or 0 in counts.shape:
        raise InvalidInputError("Counts matrix must be a non-empty 2D array.")
    if not isinstance(block_bp, int) or isinstance(block_bp, bool) or block_bp <= 0:
        raise InvalidInputError("block_bp must be a positive integer")
    n_bins, n_samples = counts.shape
    chunk_rows = max(1024, (64 * 1024 * 1024) // max(n_samples, 1))
    for r0 in range(0, n_bins, chunk_rows):
        if np.any(counts[r0 : r0 + chunk_rows] < 0):
            raise InvalidInputError("Counts matrix cannot contain negative values.")

    usable_arr = np.asarray(usable, dtype=np.bool_)
    if usable_arr.shape != (n_bins,):
        raise InvalidInputError("usable length must match counts row count")
    u_idx = np.flatnonzero(usable_arr)
    if u_idx.size == 0:
        raise InvalidInputError("No usable bins remaining after mask.")

    c_ids = _encode_contig_ids(contigs, n_bins)
    s_arr, e_arr = np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)
    if s_arr.shape != (n_bins,) or e_arr.shape != (n_bins,):
        raise InvalidInputError("starts and ends must match counts row count")
    widths = e_arr - s_arr
    if np.any(widths < 0):
        raise InvalidInputError("bin widths cannot be negative")

    bin_size = max(1, int(np.max(widths)))
    bins_per_block = max(10, round(max(int(block_bp), 10 * bin_size) / bin_size))
    half_k = max(1, bins_per_block // 2)
    u_c = c_ids[u_idx]
    run_cuts = np.concatenate(
        ([0], np.flatnonzero(u_c[1:] != u_c[:-1]) + 1, [u_idx.size])
    )
    starts_list: list[int] = []
    for r_idx in range(run_cuts.size - 1):
        r0, r1 = int(run_cuts[r_idx]), int(run_cuts[r_idx + 1])
        cuts = (
            [r0] if r1 - r0 <= bins_per_block else list(range(r0, r1, bins_per_block))
        )
        if len(cuts) > 1 and (r1 - cuts[-1]) < half_k:
            cuts.pop()
        starts_list.extend(cuts)

    b_starts = np.asarray(starts_list, dtype=np.int64)
    b_ends = np.concatenate((b_starts[1:], [u_idx.size]))
    block_sums = np.empty((b_starts.size, n_samples), dtype=np.float64)
    b0 = 0
    while b0 < b_starts.size:
        b1 = b0 + 1
        while b1 < b_starts.size and int(b_ends[b1 - 1] - b_starts[b0]) < chunk_rows:
            b1 += 1
        u0, u1 = int(b_starts[b0]), int(b_ends[b1 - 1])
        sub = np.asarray(counts[u_idx[u0:u1]], dtype=np.float64)
        block_sums[b0:b1] = np.add.reduceat(sub, b_starts[b0:b1] - u0, axis=0)
        b0 = b1
    return _size_factors_from_blocks(block_sums, has_mask=True)


def estimate_size_factors(
    counts: NDArray[np.floating] | NDArray[np.integer],
    mask: NDArray[np.bool_] | None = None,
    *,
    contigs: NDArray[np.integer] | Sequence[str] | None = None,
    starts: NDArray[np.integer] | None = None,
    ends: NDArray[np.integer] | None = None,
    block_bp: int = 1_000_000,
) -> NDArray[np.float64]:
    """Estimate median-normalized DESeq2 size factors for each sample."""
    arr = np.asarray(counts)
    if arr.ndim != 2 or 0 in arr.shape:
        raise InvalidInputError("Counts matrix must be a non-empty 2D array.")
    if contigs is not None and starts is not None and ends is not None:
        usable = (
            np.ones(arr.shape[0], dtype=np.bool_)
            if mask is None
            else np.asarray(mask, dtype=np.bool_)
        )
        return estimate_superblock_size_factors(
            arr, contigs, starts, ends, usable, block_bp=block_bp
        )

    matrix = np.asarray(arr, dtype=np.float64)
    if np.any(matrix < 0):
        raise InvalidInputError("Counts matrix cannot contain negative values.")
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=np.bool_)
        if mask_arr.shape != (matrix.shape[0],):
            raise InvalidInputError("mask length must match counts row count")
        sub = matrix[mask_arr]
        if sub.shape[0] == 0:
            raise InvalidInputError("No usable bins remaining after mask.")
    else:
        sub = matrix

    if sub.shape[0] >= 1000:
        cuts = np.arange(0, sub.shape[0], 1000, dtype=np.int64)
        if cuts.size > 1 and (sub.shape[0] - int(cuts[-1])) < 500:
            cuts = cuts[:-1]
        return _size_factors_from_blocks(
            np.add.reduceat(sub, cuts, axis=0), has_mask=mask is not None
        )

    totals = sub.sum(axis=0)
    if np.any(totals <= 0):
        suffix = " on usable bins" if mask is not None else ""
        raise InvalidInputError(
            f"Total read count is zero for at least one sample{suffix}."
        )
    pos_rows = np.all(sub > 0, axis=1)
    if np.count_nonzero(pos_rows) >= 1:
        log_gm = np.mean(np.log(sub[pos_rows]), axis=1, keepdims=True)
        raw_sf = np.exp(np.median(np.log(sub[pos_rows]) - log_gm, axis=0))
    else:
        raw_sf = totals.astype(np.float64)
    return raw_sf / float(np.median(raw_sf))


def mask_low_coverage_per_contig(
    counts: NDArray[np.floating] | NDArray[np.integer],
    contigs: NDArray[np.integer]
    | Sequence[str]
    | NDArray[np.str_]
    | NDArray[np.object_],
    exposure: NDArray[np.floating],
    *,
    min_fraction: float = _MASK_FRACTION,
) -> NDArray[np.bool_]:
    """Mask bins per contig where exposure-adjusted depth falls below min_fraction of positive median."""
    if not math.isfinite(min_fraction) or not (0.0 <= min_fraction < 1.0):
        raise InvalidInputError("min_fraction must be a finite number in [0, 1)")
    arr = np.asarray(counts)
    if arr.ndim == 1:
        totals = np.asarray(arr, dtype=np.float64)
    elif arr.ndim == 2:
        totals = np.sum(arr, axis=1, dtype=np.float64)
    else:
        raise InvalidInputError("counts must be a 1D or 2D array")

    n_bins = totals.shape[0]
    exp_arr = np.asarray(exposure, dtype=np.float64)
    if exp_arr.shape != (n_bins,):
        raise InvalidInputError("exposure length must match counts row count")
    c_ids = _encode_contig_ids(contigs, n_bins)
    if n_bins == 0:
        return np.empty(0, dtype=np.bool_)

    pos = np.isfinite(exp_arr) & (exp_arr > 0.0) & np.isfinite(totals) & (totals > 0.0)
    rate = np.where(pos, totals / np.maximum(exp_arr, 1e-6), 0.0)
    keep = np.zeros(n_bins, dtype=np.bool_)
    for c_val in np.unique(c_ids):
        on_c = c_ids == c_val
        pos_rates = rate[on_c & pos]
        if pos_rates.size > 0:
            keep[on_c] = pos[on_c] & (
                rate[on_c] > min_fraction * float(np.median(pos_rates))
            )
    return keep


def _extract_streamed_sample(
    index: int,
    table: pa.Table | SampleBuffers,
    path: Path,
    target_bin_size: int | None,
) -> _StreamedSample:
    try:
        bufs = SampleBuffers.coerce(table, validate=True)
    except SchemaError as err:
        raise ReferenceError(f"{path}: {err}") from err
    raw_meta = (
        bufs.meta.to_metadata()
        if isinstance(table, SampleBuffers)
        else decode_metadata(table.schema.metadata)
    )
    if target_bin_size is not None and bufs.meta.bin_size != target_bin_size:
        bufs = rebin_sample_buffers(bufs, target_bin_size)
        if bufs.meta.fragmentomics is None:
            raw_meta.pop("fragmentomics", None)

    is_paired = (
        bufs.meta.fragmentomics.paired_end
        if bufs.meta.fragmentomics is not None
        else (bufs.frag_bands is not None)
    )
    return _StreamedSample(
        index,
        path,
        raw_meta,
        bufs.meta,
        is_paired,
        np.asarray(bufs.count, dtype=np.uint32),
        bufs.paired_count,
        bufs.frag_len_sum,
        bufs.endpoint_count,
    )


def _read_and_extract_sample(
    index: int, path: Path, target_bin_size: int | None
) -> _StreamedSample:
    try:
        table = pq.read_table(path)
    except (FileNotFoundError, OSError, pa.ArrowInvalid, pa.ArrowException) as err:
        raise ReferenceError(f"{path}: cannot read sample Parquet ({err})") from err
    return _extract_streamed_sample(index, table, path, target_bin_size)


def _alloc_cohort_counts(
    n_bins: int, n_samples: int, max_batch_memory_mb: int
) -> NDArray[np.float32]:
    if (
        n_bins > 0
        and n_samples > 0
        and n_bins * n_samples * 4 > max_batch_memory_mb * 1024 * 1024
    ):
        with tempfile.NamedTemporaryFile(
            prefix="ccnv_cohort_", suffix=".mmap", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        mmap_arr = np.memmap(
            tmp_path, dtype=np.float32, mode="w+", shape=(n_bins, n_samples), order="F"
        )
        try:
            os.unlink(tmp_path)
        except OSError:
            weakref.finalize(mmap_arr, lambda p=tmp_path: p.unlink(missing_ok=True))
        return mmap_arr
    return np.empty((n_bins, n_samples), dtype=np.float32, order="F")


class _CohortAccumulator:
    def __init__(
        self,
        first: _StreamedSample,
        n_samples: int,
        *,
        require_same_native_bin: bool,
        max_batch_memory_mb: int,
    ) -> None:
        n_bins = first.count.size
        self.first, self.require_same_native_bin = first, require_same_native_bin
        self.counts = _alloc_cohort_counts(n_bins, n_samples, max_batch_memory_mb)
        self.sum_count = np.zeros(n_bins, dtype=np.float64)
        self.sum_paired = np.zeros(n_bins, dtype=np.float64)
        self.sum_flen = np.zeros(n_bins, dtype=np.float64)
        self.sum_ep = np.zeros(n_bins, dtype=np.float64)
        self.raw_metas: list[dict[str, object]] = [{} for _ in range(n_samples)]
        self.identities: list[object] = [None for _ in range(n_samples)]
        self.record(first)

    def record(self, sample: _StreamedSample) -> None:
        first, sm, fm = self.first, sample.sample_meta, self.first.sample_meta
        if sample.is_paired != first.is_paired:
            raise ReferenceError(
                "mixed single-end and paired-end cohort controls are not supported"
            )
        if sm.contig_lengths != fm.contig_lengths:
            raise ReferenceError(
                f"{sample.path}: contig_lengths do not match the cohort"
            )
        if self.require_same_native_bin and sm.bin_size != fm.bin_size:
            raise ReferenceError(
                f"{sample.path}: bin_size {sm.bin_size} does not match cohort {fm.bin_size}"
            )
        if sm.contig_order != fm.contig_order or sample.count.size != first.count.size:
            raise ReferenceError(
                f"{sample.path}: bin coordinates do not match the cohort"
            )

        idx = sample.index
        logger.debug(
            "loaded cohort sample %s: %d bins, %d total depth",
            sample.path.name,
            sample.count.size,
            int(np.sum(sample.count)),
        )
        self.counts[:, idx] = sample.count
        self.sum_count += sample.count
        self.sum_paired += sample.paired_count
        self.sum_flen += sample.frag_len_sum
        self.sum_ep += sample.endpoint_count
        self.raw_metas[idx] = sample.raw_meta
        self.identities[idx] = sample.raw_meta.get(
            "source_identity", {"path": str(sample.path)}
        )

    def finalize(
        self,
        paths: Sequence[Path],
        *,
        in_memory_tables: Sequence[pa.Table] = (),
    ) -> Cohort:
        codes, names, starts_u32, ends_u32 = reconstruct_sample_grid(
            self.first.sample_meta
        )
        if self.first.is_paired:
            pos = self.sum_paired > 0
            fb = (
                float(np.median(self.sum_flen[pos] / self.sum_paired[pos]))
                if np.any(pos)
                else _DEFAULT_FRAG_LEN_BP
            )
            ref_frag_len = np.where(
                pos, self.sum_flen / np.maximum(self.sum_paired, 1.0), fb
            ).astype(np.float32)
            ref_ep_rate = (self.sum_ep / np.maximum(self.sum_count, 1.0)).astype(
                np.float32
            )
        else:
            ref_frag_len = ref_ep_rate = None
        return Cohort(
            counts=self.counts,
            contigs=tuple(names[int(c)] for c in codes),
            starts=starts_u32.astype(np.int64),
            ends=ends_u32.astype(np.int64),
            contig_lengths=dict(self.first.sample_meta.contig_lengths),
            bin_size=self.first.sample_meta.bin_size,
            identities=tuple(self.identities),
            paths=tuple(paths),
            sample_metadata=tuple(self.raw_metas),
            ref_frag_len=ref_frag_len,
            ref_ep_rate=ref_ep_rate,
            in_memory_tables=tuple(in_memory_tables),
        )


def load_and_assemble_cohort(
    entries: Sequence[SamplesheetEntry],
    target_bin_size: int | None = None,
    *,
    threads: int = 1,
    max_batch_memory_mb: int = 512,
) -> Cohort:
    """Stream sample sketches into a Fortran-contiguous float32 cohort matrix."""
    if len(entries) < 1:
        raise ReferenceError("cohort must contain at least one sample")
    if target_bin_size is not None and target_bin_size <= 0:
        raise BinCompatibilityError("target_bin_size must be a positive integer")

    paths = [entry.filepath for entry in entries]
    n_samples = len(paths)
    first = _read_and_extract_sample(0, paths[0], target_bin_size)
    acc = _CohortAccumulator(
        first,
        n_samples,
        require_same_native_bin=target_bin_size is None,
        max_batch_memory_mb=max_batch_memory_mb,
    )
    if n_samples > 1:
        workers = max(1, min(threads, n_samples - 1))
        pending: deque[Future[_StreamedSample]] = deque()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for idx in range(1, min(n_samples, 1 + 2 * workers)):
                pending.append(
                    pool.submit(
                        _read_and_extract_sample, idx, paths[idx], target_bin_size
                    )
                )
            next_idx = 1 + len(pending)
            while pending:
                acc.record(pending.popleft().result())
                if next_idx < n_samples:
                    pending.append(
                        pool.submit(
                            _read_and_extract_sample,
                            next_idx,
                            paths[next_idx],
                            target_bin_size,
                        )
                    )
                    next_idx += 1
    return acc.finalize(paths)


def _weighted_sample_column(
    table: pa.Table | SampleBuffers,
    target_bin_size: int,
    w_24: NDArray[np.float64],
) -> NDArray[np.float32]:
    bufs = SampleBuffers.coerce(table, validate=True)
    if bufs.meta.bin_size != target_bin_size:
        bufs = rebin_sample_buffers(bufs, target_bin_size)
    if (
        bufs.num_rows == 0
        or bufs.frag_bands is None
        or not np.any(bufs.paired_count > 0)
    ):
        return np.asarray(bufs.count, dtype=np.float32)
    w_counts, _ = evaluate_band_weighted_counts(
        bufs.frag_bands, bufs.count, bufs.paired_count, w_24
    )
    return np.asarray(w_counts, dtype=np.float32)


def build_fisher_weighted_cohort(
    cohort: Cohort,
    band_weights_24: Sequence[float] | NDArray[np.floating],
    *,
    max_batch_memory_mb: int = 256,
    threads: int = 1,
) -> Cohort:
    """Return a Cohort whose counts matrix applies 24-band Fisher read weighting."""
    n_bins, n_samples = cohort.counts.shape
    w_24 = np.asarray(band_weights_24, dtype=np.float64)
    if n_bins == 0 or n_samples == 0 or w_24.size != NUM_FRAG_BANDS:
        return cohort

    if len(cohort.in_memory_tables) == n_samples:
        weighted = _alloc_cohort_counts(n_bins, n_samples, max_batch_memory_mb)
        for idx, tbl in enumerate(cohort.in_memory_tables):
            weighted[:, idx] = _weighted_sample_column(tbl, cohort.bin_size, w_24)
        return replace(cohort, counts=weighted)

    if len(cohort.paths) != n_samples or not all(p.is_file() for p in cohort.paths):
        return cohort

    weighted = _alloc_cohort_counts(n_bins, n_samples, max_batch_memory_mb)

    def _load_col(idx: int) -> tuple[int, NDArray[np.float32]]:
        path = cohort.paths[idx]
        try:
            tbl = pq.read_table(path)
        except (FileNotFoundError, OSError, pa.ArrowInvalid, pa.ArrowException) as err:
            raise ReferenceError(f"{path}: cannot read sample Parquet ({err})") from err
        return idx, _weighted_sample_column(tbl, cohort.bin_size, w_24)

    workers = max(1, min(threads, n_samples))
    if workers == 1:
        for idx in range(n_samples):
            _, col = _load_col(idx)
            weighted[:, idx] = col
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for idx, col in pool.map(_load_col, range(n_samples)):
                weighted[:, idx] = col
    return replace(cohort, counts=weighted)


def validate_cohort(
    tables: Sequence[pa.Table | SampleBuffers],
    paths: Sequence[Path] | None = None,
    target_bin_size: int | None = None,
) -> Cohort:
    """Validate in-memory sample sketch tables into an aligned Cohort."""
    if len(tables) < 1:
        raise ReferenceError("cohort must contain at least one sample")
    resolved = (
        list(paths)
        if paths is not None
        else [Path(f"sample_{i}") for i in range(len(tables))]
    )
    if len(resolved) != len(tables):
        raise ReferenceError("paths must align with sample tables")
    first = _extract_streamed_sample(0, tables[0], resolved[0], target_bin_size)
    acc = _CohortAccumulator(
        first,
        len(tables),
        require_same_native_bin=target_bin_size is None,
        max_batch_memory_mb=65_536,
    )
    for idx in range(1, len(tables)):
        acc.record(
            _extract_streamed_sample(idx, tables[idx], resolved[idx], target_bin_size)
        )
    return acc.finalize(resolved, in_memory_tables=tables)


def rebin_cohort(cohort: Cohort, target_bin_size: int) -> Cohort:
    """Rescale 2D cohort count matrix to target_bin_size."""
    if target_bin_size <= 0:
        raise ValueError("target_bin_size must be a positive integer")
    if cohort.bin_size == target_bin_size:
        return cohort
    rebinned = rebin_counts_matrix(
        cohort.counts,
        cohort.contigs,
        cohort.starts,
        cohort.ends,
        cohort.contig_lengths,
        cohort.bin_size,
        target_bin_size,
    )
    return Cohort(
        *rebinned,
        cohort.contig_lengths,
        target_bin_size,
        cohort.identities,
        cohort.paths,
        cohort.sample_metadata,
    )


def cohort_from_fasta(
    contigs_dict: Mapping[str, int],
    bin_size: int,
    *,
    fasta_path: Path | None = None,
) -> Cohort:
    """Build a 0-sample coordinate grid from FASTA contig lengths for flat reference models."""
    if bin_size <= 0:
        raise ReferenceError("bin_size must be a positive integer")
    resolved = (fasta_path or Path("flat_fasta")).expanduser().resolve()
    contigs, starts, ends, contig_lengths = build_coordinate_grid(
        contigs_dict, bin_size
    )
    counts = np.empty((starts.size, 0), dtype=np.float32, order="F")
    identities: tuple[object, ...] = ({"path": str(resolved), "format": "flat-fasta"},)
    return Cohort(
        counts=counts,
        contigs=contigs,
        starts=starts,
        ends=ends,
        contig_lengths=contig_lengths,
        bin_size=bin_size,
        identities=identities,
        paths=(),
    )
