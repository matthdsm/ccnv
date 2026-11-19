"""Cohort sample aggregation, streaming assembly, and super-block depth scaling."""

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

from ccnv._core import FRAG_BAND_EDGES
from ccnv.features.binning import (
    build_coordinate_grid,
    rebin_counts_matrix,
)
from ccnv.features.fragmentomics import has_paired_fragmentomics
from ccnv.features.rebinning import rebin_sample_buffers
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
from ccnv.stats import estimate_superblock_scales

logger = logging.getLogger(__name__)
_MASK_FRACTION: Final[float] = 0.05
_DEFAULT_FRAG_LEN_BP: Final[float] = 167.0
_FRAG_BAND_MIDPOINTS: Final[NDArray[np.float64]] = 0.5 * (
    np.asarray(FRAG_BAND_EDGES[:-1], dtype=np.float64)
    + np.asarray(FRAG_BAND_EDGES[1:], dtype=np.float64)
    - 1.0
)

__all__ = [
    "Cohort",
    "alloc_cohort_counts",
    "build_fisher_weighted_cohort",
    "cohort_from_fasta",
    "estimate_size_factors",
    "load_and_assemble_cohort",
    "mask_low_coverage_per_contig",
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
    in_memory_tables: tuple[SampleBuffers, ...] = ()
    ref_kappa_frag: NDArray[np.float32] | None = None


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
    bufs: SampleBuffers | None = None


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


def estimate_size_factors(
    counts: NDArray[np.floating] | NDArray[np.integer],
    mask: NDArray[np.bool_] | None = None,
    *,
    contigs: NDArray[np.integer]
    | Sequence[str]
    | NDArray[np.str_]
    | NDArray[np.object_]
    | None = None,
    starts: NDArray[np.integer] | None = None,
    ends: NDArray[np.integer] | None = None,
    block_bp: int = 1_000_000,
) -> NDArray[np.float64]:
    """Estimate DESeq2 median-of-ratios size factors over per-contig 1 Mb super-blocks."""
    arr = counts if isinstance(counts, np.ndarray) else np.asarray(counts)
    if arr.ndim != 2 or 0 in arr.shape:
        raise InvalidInputError("Counts matrix must be a non-empty 2D array.")
    if contigs is None or starts is None or ends is None:
        raise InvalidInputError(
            "contigs, starts, and ends coordinates are required for super-block size factors"
        )
    n_bins = arr.shape[0]
    c_ids = _encode_contig_ids(contigs, n_bins)
    s_arr = np.asarray(starts, dtype=np.int64)
    e_arr = np.asarray(ends, dtype=np.int64)
    if s_arr.shape != (n_bins,) or e_arr.shape != (n_bins,):
        raise InvalidInputError("starts and ends must match counts row count")
    widths = e_arr - s_arr
    if np.any(widths < 0):
        raise InvalidInputError("bin widths cannot be negative")
    bin_size = max(1, int(np.max(widths)))
    return estimate_superblock_scales(
        arr,
        c_ids,
        usable=mask,
        bin_size=bin_size,
        block_bp=block_bp,
    )


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


def _finalize_frag_covariates(
    sum_count: NDArray[np.float64],
    sum_paired: NDArray[np.float64],
    sum_flen: NDArray[np.float64],
    sum_ep: NDArray[np.float64],
    *,
    is_paired: bool,
) -> tuple[NDArray[np.float32] | None, NDArray[np.float32] | None]:
    if not is_paired:
        return None, None
    pos = sum_paired > 0.0
    fb = (
        float(np.median(sum_flen[pos] / sum_paired[pos]))
        if np.any(pos)
        else _DEFAULT_FRAG_LEN_BP
    )
    ref_frag_len = np.where(pos, sum_flen / np.maximum(sum_paired, 1e-12), fb).astype(
        np.float32
    )
    ref_ep_rate = (sum_ep / np.maximum(sum_count, 1e-12)).astype(np.float32)
    return ref_frag_len, ref_ep_rate


def _extract_streamed_sample(
    index: int,
    bufs: SampleBuffers,
    raw_meta: dict[str, object],
    path: Path,
    target_bin_size: int | None,
    *,
    retain_buffers: bool = False,
) -> _StreamedSample:
    if target_bin_size is not None and bufs.meta.bin_size != target_bin_size:
        bufs = rebin_sample_buffers(bufs, target_bin_size)
        if bufs.meta.fragmentomics is None:
            raw_meta.pop("fragmentomics", None)

    is_paired = has_paired_fragmentomics(bufs)
    return _StreamedSample(
        index=index,
        path=path,
        raw_meta=raw_meta,
        sample_meta=bufs.meta,
        is_paired=is_paired,
        count=np.asarray(bufs.count, dtype=np.uint32),
        paired_count=bufs.paired_count,
        frag_len_sum=bufs.frag_len_sum,
        endpoint_count=bufs.endpoint_count,
        bufs=bufs if retain_buffers else None,
    )


def _read_cohort_sample(path: Path) -> tuple[SampleBuffers, pa.Table]:
    try:
        table = pq.read_table(path)
        bufs = SampleBuffers.from_table(table, validate=True)
    except (FileNotFoundError, OSError, pa.ArrowInvalid, pa.ArrowException) as err:
        raise ReferenceError(f"{path}: cannot read sample Parquet ({err})") from err
    except SchemaError as err:
        raise ReferenceError(f"{path}: {err}") from err
    return bufs, table


def _read_and_extract_sample(
    index: int, path: Path, target_bin_size: int | None
) -> _StreamedSample:
    bufs, table = _read_cohort_sample(path)
    raw_meta = decode_metadata(table.schema.metadata)
    return _extract_streamed_sample(index, bufs, raw_meta, path, target_bin_size)


def alloc_cohort_counts(
    n_bins: int, n_samples: int, max_batch_memory_mb: int
) -> NDArray[np.float32]:
    """Allocate a Fortran-contiguous float32 cohort matrix, spilling to memmap above budget."""
    if (
        n_bins > 0
        and n_samples > 0
        and n_bins * n_samples * 4 > max_batch_memory_mb * 1024 * 1024
    ):
        with tempfile.NamedTemporaryFile(
            prefix="ccnv_cohort_", suffix=".mmap", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            mmap_arr = np.memmap(
                tmp_path,
                dtype=np.float32,
                mode="w+",
                shape=(n_bins, n_samples),
                order="F",
            )
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
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
        self.counts = alloc_cohort_counts(n_bins, n_samples, max_batch_memory_mb)
        self.sum_count = np.zeros(n_bins, dtype=np.float64)
        self.sum_paired = np.zeros(n_bins, dtype=np.float64)
        self.sum_flen = np.zeros(n_bins, dtype=np.float64)
        self.sum_ep = np.zeros(n_bins, dtype=np.float64)
        self.raw_metas: list[dict[str, object]] = [{} for _ in range(n_samples)]
        self.identities: list[object] = [None for _ in range(n_samples)]
        self.is_paired: list[bool] = [False for _ in range(n_samples)]
        self.record(first)

    def record(self, sample: _StreamedSample) -> None:
        first, sm, fm = self.first, sample.sample_meta, self.first.sample_meta
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
        self.is_paired[idx] = sample.is_paired

    def finalize(
        self,
        paths: Sequence[Path],
        *,
        in_memory_tables: Sequence[SampleBuffers] = (),
    ) -> Cohort:
        paired_paths = [
            str(p) for p, paired in zip(paths, self.is_paired, strict=True) if paired
        ]
        single_paths = [
            str(p)
            for p, paired in zip(paths, self.is_paired, strict=True)
            if not paired
        ]
        if paired_paths and single_paths:
            total = len(paths)
            n_paired, n_single = len(paired_paths), len(single_paths)
            if n_paired > n_single:
                detail = (
                    f"cohort is paired-end ({n_paired}/{total}), "
                    f"single-end files ({n_single}/{total}): "
                    f"{', '.join(single_paths)}"
                )
            elif n_single > n_paired:
                detail = (
                    f"cohort is single-end ({n_single}/{total}), "
                    f"paired-end files ({n_paired}/{total}): "
                    f"{', '.join(paired_paths)}"
                )
            else:
                detail = (
                    f"paired-end files ({n_paired}/{total}): "
                    f"{', '.join(paired_paths)}; "
                    f"single-end files ({n_single}/{total}): "
                    f"{', '.join(single_paths)}"
                )
            raise ReferenceError(
                "mixed single-end and paired-end cohort controls are not supported: "
                f"{detail}"
            )
        codes, names, starts_u32, ends_u32 = reconstruct_sample_grid(
            self.first.sample_meta
        )
        ref_frag_len, ref_ep_rate = _finalize_frag_covariates(
            self.sum_count,
            self.sum_paired,
            self.sum_flen,
            self.sum_ep,
            is_paired=self.first.is_paired,
        )
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


def _weighted_sample_summary(
    bufs: SampleBuffers,
    target_bin_size: int,
    w_24: NDArray[np.float64],
    w_sq_24: NDArray[np.float64],
    w_mid_24: NDArray[np.float64],
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    if bufs.meta.bin_size != target_bin_size:
        bufs = rebin_sample_buffers(bufs, target_bin_size)
    zeros = np.zeros(bufs.num_rows, dtype=np.float64)
    if (
        bufs.num_rows == 0
        or bufs.frag_bands is None
        or not np.any(bufs.paired_count > 0)
    ):
        w_counts = np.asarray(bufs.count, dtype=np.float64)
        return (
            w_counts.astype(np.float32),
            np.asarray(bufs.paired_count, dtype=np.float64),
            np.asarray(bufs.frag_len_sum, dtype=np.float64),
            np.asarray(bufs.endpoint_count, dtype=np.float64),
            zeros,
            zeros,
        )
    bands_f64 = np.asarray(bufs.frag_bands, dtype=np.float64)
    raw_cnt = np.asarray(bufs.count, dtype=np.float64)
    paired_f64 = np.asarray(bufs.paired_count, dtype=np.float64)
    raw_unpaired = np.maximum(raw_cnt - paired_f64, 0.0)
    w_paired = bands_f64 @ w_24
    w_counts = raw_unpaired + w_paired
    w_flen = bands_f64 @ w_mid_24
    ep_scale = np.where(raw_cnt > 0.0, w_counts / np.maximum(raw_cnt, 1.0), 1.0)
    w_ep = np.asarray(bufs.endpoint_count, dtype=np.float64) * ep_scale

    has_p = paired_f64 > 0.0
    w_tot_bin = np.where(has_p, w_counts, 0.0)
    w_sq_bin = np.where(has_p, raw_unpaired + (bands_f64 @ w_sq_24), 0.0)
    return (
        w_counts.astype(np.float32),
        w_paired,
        w_flen,
        w_ep,
        w_tot_bin,
        w_sq_bin,
    )


def build_fisher_weighted_cohort(
    cohort: Cohort,
    band_weights_24: Sequence[float] | NDArray[np.floating] | None,
    *,
    max_batch_memory_mb: int = 256,
    threads: int = 1,
) -> Cohort:
    """Return a Cohort whose counts and fragmentomic covariates apply 24-band Fisher read weighting."""
    if band_weights_24 is None:
        return cohort
    w_24 = np.asarray(band_weights_24, dtype=np.float64)
    if (
        w_24.ndim != 1
        or w_24.size != NUM_FRAG_BANDS
        or not np.all(np.isfinite(w_24) & (w_24 >= 0.0))
    ):
        raise ReferenceError(
            f"band_weights_24 must contain {NUM_FRAG_BANDS} finite non-negative weights"
        )

    n_bins, n_samples = cohort.counts.shape
    if n_bins == 0 or n_samples == 0:
        return cohort

    has_inmem = len(cohort.in_memory_tables) == n_samples
    has_paths = len(cohort.paths) == n_samples and all(
        p.is_file() for p in cohort.paths
    )
    if not has_inmem and not has_paths:
        raise ReferenceError(
            "cannot apply Fisher fragment weighting: cohort lacks in-memory sample buffers and readable sample paths"
        )

    w_sq_24 = np.square(w_24)
    w_mid_24 = w_24 * _FRAG_BAND_MIDPOINTS
    weighted = alloc_cohort_counts(n_bins, n_samples, max_batch_memory_mb)
    sum_count = np.zeros(n_bins, dtype=np.float64)
    sum_paired = np.zeros(n_bins, dtype=np.float64)
    sum_flen = np.zeros(n_bins, dtype=np.float64)
    sum_ep = np.zeros(n_bins, dtype=np.float64)
    sum_w_tot = np.zeros(n_bins, dtype=np.float64)
    sum_w_sq = np.zeros(n_bins, dtype=np.float64)

    def _accumulate(
        idx: int,
        col: NDArray[np.float32],
        w_p: NDArray[np.float64],
        w_fl: NDArray[np.float64],
        w_e: NDArray[np.float64],
        w_tot_b: NDArray[np.float64],
        w_sq_b: NDArray[np.float64],
    ) -> None:
        nonlocal sum_count, sum_paired, sum_flen, sum_ep
        nonlocal sum_w_tot, sum_w_sq
        weighted[:, idx] = col
        sum_count += col
        sum_paired += w_p
        sum_flen += w_fl
        sum_ep += w_e
        sum_w_tot += w_tot_b
        sum_w_sq += w_sq_b

    if has_inmem:
        for idx, bufs in enumerate(cohort.in_memory_tables):
            summary = _weighted_sample_summary(
                bufs, cohort.bin_size, w_24, w_sq_24, w_mid_24
            )
            _accumulate(idx, *summary)
    else:

        def _load_col(
            idx: int,
        ) -> tuple[
            int,
            NDArray[np.float32],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ]:
            bufs, _ = _read_cohort_sample(cohort.paths[idx])
            summary = _weighted_sample_summary(
                bufs, cohort.bin_size, w_24, w_sq_24, w_mid_24
            )
            return idx, *summary

        workers = max(1, min(threads, n_samples))
        if workers == 1:
            for idx in range(n_samples):
                _accumulate(*_load_col(idx))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for loaded in pool.map(_load_col, range(n_samples)):
                    _accumulate(*loaded)

    has_frag_covs = (cohort.ref_frag_len is not None) or (
        cohort.ref_ep_rate is not None
    )
    ref_frag_len, ref_ep_rate = _finalize_frag_covariates(
        sum_count, sum_paired, sum_flen, sum_ep, is_paired=has_frag_covs
    )
    has_paired = (sum_w_tot > 0.0) & (sum_paired > 0.0)
    ref_kappa = np.ones(n_bins, dtype=np.float32)
    ref_kappa[has_paired] = np.maximum(
        sum_w_sq[has_paired] / np.maximum(sum_w_tot[has_paired], 1e-12),
        1e-6,
    ).astype(np.float32)
    return replace(
        cohort,
        counts=weighted,
        ref_frag_len=ref_frag_len,
        ref_ep_rate=ref_ep_rate,
        ref_kappa_frag=ref_kappa,
    )


def validate_cohort(
    tables: Sequence[pa.Table | SampleBuffers],
    paths: Sequence[Path] | None = None,
    target_bin_size: int | None = None,
) -> Cohort:
    """Validate in-memory sample sketch tables or buffers into an aligned Cohort."""
    if len(tables) < 1:
        raise ReferenceError("cohort must contain at least one sample")
    resolved = (
        list(paths)
        if paths is not None
        else [Path(f"sample_{i}") for i in range(len(tables))]
    )
    if len(resolved) != len(tables):
        raise ReferenceError("paths must align with sample tables")

    def _coerce_one(idx: int) -> _StreamedSample:
        item = tables[idx]
        path = resolved[idx]
        try:
            bufs = SampleBuffers.coerce(item, validate=True)
        except SchemaError as err:
            raise ReferenceError(f"{path}: {err}") from err
        raw_meta = (
            bufs.meta.to_metadata()
            if isinstance(item, SampleBuffers)
            else decode_metadata(item.schema.metadata)
        )
        return _extract_streamed_sample(
            idx, bufs, raw_meta, path, target_bin_size, retain_buffers=True
        )

    first = _coerce_one(0)
    acc = _CohortAccumulator(
        first,
        len(tables),
        require_same_native_bin=target_bin_size is None,
        max_batch_memory_mb=65_536,
    )
    retained: list[SampleBuffers] = [first.bufs] if first.bufs is not None else []
    for idx in range(1, len(tables)):
        sample = _coerce_one(idx)
        acc.record(sample)
        if sample.bufs is not None:
            retained.append(sample.bufs)
    return acc.finalize(resolved, in_memory_tables=retained)


def rebin_cohort(cohort: Cohort, target_bin_size: int) -> Cohort:
    """Rescale 2D cohort count matrix to target_bin_size."""
    if target_bin_size <= 0:
        raise InvalidInputError("target_bin_size must be a positive integer")
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
