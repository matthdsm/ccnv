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
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from ccnv.features.binning import build_uniform_fasta_grid, rebin_counts_matrix
from ccnv.features.rebinning import rebin_sample_sketch
from ccnv.models import (
    NUM_FRAG_BANDS,
    BinCompatibilityError,
    ReferenceError,
    SampleMetadata,
    SamplesheetEntry,
    SchemaError,
    decode_metadata,
    fixed_size_list_to_numpy,
    reconstruct_sample_grid,
    validate_sample_table,
)

logger = logging.getLogger(__name__)
SexLabel = Literal["female", "male"]
_MASK_FRACTION: Final[float] = 0.05
_SEPARATION_MAD_MULTIPLIER: Final[float] = 4.0
_DEFAULT_FRAG_LEN_BP: Final[float] = 167.0

__all__ = [
    "Cohort",
    "InvalidInputError",
    "SexInference",
    "SexLabel",
    "cohort_from_fasta",
    "estimate_size_factors",
    "infer_sample_sexes",
    "load_and_assemble_cohort",
    "mask_low_coverage",
    "normalize_depth",
    "rebin_cohort",
    "rebin_sample_table",
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
    prenatal_bands: NDArray[np.uint64] | None = None


@dataclass(frozen=True, slots=True)
class SexInference:
    """Deterministic sex assignments, or a rejected inference."""

    labels: tuple[SexLabel, ...] | None
    cutoff: float | None
    fractions: tuple[float, ...]
    passed: bool
    reason: str


class InvalidInputError(ReferenceError, ValueError):
    """Raised when count matrix input validation fails."""


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
    band_totals: NDArray[np.uint64] | None


def rebin_sample_table(table: pa.Table, target_bin_size: int) -> pa.Table:
    """Coarsen a Schema v2.0.0 sample sketch table to target_bin_size."""
    return rebin_sample_sketch(table, target_bin_size)


def estimate_size_factors(
    counts: NDArray[np.floating],
    mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Estimate DESeq2 median-of-ratios size factors for each sample."""
    matrix = np.asarray(counts, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise InvalidInputError("Counts matrix must be a non-empty 2D array.")
    if np.any(matrix < 0):
        raise InvalidInputError("Counts matrix cannot contain negative values.")
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=np.bool_)
        if mask_arr.ndim != 1 or mask_arr.shape[0] != matrix.shape[0]:
            raise InvalidInputError("mask length must match counts row count")
        sub = matrix[mask_arr]
        if sub.shape[0] == 0:
            raise InvalidInputError("No usable bins remaining after mask.")
    else:
        sub = matrix

    totals = sub.sum(axis=0)
    if np.any(totals <= 0):
        suffix = " on usable bins" if mask is not None else ""
        raise InvalidInputError(
            f"Total read count is zero for at least one sample{suffix}."
        )

    pos_rows = np.all(sub > 0, axis=1)
    if np.count_nonzero(pos_rows) >= 1:
        log_geomean = np.mean(np.log(sub[pos_rows]), axis=1, keepdims=True)
        raw_sf = np.exp(np.median(np.log(sub[pos_rows]) - log_geomean, axis=0))
    else:
        pos_meds = [float(np.median(c[c > 0])) if np.any(c > 0) else 0.0 for c in sub.T]
        raw_sf = (
            np.asarray(pos_meds, dtype=np.float64)
            if all(m > 0 for m in pos_meds)
            else totals.astype(np.float64)
        )
    return raw_sf / np.exp(np.log(raw_sf).mean())


def normalize_depth(
    counts: NDArray[np.floating],
    mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Normalize sample count matrix using DESeq2 median-of-ratios depth scaling."""
    matrix = np.asarray(counts, dtype=np.float64)
    sf = estimate_size_factors(matrix, mask=mask)
    sub = matrix[mask] if mask is not None else matrix
    s = sf * float(np.median(sub.sum(axis=0) / sf))
    return matrix / s[np.newaxis, :]


def mask_low_coverage(normalized: NDArray[np.floating]) -> NDArray[np.bool_]:
    """Mask bins with no coverage or coverage below 5% of the positive median."""
    per_bin = np.asarray(normalized, dtype=np.float64).sum(axis=1)
    positive = per_bin[per_bin > 0]
    if positive.size == 0:
        return np.zeros(per_bin.shape, dtype=np.bool_)
    return per_bin > (_MASK_FRACTION * float(np.median(positive)))


def _mad(values: NDArray[np.floating]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.median(np.abs(arr - np.median(arr)))) if arr.size > 0 else 0.0


def infer_sample_sexes(
    fractions: NDArray[np.floating],
    min_group_size: int,
    min_separation: float = 0.0,
    cutoff: float | None = None,
) -> SexInference:
    """Split Y-like fractions into female/male groups when separation is clear."""
    values = np.asarray(fractions, dtype=np.float64)
    recorded = tuple(float(v) for v in values)
    if min_group_size < 1:
        raise ReferenceError("min_sex_group_size must be a positive integer")
    if cutoff is not None:
        if not math.isfinite(cutoff) or cutoff < 0.0 or cutoff > 1.0:
            raise ReferenceError("sex_cutoff must be a finite number in [0, 1]")
        labels = tuple("male" if float(v) > cutoff else "female" for v in values)
        return SexInference(labels, float(cutoff), recorded, True, "cutoff")
    if values.size < 2 * min_group_size:
        return SexInference(
            None, None, recorded, False, "cohort is too small for sex inference"
        )

    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    gaps = np.diff(sorted_vals)[min_group_size - 1 : values.size - min_group_size]
    if gaps.size == 0 or float(np.max(gaps)) <= 0:
        return SexInference(
            None, None, recorded, False, "sex-signal fractions are not separable"
        )
    best_rel = int(np.argmax(gaps))
    best_split = min_group_size + best_rel
    low, high = sorted_vals[:best_split], sorted_vals[best_split:]
    spread = max(_mad(low), _mad(high), 1e-12)
    cutoff_val = float((low[-1] + high[0]) / 2.0)
    if float(gaps[best_rel]) < max(min_separation, _SEPARATION_MAD_MULTIPLIER * spread):
        return SexInference(
            None,
            cutoff_val,
            recorded,
            False,
            "sex-signal groups are not sufficiently separated",
        )
    labels_list: list[SexLabel] = ["female"] * values.size
    for idx in order[best_split:]:
        labels_list[int(idx)] = "male"
    return SexInference(tuple(labels_list), cutoff_val, recorded, True, "separated")


def _extract_streamed_sample(
    index: int,
    table: pa.Table,
    path: Path,
    target_bin_size: int | None,
    collect_prenatal_bands: bool,
) -> _StreamedSample:
    try:
        sample_meta = validate_sample_table(table)
    except SchemaError as err:
        raise ReferenceError(f"{path}: {err}") from err
    if target_bin_size is not None and sample_meta.bin_size != target_bin_size:
        table = rebin_sample_sketch(table, target_bin_size)
        sample_meta = validate_sample_table(table)

    raw_meta = decode_metadata(table.schema.metadata)
    frag_obj = raw_meta.get("fragmentomics")
    if sample_meta.fragmentomics is not None:
        is_paired = sample_meta.fragmentomics.paired_end
    elif isinstance(frag_obj, Mapping) and isinstance(frag_obj.get("paired_end"), bool):
        is_paired = bool(frag_obj["paired_end"])
    else:
        is_paired = (
            0 < table.num_rows
            and table.column("frag_bands").null_count < table.num_rows
        )

    band_totals: NDArray[np.uint64] | None = None
    if collect_prenatal_bands:
        bands_col = table.column("frag_bands")
        has_bands = 0 < table.num_rows and bands_col.null_count < table.num_rows
        band_totals = (
            fixed_size_list_to_numpy(bands_col, NUM_FRAG_BANDS, np.uint32).sum(
                axis=0, dtype=np.uint64
            )
            if has_bands
            else np.zeros(NUM_FRAG_BANDS, dtype=np.uint64)
        )
    cols = {
        k: table.column(k).combine_chunks().to_numpy(zero_copy_only=False)
        for k in ("count", "paired_count", "frag_len_sum", "endpoint_count")
    }
    return _StreamedSample(
        index,
        path,
        raw_meta,
        sample_meta,
        is_paired,
        np.asarray(cols["count"], dtype=np.uint32),
        np.asarray(cols["paired_count"], dtype=np.uint32),
        np.asarray(cols["frag_len_sum"], dtype=np.uint64),
        np.asarray(cols["endpoint_count"], dtype=np.uint32),
        band_totals,
    )


def _read_and_extract_sample(
    index: int, path: Path, target_bin_size: int | None, collect_prenatal_bands: bool
) -> _StreamedSample:
    try:
        table = pq.read_table(path)
    except (FileNotFoundError, OSError, pa.ArrowInvalid, pa.ArrowException) as err:
        raise ReferenceError(f"{path}: cannot read sample Parquet ({err})") from err
    return _extract_streamed_sample(
        index, table, path, target_bin_size, collect_prenatal_bands
    )


def _alloc_cohort_counts(
    n_bins: int, n_samples: int, max_batch_memory_mb: int
) -> NDArray[np.float32]:
    if (
        n_bins > 0
        and n_samples > 0
        and n_bins * n_samples * 4 > max_batch_memory_mb * 4 * 1024 * 1024
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
        collect_prenatal_bands: bool,
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
        self.prenatal_bands = (
            np.zeros((n_samples, NUM_FRAG_BANDS), dtype=np.uint64)
            if collect_prenatal_bands
            else None
        )
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
        if self.prenatal_bands is not None and sample.band_totals is not None:
            self.prenatal_bands[idx] = sample.band_totals

    def finalize(self, paths: Sequence[Path]) -> Cohort:
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
            prenatal_bands=self.prenatal_bands,
        )


def load_and_assemble_cohort(
    entries: Sequence[SamplesheetEntry],
    target_bin_size: int | None = None,
    *,
    threads: int = 1,
    max_batch_memory_mb: int = 512,
    collect_prenatal_bands: bool = False,
) -> Cohort:
    """Stream sample sketches into a Fortran-contiguous float32 cohort matrix."""
    if len(entries) < 1:
        raise ReferenceError("cohort must contain at least one sample")
    if target_bin_size is not None and target_bin_size <= 0:
        raise BinCompatibilityError("target_bin_size must be a positive integer")

    paths = [entry.filepath for entry in entries]
    n_samples = len(paths)
    opts = (target_bin_size, collect_prenatal_bands)
    first = _read_and_extract_sample(0, paths[0], *opts)
    acc = _CohortAccumulator(
        first,
        n_samples,
        require_same_native_bin=target_bin_size is None,
        max_batch_memory_mb=max_batch_memory_mb,
        collect_prenatal_bands=collect_prenatal_bands,
    )
    if n_samples > 1:
        workers = max(1, min(threads, n_samples - 1))
        pending: deque[Future[_StreamedSample]] = deque()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for idx in range(1, min(n_samples, 1 + 2 * workers)):
                pending.append(
                    pool.submit(_read_and_extract_sample, idx, paths[idx], *opts)
                )
            next_idx = 1 + len(pending)
            while pending:
                acc.record(pending.popleft().result())
                if next_idx < n_samples:
                    pending.append(
                        pool.submit(
                            _read_and_extract_sample, next_idx, paths[next_idx], *opts
                        )
                    )
                    next_idx += 1
    return acc.finalize(paths)


def validate_cohort(
    tables: Sequence[pa.Table],
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
    first = _extract_streamed_sample(0, tables[0], resolved[0], target_bin_size, False)
    acc = _CohortAccumulator(
        first,
        len(tables),
        require_same_native_bin=target_bin_size is None,
        max_batch_memory_mb=65_536,
        collect_prenatal_bands=False,
    )
    for idx in range(1, len(tables)):
        acc.record(
            _extract_streamed_sample(
                idx, tables[idx], resolved[idx], target_bin_size, False
            )
        )
    return acc.finalize(resolved)


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
    n_samples: int,
    fasta_path: Path | None = None,
) -> Cohort:
    """Build identical uniform-depth samples from FASTA contig lengths."""
    if n_samples < 1:
        raise ReferenceError("n_samples must be a positive integer")
    resolved = (fasta_path or Path("flat_fasta")).expanduser().resolve()
    grid = build_uniform_fasta_grid(contigs_dict, bin_size, n_samples)
    ids = tuple(
        {"path": str(resolved), "format": "flat-fasta", "replica": i}
        for i in range(n_samples)
    )
    paths = tuple(Path(f"{resolved}#flat-{i}") for i in range(n_samples))
    return Cohort(*grid, bin_size, ids, paths)
