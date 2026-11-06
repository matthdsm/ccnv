"""Stream BAM/CRAM alignments into canonical sample Parquet bins."""

import logging
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pyarrow as pa
import pysam

from ccnv.io.alignments import (
    alignment_format as _alignment_format,
)
from ccnv.io.alignments import (
    ensure_index as _ensure_index,
)
from ccnv.io.alignments import (
    open_alignment as _open_alignment,
)
from ccnv.io.alignments import (
    rejection_reason,
    stream_contig_reads,
)
from ccnv.io.parquet import (
    read_sample_parquet,
    write_sample_parquet,
)
from ccnv.models import (
    ConvertConfig,
    ConvertError,
    GenomeProfile,
    ReadFilters,
    is_autosome_name,
    is_canonical_contig,
    normalize_chr,
    parquet_path,
    resolve_contig_name,
    select_layout,
)
from ccnv.schemas import (
    SAMPLE_SCHEMA,
    ArtifactKind,
    decode_metadata,
    metadata_for,
)

logger = logging.getLogger(__name__)

QC_COUNTERS: Final[tuple[str, ...]] = (
    "header_mapped",
    "header_unmapped",
    "header_no_coordinate",
    "reads_seen",
    "reads_kept",
    "effective_reads_examined",
    "effective_reads_kept",
    "autosomal_reads_kept",
    "sex_reads_kept",
    "filter_unmapped",
    "filter_no_start",
    "filter_secondary",
    "filter_supplementary",
    "filter_duplicate",
    "filter_improper_pair",
    "filter_mapq",
)

__all__ = [
    "QC_COUNTERS",
    "ConvertError",
    "bin_contig_with_handle",
    "bin_counts",
    "bin_intervals",
    "empty_qc",
    "is_autosome_name",
    "is_canonical_contig",
    "normalize_chr",
    "read_sample_parquet",
    "rebin_counts",
    "rebin_sample_table",
    "rejection_reason",
    "resolve_contig_name",
    "run",
    "select_layout",
    "split_thread_budget",
    "write_sample_parquet",
]


@dataclass(frozen=True, slots=True)
class _ContigTask:
    """One contig's binning work, sized to be cheap to send to a worker."""

    path: Path
    fmt: str
    fasta: Path | None
    name: str
    length: int
    bin_size: int
    filters: ReadFilters
    decode_threads: int = 1


_WORKER_HANDLES: dict[tuple[str, str, str | None, int], pysam.AlignmentFile] = {}


def split_thread_budget(budget: int, n_tasks: int) -> tuple[int, int]:
    """Split a CPU budget into process workers and per-handle htslib threads.

    The product of the returned values is at most ``budget``. A single task or a
    budget of 1 stays in-process and gives the whole budget to pysam, whose
    ``threads`` argument is passed to ``hts_set_threads`` as ``threads - 1``.
    Otherwise each handle gets two threads when the budget is at least 4, so
    CRAM decode is not stuck at pysam's default of one.
    """

    if budget <= 0:
        raise ValueError("thread budget must be a positive integer")
    if n_tasks <= 0:
        raise ValueError("n_tasks must be a positive integer")
    if budget == 1 or n_tasks == 1:
        return 1, budget
    decode_threads = 2 if budget >= 4 else 1
    workers = min(n_tasks, max(1, budget // decode_threads))
    return workers, max(1, budget // workers)


def _worker_handle(task: _ContigTask) -> pysam.AlignmentFile:
    """Reuse one alignment handle per worker process, keyed by input file."""

    key = (
        str(task.path),
        task.fmt,
        str(task.fasta) if task.fasta else None,
        task.decode_threads,
    )
    handle = _WORKER_HANDLES.get(key)
    if handle is None:
        handle = _open_alignment(
            task.path, task.fmt, task.fasta, threads=task.decode_threads
        )
        _WORKER_HANDLES[key] = handle
    return handle


def _bin_contig(task: _ContigTask) -> tuple[np.ndarray, dict[str, int]]:
    """Bin one contig inside a worker process."""

    return bin_contig_with_handle(_worker_handle(task), task)


def bin_contig_with_handle(
    handle: pysam.AlignmentFile, task: _ContigTask
) -> tuple[np.ndarray, dict[str, int]]:
    """Count reads for one contig and return its bin counts plus QC counters."""

    starts, _ = bin_intervals(task.length, task.bin_size)
    qc = empty_qc()
    starts_list = list(
        stream_contig_reads(
            handle,
            task.name,
            task.filters,
            contig_length=task.length,
            qc=qc,
        )
    )
    kept_starts = (
        np.fromiter(starts_list, dtype=np.int64, count=len(starts_list))
        if starts_list
        else np.empty(0, dtype=np.int64)
    )
    return bin_counts(kept_starts, task.bin_size, starts.size), qc


def empty_qc() -> dict[str, int]:
    """Return a complete QC counter map with deterministic key order."""

    return {name: 0 for name in QC_COUNTERS}


def bin_intervals(length: int, bin_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return 0-based half-open bin edges, keeping a partial last bin."""

    if length <= 0:
        raise ValueError("contig length must be positive")
    if bin_size <= 0:
        raise ValueError("bin_size must be a positive integer")
    n_bins = (length + bin_size - 1) // bin_size
    starts = np.arange(n_bins, dtype=np.int64) * np.int64(bin_size)
    ends = np.minimum(starts + np.int64(bin_size), np.int64(length))
    return starts, ends


def bin_counts(starts: np.ndarray, bin_size: int, n_bins: int) -> np.ndarray:
    """Count start positions into bins with a single ``numpy.bincount``."""

    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if starts.size == 0:
        return np.zeros(n_bins, dtype=np.uint64)
    indices = starts // bin_size
    valid = (indices >= 0) & (indices < n_bins)
    return np.bincount(indices[valid], minlength=n_bins).astype(np.uint64)


def rebin_counts(
    source_starts: np.ndarray,
    source_ends: np.ndarray,
    source_counts: np.ndarray,
    target_starts: np.ndarray,
    target_ends: np.ndarray,
) -> np.ndarray:
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


def rebin_sample_table(table: pa.Table, target_bin_size: int) -> pa.Table:
    """Rescale sample Parquet counts to target_bin_size without changing length."""
    if target_bin_size <= 0:
        raise ValueError("target_bin_size must be a positive integer")
    metadata = decode_metadata(table.schema.metadata)
    current_bin_size = metadata.get("bin_size")
    if current_bin_size == target_bin_size:
        return table

    contig_lengths = metadata.get("contig_lengths")
    if not isinstance(contig_lengths, dict) or not contig_lengths:
        raise ValueError("table metadata lacks contig_lengths")

    contig_arr = np.asarray(table.column("contig"))
    start_col = np.asarray(
        table.column("start").to_numpy(zero_copy_only=False), dtype=np.int64
    )
    end_col = np.asarray(
        table.column("end").to_numpy(zero_copy_only=False), dtype=np.int64
    )
    count_col = np.asarray(
        table.column("count").to_numpy(zero_copy_only=False), dtype=np.float64
    )

    new_contigs: list[np.ndarray] = []
    new_starts: list[np.ndarray] = []
    new_ends: list[np.ndarray] = []
    new_counts: list[np.ndarray] = []

    if len(contig_arr) == 0:
        contig_slices: dict[str, tuple[int, int]] = {}
    else:
        changes = np.flatnonzero(contig_arr[1:] != contig_arr[:-1]) + 1
        starts = np.concatenate(([0], changes))
        ends = np.concatenate((changes, [len(contig_arr)]))
        contig_slices = {
            str(contig_arr[s]): (int(s), int(e))
            for s, e in zip(starts, ends, strict=True)
        }

    for contig_name, length in contig_lengths.items():
        target_starts, target_ends = bin_intervals(int(length), target_bin_size)
        span = contig_slices.get(contig_name)
        if span is None:
            target_counts = np.zeros(target_starts.size, dtype=np.float64)
        else:
            s, e = span
            s_starts = start_col[s:e]
            s_ends = end_col[s:e]
            s_counts = count_col[s:e]
            target_counts = rebin_counts(
                s_starts, s_ends, s_counts, target_starts, target_ends
            )
        new_contigs.append(np.full(target_starts.size, contig_name, dtype=object))
        new_starts.append(target_starts)
        new_ends.append(target_ends)
        new_counts.append(target_counts)

    meta_values = {
        key: val
        for key, val in metadata.items()
        if key
        not in (
            "artifact_kind",
            "schema_version",
            "coordinate_system",
            "contig_lengths",
        )
    }
    meta_values["bin_size"] = target_bin_size
    new_metadata = metadata_for(
        ArtifactKind.SAMPLE,
        contig_lengths={str(k): int(v) for k, v in contig_lengths.items()},
        values=meta_values,
    )
    new_schema = pa.schema(
        [
            pa.field("contig", pa.string(), nullable=False),
            pa.field("start", pa.int64(), nullable=False),
            pa.field("end", pa.int64(), nullable=False),
            pa.field("count", pa.float64(), nullable=False),
        ]
    ).with_metadata(new_metadata)

    empty_str = np.array([], dtype=object)
    empty_i64 = np.empty(0, dtype=np.int64)
    empty_f64 = np.empty(0, dtype=np.float64)
    return pa.table(
        {
            "contig": pa.array(
                np.concatenate(new_contigs) if new_contigs else empty_str,
                type=pa.string(),
            ),
            "start": pa.array(
                np.concatenate(new_starts) if new_starts else empty_i64,
                type=pa.int64(),
            ),
            "end": pa.array(
                np.concatenate(new_ends) if new_ends else empty_i64,
                type=pa.int64(),
            ),
            "count": pa.array(
                np.concatenate(new_counts) if new_counts else empty_f64,
                type=pa.float64(),
            ),
        },
        schema=new_schema,
    )


@contextmanager
def _binned_contigs(
    tasks: Sequence[_ContigTask],
    workers: int,
    handle: pysam.AlignmentFile,
) -> Iterator[Iterator[tuple[np.ndarray, dict[str, int]]]]:
    """Yield per-contig results in task order, serially or across worker processes."""

    if workers <= 1 or len(tasks) < 2:
        yield (bin_contig_with_handle(handle, task) for task in tasks)
        return
    logger.debug("binning %d contigs across %d worker processes", len(tasks), workers)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        yield pool.map(_bin_contig, tasks)


def run(config: ConvertConfig) -> None:
    """Convert one BAM/CRAM into canonical sample Parquet."""

    alignment = config.alignment.expanduser()
    if not alignment.is_file():
        raise ConvertError(f"alignment not found: {alignment}")

    fmt = _alignment_format(alignment)
    logger.debug("opening alignment %s as %s", alignment, fmt)
    bam = _open_alignment(alignment, fmt, config.fasta)
    try:
        if bam.has_index():
            logger.debug("%s already has an index", alignment)
        bam = _ensure_index(alignment, fmt, bam, config.fasta, threads=1)
        header_names = list(bam.references)
        header_lengths = [int(length) for length in bam.lengths]
        if not header_names:
            raise ConvertError(f"{alignment} header declares no contigs")
        if any(length <= 0 for length in header_lengths):
            raise ConvertError(
                f"{alignment} header contains non-positive contig lengths"
            )
        layout = select_layout(
            header_names, header_lengths, all_contigs=config.all_contigs
        )
        if not layout:
            raise ConvertError(
                f"{alignment} header declares no canonical chromosomes; "
                "pass --all-contigs to bin every contig"
            )
        profile = GenomeProfile(layout)
        logger.debug(
            "binning %d of %d header contigs; skipped %d non-canonical",
            len(layout),
            len(header_names),
            len(header_names) - len(layout),
        )
        logger.debug(
            "selected contigs (%d bp total): %s",
            sum(c.length for c in layout),
            ", ".join(c.name for c in layout[:8]) + ("..." if len(layout) > 8 else ""),
        )
        logger.debug(
            "read filters: min_mapping_quality=%d, exclude_duplicates=%s, "
            "exclude_secondary=%s, exclude_supplementary=%s, require_proper_pair=%s",
            config.filters.min_mapping_quality,
            config.filters.exclude_duplicates,
            config.filters.exclude_secondary,
            config.filters.exclude_supplementary,
            config.filters.require_proper_pair,
        )
        contig_lengths = profile.length_map()
        workers, decode_threads = split_thread_budget(config.threads, len(layout))
        logger.debug(
            "using %d worker processes and %d htslib threads per handle",
            workers,
            decode_threads,
        )
        if workers <= 1 and decode_threads > 1:
            bam.close()
            bam = _open_alignment(alignment, fmt, config.fasta, threads=decode_threads)

        qc = empty_qc()
        qc["header_mapped"] = int(bam.mapped)
        qc["header_unmapped"] = int(bam.unmapped)
        qc["header_no_coordinate"] = int(bam.nocoordinate)

        tasks = [
            _ContigTask(
                path=alignment,
                fmt=fmt,
                fasta=config.fasta,
                name=contig.name,
                length=contig.length,
                bin_size=config.bin_size,
                filters=config.filters,
                decode_threads=decode_threads,
            )
            for contig in layout
        ]

        contig_arrays: list[np.ndarray] = []
        start_arrays: list[np.ndarray] = []
        end_arrays: list[np.ndarray] = []
        count_arrays: list[np.ndarray] = []
        with _binned_contigs(tasks, workers, bam) as results:
            for task, (counts, contig_qc) in zip(tasks, results, strict=True):
                for counter, value in contig_qc.items():
                    qc[counter] += value
                norm = normalize_chr(task.name)
                if norm in {"X", "Y"}:
                    qc["sex_reads_kept"] += contig_qc["reads_kept"]
                elif norm.isdigit():
                    qc["autosomal_reads_kept"] += contig_qc["reads_kept"]
                starts, ends = bin_intervals(task.length, config.bin_size)
                logger.debug(
                    "binned contig %s (%d bp): %d bins, %d reads kept",
                    task.name,
                    task.length,
                    starts.size,
                    contig_qc["reads_kept"],
                )
                logger.debug(
                    "contig %s filter drops: %d unmapped, %d low_mapq, "
                    "%d duplicates, %d secondary, %d supplementary",
                    task.name,
                    contig_qc["filter_unmapped"],
                    contig_qc["filter_mapq"],
                    contig_qc["filter_duplicate"],
                    contig_qc["filter_secondary"],
                    contig_qc["filter_supplementary"],
                )
                contig_arrays.append(np.full(starts.size, task.name, dtype=object))
                start_arrays.append(starts)
                end_arrays.append(ends)
                count_arrays.append(counts)
    finally:
        bam.close()

    qc["effective_reads_examined"] = qc["reads_seen"]
    qc["effective_reads_kept"] = qc["reads_kept"]

    logger.debug(
        "convert QC summary: %d reads examined, %d kept (%.1f%%), %d duplicate, "
        "%d low_mapq, %d unmapped",
        qc["reads_seen"],
        qc["reads_kept"],
        (qc["reads_kept"] / qc["reads_seen"] * 100) if qc["reads_seen"] else 0.0,
        qc["filter_duplicate"],
        qc["filter_mapq"],
        qc["filter_unmapped"],
    )
    logger.debug(
        "convert QC effective: %d autosomal kept, %d sex kept",
        qc["autosomal_reads_kept"],
        qc["sex_reads_kept"],
    )

    table = _sample_table(
        contig_arrays,
        start_arrays,
        end_arrays,
        count_arrays,
        contig_lengths=contig_lengths,
        bin_size=config.bin_size,
        source_identity={
            "path": str(alignment.resolve()),
            "format": fmt,
        },
        reference_identity=_reference_identity(config.fasta, profile.names),
        qc=qc,
        filters=asdict(config.filters),
    )
    output = parquet_path(config.prefix)
    logger.debug("writing sample parquet %s (%d bins)", output, table.num_rows)
    write_sample_parquet(output, table)
    logger.debug(
        "wrote sample parquet %s (%.1f KB)",
        output,
        output.stat().st_size / 1024,
    )


def _reference_identity(
    fasta: Path | None, contig_names: Sequence[str]
) -> dict[str, object]:
    identity: dict[str, object] = {
        "n_contigs": len(contig_names),
        "contig_order": list(contig_names),
    }
    if fasta is not None:
        identity["fasta"] = str(fasta.expanduser().resolve())
    return identity


def _sample_table(
    contigs: Sequence[np.ndarray],
    starts: Sequence[np.ndarray],
    ends: Sequence[np.ndarray],
    counts: Sequence[np.ndarray],
    *,
    contig_lengths: Mapping[str, int],
    bin_size: int,
    source_identity: Mapping[str, object],
    reference_identity: Mapping[str, object],
    qc: Mapping[str, int],
    filters: Mapping[str, object],
) -> pa.Table:
    metadata = metadata_for(
        ArtifactKind.SAMPLE,
        contig_lengths=contig_lengths,
        values={
            "bin_size": bin_size,
            "source_identity": dict(source_identity),
            "reference_identity": dict(reference_identity),
            "qc": dict(qc),
            "filters": dict(filters),
        },
    )
    return pa.table(
        {
            "contig": pa.array(
                _concat(contigs, np.array([], dtype=object)), type=pa.string()
            ),
            "start": pa.array(
                _concat(starts, np.empty(0, dtype=np.int64)), type=pa.int64()
            ),
            "end": pa.array(
                _concat(ends, np.empty(0, dtype=np.int64)), type=pa.int64()
            ),
            "count": pa.array(
                _concat(counts, np.empty(0, dtype=np.uint64)), type=pa.uint64()
            ),
        },
        schema=SAMPLE_SCHEMA.with_metadata(metadata),
    )


def _concat(parts: Sequence[np.ndarray], empty: np.ndarray) -> np.ndarray:
    return np.concatenate(parts) if parts else empty
