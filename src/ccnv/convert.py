"""Stream BAM/CRAM alignments into canonical sample Parquet bins."""

import logging
import re
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pysam

from ccnv.models import Contig, ConvertConfig, GenomeProfile, ReadFilters, parquet_path
from ccnv.schemas import (
    SAMPLE_SCHEMA,
    ArtifactKind,
    metadata_for,
    validate_table,
)

logger = logging.getLogger(__name__)

CANONICAL_AUTOSOMES: Final[int] = 22
_REFSEQ_CHROMOSOME: Final[re.Pattern[str]] = re.compile(r"NC_0000(\d{2})\.\d+")
_REFSEQ_LAST: Final[int] = 24
"""``NC_000023`` is X and ``NC_000024`` is Y; mitochondrial DNA is ``NC_012920``."""

QC_COUNTERS: Final[tuple[str, ...]] = (
    "header_mapped",
    "header_unmapped",
    "header_no_coordinate",
    "reads_seen",
    "reads_kept",
    "filter_unmapped",
    "filter_no_start",
    "filter_secondary",
    "filter_supplementary",
    "filter_duplicate",
    "filter_improper_pair",
    "filter_mapq",
)


class ConvertError(ValueError):
    """Raised when conversion cannot proceed with the given inputs."""


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
    kept_starts = _kept_starts(handle.fetch(task.name), task.filters, qc, task.length)
    return bin_counts(kept_starts, task.bin_size, starts.size), qc


def empty_qc() -> dict[str, int]:
    """Return a complete QC counter map with deterministic key order."""

    return {name: 0 for name in QC_COUNTERS}


def select_layout(
    names: Sequence[str],
    lengths: Sequence[int],
    *,
    all_contigs: bool,
) -> tuple[Contig, ...]:
    """Keep canonical chromosomes unless ``all_contigs`` is set.

    The result may be empty; callers decide how to report that. Header
    names and lengths must already be aligned.
    """

    if len(names) != len(lengths):
        raise ValueError("contig names and lengths must have equal length")
    return tuple(
        Contig(name, length)
        for name, length in zip(names, lengths, strict=True)
        if all_contigs or is_canonical_contig(name)
    )


def is_canonical_contig(name: str) -> bool:
    """Report whether a contig is a canonical autosome or sex chromosome.

    Accepts UCSC (``chr1``, ``chrX``), plain NCBI/Ensembl (``1``, ``X``), and
    RefSeq accession (``NC_000001.11``) spellings. Mitochondrial contigs, ALT
    and decoy scaffolds, unplaced scaffolds, and EBV are not canonical.
    """

    bare = name[3:] if name[:3].lower() == "chr" else name
    if bare.isdigit():
        return 1 <= int(bare) <= CANONICAL_AUTOSOMES
    if bare.upper() in {"X", "Y"}:
        return True
    match = _REFSEQ_CHROMOSOME.fullmatch(name)
    return match is not None and 1 <= int(match.group(1)) <= _REFSEQ_LAST


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


def rejection_reason(read: object, filters: ReadFilters) -> str | None:
    """Return the first matching QC counter name, or ``None`` if kept."""

    if filters.exclude_unmapped and bool(getattr(read, "is_unmapped", False)):
        return "filter_unmapped"
    start = getattr(read, "reference_start", None)
    if start is None or int(start) < 0:
        return "filter_no_start"
    if filters.exclude_secondary and bool(getattr(read, "is_secondary", False)):
        return "filter_secondary"
    if filters.exclude_supplementary and bool(getattr(read, "is_supplementary", False)):
        return "filter_supplementary"
    if filters.exclude_duplicates and bool(getattr(read, "is_duplicate", False)):
        return "filter_duplicate"
    if (
        filters.require_proper_pair
        and bool(getattr(read, "is_paired", False))
        and not bool(getattr(read, "is_proper_pair", False))
    ):
        return "filter_improper_pair"
    mapping_quality = int(getattr(read, "mapping_quality", 0))
    if mapping_quality < filters.min_mapping_quality:
        return "filter_mapq"
    return None


def write_sample_parquet(path: Path, table: pa.Table) -> None:
    """Validate the sample contract, then persist Parquet."""

    validate_table(table, ArtifactKind.SAMPLE)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def read_sample_parquet(path: Path) -> pa.Table:
    """Read sample Parquet and reject contract violations."""

    table = pq.read_table(path)
    validate_table(table, ArtifactKind.SAMPLE)
    return table


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
                starts, ends = bin_intervals(task.length, config.bin_size)
                logger.debug(
                    "binned contig %s (%d bp): %d bins, %d reads kept",
                    task.name,
                    task.length,
                    starts.size,
                    contig_qc["reads_kept"],
                )
                contig_arrays.append(np.full(starts.size, task.name, dtype=object))
                start_arrays.append(starts)
                end_arrays.append(ends)
                count_arrays.append(counts)
    finally:
        bam.close()

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


def _alignment_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".bam":
        return "bam"
    if suffix == ".cram":
        return "cram"
    raise ConvertError(f"{path} must be BAM or CRAM (found suffix {path.suffix!r})")


def _open_alignment(
    path: Path,
    fmt: str,
    fasta: Path | None,
    threads: int = 1,
) -> pysam.AlignmentFile:
    decode_threads = max(1, threads)
    if fmt == "cram":
        if fasta is None:
            raise ConvertError(
                f"{path} is CRAM and requires a reference FASTA. "
                "Pass --fasta /path/to/reference.fa (indexed with samtools faidx)."
            )
        if not fasta.is_file():
            raise ConvertError(f"CRAM reference FASTA not found: {fasta}")
        return pysam.AlignmentFile(
            str(path),
            "rc",
            reference_filename=str(fasta),
            threads=decode_threads,
        )
    return pysam.AlignmentFile(str(path), "rb", threads=decode_threads)


def _ensure_index(
    path: Path,
    fmt: str,
    handle: pysam.AlignmentFile,
    fasta: Path | None,
    threads: int = 1,
) -> pysam.AlignmentFile:
    if handle.has_index():
        logger.debug("%s already has an index", path)
        return handle
    handle.close()
    kind = "CRAM" if fmt == "cram" else "BAM"
    logger.debug("creating %s index for %s", kind, path)
    try:
        pysam.index(str(path))
    except Exception as error:
        raise ConvertError(
            f"failed to create {kind} index for {path}: {error}"
        ) from error
    handle = _open_alignment(path, fmt, fasta, threads=threads)
    if handle.has_index():
        return handle
    handle.close()
    raise ConvertError(f"failed to create {kind} index for {path}")


def _kept_starts(
    reads: Iterator[object],
    filters: ReadFilters,
    qc: dict[str, int],
    contig_length: int,
) -> np.ndarray:
    starts: list[int] = []
    for read in reads:
        qc["reads_seen"] += 1
        reason = rejection_reason(read, filters)
        if reason is not None:
            qc[reason] += 1
            continue
        start = int(read.reference_start)
        if start >= contig_length:
            qc["filter_no_start"] += 1
            continue
        starts.append(start)
        qc["reads_kept"] += 1
    if not starts:
        return np.empty(0, dtype=np.int64)
    return np.fromiter(starts, dtype=np.int64, count=len(starts))


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
