"""Stream BAM/CRAM alignments into canonical sample Parquet bins."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pyarrow as pa
import pysam

from ccnv.features.binning import bin_counts, bin_intervals, rebin_counts
from ccnv.features.cohort import rebin_sample_table
from ccnv.io.alignments import (
    alignment_format,
    ensure_index,
    open_alignment,
    rejection_reason,
    stream_contig_reads,
)
from ccnv.io.parquet import read_sample_parquet, write_sample_parquet
from ccnv.models import (
    ConvertConfig,
    ConvertError,
    GenomeProfile,
    ReadFilters,
    is_canonical_contig,
    normalize_chr,
    parquet_path,
    select_layout,
)
from ccnv.qc.sample import QC_COUNTERS, build_sample_qc, empty_qc
from ccnv.schemas import SAMPLE_SCHEMA, ArtifactKind, metadata_for

logger = logging.getLogger(__name__)

_alignment_format, _ensure_index, _open_alignment = (
    alignment_format,
    ensure_index,
    open_alignment,
)
_REEXPORTS = (
    QC_COUNTERS,
    is_canonical_contig,
    read_sample_parquet,
    rebin_counts,
    rebin_sample_table,
    rejection_reason,
)

# fmt: off
__all__ = [
    "QC_COUNTERS", "ConvertError", "bin_contig_with_handle", "bin_counts",
    "bin_intervals", "empty_qc", "is_canonical_contig", "normalize_chr",
    "read_sample_parquet", "rebin_counts", "rebin_sample_table",
    "rejection_reason", "run", "run_convert", "select_layout",
    "split_thread_budget", "write_sample_parquet",
]
# fmt: on


@dataclass(frozen=True, slots=True)
class _ContigTask:
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
    """Split CPU budget into process workers and per-handle htslib threads."""
    if budget <= 0 or n_tasks <= 0:
        raise ValueError("thread budget and n_tasks must be positive integers")
    if budget == 1 or n_tasks == 1:
        return 1, budget
    workers = min(n_tasks, max(1, budget // (2 if budget >= 4 else 1)))
    return workers, max(1, budget // workers)


def _worker_handle(t: _ContigTask) -> pysam.AlignmentFile:
    k = (str(t.path), t.fmt, str(t.fasta) if t.fasta else None, t.decode_threads)
    if k not in _WORKER_HANDLES:
        _WORKER_HANDLES[k] = _open_alignment(
            t.path, t.fmt, t.fasta, threads=t.decode_threads
        )
    return _WORKER_HANDLES[k]


def _bin_contig(t: _ContigTask) -> tuple[np.ndarray, dict[str, int]]:
    return bin_contig_with_handle(_worker_handle(t), t)


def bin_contig_with_handle(
    handle: pysam.AlignmentFile, task: _ContigTask
) -> tuple[np.ndarray, dict[str, int]]:
    starts, _ = bin_intervals(task.length, task.bin_size)
    qc = empty_qc()
    reads = list(
        stream_contig_reads(
            handle, task.name, task.filters, contig_length=task.length, qc=qc
        )
    )
    return bin_counts(np.array(reads, dtype=np.int64), task.bin_size, starts.size), qc


@contextmanager
def _binned_contigs(
    tasks: Sequence[_ContigTask], workers: int, handle: pysam.AlignmentFile
) -> Iterator[Iterator[tuple[np.ndarray, dict[str, int]]]]:
    if workers <= 1 or len(tasks) < 2:
        yield (bin_contig_with_handle(handle, t) for t in tasks)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            yield pool.map(_bin_contig, tasks)


def run(config: ConvertConfig) -> None:
    """Convert one BAM/CRAM into canonical sample Parquet."""
    alignment = config.alignment.expanduser()
    if not alignment.is_file():
        raise ConvertError(f"alignment not found: {alignment}")

    fmt = alignment_format(alignment)
    logger.debug("opening alignment %s as %s", alignment, fmt)
    bam = _open_alignment(alignment, fmt, config.fasta)
    try:
        if bam.has_index():
            logger.debug("%s already has an index", alignment)
        bam = ensure_index(alignment, fmt, bam, config.fasta, threads=1)
        h_names, h_lens = list(bam.references), [int(x) for x in bam.lengths]
        if not h_names or any(length <= 0 for length in h_lens):
            raise ConvertError(f"{alignment} header declares invalid contigs")
        layout = select_layout(h_names, h_lens, all_contigs=config.all_contigs)
        if not layout:
            msg = (
                f"{alignment} header declares no canonical chromosomes; "
                "pass --all-contigs to bin every contig"
            )
            raise ConvertError(msg)
        profile = GenomeProfile(layout)
        workers, decode = split_thread_budget(config.threads, len(layout))
        if workers <= 1 and decode > 1:
            bam.close()
            bam = _open_alignment(alignment, fmt, config.fasta, threads=decode)

        qc = empty_qc()
        qc["header_mapped"], qc["header_unmapped"], qc["header_no_coordinate"] = (
            int(bam.mapped),
            int(bam.unmapped),
            int(bam.nocoordinate),
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
        cf, flt, bs = config.fasta, config.filters, config.bin_size
        tasks = [
            _ContigTask(alignment, fmt, cf, c.name, c.length, bs, flt, decode)
            for c in layout
        ]
        c_arrs, s_arrs, e_arrs, cnt_arrs = [], [], [], []
        with _binned_contigs(tasks, workers, bam) as results:
            for task, (counts, c_qc) in zip(tasks, results, strict=True):
                for k, v in c_qc.items():
                    qc[k] += v
                norm = normalize_chr(task.name)
                key = (
                    "sex_reads_kept"
                    if norm in {"X", "Y"}
                    else "autosomal_reads_kept"
                    if norm.isdigit()
                    else None
                )
                if key:
                    qc[key] += c_qc["reads_kept"]
                starts, ends = bin_intervals(task.length, config.bin_size)
                logger.debug(
                    "binned contig %s (%d bp): %d bins, %d reads kept",
                    task.name,
                    task.length,
                    starts.size,
                    c_qc["reads_kept"],
                )
                logger.debug(
                    "contig %s filter drops: %d unmapped, %d low_mapq, "
                    "%d duplicates, %d secondary, %d supplementary",
                    task.name,
                    c_qc["filter_unmapped"],
                    c_qc["filter_mapq"],
                    c_qc["filter_duplicate"],
                    c_qc["filter_secondary"],
                    c_qc["filter_supplementary"],
                )
                c_arrs.append(np.full(starts.size, task.name, dtype=object))
                s_arrs.append(starts)
                e_arrs.append(ends)
                cnt_arrs.append(counts)
    finally:
        bam.close()

    qc = build_sample_qc(qc)
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

    ref_id = {
        "n_contigs": len(profile.names),
        "contig_order": list(profile.names),
        **({"fasta": str(config.fasta.expanduser().resolve())} if config.fasta else {}),
    }
    metadata = metadata_for(
        ArtifactKind.SAMPLE,
        contig_lengths=profile.length_map(),
        values={
            "bin_size": config.bin_size,
            "source_identity": {"path": str(alignment.resolve()), "format": fmt},
            "reference_identity": ref_id,
            "qc": dict(qc),
            "filters": dict(asdict(config.filters)),
        },
    )
    cols = {
        "contig": pa.array(np.concatenate(c_arrs) if c_arrs else [], type=pa.string()),
        "start": pa.array(np.concatenate(s_arrs) if s_arrs else [], type=pa.int64()),
        "end": pa.array(np.concatenate(e_arrs) if e_arrs else [], type=pa.int64()),
        "count": pa.array(
            np.concatenate(cnt_arrs) if cnt_arrs else [], type=pa.uint64()
        ),
    }
    table = pa.table(cols, schema=SAMPLE_SCHEMA.with_metadata(metadata))
    output = parquet_path(config.prefix)
    logger.debug("writing sample parquet %s (%d bins)", output, table.num_rows)
    write_sample_parquet(output, table)
    logger.debug(
        "wrote sample parquet %s (%.1f KB)",
        output,
        output.stat().st_size / 1024,
    )


run_convert: Final = run
