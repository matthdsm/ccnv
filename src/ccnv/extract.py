"""Stream BAM/CRAM alignments into canonical sample Parquet bins."""

from __future__ import annotations

import json
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

from ccnv.features.binning import bin_contig_fragments, bin_intervals
from ccnv.io.alignments import (
    alignment_format,
    ensure_index,
    open_alignment,
    stream_contig_fragments,
)
from ccnv.io.parquet import write_sample_parquet
from ccnv.models import (
    FRAGMENTOMICS_METADATA_KEY,
    SAMPLE_SCHEMA,
    ArtifactKind,
    ExtractConfig,
    ExtractError,
    GenomeProfile,
    ReadFilters,
    metadata_for,
    normalize_chr,
    parquet_path,
    select_layout,
)
from ccnv.qc.sample import build_sample_qc, empty_qc
from ccnv.stats.fragmentomics import (
    FRAG_MAX_LEN,
    FRAG_MIN_LEN,
    FRAG_NUM_BINS,
    estimate_sample_landmarks,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ExtractError",
    "bin_contig_with_handle",
    "extract_sample_table",
    "run",
    "run_extract",
    "split_thread_budget",
]


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
    min_frag_len: int = 30
    max_frag_len: int = 600


_WORKER_HANDLES: dict[tuple[str, str, str | None, int], pysam.AlignmentFile] = {}


def split_thread_budget(budget: int, n_tasks: int) -> tuple[int, int]:
    """Split CPU budget into process workers and per-handle htslib threads."""
    if budget <= 0 or n_tasks <= 0:
        raise ValueError("thread budget and n_tasks must be positive integers")
    if budget == 1 or n_tasks == 1:
        return 1, budget
    workers = min(n_tasks, max(1, budget // (2 if budget >= 4 else 1)))
    return workers, max(1, budget // workers)


def _worker_handle(task: _ContigTask) -> pysam.AlignmentFile:
    """Retrieve or open a cached per-process AlignmentFile handle."""
    key = (
        str(task.path),
        task.fmt,
        str(task.fasta) if task.fasta else None,
        task.decode_threads,
    )
    if key not in _WORKER_HANDLES:
        _WORKER_HANDLES[key] = open_alignment(
            task.path, task.fmt, task.fasta, threads=task.decode_threads
        )
    return _WORKER_HANDLES[key]


def _bin_contig(
    task: _ContigTask,
) -> tuple[
    np.ndarray,
    pa.Array,
    np.ndarray,
    np.ndarray,
    dict[str, int],
    np.ndarray,
    np.ndarray,
    bool,
]:
    """Bin reads for a contig task using a cached worker alignment handle."""
    return bin_contig_with_handle(_worker_handle(task), task)


def bin_contig_with_handle(
    handle: pysam.AlignmentFile, task: _ContigTask
) -> tuple[
    np.ndarray,
    pa.Array,
    np.ndarray,
    np.ndarray,
    dict[str, int],
    np.ndarray,
    np.ndarray,
    bool,
]:
    """Stream and bin reads for a contig using an open alignment handle."""
    starts, _ = bin_intervals(task.length, task.bin_size)
    qc = empty_qc()
    m_list: list[int] = []
    s_list: list[int] = []
    e_list: list[int] = []
    l_list: list[int] = []
    gc_list: list[float] = []
    contig_hist = np.zeros(FRAG_NUM_BINS, dtype=np.int64)
    contig_motifs = np.zeros(256, dtype=np.int64)
    paired_detected = False

    for m, s, e, length, gc_frac, motif, is_paired in stream_contig_fragments(
        handle,
        task.name,
        task.filters,
        contig_length=task.length,
        min_frag_len=task.min_frag_len,
        max_frag_len=task.max_frag_len,
        qc=qc,
    ):
        m_list.append(m)
        s_list.append(s)
        e_list.append(e)
        l_list.append(length)
        gc_list.append(gc_frac)
        if is_paired:
            paired_detected = True
            if FRAG_MIN_LEN <= length <= FRAG_MAX_LEN:
                contig_hist[length - FRAG_MIN_LEN] += 1
        if motif >= 0:
            contig_motifs[motif] += 1

    counts, frag_arr, ep_counts, gc_sums = bin_contig_fragments(
        np.array(m_list, dtype=np.int64),
        np.array(s_list, dtype=np.int64),
        np.array(e_list, dtype=np.int64),
        np.array(l_list, dtype=np.uint16),
        np.array(gc_list, dtype=np.float32),
        is_paired=paired_detected,
        n_bins=starts.size,
        bin_size=task.bin_size,
    )
    return (
        counts,
        frag_arr,
        ep_counts,
        gc_sums,
        qc,
        contig_hist,
        contig_motifs,
        paired_detected,
    )


@contextmanager
def _binned_contigs(
    tasks: Sequence[_ContigTask], workers: int, handle: pysam.AlignmentFile
) -> Iterator[
    Iterator[
        tuple[
            np.ndarray,
            pa.Array,
            np.ndarray,
            np.ndarray,
            dict[str, int],
            np.ndarray,
            np.ndarray,
            bool,
        ]
    ]
]:
    """Yield an iterator of binned contig counts across single-process or worker pool."""
    if workers <= 1 or len(tasks) < 2:
        yield (bin_contig_with_handle(handle, task) for task in tasks)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            yield pool.map(_bin_contig, tasks)


def extract_sample_table(
    config: ExtractConfig,
) -> tuple[pa.Table, dict[str, object]]:
    """Extract sample table and QC metadata in memory from one BAM/CRAM."""
    alignment = config.alignment.expanduser()
    if not alignment.is_file():
        raise ExtractError(f"alignment not found: {alignment}")

    fmt = alignment_format(alignment)
    logger.debug("opening alignment %s as %s", alignment, fmt)
    bam = open_alignment(alignment, fmt, config.fasta)

    try:
        if bam.has_index():
            logger.debug("%s already has an index", alignment)
        bam = ensure_index(alignment, fmt, bam, config.fasta, threads=1)
        header_names = list(bam.references)
        header_lengths = [int(length) for length in bam.lengths]
        if not header_names or any(length <= 0 for length in header_lengths):
            raise ExtractError(f"{alignment} header declares invalid contigs")
        layout = select_layout(
            header_names, header_lengths, all_contigs=config.all_contigs
        )
        if not layout:
            msg = (
                f"{alignment} header declares no canonical chromosomes; "
                "pass --all-contigs to bin every contig"
            )
            raise ExtractError(msg)
        profile = GenomeProfile(layout)
        workers, decode_threads = split_thread_budget(config.threads, len(layout))
        if workers <= 1 and decode_threads > 1:
            bam.close()
            bam = open_alignment(alignment, fmt, config.fasta, threads=decode_threads)

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
        tasks = [
            _ContigTask(
                alignment,
                fmt,
                config.fasta,
                contig.name,
                contig.length,
                config.bin_size,
                config.filters,
                decode_threads,
                config.min_frag_len,
                config.max_frag_len,
            )
            for contig in layout
        ]
        contig_arrays = []
        start_arrays = []
        end_arrays = []
        count_arrays = []
        frag_arrays = []
        ep_arrays = []
        gc_arrays = []

        hist_autosomal = np.zeros(FRAG_NUM_BINS, dtype=np.int64)
        hist_chry = np.zeros(FRAG_NUM_BINS, dtype=np.int64)
        hist_chrx = np.zeros(FRAG_NUM_BINS, dtype=np.int64)
        global_motifs = np.zeros(256, dtype=np.int64)
        any_paired = False

        with _binned_contigs(tasks, workers, bam) as results:
            for task, (
                counts,
                frag_arr,
                ep_counts,
                gc_sums,
                contig_qc,
                contig_hist,
                contig_motifs,
                paired_detected,
            ) in zip(tasks, results, strict=True):
                for counter, value in contig_qc.items():
                    qc[counter] += value
                norm_chr = normalize_chr(task.name)
                key = (
                    "sex_reads_kept"
                    if norm_chr in {"X", "Y"}
                    else "autosomal_reads_kept"
                    if norm_chr.isdigit()
                    else None
                )
                if key:
                    qc[key] += contig_qc["reads_kept"]

                if norm_chr.isdigit():
                    hist_autosomal += contig_hist
                elif norm_chr == "Y":
                    hist_chry += contig_hist
                elif norm_chr == "X":
                    hist_chrx += contig_hist
                global_motifs += contig_motifs
                if paired_detected:
                    any_paired = True

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
                frag_arrays.append(frag_arr)
                ep_arrays.append(ep_counts)
                gc_arrays.append(gc_sums)
    finally:
        bam.close()

    qc = build_sample_qc(qc)
    logger.debug(
        "extract QC summary: %d reads examined, %d kept (%.1f%%), %d duplicate, "
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

    meta_frag = estimate_sample_landmarks(
        hist_autosomal=hist_autosomal,
        hist_chry=hist_chry,
        hist_chrx=hist_chrx,
        end_motif_counts=global_motifs,
        paired_end=any_paired,
    )

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
    metadata[FRAGMENTOMICS_METADATA_KEY] = json.dumps(asdict(meta_frag)).encode("utf-8")

    columns = {
        "contig": pa.array(
            np.concatenate(contig_arrays) if contig_arrays else [], type=pa.string()
        ),
        "start": pa.array(
            np.concatenate(start_arrays) if start_arrays else [], type=pa.int64()
        ),
        "end": pa.array(
            np.concatenate(end_arrays) if end_arrays else [], type=pa.int64()
        ),
        "count": pa.array(
            np.concatenate(count_arrays) if count_arrays else [], type=pa.uint64()
        ),
        "fragment_lengths": (
            pa.concat_arrays(frag_arrays)
            if frag_arrays
            else pa.array([], type=pa.list_(pa.uint16()))
        ),
        "endpoint_count": pa.array(
            np.concatenate(ep_arrays) if ep_arrays else [], type=pa.uint32()
        ),
        "frag_gc_sum": pa.array(
            np.concatenate(gc_arrays) if gc_arrays else [], type=pa.float32()
        ),
    }
    table = pa.table(columns, schema=SAMPLE_SCHEMA.with_metadata(metadata))
    return table, qc


def run(config: ExtractConfig) -> None:
    """Extract binned counts from one BAM/CRAM into canonical sample Parquet."""
    table, _ = extract_sample_table(config)
    output = parquet_path(config.prefix)
    logger.debug("writing sample parquet %s (%d bins)", output, table.num_rows)
    write_sample_parquet(table, output)
    logger.debug(
        "wrote sample parquet %s (%.1f KB)",
        output,
        output.stat().st_size / 1024,
    )


run_extract: Final = run
