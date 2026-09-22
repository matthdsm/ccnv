"""Stream BAM/CRAM alignments into canonical sample Parquet bins."""

import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import polars_bio as pb
import pyarrow as pa
import pyarrow.parquet as pq

from ccnv.models import Contig, ConvertConfig, GenomeProfile, ReadFilters, parquet_path
from ccnv.schemas import (
    SAMPLE_SCHEMA,
    ArtifactKind,
    decode_metadata,
    metadata_for,
    validate_table,
)

logger = logging.getLogger(__name__)

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


_WORKER_LF: dict[tuple[str, str, str | None, int], pl.LazyFrame] = {}


def split_thread_budget(budget: int, n_tasks: int) -> tuple[int, int]:
    """Split a CPU budget into process workers and decode threads.

    The product of the returned values is at most ``budget``.
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


def filter_and_bin_contig(
    lf: pl.LazyFrame,
    contig: Contig,
    bin_size: int,
    n_bins: int,
    filters: ReadFilters,
) -> tuple[np.ndarray, dict[str, int]]:
    """Filter reads and compute bin counts and QC statistics for one contig."""

    qc = empty_qc()
    contig_df = (
        lf.filter(pl.col("chrom") == contig.name)
        .select(["start", "flags", "mapping_quality"])
        .collect()
    )
    n_reads = len(contig_df)
    qc["reads_seen"] = n_reads
    if n_reads == 0:
        return np.zeros(n_bins, dtype=np.uint64), qc

    conditions = []
    if filters.exclude_unmapped:
        conditions.append(((pl.col("flags") & 4) != 0, "filter_unmapped"))
    conditions.append((pl.col("start").is_null(), "filter_no_start"))
    if filters.exclude_secondary:
        conditions.append(((pl.col("flags") & 256) != 0, "filter_secondary"))
    if filters.exclude_supplementary:
        conditions.append(((pl.col("flags") & 2048) != 0, "filter_supplementary"))
    if filters.exclude_duplicates:
        conditions.append(((pl.col("flags") & 1024) != 0, "filter_duplicate"))
    if filters.require_proper_pair:
        conditions.append(
            (
                ((pl.col("flags") & 1) != 0) & ((pl.col("flags") & 2) == 0),
                "filter_improper_pair",
            )
        )
    if filters.min_mapping_quality > 0:
        conditions.append(
            (
                pl.col("mapping_quality") < filters.min_mapping_quality,
                "filter_mapq",
            )
        )
    conditions.append((pl.col("start") >= contig.length, "filter_no_start"))

    expr = pl.when(conditions[0][0]).then(pl.lit(conditions[0][1]))
    for cond, res in conditions[1:]:
        expr = expr.when(cond).then(pl.lit(res))
    expr = expr.otherwise(pl.lit("kept")).alias("reason")

    classified = contig_df.with_columns(expr)
    reason_counts = classified["reason"].value_counts()
    for row in reason_counts.iter_rows():
        r_name, r_cnt = row[0], row[1]
        if r_name in qc:
            qc[r_name] = r_cnt
        elif r_name == "kept":
            qc["reads_kept"] = r_cnt

    kept_df = classified.filter(pl.col("reason") == "kept")
    kept_starts = kept_df["start"].to_numpy()
    counts = bin_counts(kept_starts, bin_size, n_bins)
    return counts, qc


def _worker_scan(task: _ContigTask) -> pl.LazyFrame:
    """Reuse one alignment LazyFrame per worker process, keyed by input file."""

    key = (
        str(task.path),
        task.fmt,
        str(task.fasta) if task.fasta else None,
        task.decode_threads,
    )
    lf = _WORKER_LF.get(key)
    if lf is None:
        pb.set_option(pb.POLARS_BIO_MAX_THREADS, str(max(1, task.decode_threads)))
        if task.fmt == "cram":
            lf = pb.scan_cram(
                str(task.path),
                reference_path=str(task.fasta) if task.fasta else None,
                use_zero_based=True,
            )
        else:
            lf = pb.scan_bam(str(task.path), use_zero_based=True)
        _WORKER_LF[key] = lf
    return lf


def bin_contig_with_lf(
    lf: pl.LazyFrame, task: _ContigTask
) -> tuple[np.ndarray, dict[str, int]]:
    """Count reads for one contig and return its bin counts plus QC counters."""

    starts, _ = bin_intervals(task.length, task.bin_size)
    return filter_and_bin_contig(
        lf=lf,
        contig=Contig(task.name, task.length),
        bin_size=task.bin_size,
        n_bins=starts.size,
        filters=task.filters,
    )


def _bin_contig(task: _ContigTask) -> tuple[np.ndarray, dict[str, int]]:
    """Bin one contig inside a worker process."""

    return bin_contig_with_lf(_worker_scan(task), task)


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

    Accepts UCSC (``chr1``, ``chrX``) and plain NCBI/Ensembl (``1``, ``X``) spellings.
    Mitochondrial contigs, ALT/decoy scaffolds, unplaced scaffolds, and EBV
    are not canonical.
    """

    bare = name[3:] if name[:3].lower() == "chr" else name
    return (bare.isdigit() and 1 <= int(bare) <= 22) or bare.upper() in {"X", "Y"}


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

    contig_col = table.column("contig").to_pylist()
    start_col = np.asarray(table.column("start").to_pylist(), dtype=np.int64)
    end_col = np.asarray(table.column("end").to_pylist(), dtype=np.int64)
    count_col = np.asarray(table.column("count").to_pylist(), dtype=np.float64)

    new_contigs: list[np.ndarray] = []
    new_starts: list[np.ndarray] = []
    new_ends: list[np.ndarray] = []
    new_counts: list[np.ndarray] = []

    contig_indices: dict[str, list[int]] = {}
    for idx, c in enumerate(contig_col):
        contig_indices.setdefault(c, []).append(idx)

    for contig_name, length in contig_lengths.items():
        target_starts, target_ends = bin_intervals(int(length), target_bin_size)
        idxs = contig_indices.get(contig_name)
        if idxs is None or len(idxs) == 0:
            target_counts = np.zeros(target_starts.size, dtype=np.float64)
        else:
            s_starts = start_col[idxs]
            s_ends = end_col[idxs]
            s_counts = count_col[idxs]
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


def _scan_alignment(
    path: Path,
    fmt: str,
    fasta: Path | None,
    threads: int = 1,
) -> tuple[pl.LazyFrame, list[str], list[int]]:
    pb.set_option(pb.POLARS_BIO_MAX_THREADS, str(max(1, threads)))
    if fmt == "cram":
        if fasta is None:
            raise ConvertError(
                f"{path} is CRAM and requires a reference FASTA. "
                "Pass --fasta /path/to/reference.fa."
            )
        fasta_path = fasta.expanduser()
        if not fasta_path.is_file():
            raise ConvertError(f"CRAM reference FASTA not found: {fasta_path}")
        fai_path = Path(f"{fasta_path}.fai")
        if not fai_path.is_file():
            raise ConvertError(
                f"FASTA index not found: {fai_path}. "
                "CRAM decoding requires an indexed FASTA reference."
            )
        lf = pb.scan_cram(
            str(path), reference_path=str(fasta_path), use_zero_based=True
        )
    else:
        lf = pb.scan_bam(str(path), use_zero_based=True)

    meta = pb.get_metadata(lf)
    header = meta.get("header") or {}
    ref_seqs_raw = header.get("reference_sequences")
    if not ref_seqs_raw:
        raise ConvertError(f"{path} header declares no contigs")
    if isinstance(ref_seqs_raw, str):
        ref_seqs = json.loads(ref_seqs_raw)
    else:
        ref_seqs = ref_seqs_raw
    header_names = [s["name"] for s in ref_seqs]
    header_lengths = [int(s["length"]) for s in ref_seqs]
    return lf, header_names, header_lengths


@contextmanager
def _binned_contigs(
    tasks: Sequence[_ContigTask],
    workers: int,
    lf: pl.LazyFrame,
) -> Iterator[Iterator[tuple[np.ndarray, dict[str, int]]]]:
    """Yield per-contig results in task order, serially or across worker processes."""

    if workers <= 1 or len(tasks) < 2:
        yield (bin_contig_with_lf(lf, task) for task in tasks)
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
    lf, header_names, header_lengths = _scan_alignment(
        alignment, fmt, config.fasta, threads=config.threads
    )
    if not header_names:
        raise ConvertError(f"{alignment} header declares no contigs")
    if any(length <= 0 for length in header_lengths):
        raise ConvertError(f"{alignment} header contains non-positive contig lengths")
    layout = select_layout(header_names, header_lengths, all_contigs=config.all_contigs)
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
        "using %d worker processes and %d decode threads per handle",
        workers,
        decode_threads,
    )

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

    qc = empty_qc()
    contig_arrays: list[np.ndarray] = []
    start_arrays: list[np.ndarray] = []
    end_arrays: list[np.ndarray] = []
    count_arrays: list[np.ndarray] = []

    with _binned_contigs(tasks, workers, lf) as results:
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
