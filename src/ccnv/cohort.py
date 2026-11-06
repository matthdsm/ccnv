import logging
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from ccnv.convert import bin_intervals, rebin_sample_table, select_layout
from ccnv.models import ReferenceError
from ccnv.schemas import ArtifactKind, SchemaError, decode_metadata, validate_table

logger = logging.getLogger(__name__)

_ParsedSample = tuple[
    dict[str, object],
    dict[str, int],
    tuple[str, ...],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
]


@dataclass(frozen=True, slots=True)
class Cohort:
    """Aligned binned counts from compatible sample Parquet files."""

    counts: NDArray[np.float64]
    contigs: tuple[str, ...]
    starts: NDArray[np.int64]
    ends: NDArray[np.int64]
    contig_lengths: dict[str, int]
    bin_size: int
    identities: tuple[object, ...]
    paths: tuple[Path, ...]


def _read_sample(path: Path) -> pa.Table:
    try:
        return pq.read_table(path)
    except (FileNotFoundError, OSError, pa.ArrowInvalid) as error:
        raise ReferenceError(f"{path}: cannot read sample Parquet ({error})") from error


def _parse_sample_table(
    table: pa.Table,
    label: str,
    target_bin_size: int | None = None,
) -> _ParsedSample:
    try:
        validate_table(table, ArtifactKind.SAMPLE)
    except SchemaError as error:
        raise ReferenceError(f"{label}: {error}") from error

    metadata = decode_metadata(table.schema.metadata)
    contig_lengths = metadata.get("contig_lengths")
    orig_bin_size = metadata.get("bin_size")
    if not isinstance(contig_lengths, dict) or not isinstance(orig_bin_size, int):
        raise ReferenceError(f"{label}: sample metadata is incomplete")

    effective_bin_size = (
        target_bin_size if target_bin_size is not None else orig_bin_size
    )
    if orig_bin_size != effective_bin_size:
        logger.debug(
            "rebinning cohort sample %s from %s bp to %s bp",
            Path(label).name,
            orig_bin_size,
            effective_bin_size,
        )
        table = rebin_sample_table(table, effective_bin_size)
        metadata = decode_metadata(table.schema.metadata)

    contigs = tuple(str(c) for c in table.column("contig").to_pylist())
    starts = np.asarray(
        table.column("start").to_numpy(zero_copy_only=False), dtype=np.int64
    )
    ends = np.asarray(
        table.column("end").to_numpy(zero_copy_only=False), dtype=np.int64
    )
    counts = np.asarray(
        table.column("count").to_numpy(zero_copy_only=False), dtype=np.float64
    )
    lengths = {str(k): int(v) for k, v in contig_lengths.items()}
    return metadata, lengths, contigs, starts, ends, counts


def _assemble_cohort(
    parsed: Sequence[_ParsedSample],
    paths: Sequence[Path],
    target_bin_size: int | None = None,
) -> Cohort:
    (
        first_meta,
        expected_lengths,
        expected_contigs,
        expected_starts,
        expected_ends,
        _,
    ) = parsed[0]
    expected_bin_size = int(first_meta["bin_size"])
    counts_list: list[NDArray[np.float64]] = []
    identities: list[object] = []

    for path, (metadata, lengths, contigs, starts, ends, counts) in zip(
        paths, parsed, strict=True
    ):
        bin_size = int(metadata["bin_size"])
        if lengths != expected_lengths:
            raise ReferenceError(f"{path}: contig_lengths do not match the cohort")
        if target_bin_size is None and bin_size != expected_bin_size:
            raise ReferenceError(
                f"{path}: bin_size {bin_size} does not match cohort {expected_bin_size}"
            )
        if (
            contigs != expected_contigs
            or not np.array_equal(starts, expected_starts)
            or not np.array_equal(ends, expected_ends)
        ):
            raise ReferenceError(f"{path}: bin coordinates do not match the cohort")

        logger.debug(
            "loaded cohort sample %s: %d bins, %d total depth",
            path.name,
            counts.size,
            int(np.sum(counts)),
        )
        counts_list.append(counts)
        identities.append(metadata.get("source_identity", {"path": str(path)}))

    return Cohort(
        counts=np.column_stack(counts_list),
        contigs=expected_contigs,
        starts=expected_starts,
        ends=expected_ends,
        contig_lengths=expected_lengths,
        bin_size=expected_bin_size,
        identities=tuple(identities),
        paths=tuple(paths),
    )


def _load_sample_worker(args: tuple[Path, int | None]) -> _ParsedSample:
    path, target_bin_size = args
    return _parse_sample_table(_read_sample(path), str(path), target_bin_size)


def load_cohort(
    paths: tuple[Path, ...],
    target_bin_size: int | None = None,
    threads: int = 1,
) -> Cohort:
    """Load sample Parquet files and reject incompatible cohorts."""
    if not paths:
        raise ReferenceError("at least one sample Parquet path is required")

    args = [(path, target_bin_size) for path in paths]
    if threads > 1 and len(paths) > 1:
        max_workers = min(threads, len(paths))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_load_sample_worker, args))
    else:
        results = [_load_sample_worker(arg) for arg in args]

    return _assemble_cohort(results, paths, target_bin_size)


def validate_cohort(
    tables: tuple[pa.Table, ...],
    paths: tuple[Path, ...] | None = None,
    target_bin_size: int | None = None,
) -> Cohort:
    """Require matching contig lengths across samples, rebinning counts if set."""
    if len(tables) < 1:
        raise ReferenceError("cohort must contain at least one sample")
    resolved_paths = paths or tuple(
        Path(f"sample_{index}") for index in range(len(tables))
    )
    if len(resolved_paths) != len(tables):
        raise ReferenceError("paths must align with sample tables")

    parsed = [
        _parse_sample_table(table, str(path), target_bin_size)
        for path, table in zip(resolved_paths, tables, strict=True)
    ]
    return _assemble_cohort(parsed, resolved_paths, target_bin_size)


def cohort_from_fasta(
    fasta: Path,
    bin_size: int,
    *,
    n_samples: int,
    all_contigs: bool = False,
) -> Cohort:
    """Build identical uniform-depth samples from FASTA contig lengths."""
    if n_samples < 1:
        raise ReferenceError("n_samples must be a positive integer")
    names, lengths = _fasta_contigs(fasta, all_contigs=all_contigs)
    contigs: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for name, length in zip(names, lengths, strict=True):
        contig_starts, contig_ends = bin_intervals(length, bin_size)
        contigs.extend([name] * contig_starts.size)
        starts.extend(int(value) for value in contig_starts)
        ends.extend(int(value) for value in contig_ends)
    start_array = np.asarray(starts, dtype=np.int64)
    end_array = np.asarray(ends, dtype=np.int64)
    expected = np.maximum(end_array - start_array, 1).astype(np.float64)
    counts = np.broadcast_to(expected[:, np.newaxis], (expected.size, n_samples)).copy()
    resolved = fasta.expanduser().resolve()
    identities = tuple(
        {
            "path": str(resolved),
            "format": "flat-fasta",
            "replica": index,
        }
        for index in range(n_samples)
    )
    return Cohort(
        counts=counts,
        contigs=tuple(contigs),
        starts=start_array,
        ends=end_array,
        contig_lengths=dict(zip(names, lengths, strict=True)),
        bin_size=bin_size,
        identities=identities,
        paths=tuple(Path(f"{resolved}#flat-{index}") for index in range(n_samples)),
    )


def _fasta_contigs(path: Path, *, all_contigs: bool) -> tuple[list[str], list[int]]:
    import pysam

    fasta = path.expanduser()
    if not fasta.is_file():
        raise ReferenceError(f"FASTA not found: {fasta}")
    if not Path(f"{fasta}.fai").is_file():
        logger.debug("creating FASTA index for %s", fasta)
        try:
            pysam.faidx(str(fasta))
        except Exception as error:
            raise ReferenceError(f"failed to index FASTA {fasta}: {error}") from error
    handle = pysam.FastaFile(str(fasta))
    try:
        header_names = list(handle.references)
        header_lengths = [int(length) for length in handle.lengths]
    finally:
        handle.close()
    if any(length <= 0 for length in header_lengths):
        raise ReferenceError(f"{fasta} contains non-positive contig lengths")
    layout = select_layout(header_names, header_lengths, all_contigs=all_contigs)
    if not layout:
        raise ReferenceError(
            f"{fasta} declares no canonical chromosomes; "
            "pass --all-contigs to include every contig"
        )
    return [contig.name for contig in layout], [contig.length for contig in layout]
