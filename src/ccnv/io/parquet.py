"""Arrow Parquet table reading, writing, and contract validation for ccnv."""

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ccnv.models import (
    ArtifactKind,
    ReferenceError,
    SchemaError,
    decode_metadata,
    validate_table,
)

__all__ = [
    "is_sample_parquet",
    "load_cohort",
    "load_sample_or_extract",
    "read_reference_parquet",
    "read_sample_parquet",
    "read_sample_tables",
    "write_reference_parquet",
    "write_sample_parquet",
]


def is_sample_parquet(path: Path | str) -> bool:
    """Return True if path is a readable Parquet file with sample artifact metadata."""
    try:
        p = Path(path)
        metadata = decode_metadata(pq.read_schema(p).metadata)
    except OSError, TypeError, ValueError, SchemaError, pa.ArrowException:
        return False
    return metadata.get("artifact_kind") == ArtifactKind.SAMPLE.value


def read_sample_parquet(path: Path | str) -> pa.Table:
    """Read a sample Parquet table from disk and validate its schema contract."""
    p = Path(path)
    table = pq.read_table(p)
    validate_table(table, ArtifactKind.SAMPLE)
    return table


def load_sample_or_extract(
    path: Path | str,
    *,
    fasta: Path | str | None = None,
    bin_size: int = 5000,
    threads: int = 1,
) -> pa.Table:
    """Load a sample Parquet table from disk or extract directly in memory from BAM/CRAM."""
    p = Path(path)
    if p.suffix.lower() in {".bam", ".cram"}:
        from ccnv.extract import extract_sample_table
        from ccnv.models import ExtractConfig

        fasta_path = Path(fasta) if fasta is not None else None
        config = ExtractConfig(
            alignment=p,
            prefix=p.with_suffix(""),
            fasta=fasta_path,
            bin_size=bin_size,
            threads=threads,
        )
        table, _ = extract_sample_table(config)
        return table
    return read_sample_parquet(p)


def write_sample_parquet(table: pa.Table, path: Path | str) -> Path:
    """Validate sample table contract and persist as a zstd-compressed Parquet table."""
    validate_table(table, ArtifactKind.SAMPLE)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="zstd")
    return out_path


def read_reference_parquet(path: Path | str) -> pa.Table:
    """Read a reference Parquet table from disk and validate its schema contract."""
    p = Path(path)
    table = pq.read_table(p)
    validate_table(table, ArtifactKind.REFERENCE)
    return table


def write_reference_parquet(
    table: pa.Table,
    path: Path | str,
    *,
    compression_level: int = 3,
) -> Path:
    """Validate reference table contract and persist with encodings."""
    validate_table(table, ArtifactKind.REFERENCE)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        out_path,
        compression="zstd",
        compression_level=compression_level,
        use_dictionary=["branch", "contig"],
        column_encoding={
            "start": "DELTA_BINARY_PACKED",
            "end": "DELTA_BINARY_PACKED",
            "nearest_ref_indices.list.element": "DELTA_BINARY_PACKED",
        },
    )
    return out_path


def _read_sample_for_cohort(path: Path | str) -> pa.Table:
    """Read and validate a single sample Parquet table for cohort aggregation."""
    p = Path(path)
    try:
        table = pq.read_table(p)
    except (FileNotFoundError, OSError, pa.ArrowInvalid) as error:
        raise ReferenceError(f"{p}: cannot read sample Parquet ({error})") from error
    try:
        validate_table(table, ArtifactKind.SAMPLE)
    except SchemaError as error:
        raise ReferenceError(f"{p}: {error}") from error
    return table


def read_sample_tables(
    paths: Sequence[Path | str],
    *,
    threads: int = 1,
) -> tuple[pa.Table, ...]:
    """Read and validate multiple sample Parquet files into memory."""
    if not paths:
        raise ReferenceError("at least one sample Parquet path is required")
    resolved = tuple(Path(p) for p in paths)
    if threads > 1 and len(resolved) > 1:
        max_workers = min(threads, len(resolved))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            tables = list(executor.map(_read_sample_for_cohort, resolved))
    else:
        tables = [_read_sample_for_cohort(p) for p in resolved]
    return tuple(tables)


load_cohort = read_sample_tables
