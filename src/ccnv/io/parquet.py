"""Arrow Parquet table reading, writing, and contract validation for ccnv."""

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import overload

import pyarrow as pa
import pyarrow.parquet as pq

from ccnv.models import ArtifactKind, ReferenceError, SchemaError, validate_table

__all__ = [
    "load_cohort",
    "read_reference_parquet",
    "read_sample_parquet",
    "read_sample_tables",
    "write_reference_parquet",
    "write_sample_parquet",
]


def read_sample_parquet(path: Path | str) -> pa.Table:
    """Read a sample Parquet table from disk and validate its schema contract."""
    p = Path(path)
    table = pq.read_table(p)
    validate_table(table, ArtifactKind.SAMPLE)
    return table


@overload
def write_sample_parquet(table: pa.Table, path: Path | str) -> Path: ...


@overload
def write_sample_parquet(path: Path | str, table: pa.Table) -> Path: ...


def write_sample_parquet(
    arg1: pa.Table | Path | str,
    arg2: Path | str | pa.Table,
) -> Path:
    """Validate sample table contract and persist as a zstd-compressed Parquet table.

    Supports both (table, path) and legacy (path, table) calling conventions.
    """
    if isinstance(arg1, pa.Table):
        table = arg1
        path = Path(arg2)
    elif isinstance(arg2, pa.Table):
        table = arg2
        path = Path(arg1)
    else:
        raise TypeError("write_sample_parquet requires a pa.Table and a Path")

    validate_table(table, ArtifactKind.SAMPLE)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return path


def read_reference_parquet(path: Path | str) -> pa.Table:
    """Read a reference Parquet table from disk and validate its schema contract."""
    p = Path(path)
    table = pq.read_table(p)
    validate_table(table, ArtifactKind.REFERENCE)
    return table


@overload
def write_reference_parquet(
    table: pa.Table,
    path: Path | str,
    *,
    compression_level: int = 3,
) -> Path: ...


@overload
def write_reference_parquet(
    path: Path | str,
    table: pa.Table,
    *,
    compression_level: int = 3,
) -> Path: ...


def write_reference_parquet(
    arg1: pa.Table | Path | str,
    arg2: Path | str | pa.Table,
    *,
    compression_level: int = 3,
) -> Path:
    """Validate reference table contract and persist with dictionary and delta encoding.

    Supports both (table, path) and legacy (path, table) calling conventions.
    """
    if isinstance(arg1, pa.Table):
        table = arg1
        path = Path(arg2)
    elif isinstance(arg2, pa.Table):
        table = arg2
        path = Path(arg1)
    else:
        raise TypeError("write_reference_parquet requires a pa.Table and a Path")

    validate_table(table, ArtifactKind.REFERENCE)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=compression_level,
        use_dictionary=["branch", "contig"],
        column_encoding={
            "start": "DELTA_BINARY_PACKED",
            "end": "DELTA_BINARY_PACKED",
            "nearest_ref_indices.list.element": "DELTA_BINARY_PACKED",
        },
    )
    return path


def _read_sample_for_cohort(path: Path | str) -> pa.Table:
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
