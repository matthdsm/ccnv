"""Arrow Parquet table reading, writing, and contract validation for ccnv."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ccnv.models import (
    ArtifactKind,
    ReferenceMetadata,
    SampleBuffers,
    SchemaError,
    canonical_dictionary_array,
    decode_metadata,
    validate_reference_table,
    validate_sample_table,
    validate_schema,
)

__all__ = [
    "is_sample_parquet",
    "read_reference_parquet",
    "read_sample_buffers",
    "read_sample_parquet",
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
    table = pq.read_table(Path(path))
    validate_sample_table(table)
    return table


def read_sample_buffers(path: Path | str) -> SampleBuffers:
    """Read and validate a sample Parquet artifact directly into SampleBuffers."""
    return SampleBuffers.from_table(pq.read_table(Path(path)), validate=True)


def write_sample_parquet(
    table: pa.Table,
    path: Path | str,
    *,
    compression_level: int = 3,
) -> Path:
    """Validate sample table contract and persist as a zstd-compressed Parquet table."""
    validate_sample_table(table)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        out_path,
        compression="zstd",
        compression_level=compression_level,
    )
    return out_path


def _decode_stat_str(val: object) -> str | None:
    if isinstance(val, str):
        return val
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8")
        except UnicodeDecodeError:
            return None
    return None


def _select_matching_row_groups(
    pf: pq.ParquetFile,
    target_branches: set[str],
) -> tuple[list[int], bool]:
    """Identify row group indices matching target_branches via metadata or branch column."""
    matching: list[int] = []
    needs_post_filter = False
    branch_col_idx = pf.schema_arrow.get_field_index("branch")

    for rg_idx in range(pf.num_row_groups):
        rg_meta = pf.metadata.row_group(rg_idx)
        if rg_meta.num_rows == 0:
            continue
        col_meta = rg_meta.column(branch_col_idx)
        stats = col_meta.statistics if col_meta.is_stats_set else None
        if stats is not None and stats.has_min_max:
            b_min = _decode_stat_str(stats.min)
            b_max = _decode_stat_str(stats.max)
            if b_min is not None and b_max is not None:
                if b_min == b_max:
                    if b_min in target_branches:
                        matching.append(rg_idx)
                    continue
                if any(b_min <= target <= b_max for target in target_branches):
                    matching.append(rg_idx)
                    needs_post_filter = True
                continue

        rg_branch_col = pf.read_row_group(rg_idx, columns=["branch"]).column(0)
        unique_vals = {
            str(v) for v in pc.unique(rg_branch_col).to_pylist() if v is not None
        }
        if unique_vals & target_branches:
            matching.append(rg_idx)
            if not unique_vals.issubset(target_branches):
                needs_post_filter = True

    return matching, needs_post_filter


def read_reference_parquet(
    path: Path | str,
    *,
    branches: Sequence[str] | None = None,
    check_rows: bool = False,
) -> tuple[pa.Table, ReferenceMetadata]:
    """Read a reference Parquet table (optionally branch-pruned) and return (table, metadata)."""
    pf = pq.ParquetFile(Path(path))
    validate_schema(pf.schema_arrow, ArtifactKind.REFERENCE)

    if branches is None:
        table = pf.read()
    else:
        target_set = set(branches)
        if not target_set or pf.num_row_groups == 0:
            table = pf.schema_arrow.empty_table()
        else:
            matching_rgs, needs_filter = _select_matching_row_groups(pf, target_set)
            if not matching_rgs:
                table = pf.schema_arrow.empty_table()
            else:
                table = pf.read_row_groups(matching_rgs)
                if needs_filter:
                    mask = pc.is_in(
                        table.column("branch"),
                        pa.array(list(target_set), type=pa.string()),
                    )
                    table = table.filter(mask)

    meta = validate_reference_table(table, check_rows=check_rows)
    return table, meta


def write_reference_parquet(
    table: pa.Table,
    path: Path | str,
    *,
    compression_level: int = 3,
) -> Path:
    """Validate reference table contract and write one ZSTD Parquet row group per branch."""
    for col_name, idx_type in (("branch", pa.int8()), ("contig", pa.int16())):
        if col_name in table.schema.names:
            col_idx = table.schema.get_field_index(col_name)
            col_field = table.schema.field(col_idx)
            if (
                pa.types.is_string(col_field.type)
                or pa.types.is_large_string(col_field.type)
            ) and table.column(col_idx).null_count == 0:
                canon = canonical_dictionary_array(
                    table.column(col_idx), col_name, index_type=idx_type
                )
                table = table.set_column(
                    col_idx,
                    pa.field(
                        col_name,
                        pa.dictionary(idx_type, pa.string()),
                        nullable=False,
                    ),
                    canon,
                )

    validate_reference_table(table, check_rows=True)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer_kwargs: dict[str, object] = {
        "compression": "zstd",
        "compression_level": compression_level,
        "use_dictionary": ["branch", "contig"],
        "column_encoding": {
            "start": "DELTA_BINARY_PACKED",
            "end": "DELTA_BINARY_PACKED",
        },
    }
    if table.num_rows == 0:
        pq.write_table(table, out_path, **writer_kwargs)
        return out_path

    branch_idx = table.schema.get_field_index("branch")
    branch_dict = canonical_dictionary_array(
        table.column(branch_idx), "branch", index_type=pa.int8()
    )
    table = table.set_column(branch_idx, table.schema.field(branch_idx), branch_dict)
    b_codes = branch_dict.indices.to_numpy(zero_copy_only=False)
    boundaries = (np.flatnonzero(b_codes[1:] != b_codes[:-1]) + 1).tolist()
    slice_starts = [0, *boundaries]
    slice_ends = [*boundaries, table.num_rows]

    with pq.ParquetWriter(out_path, table.schema, **writer_kwargs) as writer:
        for s_idx, e_idx in zip(slice_starts, slice_ends, strict=True):
            branch_slice = table.slice(s_idx, e_idx - s_idx)
            writer.write_table(
                branch_slice, row_group_size=max(branch_slice.num_rows, 1)
            )

    return out_path
