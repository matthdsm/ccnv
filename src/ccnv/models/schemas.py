"""Versioned Arrow contracts and artifact metadata for ccnv Parquet tables."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from numpy.typing import NDArray

from ccnv.models.exceptions import SchemaError, SchemaVersionError
from ccnv.models.metadata import (
    COORDINATE_SYSTEM,
    FRAGMENTOMICS_METADATA_KEY,
    METADATA_PREFIX,
    PRENATAL_METADATA_KEY,
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    ArtifactKind,
    FragmentomicsMetadata,
    PrenatalModelMetadata,
    ReferenceMetadata,
    SampleMetadata,
    decode_metadata,
    encode_metadata,
)

NUM_FRAG_BANDS: Final[int] = 24
NUM_PROTOTYPES_PER_BIN: Final[int] = 4
NUM_GC_QUANTILES: Final[int] = 8

BRANCH_AUTOSOMAL: Final[str] = "autosomal"
BRANCH_FEMALE_SEX: Final[str] = "female_sex"
BRANCH_MALE_SEX: Final[str] = "male_sex"
BRANCH_PRENATAL_AUTOSOMAL: Final[str] = "prenatal_autosomal"
BRANCH_PRENATAL_FFY: Final[str] = "prenatal_ffy"

KNOWN_REFERENCE_BRANCHES: Final[frozenset[str]] = frozenset(
    {
        BRANCH_AUTOSOMAL,
        BRANCH_FEMALE_SEX,
        BRANCH_MALE_SEX,
        BRANCH_PRENATAL_AUTOSOMAL,
        BRANCH_PRENATAL_FFY,
    }
)
CNV_REFERENCE_BRANCHES: Final[frozenset[str]] = frozenset(
    {BRANCH_AUTOSOMAL, BRANCH_FEMALE_SEX, BRANCH_MALE_SEX}
)

SAMPLE_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field("count", pa.uint32(), nullable=False),
        pa.field("paired_count", pa.uint32(), nullable=False),
        pa.field("frag_len_sum", pa.uint64(), nullable=False),
        pa.field("short_frag_count", pa.uint32(), nullable=False),
        pa.field("endpoint_count", pa.uint32(), nullable=False),
        pa.field("frag_gc_sum", pa.float32(), nullable=False),
        pa.field("frag_bands", pa.list_(pa.uint32(), NUM_FRAG_BANDS), nullable=True),
    ]
)


def build_reference_schema(pca_rank: int) -> pa.Schema:
    """Build Schema v2.0.0 reference Arrow schema parameterized by PCA rank."""
    if not isinstance(pca_rank, int) or isinstance(pca_rank, bool) or pca_rank < 0:
        raise ValueError(f"pca_rank must be a non-negative integer, got {pca_rank!r}")
    return pa.schema(
        [
            pa.field("branch", pa.dictionary(pa.int8(), pa.string()), nullable=False),
            pa.field("contig", pa.dictionary(pa.int16(), pa.string()), nullable=False),
            pa.field("start", pa.uint32(), nullable=False),
            pa.field("end", pa.uint32(), nullable=False),
            pa.field("masked", pa.bool_(), nullable=False),
            pa.field("gc", pa.float32(), nullable=False),
            pa.field("mappability", pa.float32(), nullable=False),
            pa.field(
                "gc_quantiles", pa.list_(pa.float32(), NUM_GC_QUANTILES), nullable=True
            ),
            pa.field("ref_frag_len", pa.float32(), nullable=False),
            pa.field("ref_ep_rate", pa.float32(), nullable=False),
            pa.field("pca_mean", pa.float32(), nullable=False),
            pa.field(
                "pca_loadings",
                pa.list_(pa.float32(), pca_rank),
                nullable=(pca_rank == 0),
            ),
            pa.field(
                "proto_ids",
                pa.list_(pa.uint16(), NUM_PROTOTYPES_PER_BIN),
                nullable=False,
            ),
            pa.field(
                "proto_weights",
                pa.list_(pa.float16(), NUM_PROTOTYPES_PER_BIN),
                nullable=False,
            ),
            pa.field("mean_distance", pa.float32(), nullable=False),
            pa.field("null_mean", pa.float32(), nullable=False),
            pa.field("bio_dispersion", pa.float32(), nullable=False),
        ]
    )


REFERENCE_SCHEMA: Final[pa.Schema] = build_reference_schema(0)


def reconstruct_sample_grid(
    meta: SampleMetadata,
) -> tuple[NDArray[np.int16], tuple[str, ...], NDArray[np.uint32], NDArray[np.uint32]]:
    """Reconstruct implicit (contig_codes, contig_names, starts, ends) bin grid."""
    names = meta.contig_order
    if len(names) > np.iinfo(np.int16).max:
        raise SchemaError(f"contig count {len(names)} exceeds int16 dictionary bounds")
    bin_size_u64 = np.uint64(meta.bin_size)
    max_u32 = int(np.iinfo(np.uint32).max)

    counts_per_contig = np.empty(len(names), dtype=np.int64)
    starts_parts: list[NDArray[np.uint32]] = []
    ends_parts: list[NDArray[np.uint32]] = []
    for idx, name in enumerate(names):
        length = meta.contig_lengths[name]
        if length > max_u32:
            raise SchemaError(f"contig {name!r} length {length} exceeds uint32 bounds")
        n_bins = (length + meta.bin_size - 1) // meta.bin_size
        counts_per_contig[idx] = n_bins
        s64 = np.arange(n_bins, dtype=np.uint64) * bin_size_u64
        e64 = np.minimum(s64 + bin_size_u64, np.uint64(length))
        starts_parts.append(s64.astype(np.uint32))
        ends_parts.append(e64.astype(np.uint32))

    codes = np.repeat(np.arange(len(names), dtype=np.int16), counts_per_contig)
    starts = (
        np.concatenate(starts_parts) if starts_parts else np.empty(0, dtype=np.uint32)
    )
    ends = np.concatenate(ends_parts) if ends_parts else np.empty(0, dtype=np.uint32)
    return codes, names, starts, ends


def bin_coordinates(table: pa.Table) -> tuple[tuple[str, int, int], ...]:
    """Return row identity as ``(contig, start, end)`` tuples."""
    if "contig" not in table.column_names:
        codes, names, starts, ends = reconstruct_sample_grid(
            SampleMetadata.from_table(table)
        )
        return tuple(
            (names[int(c)], int(s), int(e))
            for c, s, e in zip(codes, starts, ends, strict=True)
        )
    c_dict = (
        pc.dictionary_encode(table.column("contig"))
        .unify_dictionaries()
        .combine_chunks()
    )
    if c_dict.dictionary.null_count > 0 or c_dict.null_count > 0:
        raise SchemaError("column 'contig' contains null entries")
    c_codes = c_dict.indices.to_numpy(zero_copy_only=False)
    c_names = tuple(
        str(c_dict.dictionary[i].as_py()) for i in range(len(c_dict.dictionary))
    )
    starts_arr = table.column("start").combine_chunks().to_numpy(zero_copy_only=False)
    ends_arr = table.column("end").combine_chunks().to_numpy(zero_copy_only=False)
    return tuple(
        (c_names[int(c)], int(s), int(e))
        for c, s, e in zip(c_codes, starts_arr, ends_arr, strict=True)
    )


def metadata_for(
    kind: ArtifactKind,
    *,
    contig_lengths: Mapping[str, int],
    values: Mapping[str, object],
) -> dict[bytes, bytes]:
    """Build complete metadata for an artifact, preserving sample contig order."""
    extra: dict[str, object] = {}
    if kind is ArtifactKind.SAMPLE and "contig_order" not in values:
        extra["contig_order"] = list(contig_lengths.keys())
    metadata: dict[str, object] = {
        "artifact_kind": kind.value,
        "schema_version": SCHEMA_VERSION,
        "coordinate_system": COORDINATE_SYSTEM,
        "contig_lengths": dict(contig_lengths),
        **extra,
        **values,
    }
    validate_metadata(metadata, kind)
    return encode_metadata(metadata)


def validate_metadata(metadata: Mapping[str, object], kind: ArtifactKind) -> None:
    """Validate required keys and cross-artifact coordinate invariants."""
    from ccnv.models.validation import validate_metadata as _impl

    _impl(metadata, kind)


def validate_schema(schema: pa.Schema, kind: ArtifactKind) -> None:
    """Validate Arrow fields and embedded file metadata."""
    from ccnv.models.validation import validate_schema as _impl

    _impl(schema, kind)


def validate_sample_table(table: pa.Table) -> SampleMetadata:
    """Validate compact Schema v2.0.0 sample table and implicit coordinate grid."""
    from ccnv.models.validation import validate_sample_table as _impl

    return _impl(table)


def validate_reference_table(
    table: pa.Table, *, check_rows: bool = True
) -> ReferenceMetadata:
    """Validate Schema v2.0.0 reference table, metadata, and optional row intervals."""
    from ccnv.models.validation import validate_reference_table as _impl

    return _impl(table, check_rows=check_rows)


def fixed_size_list_to_numpy[ScalarT: np.generic](
    col: pa.Array | pa.ChunkedArray,
    list_size: int,
    dtype: type[ScalarT],
) -> NDArray[ScalarT]:
    """Extract a FixedSizeListArray or ChunkedArray into a 2D (N, list_size) NumPy array."""
    arr = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    n = len(arr)
    np_dtype = np.dtype(dtype)
    if n == 0 or list_size == 0:
        return np.empty((n, list_size), dtype=np_dtype)
    is_float = bool(np.issubdtype(np_dtype, np.floating))
    if arr.null_count == n:
        return np.full((n, list_size), np.nan if is_float else 0, dtype=np_dtype)
    if not isinstance(arr, pa.FixedSizeListArray):
        raise SchemaError(f"expected FixedSizeListArray, found {type(arr).__name__}")
    child = arr.values.slice(arr.offset * list_size, n * list_size)
    if child.null_count > 0 and not is_float:
        child = pc.fill_null(child, 0)
    flat = child.to_numpy(zero_copy_only=False)
    out: NDArray[ScalarT] = np.asarray(flat, dtype=np_dtype).reshape(n, list_size)
    if arr.null_count > 0:
        out = out.copy()
        null_mask = arr.is_null().to_numpy(zero_copy_only=False)
        out[null_mask] = np.nan if is_float else 0
    return out


def validate_table(table: pa.Table, kind: ArtifactKind) -> None:
    """Validate schema plus canonical coordinate bounds and row identity."""
    from ccnv.models.validation import validate_table as _impl

    _impl(table, kind)


__all__ = [
    "BRANCH_AUTOSOMAL",
    "BRANCH_FEMALE_SEX",
    "BRANCH_MALE_SEX",
    "BRANCH_PRENATAL_AUTOSOMAL",
    "BRANCH_PRENATAL_FFY",
    "CNV_REFERENCE_BRANCHES",
    "COORDINATE_SYSTEM",
    "FRAGMENTOMICS_METADATA_KEY",
    "KNOWN_REFERENCE_BRANCHES",
    "METADATA_PREFIX",
    "NUM_FRAG_BANDS",
    "NUM_GC_QUANTILES",
    "NUM_PROTOTYPES_PER_BIN",
    "PRENATAL_METADATA_KEY",
    "REFERENCE_SCHEMA",
    "SAMPLE_SCHEMA",
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "ArtifactKind",
    "FragmentomicsMetadata",
    "PrenatalModelMetadata",
    "ReferenceMetadata",
    "SampleMetadata",
    "SchemaError",
    "SchemaVersionError",
    "bin_coordinates",
    "build_reference_schema",
    "decode_metadata",
    "encode_metadata",
    "fixed_size_list_to_numpy",
    "metadata_for",
    "reconstruct_sample_grid",
    "validate_metadata",
    "validate_reference_table",
    "validate_sample_table",
    "validate_schema",
    "validate_table",
]
