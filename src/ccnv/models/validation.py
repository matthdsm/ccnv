"""Vectorized Arrow schema, metadata, and coordinate-grid validation for ccnv."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Literal, overload

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ccnv.models.config import ReadFilters
from ccnv.models.exceptions import SchemaError, SchemaVersionError
from ccnv.models.metadata import (
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    ArtifactKind,
    ReferenceMetadata,
    SampleMetadata,
    decode_metadata,
)
from ccnv.models.schemas import (
    BRANCH_PRENATAL_AUTOSOMAL,
    KNOWN_REFERENCE_BRANCHES,
    SAMPLE_SCHEMA,
    build_reference_schema,
    canonical_dictionary_array,
)

_COMMON_METADATA: Final[frozenset[str]] = frozenset(
    {
        "artifact_kind",
        "schema_version",
        "coordinate_system",
        "contig_lengths",
    }
)
_KIND_METADATA: Final[Mapping[ArtifactKind, frozenset[str]]] = {
    ArtifactKind.SAMPLE: frozenset(
        {
            "bin_size",
            "source_identity",
            "reference_identity",
            "qc",
            "filters",
        }
    ),
    ArtifactKind.REFERENCE: frozenset(
        {
            "model",
            "sample_identities",
        }
    ),
}

__all__ = [
    "validate_metadata",
    "validate_reference_table",
    "validate_sample_table",
    "validate_schema",
]


def _check_version_header(metadata: Mapping[str, object], kind: ArtifactKind) -> None:
    """Fail fast with SchemaVersionError or wrong artifact_kind before field checks."""
    if "schema_version" in metadata:
        version = metadata["schema_version"]
        if not isinstance(version, str) or version not in SUPPORTED_SCHEMA_VERSIONS:
            raise SchemaVersionError(
                f"unsupported schema version {version!r}; expected {SCHEMA_VERSION!r}"
            )
    if "artifact_kind" in metadata and metadata["artifact_kind"] != kind.value:
        raise SchemaError(
            f"expected artifact_kind {kind.value!r}, got {metadata['artifact_kind']!r}"
        )


@overload
def validate_metadata(
    metadata: Mapping[str, object], kind: Literal[ArtifactKind.SAMPLE]
) -> SampleMetadata: ...


@overload
def validate_metadata(
    metadata: Mapping[str, object], kind: Literal[ArtifactKind.REFERENCE]
) -> ReferenceMetadata: ...


@overload
def validate_metadata(
    metadata: Mapping[str, object], kind: ArtifactKind
) -> SampleMetadata | ReferenceMetadata: ...


def validate_metadata(
    metadata: Mapping[str, object], kind: ArtifactKind
) -> SampleMetadata | ReferenceMetadata:
    """Validate required envelope keys, coordinate invariants, and typed metadata."""
    _check_version_header(metadata, kind)
    required = _COMMON_METADATA | _KIND_METADATA[kind]
    missing = sorted(required - metadata.keys())
    if missing:
        raise SchemaError(f"missing {kind.value} metadata: {', '.join(missing)}")

    if kind is ArtifactKind.SAMPLE:
        filters = metadata["filters"]
        if not isinstance(filters, Mapping):
            raise SchemaError("filters must be an object")
        ReadFilters.from_dict(filters, strict=True)
        return SampleMetadata.from_dict(metadata)

    model = metadata.get("model")
    if not isinstance(model, Mapping):
        raise SchemaError("model must be an object")
    return ReferenceMetadata.from_dict(metadata)


def _expected_reference_schema(schema: pa.Schema) -> pa.Schema:
    """Resolve expected reference schema parameterized by pca_loadings list size."""
    if "pca_loadings" not in schema.names:
        raise SchemaError(f"reference fields do not match schema {SCHEMA_VERSION}")
    loadings_type = schema.field("pca_loadings").type
    if (
        not isinstance(loadings_type, pa.FixedSizeListType)
        or loadings_type.value_type != pa.float32()
    ):
        raise SchemaError(f"reference fields do not match schema {SCHEMA_VERSION}")
    return build_reference_schema(loadings_type.list_size)


def _validate_arrow_fields(
    schema: pa.Schema, expected: pa.Schema, kind: ArtifactKind
) -> None:
    """Verify exact field names, order, and Arrow types."""
    if len(schema) != len(expected) or schema.names != expected.names:
        raise SchemaError(f"{kind.value} fields do not match schema {SCHEMA_VERSION}")
    for actual_field, expected_field in zip(schema, expected, strict=True):
        if actual_field.type != expected_field.type:
            raise SchemaError(
                f"{kind.value} fields do not match schema {SCHEMA_VERSION}"
            )


def _validate_non_null_columns(table: pa.Table, expected: pa.Schema) -> None:
    """Reject null entries in non-nullable columns and inner nulls in fixed-size lists."""
    for idx, expected_field in enumerate(expected):
        col = table.column(idx)
        if not expected_field.nullable and col.null_count > 0:
            raise SchemaError(
                f"column {expected_field.name!r} contains unexpected null values"
            )
        if (
            isinstance(expected_field.type, pa.FixedSizeListType)
            and expected_field.type.list_size > 0
        ):
            non_null_rows = pc.drop_null(col) if col.null_count > 0 else col
            if pc.list_flatten(non_null_rows).null_count > 0:
                raise SchemaError(
                    f"column {expected_field.name!r} contains null list elements"
                )


def validate_schema(schema: pa.Schema, kind: ArtifactKind) -> None:
    """Validate Arrow fields and embedded file metadata."""
    metadata = decode_metadata(schema.metadata)
    _check_version_header(metadata, kind)
    expected = (
        SAMPLE_SCHEMA
        if kind is ArtifactKind.SAMPLE
        else _expected_reference_schema(schema)
    )
    _validate_arrow_fields(schema, expected, kind)
    validate_metadata(metadata, kind)


def validate_sample_table(table: pa.Table) -> SampleMetadata:
    """Validate compact Schema v2.0.0 sample table and implicit coordinate grid."""
    metadata = decode_metadata(table.schema.metadata)
    _check_version_header(metadata, ArtifactKind.SAMPLE)
    _validate_arrow_fields(table.schema, SAMPLE_SCHEMA, ArtifactKind.SAMPLE)
    _validate_non_null_columns(table, SAMPLE_SCHEMA)
    sample_meta = validate_metadata(metadata, ArtifactKind.SAMPLE)

    expected_rows = sample_meta.expected_num_bins
    if table.num_rows != expected_rows:
        raise SchemaError(
            f"sample row count {table.num_rows} is outside expected grid size "
            f"{expected_rows} for bin_size={sample_meta.bin_size}"
        )
    return sample_meta


def _explicit_bin_size(
    metadata: Mapping[str, object], ref_meta: ReferenceMetadata
) -> int | None:
    model = metadata.get("model")
    if isinstance(model, Mapping) and model.get("bin_size") is not None:
        return ref_meta.bin_size
    return None


def _validate_reference_rows(
    table: pa.Table,
    metadata: Mapping[str, object],
    ref_meta: ReferenceMetadata,
) -> None:
    """Validate contiguous (branch, contig) runs, coordinate bounds, and bin spacing."""
    if table.num_rows == 0:
        return

    bin_size = _explicit_bin_size(metadata, ref_meta)
    prenatal_bin_size = ref_meta.prenatal_bin_size

    branch_dict = canonical_dictionary_array(table.column("branch"), "branch")
    contig_dict = canonical_dictionary_array(table.column("contig"), "contig")
    b_vocab = [str(v) for v in branch_dict.dictionary.to_pylist()]
    c_vocab = [str(v) for v in contig_dict.dictionary.to_pylist()]

    b_codes = branch_dict.indices.to_numpy(zero_copy_only=False).astype(
        np.int32, copy=False
    )
    c_codes = contig_dict.indices.to_numpy(zero_copy_only=False).astype(
        np.int32, copy=False
    )
    starts = (
        table.column("start")
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
        .astype(np.int64, copy=False)
    )
    ends = (
        table.column("end")
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
        .astype(np.int64, copy=False)
    )

    target_size_by_bcode = np.zeros(len(b_vocab), dtype=np.int64)
    for b_code_raw in np.unique(b_codes):
        b_idx = int(b_code_raw)
        b_name = b_vocab[b_idx]
        if b_name not in KNOWN_REFERENCE_BRANCHES:
            raise SchemaError(
                f"unknown branch {b_name!r}; "
                f"expected one of {sorted(KNOWN_REFERENCE_BRANCHES)}"
            )
        if b_name == BRANCH_PRENATAL_AUTOSOMAL and prenatal_bin_size is None:
            raise SchemaError(
                "table contains 'prenatal_autosomal' branch but prenatal_bin_size "
                "is not specified in metadata"
            )
        ts = prenatal_bin_size if b_name == BRANCH_PRENATAL_AUTOSOMAL else bin_size
        target_size_by_bcode[b_idx] = ts if ts is not None else 0

    contig_len_by_code = np.zeros(len(c_vocab), dtype=np.int64)
    for c_code_raw in np.unique(c_codes):
        c_idx = int(c_code_raw)
        c_name = c_vocab[c_idx]
        c_len = ref_meta.contig_lengths.get(c_name)
        if c_len is None:
            raise SchemaError(f"undeclared contig {c_name!r}")
        contig_len_by_code[c_idx] = c_len

    branch_run_starts = np.concatenate(
        (np.array([0], dtype=np.int64), np.flatnonzero(b_codes[1:] != b_codes[:-1]) + 1)
    )
    if np.unique(b_codes[branch_run_starts]).size != branch_run_starts.size:
        raise SchemaError("reference branches must be stored in contiguous blocks")

    pair_codes = (b_codes.astype(np.int64) << 32) | (
        c_codes.astype(np.int64) & np.int64(0xFFFF_FFFF)
    )
    same_run = pair_codes[1:] == pair_codes[:-1]
    run_starts = np.concatenate(
        (np.array([0], dtype=np.int64), np.flatnonzero(~same_run) + 1)
    )
    if np.unique(pair_codes[run_starts]).size != run_starts.size:
        raise SchemaError(
            "non-contiguous or duplicate (branch, contig) runs in reference table"
        )

    row_contig_lens = contig_len_by_code[c_codes]
    bad_bounds = (starts < 0) | (ends <= starts) | (ends > row_contig_lens)
    if np.any(bad_bounds):
        idx = int(np.flatnonzero(bad_bounds)[0])
        c_name = c_vocab[int(c_codes[idx])]
        raise SchemaError(
            f"row {idx} interval [{int(starts[idx])}, {int(ends[idx])}) is outside "
            f"{c_name!r} length {int(row_contig_lens[idx])}"
        )

    adj_mismatch = same_run & (starts[1:] != ends[:-1])
    if np.any(adj_mismatch):
        idx = int(np.flatnonzero(adj_mismatch)[0]) + 1
        raise SchemaError(
            f"row {idx} start {int(starts[idx])} does not match previous bin end "
            f"{int(ends[idx - 1])} within (branch, contig) run"
        )

    row_ts = target_size_by_bcode[b_codes]
    has_ts = row_ts > 0
    if np.any(has_ts):
        widths = ends - starts
        is_final = ends == row_contig_lens
        safe_ts = np.where(has_ts, row_ts, 1)

        bad_nonfinal_w = has_ts & (~is_final) & (widths != row_ts)
        if np.any(bad_nonfinal_w):
            idx = int(np.flatnonzero(bad_nonfinal_w)[0])
            raise SchemaError(
                f"row {idx} bin width {int(widths[idx])} does not match {int(row_ts[idx])}"
            )

        bad_final_w = has_ts & is_final & ((widths <= 0) | (widths > row_ts))
        if np.any(bad_final_w):
            idx = int(np.flatnonzero(bad_final_w)[0])
            raise SchemaError(
                f"row {idx} final bin width {int(widths[idx])} outside expected "
                f"(0, {int(row_ts[idx])}]"
            )

        bad_align = has_ts & ((starts % safe_ts) != 0)
        if np.any(bad_align):
            idx = int(np.flatnonzero(bad_align)[0])
            raise SchemaError(
                f"row {idx} start coordinate {int(starts[idx])} is not a multiple "
                f"of expected bin size {int(row_ts[idx])}"
            )


def validate_reference_table(
    table: pa.Table,
    *,
    check_rows: bool = True,
) -> ReferenceMetadata:
    """Validate Schema v2.0.0 reference table, metadata, and optional row intervals."""
    metadata = decode_metadata(table.schema.metadata)
    _check_version_header(metadata, ArtifactKind.REFERENCE)
    expected = _expected_reference_schema(table.schema)
    _validate_arrow_fields(table.schema, expected, ArtifactKind.REFERENCE)
    _validate_non_null_columns(table, expected)
    ref_meta = validate_metadata(metadata, ArtifactKind.REFERENCE)
    if check_rows:
        _validate_reference_rows(table, metadata, ref_meta)
    return ref_meta
