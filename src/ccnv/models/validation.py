"""Vectorized Arrow schema, metadata, and coordinate-grid validation for ccnv."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ccnv.models.exceptions import SchemaError, SchemaVersionError
from ccnv.models.schemas import (
    BRANCH_PRENATAL_AUTOSOMAL,
    COORDINATE_SYSTEM,
    KNOWN_REFERENCE_BRANCHES,
    SAMPLE_SCHEMA,
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    ArtifactKind,
    PrenatalModelMetadata,
    ReferenceMetadata,
    SampleMetadata,
    build_reference_schema,
    decode_metadata,
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
    "validate_table",
]


def _is_pos_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _check_version_header(metadata: Mapping[str, object]) -> None:
    """Fail fast with SchemaVersionError before field checks when version is unsupported."""
    if "schema_version" in metadata:
        version = metadata["schema_version"]
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            raise SchemaVersionError(
                f"unsupported schema version {version!r}; expected {SCHEMA_VERSION!r}"
            )


def validate_metadata(metadata: Mapping[str, object], kind: ArtifactKind) -> None:
    """Validate required keys and cross-artifact coordinate invariants."""
    _check_version_header(metadata)
    required = _COMMON_METADATA | _KIND_METADATA[kind]
    missing = sorted(required - metadata.keys())
    if missing:
        raise SchemaError(f"missing {kind.value} metadata: {', '.join(missing)}")
    if metadata["artifact_kind"] != kind.value:
        raise SchemaError(f"expected artifact kind {kind.value!r}")
    if metadata["coordinate_system"] != COORDINATE_SYSTEM:
        raise SchemaError(f"coordinates must be {COORDINATE_SYSTEM}")

    contig_lengths = metadata["contig_lengths"]
    if not isinstance(contig_lengths, dict) or not contig_lengths:
        raise SchemaError("contig_lengths must be a non-empty object")
    if any(
        not isinstance(name, str) or not name or not _is_pos_int(length)
        for name, length in contig_lengths.items()
    ):
        raise SchemaError(
            "contig_lengths must map non-empty names to positive integers"
        )

    if kind is ArtifactKind.SAMPLE:
        if not _is_pos_int(metadata["bin_size"]):
            raise SchemaError("bin_size must be a positive integer")
        filters = metadata["filters"]
        if not isinstance(filters, dict):
            raise SchemaError("filters must be an object")
        min_mapq = filters.get("min_mapping_quality")
        if not isinstance(min_mapq, int) or isinstance(min_mapq, bool) or min_mapq < 0:
            raise SchemaError(
                "filters.min_mapping_quality must be a non-negative integer"
            )
        contig_order = metadata.get("contig_order")
        if contig_order is not None:
            if not isinstance(contig_order, (list, tuple)) or any(
                not isinstance(c, str) for c in contig_order
            ):
                raise SchemaError("contig_order must be a list of contig names")
            if len(contig_order) != len(contig_lengths) or set(contig_order) != set(
                contig_lengths.keys()
            ):
                raise SchemaError("contig_order must match contig_lengths keys")

    if kind is ArtifactKind.REFERENCE:
        model = metadata.get("model")
        if not isinstance(model, dict):
            raise SchemaError("model must be an object")
        bin_size = model.get("bin_size")
        if bin_size is not None and not _is_pos_int(bin_size):
            raise SchemaError("model.bin_size must be a positive integer")
        prenatal_bin_size = model.get("prenatal_bin_size")
        if prenatal_bin_size is not None and not _is_pos_int(prenatal_bin_size):
            raise SchemaError("model.prenatal_bin_size must be a positive integer")
        mode = model.get("mode")
        if mode is not None and mode not in ("germline", "prenatal"):
            raise SchemaError(
                f"model.mode must be 'germline' or 'prenatal', got {mode!r}"
            )
        if "prenatal" in metadata:
            prenatal_obj = metadata["prenatal"]
            if not isinstance(prenatal_obj, Mapping):
                raise SchemaError("ccnv.prenatal metadata must be an object")
            try:
                PrenatalModelMetadata.from_dict(prenatal_obj)
            except (ValueError, TypeError, KeyError) as err:
                raise SchemaError(f"invalid ccnv.prenatal metadata: {err}") from err


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
    """Reject null entries in columns declared non-nullable by the contract."""
    for idx, expected_field in enumerate(expected):
        if not expected_field.nullable and table.column(idx).null_count > 0:
            raise SchemaError(
                f"column {expected_field.name!r} contains unexpected null values"
            )


def validate_schema(schema: pa.Schema, kind: ArtifactKind) -> None:
    """Validate Arrow fields and embedded file metadata."""
    metadata = decode_metadata(schema.metadata)
    _check_version_header(metadata)
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
    _check_version_header(metadata)
    _validate_arrow_fields(table.schema, SAMPLE_SCHEMA, ArtifactKind.SAMPLE)
    _validate_non_null_columns(table, SAMPLE_SCHEMA)
    validate_metadata(metadata, ArtifactKind.SAMPLE)

    sample_meta = SampleMetadata.from_dict(metadata)
    expected_rows = sample_meta.expected_num_bins
    if table.num_rows != expected_rows:
        raise SchemaError(
            f"sample row count {table.num_rows} is outside expected grid size "
            f"{expected_rows} for bin_size={sample_meta.bin_size}"
        )
    return sample_meta


def _resolve_target_bin_sizes(
    metadata: Mapping[str, object],
) -> tuple[int | None, int | None]:
    model_obj = metadata.get("model")
    model = model_obj if isinstance(model_obj, Mapping) else {}
    raw_bin = model.get("bin_size")
    bin_size = (
        int(raw_bin) if isinstance(raw_bin, int) and _is_pos_int(raw_bin) else None
    )

    raw_p = model.get("prenatal_bin_size")
    if raw_p is None and isinstance(metadata.get("prenatal"), Mapping):
        prenatal_map = metadata["prenatal"]
        if isinstance(prenatal_map, Mapping):
            raw_p = prenatal_map.get("prenatal_bin_size")
    prenatal_bin_size = (
        int(raw_p) if isinstance(raw_p, int) and _is_pos_int(raw_p) else None
    )
    return bin_size, prenatal_bin_size


def _validate_reference_rows(
    table: pa.Table,
    metadata: Mapping[str, object],
    ref_meta: ReferenceMetadata,
) -> None:
    """Validate contiguous (branch, contig) runs, coordinate bounds, and bin spacing."""
    if table.num_rows == 0:
        return

    bin_size, prenatal_bin_size = _resolve_target_bin_sizes(metadata)

    branch_dict = (
        pc.dictionary_encode(table.column("branch"))
        .unify_dictionaries()
        .combine_chunks()
    )
    contig_dict = (
        pc.dictionary_encode(table.column("contig"))
        .unify_dictionaries()
        .combine_chunks()
    )

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

    target_size_by_bcode = np.zeros(len(branch_dict.dictionary), dtype=np.int64)
    for b_code_raw in np.unique(b_codes):
        b_code = int(b_code_raw)
        b_name = str(branch_dict.dictionary[b_code].as_py())
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
        target_size_by_bcode[b_code] = ts if ts is not None else 0

    contig_len_by_code = np.zeros(len(contig_dict.dictionary), dtype=np.int64)
    contig_name_by_code: dict[int, str] = {}
    for c_code_raw in np.unique(c_codes):
        c_code = int(c_code_raw)
        c_name = str(contig_dict.dictionary[c_code].as_py())
        c_len = ref_meta.contig_lengths.get(c_name)
        if c_len is None:
            raise SchemaError(f"undeclared contig {c_name!r}")
        contig_len_by_code[c_code] = c_len
        contig_name_by_code[c_code] = c_name

    # Enforce contiguous branch blocks and unique contiguous (branch, contig) runs
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
        c_name = contig_name_by_code[int(c_codes[idx])]
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
    _check_version_header(metadata)
    expected = _expected_reference_schema(table.schema)
    _validate_arrow_fields(table.schema, expected, ArtifactKind.REFERENCE)
    _validate_non_null_columns(table, expected)
    validate_metadata(metadata, ArtifactKind.REFERENCE)

    ref_meta = ReferenceMetadata.from_dict(metadata)
    if check_rows:
        _validate_reference_rows(table, metadata, ref_meta)
    return ref_meta


def validate_table(table: pa.Table, kind: ArtifactKind) -> None:
    """Validate schema plus canonical coordinate bounds and row identity."""
    if kind is ArtifactKind.SAMPLE:
        validate_sample_table(table)
    else:
        validate_reference_table(table, check_rows=True)
