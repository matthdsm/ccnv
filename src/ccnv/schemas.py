"""Versioned Arrow contracts for ccnv Parquet artifacts."""

import json
from collections.abc import Mapping
from enum import StrEnum
from typing import Final

import pyarrow as pa

SCHEMA_VERSION: Final[str] = "1.0.0"
SUPPORTED_SCHEMA_VERSIONS: Final[frozenset[str]] = frozenset({"1.0.0"})
COORDINATE_SYSTEM: Final[str] = "0-based-half-open"
METADATA_PREFIX: Final[str] = "ccnv."


class ArtifactKind(StrEnum):
    SAMPLE = "sample"
    REFERENCE = "reference"


class SchemaError(ValueError):
    """Raised when an artifact violates a storage contract."""


SAMPLE_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field("contig", pa.string(), nullable=False),
        pa.field("start", pa.int64(), nullable=False),
        pa.field("end", pa.int64(), nullable=False),
        pa.field("count", pa.uint64(), nullable=False),
    ]
)

REFERENCE_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field("branch", pa.string(), nullable=False),
        pa.field("contig", pa.string(), nullable=False),
        pa.field("start", pa.int64(), nullable=False),
        pa.field("end", pa.int64(), nullable=False),
        pa.field("masked", pa.bool_(), nullable=False),
        pa.field("pca_mean", pa.float64(), nullable=False),
        pa.field("pca_loadings", pa.list_(pa.float64()), nullable=False),
        pa.field("nearest_ref_indices", pa.list_(pa.int32()), nullable=False),
        pa.field("mean_distance", pa.float32(), nullable=False),
        pa.field("null_mean", pa.float64(), nullable=False),
        pa.field("null_std", pa.float64(), nullable=False),
    ]
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


def encode_metadata(values: Mapping[str, object]) -> dict[bytes, bytes]:
    """Encode namespaced metadata values as deterministic JSON."""

    return {
        f"{METADATA_PREFIX}{key}".encode(): json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ).encode()
        for key, value in values.items()
    }


def decode_metadata(metadata: Mapping[bytes, bytes] | None) -> dict[str, object]:
    """Decode ccnv metadata and reject malformed JSON."""

    if metadata is None:
        return {}
    decoded: dict[str, object] = {}
    prefix = METADATA_PREFIX.encode()
    for raw_key, raw_value in metadata.items():
        if not raw_key.startswith(prefix):
            continue
        key = raw_key[len(prefix) :].decode()
        try:
            decoded[key] = json.loads(raw_value)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SchemaError(f"invalid metadata value for {key!r}") from error
    return decoded


def metadata_for(
    kind: ArtifactKind,
    *,
    contig_lengths: Mapping[str, int],
    values: Mapping[str, object],
) -> dict[bytes, bytes]:
    """Build complete metadata for an artifact."""

    metadata: dict[str, object] = {
        "artifact_kind": kind.value,
        "schema_version": SCHEMA_VERSION,
        "coordinate_system": COORDINATE_SYSTEM,
        "contig_lengths": dict(contig_lengths),
        **values,
    }
    validate_metadata(metadata, kind)
    return encode_metadata(metadata)


def validate_metadata(metadata: Mapping[str, object], kind: ArtifactKind) -> None:
    """Validate required keys and cross-artifact coordinate invariants."""

    required = _COMMON_METADATA | _KIND_METADATA[kind]
    missing = sorted(required - metadata.keys())
    if missing:
        raise SchemaError(f"missing {kind.value} metadata: {', '.join(missing)}")
    if metadata["artifact_kind"] != kind.value:
        raise SchemaError(f"expected artifact kind {kind.value!r}")
    version = metadata.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise SchemaError(
            f"unsupported schema version {version!r}; expected {SCHEMA_VERSION!r}"
        )
    if metadata["coordinate_system"] != COORDINATE_SYSTEM:
        raise SchemaError(f"coordinates must be {COORDINATE_SYSTEM}")

    contig_lengths = metadata["contig_lengths"]
    if not isinstance(contig_lengths, dict) or not contig_lengths:
        raise SchemaError("contig_lengths must be a non-empty object")
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(length, int)
        or isinstance(length, bool)
        or length <= 0
        for name, length in contig_lengths.items()
    ):
        raise SchemaError(
            "contig_lengths must map non-empty names to positive integers"
        )

    if kind is ArtifactKind.SAMPLE:
        bin_size = metadata["bin_size"]
        if not isinstance(bin_size, int) or isinstance(bin_size, bool) or bin_size <= 0:
            raise SchemaError("bin_size must be a positive integer")
        filters = metadata["filters"]
        if not isinstance(filters, dict):
            raise SchemaError("filters must be an object")
        min_mapping_quality = filters.get("min_mapping_quality")
        if (
            not isinstance(min_mapping_quality, int)
            or isinstance(min_mapping_quality, bool)
            or min_mapping_quality < 0
        ):
            raise SchemaError(
                "filters.min_mapping_quality must be a non-negative integer"
            )


def validate_schema(schema: pa.Schema, kind: ArtifactKind) -> None:
    """Validate Arrow fields and embedded file metadata."""

    expected = SAMPLE_SCHEMA if kind is ArtifactKind.SAMPLE else REFERENCE_SCHEMA
    if not schema.equals(expected, check_metadata=False):
        raise SchemaError(f"{kind.value} fields do not match schema {SCHEMA_VERSION}")

    metadata = decode_metadata(schema.metadata)
    validate_metadata(metadata, kind)


def bin_coordinates(table: pa.Table) -> tuple[tuple[str, int, int], ...]:
    """Return row identity as ``(contig, start, end)`` tuples."""

    return tuple(
        zip(
            table.column("contig").to_pylist(),
            table.column("start").to_pylist(),
            table.column("end").to_pylist(),
            strict=True,
        )
    )


def validate_table(table: pa.Table, kind: ArtifactKind) -> None:
    """Validate schema plus canonical coordinate bounds and row identity."""

    validate_schema(table.schema, kind)
    metadata = decode_metadata(table.schema.metadata)
    contig_lengths = metadata["contig_lengths"]
    assert isinstance(contig_lengths, dict)

    # Fast path for large tables: check bounds and duplicates via PyArrow/NumPy
    if table.num_rows > 1000:
        import numpy as np

        starts = table.column("start").to_numpy(zero_copy_only=False)
        ends = table.column("end").to_numpy(zero_copy_only=False)
        contig_arr = np.asarray(table.column("contig").to_numpy(zero_copy_only=False))
        unique_contigs = set(table.column("contig").unique().to_pylist())

        all_contigs_declared = unique_contigs.issubset(contig_lengths.keys())
        basic_bounds_ok = bool(np.all(starts >= 0) and np.all(ends > starts))
        lengths_ok = all_contigs_declared and all(
            bool(np.all(ends[contig_arr == c] <= contig_lengths[c]))
            for c in unique_contigs
        )

        cols = (
            ["branch", "contig", "start", "end"]
            if kind is ArtifactKind.REFERENCE
            else ["contig", "start", "end"]
        )
        no_duplicates = table.group_by(cols).aggregate([]).num_rows == table.num_rows

        if all_contigs_declared and basic_bounds_ok and lengths_ok and no_duplicates:
            return

    contigs: list[str] = table.column("contig").to_pylist()
    starts_list: list[int] = table.column("start").to_pylist()
    ends_list: list[int] = table.column("end").to_pylist()
    branches: list[str] | None = (
        table.column("branch").to_pylist() if kind is ArtifactKind.REFERENCE else None
    )
    seen: set[tuple[object, ...]] = set()
    for index, (contig, start, end) in enumerate(
        zip(contigs, starts_list, ends_list, strict=True)
    ):
        length = contig_lengths.get(contig)
        if not isinstance(length, int):
            raise SchemaError(f"row {index} uses undeclared contig {contig!r}")
        if start < 0 or end <= start or end > length:
            raise SchemaError(
                f"row {index} interval [{start}, {end}) is outside "
                f"{contig!r} length {length}"
            )
        identity: tuple[object, ...]
        if branches is None:
            identity = (contig, start, end)
        else:
            identity = (branches[index], contig, start, end)
        if identity in seen:
            raise SchemaError(f"row {index} duplicates bin identity {identity!r}")
        seen.add(identity)
