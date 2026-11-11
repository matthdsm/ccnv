"""Versioned Arrow contracts and artifact metadata for ccnv Parquet tables."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import StrEnum
from typing import Final

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from numpy.typing import NDArray

from ccnv.models.config import DEFAULT_BIN_SIZE, DEFAULT_MIN_MAPPABILITY, AnalysisMode
from ccnv.models.exceptions import SchemaError, SchemaVersionError
from ccnv.models.metadata import (
    FRAGMENTOMICS_METADATA_KEY,
    PRENATAL_METADATA_KEY,
    FragmentomicsMetadata,
    PrenatalModelMetadata,
)

SCHEMA_VERSION: Final[str] = "2.0.0"
SUPPORTED_SCHEMA_VERSIONS: Final[frozenset[str]] = frozenset({"2.0.0"})
COORDINATE_SYSTEM: Final[str] = "0-based-half-open"
METADATA_PREFIX: Final[str] = "ccnv."

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


class ArtifactKind(StrEnum):
    SAMPLE = "sample"
    REFERENCE = "reference"


def _as_int(raw: object, name: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise TypeError(f"{name} must be an integer, got {raw!r}")
    return int(raw)


def _as_float(raw: object, name: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise TypeError(f"{name} must be numeric, got {raw!r}")
    return float(raw)


def _as_str_tuple(raw: object) -> tuple[str, ...]:
    return tuple(str(x) for x in raw) if isinstance(raw, (list, tuple)) else ()


def _as_obj_dict(raw: object) -> dict[str, object]:
    return {str(k): v for k, v in raw.items()} if isinstance(raw, Mapping) else {}


def _check_pos_int(val: object, name: str) -> None:
    if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
        raise ValueError(f"{name} must be a positive integer, got {val!r}")


@dataclass(frozen=True, slots=True)
class SampleMetadata:
    """Strongly-typed metadata container for Schema v2.0.0 sample sketch artifacts."""

    bin_size: int
    contig_lengths: dict[str, int]
    contig_order: tuple[str, ...]
    source_identity: dict[str, object]
    reference_identity: dict[str, object]
    qc: dict[str, object]
    filters: dict[str, object]
    schema_version: str = SCHEMA_VERSION
    coordinate_system: str = COORDINATE_SYSTEM
    fragmentomics: FragmentomicsMetadata | None = None

    @property
    def paired_end(self) -> bool:
        """Whether paired-end fragments were detected during sample extraction."""
        return (
            self.fragmentomics.paired_end if self.fragmentomics is not None else False
        )

    @property
    def expected_num_bins(self) -> int:
        """Total row count implied by contig_order, contig_lengths, and bin_size."""
        return sum(
            (self.contig_lengths[c] + self.bin_size - 1) // self.bin_size
            for c in self.contig_order
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SampleMetadata:
        """Construct sample metadata from a decoded Arrow metadata dictionary."""
        raw_lengths = data.get("contig_lengths")
        if not isinstance(raw_lengths, Mapping) or not raw_lengths:
            raise SchemaError("contig_lengths must be a non-empty object")
        contig_lengths = {
            str(k): _as_int(v, "contig_length") for k, v in raw_lengths.items()
        }
        ref_id = _as_obj_dict(data.get("reference_identity"))
        raw_order = data.get("contig_order", ref_id.get("contig_order"))
        contig_order = (
            _as_str_tuple(raw_order)
            if isinstance(raw_order, (list, tuple))
            else tuple(contig_lengths.keys())
        )
        if len(contig_order) != len(contig_lengths) or set(contig_order) != set(
            contig_lengths.keys()
        ):
            raise SchemaError("contig_order does not match contig_lengths keys")

        frag_obj = data.get("fragmentomics")
        frag_meta: FragmentomicsMetadata | None = None
        if isinstance(frag_obj, Mapping):
            try:
                frag_meta = FragmentomicsMetadata.from_dict(frag_obj)
            except (KeyError, TypeError, ValueError) as err:
                raise SchemaError(
                    f"invalid ccnv.fragmentomics metadata: {err}"
                ) from err

        return cls(
            bin_size=_as_int(data["bin_size"], "bin_size"),
            contig_lengths=contig_lengths,
            contig_order=contig_order,
            source_identity=_as_obj_dict(data.get("source_identity")),
            reference_identity=ref_id,
            qc=_as_obj_dict(data.get("qc")),
            filters=_as_obj_dict(data.get("filters")),
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            coordinate_system=str(data.get("coordinate_system", COORDINATE_SYSTEM)),
            fragmentomics=frag_meta,
        )

    @classmethod
    def from_schema(cls, schema: pa.Schema) -> SampleMetadata:
        """Construct sample metadata from an Arrow Schema."""
        return cls.from_dict(decode_metadata(schema.metadata))

    @classmethod
    def from_table(cls, table: pa.Table) -> SampleMetadata:
        """Construct sample metadata from an Arrow Table."""
        return cls.from_schema(table.schema)


_REF_TOP_KEYS: Final[frozenset[str]] = frozenset(
    {
        "contig_lengths",
        "sample_identities",
        "schema_version",
        "coordinate_system",
        "prenatal",
    }
)


@dataclass(frozen=True, slots=True)
class ReferenceMetadata:
    """Strongly-typed metadata container for ccnv reference artifacts."""

    bin_size: int
    contig_lengths: dict[str, int]
    branches: tuple[str, ...]
    autosomal_contigs: tuple[str, ...] = ()
    sex_contigs: tuple[str, ...] = ()
    pca_components: int = 5
    nearest_bins: int = 300
    mode: AnalysisMode = "germline"
    prenatal_bin_size: int | None = None
    sample_identities: tuple[object, ...] = ()
    sex_signal_contig: str | None = None
    sex_cutoff: float | None = None
    sex_source: str | None = None
    sample_sexes: tuple[str, ...] | None = None
    sex_inference: dict[str, object] | None = None
    max_batch_memory_mb: int = 256
    flat: bool = False
    gc_corrected: bool = False
    mappability_corrected: bool = False
    fragmentomics_corrected: bool = False
    min_mappability: float = DEFAULT_MIN_MAPPABILITY
    schema_version: str = SCHEMA_VERSION
    coordinate_system: str = COORDINATE_SYSTEM
    prenatal: PrenatalModelMetadata | None = None

    def __post_init__(self) -> None:
        """Validate reference metadata configuration and contig lengths."""
        _check_pos_int(self.bin_size, "bin_size")
        if self.prenatal_bin_size is not None:
            _check_pos_int(self.prenatal_bin_size, "prenatal_bin_size")
        if self.mode not in ("germline", "prenatal"):
            raise ValueError(
                f"mode must be 'germline' or 'prenatal', got {self.mode!r}"
            )
        if not self.contig_lengths:
            raise ValueError("contig_lengths must be a non-empty mapping")

    def to_metadata(self) -> dict[str, object]:
        """Convert reference metadata to a dictionary for Arrow table persistence."""
        model: dict[str, object] = {}
        for f in fields(self):
            if f.name in _REF_TOP_KEYS:
                continue
            val = getattr(self, f.name)
            model[f.name] = list(val) if isinstance(val, tuple) else val
        values: dict[str, object] = {
            "sample_identities": list(self.sample_identities),
            "model": model,
        }
        if self.prenatal is not None:
            values["prenatal"] = self.prenatal.to_dict()
        return values

    def build_arrow_metadata(self) -> dict[bytes, bytes]:
        """Build Arrow metadata mapping containing reference metadata."""
        return metadata_for(
            ArtifactKind.REFERENCE,
            contig_lengths=self.contig_lengths,
            values=self.to_metadata(),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ReferenceMetadata:
        """Construct reference metadata from a decoded metadata dictionary."""
        model = _as_obj_dict(data.get("model"))
        raw_clen = data.get("contig_lengths", {})
        contig_lengths = (
            {str(k): _as_int(v, "contig_length") for k, v in raw_clen.items()}
            if isinstance(raw_clen, Mapping)
            else {}
        )
        raw_ids = data.get("sample_identities", ())
        sample_identities: tuple[object, ...] = (
            tuple(dict(x) if isinstance(x, Mapping) else str(x) for x in raw_ids)
            if isinstance(raw_ids, (list, tuple))
            else ()
        )
        prenatal_obj = data.get("prenatal")
        prenatal_meta = (
            PrenatalModelMetadata.from_dict(prenatal_obj)
            if isinstance(prenatal_obj, Mapping)
            else None
        )
        raw_mode = str(model.get("mode", "germline"))
        if raw_mode not in ("germline", "prenatal"):
            raise ValueError(f"mode must be 'germline' or 'prenatal', got {raw_mode!r}")
        mode: AnalysisMode = "prenatal" if raw_mode == "prenatal" else "germline"
        raw_pbs = model.get("prenatal_bin_size")
        prenatal_bs = (
            _as_int(raw_pbs, "prenatal_bin_size")
            if raw_pbs is not None
            else (
                prenatal_meta.prenatal_bin_size if prenatal_meta is not None else None
            )
        )
        raw_sexes = model.get("sample_sexes")
        raw_cutoff = model.get("sex_cutoff")
        raw_sex_inf = model.get("sex_inference")
        return cls(
            bin_size=_as_int(model.get("bin_size", DEFAULT_BIN_SIZE), "bin_size"),
            contig_lengths=contig_lengths,
            branches=_as_str_tuple(model.get("branches")),
            autosomal_contigs=_as_str_tuple(model.get("autosomal_contigs")),
            sex_contigs=_as_str_tuple(model.get("sex_contigs")),
            pca_components=_as_int(model.get("pca_components", 5), "pca_components"),
            nearest_bins=_as_int(model.get("nearest_bins", 300), "nearest_bins"),
            mode=mode,
            prenatal_bin_size=prenatal_bs,
            sample_identities=sample_identities,
            sex_signal_contig=(
                None
                if model.get("sex_signal_contig") is None
                else str(model["sex_signal_contig"])
            ),
            sex_cutoff=(
                None if raw_cutoff is None else _as_float(raw_cutoff, "sex_cutoff")
            ),
            sex_source=(
                None if model.get("sex_source") is None else str(model["sex_source"])
            ),
            sample_sexes=(
                _as_str_tuple(raw_sexes)
                if isinstance(raw_sexes, (list, tuple))
                else None
            ),
            sex_inference=(
                _as_obj_dict(raw_sex_inf) if isinstance(raw_sex_inf, Mapping) else None
            ),
            max_batch_memory_mb=_as_int(
                model.get("max_batch_memory_mb", 256), "max_batch_memory_mb"
            ),
            flat=bool(model.get("flat", False)),
            gc_corrected=bool(model.get("gc_corrected", False)),
            mappability_corrected=bool(model.get("mappability_corrected", False)),
            fragmentomics_corrected=bool(model.get("fragmentomics_corrected", False)),
            min_mappability=_as_float(
                model.get("min_mappability", DEFAULT_MIN_MAPPABILITY), "min_mappability"
            ),
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            coordinate_system=str(data.get("coordinate_system", COORDINATE_SYSTEM)),
            prenatal=prenatal_meta,
        )

    @classmethod
    def from_schema(cls, schema: pa.Schema) -> ReferenceMetadata:
        """Construct reference metadata from an Arrow Schema."""
        return cls.from_dict(decode_metadata(schema.metadata))

    @classmethod
    def from_table(cls, table: pa.Table) -> ReferenceMetadata:
        """Construct reference metadata from an Arrow Table."""
        return cls.from_schema(table.schema)


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
    "metadata_for",
    "reconstruct_sample_grid",
    "validate_metadata",
    "validate_reference_table",
    "validate_sample_table",
    "validate_schema",
    "validate_table",
]
