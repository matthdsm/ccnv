"""Typed metadata containers and JSON codec for ccnv sample and reference artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import StrEnum
from typing import Final

import pyarrow as pa

from ccnv.models.config import DEFAULT_BIN_SIZE, DEFAULT_MIN_MAPPABILITY, AnalysisMode
from ccnv.models.exceptions import SchemaError, SchemaVersionError

SCHEMA_VERSION: Final[str] = "2.0.0"
SUPPORTED_SCHEMA_VERSIONS: Final[frozenset[str]] = frozenset({"2.0.0"})
COORDINATE_SYSTEM: Final[str] = "0-based-half-open"
METADATA_PREFIX: Final[str] = "ccnv."

PRENATAL_METADATA_KEY: Final[bytes] = b"ccnv.prenatal"
FRAGMENTOMICS_METADATA_KEY: Final[bytes] = b"ccnv.fragmentomics"

_MISSING: Final[object] = object()


class ArtifactKind(StrEnum):
    SAMPLE = "sample"
    REFERENCE = "reference"


__all__ = [
    "COORDINATE_SYSTEM",
    "FRAGMENTOMICS_METADATA_KEY",
    "METADATA_PREFIX",
    "PRENATAL_METADATA_KEY",
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "ArtifactKind",
    "FragmentomicsMetadata",
    "PrenatalModelMetadata",
    "ReferenceMetadata",
    "SampleMetadata",
    "decode_metadata",
    "encode_metadata",
]


def _as_bool(raw: object, name: str) -> bool:
    if not isinstance(raw, bool):
        raise TypeError(f"{name} must be a boolean, got {raw!r}")
    return raw


def _as_int(raw: object, name: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise TypeError(f"{name} must be an integer, got {raw!r}")
    return int(raw)


def _as_float(raw: object, name: str, *, finite: bool = False) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise TypeError(f"{name} must be numeric, got {raw!r}")
    val = float(raw)
    if finite and not math.isfinite(val):
        raise ValueError(f"{name} must be finite, got {raw!r}")
    return val


def _req_bool(data: Mapping[str, object], key: str, default: object = _MISSING) -> bool:
    raw = data[key] if default is _MISSING else data.get(key, default)
    return _as_bool(raw, key)


def _req_int(data: Mapping[str, object], key: str, default: object = _MISSING) -> int:
    raw = data[key] if default is _MISSING else data.get(key, default)
    return _as_int(raw, key)


def _opt_int(data: Mapping[str, object], key: str) -> int | None:
    raw = data.get(key)
    return None if raw is None else _as_int(raw, key)


def _req_float(
    data: Mapping[str, object],
    key: str,
    default: object = _MISSING,
    *,
    finite: bool = False,
) -> float:
    raw = data[key] if default is _MISSING else data.get(key, default)
    return _as_float(raw, key, finite=finite)


def _opt_float(data: Mapping[str, object], key: str) -> float | None:
    raw = data.get(key)
    return None if raw is None else _as_float(raw, key)


def _opt_str(data: Mapping[str, object], key: str) -> str | None:
    raw = data.get(key)
    return None if raw is None else str(raw)


def _req_float_list(data: Mapping[str, object], key: str) -> list[float]:
    raw = data[key]
    if not isinstance(raw, (list, tuple)):
        raise TypeError(f"{key} must be a sequence of floats, got {raw!r}")
    return [_as_float(item, key) for item in raw]


def _opt_float_list(data: Mapping[str, object], key: str) -> list[float] | None:
    if data.get(key) is None:
        return None
    return _req_float_list(data, key)


def _req_int_list(data: Mapping[str, object], key: str) -> list[int]:
    raw = data[key]
    if not isinstance(raw, (list, tuple)):
        raise TypeError(f"{key} must be a sequence of integers, got {raw!r}")
    return [_as_int(item, key) for item in raw]


def _opt_int_list(data: Mapping[str, object], key: str) -> list[int] | None:
    if data.get(key) is None:
        return None
    return _req_int_list(data, key)


def _as_str_tuple(raw: object, name: str) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple)) or any(not isinstance(x, str) for x in raw):
        raise TypeError(f"{name} must be a sequence of strings, got {raw!r}")
    return tuple(raw)


def _opt_str_tuple(data: Mapping[str, object], key: str) -> tuple[str, ...]:
    raw = data.get(key)
    return () if raw is None else _as_str_tuple(raw, key)


def _opt_nullable_str_tuple(
    data: Mapping[str, object], key: str
) -> tuple[str, ...] | None:
    raw = data.get(key)
    return None if raw is None else _as_str_tuple(raw, key)


def _opt_obj_dict(data: Mapping[str, object], key: str) -> dict[str, object]:
    raw = data.get(key)
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"{key} must be a mapping, got {raw!r}")
    return {str(k): v for k, v in raw.items()}


def _opt_nullable_obj_dict(
    data: Mapping[str, object], key: str
) -> dict[str, object] | None:
    return None if data.get(key) is None else _opt_obj_dict(data, key)


def _is_pos_int(val: object) -> bool:
    return isinstance(val, int) and not isinstance(val, bool) and val > 0


def _schema_pos_int(val: object, name: str) -> None:
    if not _is_pos_int(val):
        raise SchemaError(f"{name} must be a positive integer, got {val!r}")


def _schema_nonneg_int(val: object, name: str) -> None:
    if not isinstance(val, int) or isinstance(val, bool) or val < 0:
        raise SchemaError(f"{name} must be a non-negative integer, got {val!r}")


def _validate_contig_lengths(contig_lengths: object) -> None:
    if not isinstance(contig_lengths, Mapping) or not contig_lengths:
        raise SchemaError("contig_lengths must be a non-empty object")
    if any(
        not isinstance(k, str) or not k or not _is_pos_int(v)
        for k, v in contig_lengths.items()
    ):
        raise SchemaError(
            "contig_lengths must map non-empty names to positive integers"
        )


def _reject_json_constant(token: str) -> object:
    raise ValueError(f"non-RFC-8259 JSON constant {token!r}")


def encode_metadata(values: Mapping[str, object]) -> dict[bytes, bytes]:
    """Encode namespaced metadata values as deterministic RFC-8259 JSON."""
    encoded: dict[bytes, bytes] = {}
    for key, value in values.items():
        try:
            payload = json.dumps(
                value, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        except (TypeError, ValueError) as error:
            raise SchemaError(f"invalid metadata value for {key!r}: {error}") from error
        encoded[f"{METADATA_PREFIX}{key}".encode()] = payload
    return encoded


def decode_metadata(metadata: Mapping[bytes, bytes] | None) -> dict[str, object]:
    """Decode ccnv metadata and reject malformed or non-finite JSON."""
    if metadata is None:
        return {}
    decoded: dict[str, object] = {}
    prefix = METADATA_PREFIX.encode()
    for raw_key, raw_value in metadata.items():
        if not raw_key.startswith(prefix):
            continue
        key = raw_key[len(prefix) :].decode()
        try:
            decoded[key] = json.loads(raw_value, parse_constant=_reject_json_constant)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise SchemaError(f"invalid metadata value for {key!r}") from error
    return decoded


@dataclass(frozen=True, slots=True)
class FragmentomicsMetadata:
    """Sample-level multi-modal fragmentomic metrics serialized under ccnv.fragmentomics."""

    paired_end: bool
    total_paired_fragments: int
    chromatosome_mode_bp: float
    core_shoulder_bp: float
    sizing_shift_bp: float
    self_referenced_peak_ratio: float
    dinucleosome_fraction: float
    high_mw_fraction: float
    helical_10bp_amplitude: float
    helical_10bp_phase: float
    motif_diversity_score: float
    hist_autosomal: list[int]
    hist_chry: list[int]
    hist_chrx: list[int]
    end_motif_counts: list[int]

    def __post_init__(self) -> None:
        """Validate fragmentomics vector dimensions."""
        for attr, expected in (
            ("hist_autosomal", 571),
            ("hist_chry", 571),
            ("hist_chrx", 571),
            ("end_motif_counts", 256),
        ):
            actual = len(getattr(self, attr))
            if actual != expected:
                raise ValueError(
                    f"{attr} must contain {expected} elements, got {actual}"
                )

    def to_dict(self) -> dict[str, object]:
        """Convert fragmentomics metadata container to a JSON-serializable dictionary."""
        return {
            "paired_end": bool(self.paired_end),
            "total_paired_fragments": int(self.total_paired_fragments),
            "chromatosome_mode_bp": float(self.chromatosome_mode_bp),
            "core_shoulder_bp": float(self.core_shoulder_bp),
            "sizing_shift_bp": float(self.sizing_shift_bp),
            "self_referenced_peak_ratio": float(self.self_referenced_peak_ratio),
            "dinucleosome_fraction": float(self.dinucleosome_fraction),
            "high_mw_fraction": float(self.high_mw_fraction),
            "helical_10bp_amplitude": float(self.helical_10bp_amplitude),
            "helical_10bp_phase": float(self.helical_10bp_phase),
            "motif_diversity_score": float(self.motif_diversity_score),
            "hist_autosomal": [int(x) for x in self.hist_autosomal],
            "hist_chry": [int(x) for x in self.hist_chry],
            "hist_chrx": [int(x) for x in self.hist_chrx],
            "end_motif_counts": [int(x) for x in self.end_motif_counts],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> FragmentomicsMetadata:
        """Construct FragmentomicsMetadata from a decoded JSON dictionary."""
        return cls(
            paired_end=_req_bool(data, "paired_end"),
            total_paired_fragments=_req_int(data, "total_paired_fragments"),
            chromatosome_mode_bp=_req_float(data, "chromatosome_mode_bp"),
            core_shoulder_bp=_req_float(data, "core_shoulder_bp"),
            sizing_shift_bp=_req_float(data, "sizing_shift_bp"),
            self_referenced_peak_ratio=_req_float(data, "self_referenced_peak_ratio"),
            dinucleosome_fraction=_req_float(data, "dinucleosome_fraction"),
            high_mw_fraction=_req_float(data, "high_mw_fraction"),
            helical_10bp_amplitude=_req_float(data, "helical_10bp_amplitude"),
            helical_10bp_phase=_req_float(data, "helical_10bp_phase"),
            motif_diversity_score=_req_float(data, "motif_diversity_score"),
            hist_autosomal=_req_int_list(data, "hist_autosomal"),
            hist_chry=_req_int_list(data, "hist_chry"),
            hist_chrx=_req_int_list(data, "hist_chrx"),
            end_motif_counts=_req_int_list(data, "end_motif_counts"),
        )


@dataclass(frozen=True, slots=True)
class PrenatalModelMetadata:
    """Scalar parameters for prenatal models serialized under ccnv.prenatal."""

    prenatal_bin_size: int
    yfrac_threshold: float
    ffy_slope: float
    olm_weights: list[float]
    olm_bias: float
    female_chrx_ratio_mean: float
    female_chrx_ratio_std: float
    alpha_1: float = -0.5
    ffy_std: float = 0.01
    ffa_std: float = 0.02
    ffa_logit_mode: bool = True
    p_fetal_density: list[float] | None = None
    p_maternal_density: list[float] | None = None
    ff_frag_weights: list[float] | None = None
    ff_frag_bias: float | None = None
    ff_frag_feature_mean: list[float] | None = None
    ff_frag_feature_scale: list[float] | None = None
    ff_size_std: float = 0.018
    msy_bg_rate: float | None = None
    clean_msy_bins: list[int] | None = None
    quarantined_samples: list[int] | None = None
    band_fisher_weights: list[float] | None = None
    mu_fetal_w: float | None = None
    mu_maternal_w: float | None = None

    def __post_init__(self) -> None:
        """Validate prenatal model metadata invariants and forward-model parameters."""
        if not _is_pos_int(self.prenatal_bin_size):
            raise ValueError(
                f"prenatal_bin_size must be a positive integer, got {self.prenatal_bin_size!r}"
            )
        for attr in ("female_chrx_ratio_mean", "female_chrx_ratio_std", "alpha_1"):
            _as_float(getattr(self, attr), attr, finite=True)
        if self.female_chrx_ratio_std < 0.0:
            raise ValueError(
                f"female_chrx_ratio_std must be non-negative, got {self.female_chrx_ratio_std!r}"
            )
        if self.alpha_1 >= 0.0:
            raise ValueError(f"alpha_1 must be negative, got {self.alpha_1!r}")

    def to_dict(self) -> dict[str, object]:
        """Convert metadata container to a JSON-serializable dictionary."""
        out: dict[str, object] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if val is None or isinstance(val, bool):
                out[f.name] = val
            elif f.name == "prenatal_bin_size":
                out[f.name] = int(val)
            elif f.name in ("clean_msy_bins", "quarantined_samples") and isinstance(
                val, (list, tuple)
            ):
                out[f.name] = [int(x) for x in val]
            elif isinstance(val, (list, tuple)):
                out[f.name] = [float(x) for x in val]
            else:
                out[f.name] = float(val)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> PrenatalModelMetadata:
        """Construct prenatal model metadata from a parsed dictionary."""
        return cls(
            prenatal_bin_size=_req_int(data, "prenatal_bin_size"),
            yfrac_threshold=_req_float(data, "yfrac_threshold"),
            ffy_slope=_req_float(data, "ffy_slope"),
            olm_weights=_req_float_list(data, "olm_weights"),
            olm_bias=_req_float(data, "olm_bias"),
            female_chrx_ratio_mean=_req_float(
                data, "female_chrx_ratio_mean", finite=True
            ),
            female_chrx_ratio_std=_req_float(
                data, "female_chrx_ratio_std", finite=True
            ),
            alpha_1=_req_float(data, "alpha_1", -0.5, finite=True),
            ffy_std=_req_float(data, "ffy_std", 0.01),
            ffa_std=_req_float(data, "ffa_std", 0.02),
            ffa_logit_mode=_req_bool(data, "ffa_logit_mode", True),
            p_fetal_density=_opt_float_list(data, "p_fetal_density"),
            p_maternal_density=_opt_float_list(data, "p_maternal_density"),
            ff_frag_weights=_opt_float_list(data, "ff_frag_weights"),
            ff_frag_bias=_opt_float(data, "ff_frag_bias"),
            ff_frag_feature_mean=_opt_float_list(data, "ff_frag_feature_mean"),
            ff_frag_feature_scale=_opt_float_list(data, "ff_frag_feature_scale"),
            ff_size_std=_req_float(data, "ff_size_std", 0.018),
            msy_bg_rate=_opt_float(data, "msy_bg_rate"),
            clean_msy_bins=_opt_int_list(data, "clean_msy_bins"),
            quarantined_samples=_opt_int_list(data, "quarantined_samples"),
            band_fisher_weights=_opt_float_list(data, "band_fisher_weights"),
            mu_fetal_w=_opt_float(data, "mu_fetal_w"),
            mu_maternal_w=_opt_float(data, "mu_maternal_w"),
        )

    def encode(self) -> bytes:
        """Serialize metadata container to canonical RFC-8259 JSON UTF-8 bytes."""
        try:
            return json.dumps(
                self.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        except (TypeError, ValueError) as error:
            raise SchemaError(f"invalid prenatal metadata: {error}") from error

    @classmethod
    def decode(cls, raw: bytes | str | Mapping[str, object]) -> PrenatalModelMetadata:
        """Decode prenatal model metadata from JSON bytes, string, or mapping."""
        try:
            parsed: object = (
                raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
            )
            if isinstance(parsed, str):
                parsed = json.loads(parsed, parse_constant=_reject_json_constant)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise SchemaError(f"invalid prenatal metadata JSON: {error}") from error
        if not isinstance(parsed, Mapping):
            raise SchemaError(
                f"invalid prenatal metadata payload: expected mapping, got {type(parsed)}"
            )
        try:
            return cls.from_dict(parsed)
        except (ValueError, TypeError, KeyError) as error:
            raise SchemaError(f"invalid prenatal metadata: {error}") from error

    @classmethod
    def from_metadata(
        cls, metadata: Mapping[bytes, bytes] | None
    ) -> PrenatalModelMetadata | None:
        """Extract and decode prenatal model metadata from Arrow schema metadata."""
        raw = None if metadata is None else metadata.get(PRENATAL_METADATA_KEY)
        return None if raw is None else cls.decode(raw)

    @classmethod
    def from_table(cls, table: pa.Table) -> PrenatalModelMetadata | None:
        """Extract and decode prenatal model metadata from an Arrow Table."""
        return cls.from_metadata(table.schema.metadata)


def _check_envelope_invariants(
    schema_version: object, coordinate_system: object
) -> None:
    if (
        not isinstance(schema_version, str)
        or schema_version not in SUPPORTED_SCHEMA_VERSIONS
    ):
        raise SchemaVersionError(
            f"unsupported schema version {schema_version!r}; expected {SCHEMA_VERSION!r}"
        )
    if not isinstance(coordinate_system, str) or coordinate_system != COORDINATE_SYSTEM:
        raise SchemaError(f"coordinates must be {COORDINATE_SYSTEM}")


def _parse_schema_version(data: Mapping[str, object]) -> str:
    version = data.get("schema_version", SCHEMA_VERSION)
    if not isinstance(version, str) or version not in SUPPORTED_SCHEMA_VERSIONS:
        raise SchemaVersionError(
            f"unsupported schema version {version!r}; expected {SCHEMA_VERSION!r}"
        )
    return version


def _parse_coordinate_system(data: Mapping[str, object]) -> str:
    coord = data.get("coordinate_system", COORDINATE_SYSTEM)
    if not isinstance(coord, str) or coord != COORDINATE_SYSTEM:
        raise SchemaError(f"coordinates must be {COORDINATE_SYSTEM}")
    return coord


def _parse_contig_lengths(raw_lengths: object) -> dict[str, int]:
    _validate_contig_lengths(raw_lengths)
    if not isinstance(raw_lengths, Mapping):
        raise SchemaError("contig_lengths must be a non-empty object")
    return {
        k: int(v)
        for k, v in raw_lengths.items()
        if isinstance(k, str) and isinstance(v, int)
    }


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

    def __post_init__(self) -> None:
        """Validate sample metadata invariants."""
        _check_envelope_invariants(self.schema_version, self.coordinate_system)
        _schema_pos_int(self.bin_size, "bin_size")
        _validate_contig_lengths(self.contig_lengths)
        if not isinstance(self.contig_order, tuple) or any(
            not isinstance(c, str) or not c for c in self.contig_order
        ):
            raise SchemaError("contig_order must be a tuple of non-empty contig names")
        if len(self.contig_order) != len(self.contig_lengths) or set(
            self.contig_order
        ) != set(self.contig_lengths.keys()):
            raise SchemaError("contig_order does not match contig_lengths keys")
        for attr in ("source_identity", "reference_identity", "qc", "filters"):
            if not isinstance(getattr(self, attr), Mapping):
                raise SchemaError(f"{attr} must be a mapping")
        if self.fragmentomics is not None and not isinstance(
            self.fragmentomics, FragmentomicsMetadata
        ):
            raise SchemaError("fragmentomics must be a FragmentomicsMetadata instance")

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

    def to_metadata(self) -> dict[str, object]:
        """Convert sample metadata to a dictionary for Arrow table persistence."""
        values: dict[str, object] = {
            "artifact_kind": ArtifactKind.SAMPLE.value,
            "schema_version": self.schema_version,
            "coordinate_system": self.coordinate_system,
            "contig_lengths": dict(self.contig_lengths),
            "contig_order": list(self.contig_order),
            "bin_size": int(self.bin_size),
            "source_identity": dict(self.source_identity),
            "reference_identity": dict(self.reference_identity),
            "qc": dict(self.qc),
            "filters": dict(self.filters),
        }
        if self.fragmentomics is not None:
            values["fragmentomics"] = self.fragmentomics.to_dict()
        return values

    def build_arrow_metadata(self) -> dict[bytes, bytes]:
        """Build Arrow metadata mapping containing sample metadata."""
        return encode_metadata(self.to_metadata())

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SampleMetadata:
        """Construct sample metadata from a decoded Arrow metadata dictionary."""
        try:
            schema_version = _parse_schema_version(data)
            coordinate_system = _parse_coordinate_system(data)
            contig_lengths = _parse_contig_lengths(data.get("contig_lengths"))
            ref_id = _opt_obj_dict(data, "reference_identity")
            raw_order = data.get("contig_order", ref_id.get("contig_order"))
            contig_order = (
                _as_str_tuple(raw_order, "contig_order")
                if raw_order is not None
                else tuple(contig_lengths.keys())
            )
            frag_obj = data.get("fragmentomics")
            frag_meta: FragmentomicsMetadata | None = None
            if frag_obj is not None:
                if not isinstance(frag_obj, Mapping):
                    raise SchemaError("ccnv.fragmentomics metadata must be an object")
                frag_meta = FragmentomicsMetadata.from_dict(frag_obj)

            return cls(
                bin_size=_req_int(data, "bin_size"),
                contig_lengths=contig_lengths,
                contig_order=contig_order,
                source_identity=_opt_obj_dict(data, "source_identity"),
                reference_identity=ref_id,
                qc=_opt_obj_dict(data, "qc"),
                filters=_opt_obj_dict(data, "filters"),
                schema_version=schema_version,
                coordinate_system=coordinate_system,
                fragmentomics=frag_meta,
            )
        except SchemaError:
            raise
        except (KeyError, TypeError, ValueError) as err:
            raise SchemaError(f"invalid sample metadata: {err}") from err

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
    pca_components: int = 0
    mode: AnalysisMode = AnalysisMode.germline
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
    branch_pca_ranks: dict[str, int] | None = None
    filters: dict[str, object] | None = None
    schema_version: str = SCHEMA_VERSION
    coordinate_system: str = COORDINATE_SYSTEM
    prenatal: PrenatalModelMetadata | None = None

    def __post_init__(self) -> None:
        """Validate reference metadata configuration and contig lengths."""
        _check_envelope_invariants(self.schema_version, self.coordinate_system)
        _schema_pos_int(self.bin_size, "bin_size")
        if self.prenatal_bin_size is not None:
            _schema_pos_int(self.prenatal_bin_size, "prenatal_bin_size")
        _schema_nonneg_int(self.pca_components, "pca_components")
        _schema_pos_int(self.max_batch_memory_mb, "max_batch_memory_mb")
        if self.branch_pca_ranks is not None:
            for b_name, b_rank in self.branch_pca_ranks.items():
                _schema_nonneg_int(b_rank, f"branch_pca_ranks[{b_name!r}]")
        if not (0.0 <= self.min_mappability <= 1.0):
            raise SchemaError(
                f"min_mappability must be within [0.0, 1.0], got {self.min_mappability!r}"
            )
        if self.mode not in ("germline", "prenatal"):
            raise SchemaError(
                f"mode must be 'germline' or 'prenatal', got {self.mode!r}"
            )
        _validate_contig_lengths(self.contig_lengths)
        if (
            self.prenatal_bin_size is not None
            and self.prenatal is not None
            and self.prenatal_bin_size != self.prenatal.prenatal_bin_size
        ):
            raise SchemaError(
                f"prenatal_bin_size ({self.prenatal_bin_size}) must match "
                f"prenatal.prenatal_bin_size ({self.prenatal.prenatal_bin_size})"
            )

    def to_metadata(self) -> dict[str, object]:
        """Convert reference metadata to a dictionary for Arrow table persistence."""
        model: dict[str, object] = {}
        for f in fields(self):
            if f.name in _REF_TOP_KEYS:
                continue
            val = getattr(self, f.name)
            if f.name == "prenatal_bin_size" and val is None:
                continue
            if f.name == "mode":
                model[f.name] = str(val)
            else:
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
        return encode_metadata(
            {
                "artifact_kind": ArtifactKind.REFERENCE.value,
                "schema_version": self.schema_version,
                "coordinate_system": self.coordinate_system,
                "contig_lengths": dict(self.contig_lengths),
                **self.to_metadata(),
            }
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ReferenceMetadata:
        """Construct reference metadata from a decoded metadata dictionary."""
        try:
            schema_version = _parse_schema_version(data)
            coordinate_system = _parse_coordinate_system(data)
            model = _opt_obj_dict(data, "model")
            contig_lengths = _parse_contig_lengths(data.get("contig_lengths"))
            raw_ids = data.get("sample_identities", ())
            if not isinstance(raw_ids, (list, tuple)):
                raise TypeError("sample_identities must be a list or tuple")
            sample_identities: tuple[object, ...] = tuple(
                dict(x) if isinstance(x, Mapping) else str(x) for x in raw_ids
            )
            prenatal_obj = data.get("prenatal")
            if prenatal_obj is not None and not isinstance(prenatal_obj, Mapping):
                raise SchemaError("ccnv.prenatal metadata must be an object")
            prenatal_meta = (
                PrenatalModelMetadata.from_dict(prenatal_obj)
                if prenatal_obj is not None
                else None
            )
            raw_mode = model.get("mode", "germline")
            if not isinstance(raw_mode, str) or raw_mode not in (
                "germline",
                "prenatal",
            ):
                raise SchemaError(
                    f"model.mode must be 'germline' or 'prenatal', got {raw_mode!r}"
                )
            prenatal_bs = _opt_int(model, "prenatal_bin_size")
            if prenatal_bs is None and prenatal_meta is not None:
                prenatal_bs = prenatal_meta.prenatal_bin_size

            raw_ranks = model.get("branch_pca_ranks")
            branch_pca_ranks: dict[str, int] | None = None
            if raw_ranks is not None:
                if not isinstance(raw_ranks, Mapping) or any(
                    not isinstance(k, str) or not k for k in raw_ranks
                ):
                    raise TypeError("branch_pca_ranks must map non-empty names to ints")
                branch_pca_ranks = {
                    k: _as_int(v, "branch_pca_rank")
                    for k, v in raw_ranks.items()
                    if isinstance(k, str)
                }
            return cls(
                bin_size=_req_int(model, "bin_size", DEFAULT_BIN_SIZE),
                contig_lengths=contig_lengths,
                branches=_opt_str_tuple(model, "branches"),
                autosomal_contigs=_opt_str_tuple(model, "autosomal_contigs"),
                sex_contigs=_opt_str_tuple(model, "sex_contigs"),
                pca_components=_req_int(model, "pca_components", 0),
                mode=AnalysisMode(raw_mode),
                prenatal_bin_size=prenatal_bs,
                sample_identities=sample_identities,
                sex_signal_contig=_opt_str(model, "sex_signal_contig"),
                sex_cutoff=_opt_float(model, "sex_cutoff"),
                sex_source=_opt_str(model, "sex_source"),
                sample_sexes=_opt_nullable_str_tuple(model, "sample_sexes"),
                sex_inference=_opt_nullable_obj_dict(model, "sex_inference"),
                max_batch_memory_mb=_req_int(model, "max_batch_memory_mb", 256),
                flat=_req_bool(model, "flat", False),
                gc_corrected=_req_bool(model, "gc_corrected", False),
                mappability_corrected=_req_bool(model, "mappability_corrected", False),
                fragmentomics_corrected=_req_bool(
                    model, "fragmentomics_corrected", False
                ),
                min_mappability=_req_float(
                    model, "min_mappability", DEFAULT_MIN_MAPPABILITY
                ),
                branch_pca_ranks=branch_pca_ranks,
                filters=_opt_nullable_obj_dict(model, "filters"),
                schema_version=schema_version,
                coordinate_system=coordinate_system,
                prenatal=prenatal_meta,
            )
        except SchemaError:
            raise
        except (KeyError, TypeError, ValueError) as err:
            raise SchemaError(f"invalid reference metadata: {err}") from err

    @classmethod
    def from_schema(cls, schema: pa.Schema) -> ReferenceMetadata:
        """Construct reference metadata from an Arrow Schema."""
        return cls.from_dict(decode_metadata(schema.metadata))

    @classmethod
    def from_table(cls, table: pa.Table) -> ReferenceMetadata:
        """Construct reference metadata from an Arrow Table."""
        return cls.from_schema(table.schema)
