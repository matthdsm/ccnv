"""Typed metadata containers and JSON codec for ccnv sample and reference artifacts."""

from __future__ import annotations

import json
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


def _req_bool(raw: object, name: str) -> bool:
    if not isinstance(raw, bool):
        raise TypeError(f"{name} must be a boolean, got {raw!r}")
    return raw


def _req_int(raw: object, name: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise TypeError(f"{name} must be an integer, got {raw!r}")
    return int(raw)


def _req_float(raw: object, name: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise TypeError(f"{name} must be numeric, got {raw!r}")
    return float(raw)


def _req_float_list(raw: object, name: str) -> list[float]:
    if not isinstance(raw, (list, tuple)):
        raise TypeError(f"{name} must be a sequence of floats, got {raw!r}")
    return [_req_float(item, name) for item in raw]


def _opt_float_list(data: Mapping[str, object], key: str) -> list[float] | None:
    raw = data.get(key)
    return None if raw is None else _req_float_list(raw, key)


def _req_int_list(data: Mapping[str, object], key: str) -> list[int]:
    raw = data[key]
    if not isinstance(raw, (list, tuple)):
        raise TypeError(f"{key} must be a sequence of integers, got {raw!r}")
    return [_req_int(item, key) for item in raw]


def _opt_int_list(data: Mapping[str, object], key: str) -> list[int] | None:
    raw = data.get(key)
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise TypeError(f"{key} must be a sequence of integers, got {raw!r}")
    return [_req_int(item, key) for item in raw]


def _req_str_tuple(raw: object, name: str) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple)) or any(not isinstance(x, str) for x in raw):
        raise TypeError(f"{name} must be a sequence of strings, got {raw!r}")
    return tuple(raw)


def _opt_str_tuple(raw: object, name: str) -> tuple[str, ...]:
    return () if raw is None else _req_str_tuple(raw, name)


def _opt_obj_dict(raw: object, name: str) -> dict[str, object]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"{name} must be a mapping, got {raw!r}")
    return {str(k): v for k, v in raw.items()}


def _check_pos_int(val: object, name: str) -> None:
    if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
        raise ValueError(f"{name} must be a positive integer, got {val!r}")


def _check_nonneg_int(val: object, name: str) -> None:
    if not isinstance(val, int) or isinstance(val, bool) or val < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {val!r}")


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

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> FragmentomicsMetadata:
        """Construct FragmentomicsMetadata from a decoded JSON dictionary."""
        return cls(
            paired_end=_req_bool(data["paired_end"], "paired_end"),
            total_paired_fragments=_req_int(
                data["total_paired_fragments"], "total_paired_fragments"
            ),
            chromatosome_mode_bp=_req_float(
                data["chromatosome_mode_bp"], "chromatosome_mode_bp"
            ),
            core_shoulder_bp=_req_float(data["core_shoulder_bp"], "core_shoulder_bp"),
            sizing_shift_bp=_req_float(data["sizing_shift_bp"], "sizing_shift_bp"),
            self_referenced_peak_ratio=_req_float(
                data["self_referenced_peak_ratio"], "self_referenced_peak_ratio"
            ),
            dinucleosome_fraction=_req_float(
                data["dinucleosome_fraction"], "dinucleosome_fraction"
            ),
            high_mw_fraction=_req_float(data["high_mw_fraction"], "high_mw_fraction"),
            helical_10bp_amplitude=_req_float(
                data["helical_10bp_amplitude"], "helical_10bp_amplitude"
            ),
            helical_10bp_phase=_req_float(
                data["helical_10bp_phase"], "helical_10bp_phase"
            ),
            motif_diversity_score=_req_float(
                data["motif_diversity_score"], "motif_diversity_score"
            ),
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
    ffy_intercept: float
    ffy_slope: float
    olm_weights: list[float]
    olm_bias: float
    ffx_intercept: float
    ffx_slope: float
    female_chrx_ratio_mean: float
    female_chrx_ratio_std: float
    ffy_std: float = 0.01
    ffa_std: float = 0.02
    ffx_std: float = 0.025
    ffa_logit_mode: bool = False
    p_fetal_density: list[float] | None = None
    p_maternal_density: list[float] | None = None
    ff_frag_weights: list[float] | None = None
    ff_frag_bias: float | None = None
    ff_frag_feature_mean: list[float] | None = None
    ff_frag_feature_scale: list[float] | None = None
    ff_size_std: float = 0.018
    msy_bg_rate: float | None = None
    clean_msy_bins: list[int] | None = None
    alpha_0: float | None = None
    alpha_1: float | None = None
    sigma_x_sys: float | None = None
    quarantined_samples: list[int] | None = None
    band_fisher_weights: list[float] | None = None
    mu_fetal_w: float | None = None
    mu_maternal_w: float | None = None

    def __post_init__(self) -> None:
        """Validate prenatal model metadata dimensions and types."""
        _check_pos_int(self.prenatal_bin_size, "prenatal_bin_size")

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
        raw_bias = data.get("ff_frag_bias")
        return cls(
            prenatal_bin_size=_req_int(data["prenatal_bin_size"], "prenatal_bin_size"),
            yfrac_threshold=_req_float(data["yfrac_threshold"], "yfrac_threshold"),
            ffy_intercept=_req_float(data["ffy_intercept"], "ffy_intercept"),
            ffy_slope=_req_float(data["ffy_slope"], "ffy_slope"),
            olm_weights=_req_float_list(data["olm_weights"], "olm_weights"),
            olm_bias=_req_float(data["olm_bias"], "olm_bias"),
            ffx_intercept=_req_float(data["ffx_intercept"], "ffx_intercept"),
            ffx_slope=_req_float(data["ffx_slope"], "ffx_slope"),
            female_chrx_ratio_mean=_req_float(
                data["female_chrx_ratio_mean"], "female_chrx_ratio_mean"
            ),
            female_chrx_ratio_std=_req_float(
                data["female_chrx_ratio_std"], "female_chrx_ratio_std"
            ),
            ffy_std=_req_float(data.get("ffy_std", 0.01), "ffy_std"),
            ffa_std=_req_float(data.get("ffa_std", 0.02), "ffa_std"),
            ffx_std=_req_float(data.get("ffx_std", 0.025), "ffx_std"),
            ffa_logit_mode=_req_bool(
                data.get("ffa_logit_mode", False), "ffa_logit_mode"
            ),
            p_fetal_density=_opt_float_list(data, "p_fetal_density"),
            p_maternal_density=_opt_float_list(data, "p_maternal_density"),
            ff_frag_weights=_opt_float_list(data, "ff_frag_weights"),
            ff_frag_bias=(
                None if raw_bias is None else _req_float(raw_bias, "ff_frag_bias")
            ),
            ff_frag_feature_mean=_opt_float_list(data, "ff_frag_feature_mean"),
            ff_frag_feature_scale=_opt_float_list(data, "ff_frag_feature_scale"),
            ff_size_std=_req_float(data.get("ff_size_std", 0.018), "ff_size_std"),
            msy_bg_rate=(
                None
                if data.get("msy_bg_rate") is None
                else _req_float(data["msy_bg_rate"], "msy_bg_rate")
            ),
            clean_msy_bins=_opt_int_list(data, "clean_msy_bins"),
            alpha_0=(
                None
                if data.get("alpha_0") is None
                else _req_float(data["alpha_0"], "alpha_0")
            ),
            alpha_1=(
                None
                if data.get("alpha_1") is None
                else _req_float(data["alpha_1"], "alpha_1")
            ),
            sigma_x_sys=(
                None
                if data.get("sigma_x_sys") is None
                else _req_float(data["sigma_x_sys"], "sigma_x_sys")
            ),
            quarantined_samples=_opt_int_list(data, "quarantined_samples"),
            band_fisher_weights=_opt_float_list(data, "band_fisher_weights"),
            mu_fetal_w=(
                None
                if data.get("mu_fetal_w") is None
                else _req_float(data["mu_fetal_w"], "mu_fetal_w")
            ),
            mu_maternal_w=(
                None
                if data.get("mu_maternal_w") is None
                else _req_float(data["mu_maternal_w"], "mu_maternal_w")
            ),
        )

    def encode(self) -> bytes:
        """Serialize metadata container to canonical JSON UTF-8 bytes."""
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()

    @classmethod
    def decode(cls, raw: bytes | str | Mapping[str, object]) -> PrenatalModelMetadata:
        """Decode prenatal model metadata from JSON bytes, string, or mapping."""
        parsed: object = (
            raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        )
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
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
        if (
            not isinstance(self.bin_size, int)
            or isinstance(self.bin_size, bool)
            or self.bin_size <= 0
        ):
            raise SchemaError(
                f"bin_size must be a positive integer, got {self.bin_size!r}"
            )
        if not isinstance(self.contig_lengths, Mapping) or not self.contig_lengths:
            raise SchemaError("contig_lengths must be a non-empty object")
        if any(
            not isinstance(k, str)
            or not k
            or not isinstance(v, int)
            or isinstance(v, bool)
            or v <= 0
            for k, v in self.contig_lengths.items()
        ):
            raise SchemaError(
                "contig_lengths must map non-empty names to positive integers"
            )
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

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SampleMetadata:
        """Construct sample metadata from a decoded Arrow metadata dictionary."""
        try:
            schema_version = str(data.get("schema_version", SCHEMA_VERSION))
            if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
                raise SchemaVersionError(
                    f"unsupported schema version {schema_version!r}; "
                    f"expected {SCHEMA_VERSION!r}"
                )
            raw_lengths = data.get("contig_lengths")
            if not isinstance(raw_lengths, Mapping) or not raw_lengths:
                raise SchemaError("contig_lengths must be a non-empty object")
            contig_lengths = {
                str(k): _req_int(v, "contig_length") for k, v in raw_lengths.items()
            }
            ref_id = _opt_obj_dict(data.get("reference_identity"), "reference_identity")
            raw_order = data.get("contig_order", ref_id.get("contig_order"))
            contig_order = (
                _req_str_tuple(raw_order, "contig_order")
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
                bin_size=_req_int(data["bin_size"], "bin_size"),
                contig_lengths=contig_lengths,
                contig_order=contig_order,
                source_identity=_opt_obj_dict(
                    data.get("source_identity"), "source_identity"
                ),
                reference_identity=ref_id,
                qc=_opt_obj_dict(data.get("qc"), "qc"),
                filters=_opt_obj_dict(data.get("filters"), "filters"),
                schema_version=schema_version,
                coordinate_system=str(data.get("coordinate_system", COORDINATE_SYSTEM)),
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
    branch_pca_ranks: dict[str, int] | None = None
    filters: dict[str, object] | None = None
    schema_version: str = SCHEMA_VERSION
    coordinate_system: str = COORDINATE_SYSTEM
    prenatal: PrenatalModelMetadata | None = None

    def __post_init__(self) -> None:
        """Validate reference metadata configuration and contig lengths."""
        _check_pos_int(self.bin_size, "bin_size")
        if self.prenatal_bin_size is not None:
            _check_pos_int(self.prenatal_bin_size, "prenatal_bin_size")
        _check_nonneg_int(self.pca_components, "pca_components")
        _check_nonneg_int(self.nearest_bins, "nearest_bins")
        _check_nonneg_int(self.max_batch_memory_mb, "max_batch_memory_mb")
        if self.branch_pca_ranks is not None:
            for b_name, b_rank in self.branch_pca_ranks.items():
                _check_nonneg_int(b_rank, f"branch_pca_ranks[{b_name!r}]")
        if not (0.0 <= self.min_mappability <= 1.0):
            raise ValueError(
                f"min_mappability must be within [0.0, 1.0], got {self.min_mappability!r}"
            )
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
        from ccnv.models.schemas import metadata_for

        return metadata_for(
            ArtifactKind.REFERENCE,
            contig_lengths=self.contig_lengths,
            values=self.to_metadata(),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ReferenceMetadata:
        """Construct reference metadata from a decoded metadata dictionary."""
        try:
            schema_version = str(data.get("schema_version", SCHEMA_VERSION))
            if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
                raise SchemaVersionError(
                    f"unsupported schema version {schema_version!r}; "
                    f"expected {SCHEMA_VERSION!r}"
                )
            model = _opt_obj_dict(data.get("model"), "model")
            raw_clen = data.get("contig_lengths", {})
            if not isinstance(raw_clen, Mapping):
                raise TypeError("contig_lengths must be a mapping")
            contig_lengths = {
                str(k): _req_int(v, "contig_length") for k, v in raw_clen.items()
            }
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
            raw_mode = str(model.get("mode", "germline"))
            if raw_mode not in ("germline", "prenatal"):
                raise ValueError(
                    f"mode must be 'germline' or 'prenatal', got {raw_mode!r}"
                )
            mode: AnalysisMode = "prenatal" if raw_mode == "prenatal" else "germline"
            raw_pbs = model.get("prenatal_bin_size")
            prenatal_bs = (
                _req_int(raw_pbs, "prenatal_bin_size")
                if raw_pbs is not None
                else (
                    prenatal_meta.prenatal_bin_size
                    if prenatal_meta is not None
                    else None
                )
            )
            raw_sexes = model.get("sample_sexes")
            raw_cutoff = model.get("sex_cutoff")
            raw_ranks = model.get("branch_pca_ranks")
            branch_pca_ranks: dict[str, int] | None = None
            if raw_ranks is not None:
                if not isinstance(raw_ranks, Mapping):
                    raise TypeError("branch_pca_ranks must be a mapping")
                branch_pca_ranks = {
                    str(k): _req_int(v, "branch_pca_rank") for k, v in raw_ranks.items()
                }
            raw_filters = model.get("filters")
            filters = (
                None if raw_filters is None else _opt_obj_dict(raw_filters, "filters")
            )
            return cls(
                bin_size=_req_int(model.get("bin_size", DEFAULT_BIN_SIZE), "bin_size"),
                contig_lengths=contig_lengths,
                branches=_opt_str_tuple(model.get("branches"), "branches"),
                autosomal_contigs=_opt_str_tuple(
                    model.get("autosomal_contigs"), "autosomal_contigs"
                ),
                sex_contigs=_opt_str_tuple(model.get("sex_contigs"), "sex_contigs"),
                pca_components=_req_int(
                    model.get("pca_components", 5), "pca_components"
                ),
                nearest_bins=_req_int(model.get("nearest_bins", 300), "nearest_bins"),
                mode=mode,
                prenatal_bin_size=prenatal_bs,
                sample_identities=sample_identities,
                sex_signal_contig=(
                    None
                    if model.get("sex_signal_contig") is None
                    else str(model["sex_signal_contig"])
                ),
                sex_cutoff=(
                    None if raw_cutoff is None else _req_float(raw_cutoff, "sex_cutoff")
                ),
                sex_source=(
                    None
                    if model.get("sex_source") is None
                    else str(model["sex_source"])
                ),
                sample_sexes=(
                    None
                    if raw_sexes is None
                    else _req_str_tuple(raw_sexes, "sample_sexes")
                ),
                sex_inference=(
                    None
                    if model.get("sex_inference") is None
                    else _opt_obj_dict(model.get("sex_inference"), "sex_inference")
                ),
                max_batch_memory_mb=_req_int(
                    model.get("max_batch_memory_mb", 256), "max_batch_memory_mb"
                ),
                flat=_req_bool(model.get("flat", False), "flat"),
                gc_corrected=_req_bool(
                    model.get("gc_corrected", False), "gc_corrected"
                ),
                mappability_corrected=_req_bool(
                    model.get("mappability_corrected", False),
                    "mappability_corrected",
                ),
                fragmentomics_corrected=_req_bool(
                    model.get("fragmentomics_corrected", False),
                    "fragmentomics_corrected",
                ),
                min_mappability=_req_float(
                    model.get("min_mappability", DEFAULT_MIN_MAPPABILITY),
                    "min_mappability",
                ),
                branch_pca_ranks=branch_pca_ranks,
                filters=filters,
                schema_version=schema_version,
                coordinate_system=str(data.get("coordinate_system", COORDINATE_SYSTEM)),
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
