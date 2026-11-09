"""Versioned Arrow contracts for ccnv Parquet artifacts."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import numpy as np
import pyarrow as pa

from ccnv.models.config import DEFAULT_BIN_SIZE, DEFAULT_MIN_MAPPABILITY, AnalysisMode
from ccnv.models.exceptions import SchemaError

SCHEMA_VERSION: Final[str] = "2.0.0"
SUPPORTED_SCHEMA_VERSIONS: Final[frozenset[str]] = frozenset({"2.0.0"})
COORDINATE_SYSTEM: Final[str] = "0-based-half-open"
METADATA_PREFIX: Final[str] = "ccnv."
PRENATAL_METADATA_KEY: Final[bytes] = b"ccnv.prenatal"

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
    {
        BRANCH_AUTOSOMAL,
        BRANCH_FEMALE_SEX,
        BRANCH_MALE_SEX,
    }
)


class ArtifactKind(StrEnum):
    SAMPLE = "sample"
    REFERENCE = "reference"


@dataclass(frozen=True, slots=True)
class PrenatalModelMetadata:
    """Scalar parameters for prenatal models serialized under ccnv.prenatal."""

    prenatal_bin_size: int
    yfrac_threshold: float
    ffy_intercept: float
    ffy_slope: float
    mlp_w1: list[list[float]]
    mlp_b1: list[float]
    mlp_w2: list[list[float]]
    mlp_b2: list[float]
    mlp_skew_intercept: float
    mlp_skew_slope: float
    olm_weights: list[float]
    olm_bias: float
    olm_skew_intercept: float
    olm_skew_slope: float
    ffx_intercept: float
    ffx_slope: float
    female_chrx_ratio_mean: float
    female_chrx_ratio_std: float
    ffy_std: float = 0.01
    ffa_std: float = 0.02
    ffx_std: float = 0.025
    ffa_logit_mode: bool = False

    @property
    def preface_std(self) -> float:
        """Deprecated alias for ffa_std."""
        return self.ffa_std

    @property
    def preface_logit_mode(self) -> bool:
        """Deprecated alias for ffa_logit_mode."""
        return self.ffa_logit_mode

    def __post_init__(self) -> None:
        if (
            not isinstance(self.prenatal_bin_size, int)
            or isinstance(self.prenatal_bin_size, bool)
            or self.prenatal_bin_size <= 0
        ):
            raise ValueError(
                f"prenatal_bin_size must be a positive integer, "
                f"got {self.prenatal_bin_size!r}"
            )
        if not isinstance(self.mlp_b2, (list, tuple)):
            object.__setattr__(self, "mlp_b2", [float(self.mlp_b2)])

    def to_dict(self) -> dict[str, Any]:
        return {
            "prenatal_bin_size": int(self.prenatal_bin_size),
            "yfrac_threshold": float(self.yfrac_threshold),
            "ffy_intercept": float(self.ffy_intercept),
            "ffy_slope": float(self.ffy_slope),
            "mlp_w1": [[float(x) for x in row] for row in self.mlp_w1],
            "mlp_b1": [float(x) for x in self.mlp_b1],
            "mlp_w2": [[float(x) for x in row] for row in self.mlp_w2],
            "mlp_b2": [float(x) for x in self.mlp_b2],
            "mlp_skew_intercept": float(self.mlp_skew_intercept),
            "mlp_skew_slope": float(self.mlp_skew_slope),
            "olm_weights": [float(x) for x in self.olm_weights],
            "olm_bias": float(self.olm_bias),
            "olm_skew_intercept": float(self.olm_skew_intercept),
            "olm_skew_slope": float(self.olm_skew_slope),
            "ffx_intercept": float(self.ffx_intercept),
            "ffx_slope": float(self.ffx_slope),
            "female_chrx_ratio_mean": float(self.female_chrx_ratio_mean),
            "female_chrx_ratio_std": float(self.female_chrx_ratio_std),
            "ffy_std": float(self.ffy_std),
            "ffa_std": float(self.ffa_std),
            "ffx_std": float(self.ffx_std),
            "ffa_logit_mode": bool(self.ffa_logit_mode),
            "preface_std": float(self.ffa_std),
            "preface_logit_mode": bool(self.ffa_logit_mode),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PrenatalModelMetadata:
        raw_bin_size = data["prenatal_bin_size"]
        if isinstance(raw_bin_size, bool):
            raise TypeError(
                f"prenatal_bin_size must be a positive integer, got {raw_bin_size!r}"
            )
        raw_b2 = data["mlp_b2"]
        mlp_b2 = (
            [float(x) for x in raw_b2]
            if isinstance(raw_b2, (list, tuple))
            else [float(raw_b2)]
        )
        ffa_std_val = float(data.get("ffa_std", data.get("preface_std", 0.02)))
        ffa_logit_val = bool(
            data.get("ffa_logit_mode", data.get("preface_logit_mode", False))
        )
        return cls(
            prenatal_bin_size=int(raw_bin_size),
            yfrac_threshold=float(data["yfrac_threshold"]),
            ffy_intercept=float(data["ffy_intercept"]),
            ffy_slope=float(data["ffy_slope"]),
            mlp_w1=[[float(x) for x in row] for row in data["mlp_w1"]],
            mlp_b1=[float(x) for x in data["mlp_b1"]],
            mlp_w2=[[float(x) for x in row] for row in data["mlp_w2"]],
            mlp_b2=mlp_b2,
            mlp_skew_intercept=float(data["mlp_skew_intercept"]),
            mlp_skew_slope=float(data["mlp_skew_slope"]),
            olm_weights=[float(x) for x in data["olm_weights"]],
            olm_bias=float(data["olm_bias"]),
            olm_skew_intercept=float(data["olm_skew_intercept"]),
            olm_skew_slope=float(data["olm_skew_slope"]),
            ffx_intercept=float(data["ffx_intercept"]),
            ffx_slope=float(data["ffx_slope"]),
            female_chrx_ratio_mean=float(data["female_chrx_ratio_mean"]),
            female_chrx_ratio_std=float(data["female_chrx_ratio_std"]),
            ffy_std=float(data.get("ffy_std", 0.01)),
            ffa_std=ffa_std_val,
            ffx_std=float(data.get("ffx_std", 0.025)),
            ffa_logit_mode=ffa_logit_val,
        )

    def encode(self) -> bytes:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()

    @classmethod
    def decode(cls, raw: bytes | str | Mapping[str, Any]) -> PrenatalModelMetadata:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, Mapping):
            raise SchemaError(
                f"invalid prenatal metadata payload: expected mapping, got {type(raw)}"
            )
        try:
            return cls.from_dict(raw)
        except (ValueError, TypeError, KeyError) as error:
            raise SchemaError(f"invalid prenatal metadata: {error}") from error

    @classmethod
    def from_metadata(
        cls, metadata: Mapping[bytes, bytes] | None
    ) -> PrenatalModelMetadata | None:
        if metadata is None:
            return None
        raw = metadata.get(PRENATAL_METADATA_KEY)
        if raw is None:
            return None
        return cls.decode(raw)

    @classmethod
    def from_table(cls, table: pa.Table) -> PrenatalModelMetadata | None:
        return cls.from_metadata(table.schema.metadata)


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
    min_mappability: float = DEFAULT_MIN_MAPPABILITY
    schema_version: str = SCHEMA_VERSION
    coordinate_system: str = COORDINATE_SYSTEM
    prenatal: PrenatalModelMetadata | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.bin_size, int)
            or isinstance(self.bin_size, bool)
            or self.bin_size <= 0
        ):
            raise ValueError(
                f"bin_size must be a positive integer, got {self.bin_size!r}"
            )
        if self.prenatal_bin_size is not None and (
            not isinstance(self.prenatal_bin_size, int)
            or isinstance(self.prenatal_bin_size, bool)
            or self.prenatal_bin_size <= 0
        ):
            raise ValueError(
                f"prenatal_bin_size must be a positive integer, "
                f"got {self.prenatal_bin_size!r}"
            )
        if self.mode not in ("germline", "prenatal"):
            raise ValueError(
                f"mode must be 'germline' or 'prenatal', got {self.mode!r}"
            )
        if not self.contig_lengths:
            raise ValueError("contig_lengths must be a non-empty mapping")

    def to_metadata(self) -> dict[str, object]:
        model: dict[str, object] = {
            "bin_size": self.bin_size,
            "branches": list(self.branches),
            "autosomal_contigs": list(self.autosomal_contigs),
            "sex_contigs": list(self.sex_contigs),
            "pca_components": self.pca_components,
            "nearest_bins": self.nearest_bins,
            "mode": self.mode,
            "prenatal_bin_size": self.prenatal_bin_size,
            "sex_signal_contig": self.sex_signal_contig,
            "sex_cutoff": self.sex_cutoff,
            "sex_source": self.sex_source,
            "sample_sexes": (
                list(self.sample_sexes) if self.sample_sexes is not None else None
            ),
            "sex_inference": self.sex_inference,
            "max_batch_memory_mb": self.max_batch_memory_mb,
            "flat": self.flat,
            "gc_corrected": self.gc_corrected,
            "mappability_corrected": self.mappability_corrected,
            "min_mappability": self.min_mappability,
        }
        values: dict[str, object] = {
            "sample_identities": list(self.sample_identities),
            "model": model,
        }
        if self.prenatal is not None:
            values["prenatal"] = self.prenatal.to_dict()
        return values

    def build_arrow_metadata(self) -> dict[bytes, bytes]:
        return metadata_for(
            ArtifactKind.REFERENCE,
            contig_lengths=self.contig_lengths,
            values=self.to_metadata(),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ReferenceMetadata:
        model_obj = data.get("model")
        model = dict(model_obj) if isinstance(model_obj, Mapping) else {}
        contig_lengths_obj = data.get("contig_lengths", {})
        contig_lengths = (
            {str(k): int(v) for k, v in contig_lengths_obj.items()}
            if isinstance(contig_lengths_obj, Mapping)
            else {}
        )
        sample_ids_obj = data.get("sample_identities", ())
        sample_identities: tuple[object, ...] = (
            tuple(dict(x) if isinstance(x, Mapping) else str(x) for x in sample_ids_obj)
            if isinstance(sample_ids_obj, (list, tuple))
            else ()
        )
        branches_obj = model.get("branches", ())
        branches = (
            tuple(str(x) for x in branches_obj)
            if isinstance(branches_obj, (list, tuple))
            else ()
        )
        autosomal_obj = model.get("autosomal_contigs", ())
        autosomal_contigs = (
            tuple(str(x) for x in autosomal_obj)
            if isinstance(autosomal_obj, (list, tuple))
            else ()
        )
        sex_obj = model.get("sex_contigs", ())
        sex_contigs = (
            tuple(str(x) for x in sex_obj) if isinstance(sex_obj, (list, tuple)) else ()
        )
        sample_sexes_obj = model.get("sample_sexes")
        sample_sexes = (
            tuple(str(x) for x in sample_sexes_obj)
            if isinstance(sample_sexes_obj, (list, tuple))
            else None
        )
        prenatal_meta: PrenatalModelMetadata | None = None
        if "prenatal" in data and isinstance(data["prenatal"], Mapping):
            prenatal_meta = PrenatalModelMetadata.from_dict(data["prenatal"])

        raw_mode = str(model.get("mode", "germline"))
        if raw_mode not in ("germline", "prenatal"):
            raise ValueError(f"mode must be 'germline' or 'prenatal', got {raw_mode!r}")
        mode: AnalysisMode = "prenatal" if raw_mode == "prenatal" else "germline"

        return cls(
            bin_size=int(model.get("bin_size", DEFAULT_BIN_SIZE)),
            contig_lengths=contig_lengths,
            branches=branches,
            autosomal_contigs=autosomal_contigs,
            sex_contigs=sex_contigs,
            pca_components=int(model.get("pca_components", 5)),
            nearest_bins=int(model.get("nearest_bins", 300)),
            mode=mode,
            prenatal_bin_size=(
                int(model["prenatal_bin_size"])
                if model.get("prenatal_bin_size") is not None
                else (
                    prenatal_meta.prenatal_bin_size
                    if prenatal_meta is not None
                    else None
                )
            ),
            sample_identities=sample_identities,
            sex_signal_contig=(
                str(model["sex_signal_contig"])
                if model.get("sex_signal_contig") is not None
                else None
            ),
            sex_cutoff=(
                float(model["sex_cutoff"])
                if model.get("sex_cutoff") is not None
                else None
            ),
            sex_source=(
                str(model["sex_source"])
                if model.get("sex_source") is not None
                else None
            ),
            sample_sexes=sample_sexes,
            sex_inference=(
                dict(model["sex_inference"])
                if isinstance(model.get("sex_inference"), Mapping)
                else None
            ),
            max_batch_memory_mb=int(model.get("max_batch_memory_mb", 256)),
            flat=bool(model.get("flat", False)),
            gc_corrected=bool(model.get("gc_corrected", False)),
            mappability_corrected=bool(model.get("mappability_corrected", False)),
            min_mappability=float(
                model.get("min_mappability", DEFAULT_MIN_MAPPABILITY)
            ),
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            coordinate_system=str(data.get("coordinate_system", COORDINATE_SYSTEM)),
            prenatal=prenatal_meta,
        )

    @classmethod
    def from_schema(cls, schema: pa.Schema) -> ReferenceMetadata:
        return cls.from_dict(decode_metadata(schema.metadata))

    @classmethod
    def from_table(cls, table: pa.Table) -> ReferenceMetadata:
        return cls.from_schema(table.schema)


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
        pa.field("gc", pa.float32(), nullable=False),
        pa.field("mappability", pa.float32(), nullable=False),
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

    if kind is ArtifactKind.REFERENCE:
        model = metadata.get("model")
        if not isinstance(model, dict):
            raise SchemaError("model must be an object")
        bin_size = model.get("bin_size")
        if bin_size is not None and (
            not isinstance(bin_size, int) or isinstance(bin_size, bool) or bin_size <= 0
        ):
            raise SchemaError("model.bin_size must be a positive integer")
        prenatal_bin_size = model.get("prenatal_bin_size")
        if prenatal_bin_size is not None and (
            not isinstance(prenatal_bin_size, int)
            or isinstance(prenatal_bin_size, bool)
            or prenatal_bin_size <= 0
        ):
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


def _validate_reference_table_content(
    table: pa.Table,
    metadata: Mapping[str, object],
    contig_lengths: Mapping[str, int],
) -> None:
    """Validate reference branches, coordinate bounds, and per-branch bin spacing."""

    model_obj = metadata.get("model")
    model = model_obj if isinstance(model_obj, Mapping) else {}
    bin_size = model.get("bin_size")
    if not isinstance(bin_size, int) or isinstance(bin_size, bool) or bin_size <= 0:
        bin_size = None

    prenatal_bin_size = model.get("prenatal_bin_size")
    if prenatal_bin_size is None and isinstance(metadata.get("prenatal"), Mapping):
        prenatal_bin_size = metadata["prenatal"].get("prenatal_bin_size")
    if (
        not isinstance(prenatal_bin_size, int)
        or isinstance(prenatal_bin_size, bool)
        or prenatal_bin_size <= 0
    ):
        prenatal_bin_size = None

    unique_branches = set(table.column("branch").unique().to_pylist())
    unknown_branches = unique_branches - KNOWN_REFERENCE_BRANCHES
    if unknown_branches:
        raise SchemaError(
            f"unknown branch(es): {sorted(unknown_branches)}; "
            f"expected one of {sorted(KNOWN_REFERENCE_BRANCHES)}"
        )

    if BRANCH_PRENATAL_AUTOSOMAL in unique_branches and prenatal_bin_size is None:
        raise SchemaError(
            "table contains 'prenatal_autosomal' branch but prenatal_bin_size "
            "is not specified in metadata"
        )

    # Fast path for large tables (> 1000 rows)
    if table.num_rows > 1000:
        starts = table.column("start").to_numpy(zero_copy_only=False)
        ends = table.column("end").to_numpy(zero_copy_only=False)
        contig_arr = np.asarray(table.column("contig").to_numpy(zero_copy_only=False))
        branch_arr = np.asarray(table.column("branch").to_numpy(zero_copy_only=False))
        unique_contigs = set(table.column("contig").unique().to_pylist())

        all_contigs_declared = unique_contigs.issubset(contig_lengths.keys())
        basic_bounds_ok = bool(np.all(starts >= 0) and np.all(ends > starts))
        lengths_ok = all_contigs_declared and all(
            bool(np.all(ends[contig_arr == c] <= contig_lengths[c]))
            for c in unique_contigs
        )
        no_duplicates = (
            table.group_by(["branch", "contig", "start", "end"]).aggregate([]).num_rows
            == table.num_rows
        )

        spacing_ok = all_contigs_declared
        if spacing_ok:
            for b in unique_branches:
                target_size = (
                    prenatal_bin_size if b == BRANCH_PRENATAL_AUTOSOMAL else bin_size
                )
                if target_size is not None:
                    mask = branch_arr == b
                    b_starts = starts[mask]
                    b_ends = ends[mask]
                    b_contigs = contig_arr[mask]
                    widths = b_ends - b_starts

                    if not bool(np.all(b_starts % target_size == 0)):
                        spacing_ok = False
                        break

                    for c in set(b_contigs):
                        c_mask = b_contigs == c
                        c_ends = b_ends[c_mask]
                        c_widths = widths[c_mask]
                        c_len = contig_lengths[c]
                        is_final = c_ends == c_len
                        if not bool(
                            np.all(c_widths[~is_final] == target_size)
                            and np.all(
                                (c_widths[is_final] > 0)
                                & (c_widths[is_final] <= target_size)
                            )
                        ):
                            spacing_ok = False
                            break
                    if not spacing_ok:
                        break

        if (
            all_contigs_declared
            and basic_bounds_ok
            and lengths_ok
            and no_duplicates
            and spacing_ok
        ):
            return

    # Row-by-row validation
    branches: list[str] = table.column("branch").to_pylist()
    contigs: list[str] = table.column("contig").to_pylist()
    starts_list: list[int] = table.column("start").to_pylist()
    ends_list: list[int] = table.column("end").to_pylist()

    seen: set[tuple[str, str, int, int]] = set()
    for index, (branch, contig, start, end) in enumerate(
        zip(branches, contigs, starts_list, ends_list, strict=True)
    ):
        if branch not in KNOWN_REFERENCE_BRANCHES:
            raise SchemaError(
                f"row {index} uses unknown branch {branch!r}; "
                f"expected one of {sorted(KNOWN_REFERENCE_BRANCHES)}"
            )
        length = contig_lengths.get(contig)
        if not isinstance(length, int):
            raise SchemaError(f"row {index} uses undeclared contig {contig!r}")
        if start < 0 or end <= start or end > length:
            raise SchemaError(
                f"row {index} interval [{start}, {end}) is outside "
                f"{contig!r} length {length}"
            )

        target_size = (
            prenatal_bin_size if branch == BRANCH_PRENATAL_AUTOSOMAL else bin_size
        )
        if target_size is not None:
            width = end - start
            if end == length:
                if width <= 0 or width > target_size:
                    raise SchemaError(
                        f"row {index} ({branch}, {contig}:{start}-{end}) "
                        f"final bin width {width} outside expected (0, {target_size}]"
                    )
            else:
                if width != target_size:
                    raise SchemaError(
                        f"row {index} ({branch}, {contig}:{start}-{end}) "
                        f"bin width {width} does not match {target_size}"
                    )
            if start % target_size != 0:
                raise SchemaError(
                    f"row {index} ({branch}, {contig}:{start}-{end}) "
                    f"start coordinate {start} is not a multiple of "
                    f"expected bin size {target_size}"
                )

        identity = (branch, contig, start, end)
        if identity in seen:
            raise SchemaError(f"row {index} duplicates bin identity {identity!r}")
        seen.add(identity)


def _validate_sample_table_content(
    table: pa.Table,
    contig_lengths: Mapping[str, int],
) -> None:
    """Validate sample coordinate bounds and bin identity."""

    # Fast path for large tables (> 1000 rows)
    if table.num_rows > 1000:
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
        no_duplicates = (
            table.group_by(["contig", "start", "end"]).aggregate([]).num_rows
            == table.num_rows
        )
        if all_contigs_declared and basic_bounds_ok and lengths_ok and no_duplicates:
            return

    contigs: list[str] = table.column("contig").to_pylist()
    starts_list: list[int] = table.column("start").to_pylist()
    ends_list: list[int] = table.column("end").to_pylist()

    seen: set[tuple[str, int, int]] = set()
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
        identity = (contig, start, end)
        if identity in seen:
            raise SchemaError(f"row {index} duplicates bin identity {identity!r}")
        seen.add(identity)


def validate_table(table: pa.Table, kind: ArtifactKind) -> None:
    """Validate schema plus canonical coordinate bounds and row identity."""

    validate_schema(table.schema, kind)
    metadata = decode_metadata(table.schema.metadata)
    contig_lengths = metadata["contig_lengths"]
    assert isinstance(contig_lengths, dict)

    if kind is ArtifactKind.REFERENCE:
        _validate_reference_table_content(table, metadata, contig_lengths)
    else:
        _validate_sample_table_content(table, contig_lengths)


def validate_sample_table(table: pa.Table) -> None:
    """Validate sample schema, version 2.0.0 metadata, and coordinate intervals."""

    validate_table(table, ArtifactKind.SAMPLE)


def validate_reference_table(table: pa.Table) -> None:
    """Validate reference schema, version 2.0.0 metadata, branches, and bin spacing."""

    validate_table(table, ArtifactKind.REFERENCE)


__all__ = [
    "BRANCH_AUTOSOMAL",
    "BRANCH_FEMALE_SEX",
    "BRANCH_MALE_SEX",
    "BRANCH_PRENATAL_AUTOSOMAL",
    "BRANCH_PRENATAL_FFY",
    "CNV_REFERENCE_BRANCHES",
    "COORDINATE_SYSTEM",
    "KNOWN_REFERENCE_BRANCHES",
    "METADATA_PREFIX",
    "PRENATAL_METADATA_KEY",
    "REFERENCE_SCHEMA",
    "SAMPLE_SCHEMA",
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "ArtifactKind",
    "PrenatalModelMetadata",
    "ReferenceMetadata",
    "SchemaError",
    "bin_coordinates",
    "decode_metadata",
    "encode_metadata",
    "metadata_for",
    "validate_metadata",
    "validate_reference_table",
    "validate_sample_table",
    "validate_schema",
    "validate_table",
]
