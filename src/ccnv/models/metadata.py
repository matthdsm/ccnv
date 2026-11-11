"""Typed metadata containers for sample fragmentomics and prenatal reference models."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Final

import pyarrow as pa

from ccnv.models.exceptions import SchemaError

PRENATAL_METADATA_KEY: Final[bytes] = b"ccnv.prenatal"
FRAGMENTOMICS_METADATA_KEY: Final[bytes] = b"ccnv.fragmentomics"

__all__ = [
    "FRAGMENTOMICS_METADATA_KEY",
    "PRENATAL_METADATA_KEY",
    "FragmentomicsMetadata",
    "PrenatalModelMetadata",
]


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
            paired_end=bool(data["paired_end"]),
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

    def __post_init__(self) -> None:
        """Validate prenatal model metadata dimensions and types."""
        if (
            not isinstance(self.prenatal_bin_size, int)
            or isinstance(self.prenatal_bin_size, bool)
            or self.prenatal_bin_size <= 0
        ):
            raise ValueError(
                f"prenatal_bin_size must be a positive integer, "
                f"got {self.prenatal_bin_size!r}"
            )

    def to_dict(self) -> dict[str, object]:
        """Convert metadata container to a JSON-serializable dictionary."""
        out: dict[str, object] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if val is None or isinstance(val, bool):
                out[f.name] = val
            elif f.name == "prenatal_bin_size":
                out[f.name] = int(val)
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
            ffa_logit_mode=bool(data.get("ffa_logit_mode", False)),
            p_fetal_density=_opt_float_list(data, "p_fetal_density"),
            p_maternal_density=_opt_float_list(data, "p_maternal_density"),
            ff_frag_weights=_opt_float_list(data, "ff_frag_weights"),
            ff_frag_bias=(
                None if raw_bias is None else _req_float(raw_bias, "ff_frag_bias")
            ),
            ff_frag_feature_mean=_opt_float_list(data, "ff_frag_feature_mean"),
            ff_frag_feature_scale=_opt_float_list(data, "ff_frag_feature_scale"),
            ff_size_std=_req_float(data.get("ff_size_std", 0.018), "ff_size_std"),
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
