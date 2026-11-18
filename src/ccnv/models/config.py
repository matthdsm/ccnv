"""Configuration objects for ccnv pipeline stages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, Literal

from ccnv.models.domain import SamplesheetEntry
from ccnv.models.exceptions import SchemaError

DEFAULT_BIN_SIZE: Final[int] = 5000
DEFAULT_MIN_MAPPABILITY: Final[float] = 0.5

_BOOL_FILTER_KEYS: Final[tuple[str, ...]] = (
    "exclude_unmapped",
    "exclude_secondary",
    "exclude_supplementary",
    "exclude_duplicates",
    "require_proper_pair",
)


def _require_pos_int(val: object, name: str) -> None:
    if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonneg_int(val: object, name: str) -> None:
    if isinstance(val, bool) or not isinstance(val, int) or val < 0:
        raise ValueError(f"{name} must be a non-negative integer")


class AnalysisMode(StrEnum):
    """Allowed pipeline analysis modes."""

    germline = "germline"
    prenatal = "prenatal"


class SexBranch(StrEnum):
    """Allowed predict sex-branch values."""

    autosomal = "autosomal"
    female = "female"
    male = "male"


def parquet_path(prefix: Path) -> Path:
    """Canonical sample or reference Parquet path for a CLI prefix."""

    return Path(f"{prefix}.ccnv")


@dataclass(frozen=True, slots=True)
class ReadFilters:
    """Explicit per-read inclusion rules applied during extract."""

    min_mapping_quality: int = 1
    exclude_unmapped: bool = True
    exclude_secondary: bool = True
    exclude_supplementary: bool = True
    exclude_duplicates: bool = True
    require_proper_pair: bool = True

    def __post_init__(self) -> None:
        """Validate filter thresholds and boolean flags."""
        _require_nonneg_int(self.min_mapping_quality, "min_mapping_quality")
        for key in _BOOL_FILTER_KEYS:
            if not isinstance(getattr(self, key), bool):
                raise TypeError(f"{key} must be a boolean")

    def to_dict(self) -> dict[str, int | bool]:
        """Serialize read filter configuration to a plain dictionary."""
        return {
            "min_mapping_quality": int(self.min_mapping_quality),
            "exclude_unmapped": bool(self.exclude_unmapped),
            "exclude_secondary": bool(self.exclude_secondary),
            "exclude_supplementary": bool(self.exclude_supplementary),
            "exclude_duplicates": bool(self.exclude_duplicates),
            "require_proper_pair": bool(self.require_proper_pair),
        }

    @classmethod
    def from_dict(
        cls, data: Mapping[str, object], *, strict: bool = False
    ) -> ReadFilters:
        """Parse read filters from metadata dictionary with optional strict validation."""
        if not isinstance(data, Mapping):
            if strict:
                raise SchemaError("filters must be an object")
            return cls()

        defaults = cls()
        raw_mq = data.get("min_mapping_quality")
        if isinstance(raw_mq, int) and not isinstance(raw_mq, bool) and raw_mq >= 0:
            min_mq = int(raw_mq)
        elif strict:
            raise SchemaError(
                "filters.min_mapping_quality must be a non-negative integer"
            )
        else:
            min_mq = defaults.min_mapping_quality

        bool_vals: dict[str, bool] = {}
        for key in _BOOL_FILTER_KEYS:
            if key not in data:
                bool_vals[key] = bool(getattr(defaults, key))
                continue
            raw_val = data[key]
            if isinstance(raw_val, bool):
                bool_vals[key] = raw_val
            elif strict:
                raise SchemaError(f"filters.{key} must be a boolean")
            else:
                bool_vals[key] = bool(getattr(defaults, key))

        return cls(
            min_mapping_quality=min_mq,
            exclude_unmapped=bool_vals["exclude_unmapped"],
            exclude_secondary=bool_vals["exclude_secondary"],
            exclude_supplementary=bool_vals["exclude_supplementary"],
            exclude_duplicates=bool_vals["exclude_duplicates"],
            require_proper_pair=bool_vals["require_proper_pair"],
        )


@dataclass(frozen=True, slots=True)
class ExtractConfig:
    alignment: Path
    prefix: Path
    bin_size: int = DEFAULT_BIN_SIZE
    fasta: Path | None = None
    filters: ReadFilters = ReadFilters()
    threads: int = 1
    all_contigs: bool = False

    def __post_init__(self) -> None:
        """Validate extraction configuration invariants."""
        _require_pos_int(self.bin_size, "bin_size")
        _require_pos_int(self.threads, "threads")


@dataclass(frozen=True, slots=True)
class ReferenceConfig:
    prefix: Path = Path()
    samplesheet_path: Path | None = None
    entries: tuple[SamplesheetEntry, ...] = ()
    fasta: Path | None = None
    autosomal_contigs: tuple[str, ...] | None = None
    sex_contigs: tuple[str, ...] | None = None
    sex_signal_contig: str | None = None
    sex_cutoff: float | None = None
    pca_components: int | None = None
    max_batch_memory_mb: int = 256
    min_cohort_size: int = 10
    min_sex_group_size: int = 5
    bin_size: int | None = None
    all_contigs: bool = False
    qc_report: bool = True
    threads: int = 1
    mappability: Path | None = None
    gc: Path | None = None
    min_mappability: float = DEFAULT_MIN_MAPPABILITY
    mode: AnalysisMode | Literal["germline", "prenatal"] = AnalysisMode.germline
    prenatal_bin_size: int = 100000
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate reference configuration invariants."""
        _require_pos_int(self.threads, "threads")
        if self.pca_components is not None:
            _require_pos_int(self.pca_components, "pca_components")
        _require_pos_int(self.max_batch_memory_mb, "max_batch_memory_mb")
        _require_pos_int(self.min_cohort_size, "min_cohort_size")
        _require_nonneg_int(self.min_sex_group_size, "min_sex_group_size")
        if self.bin_size is not None:
            _require_pos_int(self.bin_size, "bin_size")
        _require_pos_int(self.prenatal_bin_size, "prenatal_bin_size")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not (0.0 <= self.min_mappability <= 1.0):
            raise ValueError("min_mappability must be between 0.0 and 1.0")
        if self.mode not in ("germline", "prenatal"):
            raise ValueError(
                f"mode must be 'germline' or 'prenatal', got {self.mode!r}"
            )


@dataclass(frozen=True, slots=True)
class ReferenceQCConfig:
    reference: Path
    prefix: Path | None = None

    def __post_init__(self) -> None:
        """Validate reference QC configuration paths."""
        if not isinstance(self.reference, Path):
            object.__setattr__(self, "reference", Path(self.reference))
        if self.prefix is not None and not isinstance(self.prefix, Path):
            object.__setattr__(self, "prefix", Path(self.prefix))


@dataclass(frozen=True, slots=True)
class PredictConfig:
    sample: Path
    reference: Path
    prefix: Path
    sex: SexBranch | Literal["autosomal", "female", "male"] = SexBranch.autosomal
    penalty: float = 1.0
    z_threshold: float = 3.0
    ratio_threshold: float | None = None
    min_reference_bins: int = 3
    mask_repeats: int = 3
    max_blacklist_gap: int = 2_000_000
    plot: bool = True
    vcf: bool = True
    bins_vcf: bool = False
    cytobands: Path | None = None
    mappability: Path | None = None
    gc: Path | None = None
    fasta: Path | None = None
    threads: int = 1
    min_mappability: float = DEFAULT_MIN_MAPPABILITY
    mode: AnalysisMode | Literal["germline", "prenatal"] = AnalysisMode.germline
    min_fetal_fraction_warning: float = 0.04
    twins: bool = False

    def __post_init__(self) -> None:
        """Validate prediction configuration invariants."""
        _require_pos_int(self.threads, "threads")
        _require_pos_int(self.min_reference_bins, "min_reference_bins")
        _require_nonneg_int(self.mask_repeats, "mask_repeats")
        _require_nonneg_int(self.max_blacklist_gap, "max_blacklist_gap")
        if not (0.0 <= self.min_mappability <= 1.0):
            raise ValueError("min_mappability must be between 0.0 and 1.0")
        if self.sex not in ("autosomal", "female", "male"):
            raise ValueError(
                f"sex must be 'autosomal', 'female', or 'male', got {self.sex!r}"
            )
        if self.mode not in ("germline", "prenatal"):
            raise ValueError(
                f"mode must be 'germline' or 'prenatal', got {self.mode!r}"
            )
        if not (0.0 <= self.min_fetal_fraction_warning <= 1.0):
            raise ValueError("min_fetal_fraction_warning must be between 0.0 and 1.0")
        if not isinstance(self.twins, bool):
            raise TypeError("twins must be a boolean")


__all__ = [
    "DEFAULT_BIN_SIZE",
    "DEFAULT_MIN_MAPPABILITY",
    "AnalysisMode",
    "ExtractConfig",
    "PredictConfig",
    "ReadFilters",
    "ReferenceConfig",
    "ReferenceQCConfig",
    "SexBranch",
    "parquet_path",
]
