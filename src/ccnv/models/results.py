"""Typed prediction results shared by the numeric pipeline and writers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

CallType = Literal["gain", "loss"]
SCACall = Literal["XX", "XY", "XO", "XXX", "XXY", "XYY", "DISCORDANT_MIXED"]
FetalSex = Literal["M", "F"]
ActiveFFSource = Literal["FFY", "FFA"]


@dataclass(frozen=True, slots=True)
class BinResult:
    """Per-bin normalized copy-number evidence."""

    branch: str
    contigs: tuple[str, ...]
    starts: NDArray[np.int64]
    ends: NDArray[np.int64]
    ratios: NDArray[np.float64]
    zscores: NDArray[np.float64]
    weights: NDArray[np.float64]
    reference_sizes: NDArray[np.int64]
    masked: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class SegmentResult:
    """One segmented interval with aggregate evidence and an optional call."""

    contig: str
    start: int
    end: int
    ratio: float
    zscore: float
    weight: float
    bin_count: int
    call: CallType | None
    expected_ratio: float | None = None
    implied_ff: float | None = None
    discordant: bool = False
    discordance_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SCAResult:
    """Sex Chromosome Aneuploidy (SCA) classification result."""

    call: SCACall
    is_aneuploid: bool
    confidence: float
    chrx_zscore: float
    ff_delta: float | None = None

    def __post_init__(self) -> None:
        if self.call not in ("XX", "XY", "XO", "XXX", "XXY", "XYY", "DISCORDANT_MIXED"):
            raise ValueError(f"invalid SCA call: {self.call!r}")


@dataclass(frozen=True, slots=True)
class FetalFractionResult:
    """Prenatal fetal fraction and sex chromosome analysis results."""

    fetal_sex: FetalSex
    active_ff: float
    active_ff_source: ActiveFFSource
    ffy: float | None
    ffx: float
    ffa: float
    chrx_ratio: float
    chry_score: float
    yfrac_threshold: float
    sca: SCAResult
    low_ff_warning: bool = False
    warnings: tuple[str, ...] = ()

    @property
    def preface(self) -> float:
        """Alias for ffa (fetal fraction autosomal)."""
        return self.ffa

    def __post_init__(self) -> None:
        if self.fetal_sex not in ("M", "F"):
            raise ValueError(f"fetal_sex must be 'M' or 'F', got {self.fetal_sex!r}")
        if self.active_ff_source not in ("FFY", "FFA"):
            raise ValueError(
                f"active_ff_source must be 'FFY' or 'FFA', "
                f"got {self.active_ff_source!r}"
            )
        if not isinstance(self.warnings, tuple):
            object.__setattr__(self, "warnings", tuple(self.warnings))


@dataclass(frozen=True, slots=True)
class PredictionOutputs:
    """Paths written by prediction."""

    bins_bed: Path
    segments_bed: Path
    calls_bed: Path
    plot_html: Path | None
    bins_vcf: Path | None
    segments_vcf: Path | None
    calls_vcf: Path | None
    bins: tuple[BinResult, ...]
    segments: tuple[SegmentResult, ...]
    qc_json_path: Path | None = None
    prenatal_result: FetalFractionResult | None = None


@dataclass(frozen=True, slots=True)
class ReferenceQCOutputs:
    """Artifacts and verdict produced by reference QC."""

    json_path: Path
    overall_verdict: str
    worst_severity: int
    data: dict[str, object]


__all__ = [
    "ActiveFFSource",
    "BinResult",
    "CallType",
    "FetalFractionResult",
    "FetalSex",
    "PredictionOutputs",
    "ReferenceQCOutputs",
    "SCACall",
    "SCAResult",
    "SegmentResult",
]
