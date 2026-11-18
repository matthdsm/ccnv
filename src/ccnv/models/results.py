"""Typed prediction results shared by the numeric pipeline and writers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

CallType = Literal["gain", "loss"]
SCACall = Literal["XX", "XY", "XO", "XXX", "XXY", "XYY", "DISCORDANT_MIXED"]
FetalSex = Literal["M", "F"]
ActiveFFSource = Literal["FFY", "FFA", "FFX", "FF_SIZE", "FUSED"]
GestationType = Literal["singleton", "twin"]
TwinSexPair = Literal["FF", "FM", "MM"]


class ScanQcCounts(TypedDict):
    """Effective read filter counters returned by the Rust alignment scanner."""

    reads_seen: int
    reads_kept: int
    mate_dedup_skipped: int
    reads_filtered: int
    autosomal_reads_kept: int
    sex_reads_kept: int
    filter_unmapped: int
    filter_no_start: int
    filter_secondary: int
    filter_supplementary: int
    filter_duplicate: int
    filter_improper_pair: int
    filter_mapq: int


class ScanAlignmentResult(TypedDict):
    """Per-bin channels, histograms, and QC counters returned by scan_alignment_bins."""

    count: NDArray[np.uint32]
    paired_count: NDArray[np.uint32]
    frag_len_sum: NDArray[np.uint64]
    short_frag_count: NDArray[np.uint32]
    endpoint_count: NDArray[np.uint32]
    frag_gc_sum: NDArray[np.float32]
    frag_bands: NDArray[np.uint32] | None
    hist_autosomal: NDArray[np.uint64]
    hist_chrx: NDArray[np.uint64]
    hist_chry: NDArray[np.uint64]
    motif_counts: NDArray[np.uint64]
    paired_end: bool
    qc_counts: ScanQcCounts


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
    counts: NDArray[np.float64] | None = None
    expected_counts: NDArray[np.float64] | None = None
    background_counts: NDArray[np.float64] | None = None
    baseline_log2_var: float = 0.0


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
    frag_short_ratio: float | None = None
    frag_llr_size: float | None = None
    origin_call: str | None = None
    copy_number: int | None = None
    log_bf: float | None = None
    q_value: float = 1.0
    mosaic_fraction: float | None = None
    mosaic_ci: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class PrenatalLoDMetrics:
    """Analytical Limit of Detection (LoD) minimum fetal fraction thresholds."""

    lod_ff_35mb: float
    lod_ff_3mb: float


@dataclass(frozen=True, slots=True)
class SCAResult:
    """Sex Chromosome Aneuploidy (SCA) classification result."""

    call: SCACall
    is_aneuploid: bool
    confidence: float
    chrx_zscore: float
    ff_delta: float | None = None

    def __post_init__(self) -> None:
        """Validate SCA call validity."""
        if self.call not in ("XX", "XY", "XO", "XXX", "XXY", "XYY", "DISCORDANT_MIXED"):
            raise ValueError(f"invalid SCA call: {self.call!r}")


@dataclass(frozen=True, slots=True)
class TwinResult:
    """Twin pregnancy and fetal sex-pair prediction result."""

    call: TwinSexPair
    fetal_sexes: tuple[FetalSex, FetalSex]
    is_twin: bool
    twin_probability: float
    confidence: float
    posteriors: dict[TwinSexPair, float]
    male_ff: float
    female_ff: float

    def __post_init__(self) -> None:
        """Validate twin sex-pair call and fetal sexes."""
        if self.call not in ("FF", "FM", "MM"):
            raise ValueError(f"invalid twin call: {self.call!r}")
        sex_tuple = tuple(self.fetal_sexes)
        if sex_tuple not in (("F", "F"), ("F", "M"), ("M", "M")):
            raise ValueError(f"invalid twin fetal_sexes: {self.fetal_sexes!r}")
        if sex_tuple != (self.call[0], self.call[1]):
            raise ValueError(
                f"twin call {self.call!r} does not match fetal_sexes {self.fetal_sexes!r}"
            )
        if not isinstance(self.fetal_sexes, tuple):
            object.__setattr__(self, "fetal_sexes", sex_tuple)


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
    ff_size: float | None = None
    low_ff_warning: bool = False
    warnings: tuple[str, ...] = ()
    active_ff_se: float | None = None
    effective_ff: float | None = None
    lod_metrics: PrenatalLoDMetrics | None = None
    twin: TwinResult | None = None
    gestation: GestationType = "singleton"
    fetal_sexes: tuple[FetalSex, FetalSex] | None = None
    male_ff: float | None = None
    female_ff: float | None = None
    twin_confidence: float | None = None

    def __post_init__(self) -> None:
        """Validate fetal sex, active FF source, and twin gestation invariants."""
        if self.fetal_sex not in ("M", "F"):
            raise ValueError(f"fetal_sex must be 'M' or 'F', got {self.fetal_sex!r}")
        if self.active_ff_source not in ("FFY", "FFA", "FFX", "FF_SIZE", "FUSED"):
            raise ValueError(
                "active_ff_source must be 'FFY', 'FFA', 'FFX', 'FF_SIZE', or 'FUSED', "
                f"got {self.active_ff_source!r}"
            )
        if self.fetal_sexes is not None:
            sex_pair = tuple(self.fetal_sexes)
            if sex_pair not in (("F", "F"), ("F", "M"), ("M", "M")):
                raise ValueError(
                    "fetal_sexes must be ('F', 'F'), ('F', 'M'), or ('M', 'M'), "
                    f"got {self.fetal_sexes!r}"
                )
            if not isinstance(self.fetal_sexes, tuple):
                object.__setattr__(self, "fetal_sexes", sex_pair)
        if self.twin is not None:
            if (
                self.twin.is_twin
                and self.gestation == "singleton"
                and self.fetal_sexes is None
            ):
                object.__setattr__(self, "gestation", "twin")
            if (self.gestation == "twin") != self.twin.is_twin:
                raise ValueError(
                    f"gestation {self.gestation!r} conflicts with twin.is_twin={self.twin.is_twin}"
                )
            if (
                self.fetal_sexes is not None
                and self.fetal_sexes != self.twin.fetal_sexes
            ):
                raise ValueError(
                    f"fetal_sexes {self.fetal_sexes!r} conflicts with twin.fetal_sexes={self.twin.fetal_sexes!r}"
                )
            if self.gestation == "twin":
                if self.fetal_sexes is None:
                    object.__setattr__(self, "fetal_sexes", self.twin.fetal_sexes)
                if self.twin_confidence is None:
                    object.__setattr__(self, "twin_confidence", self.twin.confidence)
                if (
                    self.fetal_sex == "F"
                    or self.ffy is not None
                    or self.twin.call == "FM"
                    or self.twin.confidence >= 0.55
                ):
                    if self.male_ff is None:
                        object.__setattr__(self, "male_ff", self.twin.male_ff)
                    if self.female_ff is None:
                        object.__setattr__(self, "female_ff", self.twin.female_ff)
        if self.gestation not in ("singleton", "twin"):
            raise ValueError(
                f"gestation must be 'singleton' or 'twin', got {self.gestation!r}"
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
class BranchQCStatus:
    """Wrapper around branch QC metrics dictionary providing status attribute."""

    data: dict[str, object]

    @property
    def status(self) -> str:
        """Branch quality verdict string (PASS, WARN, or FAIL)."""
        return str(self.data.get("verdict", ""))

    def __getitem__(self, key: str) -> object:
        return self.data[key]

    def get(self, key: str, default: object = None) -> object:
        return self.data.get(key, default)


@dataclass(frozen=True, slots=True)
class ReferenceQCOutputs:
    """Artifacts and verdict produced by reference QC."""

    json_path: Path | None
    overall_verdict: str
    worst_severity: int
    data: dict[str, object]

    @property
    def branches(self) -> dict[str, BranchQCStatus]:
        """Typed map of branch QC results by branch name."""
        raw = self.data.get("branches")
        if isinstance(raw, dict):
            return {
                str(k): (
                    BranchQCStatus(v) if isinstance(v, dict) else BranchQCStatus({})
                )
                for k, v in raw.items()
            }
        return {}


__all__ = [
    "ActiveFFSource",
    "BinResult",
    "BranchQCStatus",
    "CallType",
    "FetalFractionResult",
    "FetalSex",
    "GestationType",
    "PredictionOutputs",
    "PrenatalLoDMetrics",
    "ReferenceQCOutputs",
    "SCACall",
    "SCAResult",
    "ScanAlignmentResult",
    "ScanQcCounts",
    "SegmentResult",
    "TwinResult",
    "TwinSexPair",
]
