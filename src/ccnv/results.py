"""Typed prediction results shared by the numeric pipeline and writers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

CallType = Literal["gain", "loss"]


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


@dataclass(frozen=True, slots=True)
class ReferenceQCOutputs:
    """Artifacts and verdict produced by reference QC."""

    json_path: Path
    overall_verdict: str
    worst_severity: int
    data: dict[str, object]
