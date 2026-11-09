"""Genomic covariate masking, feature assembly, and multiplicative bias correction."""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import scipy.stats
from numpy.typing import NDArray

from ccnv.models import (
    DEFAULT_MIN_MAPPABILITY,
    CovariateError,
)
from ccnv.stats.regression import apply_1d_correction, build_joint_covariate_basis

__all__ = [
    "DEFAULT_MIN_MAPPABILITY",
    "correct_covariate_bias",
    "covariate_features",
    "mask_covariates",
    "validate_min_mappability",
]

_MIN_VARIANCE_STD: Final[float] = 1e-6


def validate_min_mappability(min_mappability: float) -> None:
    """Reject non-finite or out-of-range mappability thresholds."""
    if (
        isinstance(min_mappability, bool)
        or not math.isfinite(min_mappability)
        or min_mappability < 0.0
        or min_mappability > 1.0
    ):
        raise CovariateError("min_mappability must be a finite number in [0, 1]")


def mask_covariates(
    usable: NDArray[np.bool_],
    *,
    gc: NDArray[np.floating] | None = None,
    mappability: NDArray[np.floating] | None = None,
    n_mask: NDArray[np.bool_] | None = None,
    min_mappability: float = DEFAULT_MIN_MAPPABILITY,
) -> NDArray[np.bool_]:
    """Filter usable bins by mappability threshold, valid GC, and N-base mask."""
    validate_min_mappability(min_mappability)
    out = np.asarray(usable, dtype=np.bool_).copy()
    if mappability is not None:
        map_arr = np.asarray(mappability, dtype=np.float64)
        if np.any(np.isfinite(map_arr)):
            out &= np.isfinite(map_arr) & (map_arr >= min_mappability)
    if gc is not None:
        gc_arr = np.asarray(gc, dtype=np.float64)
        if np.any(np.isfinite(gc_arr)) or n_mask is not None:
            out &= np.isfinite(gc_arr) & (gc_arr >= 0.0) & (gc_arr <= 1.0)
    if n_mask is not None:
        out &= ~np.asarray(n_mask, dtype=np.bool_)
    return out


def correct_covariate_bias(
    normalized: NDArray[np.floating],
    usable: NDArray[np.bool_],
    gc: NDArray[np.floating] | None = None,
    mappability: NDArray[np.floating] | None = None,
) -> NDArray[np.float64]:
    """Remove per-sample multiplicative GC and mappability trends on usable bins."""
    data = np.asarray(normalized, dtype=np.float64)
    mask = np.asarray(usable, dtype=np.bool_)
    if data.ndim not in (1, 2):
        raise CovariateError("normalized coverage array must be 1D or 2D")
    n_usable = int(np.count_nonzero(mask))
    if n_usable < 4 or (gc is None and mappability is None):
        return data.copy()

    gc_sub = np.asarray(gc, dtype=np.float64)[mask] if gc is not None else None
    map_sub = (
        np.asarray(mappability, dtype=np.float64)[mask]
        if mappability is not None
        else None
    )
    joint_design = build_joint_covariate_basis(gc_sub, map_sub)
    if joint_design is None:
        return data.copy()

    out = data.copy()
    sub = out[mask]
    sub = apply_1d_correction(sub, joint_design)
    out[mask] = sub
    return out


def covariate_features(
    usable: NDArray[np.bool_],
    gc: NDArray[np.floating] | None = None,
    mappability: NDArray[np.floating] | None = None,
) -> NDArray[np.float64] | None:
    """Return standardized (GC, mappability) feature matrix for unmasked bins."""
    mask = np.asarray(usable, dtype=np.bool_)
    n_usable = int(np.count_nonzero(mask))
    if n_usable == 0:
        return None

    features: list[NDArray[np.float64]] = []
    for arr in (gc, mappability):
        if arr is None:
            continue
        sub = np.asarray(arr, dtype=np.float64)[mask]
        if not np.all(np.isfinite(sub)):
            continue
        std = float(np.std(sub))
        if std > _MIN_VARIANCE_STD:
            features.append(
                np.asarray(scipy.stats.zscore(sub, ddof=0), dtype=np.float64)
            )

    if not features:
        return None
    return np.column_stack(features)
