"""Genomic covariate masking and standardized exogenous feature extraction."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ccnv.models import (
    DEFAULT_MIN_MAPPABILITY,
    CovariateError,
)
from ccnv.stats import standardize_column

__all__ = [
    "covariate_features",
    "mask_covariates",
    "validate_min_mappability",
]


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


def covariate_features(
    usable: NDArray[np.bool_],
    gc: NDArray[np.floating] | None = None,
    mappability: NDArray[np.floating] | None = None,
    ref_frag_len: NDArray[np.floating] | None = None,
    ref_ep_rate: NDArray[np.floating] | None = None,
) -> NDArray[np.float64] | None:
    """Return standardized exogenous feature matrix for unmasked bins."""
    mask = np.asarray(usable, dtype=np.bool_)
    if int(np.count_nonzero(mask)) == 0:
        return None

    features: list[NDArray[np.float64]] = []
    for arr in (gc, mappability, ref_frag_len, ref_ep_rate):
        if arr is None:
            continue
        sub = np.asarray(arr, dtype=np.float64)[mask]
        z_col = standardize_column(sub)
        if z_col is not None:
            features.append(z_col)

    if not features:
        return None
    return np.column_stack(features)
