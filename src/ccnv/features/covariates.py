"""Genomic covariate masking, exogenous basis assembly, and Poisson bias regression."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ccnv.models import (
    DEFAULT_MIN_MAPPABILITY,
    CovariateError,
)
from ccnv.stats.regression import (
    _standardize_column,
    apply_1d_correction,
    build_exogenous_covariate_basis,
    build_joint_covariate_basis,
    fit_poisson_covariate_trends,
)

__all__ = [
    "DEFAULT_MIN_MAPPABILITY",
    "build_exogenous_covariate_basis",
    "correct_covariate_bias",
    "covariate_features",
    "fit_poisson_covariate_trends",
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


def correct_covariate_bias(
    normalized: NDArray[np.floating],
    usable: NDArray[np.bool_],
    gc: NDArray[np.floating] | None = None,
    mappability: NDArray[np.floating] | None = None,
    *,
    ref_frag_len: NDArray[np.floating] | None = None,
    ref_ep_rate: NDArray[np.floating] | None = None,
    mean_frag_len: NDArray[np.floating] | None = None,
    endpoint_ratio: NDArray[np.floating] | None = None,
) -> NDArray[np.float64]:
    """Compatibility wrapper removing exogenous multiplicative trends on usable bins."""
    del mean_frag_len, endpoint_ratio
    data = np.asarray(normalized, dtype=np.float64)
    mask = np.asarray(usable, dtype=np.bool_)
    if data.ndim not in (1, 2):
        raise CovariateError("normalized coverage array must be 1D or 2D")
    n_usable = int(np.count_nonzero(mask))
    if n_usable < 4 or (
        gc is None
        and mappability is None
        and ref_frag_len is None
        and ref_ep_rate is None
    ):
        return data.copy()

    gc_sub = np.asarray(gc, dtype=np.float64)[mask] if gc is not None else None
    map_sub = (
        np.asarray(mappability, dtype=np.float64)[mask]
        if mappability is not None
        else None
    )
    len_sub = (
        np.asarray(ref_frag_len, dtype=np.float64)[mask]
        if ref_frag_len is not None
        else None
    )
    ep_sub = (
        np.asarray(ref_ep_rate, dtype=np.float64)[mask]
        if ref_ep_rate is not None
        else None
    )
    joint_design = build_joint_covariate_basis(
        gc_sub,
        map_sub,
        ref_frag_len=len_sub,
        ref_ep_rate=ep_sub,
    )
    if joint_design is None:
        return data.copy()

    out = data.copy()
    out[mask] = apply_1d_correction(out[mask], joint_design)
    return out


def covariate_features(
    usable: NDArray[np.bool_],
    gc: NDArray[np.floating] | None = None,
    mappability: NDArray[np.floating] | None = None,
    ref_frag_len: NDArray[np.floating] | None = None,
    ref_ep_rate: NDArray[np.floating] | None = None,
    *,
    mean_frag_len: NDArray[np.floating] | None = None,
    endpoint_ratio: NDArray[np.floating] | None = None,
) -> NDArray[np.float64] | None:
    """Return standardized exogenous feature matrix for unmasked bins."""
    mask = np.asarray(usable, dtype=np.bool_)
    if int(np.count_nonzero(mask)) == 0:
        return None

    frag_track = ref_frag_len if ref_frag_len is not None else mean_frag_len
    ep_track = ref_ep_rate if ref_ep_rate is not None else endpoint_ratio

    features: list[NDArray[np.float64]] = []
    for arr in (gc, mappability, frag_track, ep_track):
        if arr is None:
            continue
        sub = np.asarray(arr, dtype=np.float64)[mask]
        z_col = _standardize_column(sub)
        if z_col is not None:
            features.append(z_col)

    if not features:
        return None
    return np.column_stack(features)
