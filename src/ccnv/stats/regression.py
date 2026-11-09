"""Robust Huber spline regression and bias correction kernels."""

from __future__ import annotations

import math
import warnings
from typing import Final

import numpy as np
import scipy.stats
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import SplineTransformer

_MIN_VARIANCE_STD: Final[float] = 1e-6
_HUBER_ALPHA: Final[float] = 1e-4
_HUBER_EPSILON: Final[float] = 1.35
_HUBER_MAX_ITER: Final[int] = 100

__all__ = [
    "apply_1d_correction",
    "build_1d_basis",
    "build_joint_covariate_basis",
    "fit_sample_trend",
]


def _is_valid_covariate(arr: NDArray[np.float64] | None) -> bool:
    if arr is None or arr.size < 2:
        return False
    if not np.all(np.isfinite(arr)):
        return False
    return float(np.std(arr)) > _MIN_VARIANCE_STD


def build_1d_basis(
    values: NDArray[np.float64],
    degree: int,
    n_knots: int = 2,
) -> NDArray[np.float64] | None:
    """Build a natural cubic or polynomial B-spline design matrix."""
    val_arr = np.asarray(values, dtype=np.float64)
    if not _is_valid_covariate(val_arr):
        return None
    unique_count = int(np.unique(val_arr).size)
    k = min(n_knots, max(2, unique_count))
    spline = SplineTransformer(
        n_knots=k,
        degree=degree,
        knots="quantile",
        extrapolation="linear",
        include_bias=True,
    )
    return np.asarray(spline.fit_transform(val_arr.reshape(-1, 1)), dtype=np.float64)


def build_joint_covariate_basis(
    gc: NDArray[np.float64] | None,
    mappability: NDArray[np.float64] | None,
) -> NDArray[np.float64] | None:
    """Build joint B-spline covariate basis capturing GC, mappability, and interaction."""
    gc_arr = np.asarray(gc, dtype=np.float64) if gc is not None else None
    map_arr = (
        np.asarray(mappability, dtype=np.float64) if mappability is not None else None
    )

    gc_ok = _is_valid_covariate(gc_arr)
    map_ok = _is_valid_covariate(map_arr)

    if not gc_ok and not map_ok:
        return None

    if gc_ok and not map_ok:
        assert gc_arr is not None
        return build_1d_basis(gc_arr, degree=3, n_knots=5)

    if map_ok and not gc_ok:
        assert map_arr is not None
        return build_1d_basis(map_arr, degree=2, n_knots=4)

    assert gc_arr is not None and map_arr is not None
    if gc_arr.size != map_arr.size:
        return None

    n_rows = gc_arr.size
    k_gc = min(5, max(2, int(np.unique(gc_arr).size)))
    spline_gc = SplineTransformer(
        n_knots=k_gc,
        degree=3,
        knots="quantile",
        extrapolation="linear",
        include_bias=False,
    )
    b_gc = np.asarray(spline_gc.fit_transform(gc_arr.reshape(-1, 1)), dtype=np.float64)

    k_map = min(4, max(2, int(np.unique(map_arr).size)))
    spline_map = SplineTransformer(
        n_knots=k_map,
        degree=2,
        knots="quantile",
        extrapolation="linear",
        include_bias=False,
    )
    b_map = np.asarray(
        spline_map.fit_transform(map_arr.reshape(-1, 1)), dtype=np.float64
    )

    z_gc = np.asarray(scipy.stats.zscore(gc_arr, ddof=0), dtype=np.float64)
    z_map = np.asarray(scipy.stats.zscore(map_arr, ddof=0), dtype=np.float64)
    interaction = (z_gc * z_map)[:, np.newaxis]
    ones = np.ones((n_rows, 1), dtype=np.float64)

    return np.hstack([ones, b_gc, b_map, interaction], dtype=np.float64)


def fit_sample_trend(
    design: NDArray[np.float64],
    y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Fit a Huber-weighted spline coverage trend centered at unit median."""
    valid = np.isfinite(y) & (y > 0)
    n_valid = int(np.count_nonzero(valid))
    n_cols = design.shape[1]
    if n_valid <= n_cols or n_valid == 0:
        return np.ones(y.size, dtype=np.float64)

    med_y = float(np.median(y[valid]))
    if not math.isfinite(med_y) or med_y <= 0:
        return np.ones(y.size, dtype=np.float64)
    scaled_y = y / med_y

    if n_valid < 4:
        beta = np.linalg.lstsq(design[valid], scaled_y[valid], rcond=None)[0]
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                reg = HuberRegressor(
                    fit_intercept=False,
                    alpha=_HUBER_ALPHA,
                    epsilon=_HUBER_EPSILON,
                    max_iter=_HUBER_MAX_ITER,
                )
                reg.fit(design[valid], scaled_y[valid])
                beta = reg.coef_
        except ValueError:
            beta = np.linalg.lstsq(design[valid], scaled_y[valid], rcond=None)[0]

    if not np.all(np.isfinite(beta)):
        return np.ones(y.size, dtype=np.float64)

    trend = design @ beta
    pos_trend = trend[valid & (trend > 0)]
    if pos_trend.size == 0:
        return np.ones(y.size, dtype=np.float64)
    median_trend = float(np.median(pos_trend))
    if not math.isfinite(median_trend) or median_trend <= 0:
        return np.ones(y.size, dtype=np.float64)
    return np.clip(trend / median_trend, 0.20, 5.0)


def apply_1d_correction(
    sub: NDArray[np.float64],
    design: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Divide 1D or 2D usable coverage array by fitted trend."""
    if design is None:
        return sub
    out = sub.copy()
    arr2d = out[:, np.newaxis] if out.ndim == 1 else out
    for s_idx in range(arr2d.shape[1]):
        col = arr2d[:, s_idx]
        orig_sum = float(np.sum(col))
        trend = fit_sample_trend(design, col)
        corrected = col / trend
        new_sum = float(np.sum(corrected))
        if orig_sum > 0 and new_sum > 0:
            corrected *= orig_sum / new_sum
        arr2d[:, s_idx] = corrected
    return out
