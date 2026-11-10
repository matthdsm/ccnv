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
    """Check if covariate array is non-empty, finite, and has variance above threshold."""
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


def _fit_spline_terms(
    values: NDArray[np.float64],
    degree: int,
    max_knots: int,
) -> NDArray[np.float64]:
    """Fit quantile B-spline terms without intercept."""
    k = min(max_knots, max(2, int(np.unique(values).size)))
    spline = SplineTransformer(
        n_knots=k,
        degree=degree,
        knots="quantile",
        extrapolation="linear",
        include_bias=False,
    )
    return np.asarray(spline.fit_transform(values.reshape(-1, 1)), dtype=np.float64)


def build_joint_covariate_basis(
    gc: NDArray[np.float64] | None,
    mappability: NDArray[np.float64] | None,
    *,
    mean_frag_len: NDArray[np.float64] | None = None,
    endpoint_ratio: NDArray[np.float64] | None = None,
) -> NDArray[np.float64] | None:
    """Build joint B-spline covariate basis capturing GC, mappability, fragmentomics, and interactions."""
    gc_arr = np.asarray(gc, dtype=np.float64) if gc is not None else None
    map_arr = (
        np.asarray(mappability, dtype=np.float64) if mappability is not None else None
    )
    len_arr = (
        np.asarray(mean_frag_len, dtype=np.float64)
        if mean_frag_len is not None
        else None
    )
    ep_arr = (
        np.asarray(endpoint_ratio, dtype=np.float64)
        if endpoint_ratio is not None
        else None
    )

    gc_ok = _is_valid_covariate(gc_arr)
    map_ok = _is_valid_covariate(map_arr)
    len_ok = _is_valid_covariate(len_arr)
    ep_ok = _is_valid_covariate(ep_arr)

    if not gc_ok and not map_ok and not len_ok and not ep_ok:
        return None

    if gc_ok and not map_ok:
        assert gc_arr is not None
        base_design = build_1d_basis(gc_arr, degree=3, n_knots=5)
    elif map_ok and not gc_ok:
        assert map_arr is not None
        base_design = build_1d_basis(map_arr, degree=2, n_knots=4)
    elif gc_ok and map_ok:
        assert gc_arr is not None and map_arr is not None
        if gc_arr.size != map_arr.size:
            return None

        n_rows = gc_arr.size
        b_gc = _fit_spline_terms(gc_arr, degree=3, max_knots=5)
        b_map = _fit_spline_terms(map_arr, degree=2, max_knots=4)

        z_gc = np.asarray(scipy.stats.zscore(gc_arr, ddof=0), dtype=np.float64)
        z_map = np.asarray(scipy.stats.zscore(map_arr, ddof=0), dtype=np.float64)
        interaction = (z_gc * z_map)[:, np.newaxis]
        ones = np.ones((n_rows, 1), dtype=np.float64)

        base_design = np.hstack([ones, b_gc, b_map, interaction], dtype=np.float64)
    else:
        first_valid = len_arr if len_ok else ep_arr
        assert first_valid is not None
        n_rows = first_valid.size
        base_design = np.ones((n_rows, 1), dtype=np.float64)

    if not len_ok and not ep_ok:
        return base_design

    assert base_design is not None
    extra_cols: list[NDArray[np.float64]] = [base_design]
    n_rows = base_design.shape[0]

    if len_ok:
        assert len_arr is not None
        if len_arr.size == n_rows:
            b_len = _fit_spline_terms(len_arr, degree=3, max_knots=4)
            extra_cols.append(b_len)
            if gc_ok:
                assert gc_arr is not None
                z_gc = np.asarray(scipy.stats.zscore(gc_arr, ddof=0), dtype=np.float64)
                z_len = np.asarray(
                    scipy.stats.zscore(len_arr, ddof=0), dtype=np.float64
                )
                extra_cols.append((z_gc * z_len)[:, np.newaxis])

    if ep_ok:
        assert ep_arr is not None
        if ep_arr.size == n_rows:
            b_ep = _fit_spline_terms(ep_arr, degree=3, max_knots=4)
            extra_cols.append(b_ep)

    return np.hstack(extra_cols, dtype=np.float64)


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
