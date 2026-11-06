"""Robust Huber IRLS polynomial regression and bias correction kernels."""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import scipy.stats
from numpy.polynomial.polynomial import polyvander
from numpy.typing import NDArray

_MIN_VARIANCE_STD: Final[float] = 1e-6
_IRLS_ITERATIONS: Final[int] = 3
_OUTLIER_SIGMA_CUTOFF: Final[float] = 2.5

__all__ = [
    "apply_1d_correction",
    "build_1d_basis",
    "fit_sample_trend",
]


def build_1d_basis(
    values: NDArray[np.float64],
    degree: int,
) -> NDArray[np.float64] | None:
    """Build a standardized, tail-clamped 1D polynomial design matrix."""

    if not np.all(np.isfinite(values)):
        return None
    std = float(np.std(values))
    if std <= _MIN_VARIANCE_STD:
        return None
    z = np.clip(
        np.asarray(scipy.stats.zscore(values, ddof=0), dtype=np.float64),
        -2.5,
        2.5,
    )
    n_rows = values.size
    max_cols = min(degree + 1, max(1, n_rows - 1))
    if max_cols <= 1:
        return None
    return np.asarray(polyvander(z, max_cols - 1), dtype=np.float64)


def fit_sample_trend(
    design: NDArray[np.float64],
    y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Fit a Huber-weighted polynomial coverage trend centered at unit median."""

    valid = np.isfinite(y) & (y > 0)
    n_cols = design.shape[1]
    if int(np.count_nonzero(valid)) <= n_cols:
        return np.ones(y.size, dtype=np.float64)

    med_y = float(np.median(y[valid]))
    if not math.isfinite(med_y) or med_y <= 0:
        return np.ones(y.size, dtype=np.float64)
    scaled_y = y / med_y

    weights = np.where(valid, 1.0, 0.0)
    beta: NDArray[np.float64] | None = None
    for _ in range(_IRLS_ITERATIONS):
        sqrt_w = np.sqrt(weights[valid])
        beta, *_ = np.linalg.lstsq(
            design[valid] * sqrt_w[:, np.newaxis],
            scaled_y[valid] * sqrt_w,
            rcond=None,
        )
        pred = design @ beta
        residuals = scaled_y[valid] - pred[valid]
        med_res = float(np.median(residuals))
        sigma = float(scipy.stats.median_abs_deviation(residuals, scale="normal"))
        delta = max(_OUTLIER_SIGMA_CUTOFF * sigma, 0.08)
        abs_dev = np.abs((scaled_y - pred) - med_res)
        huber_w = np.where(
            abs_dev <= delta,
            1.0,
            np.maximum(0.05, delta / np.maximum(abs_dev, 1e-12)),
        )
        weights = np.where(valid, huber_w, 0.0)

    if beta is None:
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
    """Divide 1D or 2D usable coverage array by fitted 1D polynomial trend."""

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


# Private aliases for backward compatibility with covariates.py
_build_1d_basis = build_1d_basis
_fit_sample_trend = fit_sample_trend
_apply_1d_correction = apply_1d_correction
