"""Exogenous log-link Poisson covariate regression, logit ensembles, and robust location estimators."""

from __future__ import annotations

import math
import sys
import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import Final

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV, PoissonRegressor, RidgeCV
from sklearn.preprocessing import SplineTransformer

from ccnv.models.exceptions import InvalidInputError

_MIN_VARIANCE_STD: Final[float] = 1e-6
_MIN_DISTINCT_VALUES: Final[int] = 5
_DISTINCT_CHUNK_SIZE: Final[int] = 65_536
_GC_QUANTILE_MIN_BIN_SIZE: Final[int] = 25_000
_GC_QUANTILE_NUM_BANDS: Final[int] = 8
_GC_BAND_MIDPOINTS: Final[tuple[float, ...]] = (
    0.15,
    0.325,
    0.375,
    0.425,
    0.475,
    0.525,
    0.60,
    0.825,
)
_POISSON_ALPHA: Final[float] = 1e-4
_POISSON_MAX_ITER: Final[int] = 100
_POISSON_PRECAP_Z: Final[float] = 6.0
_POISSON_WINSOR_Z: Final[float] = 3.5
_POISSON_WINSOR_PASSES: Final[int] = 2
_POISSON_EXP_CLIP: Final[float] = 20.0
_GEMM_ROW_CHUNK: Final[int] = 65_536
_GEMM_SAMPLE_BATCH: Final[int] = 128

__all__ = [
    "build_exogenous_covariate_basis",
    "fit_logit_linear_ensemble",
    "fit_poisson_covariate_trends",
    "half_sample_mode",
    "hsm_biweight_mode",
    "standardize_column",
]


def _count_distinct_up_to(
    arr: NDArray[np.floating],
    limit: int,
) -> int:
    """Count distinct finite values in arr up to limit via O(1) probe + chunk scan."""
    if limit <= 0:
        return 0
    flat = np.asarray(arr, dtype=np.float64).ravel()
    if flat.size == 0 or not np.all(np.isfinite(flat)):
        return 0

    step = max(1, flat.size // 64)
    seen: set[float] = set()
    for val in flat[::step]:
        seen.add(float(val))
        if len(seen) >= limit:
            return limit
    if step == 1:
        return len(seen)

    for start in range(0, flat.size, _DISTINCT_CHUNK_SIZE):
        chunk = flat[start : start + _DISTINCT_CHUNK_SIZE]
        unseen = np.ones(chunk.shape[0], dtype=np.bool_)
        for known in seen:
            unseen &= chunk != known
        while np.any(unseen):
            first_new = float(chunk[int(np.argmax(unseen))])
            seen.add(first_new)
            if len(seen) >= limit:
                return limit
            unseen &= chunk != first_new
    return len(seen)


def _has_min_distinct(
    arr: NDArray[np.floating],
    min_distinct: int = _MIN_DISTINCT_VALUES,
) -> bool:
    """Return True if arr is all-finite and contains at least min_distinct distinct values."""
    if min_distinct <= 0:
        return True
    return _count_distinct_up_to(arr, min_distinct) >= min_distinct


def standardize_column(arr: NDArray[np.float64]) -> NDArray[np.float64] | None:
    """Center and scale a 1D float64 vector when population std exceeds threshold."""
    if arr.size < 2 or not np.all(np.isfinite(arr)):
        return None
    centered = arr - float(np.mean(arr))
    std = float(np.sqrt(np.mean(np.square(centered))))
    if not math.isfinite(std) or std <= _MIN_VARIANCE_STD:
        return None
    return centered / std


def half_sample_mode(values: NDArray[np.floating]) -> float:
    """Compute Robertson-Cryer O(N log N) iterative Half-Sample Mode."""
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    s = np.sort(finite)
    while s.size > 3:
        n = int(s.size)
        half = (n + 1) // 2
        widths = s[half - 1 :] - s[: n - half + 1]
        min_w = float(np.min(widths))
        ties = np.flatnonzero(widths <= min_w + 1e-15)
        best = int(ties[ties.size // 2])
        s = s[best : best + half]
    if s.size == 1:
        return float(s[0])
    if s.size == 2:
        return 0.5 * float(s[0] + s[1])
    d0 = float(s[1] - s[0])
    d1 = float(s[2] - s[1])
    if d0 < d1:
        return 0.5 * float(s[0] + s[1])
    if d1 < d0:
        return 0.5 * float(s[1] + s[2])
    return float(s[1])


def hsm_biweight_mode(
    values: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
) -> float:
    """Refine Half-Sample Mode with 2-step precision-weighted Tukey biweight."""
    vals = np.asarray(values, dtype=np.float64)
    finite_mask = np.isfinite(vals)
    if weights is not None:
        w_arr = np.asarray(weights, dtype=np.float64)
        if w_arr.shape == finite_mask.shape:
            weights = w_arr[finite_mask]
    vals = vals[finite_mask]
    if vals.size == 0:
        return 0.0
    m = half_sample_mode(vals)
    if vals.size < 6:
        return m
    mad = 1.4826 * float(np.median(np.abs(vals - m)))
    if mad <= 1e-8:
        return m
    c = 3.5 * mad
    base_w = np.ones_like(vals) if weights is None else np.maximum(weights, 0.0)
    for _ in range(2):
        u = (vals - m) / c
        inlier = np.abs(u) < 1.0
        if not np.any(inlier):
            return m
        w = np.where(inlier, base_w * np.square(1.0 - np.square(u)), 0.0)
        w_sum = float(np.sum(w))
        if w_sum <= 0.0:
            return m
        m = float(np.sum(w * vals)) / w_sum
    return m


def fit_logit_linear_ensemble(
    X: NDArray[np.float64],
    u: NDArray[np.float64],
    *,
    ridge_alphas: NDArray[np.float64],
    l1_ratio: Sequence[float] = (0.1, 0.5),
    enet_alphas: NDArray[np.float64] | None = None,
    seed: int = 42,
    max_iter: int = 3000,
) -> tuple[NDArray[np.float64], float]:
    """Fit a 50/50 convex RidgeCV + ElasticNetCV linear model in logit space."""
    cv_folds = min(5, max(2, int(X.shape[0])))
    ridge = RidgeCV(alphas=ridge_alphas)
    enet = ElasticNetCV(
        l1_ratio=list(l1_ratio),
        alphas=100 if enet_alphas is None else enet_alphas,
        cv=cv_folds,
        random_state=seed,
        max_iter=max_iter,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", (ConvergenceWarning, UserWarning))
        ridge.fit(X, u)
        enet.fit(X, u)
    w_ens = 0.5 * (
        np.asarray(ridge.coef_, dtype=np.float64)
        + np.asarray(enet.coef_, dtype=np.float64)
    )
    b_ens = 0.5 * (float(ridge.intercept_) + float(enet.intercept_))
    return w_ens, b_ens


def _clamped_quantile_splines(
    values: NDArray[np.float64],
    *,
    degree: int,
    n_knots: int,
) -> NDArray[np.float64]:
    """Fit clamped quantile B-spline terms without bias column."""
    spline = SplineTransformer(
        n_knots=n_knots,
        degree=degree,
        knots="quantile",
        extrapolation="constant",
        include_bias=False,
    )
    return np.asarray(spline.fit_transform(values.reshape(-1, 1)), dtype=np.float64)


def _append_standardized_columns(
    dest: list[NDArray[np.float64]],
    matrix: NDArray[np.float64],
) -> None:
    """Append standardized columns of matrix whose population std exceeds threshold."""
    for col_idx in range(matrix.shape[1]):
        sc = standardize_column(matrix[:, col_idx])
        if sc is not None:
            dest.append(sc)


def _validate_optional_1d_track(
    track: NDArray[np.floating] | None,
    n_bins: int,
    name: str,
) -> NDArray[np.floating] | None:
    """Ensure optional 1D track matches full-genome bin count."""
    if track is None:
        return None
    arr = np.asarray(track)
    if arr.ndim != 1 or arr.shape[0] != n_bins:
        raise InvalidInputError(
            f"{name} must be a 1D array of length {n_bins} matching usable"
        )
    return arr


def _usable_spline_covariate(
    track: NDArray[np.floating] | None,
    mask: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Extract usable slice and z-score when at least 5 distinct finite values exist."""
    if track is None:
        return None
    sub = np.asarray(track[mask], dtype=np.float64)
    if not _has_min_distinct(sub, _MIN_DISTINCT_VALUES):
        return None
    z_col = standardize_column(sub)
    if z_col is None:
        return None
    return sub, z_col


def build_exogenous_covariate_basis(
    usable: NDArray[np.bool_],
    *,
    bin_size: int,
    gc: NDArray[np.floating] | None = None,
    gc_quantiles: NDArray[np.floating] | None = None,
    mappability: NDArray[np.floating] | None = None,
    ref_frag_len: NDArray[np.floating] | None = None,
    ref_ep_rate: NDArray[np.floating] | None = None,
) -> NDArray[np.float64]:
    """Build standardized exogenous covariate design matrix (U_u, K) on usable bins."""
    if not isinstance(bin_size, int) or isinstance(bin_size, bool) or bin_size <= 0:
        raise InvalidInputError("bin_size must be a positive integer")
    mask = np.asarray(usable, dtype=np.bool_)
    if mask.ndim != 1:
        raise InvalidInputError("usable must be a 1D boolean mask")

    n_bins = mask.shape[0]
    gc_arr = _validate_optional_1d_track(gc, n_bins, "gc")
    map_arr = _validate_optional_1d_track(mappability, n_bins, "mappability")
    len_arr = _validate_optional_1d_track(ref_frag_len, n_bins, "ref_frag_len")
    ep_arr = _validate_optional_1d_track(ref_ep_rate, n_bins, "ref_ep_rate")

    q_arr: NDArray[np.floating] | None = None
    if gc_quantiles is not None:
        q_arr = np.asarray(gc_quantiles)
        if q_arr.ndim != 2 or q_arr.shape != (n_bins, _GC_QUANTILE_NUM_BANDS):
            raise InvalidInputError(
                f"gc_quantiles must have shape ({n_bins}, {_GC_QUANTILE_NUM_BANDS})"
            )

    n_usable = int(np.count_nonzero(mask))
    if n_usable < _MIN_DISTINCT_VALUES:
        return np.empty((n_usable, 0), dtype=np.float64)

    cols: list[NDArray[np.float64]] = []
    z_gc: NDArray[np.float64] | None = None
    used_gc_quantiles = False

    if q_arr is not None and bin_size >= _GC_QUANTILE_MIN_BIN_SIZE:
        q_usable = np.asarray(q_arr[mask], dtype=np.float64)
        if np.all(np.isfinite(q_usable)):
            q_std_cols: list[NDArray[np.float64]] = []
            _append_standardized_columns(q_std_cols, q_usable[:, :7])
            if q_std_cols:
                cols.extend(q_std_cols)
                used_gc_quantiles = True
                if gc_arr is not None:
                    z_gc = standardize_column(
                        np.asarray(gc_arr[mask], dtype=np.float64)
                    )
                if z_gc is None:
                    midpoints = np.asarray(_GC_BAND_MIDPOINTS, dtype=np.float64)
                    z_gc = standardize_column(q_usable @ midpoints)

    if not used_gc_quantiles:
        gc_pair = _usable_spline_covariate(gc_arr, mask)
        if gc_pair is not None:
            gc_sub, z_gc = gc_pair
            _append_standardized_columns(
                cols, _clamped_quantile_splines(gc_sub, degree=3, n_knots=5)
            )

    map_pair = _usable_spline_covariate(map_arr, mask)
    if map_pair is not None:
        map_sub, z_map = map_pair
        _append_standardized_columns(
            cols, _clamped_quantile_splines(map_sub, degree=2, n_knots=4)
        )
        if z_gc is not None:
            inter_gm = standardize_column(z_gc * z_map)
            if inter_gm is not None:
                cols.append(inter_gm)

    len_pair = _usable_spline_covariate(len_arr, mask)
    if len_pair is not None:
        len_sub, z_len = len_pair
        _append_standardized_columns(
            cols, _clamped_quantile_splines(len_sub, degree=3, n_knots=4)
        )
        if z_gc is not None:
            inter_gl = standardize_column(z_gc * z_len)
            if inter_gl is not None:
                cols.append(inter_gl)

    ep_pair = _usable_spline_covariate(ep_arr, mask)
    if ep_pair is not None:
        ep_sub, _ = ep_pair
        _append_standardized_columns(
            cols, _clamped_quantile_splines(ep_sub, degree=3, n_knots=4)
        )

    if not cols:
        return np.empty((n_usable, 0), dtype=np.float64)
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.float64)


def _poisson_deviance_residuals(
    counts: NDArray[np.float64],
    mu: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute signed unit Poisson deviance residuals with exact 0 * log(0) == 0."""
    mu_safe = np.maximum(mu, 1e-12)
    c_safe = np.maximum(counts, 1e-12)
    half_dev = np.where(
        counts > 0.0,
        counts * np.log(c_safe / mu_safe) - (counts - mu_safe),
        mu_safe,
    )
    return np.sign(counts - mu_safe) * np.sqrt(2.0 * np.maximum(half_dev, 0.0))


def _winsorize_poisson_counts(
    counts: NDArray[np.float64],
    mu: NDArray[np.float64],
    z_threshold: float = _POISSON_WINSOR_Z,
    *,
    upper_only: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None, bool]:
    """Clip outlier counts and compute inverse-square deviance damping weights."""
    dev = _poisson_deviance_residuals(counts, mu)
    med = float(np.median(dev))
    mad = float(np.median(np.abs(dev - med)))
    tau = z_threshold * max(1.0, 1.4826 * mad)

    hi = dev > tau
    lo = (dev < -tau) if not upper_only else np.zeros_like(hi)
    outliers = hi | lo
    if not np.any(outliers):
        return counts, None, False

    clipped = counts.copy()
    if np.any(hi):
        m_hi = np.maximum(mu[hi], 1e-12)
        delta_hi = 0.5 * (tau * tau) / m_hi
        u_hi = 1.0 + tau / np.sqrt(m_hi) + (tau * tau) / (3.0 * m_hi)
        for _ in range(3):
            u_hi = u_hi - (u_hi * np.log(u_hi) - u_hi + 1.0 - delta_hi) / np.log(u_hi)
        clipped[hi] = m_hi * u_hi

    if np.any(lo):
        m_lo = mu[lo]
        delta_lo = 0.5 * (tau * tau) / m_lo
        u_lo = np.maximum(1e-6, np.square(1.0 - 0.5 * tau / np.sqrt(m_lo)))
        for _ in range(4):
            step = (u_lo * np.log(u_lo) - u_lo + 1.0 - delta_lo) / np.log(u_lo)
            u_lo = np.clip(u_lo - step, 1e-9, 1.0 - 1e-9)
        clipped[lo] = m_lo * u_lo

    abs_dev = np.abs(dev)
    w_scale = np.ones(counts.shape[0], dtype=np.float64)
    w_scale[outliers] = np.square(tau / np.maximum(abs_dev[outliers], tau))
    return clipped, w_scale, True


def _fit_single_sample_poisson(
    c_fit: NDArray[np.float64],
    sf_val: float,
    exp_fit: NDArray[np.float64],
    x_fit: NDArray[np.float64],
    *,
    suppress_warnings: bool = True,
) -> tuple[float, NDArray[np.float64]]:
    """Fit zero-inclusive offset-equivalent Poisson GLM with outlier Winsorization."""
    if not np.all(np.isfinite(c_fit)) or np.any(c_fit < 0.0):
        raise InvalidInputError("counts must be finite and non-negative on usable bins")

    n_features = x_fit.shape[1]
    zero_coef = np.zeros(n_features, dtype=np.float64)
    w_fit = sf_val * exp_fit
    w_sum = float(np.sum(w_fit))
    c_total = float(np.sum(c_fit))
    if c_total <= 0.0 or w_sum <= 0.0:
        return -_POISSON_EXP_CLIP, zero_coef

    y_raw = c_fit / w_fit
    q95 = float(np.quantile(y_raw, 0.95))
    if q95 > 0.0:
        keep = y_raw <= q95
        r0 = float(np.sum(c_fit[keep]) / np.sum(w_fit[keep]))
    else:
        r0 = c_total / w_sum
    c_prev, w_scale0, _ = _winsorize_poisson_counts(
        c_fit, w_fit * max(r0, 1e-12), _POISSON_PRECAP_Z, upper_only=True
    )
    w_eff = w_fit * w_scale0 if w_scale0 is not None else w_fit

    if n_features == 0:
        rate = max(
            float(np.sum(c_prev * w_scale0) / np.sum(w_eff))
            if w_scale0 is not None
            else float(np.sum(c_prev)) / w_sum,
            1e-12,
        )
        for _ in range(_POISSON_WINSOR_PASSES):
            c_next, w_scale, changed = _winsorize_poisson_counts(
                c_fit, w_fit * rate, _POISSON_WINSOR_Z
            )
            rate = max(
                float(np.sum(c_next * w_scale) / np.sum(w_fit * w_scale))
                if w_scale is not None
                else float(np.sum(c_next)) / w_sum,
                1e-12,
            )
            if not changed or np.allclose(c_next, c_prev, rtol=1e-4, atol=1e-4):
                break
            c_prev = c_next
        log_rate = float(np.clip(math.log(rate), -_POISSON_EXP_CLIP, _POISSON_EXP_CLIP))
        return log_rate, zero_coef

    warn_ctx = warnings.catch_warnings() if suppress_warnings else nullcontext()
    with warn_ctx:
        if suppress_warnings:
            warnings.simplefilter("ignore", ConvergenceWarning)
        reg = PoissonRegressor(
            alpha=_POISSON_ALPHA,
            max_iter=_POISSON_MAX_ITER,
            warm_start=True,
        )
        reg.fit(x_fit, c_prev / w_fit, sample_weight=w_eff)
        for _ in range(_POISSON_WINSOR_PASSES):
            lin = np.clip(
                float(reg.intercept_) + x_fit @ reg.coef_,
                -_POISSON_EXP_CLIP,
                _POISSON_EXP_CLIP,
            )
            mu_fit = w_fit * np.exp(lin)
            c_next, w_scale, changed = _winsorize_poisson_counts(
                c_fit, mu_fit, _POISSON_WINSOR_Z
            )
            if not changed or np.allclose(c_next, c_prev, rtol=1e-4, atol=1e-4):
                break
            c_prev = c_next
            w_refit = w_fit * w_scale if w_scale is not None else w_fit
            reg.fit(x_fit, c_next / w_fit, sample_weight=w_refit)

    coef = np.asarray(reg.coef_, dtype=np.float64)
    intercept = float(reg.intercept_)
    if not math.isfinite(intercept) or not np.all(np.isfinite(coef)):
        fallback_rate = max(c_total / w_sum, 1e-12)
        return (
            float(
                np.clip(math.log(fallback_rate), -_POISSON_EXP_CLIP, _POISSON_EXP_CLIP)
            ),
            zero_coef,
        )
    return (
        float(np.clip(intercept, -_POISSON_EXP_CLIP, _POISSON_EXP_CLIP)),
        coef,
    )


def fit_poisson_covariate_trends(
    counts: NDArray[np.floating] | NDArray[np.integer],
    size_factors: NDArray[np.floating],
    exposure: NDArray[np.floating],
    usable: NDArray[np.bool_],
    design_usable: NDArray[np.floating],
    *,
    fit_mask: NDArray[np.bool_] | None = None,
    max_fit_bins: int = 65_536,
    threads: int = 1,
) -> NDArray[np.float32]:
    """Fit strided zero-inclusive log-link Poisson trends and evaluate full genome via BLAS."""
    if (
        not isinstance(max_fit_bins, int)
        or isinstance(max_fit_bins, bool)
        or max_fit_bins < 1
    ):
        raise InvalidInputError("max_fit_bins must be a positive integer")
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise InvalidInputError("threads must be a positive integer")

    counts_arr = counts if isinstance(counts, np.ndarray) else np.asarray(counts)
    if counts_arr.ndim == 1:
        counts_2d = counts_arr[:, np.newaxis]
    elif counts_arr.ndim == 2:
        counts_2d = counts_arr
    else:
        raise InvalidInputError("counts must be a 1D or 2D array")
    n_bins, n_samples = counts_2d.shape

    mask = np.asarray(usable, dtype=np.bool_)
    if mask.ndim != 1 or mask.shape[0] != n_bins:
        raise InvalidInputError("usable must be a 1D boolean mask aligned with counts")
    n_usable = int(np.count_nonzero(mask))

    exp_arr = np.asarray(exposure, dtype=np.float64)
    if exp_arr.ndim != 1 or exp_arr.shape[0] != n_bins:
        raise InvalidInputError("exposure must be a 1D array aligned with counts")

    sf_arr = np.asarray(size_factors, dtype=np.float64)
    if sf_arr.ndim == 0:
        sf_arr = sf_arr.reshape(1)
    if sf_arr.ndim != 1 or sf_arr.shape[0] != n_samples:
        raise InvalidInputError(
            "size_factors must be a 1D array aligned with sample count"
        )
    if n_samples > 0 and (not np.all(np.isfinite(sf_arr)) or np.any(sf_arr <= 0.0)):
        raise InvalidInputError("size_factors must be finite and positive")

    design = np.ascontiguousarray(design_usable, dtype=np.float64)
    if design.ndim != 2 or design.shape[0] != n_usable:
        raise InvalidInputError(
            f"design_usable must have shape ({n_usable}, K) matching usable bins"
        )
    if design.size > 0 and not np.all(np.isfinite(design)):
        raise InvalidInputError("design_usable must contain only finite values")

    trends = np.zeros((n_bins, n_samples), dtype=np.float32)
    if n_usable == 0 or n_samples == 0:
        return trends

    usable_idx = np.flatnonzero(mask)
    exp_usable = exp_arr[usable_idx]
    if not np.all(np.isfinite(exp_usable)) or np.any(exp_usable <= 0.0):
        raise InvalidInputError("exposure must be finite and positive on usable bins")

    if fit_mask is not None:
        fm = np.asarray(fit_mask, dtype=np.bool_)
        if fm.ndim != 1 or fm.shape[0] != n_bins:
            raise InvalidInputError(
                "fit_mask must be a 1D boolean mask aligned with counts"
            )
        cand_rel = np.flatnonzero(fm[usable_idx])
        if cand_rel.size == 0:
            cand_rel = np.arange(n_usable, dtype=np.int64)
    else:
        cand_rel = np.arange(n_usable, dtype=np.int64)

    n_fit = min(int(cand_rel.size), max_fit_bins)
    fit_rel_idx = cand_rel[
        np.linspace(0, int(cand_rel.size) - 1, n_fit, dtype=np.int64)
    ]
    fit_abs_idx = usable_idx[fit_rel_idx]
    exp_fit = exp_usable[fit_rel_idx]
    x_fit = np.ascontiguousarray(design[fit_rel_idx], dtype=np.float64)
    n_features = x_fit.shape[1]

    workers = min(threads, n_samples)
    worker_suppress = workers > 1 and bool(
        getattr(sys.flags, "context_aware_warnings", 0)
    )

    def _fit_idx(s_idx: int) -> tuple[float, NDArray[np.float64]]:
        c_col = np.asarray(counts_2d[fit_abs_idx, s_idx], dtype=np.float64)
        return _fit_single_sample_poisson(
            c_col,
            float(sf_arr[s_idx]),
            exp_fit,
            x_fit,
            suppress_warnings=worker_suppress,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if workers <= 1:
            fitted = [_fit_idx(s_idx) for s_idx in range(n_samples)]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                fitted = list(pool.map(_fit_idx, range(n_samples)))

    intercepts = np.array([pair[0] for pair in fitted], dtype=np.float64)
    if n_features == 0:
        rates = np.exp(intercepts)
        for r0 in range(0, n_usable, _GEMM_ROW_CHUNK):
            r1 = min(n_usable, r0 + _GEMM_ROW_CHUNK)
            block = exp_usable[r0:r1, np.newaxis] * rates[np.newaxis, :]
            trends[usable_idx[r0:r1], :] = np.maximum(
                block.astype(np.float32), np.float32(1e-30)
            )
        return trends

    coefs = np.column_stack([pair[1] for pair in fitted])
    for r0 in range(0, n_usable, _GEMM_ROW_CHUNK):
        r1 = min(n_usable, r0 + _GEMM_ROW_CHUNK)
        x_block = design[r0:r1]
        exp_block = exp_usable[r0:r1, np.newaxis]
        u_block = usable_idx[r0:r1]
        for s0 in range(0, n_samples, _GEMM_SAMPLE_BATCH):
            s1 = min(n_samples, s0 + _GEMM_SAMPLE_BATCH)
            lin_block = intercepts[np.newaxis, s0:s1] + x_block @ coefs[:, s0:s1]
            np.clip(lin_block, -_POISSON_EXP_CLIP, _POISSON_EXP_CLIP, out=lin_block)
            np.exp(lin_block, out=lin_block)
            lin_block *= exp_block
            trends[u_block, s0:s1] = np.maximum(
                lin_block.astype(np.float32), np.float32(1e-30)
            )
    return trends
