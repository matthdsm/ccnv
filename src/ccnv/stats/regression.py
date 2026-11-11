"""Exogenous log-link Poisson covariate regression and legacy Huber spline kernels."""

from __future__ import annotations

import math
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Final

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor, PoissonRegressor
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
_HUBER_ALPHA: Final[float] = 1e-4
_HUBER_EPSILON: Final[float] = 1.35
_HUBER_MAX_ITER: Final[int] = 100

__all__ = [
    "apply_1d_correction",
    "build_1d_basis",
    "build_exogenous_covariate_basis",
    "build_joint_covariate_basis",
    "fit_poisson_covariate_trends",
    "fit_sample_trend",
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


def _standardize_column(arr: NDArray[np.float64]) -> NDArray[np.float64] | None:
    """Center and scale a 1D float64 vector when population std exceeds threshold."""
    if arr.size < 2 or not np.all(np.isfinite(arr)):
        return None
    centered = arr - float(np.mean(arr))
    std = float(np.sqrt(np.mean(np.square(centered))))
    if not math.isfinite(std) or std <= _MIN_VARIANCE_STD:
        return None
    return centered / std


def _prepare_covariate(arr: NDArray[np.floating] | None) -> NDArray[np.float64] | None:
    """Convert optional 1D covariate to float64 when finite and non-constant."""
    if arr is None:
        return None
    val_arr = np.asarray(arr, dtype=np.float64)
    if val_arr.ndim != 1 or val_arr.size < 2 or not np.all(np.isfinite(val_arr)):
        return None
    if float(np.std(val_arr)) <= _MIN_VARIANCE_STD:
        return None
    return val_arr


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
        sc = _standardize_column(matrix[:, col_idx])
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
    z_col = _standardize_column(sub)
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
                    z_gc = _standardize_column(
                        np.asarray(gc_arr[mask], dtype=np.float64)
                    )
                if z_gc is None:
                    midpoints = np.asarray(_GC_BAND_MIDPOINTS, dtype=np.float64)
                    z_gc = _standardize_column(q_usable @ midpoints)

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
            inter_gm = _standardize_column(z_gc * z_map)
            if inter_gm is not None:
                cols.append(inter_gm)

    len_pair = _usable_spline_covariate(len_arr, mask)
    if len_pair is not None:
        len_sub, z_len = len_pair
        _append_standardized_columns(
            cols, _clamped_quantile_splines(len_sub, degree=3, n_knots=4)
        )
        if z_gc is not None:
            inter_gl = _standardize_column(z_gc * z_len)
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

    with warnings.catch_warnings():
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

    n_fit = min(n_usable, max_fit_bins)
    fit_rel_idx = np.linspace(0, n_usable - 1, n_fit, dtype=np.int64)
    fit_abs_idx = usable_idx[fit_rel_idx]
    exp_fit = exp_usable[fit_rel_idx]
    x_fit = np.ascontiguousarray(design[fit_rel_idx], dtype=np.float64)
    n_features = x_fit.shape[1]

    def _fit_idx(s_idx: int) -> tuple[float, NDArray[np.float64]]:
        c_col = np.asarray(counts_2d[fit_abs_idx, s_idx], dtype=np.float64)
        return _fit_single_sample_poisson(c_col, float(sf_arr[s_idx]), exp_fit, x_fit)

    workers = min(threads, n_samples)
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


def build_1d_basis(
    values: NDArray[np.floating],
    degree: int,
    n_knots: int = 2,
) -> NDArray[np.float64] | None:
    """Build a natural cubic or polynomial B-spline design matrix."""
    val_arr = _prepare_covariate(values)
    if val_arr is None:
        return None
    k = min(n_knots, max(2, _count_distinct_up_to(val_arr, max(2, n_knots))))
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
    k = min(max_knots, max(2, _count_distinct_up_to(values, max(2, max_knots))))
    spline = SplineTransformer(
        n_knots=k,
        degree=degree,
        knots="quantile",
        extrapolation="linear",
        include_bias=False,
    )
    return np.asarray(spline.fit_transform(values.reshape(-1, 1)), dtype=np.float64)


def build_joint_covariate_basis(
    gc: NDArray[np.floating] | None,
    mappability: NDArray[np.floating] | None,
    *,
    ref_frag_len: NDArray[np.floating] | None = None,
    ref_ep_rate: NDArray[np.floating] | None = None,
    mean_frag_len: NDArray[np.floating] | None = None,
    endpoint_ratio: NDArray[np.floating] | None = None,
) -> NDArray[np.float64] | None:
    """Build legacy Huber B-spline covariate basis with single-pass standardization."""
    gc_arr = _prepare_covariate(gc)
    map_arr = _prepare_covariate(mappability)
    len_arr = _prepare_covariate(
        ref_frag_len if ref_frag_len is not None else mean_frag_len
    )
    ep_arr = _prepare_covariate(
        ref_ep_rate if ref_ep_rate is not None else endpoint_ratio
    )

    if gc_arr is None and map_arr is None and len_arr is None and ep_arr is None:
        return None

    z_gc = _standardize_column(gc_arr) if gc_arr is not None else None
    z_map = _standardize_column(map_arr) if map_arr is not None else None

    base_design: NDArray[np.float64] | None
    if gc_arr is not None and map_arr is None:
        base_design = build_1d_basis(gc_arr, degree=3, n_knots=5)
    elif map_arr is not None and gc_arr is None:
        base_design = build_1d_basis(map_arr, degree=2, n_knots=4)
    elif (
        gc_arr is not None
        and map_arr is not None
        and z_gc is not None
        and z_map is not None
    ):
        if gc_arr.size != map_arr.size:
            return None
        n_rows = gc_arr.size
        b_gc = _fit_spline_terms(gc_arr, degree=3, max_knots=5)
        b_map = _fit_spline_terms(map_arr, degree=2, max_knots=4)
        interaction = (z_gc * z_map)[:, np.newaxis]
        ones = np.ones((n_rows, 1), dtype=np.float64)
        base_design = np.hstack([ones, b_gc, b_map, interaction], dtype=np.float64)
    else:
        first_valid = len_arr if len_arr is not None else ep_arr
        if first_valid is None:
            return None
        base_design = np.ones((first_valid.size, 1), dtype=np.float64)

    if base_design is None or (len_arr is None and ep_arr is None):
        return base_design

    extra_cols: list[NDArray[np.float64]] = [base_design]
    n_rows = base_design.shape[0]

    if len_arr is not None and len_arr.size == n_rows:
        z_len = _standardize_column(len_arr)
        extra_cols.append(_fit_spline_terms(len_arr, degree=3, max_knots=4))
        if z_gc is not None and z_len is not None and z_gc.size == n_rows:
            extra_cols.append((z_gc * z_len)[:, np.newaxis])

    if ep_arr is not None and ep_arr.size == n_rows:
        extra_cols.append(_fit_spline_terms(ep_arr, degree=3, max_knots=4))

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
