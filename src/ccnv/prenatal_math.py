"""Pure NumPy and SciPy numerical kernels for prenatal copy number analysis."""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.sparse.linalg
import scipy.special
from numpy.typing import NDArray

_RPROP_ETA_PLUS: Final[float] = 1.2
_RPROP_ETA_MINUS: Final[float] = 0.5
_RPROP_DELTA_0: Final[float] = 0.01
_RPROP_DELTA_MIN: Final[float] = 1e-6
_RPROP_DELTA_MAX: Final[float] = 50.0
_HUBER_SCALE_FACTOR: Final[float] = 1.4826

__all__ = [
    "fit_lasso_fista",
    "fit_rprop_mlp",
    "infer_sex_gmm_1d",
    "ols_fit",
    "pca_irlba_numpy",
    "rlm_huber",
    "train_olm",
]


def infer_sex_gmm_1d(
    yfracs: NDArray[np.floating] | np.ndarray,
    seed: int = 42,
) -> tuple[NDArray[np.bool_], float]:
    """Fit a 2-component 1D Gaussian Mixture Model on chrY read fractions.

    Uses deterministic quantile initialization (25th and 75th percentiles) and
    expectation-maximization updates with variance regularization, returning
    boolean is_male classifications and the posterior probability P = 0.5 decision
    threshold.
    """
    y = np.asarray(yfracs, dtype=np.float64).ravel()
    n = y.size
    if n == 0:
        return np.empty(0, dtype=np.bool_), 0.0
    if n == 1:
        return np.zeros(1, dtype=np.bool_), float(y[0])

    mu0 = float(np.percentile(y, 25.0))
    mu1 = float(np.percentile(y, 75.0))
    sample_var = float(np.var(y))
    var0 = max(sample_var * 0.5, 1e-8)
    var1 = max(sample_var * 0.5, 1e-8)
    pi0 = 0.5
    pi1 = 0.5

    if mu0 > mu1:
        mu0, mu1 = mu1, mu0

    if abs(mu1 - mu0) >= 1e-8 and sample_var >= 1e-8:
        for _ in range(100):
            prev_mu0, prev_mu1 = mu0, mu1

            log_p0 = (
                math.log(max(pi0, 1e-12))
                - 0.5 * math.log(2.0 * math.pi * var0)
                - 0.5 * ((y - mu0) ** 2) / var0
            )
            log_p1 = (
                math.log(max(pi1, 1e-12))
                - 0.5 * math.log(2.0 * math.pi * var1)
                - 0.5 * ((y - mu1) ** 2) / var1
            )

            max_log = np.maximum(log_p0, log_p1)
            exp0 = np.exp(log_p0 - max_log)
            exp1 = np.exp(log_p1 - max_log)
            denom = exp0 + exp1
            gamma0 = exp0 / denom
            gamma1 = 1.0 - gamma0

            n0 = float(np.sum(gamma0))
            n1 = float(np.sum(gamma1))
            if n0 < 1e-6 or n1 < 1e-6:
                break

            pi0 = float(np.clip(n0 / n, 1e-6, 1.0 - 1e-6))
            pi1 = 1.0 - pi0
            mu0 = float(np.sum(gamma0 * y) / n0)
            mu1 = float(np.sum(gamma1 * y) / n1)
            var0 = max(float(np.sum(gamma0 * ((y - mu0) ** 2)) / n0), 1e-8)
            var1 = max(float(np.sum(gamma1 * ((y - mu1) ** 2)) / n1), 1e-8)

            if abs(mu0 - prev_mu0) < 1e-6 and abs(mu1 - prev_mu1) < 1e-6:
                break

        if mu0 > mu1:
            mu0, mu1 = mu1, mu0
            var0, var1 = var1, var0
            pi0, pi1 = pi1, pi0

    if mu1 - mu0 < 1e-8:
        thresh = float(mu0)
        return y >= thresh, thresh

    def _log_posterior_diff(val: float) -> float:
        lp0 = (
            math.log(pi0)
            - 0.5 * math.log(2.0 * math.pi * var0)
            - 0.5 * ((val - mu0) ** 2) / var0
        )
        lp1 = (
            math.log(pi1)
            - 0.5 * math.log(2.0 * math.pi * var1)
            - 0.5 * ((val - mu1) ** 2) / var1
        )
        return lp1 - lp0

    f_low = _log_posterior_diff(mu0)
    f_high = _log_posterior_diff(mu1)

    if f_low < 0.0 < f_high or f_high < 0.0 < f_low:
        thresh = float(scipy.optimize.brentq(_log_posterior_diff, mu0, mu1))
    else:
        a = (1.0 / (2.0 * var0)) - (1.0 / (2.0 * var1))
        b = -(mu0 / var0) + (mu1 / var1)
        c = (
            (mu0**2 / (2.0 * var0))
            - (mu1**2 / (2.0 * var1))
            + math.log(pi0 / pi1)
            + 0.5 * math.log(var1 / var0)
        )
        if abs(a) < 1e-12:
            thresh = -c / b if abs(b) > 1e-12 else 0.5 * (mu0 + mu1)
        else:
            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                sqrt_disc = math.sqrt(disc)
                r1 = (-b - sqrt_disc) / (2.0 * a)
                r2 = (-b + sqrt_disc) / (2.0 * a)
                valid = [r for r in (r1, r2) if mu0 <= r <= mu1]
                if valid:
                    thresh = valid[0]
                else:
                    mid = 0.5 * (mu0 + mu1)
                    thresh = r1 if abs(r1 - mid) < abs(r2 - mid) else r2
            else:
                thresh = 0.5 * (mu0 + mu1)

    is_male = y >= thresh
    return is_male, float(thresh)


def fit_lasso_fista(
    X: NDArray[np.floating] | np.ndarray,
    y: NDArray[np.floating] | np.ndarray,
    alpha: float,
    max_iter: int = 2000,
    tol: float = 1e-6,
) -> NDArray[np.float64]:
    """Solve non-negative Lasso regression via FISTA with Nesterov momentum.

    Minimizes (1 / (2 * N)) * ||Xw - y||_2^2 + alpha * ||w||_1 subject to w >= 0.
    """
    x_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n_samples, n_features = x_arr.shape

    if n_samples == 0 or n_features == 0:
        return np.zeros(n_features, dtype=np.float64)

    s_max = float(scipy.linalg.svdvals(x_arr)[0]) if n_features > 0 else 0.0
    lipschitz = max((s_max * s_max) / n_samples, 1e-12)
    step = 1.0 / lipschitz
    thresh = step * alpha

    w = np.zeros(n_features, dtype=np.float64)
    v = np.zeros(n_features, dtype=np.float64)
    t = 1.0
    prev_obj = 0.5 * float(np.mean(y_arr * y_arr))

    for _ in range(max_iter):
        grad = (x_arr.T @ (x_arr @ v - y_arr)) / n_samples
        w_next = np.maximum(0.0, v - step * grad - thresh)

        diff_w = float(np.max(np.abs(w_next - w)))
        scale_w = max(float(np.max(np.abs(w_next))), 1e-12)
        rel_w_diff = diff_w / scale_w

        residuals = x_arr @ w_next - y_arr
        obj = 0.5 * float(np.mean(residuals * residuals)) + alpha * float(
            np.sum(w_next)
        )
        rel_obj_diff = abs(obj - prev_obj) / max(abs(prev_obj), 1e-12)

        if rel_w_diff < tol or rel_obj_diff < tol:
            w = w_next
            break

        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        v = w_next + ((t - 1.0) / t_next) * (w_next - w)
        w = w_next
        t = t_next
        prev_obj = obj

    return w


def pca_irlba_numpy(
    X: NDArray[np.floating] | np.ndarray,
    n_components: int,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute truncated unscaled PCA with sign canonicalization.

    Returns (scores, center, rotation) matching prcomp_irlba.
    """
    x_arr = np.asarray(X, dtype=np.float64)
    n_samples, n_features = x_arr.shape
    k = max(0, min(n_components, n_samples, n_features))

    center = np.nanmean(x_arr, axis=0)
    if not np.all(np.isfinite(center)):
        center = np.nan_to_num(center, nan=0.0)
    x_centered = np.nan_to_num(x_arr - center, nan=0.0)

    if k == 0:
        return (
            np.zeros((n_samples, n_components), dtype=np.float64),
            center,
            np.zeros((n_features, n_components), dtype=np.float64),
        )

    if k < min(n_samples, n_features) and k >= 1:
        u_k, s_k, vh_k = scipy.sparse.linalg.svds(x_centered, k=k, random_state=seed)
        order = np.argsort(s_k)[::-1]
        rotation = vh_k[order, :].T
    else:
        if n_samples <= n_features:
            gram = x_centered @ x_centered.T
            evals, evecs = scipy.linalg.eigh(gram)
            evals = np.maximum(evals[::-1][:k], 0.0)
            u_k = evecs[:, ::-1][:, :k]
            s_k = np.sqrt(evals)
            safe_s = np.where(s_k > 1e-12, s_k, 1.0)
            rotation = (x_centered.T @ u_k) / safe_s[np.newaxis, :]
        else:
            _, _, vh_full = scipy.linalg.svd(x_centered, full_matrices=False)
            rotation = vh_full[:k, :].T

    for j in range(k):
        col = rotation[:, j]
        max_idx = int(np.argmax(np.abs(col)))
        if col[max_idx] < 0.0:
            rotation[:, j] = -col

    scores = x_centered @ rotation

    if k < n_components:
        padded_scores = np.zeros((n_samples, n_components), dtype=np.float64)
        padded_scores[:, :k] = scores
        padded_rotation = np.zeros((n_features, n_components), dtype=np.float64)
        padded_rotation[:, :k] = rotation
        return padded_scores, center, padded_rotation

    return scores, center, rotation


def fit_rprop_mlp(
    X: NDArray[np.floating] | np.ndarray,
    y: NDArray[np.floating] | np.ndarray,
    hidden_dim: int = 2,
    n_inits: int = 5,
    max_epochs: int = 1500,
    seed: int = 42,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Fit D -> hidden_dim -> 1 Sigmoid neural network with full-batch Rprop-.

    Evaluates across n_inits deterministic initializations, returning the weights
    and biases (w1, b1, w2, b2) achieving the lowest training MSE.
    """
    x_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    n_samples, n_features = x_arr.shape

    if n_samples < 2 or n_features == 0:
        return (
            np.zeros((n_features, hidden_dim), dtype=np.float64),
            np.zeros(hidden_dim, dtype=np.float64),
            np.zeros((hidden_dim, 1), dtype=np.float64),
            np.zeros(1, dtype=np.float64),
        )

    best_mse = float("inf")
    best_weights: (
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ]
        | None
    ) = None

    for init_idx in range(n_inits):
        rng = np.random.default_rng(seed + init_idx)
        limit1 = math.sqrt(6.0 / (n_features + hidden_dim))
        w1 = rng.uniform(-limit1, limit1, size=(n_features, hidden_dim)).astype(
            np.float64
        )
        b1 = np.zeros(hidden_dim, dtype=np.float64)

        limit2 = math.sqrt(6.0 / (hidden_dim + 1))
        w2 = rng.uniform(-limit2, limit2, size=(hidden_dim, 1)).astype(np.float64)
        b2 = np.zeros(1, dtype=np.float64)

        delta_w1 = np.full_like(w1, _RPROP_DELTA_0)
        delta_b1 = np.full_like(b1, _RPROP_DELTA_0)
        delta_w2 = np.full_like(w2, _RPROP_DELTA_0)
        delta_b2 = np.full_like(b2, _RPROP_DELTA_0)

        prev_gw1 = np.zeros_like(w1)
        prev_gb1 = np.zeros_like(b1)
        prev_gw2 = np.zeros_like(w2)
        prev_gb2 = np.zeros_like(b2)

        for _ in range(max_epochs):
            h = scipy.special.expit(x_arr @ w1 + b1)
            pred = scipy.special.expit(h @ w2 + b2)

            error = pred - y_arr
            d_out = (error * pred * (1.0 - pred)) / n_samples
            gw2 = h.T @ d_out
            gb2 = np.sum(d_out, axis=0)

            d_hidden = (d_out @ w2.T) * h * (1.0 - h)
            gw1 = x_arr.T @ d_hidden
            gb1 = np.sum(d_hidden, axis=0)

            max_grad = max(
                float(np.max(np.abs(gw1))),
                float(np.max(np.abs(gb1))),
                float(np.max(np.abs(gw2))),
                float(np.max(np.abs(gb2))),
            )
            if max_grad < 1e-4:
                break

            for p, g, prev_g, delta in (
                (w1, gw1, prev_gw1, delta_w1),
                (b1, gb1, prev_gb1, delta_b1),
                (w2, gw2, prev_gw2, delta_w2),
                (b2, gb2, prev_gb2, delta_b2),
            ):
                sign_change = g * prev_g
                np.copyto(
                    delta,
                    np.where(
                        sign_change > 0.0,
                        np.minimum(delta * _RPROP_ETA_PLUS, _RPROP_DELTA_MAX),
                        np.where(
                            sign_change < 0.0,
                            np.maximum(delta * _RPROP_ETA_MINUS, _RPROP_DELTA_MIN),
                            delta,
                        ),
                    ),
                )
                p -= np.sign(g) * delta
                np.copyto(prev_g, np.where(sign_change < 0.0, 0.0, g))

        final_h = scipy.special.expit(x_arr @ w1 + b1)
        final_pred = scipy.special.expit(final_h @ w2 + b2)
        mse = float(np.mean((final_pred - y_arr) ** 2))

        if mse < best_mse or best_weights is None:
            best_mse = mse
            best_weights = (w1.copy(), b1.copy(), w2.copy(), b2.copy())

    assert best_weights is not None
    return best_weights


def train_olm(
    X: NDArray[np.floating] | np.ndarray,
    y: NDArray[np.floating] | np.ndarray,
) -> tuple[NDArray[np.float64], float]:
    """Fit ordinary linear model via least squares returning (weights, bias)."""
    x_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n_samples, n_features = x_arr.shape

    if n_samples < 2 or n_features == 0:
        return np.zeros(n_features, dtype=np.float64), 0.0

    design = np.column_stack([np.ones(n_samples, dtype=np.float64), x_arr])
    coeffs, _, _, _ = np.linalg.lstsq(design, y_arr, rcond=None)
    bias = float(coeffs[0])
    weights = coeffs[1:].astype(np.float64, copy=False)
    return weights, bias


def ols_fit(
    x: NDArray[np.floating] | np.ndarray,
    y: NDArray[np.floating] | np.ndarray,
) -> tuple[float, float]:
    """Fit ordinary least squares returning (intercept, slope) with fallback."""
    xv = np.asarray(x, dtype=np.float64).ravel()
    yv = np.asarray(y, dtype=np.float64).ravel()
    if xv.size < 2 or float(np.var(xv)) < 1e-12:
        return 0.0, 1.0

    design = np.column_stack([np.ones_like(xv), xv])
    coeffs, _, _, _ = np.linalg.lstsq(design, yv, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def rlm_huber(
    x: NDArray[np.floating] | np.ndarray,
    y: NDArray[np.floating] | np.ndarray,
    k: float = 1.345,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> tuple[float, float]:
    """Robust linear model using Huber M-estimation and residual MAD scale."""
    xv = np.asarray(x, dtype=np.float64).ravel()
    yv = np.asarray(y, dtype=np.float64).ravel()
    if xv.size < 2 or float(np.var(xv)) < 1e-12:
        return ols_fit(xv, yv)

    design = np.column_stack([np.ones_like(xv), xv])
    coeffs, _, _, _ = np.linalg.lstsq(design, yv, rcond=None)
    fitted = design @ coeffs
    resid = yv - fitted

    for _ in range(max_iter):
        med_res = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med_res)))
        scale = _HUBER_SCALE_FACTOR * mad
        if scale < 1e-12:
            break

        u = np.abs(resid / scale)
        weights = np.where(u <= k, 1.0, k / np.maximum(u, 1e-12))

        sqrt_w = np.sqrt(weights)
        new_coeffs, _, _, _ = np.linalg.lstsq(
            design * sqrt_w[:, np.newaxis],
            yv * sqrt_w,
            rcond=None,
        )
        diff = float(np.max(np.abs(new_coeffs - coeffs)))
        coeffs = new_coeffs
        fitted = design @ coeffs
        resid = yv - fitted
        if diff < tol:
            break

    return float(coeffs[0]), float(coeffs[1])
