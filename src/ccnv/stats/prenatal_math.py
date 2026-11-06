"""NumPy, SciPy, and scikit-learn numerical kernels for prenatal analysis."""

from __future__ import annotations

import math
import warnings

import numpy as np
import scipy.optimize
import scipy.special
import scipy.stats
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor, Lasso, LinearRegression
from sklearn.mixture import GaussianMixture

from ccnv.stats.pca import pca_irlba_numpy

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

    Uses deterministic quantile initialization (25th and 75th percentiles) with
    scikit-learn GaussianMixture and brentq root-finding for the P = 0.5
    posterior decision threshold.
    """
    y = np.asarray(yfracs, dtype=np.float64).ravel()
    n = y.size
    if n == 0:
        return np.empty(0, dtype=np.bool_), 0.0
    if n == 1:
        return np.zeros(1, dtype=np.bool_), float(y[0])

    mu0 = float(np.percentile(y, 25.0))
    mu1 = float(np.percentile(y, 75.0))
    if mu0 > mu1:
        mu0, mu1 = mu1, mu0

    sample_var = float(np.var(y))
    if mu1 - mu0 < 1e-8 or sample_var < 1e-8:
        thresh = float(mu0)
        return y >= thresh, thresh

    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        tol=1e-6,
        reg_covar=1e-8,
        max_iter=100,
        means_init=np.array([[mu0], [mu1]], dtype=np.float64),
        weights_init=np.array([0.5, 0.5], dtype=np.float64),
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gmm.fit(y.reshape(-1, 1))

    means = gmm.means_.ravel()
    male_idx = int(np.argmax(means))
    female_idx = 1 - male_idx
    mu_low = float(means[female_idx])
    mu_high = float(means[male_idx])

    if mu_high - mu_low < 1e-8:
        thresh = mu_low
        return y >= thresh, thresh

    def _posterior_diff(val: float) -> float:
        probs = gmm.predict_proba(np.array([[val]], dtype=np.float64))
        return float(probs[0, male_idx] - 0.5)

    f_low = _posterior_diff(mu_low)
    f_high = _posterior_diff(mu_high)
    if f_low * f_high <= 0.0:
        thresh = float(scipy.optimize.brentq(_posterior_diff, mu_low, mu_high))
    else:
        thresh = 0.5 * (mu_low + mu_high)

    return y >= thresh, float(thresh)


def fit_lasso_fista(
    X: NDArray[np.floating] | np.ndarray,
    y: NDArray[np.floating] | np.ndarray,
    alpha: float,
    max_iter: int = 2000,
    tol: float = 1e-6,
) -> NDArray[np.float64]:
    """Solve non-negative Lasso regression via scikit-learn coordinate descent.

    Minimizes (1 / (2 * N)) * ||Xw - y||_2^2 + alpha * ||w||_1 subject to w >= 0.
    """
    x_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n_samples, n_features = x_arr.shape

    if (
        n_samples == 0
        or n_features == 0
        or np.allclose(x_arr, 0.0)
        or np.allclose(y_arr, 0.0)
    ):
        return np.zeros(n_features, dtype=np.float64)

    if alpha <= 0.0:
        w, _ = scipy.optimize.nnls(x_arr, y_arr, maxiter=max_iter)
        return np.asarray(w, dtype=np.float64)

    model = Lasso(
        alpha=float(alpha),
        fit_intercept=False,
        positive=True,
        max_iter=max_iter,
        tol=tol,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        model.fit(x_arr, y_arr)
    return np.asarray(model.coef_, dtype=np.float64)


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
    """Fit D -> hidden_dim -> 1 Sigmoid neural network via L-BFGS-B optimization.

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

    s1 = n_features * hidden_dim
    s2 = s1 + hidden_dim
    s3 = s2 + hidden_dim

    def _loss_and_grad(
        theta: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        w1 = theta[:s1].reshape(n_features, hidden_dim)
        b1 = theta[s1:s2]
        w2 = theta[s2:s3].reshape(hidden_dim, 1)
        b2 = theta[s3:]

        h = scipy.special.expit(x_arr @ w1 + b1)
        pred = scipy.special.expit(h @ w2 + b2)

        error = pred - y_arr
        loss = 0.5 * float(np.mean(error * error))

        d_out = (error * pred * (1.0 - pred)) / n_samples
        gw2 = h.T @ d_out
        gb2 = np.sum(d_out, axis=0)

        d_hidden = (d_out @ w2.T) * h * (1.0 - h)
        gw1 = x_arr.T @ d_hidden
        gb1 = np.sum(d_hidden, axis=0)

        grad = np.concatenate([gw1.ravel(), gb1, gw2.ravel(), gb2])
        return loss, grad

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

    limit1 = math.sqrt(6.0 / (n_features + hidden_dim))
    limit2 = math.sqrt(6.0 / (hidden_dim + 1))

    for init_idx in range(n_inits):
        rng = np.random.default_rng(seed + init_idx)
        w1_0 = rng.uniform(-limit1, limit1, size=(n_features, hidden_dim)).astype(
            np.float64
        )
        b1_0 = np.zeros(hidden_dim, dtype=np.float64)
        w2_0 = rng.uniform(-limit2, limit2, size=(hidden_dim, 1)).astype(np.float64)
        b2_0 = np.zeros(1, dtype=np.float64)
        theta0 = np.concatenate([w1_0.ravel(), b1_0, w2_0.ravel(), b2_0])

        res = scipy.optimize.minimize(
            _loss_and_grad,
            theta0,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": max_epochs, "gtol": 1e-4},
        )
        theta_opt = np.asarray(res.x, dtype=np.float64)
        w1 = theta_opt[:s1].reshape(n_features, hidden_dim).copy()
        b1 = theta_opt[s1:s2].copy()
        w2 = theta_opt[s2:s3].reshape(hidden_dim, 1).copy()
        b2 = theta_opt[s3:].copy()

        final_h = scipy.special.expit(x_arr @ w1 + b1)
        final_pred = scipy.special.expit(final_h @ w2 + b2)
        mse = float(np.mean((final_pred - y_arr) ** 2))

        if mse < best_mse or best_weights is None:
            best_mse = mse
            best_weights = (w1, b1, w2, b2)

    assert best_weights is not None
    return best_weights


def train_olm(
    X: NDArray[np.floating] | np.ndarray,
    y: NDArray[np.floating] | np.ndarray,
) -> tuple[NDArray[np.float64], float]:
    """Fit ordinary linear model via scikit-learn returning (weights, bias)."""
    x_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n_samples, n_features = x_arr.shape

    if n_samples < 2 or n_features == 0:
        return np.zeros(n_features, dtype=np.float64), 0.0

    reg = LinearRegression().fit(x_arr, y_arr)
    return np.asarray(reg.coef_, dtype=np.float64), float(reg.intercept_)


def ols_fit(
    x: NDArray[np.floating] | np.ndarray,
    y: NDArray[np.floating] | np.ndarray,
) -> tuple[float, float]:
    """Fit ordinary least squares returning (intercept, slope) with fallback."""
    xv = np.asarray(x, dtype=np.float64).ravel()
    yv = np.asarray(y, dtype=np.float64).ravel()
    if xv.size < 2 or float(np.var(xv)) < 1e-12:
        return 0.0, 1.0

    res = scipy.stats.linregress(xv, yv)
    return float(res.intercept), float(res.slope)


def rlm_huber(
    x: NDArray[np.floating] | np.ndarray,
    y: NDArray[np.floating] | np.ndarray,
    k: float = 1.345,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> tuple[float, float]:
    """Robust linear model using scikit-learn HuberRegressor."""
    xv = np.asarray(x, dtype=np.float64).ravel()
    yv = np.asarray(y, dtype=np.float64).ravel()
    if xv.size < 2 or float(np.var(xv)) < 1e-12 or float(np.var(yv)) < 1e-12:
        return ols_fit(xv, yv)

    model = HuberRegressor(
        epsilon=max(float(k), 1.0),
        max_iter=max(max_iter, 100),
        tol=tol,
        alpha=0.0,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(xv.reshape(-1, 1), yv)
    except ValueError:
        return ols_fit(xv, yv)
    intercept = float(model.intercept_)
    slope = float(model.coef_[0])
    if math.isfinite(intercept) and math.isfinite(slope):
        return intercept, slope
    return ols_fit(xv, yv)
