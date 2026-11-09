"""Centered unscaled PCA and deterministic PCA fitting routines."""

from __future__ import annotations

import logging
import warnings

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.utils.extmath import svd_flip

from ccnv.models.exceptions import ReferenceError

logger = logging.getLogger(__name__)

__all__ = [
    "fit_pca",
    "optimal_svht_rank",
    "pca_irlba_numpy",
]


def optimal_svht_rank(
    data: NDArray[np.floating],
    max_components: int = 10,
) -> int:
    """Select optimal PCA rank using Gavish-Donoho (2014) SVHT."""
    data_arr = np.asarray(data, dtype=np.float64)
    if data_arr.ndim != 2 or data_arr.size == 0:
        return 1
    samples = data_arr.T
    s_dim, u_dim = samples.shape
    if s_dim < 1 or u_dim < 1:
        return 1
    mean = np.mean(samples, axis=0)
    centered = samples - mean
    s = np.linalg.svd(centered, compute_uv=False)
    if s.size == 0:
        return 1
    beta = min(s_dim, u_dim) / max(s_dim, u_dim)
    omega = 0.56 * (beta**3) - 0.95 * (beta**2) + 1.82 * beta + 1.43
    tau = omega * float(np.median(s))
    n_above = int(np.count_nonzero(s > tau))
    upper = min(max_components, s_dim, u_dim)
    return int(np.clip(n_above, 1, upper))


def fit_pca(
    masked_normalized: NDArray[np.floating],
    n_components: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Fit a deterministic PCA model on samples-by-bins and correct coverage."""
    data = np.asarray(masked_normalized, dtype=np.float64)
    if data.ndim != 2 or data.size == 0:
        raise ReferenceError("PCA requires a non-empty bins-by-samples matrix")
    samples = data.T
    if n_components is None:
        n_components = optimal_svht_rank(data, max_components=10)
    else:
        n_components = min(n_components, samples.shape[0], samples.shape[1])
    if n_components < 1:
        raise ReferenceError("PCA requires at least one sample and one unmasked bin")
    mean = np.mean(samples, axis=0)
    if np.allclose(samples, mean):
        return (
            np.ones(data.shape, dtype=np.float64),
            mean.astype(np.float64, copy=False),
            np.zeros((n_components, samples.shape[1]), dtype=np.float64),
        )
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(samples)
    if (
        hasattr(pca, "explained_variance_ratio_")
        and pca.explained_variance_ratio_ is not None
    ):
        finite_vars = np.nan_to_num(pca.explained_variance_ratio_ * 100.0)
        logger.debug(
            "PCA fitted %d components: explained variance [%s] (total=%.2f%%)",
            n_components,
            ", ".join(f"{v:.2f}%" for v in finite_vars),
            float(np.sum(finite_vars)),
        )
    reconstructed = pca.inverse_transform(pca.transform(samples))
    with np.errstate(divide="ignore", invalid="ignore"):
        corrected = samples / reconstructed
    corrected = np.nan_to_num(corrected, nan=1.0, posinf=1.0, neginf=1.0)
    return (
        corrected.T.astype(np.float64, copy=False),
        pca.mean_.astype(np.float64, copy=False),
        pca.components_.astype(np.float64, copy=False),
    )


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

    if n_samples == 0 or n_features == 0:
        return (
            np.zeros((n_samples, n_components), dtype=np.float64),
            np.zeros(n_features, dtype=np.float64),
            np.zeros((n_features, n_components), dtype=np.float64),
        )

    nan_mask = ~np.isfinite(x_arr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        center = np.nan_to_num(
            np.nanmean(np.where(nan_mask, np.nan, x_arr), axis=0),
            nan=0.0,
        )
    x_centered = np.where(nan_mask, 0.0, x_arr - center)

    if k == 0 or np.allclose(x_centered, 0.0):
        return (
            np.zeros((n_samples, n_components), dtype=np.float64),
            center,
            np.zeros((n_features, n_components), dtype=np.float64),
        )

    solver = "arpack" if k < min(n_samples, n_features) - 1 else "full"
    pca = PCA(n_components=k, svd_solver=solver, random_state=seed)
    pca.fit(x_centered)
    _, components = svd_flip(
        np.zeros((n_samples, k), dtype=np.float64),
        pca.components_,
        u_based_decision=False,
    )
    rotation = np.asarray(components.T, dtype=np.float64)
    scores = x_centered @ rotation

    if k < n_components:
        padded_scores = np.zeros((n_samples, n_components), dtype=np.float64)
        padded_scores[:, :k] = scores
        padded_rotation = np.zeros((n_features, n_components), dtype=np.float64)
        padded_rotation[:, :k] = rotation
        return padded_scores, center, padded_rotation

    return scores, center, rotation
