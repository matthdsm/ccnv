"""Batched nearest-neighbor selection and null model computation."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.spatial import KDTree
from scipy.special import digamma, polygamma

logger = logging.getLogger(__name__)

_MAX_NULL_SAMPLES: Final[int] = 100

__all__ = [
    "build_signal_space_features",
    "compute_null_models",
    "distance_cutoff",
    "select_nearest_bins",
    "squeeze_var",
]


def build_signal_space_features(
    pca_loadings: NDArray[np.floating],
    singular_values: NDArray[np.floating] | None = None,
    covariate_mat: NDArray[np.floating] | None = None,
) -> NDArray[np.float64]:
    """Assemble low-dimensional signal-space feature matrix for nearest-bin search."""
    loadings = np.asarray(pca_loadings, dtype=np.float64)
    if loadings.ndim == 2:
        feats = loadings.T
    elif loadings.ndim == 1:
        feats = loadings[:, np.newaxis]
    else:
        feats = np.empty((0, 0), dtype=np.float64)

    if singular_values is not None and feats.size > 0:
        s = np.asarray(singular_values, dtype=np.float64)
        feats = feats * s

    if covariate_mat is not None:
        cov = np.asarray(covariate_mat, dtype=np.float64)
        if cov.ndim == 1:
            cov = cov[:, np.newaxis]
        if cov.size > 0:
            if feats.size > 0 and not np.allclose(feats, 0.0):
                feats = np.column_stack([feats, cov])
            else:
                feats = cov

    return np.asarray(feats, dtype=np.float64)


def select_nearest_bins(
    profiles: NDArray[np.floating],
    contig_ids: NDArray[np.integer],
    k: int,
    max_batch_memory_mb: int = 256,
    threads: int = 1,
) -> tuple[list[list[int]], list[list[float]]]:
    """Pick k nearest off-contig bins via per-contig leave-one-out KDTree query."""
    data = np.asarray(profiles, dtype=np.float64)
    contig_ids = np.asarray(contig_ids)
    n_bins = data.shape[0]
    if k < 1:
        raise ValueError("nearest_bins must be a positive integer")
    if n_bins == 0:
        return [], []

    out_indices: list[list[int]] = [[] for _ in range(n_bins)]
    out_distances: list[list[float]] = [[] for _ in range(n_bins)]

    unique_cids = np.unique(contig_ids)
    for cid in unique_cids:
        on_contig = np.flatnonzero(contig_ids == cid)
        off_contig = np.flatnonzero(contig_ids != cid)
        if len(off_contig) == 0:
            continue

        take = min(k, len(off_contig))
        tree = KDTree(data[off_contig])
        dists, local_idx = tree.query(data[on_contig], k=take, p=2, workers=threads)
        dists = np.asarray(dists, dtype=np.float64).reshape(len(on_contig), take)
        local_idx = np.asarray(local_idx, dtype=np.int64).reshape(len(on_contig), take)
        global_idx = off_contig[local_idx]
        sq_dists = dists**2

        for r_idx, b_idx in enumerate(on_contig):
            out_indices[b_idx] = [int(x) for x in global_idx[r_idx]]
            out_distances[b_idx] = [float(x) for x in sq_dists[r_idx]]

    return out_indices, out_distances


def squeeze_var(
    var: NDArray[np.floating],
    df: float,
    winsor_tail_p: tuple[float, float] = (0.05, 0.10),
) -> tuple[NDArray[np.float64], float, float]:
    """Empirical Bayes variance moderation matching Smyth (2004) / limma::squeezeVar."""
    var_arr = np.asarray(var, dtype=np.float64)
    valid = np.isfinite(var_arr) & (var_arr > 1e-16)
    if np.count_nonzero(valid) < 2:
        return var_arr.copy(), 1e6, 0.0

    v = var_arr[valid]
    z = np.log(v) - float(digamma(df / 2.0)) + np.log(df / 2.0)

    low_p, high_p = winsor_tail_p
    if low_p > 0.0 or high_p > 0.0:
        q_low = float(np.quantile(z, low_p)) if low_p > 0.0 else -np.inf
        q_high = float(np.quantile(z, 1.0 - high_p)) if high_p > 0.0 else np.inf
        z_winsor = np.clip(z, q_low, q_high)
    else:
        z_winsor = z

    mean_z = float(np.mean(z_winsor))
    var_z = float(np.var(z_winsor, ddof=1))
    excess_var = var_z - float(polygamma(1, df / 2.0))

    if excess_var <= 1e-7:
        d0_hat = 1e6
        s0_sq_hat = float(np.exp(mean_z + digamma(d0_hat / 2.0) - np.log(d0_hat / 2.0)))
    else:

        def func(d0: float) -> float:
            """Root objective for prior degrees of freedom d0."""
            return float(polygamma(1, d0 / 2.0)) - excess_var

        d0_hat = float(brentq(func, 1e-8, 1e8))
        s0_sq_hat = float(np.exp(mean_z + digamma(d0_hat / 2.0) - np.log(d0_hat / 2.0)))

    var_post = var_arr.copy()
    var_post[valid] = (d0_hat * s0_sq_hat + df * v) / (d0_hat + df)
    return var_post, d0_hat, s0_sq_hat


def compute_null_models(
    corrected: NDArray[np.float64],
    nearest_indices: Sequence[Sequence[int]],
    threads: int = 1,
    n_pca_components: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute per-bin reference medians and DOF-corrected empirical Bayes null statistics."""
    n_bins, n_samples = corrected.shape
    n_null = min(n_samples, _MAX_NULL_SAMPLES)
    if n_null == 0 or n_bins == 0:
        return np.zeros(n_bins, dtype=np.float64), np.zeros(n_bins, dtype=np.float64)

    max_k = max((len(idx) for idx in nearest_indices), default=0)
    if max_k == 0:
        return np.zeros(n_bins, dtype=np.float64), np.zeros(n_bins, dtype=np.float64)

    pad_idx = np.full((n_bins, max_k), -1, dtype=np.int64)
    for i, idx in enumerate(nearest_indices):
        if idx:
            pad_idx[i, : len(idx)] = idx

    d = max(1, n_null - 1 - n_pca_components)
    dof_scale = (n_null - 1) / d
    sub = corrected[:, :n_null]
    sub_with_nan = np.vstack([sub, np.full((1, n_null), np.nan)])

    out_mean = np.zeros(n_bins, dtype=np.float64)
    raw_vars = np.zeros(n_bins, dtype=np.float64)

    bytes_per_row = max_k * n_null * 8
    chunk_size = min(4096, max(1, (256 * 1024 * 1024) // max(bytes_per_row, 1)))

    for c_start in range(0, n_bins, chunk_size):
        c_stop = min(c_start + chunk_size, n_bins)
        chunk_pad = pad_idx[c_start:c_stop]
        has_neighbors = (chunk_pad != -1).any(axis=1)

        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            ref = np.nanmedian(sub_with_nan[chunk_pad], axis=1)
            values = np.log2(sub[c_start:c_stop] / ref)
            values_clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            out_mean[c_start:c_stop] = np.where(
                has_neighbors, np.mean(values_clean, axis=1), 0.0
            )
            if n_null > 1:
                chunk_vars = np.var(values_clean, axis=1, ddof=1) * dof_scale
            else:
                chunk_vars = np.zeros(c_stop - c_start, dtype=np.float64)
            raw_vars[c_start:c_stop] = np.where(has_neighbors, chunk_vars, 0.0)

    valid_bins = (pad_idx != -1).any(axis=1) & (raw_vars > 1e-16)
    if n_bins >= 10 and np.count_nonzero(valid_bins) >= 2:
        mod_vars, _, _ = squeeze_var(raw_vars, df=float(d))
        out_std = np.sqrt(np.maximum(mod_vars, 0.0))
        out_std[~valid_bins] = 0.0
    else:
        out_std = np.sqrt(np.maximum(raw_vars, 0.0))
        out_std[~valid_bins] = 0.0

    out_std[~np.isfinite(out_std) | (out_std <= 1e-8)] = 0.0
    return out_mean, out_std


def distance_cutoff(
    distances: (
        tuple[NDArray[np.float64], ...]
        | NDArray[np.floating]
        | Sequence[Sequence[float]]
    ),
    repeats: int = 3,
) -> float:
    """Return an iterative three-sigma cutoff for nearest-bin distances."""
    if isinstance(distances, np.ndarray):
        values = distances.ravel()
    elif (
        isinstance(distances, tuple)
        and distances
        and isinstance(distances[0], np.ndarray)
    ):
        values = np.concatenate(distances)
    else:
        values = np.asarray([d for sub in distances for d in sub], dtype=np.float64)

    values = values[np.isfinite(values) & (values >= 0)]
    if values.size == 0:
        return float("inf")
    cutoff = float("inf")
    for _ in range(repeats):
        selected = values[values < cutoff]
        if selected.size == 0:
            break
        cutoff = float(np.mean(selected) + 3.0 * np.std(selected))
    return cutoff
