"""Variance-whitened outlier-clipped PCA and deterministic PCA fitting routines."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Final

import numpy as np
import scipy.linalg
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd, svd_flip

from ccnv.models.exceptions import ReferenceError

logger = logging.getLogger(__name__)

_MAX_PCA_CHUNK_ROWS: Final[int] = 32_768
_TARGET_CHUNK_BYTES: Final[int] = 64 * 1024 * 1024
_WINSOR_Z_CLIP: Final[float] = 3.0
_LOG_WAVE_CLIP: Final[float] = 4.0

__all__ = [
    "WhitenedPCAResult",
    "fit_pca",
    "fit_whitened_pca",
    "optimal_svht_rank",
    "pca_irlba_numpy",
    "reconstruct_whitened_pca_expected",
]


@dataclass(frozen=True, slots=True)
class WhitenedPCAResult:
    """Fitted variance-whitened outlier-clipped PCA model on usable bins."""

    mean_rate: NDArray[np.float32]
    loadings: NDArray[np.float32]
    singular_values: NDArray[np.float64]
    mp_sigma: float


def _pca_chunk_rows(n_samples: int) -> int:
    """Compute bounded row chunk size for streaming memmap/dense matrices."""
    denom = max(n_samples, 1) * 8
    return min(_MAX_PCA_CHUNK_ROWS, max(256, _TARGET_CHUNK_BYTES // denom))


def _fit_chunk_mean_and_whitened(
    c_chunk: NDArray[np.floating],
    e_chunk: NDArray[np.floating],
) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
    """Estimate outlier-clipped bin capture rate mu_i and centered whitened Z."""
    c = np.asarray(c_chunk, dtype=np.float64)
    e = np.asarray(e_chunk, dtype=np.float64)
    if (
        not np.all(np.isfinite(c))
        or np.any(c < 0.0)
        or not np.all(np.isfinite(e))
        or np.any(e <= 0.0)
    ):
        raise ReferenceError(
            "counts must be finite non-negative and expected must be finite positive on usable bins"
        )

    n_samples = c.shape[1]
    r_init = (c + 0.5) / (e + 0.5)
    med_r = np.median(r_init, axis=1, keepdims=True)
    c_cap = np.minimum(c, (e + 0.5) * np.maximum(2.0 * med_r, 2.0))
    e_sum = np.maximum(np.sum(e, axis=1), 1e-12)
    mu = np.clip(np.sum(c_cap, axis=1) / e_sum, 1e-4, 1e4)

    min_inliers = max(1, (n_samples + 1) // 2)
    for _ in range(2):
        e0 = e * mu[:, np.newaxis]
        w = np.sqrt(e0 + 0.5)
        z_raw = np.log((c + 0.5) / (e0 + 0.5)) * w
        inlier = np.abs(z_raw) <= _WINSOR_Z_CLIP
        in_cnt = np.sum(inlier, axis=1)
        c_in = np.sum(np.where(inlier, c, 0.0), axis=1)
        e_in = np.sum(np.where(inlier, e, 0.0), axis=1)
        z_clip = np.clip(z_raw, -_WINSOR_Z_CLIP, _WINSOR_Z_CLIP)
        c_win = np.maximum((e0 + 0.5) * np.exp(z_clip / w) - 0.5, 0.0)
        mu_win = np.sum(c_win, axis=1) / e_sum
        use_in = (in_cnt >= min_inliers) & (e_in > 1e-12)
        mu_cand = np.where(use_in, c_in / np.maximum(e_in, 1e-12), mu_win)
        mu = np.clip(np.where(mu_cand > 0.0, mu_cand, mu_win), 1e-4, 1e4)

    mu_f32 = mu.astype(np.float32)
    zc = _whitened_centered_from_mu(c, e, mu_f32)
    return mu_f32, zc


def _whitened_centered_from_mu(
    c: NDArray[np.float64],
    e: NDArray[np.float64],
    mu_f32: NDArray[np.float32],
) -> NDArray[np.float64]:
    """Compute 2-pass winsorized sample-centered whitened residuals from fixed mu_i."""
    if c.shape[1] <= 1:
        return np.zeros(c.shape, dtype=np.float64)
    e0 = e * mu_f32.astype(np.float64)[:, np.newaxis]
    w = np.sqrt(e0 + 0.5)
    z1 = np.clip(np.log((c + 0.5) / (e0 + 0.5)) * w, -_WINSOR_Z_CLIP, _WINSOR_Z_CLIP)
    zc = z1 - np.mean(z1, axis=1, keepdims=True)
    zc = np.clip(zc, -_WINSOR_Z_CLIP, _WINSOR_Z_CLIP)
    zc -= np.mean(zc, axis=1, keepdims=True)
    return zc


def _svht_noise_cutoff(
    singular_values: NDArray[np.float64],
    s_eff: int,
    u_count: int,
) -> float:
    """Compute Gavish-Donoho (2014) SVHT threshold excluding structural zero mode."""
    if singular_values.size == 0 or s_eff < 1 or u_count < 1:
        return 0.0
    beta = min(s_eff, u_count) / max(s_eff, u_count)
    omega = 0.56 * (beta**3) - 0.95 * (beta**2) + 1.82 * beta + 1.43
    return float(omega * float(np.median(singular_values)))


def fit_whitened_pca(
    counts: NDArray[np.float32],
    expected: NDArray[np.float32],
    usable: NDArray[np.bool_],
    *,
    max_components: int | None = None,
) -> WhitenedPCAResult:
    """Fit chunked variance-whitened outlier-clipped PCA with SVHT rank selection."""
    counts_arr = counts if isinstance(counts, np.ndarray) else np.asarray(counts)
    exp_arr = expected if isinstance(expected, np.ndarray) else np.asarray(expected)
    if (
        counts_arr.ndim != 2
        or exp_arr.ndim != 2
        or counts_arr.shape != exp_arr.shape
        or 0 in counts_arr.shape
    ):
        raise ReferenceError(
            "counts and expected must be non-empty 2D arrays with identical shape"
        )
    if max_components is not None and (
        not isinstance(max_components, int)
        or isinstance(max_components, bool)
        or max_components < 0
    ):
        raise ReferenceError("max_components must be None or a non-negative integer")

    n_bins, n_samples = counts_arr.shape
    mask = np.asarray(usable, dtype=np.bool_)
    if mask.ndim != 1 or mask.shape[0] != n_bins:
        raise ReferenceError("usable must be a 1D boolean mask aligned with counts")
    u_idx = np.flatnonzero(mask)
    u_count = int(u_idx.size)
    if u_count == 0:
        raise ReferenceError("fit_whitened_pca requires at least one usable bin")

    mean_rate = np.zeros(n_bins, dtype=np.float32)
    mu_u = np.empty(u_count, dtype=np.float32)
    chunk_rows = _pca_chunk_rows(n_samples)
    s_eff = max(0, n_samples - 1)
    m_nontrivial = min(u_count, s_eff)

    if n_samples <= u_count:
        gram = np.zeros((n_samples, n_samples), dtype=np.float64)
        for c0 in range(0, u_count, chunk_rows):
            c1 = min(u_count, c0 + chunk_rows)
            rows = u_idx[c0:c1]
            mu_chunk, zc = _fit_chunk_mean_and_whitened(counts_arr[rows], exp_arr[rows])
            mu_u[c0:c1] = mu_chunk
            if m_nontrivial > 0:
                gram += zc.T @ zc
        mean_rate[u_idx] = mu_u
        if m_nontrivial == 0:
            logger.debug(
                "PCA fitted 0 components: explained variance [] (mp_sigma=0.0000)"
            )
            return WhitenedPCAResult(
                mean_rate=mean_rate,
                loadings=np.empty((0, u_count), dtype=np.float32),
                singular_values=np.empty(0, dtype=np.float64),
                mp_sigma=0.0,
            )
        evals, evecs = scipy.linalg.eigh(0.5 * (gram + gram.T))
        s_all = np.sqrt(np.maximum(evals[::-1][:m_nontrivial], 0.0))
        u_samp_all = evecs[:, ::-1][:, :m_nontrivial]
        u_bins_all: NDArray[np.float64] | None = None
    else:
        zc_full = np.empty((u_count, n_samples), dtype=np.float64)
        for c0 in range(0, u_count, chunk_rows):
            c1 = min(u_count, c0 + chunk_rows)
            rows = u_idx[c0:c1]
            mu_chunk, zc = _fit_chunk_mean_and_whitened(counts_arr[rows], exp_arr[rows])
            mu_u[c0:c1] = mu_chunk
            zc_full[c0:c1] = zc
        mean_rate[u_idx] = mu_u
        u_bins_raw, s_raw, _ = randomized_svd(
            zc_full, n_components=m_nontrivial, random_state=0
        )
        s_all = np.asarray(s_raw, dtype=np.float64)
        u_bins_all = np.asarray(u_bins_raw, dtype=np.float64)
        u_samp_all = np.empty((0, 0), dtype=np.float64)

    mp_sigma = _svht_noise_cutoff(s_all, s_eff, u_count)
    if n_samples <= 2 or u_count <= 1:
        rank = 0
    else:
        max_rank = (
            min(60, max(5, n_samples // 5), max(0, n_samples - 2), max(0, u_count - 1))
            if max_components is None
            else min(max_components, max(0, n_samples - 1), u_count)
        )
        rank = min(int(np.count_nonzero(s_all > mp_sigma)), max_rank)

    if rank == 0:
        logger.debug(
            "PCA fitted 0 components: explained variance [] (mp_sigma=%.4f)",
            mp_sigma,
        )
        return WhitenedPCAResult(
            mean_rate=mean_rate,
            loadings=np.empty((0, u_count), dtype=np.float32),
            singular_values=np.empty(0, dtype=np.float64),
            mp_sigma=mp_sigma,
        )

    if u_bins_all is not None:
        v_bins = u_bins_all[:, :rank].copy()
    else:
        u_top = u_samp_all[:, :rank]
        v_bins = np.empty((u_count, rank), dtype=np.float64)
        for c0 in range(0, u_count, chunk_rows):
            c1 = min(u_count, c0 + chunk_rows)
            rows = u_idx[c0:c1]
            zc = _whitened_centered_from_mu(
                np.asarray(counts_arr[rows], dtype=np.float64),
                np.asarray(exp_arr[rows], dtype=np.float64),
                mu_u[c0:c1],
            )
            v_bins[c0:c1] = zc @ u_top

    norms = np.linalg.norm(v_bins, axis=0)
    keep = norms > 1e-12
    if not np.any(keep):
        return WhitenedPCAResult(
            mean_rate=mean_rate,
            loadings=np.empty((0, u_count), dtype=np.float32),
            singular_values=np.empty(0, dtype=np.float64),
            mp_sigma=mp_sigma,
        )
    v_unit = v_bins[:, keep] / norms[np.newaxis, keep]
    s_kept = s_all[:rank][keep]
    _, flipped = svd_flip(
        np.zeros((n_samples, int(s_kept.size)), dtype=np.float64),
        v_unit.T,
        u_based_decision=False,
    )
    total_sq = max(float(np.sum(np.square(s_all))), 1e-12)
    pct_vars = (np.square(s_kept) / total_sq) * 100.0
    logger.debug(
        "PCA fitted %d components: explained variance [%s] (mp_sigma=%.4f)",
        int(s_kept.size),
        ", ".join(f"{v:.2f}%" for v in pct_vars),
        mp_sigma,
    )
    return WhitenedPCAResult(
        mean_rate=mean_rate,
        loadings=np.asarray(flipped, dtype=np.float32),
        singular_values=np.asarray(s_kept, dtype=np.float64),
        mp_sigma=mp_sigma,
    )


def reconstruct_whitened_pca_expected(
    counts: NDArray[np.float32],
    expected: NDArray[np.float32],
    usable: NDArray[np.bool_],
    pca_result: WhitenedPCAResult,
) -> NDArray[np.float32]:
    """Reconstruct positive per-sample expected counts lambda_hat from whitened PCA."""
    n_bins, n_samples = expected.shape
    mask = np.asarray(usable, dtype=np.bool_)
    u_idx = np.flatnonzero(mask)
    u_count = int(u_idx.size)
    recon = np.zeros((n_bins, n_samples), dtype=np.float32)
    if u_count == 0:
        return recon

    mu_u = pca_result.mean_rate[u_idx]
    rank = int(pca_result.loadings.shape[0])
    chunk_rows = _pca_chunk_rows(n_samples)
    if rank == 0:
        for c0 in range(0, u_count, chunk_rows):
            c1 = min(u_count, c0 + chunk_rows)
            rows = u_idx[c0:c1]
            e0 = np.asarray(expected[rows], dtype=np.float64) * mu_u[
                c0:c1, np.newaxis
            ].astype(np.float64)
            recon[rows] = np.maximum(e0, 1e-12).astype(np.float32)
        return recon

    v_f64 = np.asarray(pca_result.loadings, dtype=np.float64)
    scores = np.zeros((rank, n_samples), dtype=np.float64)
    for c0 in range(0, u_count, chunk_rows):
        c1 = min(u_count, c0 + chunk_rows)
        rows = u_idx[c0:c1]
        zc = _whitened_centered_from_mu(
            np.asarray(counts[rows], dtype=np.float64),
            np.asarray(expected[rows], dtype=np.float64),
            mu_u[c0:c1],
        )
        scores += v_f64[:, c0:c1] @ zc

    for c0 in range(0, u_count, chunk_rows):
        c1 = min(u_count, c0 + chunk_rows)
        rows = u_idx[c0:c1]
        e0 = np.asarray(expected[rows], dtype=np.float64) * mu_u[
            c0:c1, np.newaxis
        ].astype(np.float64)
        hat_z = v_f64[:, c0:c1].T @ scores
        eta = np.clip(hat_z / np.sqrt(e0 + 0.5), -_LOG_WAVE_CLIP, _LOG_WAVE_CLIP)
        recon[rows] = np.maximum(e0 * np.exp(eta), 1e-12).astype(np.float32)
    return recon


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
    if n_components is not None and n_components < 1:
        raise ReferenceError("PCA requires at least one sample and one unmasked bin")
    max_k = min(
        10 if n_components is None else n_components,
        samples.shape[0],
        samples.shape[1],
    )
    if max_k < 1:
        raise ReferenceError("PCA requires at least one sample and one unmasked bin")
    rank = optimal_svht_rank(data, max_components=max_k)
    rank = min(rank, samples.shape[0], samples.shape[1])
    mean = np.mean(samples, axis=0)
    if np.allclose(samples, mean):
        return (
            np.ones(data.shape, dtype=np.float64),
            mean.astype(np.float64, copy=False),
            np.zeros((rank, samples.shape[1]), dtype=np.float64),
        )
    pca = PCA(n_components=rank, svd_solver="full")
    pca.fit(samples)
    if (
        hasattr(pca, "explained_variance_ratio_")
        and pca.explained_variance_ratio_ is not None
    ):
        finite_vars = np.nan_to_num(pca.explained_variance_ratio_ * 100.0)
        logger.debug(
            "PCA fitted %d components: explained variance [%s] (total=%.2f%%)",
            rank,
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
