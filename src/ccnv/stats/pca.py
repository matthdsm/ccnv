"""Variance-whitened outlier-clipped PCA and deterministic PCA fitting routines."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike, NDArray
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor
from sklearn.utils.extmath import svd_flip

from ccnv.models.exceptions import InvalidInputError

logger = logging.getLogger(__name__)

LOG_WAVE_CLIP: Final[float] = 4.0
_MAX_PCA_CHUNK_ROWS: Final[int] = 32_768
_TARGET_CHUNK_BYTES: Final[int] = 64 * 1024 * 1024
_WINSOR_Z_CLIP: Final[float] = 3.0

__all__ = [
    "LOG_WAVE_CLIP",
    "WhitenedPCAResult",
    "fit_whitened_pca",
    "pca_irlba_numpy",
    "project_huber_pca_wave",
    "reconstruct_whitened_pca_expected",
]


@dataclass(frozen=True, slots=True)
class WhitenedPCAResult:
    """Fitted variance-whitened outlier-clipped PCA model on usable bins."""

    mean_rate: NDArray[np.float32]
    loadings: NDArray[np.float32]
    singular_values: NDArray[np.float64]
    mp_sigma: float
    scores: NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )


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
        raise InvalidInputError(
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
    counts: NDArray[np.floating],
    expected: NDArray[np.floating],
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
        raise InvalidInputError(
            "counts and expected must be non-empty 2D arrays with identical shape"
        )
    if max_components is not None and (
        not isinstance(max_components, int)
        or isinstance(max_components, bool)
        or max_components < 0
    ):
        raise InvalidInputError("max_components must be None or a non-negative integer")

    n_bins, n_samples = counts_arr.shape
    mask = np.asarray(usable, dtype=np.bool_)
    if mask.ndim != 1 or mask.shape[0] != n_bins:
        raise InvalidInputError("usable must be a 1D boolean mask aligned with counts")
    u_idx = np.flatnonzero(mask)
    u_count = int(u_idx.size)
    if u_count == 0:
        raise InvalidInputError("fit_whitened_pca requires at least one usable bin")

    mean_rate = np.zeros(n_bins, dtype=np.float32)
    mu_u = np.empty(u_count, dtype=np.float32)
    chunk_rows = _pca_chunk_rows(n_samples)
    s_eff = max(0, n_samples - 1)
    m_nontrivial = min(u_count, s_eff)
    empty_scores = np.empty((0, n_samples), dtype=np.float64)

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
                scores=empty_scores,
            )
        evals, evecs = scipy.linalg.eigh(0.5 * (gram + gram.T))
        s_all = np.sqrt(np.maximum(evals[::-1][:m_nontrivial], 0.0))
        u_samp_all = np.asarray(evecs[:, ::-1][:, :m_nontrivial], dtype=np.float64)
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
        if m_nontrivial == 0:
            return WhitenedPCAResult(
                mean_rate=mean_rate,
                loadings=np.empty((0, u_count), dtype=np.float32),
                singular_values=np.empty(0, dtype=np.float64),
                mp_sigma=0.0,
                scores=empty_scores,
            )
        u_bins_raw, s_raw, vt_raw = scipy.linalg.svd(zc_full, full_matrices=False)
        s_all = np.asarray(s_raw[:m_nontrivial], dtype=np.float64)
        u_bins_all = np.asarray(u_bins_raw[:, :m_nontrivial], dtype=np.float64)
        u_samp_all = np.asarray(vt_raw[:m_nontrivial, :], dtype=np.float64).T

    mp_sigma = _svht_noise_cutoff(s_all, s_eff, u_count)
    if n_samples <= 2 or u_count <= 1:
        rank = 0
    else:
        cap = (
            min(60, max(5, n_samples // 5))
            if max_components is None
            else max_components
        )
        max_rank = max(0, min(cap, n_samples - 2, u_count - 1))
        rank = min(int(np.count_nonzero(s_all > max(mp_sigma, 1e-12))), max_rank)

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
            scores=empty_scores,
        )

    u_top = u_samp_all[:, :rank]
    if u_bins_all is not None:
        v_bins = u_bins_all[:, :rank].copy()
    else:
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
    keep = (norms > 1e-12) & (s_all[:rank] > 1e-12)
    if not np.any(keep):
        return WhitenedPCAResult(
            mean_rate=mean_rate,
            loadings=np.empty((0, u_count), dtype=np.float32),
            singular_values=np.empty(0, dtype=np.float64),
            mp_sigma=mp_sigma,
            scores=empty_scores,
        )

    v_unit = v_bins[:, keep] / norms[np.newaxis, keep]
    u_samp_kept = u_top[:, keep]
    s_kept = s_all[:rank][keep]
    u_samp_flipped, flipped = svd_flip(
        u_samp_kept,
        v_unit.T,
        u_based_decision=False,
    )
    scores = np.asarray(s_kept[:, np.newaxis] * u_samp_flipped.T, dtype=np.float64)
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
        scores=scores,
    )


def reconstruct_whitened_pca_expected(
    expected: NDArray[np.floating],
    usable: NDArray[np.bool_],
    pca_result: WhitenedPCAResult,
) -> NDArray[np.float32]:
    """Reconstruct positive per-sample expected counts lambda_hat from whitened PCA."""
    exp_arr = expected if isinstance(expected, np.ndarray) else np.asarray(expected)
    if exp_arr.ndim != 2:
        raise InvalidInputError("expected must be a 2D array")
    n_bins, n_samples = exp_arr.shape
    rank = int(pca_result.loadings.shape[0])
    if pca_result.scores.shape != (rank, n_samples):
        raise InvalidInputError(
            f"pca_result.scores shape {pca_result.scores.shape} does not match ({rank}, {n_samples})"
        )

    mask = np.asarray(usable, dtype=np.bool_)
    if mask.ndim != 1 or mask.shape[0] != n_bins:
        raise InvalidInputError(
            "usable must be a 1D boolean mask aligned with expected"
        )
    u_idx = np.flatnonzero(mask)
    u_count = int(u_idx.size)
    recon = np.zeros((n_bins, n_samples), dtype=np.float32)
    if u_count == 0:
        return recon
    if int(pca_result.loadings.shape[1]) != u_count:
        raise InvalidInputError(
            f"pca_result.loadings width {pca_result.loadings.shape[1]} does not match usable bin count {u_count}"
        )

    mu_u = pca_result.mean_rate[u_idx]
    v_f64 = np.asarray(pca_result.loadings, dtype=np.float64)
    scores = np.asarray(pca_result.scores, dtype=np.float64)
    chunk_rows = _pca_chunk_rows(n_samples)

    for c0 in range(0, u_count, chunk_rows):
        c1 = min(u_count, c0 + chunk_rows)
        rows = u_idx[c0:c1]
        e0 = np.asarray(exp_arr[rows], dtype=np.float64) * mu_u[
            c0:c1, np.newaxis
        ].astype(np.float64)
        if rank == 0:
            recon[rows] = np.maximum(e0, 1e-12).astype(np.float32)
        else:
            hat_z = v_f64[:, c0:c1].T @ scores
            eta = np.clip(hat_z / np.sqrt(e0 + 0.5), -LOG_WAVE_CLIP, LOG_WAVE_CLIP)
            recon[rows] = np.maximum(e0 * np.exp(eta), 1e-12).astype(np.float32)
    return recon


def project_huber_pca_wave(
    log_signal: NDArray[np.floating],
    loadings: NDArray[np.floating],
    *,
    whitening_scale: NDArray[np.floating] | None = None,
    eval_loadings: NDArray[np.floating] | None = None,
    eval_whitening_scale: NDArray[np.floating] | None = None,
    epsilon: float = 1.35,
    alpha: float = 1e-4,
    clip: float | None = LOG_WAVE_CLIP,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Project 1D signal onto whitened PCA loadings via rank-safe Huber M-estimation."""
    y = np.asarray(log_signal, dtype=np.float64).ravel()
    u_len = int(y.size)
    ld = np.asarray(loadings, dtype=np.float64)
    if ld.ndim != 2:
        raise InvalidInputError("loadings must be a 2D array of shape (K, U)")
    k_rank, u_cols = ld.shape
    if (k_rank > 0 and u_cols != u_len) or (k_rank == 0 and u_cols not in (0, u_len)):
        raise InvalidInputError(
            f"loadings width {u_cols} does not match log_signal length {u_len}"
        )
    ev_ld = ld if eval_loadings is None else np.asarray(eval_loadings, dtype=np.float64)
    if ev_ld.ndim != 2 or ev_ld.shape[0] != k_rank:
        raise InvalidInputError(
            "eval_loadings must be a 2D array with matching component rank"
        )
    eval_len = u_len if eval_loadings is None else int(ev_ld.shape[1])

    if (
        eval_loadings is None
        and eval_whitening_scale is None
        and whitening_scale is not None
    ):
        eval_whitening_scale = whitening_scale
    elif eval_loadings is not None and (whitening_scale is None) != (
        eval_whitening_scale is None
    ):
        raise InvalidInputError(
            "whitening_scale and eval_whitening_scale must both be provided or both omitted when eval_loadings is set"
        )

    w_arr: NDArray[np.float64] | None = None
    if whitening_scale is not None:
        w_arr = np.asarray(whitening_scale, dtype=np.float64).ravel()
        if w_arr.size != u_len:
            raise InvalidInputError(
                f"whitening_scale length {w_arr.size} does not match log_signal length {u_len}"
            )
        if u_len > 0 and bool(np.any(np.isfinite(w_arr) & (w_arr < 0.0))):
            raise InvalidInputError("whitening_scale cannot contain negative entries")

    ew_arr: NDArray[np.float64] | None = None
    if eval_whitening_scale is not None:
        ew_arr = np.asarray(eval_whitening_scale, dtype=np.float64).ravel()
        if ew_arr.size != eval_len:
            raise InvalidInputError(
                f"eval_whitening_scale length {ew_arr.size} does not match eval_loadings width {eval_len}"
            )
        if eval_len > 0 and bool(np.any(np.isfinite(ew_arr) & (ew_arr < 0.0))):
            raise InvalidInputError(
                "eval_whitening_scale cannot contain negative entries"
            )

    if k_rank == 0 or u_len == 0:
        return np.zeros(k_rank, dtype=np.float64), np.zeros(eval_len, dtype=np.float64)

    finite = np.isfinite(y) & np.all(np.isfinite(ld), axis=0)
    if w_arr is not None:
        finite &= np.isfinite(w_arr) & (w_arr > 0.0)

    n_valid = int(np.count_nonzero(finite))
    if n_valid < 2:
        return np.zeros(k_rank, dtype=np.float64), np.zeros(eval_len, dtype=np.float64)

    v_sub = ld[:, finite]
    y_sub = y[finite]
    raw_gram = v_sub @ v_sub.T
    gram_v = 0.5 * (raw_gram + raw_gram.T)
    evals_v, evecs_v = np.linalg.eigh(gram_v)
    tol_v = 1e-8 * max(float(np.max(evals_v)), 1.0)
    keep_v = evals_v > tol_v
    rank_eff = int(np.count_nonzero(keep_v))
    if rank_eff == 0 or n_valid <= rank_eff:
        return np.zeros(k_rank, dtype=np.float64), np.zeros(eval_len, dtype=np.float64)

    proj_basis = evecs_v[:, keep_v].T
    v_red_fit = proj_basis @ v_sub
    x_fit = v_red_fit.T
    y_fit = y_sub * w_arr[finite] if w_arr is not None else y_sub

    coef_red: NDArray[np.float64] | None = None
    if n_valid >= max(rank_eff + 2, 4) and float(np.std(y_fit)) > 1e-12:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                huber = HuberRegressor(
                    fit_intercept=False,
                    alpha=alpha,
                    max_iter=200,
                    epsilon=epsilon,
                )
                huber.fit(x_fit, y_fit)
                cand = np.asarray(huber.coef_, dtype=np.float64)
                if np.all(np.isfinite(cand)):
                    coef_red = cand
        except ValueError:
            coef_red = None

    if coef_red is None:
        coef_red = np.asarray(
            np.linalg.lstsq(x_fit, y_fit, rcond=None)[0], dtype=np.float64
        )

    coef = np.asarray(proj_basis.T @ coef_red, dtype=np.float64)
    ev_safe = np.where(np.isfinite(ev_ld), ev_ld, 0.0)
    wave = ev_safe.T @ coef
    if ew_arr is not None:
        ew_pos = np.isfinite(ew_arr) & (ew_arr > 0.0)
        wave = np.where(ew_pos, wave / np.where(ew_pos, ew_arr, 1.0), 0.0)
    if clip is not None:
        wave = np.clip(wave, -clip, clip)
    return coef, np.asarray(wave, dtype=np.float64)


def pca_irlba_numpy(
    X: ArrayLike,
    n_components: int,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute truncated unscaled PCA with sign canonicalization.

    Returns (scores, center, rotation) matching prcomp_irlba.
    """
    x_arr = np.asarray(X, dtype=np.float64)
    if x_arr.ndim != 2:
        raise InvalidInputError("X must be a 2D array")
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
