"""Prototype Graph fitting, LOCO evaluation, and Empirical Bayes dispersion moderation."""

from __future__ import annotations

import logging
import tempfile
import warnings
from contextlib import ExitStack
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.spatial import KDTree
from scipy.special import digamma, polygamma
from scipy.stats import poisson
from sklearn.cluster import MiniBatchKMeans
from sklearn.exceptions import ConvergenceWarning

logger = logging.getLogger(__name__)

LN2_SQ: Final[float] = float(np.log(2.0) ** 2)
_MAX_INMEM_RATIO_ELEMENTS: Final[int] = 32_000_000
_DISPERSION_ROW_CHUNK: Final[int] = 32_768
_WINSOR_DISPERSION_SIGMA: Final[float] = 3.5

__all__ = [
    "LN2_SQ",
    "build_whitened_subspace_features",
    "calibrate_prototype_dispersion",
    "compute_kish_loco_inflation",
    "compute_loco_prototype_support",
    "evaluate_loco_prototype_baseline",
    "exact_poisson_log2_jensen_bias",
    "exact_poisson_log_variance",
    "fit_prototype_graph",
    "squeeze_var",
]


def _build_poisson_log_table() -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Precompute exact mean bias and variance of ln(Poisson(lambda) + 0.5)."""
    knots = np.geomspace(1e-3, 250.0, 160)
    log_knots = np.log(knots)
    jensen_log2 = np.empty_like(knots)
    variances = np.empty_like(knots)
    inv_ln2 = 1.0 / float(np.log(2.0))
    for idx, lam in enumerate(knots):
        k_max = max(32, int(lam + 10.0 * np.sqrt(lam) + 20.0))
        k = np.arange(k_max + 1, dtype=np.float64)
        pmf = poisson.pmf(k, lam)
        vals = np.log(k + 0.5)
        mu = float(np.sum(pmf * vals))
        jensen_log2[idx] = (mu - float(np.log(lam + 0.5))) * inv_ln2
        variances[idx] = float(np.sum(pmf * np.square(vals - mu)))
    return log_knots, jensen_log2, variances


_POISSON_LOG_KNOTS, _POISSON_JENSEN_LOG2, _POISSON_LOG_VARS = _build_poisson_log_table()


def exact_poisson_log2_jensen_bias(lam: NDArray[np.floating]) -> NDArray[np.float64]:
    """Evaluate E_{C~Pois(lam)}[log2((C + 0.5) / (lam + 0.5))] in log2 units."""
    l_clamped = np.maximum(np.asarray(lam, dtype=np.float64), 1e-3)
    out = np.interp(np.log(l_clamped), _POISSON_LOG_KNOTS, _POISSON_JENSEN_LOG2)
    hi = l_clamped > 250.0
    if np.any(hi):
        lh = l_clamped[hi] + 0.5
        out[hi] = -l_clamped[hi] / (2.0 * float(np.log(2.0)) * np.square(lh))
    return out


def exact_poisson_log_variance(lam: NDArray[np.floating]) -> NDArray[np.float64]:
    """Evaluate exact discrete Poisson shot variance of ln(C + 0.5) in natural-log^2 units."""
    l_clamped = np.maximum(np.asarray(lam, dtype=np.float64), 1e-3)
    out = np.interp(np.log(l_clamped), _POISSON_LOG_KNOTS, _POISSON_LOG_VARS)
    hi = l_clamped > 250.0
    if np.any(hi):
        lh = l_clamped[hi] + 0.5
        out[hi] = (1.0 / lh) + (1.0 / np.square(lh))
    return out


def build_whitened_subspace_features(
    pca_loadings: NDArray[np.floating],
    singular_values: NDArray[np.floating],
    mp_sigma: float,
    *,
    covariate_mat: NDArray[np.floating] | None = None,
    fallback_rate: NDArray[np.floating] | None = None,
) -> NDArray[np.float32]:
    """Build signal-weighted PC + standardized covariate coordinates on usable bins."""
    loadings = np.asarray(pca_loadings, dtype=np.float64)
    s_vals = np.asarray(singular_values, dtype=np.float64)
    blocks: list[NDArray[np.float64]] = []
    u_count = 0

    if loadings.ndim == 2 and loadings.shape[1] > 0:
        u_count = int(loadings.shape[1])
        if loadings.shape[0] > 0 and s_vals.size == loadings.shape[0]:
            s1_sq = max(float(s_vals[0]) ** 2, 1e-12)
            excess = np.maximum(0.0, np.square(s_vals) - float(mp_sigma) ** 2)
            weights = np.sqrt(excess / s1_sq) * np.sqrt(float(u_count))
            keep = weights > 1e-12
            if np.any(keep):
                blocks.append(loadings[keep].T * weights[np.newaxis, keep])

    if covariate_mat is not None:
        cov = np.asarray(covariate_mat, dtype=np.float64)
        if cov.ndim == 1:
            cov = cov[:, np.newaxis]
        if cov.shape[0] > 0:
            u_count = int(cov.shape[0])
        if cov.size > 0 and not np.allclose(cov, 0.0):
            blocks.append(cov)

    if not blocks and fallback_rate is not None:
        rate = np.asarray(fallback_rate, dtype=np.float64)
        u_count = int(rate.size)
        pos = np.isfinite(rate) & (rate > 0.0)
        if np.count_nonzero(pos) >= 2:
            log_r = np.zeros(u_count, dtype=np.float64)
            log_r[pos] = np.log(rate[pos])
            sd = float(np.std(log_r[pos]))
            if sd > 1e-6:
                blocks.append(
                    ((log_r - float(np.mean(log_r[pos]))) / sd)[:, np.newaxis]
                )

    if not blocks:
        return np.empty((u_count, 0), dtype=np.float32)
    return np.hstack(blocks).astype(np.float32)


def fit_prototype_graph(
    features: NDArray[np.floating],
    *,
    max_prototypes: int = 2048,
    top_r: int = 4,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.uint16],
    NDArray[np.float16],
    NDArray[np.float32],
]:
    """Cluster signal-space features into M <= max_prototypes prototypes and query top_r."""
    if max_prototypes < 1 or max_prototypes > 65535:
        raise ValueError("max_prototypes must be in [1, 65535]")
    if top_r < 1:
        raise ValueError("top_r must be a positive integer")

    feats = np.asarray(features, dtype=np.float32)
    if feats.ndim == 1:
        feats = feats[:, np.newaxis]
    if feats.ndim != 2:
        raise ValueError("features must be a 1D or 2D array")

    u_count, n_dim = feats.shape
    if u_count == 0:
        return (
            np.empty((0, n_dim), dtype=np.float32),
            np.zeros((0, top_r), dtype=np.uint16),
            np.zeros((0, top_r), dtype=np.float16),
            np.empty(0, dtype=np.float32),
        )

    if n_dim == 0:
        proto_ids = np.zeros((u_count, top_r), dtype=np.uint16)
        proto_weights = np.zeros((u_count, top_r), dtype=np.float16)
        proto_weights[:, 0] = np.float16(1.0)
        return (
            np.empty((1, 0), dtype=np.float32),
            proto_ids,
            proto_weights,
            np.zeros(u_count, dtype=np.float32),
        )

    scaled_cap = max(8, u_count // 64) if u_count >= 8 else u_count
    m_count = min(max_prototypes, u_count, max(1, scaled_cap))
    k_query = min(top_r, m_count)
    logger.debug(
        "fitting prototype graph: %d usable bins -> %d prototypes (top_r=%d)",
        u_count,
        m_count,
        k_query,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        kmeans = MiniBatchKMeans(
            n_clusters=m_count,
            random_state=0,
            batch_size=min(8192, max(256, u_count)),
            n_init=3,
        )
        kmeans.fit(feats)

    centers = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
    tree = KDTree(centers)
    raw_dists, raw_ids = tree.query(feats, k=k_query, p=2, workers=-1)
    dists = np.asarray(raw_dists, dtype=np.float64).reshape(u_count, k_query)
    ids = np.asarray(raw_ids, dtype=np.uint16).reshape(u_count, k_query)

    inv_d = 1.0 / (dists + 1e-6)
    norm_w = inv_d / np.sum(inv_d, axis=1, keepdims=True)
    w_f16 = norm_w.astype(np.float16)
    w_f32 = w_f16.astype(np.float32)
    w_sum = np.maximum(np.sum(w_f32, axis=1, keepdims=True), 1e-12)
    w_f16 = (w_f32 / w_sum).astype(np.float16)

    proto_ids = np.zeros((u_count, top_r), dtype=np.uint16)
    proto_ids[:, :k_query] = ids
    proto_weights = np.zeros((u_count, top_r), dtype=np.float16)
    proto_weights[:, :k_query] = w_f16
    mean_distance = np.mean(dists, axis=1).astype(np.float32)
    return centers, proto_ids, proto_weights, mean_distance


def _resolve_prototype_weights(
    pids: NDArray[np.intp],
    pw: NDArray[np.float64],
    valid_proto: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Resolve effective prototype weights and deduplicated support mask per bin."""
    u_count, r_cols = pids.shape
    first_occ = np.ones((u_count, r_cols), dtype=np.bool_)
    for col_j in range(1, r_cols):
        first_occ[:, col_j] = np.all(
            pids[:, :col_j] != pids[:, col_j : col_j + 1], axis=1
        )
    valid_w = valid_proto & (pw > 0.0)
    eff_w = np.where(valid_w, pw, 0.0)
    w_sum = np.sum(eff_w, axis=1)
    fallback_proto = valid_proto & first_occ
    need_unif = (w_sum <= 0.0) & np.any(fallback_proto, axis=1)
    if np.any(need_unif):
        eff_w = np.where(
            need_unif[:, np.newaxis], np.where(fallback_proto, 1.0, 0.0), eff_w
        )
        w_sum = np.sum(eff_w, axis=1)
    used_mask = np.where(need_unif[:, np.newaxis], fallback_proto, valid_w & first_occ)
    return eff_w, w_sum, used_mask


def compute_loco_prototype_support(
    contig_ids: NDArray[np.integer],
    proto_ids: NDArray[np.uint16],
    proto_weights: NDArray[np.floating],
    *,
    eligible: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Compute O(U) LOCO prototype donor support counts and inverse effective sample sizes."""
    u_count = int(contig_ids.size)
    if u_count == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    if proto_ids.size == 0:
        return np.zeros(u_count, dtype=np.int64), np.ones(u_count, dtype=np.float64)

    elig = (
        np.ones(u_count, dtype=np.bool_)
        if eligible is None
        else np.asarray(eligible, dtype=np.bool_)
    )
    e_total = int(np.count_nonzero(elig))
    if e_total == 0:
        return np.zeros(u_count, dtype=np.int64), np.ones(u_count, dtype=np.float64)

    _, c_compact = np.unique(
        np.asarray(contig_ids, dtype=np.int64), return_inverse=True
    )
    pids = np.asarray(proto_ids, dtype=np.intp)
    pw = np.asarray(proto_weights, dtype=np.float64)
    n_contigs = int(np.max(c_compact)) + 1
    n_protos = int(np.max(pids)) + 1

    e_prim = pids[elig, 0]
    e_c = c_compact[elig]
    full_cnt = np.bincount(e_prim, minlength=n_protos)
    c_elig_cnt = np.bincount(e_c, minlength=n_contigs)
    if n_contigs > 1:
        pc_cnt = np.zeros((n_protos, n_contigs), dtype=np.int64)
        np.add.at(pc_cnt, (e_prim, e_c), 1)
        loco_cnt = full_cnt[:, np.newaxis] - pc_cnt
    else:
        loco_cnt = np.where(full_cnt == 1, 0, full_cnt)[:, np.newaxis]

    p_cnt = loco_cnt[pids, c_compact[:, np.newaxis]]
    eff_w, w_sum, used_mask = _resolve_prototype_weights(pids, pw, p_cnt > 0)
    has_est = w_sum > 0.0
    support = np.sum(np.where(used_mask, p_cnt, 0), axis=1, dtype=np.int64)

    if np.any(~has_est):
        off_cnt = (
            e_total - c_elig_cnt[c_compact]
            if n_contigs > 1
            else np.zeros(u_count, dtype=np.int64)
        )
        lso_cnt = (
            np.ones(u_count, dtype=np.int64)
            if e_total == 1
            else (e_total - elig.astype(np.int64))
        )
        fallback_sup = np.where(off_cnt > 0, off_cnt, lso_cnt)
        support = np.where(has_est, support, fallback_sup)

    w_norm = eff_w / np.where(has_est, w_sum, 1.0)[:, np.newaxis]
    inv_keff = np.where(
        has_est,
        np.sum(
            np.where(used_mask, np.square(w_norm) / np.maximum(p_cnt, 1), 0.0), axis=1
        ),
        1.0 / np.maximum(support.astype(np.float64), 1.0),
    )
    return support, inv_keff


def evaluate_loco_prototype_baseline(
    values: NDArray[np.floating],
    contig_ids: NDArray[np.integer],
    proto_ids: NDArray[np.uint16],
    proto_weights: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Evaluate O(U) leave-one-contig-out prototype baselines and bin support counts."""
    vals = np.asarray(values, dtype=np.float64)
    cids = np.asarray(contig_ids, dtype=np.int64)
    pids = np.asarray(proto_ids, dtype=np.intp)
    pw = np.asarray(proto_weights, dtype=np.float64)
    u_count = vals.size
    if u_count == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)

    elig = np.isfinite(vals) & (vals > 0.0)
    if not np.any(elig) or pids.size == 0:
        return np.full(u_count, np.nan, dtype=np.float64), np.zeros(
            u_count, dtype=np.int64
        )

    _, c_compact = np.unique(cids, return_inverse=True)
    n_contigs = int(np.max(c_compact)) + 1
    n_protos = int(np.max(pids)) + 1

    e_idx = np.flatnonzero(elig)
    e_prim = pids[e_idx, 0]
    e_c = c_compact[e_idx]
    e_v = vals[e_idx]

    order = np.lexsort((e_c, e_prim))
    s_prim = e_prim[order]
    s_c = e_c[order]
    s_v = e_v[order]

    proto_range = np.arange(n_protos, dtype=np.intp)
    p_starts = np.searchsorted(s_prim, proto_range, side="left")
    p_ends = np.searchsorted(s_prim, proto_range, side="right")

    full_med = np.full(n_protos, np.nan, dtype=np.float64)
    full_cnt = (p_ends - p_starts).astype(np.int64)
    for m in range(n_protos):
        lo = int(p_starts[m])
        hi = int(p_ends[m])
        if hi > lo:
            full_med[m] = float(np.median(s_v[lo:hi]))

    loco_med = np.broadcast_to(full_med[:, np.newaxis], (n_protos, n_contigs)).copy()
    loco_cnt = np.broadcast_to(full_cnt[:, np.newaxis], (n_protos, n_contigs)).copy()

    for m in range(n_protos):
        lo = int(p_starts[m])
        hi = int(p_ends[m])
        if hi <= lo:
            continue
        if n_contigs == 1:
            if hi - lo == 1:
                loco_med[m, 0] = np.nan
                loco_cnt[m, 0] = 0
            continue
        c_m = s_c[lo:hi]
        if c_m[0] == c_m[-1]:
            ck_only = int(c_m[0])
            loco_med[m, ck_only] = np.nan
            loco_cnt[m, ck_only] = 0
            continue
        v_m = s_v[lo:hi]
        cuts = np.flatnonzero(c_m[1:] != c_m[:-1]) + 1
        r_starts = [0, *cuts.tolist()]
        r_ends = [*cuts.tolist(), int(v_m.size)]
        for r_lo, r_hi in zip(r_starts, r_ends, strict=True):
            ck = int(c_m[r_lo])
            off_v = np.concatenate((v_m[:r_lo], v_m[r_hi:]))
            loco_med[m, ck] = float(np.median(off_v))
            loco_cnt[m, ck] = int(off_v.size)

    p_est = loco_med[pids, c_compact[:, np.newaxis]]
    p_cnt = loco_cnt[pids, c_compact[:, np.newaxis]]
    valid_proto = np.isfinite(p_est) & (p_est > 0.0) & (p_cnt > 0)
    eff_w, w_sum, used_mask = _resolve_prototype_weights(pids, pw, valid_proto)
    has_est = w_sum > 0.0
    baselines = np.where(
        has_est,
        np.sum(eff_w * np.where(valid_proto, p_est, 0.0), axis=1)
        / np.where(has_est, w_sum, 1.0),
        np.nan,
    )
    support = np.sum(np.where(used_mask, p_cnt, 0), axis=1, dtype=np.int64)

    missing_rows = np.flatnonzero(~has_est)
    if missing_rows.size > 0:
        lso_rows: list[int] = []
        if n_contigs > 1:
            for ck in np.unique(c_compact[missing_rows]):
                ck_int = int(ck)
                ck_rows = missing_rows[c_compact[missing_rows] == ck_int]
                off_elig = e_v[e_c != ck_int]
                if off_elig.size > 0:
                    baselines[ck_rows] = float(np.median(off_elig))
                    support[ck_rows] = int(off_elig.size)
                else:
                    lso_rows.extend(int(r) for r in ck_rows)
        else:
            lso_rows = [int(r) for r in missing_rows]

        if lso_rows and e_v.size >= 2:
            sv = np.sort(e_v)
            e_len = int(sv.size)
            full_e_med = float(np.median(sv))
            m_len = e_len - 1
            for r_i in lso_rows:
                if not elig[r_i]:
                    baselines[r_i] = full_e_med
                    support[r_i] = e_len
                else:
                    pos = int(np.searchsorted(sv, vals[r_i], side="left"))
                    if m_len % 2 == 1:
                        q = m_len // 2
                        baselines[r_i] = float(sv[q if q < pos else q + 1])
                    else:
                        q0, q1 = m_len // 2 - 1, m_len // 2
                        i0 = q0 if q0 < pos else q0 + 1
                        i1 = q1 if q1 < pos else q1 + 1
                        baselines[r_i] = 0.5 * float(sv[i0] + sv[i1])
                    support[r_i] = m_len
        elif lso_rows and e_v.size == 1:
            baselines[lso_rows] = float(e_v[0])
            support[lso_rows] = 1

    return baselines, support


def compute_kish_loco_inflation(
    contig_ids: NDArray[np.integer],
    proto_ids: NDArray[np.uint16],
    proto_weights: NDArray[np.floating],
    support: NDArray[np.integer],
    *,
    eligible: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Compute Kish LOCO prototype median variance factor 1 + pi / (2 * K_eff)."""
    u_count = int(contig_ids.size)
    if u_count == 0:
        return np.empty(0, dtype=np.float64)
    if proto_ids.size == 0:
        return 1.0 + 0.5 * np.pi / np.maximum(
            np.asarray(support, dtype=np.float64), 1.0
        )
    sup_calc, inv_keff = compute_loco_prototype_support(
        contig_ids, proto_ids, proto_weights, eligible=eligible
    )
    if np.all(sup_calc == 0):
        return 1.0 + 0.5 * np.pi / np.maximum(
            np.asarray(support, dtype=np.float64), 1.0
        )
    return 1.0 + 0.5 * np.pi * inv_keff


def _winsorized_anscombe_mean_and_var(
    r_chunk: NDArray[np.float64],
    mean_lam: NDArray[np.float64],
    ddof: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Winsorize log2 ratios in skewness-free Anscombe space and compute mean and variance."""
    n_rows, n_samples = r_chunk.shape
    if n_samples == 0:
        return np.zeros(n_rows, dtype=np.float64), np.zeros(n_rows, dtype=np.float64)

    lam_col = np.maximum(mean_lam, 1e-6)[:, np.newaxis]
    j_col = exact_poisson_log2_jensen_bias(lam_col.ravel())[:, np.newaxis]
    root_lam = np.sqrt(lam_col + 0.375)
    finite_mask = np.isfinite(r_chunk)
    all_finite = bool(np.all(finite_mask))
    exp_arg = np.clip(np.where(finite_mask, r_chunk + j_col, 0.0), -20.0, 20.0)
    c_p375 = np.maximum(0.375, (lam_col + 0.5) * np.exp2(exp_arg) - 0.125)
    a_mat = np.where(finite_mask, 2.0 * (np.sqrt(c_p375) - root_lam), np.nan)

    if all_finite:
        med_a = np.median(a_mat, axis=1, keepdims=True)
        mad_a = 1.4826 * np.median(np.abs(a_mat - med_a), axis=1, keepdims=True)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med_a = np.nan_to_num(np.nanmedian(a_mat, axis=1, keepdims=True), nan=0.0)
            mad_a = np.nan_to_num(
                1.4826 * np.nanmedian(np.abs(a_mat - med_a), axis=1, keepdims=True),
                nan=1.0,
            )

    scale_a = np.maximum(1.0, mad_a)
    a_win = np.clip(
        np.where(finite_mask, a_mat, med_a),
        med_a - _WINSOR_DISPERSION_SIGMA * scale_a,
        med_a + _WINSOR_DISPERSION_SIGMA * scale_a,
    )
    c_win_p5 = np.square(np.maximum(0.0, root_lam + 0.5 * a_win)) + 0.125
    r_mapped = np.log2(c_win_p5 / (lam_col + 0.5)) - j_col

    if all_finite:
        r_win = np.where(a_win != a_mat, r_mapped, r_chunk)
        mean_out = np.mean(r_win, axis=1)
        if n_samples <= 1:
            return mean_out, np.zeros(n_rows, dtype=np.float64)
        sse = np.sum(np.square(r_win - mean_out[:, np.newaxis]), axis=1)
        return mean_out, sse / max(1.0, float(n_samples) - ddof)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r_win = np.where(
            finite_mask, np.where(a_win != a_mat, r_mapped, r_chunk), np.nan
        )
        cnt = np.sum(finite_mask, axis=1)
        mean_out = np.where(
            cnt > 0, np.nan_to_num(np.nanmean(r_win, axis=1), nan=0.0), 0.0
        )
        sse = np.nansum(np.square(r_win - mean_out[:, np.newaxis]), axis=1)
        var_out = np.where(
            cnt > 1, sse / np.maximum(1.0, cnt.astype(np.float64) - ddof), 0.0
        )
    return mean_out, var_out


def calibrate_prototype_dispersion(
    counts: NDArray[np.float32],
    recon_expected: NDArray[np.float32],
    contigs: NDArray[np.int16],
    usable: NDArray[np.bool_],
    proto_ids: NDArray[np.uint16],
    proto_weights: NDArray[np.float16],
    *,
    pca_rank: int,
    n_prototypes: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Calibrate full-cohort null_mean and depth-decomposed EB bio_dispersion across all samples."""
    del n_prototypes
    counts_arr = counts if isinstance(counts, np.ndarray) else np.asarray(counts)
    exp_arr = (
        recon_expected
        if isinstance(recon_expected, np.ndarray)
        else np.asarray(recon_expected)
    )
    if counts_arr.ndim != 2 or exp_arr.ndim != 2 or counts_arr.shape != exp_arr.shape:
        raise ValueError(
            "counts and recon_expected must be 2D arrays of matching shape"
        )

    n_bins, n_samples = counts_arr.shape
    mask = np.asarray(usable, dtype=np.bool_)
    if mask.ndim != 1 or mask.shape[0] != n_bins:
        raise ValueError("usable must be a 1D boolean mask aligned with counts")

    null_mean = np.zeros(n_bins, dtype=np.float32)
    bio_disp = np.zeros(n_bins, dtype=np.float32)
    u_idx = np.flatnonzero(mask)
    u_count = int(u_idx.size)
    if u_count == 0 or n_samples == 0:
        return null_mean, bio_disp

    c_ids_u = np.asarray(contigs, dtype=np.int64)[u_idx]
    if proto_ids.shape[0] == u_count and proto_weights.shape[0] == u_count:
        pids_u = proto_ids
        pw_u = proto_weights
    elif proto_ids.shape[0] == n_bins and proto_weights.shape[0] == n_bins:
        pids_u = proto_ids[u_idx]
        pw_u = proto_weights[u_idx]
    else:
        raise ValueError(
            "proto_ids and proto_weights must align with usable or full bins"
        )

    shot_sum = np.zeros(u_count, dtype=np.float64)
    lam_sum = np.zeros(u_count, dtype=np.float64)
    shot_cnt = np.zeros(u_count, dtype=np.int32)
    last_sup = np.zeros(u_count, dtype=np.int64)

    def _populate_ratios(r_store: NDArray[np.float32]) -> None:
        nonlocal last_sup
        for s_idx in range(n_samples):
            c_s = np.asarray(counts_arr[u_idx, s_idx], dtype=np.float64)
            lam_s = np.asarray(exp_arr[u_idx, s_idx], dtype=np.float64)
            valid_in = (
                np.isfinite(c_s) & (c_s >= 0.0) & np.isfinite(lam_s) & (lam_s > 0.0)
            )
            q_s = np.where(valid_in, (c_s + 0.5) / (lam_s + 0.5), np.nan)
            base_s, sup_s = evaluate_loco_prototype_baseline(q_s, c_ids_u, pids_u, pw_u)
            b_pos = valid_in & np.isfinite(base_s) & (base_s > 0.0) & (sup_s > 0)
            if u_count >= 20 and int(np.count_nonzero(b_pos)) >= 8:
                b_med = float(np.median(base_s[b_pos]))
                if b_med > 0.0:
                    base_s = base_s / b_med
            last_sup = np.maximum(last_sup, sup_s)
            ok = valid_in & np.isfinite(base_s) & (base_s > 0.0) & (sup_s > 0)
            eff_lam = np.where(ok, lam_s * base_s, 1.0)
            r_col = np.full(u_count, np.nan, dtype=np.float32)
            r_unbiased = np.log2(
                (c_s[ok] + 0.5) / (eff_lam[ok] + 0.5)
            ) - exact_poisson_log2_jensen_bias(eff_lam[ok])
            r_col[ok] = r_unbiased.astype(np.float32)
            r_store[:, s_idx] = r_col
            shot_sum[ok] += exact_poisson_log_variance(eff_lam[ok]) / LN2_SQ
            lam_sum[ok] += eff_lam[ok]
            shot_cnt[ok] += 1

    ddof = 1.0 + float(max(0, pca_rank)) / float(max(n_samples, 1))
    mean_u = np.zeros(u_count, dtype=np.float64)
    obs_var_u = np.zeros(u_count, dtype=np.float64)

    with ExitStack() as stack:
        if u_count * n_samples <= _MAX_INMEM_RATIO_ELEMENTS:
            r_store: NDArray[np.float32] = np.empty(
                (u_count, n_samples), dtype=np.float32
            )
        else:
            tmp_dir = stack.enter_context(tempfile.TemporaryDirectory())
            r_store = np.memmap(
                Path(tmp_dir) / "loco_ratios.f32",
                dtype=np.float32,
                mode="w+",
                shape=(u_count, n_samples),
            )
        try:
            _populate_ratios(r_store)
            if isinstance(r_store, np.memmap):
                r_store.flush()
            mean_lam_u = np.where(shot_cnt > 0, lam_sum / np.maximum(shot_cnt, 1), 1.0)
            for c0 in range(0, u_count, _DISPERSION_ROW_CHUNK):
                c1 = min(u_count, c0 + _DISPERSION_ROW_CHUNK)
                m_c, v_c = _winsorized_anscombe_mean_and_var(
                    np.asarray(r_store[c0:c1], dtype=np.float64),
                    mean_lam_u[c0:c1],
                    ddof,
                )
                mean_u[c0:c1] = m_c
                obs_var_u[c0:c1] = v_c
        finally:
            del r_store

    has_support = (shot_cnt > 0) & (last_sup > 0)
    null_mean[u_idx] = np.where(has_support, mean_u, 0.0).astype(np.float32)
    if n_samples <= 1:
        return null_mean, bio_disp

    d_eb = float(max(1, n_samples - 1 - max(0, pca_rank)))
    if n_samples > 1 and pca_rank > 0:
        obs_var_u = obs_var_u * (float(n_samples) - ddof) / float(d_eb)

    f_loco = compute_kish_loco_inflation(c_ids_u, pids_u, pw_u, last_sup)
    v_shot = np.where(shot_cnt > 0, shot_sum / np.maximum(shot_cnt, 1), 0.0)
    v_tot = obs_var_u / f_loco
    raw_phi = np.where(has_support, LN2_SQ * np.maximum(0.0, v_tot - v_shot), 0.0)

    pos_bins = has_support & (raw_phi > 1e-12)
    disp_u = np.zeros(u_count, dtype=np.float64)
    if u_count >= 10 and int(np.count_nonzero(pos_bins)) >= 2:
        shrunk_pos, _, _ = squeeze_var(raw_phi[pos_bins], df=d_eb)
        disp_u[pos_bins] = np.maximum(shrunk_pos, 0.0)
    else:
        disp_u[pos_bins] = raw_phi[pos_bins]

    disp_u[~np.isfinite(disp_u) | (disp_u <= 1e-16)] = 0.0
    bio_disp[u_idx] = disp_u.astype(np.float32)
    logger.debug(
        "null models computed for %d/%d usable bins across %d samples",
        int(np.count_nonzero(has_support)),
        u_count,
        n_samples,
    )
    return null_mean, bio_disp


def squeeze_var(
    var: NDArray[np.floating],
    df: float,
    winsor_tail_p: tuple[float, float] = (0.0, 0.0),
) -> tuple[NDArray[np.float64], float, float]:
    """Empirical Bayes variance moderation matching Smyth (2004) / limma::squeezeVar."""
    del winsor_tail_p
    var_arr = np.asarray(var, dtype=np.float64)
    valid = np.isfinite(var_arr) & (var_arr > 1e-16)
    if np.count_nonzero(valid) < 2:
        return var_arr.copy(), 1e6, 0.0

    v = var_arr[valid]
    z = np.log(v) - float(digamma(df / 2.0)) + np.log(df / 2.0)

    mean_z = float(np.mean(z))
    var_z = float(np.var(z, ddof=1))
    excess_var = var_z - float(polygamma(1, df / 2.0))

    if excess_var <= 1e-7:
        d0_hat = 1e6
        s0_sq_hat = float(np.exp(mean_z + digamma(d0_hat / 2.0) - np.log(d0_hat / 2.0)))
    else:

        def func(d0: float) -> float:
            return float(polygamma(1, d0 / 2.0)) - excess_var

        d0_hat = float(brentq(func, 1e-8, 1e8))
        s0_sq_hat = float(np.exp(mean_z + digamma(d0_hat / 2.0) - np.log(d0_hat / 2.0)))

    var_post = var_arr.copy()
    var_post[valid] = (d0_hat * s0_sq_hat + df * v) / (d0_hat + df)
    return var_post, d0_hat, s0_sq_hat
