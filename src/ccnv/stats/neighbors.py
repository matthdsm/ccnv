"""Prototype Graph fitting, LOCO evaluation, nearest-neighbor search, and null moderation."""

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
from sklearn.cluster import MiniBatchKMeans
from sklearn.exceptions import ConvergenceWarning

logger = logging.getLogger(__name__)

_MAX_NULL_SAMPLES: Final[int] = 100

__all__ = [
    "build_signal_space_features",
    "compute_null_models",
    "compute_prototype_null_models",
    "distance_cutoff",
    "evaluate_loco_prototype_baseline",
    "fit_prototype_graph",
    "select_nearest_bins",
    "squeeze_var",
]


def build_signal_space_features(
    pca_loadings: NDArray[np.floating],
    singular_values: NDArray[np.floating] | None = None,
    covariate_mat: NDArray[np.floating] | None = None,
) -> NDArray[np.float64]:
    """Assemble low-dimensional signal-space feature matrix for prototype/neighbor search."""
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

    r_cols = pids.shape[1]
    first_occ = np.ones((u_count, r_cols), dtype=np.bool_)
    for col_j in range(1, r_cols):
        first_occ[:, col_j] = np.all(
            pids[:, :col_j] != pids[:, col_j : col_j + 1], axis=1
        )

    p_est = loco_med[pids, c_compact[:, np.newaxis]]
    p_cnt = loco_cnt[pids, c_compact[:, np.newaxis]]
    valid_proto = np.isfinite(p_est) & (p_est > 0.0) & (p_cnt > 0)
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


def compute_prototype_null_models(
    corrected: NDArray[np.floating],
    contig_ids: NDArray[np.integer],
    proto_ids: NDArray[np.uint16],
    proto_weights: NDArray[np.floating],
    *,
    n_pca_components: int = 0,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Compute Empirical Bayes moderated null_mean and bio_dispersion (variance) via prototypes."""
    mat = np.asarray(corrected, dtype=np.float64)
    n_bins, n_samples = mat.shape
    n_null = min(n_samples, _MAX_NULL_SAMPLES)
    if n_bins == 0 or n_null == 0:
        return np.zeros(n_bins, dtype=np.float32), np.zeros(n_bins, dtype=np.float32)

    sub = mat[:, :n_null]
    ratios = np.full((n_bins, n_null), np.nan, dtype=np.float64)
    has_support = np.zeros(n_bins, dtype=np.bool_)
    for s_idx in range(n_null):
        col = sub[:, s_idx]
        base, sup = evaluate_loco_prototype_baseline(
            col, contig_ids, proto_ids, proto_weights
        )
        valid = (
            np.isfinite(col)
            & (col > 0.0)
            & np.isfinite(base)
            & (base > 0.0)
            & (sup > 0)
        )
        has_support |= sup > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios[valid, s_idx] = np.log2(col[valid] / base[valid])

    clean = np.nan_to_num(ratios, nan=0.0, posinf=0.0, neginf=0.0)
    out_mean = np.where(has_support, np.mean(clean, axis=1), 0.0)

    d = max(1, n_null - 1 - n_pca_components)
    dof_scale = (n_null - 1) / d if n_null > 1 else 0.0
    raw_vars = (
        np.where(has_support, np.var(clean, axis=1, ddof=1) * dof_scale, 0.0)
        if n_null > 1
        else np.zeros(n_bins, dtype=np.float64)
    )

    valid_bins = has_support & (raw_vars > 1e-16)
    if n_bins >= 10 and np.count_nonzero(valid_bins) >= 2:
        mod_vars, _, _ = squeeze_var(raw_vars, df=float(d))
        disp = np.where(valid_bins, np.maximum(mod_vars, 0.0), 0.0)
    else:
        disp = np.where(valid_bins, np.maximum(raw_vars, 0.0), 0.0)

    disp[~np.isfinite(disp) | (disp <= 1e-16)] = 0.0
    return out_mean.astype(np.float32), disp.astype(np.float32)


def select_nearest_bins(
    profiles: NDArray[np.floating],
    contig_ids: NDArray[np.integer],
    k: int,
    max_batch_memory_mb: int = 256,
    threads: int = 1,
) -> tuple[list[list[int]], list[list[float]]]:
    """Pick k nearest off-contig bins via per-contig leave-one-out KDTree query."""
    del max_batch_memory_mb
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


def compute_null_models(
    corrected: NDArray[np.float64],
    nearest_indices: Sequence[Sequence[int]],
    threads: int = 1,
    n_pca_components: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute per-bin reference medians and DOF-corrected empirical Bayes null statistics."""
    del threads
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
