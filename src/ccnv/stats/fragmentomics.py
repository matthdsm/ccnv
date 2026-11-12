"""Stateless mathematical and machine learning kernels for cfDNA fragmentomics."""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage, optimize, signal
from sklearn.model_selection import KFold

from ccnv._core import FRAG_BAND_EDGES
from ccnv.models.schemas import FragmentomicsMetadata, PrenatalModelMetadata
from ccnv.stats.regression import fit_logit_linear_ensemble

FRAG_MIN_LEN: Final[int] = 30
FRAG_MAX_LEN: Final[int] = 600
FRAG_NUM_BINS: Final[int] = 571
CANONICAL_CHROMATOSOME_BP: Final[float] = 166.0

_BASE_MAP: Final[dict[int, int]] = {
    ord("A"): 0,
    ord("a"): 0,
    ord("C"): 1,
    ord("c"): 1,
    ord("G"): 2,
    ord("g"): 2,
    ord("T"): 3,
    ord("t"): 3,
}

_COMP_BASE_MAP: Final[dict[int, int]] = {
    ord("A"): 3,
    ord("a"): 3,
    ord("C"): 2,
    ord("c"): 2,
    ord("G"): 1,
    ord("g"): 1,
    ord("T"): 0,
    ord("t"): 0,
}


def encode_kmer_4(seq: str | bytes, reverse_complement: bool = False) -> int | None:
    """Encode a 4-mer nucleotide sequence into a base-4 integer in [0, 255]."""
    if len(seq) != 4:
        return None

    raw_bytes = seq.encode("ascii") if isinstance(seq, str) else seq
    mapping = _COMP_BASE_MAP if reverse_complement else _BASE_MAP

    if reverse_complement:
        b0 = mapping.get(raw_bytes[3])
        b1 = mapping.get(raw_bytes[2])
        b2 = mapping.get(raw_bytes[1])
        b3 = mapping.get(raw_bytes[0])
    else:
        b0 = mapping.get(raw_bytes[0])
        b1 = mapping.get(raw_bytes[1])
        b2 = mapping.get(raw_bytes[2])
        b3 = mapping.get(raw_bytes[3])

    if b0 is None or b1 is None or b2 is None or b3 is None:
        return None

    return (b0 << 6) | (b1 << 4) | (b2 << 2) | b3


def compute_motif_diversity_score(motif_counts: NDArray[np.number]) -> float:
    """Compute the normalized Shannon entropy (MDS) over 256 5' end motifs in [0, 1]."""
    counts = np.asarray(motif_counts, dtype=np.float64)
    total = np.sum(counts)
    if total <= 0.0:
        return 0.0

    p = counts / total
    nonzero = p > 0.0
    if not np.any(nonzero):
        return 0.0

    entropy = -np.sum(p[nonzero] * np.log(p[nonzero]))
    max_entropy = np.log(256.0)
    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


def compute_helical_periodicity(hist: NDArray[np.number]) -> tuple[float, float]:
    """Evaluate 10.4 bp helical cleavage periodicity via DFT in the sub-nucleosomal band."""
    h = np.asarray(hist, dtype=np.float64)
    idx_start = 80 - FRAG_MIN_LEN
    idx_end = 155 - FRAG_MIN_LEN
    if len(h) < idx_end + 1:
        return 0.0, 0.0

    total = float(np.sum(h))
    if total <= 0.0:
        return 0.0, 0.0

    p = h / total
    sub_p = p[idx_start : idx_end + 1]

    detrended = signal.detrend(sub_p)
    lengths = np.arange(80, 156, dtype=np.float64)
    f_helix = 1.0 / 10.4

    fourier_comp = np.sum(detrended * np.exp(-2.0j * np.pi * f_helix * lengths))
    amp = float(np.abs(fourier_comp) * (2.0 / len(sub_p)))
    phase = float(np.angle(fourier_comp))
    return amp, phase


def align_histogram(hist: NDArray[np.number], shift_bp: float) -> NDArray[np.float64]:
    """Sub-pixel linear shift of a 1-bp histogram onto the canonical frame."""
    h = np.asarray(hist, dtype=np.float64)
    if abs(shift_bp) < 1e-6 or np.sum(h) <= 0.0:
        return h.copy()

    n = len(h)
    x = np.arange(n, dtype=np.float64)
    aligned = np.interp(x + shift_bp, x, h, left=0.0, right=0.0)

    total_orig = np.sum(h)
    total_new = np.sum(aligned)
    if total_new > 0.0 and total_orig > 0.0:
        aligned *= total_orig / total_new

    return aligned


def estimate_sample_landmarks(
    hist_autosomal: NDArray[np.number],
    hist_chry: NDArray[np.number],
    hist_chrx: NDArray[np.number],
    end_motif_counts: NDArray[np.number],
    paired_end: bool,
) -> FragmentomicsMetadata:
    """Extract sample structural landmarks, helical periodicity, and pre-analytical QC."""
    auto = np.asarray(hist_autosomal, dtype=np.int64)
    chry = np.asarray(hist_chry, dtype=np.int64)
    chrx = np.asarray(hist_chrx, dtype=np.int64)
    motifs = np.asarray(end_motif_counts, dtype=np.int64)

    total_paired = int(np.sum(auto))
    if not paired_end or total_paired == 0:
        return FragmentomicsMetadata(
            paired_end=paired_end,
            total_paired_fragments=total_paired,
            chromatosome_mode_bp=CANONICAL_CHROMATOSOME_BP,
            core_shoulder_bp=143.0,
            sizing_shift_bp=0.0,
            self_referenced_peak_ratio=0.0,
            dinucleosome_fraction=0.0,
            high_mw_fraction=0.0,
            helical_10bp_amplitude=0.0,
            helical_10bp_phase=0.0,
            motif_diversity_score=compute_motif_diversity_score(motifs),
            hist_autosomal=auto.tolist(),
            hist_chry=chry.tolist(),
            hist_chrx=chrx.tolist(),
            end_motif_counts=motifs.tolist(),
        )

    smoothed = ndimage.gaussian_filter1d(auto.astype(np.float64), sigma=2.0)

    idx_155 = 155 - FRAG_MIN_LEN
    idx_180 = 180 - FRAG_MIN_LEN
    local_argmax = int(np.argmax(smoothed[idx_155 : idx_180 + 1]))
    peak_idx = idx_155 + local_argmax

    if 0 < peak_idx < len(smoothed) - 1:
        y0 = smoothed[peak_idx - 1]
        y1 = smoothed[peak_idx]
        y2 = smoothed[peak_idx + 1]
        denom = 2.0 * (y0 - 2.0 * y1 + y2)
        if denom < -1e-9:
            delta = (y0 - y2) / denom
            sub_bin = peak_idx + np.clip(delta, -1.0, 1.0)
        else:
            sub_bin = float(peak_idx)
    else:
        sub_bin = float(peak_idx)

    chrom_mode = float(np.clip(FRAG_MIN_LEN + sub_bin, 155.0, 180.0))
    sizing_shift = float(chrom_mode - CANONICAL_CHROMATOSOME_BP)

    idx_135 = 135 - FRAG_MIN_LEN
    idx_150 = 150 - FRAG_MIN_LEN
    core_argmax = int(np.argmax(smoothed[idx_135 : idx_150 + 1]))
    core_shoulder = float(FRAG_MIN_LEN + idx_135 + core_argmax)

    idx_160 = 160 - FRAG_MIN_LEN
    idx_175 = 175 - FRAG_MIN_LEN
    sum_core = float(np.sum(auto[idx_135 : idx_150 + 1]))
    sum_chrom = float(np.sum(auto[idx_160 : idx_175 + 1]))
    self_ref_ratio = float(sum_core / max(sum_chrom, 1.0))

    idx_250 = 250 - FRAG_MIN_LEN
    high_mw = float(np.sum(auto[idx_250:]) / max(total_paired, 1))

    idx_280 = 280 - FRAG_MIN_LEN
    idx_380 = 380 - FRAG_MIN_LEN
    dinucl = float(np.sum(auto[idx_280 : idx_380 + 1]) / max(total_paired, 1))

    helical_amp, helical_phase = compute_helical_periodicity(auto)
    mds = compute_motif_diversity_score(motifs)

    return FragmentomicsMetadata(
        paired_end=True,
        total_paired_fragments=total_paired,
        chromatosome_mode_bp=chrom_mode,
        core_shoulder_bp=core_shoulder,
        sizing_shift_bp=sizing_shift,
        self_referenced_peak_ratio=self_ref_ratio,
        dinucleosome_fraction=dinucl,
        high_mw_fraction=high_mw,
        helical_10bp_amplitude=helical_amp,
        helical_10bp_phase=helical_phase,
        motif_diversity_score=mds,
        hist_autosomal=auto.tolist(),
        hist_chry=chry.tolist(),
        hist_chrx=chrx.tolist(),
        end_motif_counts=motifs.tolist(),
    )


def deconvolve_fetal_maternal_em(
    hist_auto: NDArray[np.float64],
    hist_chry: NDArray[np.float64],
    f_estimates: NDArray[np.float64],
    is_male: NDArray[np.bool_],
    is_female: NDArray[np.bool_],
    *,
    smooth_sigma: float = 1.2,
    em_steps: int = 8,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cohort deconvolution of pure fetal and maternal cfDNA fragment length densities."""
    lengths = np.arange(FRAG_MIN_LEN, FRAG_MAX_LEN + 1, dtype=np.float64)
    auto = np.asarray(hist_auto, dtype=np.float64)
    chry = np.asarray(hist_chry, dtype=np.float64)
    ff = np.asarray(f_estimates, dtype=np.float64)
    male_arr = np.asarray(is_male, dtype=bool)
    female_arr = np.asarray(is_female, dtype=bool)

    n_samples = auto.shape[0] if auto.ndim == 2 else 0
    if n_samples == 0:
        param_f = np.exp(-0.5 * ((lengths - 143.0) / 10.0) ** 2)
        param_m = np.exp(-0.5 * ((lengths - 166.0) / 12.0) ** 2)
        return param_f / np.sum(param_f), param_m / np.sum(param_m)

    n_reads = np.sum(auto, axis=1)
    eligible_males = male_arr & np.isfinite(ff) & (ff >= 0.02)
    if np.any(eligible_males) and np.sum(chry[eligible_males]) > 0.0:
        h_male_y = np.sum(chry[eligible_males], axis=0)
        h_female_y = (
            np.sum(chry[female_arr], axis=0)
            if np.any(female_arr)
            else np.zeros(FRAG_NUM_BINS, dtype=np.float64)
        )
        s_male_y = ndimage.gaussian_filter1d(h_male_y, sigma=smooth_sigma)
        s_female_y = ndimage.gaussian_filter1d(h_female_y, sigma=smooth_sigma)

        sum_male_n = float(np.sum((1.0 - ff[eligible_males]) * n_reads[eligible_males]))
        sum_female_n = float(np.sum(n_reads[female_arr])) if np.any(female_arr) else 0.0
        bg_scale = sum_male_n / max(sum_female_n, 1.0) if sum_female_n > 0.0 else 0.0

        h_f0 = np.maximum(s_male_y - bg_scale * s_female_y, 0.0)
        sum_f0 = float(np.sum(h_f0))
        if sum_f0 > 0.0:
            p_f0 = h_f0 / sum_f0
        else:
            param_f = np.exp(-0.5 * ((lengths - 143.0) / 10.0) ** 2)
            p_f0 = param_f / np.sum(param_f)
    else:
        param_f = np.exp(-0.5 * ((lengths - 143.0) / 10.0) ** 2)
        p_f0 = param_f / np.sum(param_f)

    h_m0 = np.zeros(FRAG_NUM_BINS, dtype=np.float64)
    for i in np.flatnonzero(eligible_males):
        f_s = float(np.clip(ff[i], 0.002, 0.55))
        n_s = float(n_reads[i])
        if n_s > 0.0:
            p_s = auto[i] / n_s
            h_m0 += n_s * np.maximum(p_s - f_s * p_f0, 0.0)

    sum_m0 = float(np.sum(h_m0))
    if sum_m0 > 0.0:
        p_m0 = h_m0 / sum_m0
    else:
        param_m = np.exp(-0.5 * ((lengths - 166.0) / 12.0) ** 2)
        p_m0 = param_m / np.sum(param_m)

    p_f = ndimage.gaussian_filter1d(p_f0, sigma=1.0)
    p_f = np.maximum(p_f, 0.0)
    p_f /= np.sum(p_f)

    p_m = ndimage.gaussian_filter1d(p_m0, sigma=1.0)
    p_m = np.maximum(p_m, 0.0)
    p_m /= np.sum(p_m)

    f_curr = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        if eligible_males[i]:
            f_curr[i] = float(np.clip(ff[i], 0.002, 0.55))
        else:
            f_curr[i] = 0.05

    for _ in range(max(1, em_steps)):
        for i in range(n_samples):
            if eligible_males[i] or n_reads[i] <= 0.0:
                continue
            h_i = auto[i]

            def _neg_ll(
                f_val: float,
                h_target: NDArray[np.float64] = h_i,
                pf_step: NDArray[np.float64] = p_f,
                pm_step: NDArray[np.float64] = p_m,
            ) -> float:
                p_mix = f_val * pf_step + (1.0 - f_val) * pm_step
                return -float(np.sum(h_target * np.log(np.maximum(p_mix, 1e-12))))

            res = optimize.minimize_scalar(
                _neg_ll, bounds=(0.002, 0.55), method="bounded"
            )
            f_curr[i] = float(np.clip(res.x, 0.002, 0.55))

        mix = (
            f_curr[:, np.newaxis] * p_f[np.newaxis, :]
            + (1.0 - f_curr[:, np.newaxis]) * p_m[np.newaxis, :]
        )
        gamma_f = (f_curr[:, np.newaxis] * p_f[np.newaxis, :]) / np.maximum(mix, 1e-12)
        h_f_acc = np.sum(auto * gamma_f, axis=0)
        h_m_acc = np.sum(auto * (1.0 - gamma_f), axis=0)

        if np.sum(h_f_acc) > 0.0:
            s_f = ndimage.gaussian_filter1d(h_f_acc, sigma=1.0)
            p_f = np.maximum(s_f, 0.0)
            p_f /= np.sum(p_f)
        if np.sum(h_m_acc) > 0.0:
            s_m = ndimage.gaussian_filter1d(h_m_acc, sigma=1.0)
            p_m = np.maximum(s_m, 0.0)
            p_m /= np.sum(p_m)

    return p_f, p_m


def deconvolve_fetal_maternal_densities(
    aligned_auto_hists: NDArray[np.number],
    aligned_chry_hists: NDArray[np.number],
    male_mask: NDArray[np.bool_],
    ffy_values: NDArray[np.number],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cohort deconvolution of pure fetal and maternal cfDNA fragment length densities."""
    auto = np.asarray(aligned_auto_hists, dtype=np.float64)
    chry = np.asarray(aligned_chry_hists, dtype=np.float64)
    males = np.asarray(male_mask, dtype=bool)
    ff = np.asarray(ffy_values, dtype=np.float64)
    females = ~males
    return deconvolve_fetal_maternal_em(
        auto, chry, ff, males, females, smooth_sigma=1.2, em_steps=8
    )


def predict_ff_kl_deconvolution(
    hist_sample: NDArray[np.float64],
    p_fetal: NDArray[np.float64],
    p_maternal: NDArray[np.float64],
) -> tuple[float, float]:
    """Zero-parameter maximum likelihood estimation of fetal fraction via KL deconvolution."""
    h = np.asarray(hist_sample, dtype=np.float64)
    pf = np.asarray(p_fetal, dtype=np.float64)
    pm = np.asarray(p_maternal, dtype=np.float64)

    total = float(np.sum(h))
    if total <= 0.0 or pf.size != FRAG_NUM_BINS or pm.size != FRAG_NUM_BINS:
        return 0.0, 0.05

    def _neg_ll(f: float) -> float:
        mix = f * pf + (1.0 - f) * pm
        return -float(np.sum(h * np.log(np.maximum(mix, 1e-12))))

    res = optimize.minimize_scalar(_neg_ll, bounds=(0.002, 0.55), method="bounded")
    hat_f = float(np.clip(res.x, 0.002, 0.55))

    delta_p = pf - pm
    p_mix_hat = hat_f * pf + (1.0 - hat_f) * pm
    score_sq = (delta_p / np.maximum(p_mix_hat, 1e-12)) ** 2
    obs_info = float(np.sum(h * score_sq))

    sigma_kl = float(1.0 / math.sqrt(obs_info)) if obs_info > 1e-9 else 0.05
    return hat_f, sigma_kl


def compute_band_fisher_weights(
    p_fetal_571: NDArray[np.float64],
    p_maternal_571: NDArray[np.float64],
    *,
    f0: float = 0.08,
) -> tuple[NDArray[np.float64], float, float]:
    """Derive Fisher-optimal read weights mapped to 24 discrete fragmentomic bands."""
    pf = np.asarray(p_fetal_571, dtype=np.float64)
    pm = np.asarray(p_maternal_571, dtype=np.float64)
    edges = np.asarray(FRAG_BAND_EDGES, dtype=np.int64)
    n_bands = len(edges) - 1

    pf_24 = np.zeros(n_bands, dtype=np.float64)
    pm_24 = np.zeros(n_bands, dtype=np.float64)
    for k in range(n_bands):
        i_s = int(edges[k]) - FRAG_MIN_LEN
        i_e = int(edges[k + 1]) - FRAG_MIN_LEN
        pf_24[k] = float(np.sum(pf[i_s:i_e]))
        pm_24[k] = float(np.sum(pm[i_s:i_e]))

    sum_pf = float(np.sum(pf_24))
    if sum_pf > 0.0:
        pf_24 /= sum_pf
    sum_pm = float(np.sum(pm_24))
    if sum_pm > 0.0:
        pm_24 /= sum_pm

    eff_f0 = float(np.clip(f0, 0.01, 0.40))
    p_mix = eff_f0 * pf_24 + (1.0 - eff_f0) * pm_24
    raw_weights = pf_24 / np.maximum(p_mix, 1e-12)

    low, high = 0.05, 20.0
    for _ in range(30):
        mid = 0.5 * (low + high)
        val = float(np.sum(pm_24 * np.clip(mid * raw_weights, 0.25, 3.0)))
        if val < 1.0:
            low = mid
        else:
            high = mid

    alpha = 0.5 * (low + high)
    w_24 = np.clip(alpha * raw_weights, 0.25, 3.0)

    mu_m_w = float(np.sum(pm_24 * w_24))
    mu_f_w = float(np.sum(pf_24 * w_24))
    return w_24, mu_f_w, mu_m_w


def evaluate_band_weighted_counts(
    frag_bands: NDArray[np.uint32],
    raw_counts: NDArray[np.uint32] | NDArray[np.float64],
    paired_counts: NDArray[np.uint32] | NDArray[np.float64],
    band_weights_24: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
    """Evaluate Fisher-weighted coverage and per-bin Kish design factor from 24 frag_bands."""
    bands_f64 = np.asarray(frag_bands, dtype=np.float64)
    raw_f64 = np.asarray(raw_counts, dtype=np.float64)
    paired_f64 = np.asarray(paired_counts, dtype=np.float64)
    w_24 = np.asarray(band_weights_24, dtype=np.float64)

    w_paired = bands_f64 @ w_24
    raw_unpaired = np.maximum(raw_f64 - paired_f64, 0.0)
    w_counts = raw_unpaired + w_paired

    sum_w_sq = bands_f64 @ (w_24**2)
    w_paired_sq = np.maximum(w_paired**2, 1e-12)
    has_paired = paired_f64 > 0.0

    kappa = np.ones(raw_f64.shape, dtype=np.float32)
    kappa[has_paired] = np.maximum(
        paired_f64[has_paired] * sum_w_sq[has_paired] / w_paired_sq[has_paired],
        1.0,
    ).astype(np.float32)

    return w_counts, kappa


def build_fragmentomic_feature_vector(
    meta: FragmentomicsMetadata,
) -> NDArray[np.float64]:
    """Extract a 36-dimensional feature vector for the regularized FF_size ensemble."""
    auto = np.asarray(meta.hist_autosomal, dtype=np.float64)
    total = np.sum(auto)
    if total <= 0.0:
        return np.zeros(36, dtype=np.float64)

    p = auto / total
    features: list[float] = []

    features.append(float(meta.chromatosome_mode_bp))

    idx_164 = 164 - FRAG_MIN_LEN
    idx_168 = 168 - FRAG_MIN_LEN
    ref_density = float(np.mean(p[idx_164 : idx_168 + 1]))
    ref_safe = max(ref_density, 1e-8)

    for start_bp in range(85, 220, 5):
        i_s = start_bp - FRAG_MIN_LEN
        i_e = i_s + 5
        band_density = float(np.mean(p[i_s:i_e]))
        features.append(math.log(max(band_density, 1e-8) / ref_safe))

    features.append(float(meta.self_referenced_peak_ratio))

    for bp in (112, 122, 133, 143):
        idx = bp - FRAG_MIN_LEN
        features.append(float(p[idx] / ref_safe))

    features.append(float(meta.helical_10bp_amplitude))
    features.append(float(meta.motif_diversity_score))

    motifs = np.asarray(meta.end_motif_counts, dtype=np.float64)
    sum_motifs = max(float(np.sum(motifs)), 1.0)
    top_indices = [
        encode_kmer_4("CCCA") or 84,
        encode_kmer_4("CCAG") or 82,
        encode_kmer_4("CCTG") or 94,
    ]
    top_ratio = float(np.sum(motifs[top_indices]) / sum_motifs)
    features.append(top_ratio)

    return np.asarray(features, dtype=np.float64)


def fit_ff_fragmentomics_ensemble(
    features: NDArray[np.number],
    ff_targets: NDArray[np.number],
) -> tuple[NDArray[np.float64], float, NDArray[np.float64], NDArray[np.float64], float]:
    """Train a convex logit-link RidgeCV + ElasticNetCV ensemble for FF_size."""
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(ff_targets, dtype=np.float64)
    n_samples = X.shape[0]
    if n_samples < 5:
        raise ValueError(
            f"fit_ff_fragmentomics_ensemble requires at least 5 samples, got {n_samples}"
        )

    mean = np.mean(X, axis=0)
    scale = np.std(X, axis=0)
    scale[scale < 1e-6] = 1.0
    Z = (X - mean) / scale

    y_clipped = np.clip(y, 0.005, 0.85)
    u = np.log(y_clipped / (1.0 - y_clipped))
    ridge_alphas = np.logspace(-2, 4, 20)
    enet_alphas = np.logspace(-3, 2, 20)

    weights, bias = fit_logit_linear_ensemble(
        Z,
        u,
        ridge_alphas=ridge_alphas,
        l1_ratio=(0.1, 0.5, 0.9),
        enet_alphas=enet_alphas,
        seed=42,
        max_iter=2000,
    )
    oof_pred = np.empty(n_samples, dtype=np.float64)
    kfold = KFold(n_splits=min(5, n_samples), shuffle=True, random_state=42)
    for train_idx, val_idx in kfold.split(X):
        f_mean = np.mean(X[train_idx], axis=0)
        f_scale = np.std(X[train_idx], axis=0)
        f_scale[f_scale < 1e-6] = 1.0
        Z_tr = (X[train_idx] - f_mean) / f_scale
        Z_va = (X[val_idx] - f_mean) / f_scale
        w_fold, b_fold = fit_logit_linear_ensemble(
            Z_tr,
            u[train_idx],
            ridge_alphas=ridge_alphas,
            l1_ratio=(0.1, 0.5, 0.9),
            enet_alphas=enet_alphas,
            seed=42,
            max_iter=2000,
        )
        oof_pred[val_idx] = 1.0 / (1.0 + np.exp(-(Z_va @ w_fold + b_fold)))

    loocv_std = float(max(np.std(y - oof_pred), 0.015))
    return weights, bias, mean, scale, loocv_std


def predict_ff_fragmentomics(
    meta: FragmentomicsMetadata,
    model: PrenatalModelMetadata,
) -> float | None:
    """Predict fetal fraction from fragmentomics metadata via logit ensemble or KL MLE."""
    if not meta.paired_end:
        return None

    f_kl: float | None = None
    if (
        model.p_fetal_density is not None
        and model.p_maternal_density is not None
        and len(meta.hist_autosomal) == FRAG_NUM_BINS
    ):
        f_val, _ = predict_ff_kl_deconvolution(
            np.asarray(meta.hist_autosomal, dtype=np.float64),
            np.asarray(model.p_fetal_density, dtype=np.float64),
            np.asarray(model.p_maternal_density, dtype=np.float64),
        )
        f_kl = f_val

    f_ens: float | None = None
    if (
        model.ff_frag_weights is not None
        and model.ff_frag_feature_mean is not None
        and model.ff_frag_feature_scale is not None
    ):
        feat = build_fragmentomic_feature_vector(meta)
        mean = np.asarray(model.ff_frag_feature_mean, dtype=np.float64)
        scale = np.asarray(model.ff_frag_feature_scale, dtype=np.float64)
        weights = np.asarray(model.ff_frag_weights, dtype=np.float64)
        bias = float(model.ff_frag_bias or 0.0)

        z = (feat - mean) / scale
        u = float(np.dot(z, weights) + bias)
        f_ens = 1.0 / (1.0 + math.exp(-u))

    if f_kl is not None and f_ens is not None:
        f_final = 0.5 * f_kl + 0.5 * f_ens
    elif f_kl is not None:
        f_final = f_kl
    elif f_ens is not None:
        f_final = f_ens
    else:
        return None

    return float(np.clip(f_final, 0.005, 0.50))


__all__ = [
    "CANONICAL_CHROMATOSOME_BP",
    "FRAG_MAX_LEN",
    "FRAG_MIN_LEN",
    "FRAG_NUM_BINS",
    "align_histogram",
    "build_fragmentomic_feature_vector",
    "compute_band_fisher_weights",
    "compute_helical_periodicity",
    "compute_motif_diversity_score",
    "deconvolve_fetal_maternal_densities",
    "deconvolve_fetal_maternal_em",
    "encode_kmer_4",
    "estimate_sample_landmarks",
    "evaluate_band_weighted_counts",
    "fit_ff_fragmentomics_ensemble",
    "predict_ff_fragmentomics",
    "predict_ff_kl_deconvolution",
]
