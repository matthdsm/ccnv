"""Stateless mathematical and machine learning kernels for cfDNA fragmentomics."""

from __future__ import annotations

import math
import warnings
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage, signal
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import KFold

from ccnv.models.schemas import FragmentomicsMetadata, PrenatalModelMetadata

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
    ord("A"): 3,  # comp is T
    ord("a"): 3,
    ord("C"): 2,  # comp is G
    ord("c"): 2,
    ord("G"): 1,  # comp is C
    ord("g"): 1,
    ord("T"): 0,  # comp is A
    ord("t"): 0,
}


def encode_kmer_4(seq: str | bytes, reverse_complement: bool = False) -> int | None:
    """Encode a 4-mer nucleotide sequence into a base-4 integer in [0, 255]."""
    if len(seq) != 4:
        return None

    raw_bytes = seq.encode("ascii") if isinstance(seq, str) else seq
    mapping = _COMP_BASE_MAP if reverse_complement else _BASE_MAP

    if reverse_complement:
        # Reverse complement of s[0]s[1]s[2]s[3] is comp(s[3]) comp(s[2]) comp(s[1]) comp(s[0])
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
    # Band [80, 155] bp corresponds to indices [80 - 30, 155 - 30] = [50, 125]
    idx_start = 80 - FRAG_MIN_LEN
    idx_end = 155 - FRAG_MIN_LEN
    if len(h) < idx_end + 1:
        return 0.0, 0.0

    sub_hist = h[idx_start : idx_end + 1]
    if np.sum(sub_hist) <= 0.0:
        return 0.0, 0.0

    # Detrend linear background
    detrended = signal.detrend(sub_hist)
    lengths = np.arange(80, 156, dtype=np.float64)
    f_helix = 1.0 / 10.4

    # Discrete Fourier component at f_helix
    fourier_comp = np.sum(detrended * np.exp(-2.0j * np.pi * f_helix * lengths))
    amp = float(np.abs(fourier_comp) * (2.0 / len(sub_hist)))
    phase = float(np.angle(fourier_comp))
    return amp, phase


def align_histogram(hist: NDArray[np.number], shift_bp: float) -> NDArray[np.float64]:
    """Sub-pixel linear shift of a 1-bp histogram onto the canonical frame."""
    h = np.asarray(hist, dtype=np.float64)
    if abs(shift_bp) < 1e-6 or np.sum(h) <= 0.0:
        return h.copy()

    n = len(h)
    x = np.arange(n, dtype=np.float64)
    # Target index x samples from source index x + shift_bp (shifting by +shift_bp moves peak to lower index)
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

    # Gaussian smoothing with sigma=2.0 bins
    smoothed = ndimage.gaussian_filter1d(auto.astype(np.float64), sigma=2.0)

    # Chromatosome mode search in [155, 180] bp -> indices [125, 150]
    idx_155 = 155 - FRAG_MIN_LEN
    idx_180 = 180 - FRAG_MIN_LEN
    local_argmax = int(np.argmax(smoothed[idx_155 : idx_180 + 1]))
    peak_idx = idx_155 + local_argmax

    # Parabolic sub-bin interpolation
    if 0 < peak_idx < len(smoothed) - 1:
        y0 = smoothed[peak_idx - 1]
        y1 = smoothed[peak_idx]
        y2 = smoothed[peak_idx + 1]
        denom = 2.0 * (y0 - 2.0 * y1 + y2)
        if abs(denom) > 1e-9:
            delta = (y0 - y2) / denom
            sub_bin = peak_idx + np.clip(delta, -1.0, 1.0)
        else:
            sub_bin = float(peak_idx)
    else:
        sub_bin = float(peak_idx)

    chrom_mode = float(np.clip(FRAG_MIN_LEN + sub_bin, 155.0, 180.0))
    sizing_shift = float(chrom_mode - CANONICAL_CHROMATOSOME_BP)

    # Core shoulder search in [135, 150] bp -> indices [105, 120]
    idx_135 = 135 - FRAG_MIN_LEN
    idx_150 = 150 - FRAG_MIN_LEN
    core_argmax = int(np.argmax(smoothed[idx_135 : idx_150 + 1]))
    core_shoulder = float(FRAG_MIN_LEN + idx_135 + core_argmax)

    # Self-referenced peak ratio: sum[135..150] / sum[160..175]
    idx_160 = 160 - FRAG_MIN_LEN
    idx_175 = 175 - FRAG_MIN_LEN
    sum_core = float(np.sum(auto[idx_135 : idx_150 + 1]))
    sum_chrom = float(np.sum(auto[idx_160 : idx_175 + 1]))
    self_ref_ratio = float(sum_core / max(sum_chrom, 1.0))

    # High MW fraction: sum[250..600] / total
    idx_250 = 250 - FRAG_MIN_LEN
    high_mw = float(np.sum(auto[idx_250:]) / max(total_paired, 1))

    # Dinucleosome fraction: sum[280..380] / total
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


def deconvolve_fetal_maternal_densities(
    aligned_auto_hists: NDArray[np.number],
    aligned_chry_hists: NDArray[np.number],
    male_mask: NDArray[np.bool_],
    ffy_values: NDArray[np.number],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cohort deconvolution of pure fetal and maternal cfDNA fragment length densities."""
    lengths = np.arange(FRAG_MIN_LEN, FRAG_MAX_LEN + 1, dtype=np.float64)
    male_arr = np.asarray(male_mask, dtype=bool)
    ff = np.asarray(ffy_values, dtype=np.float64)
    eligible_males = male_arr & np.isfinite(ff) & (ff >= 0.08)
    female_mask = ~male_arr
    n_eligible_males = int(np.sum(eligible_males))
    n_females = int(np.sum(female_mask))

    chry_arr = np.asarray(aligned_chry_hists, dtype=np.float64)
    auto_arr = np.asarray(aligned_auto_hists, dtype=np.float64)

    male_chry_reads = (
        float(np.sum(chry_arr[eligible_males])) if n_eligible_males > 0 else 0.0
    )

    # 1. Pure Fetal Density p_F(L)
    if n_eligible_males >= 5 and male_chry_reads >= 500.0:
        pooled_male = np.sum(chry_arr[eligible_males], axis=0)
        pooled_female = (
            np.sum(chry_arr[female_mask], axis=0)
            if n_females > 0
            else np.zeros(FRAG_NUM_BINS)
        )
        bg_scale = float(n_eligible_males) / max(float(n_females), 1.0)
        h_fetal = np.maximum(pooled_male - bg_scale * pooled_female, 0.0)
        # Smooth with Gaussian sigma=1.5
        smoothed_f = ndimage.gaussian_filter1d(h_fetal, sigma=1.5)
        sum_f = np.sum(smoothed_f)
        p_fetal = (
            smoothed_f / sum_f
            if sum_f > 0.0
            else np.ones(FRAG_NUM_BINS) / FRAG_NUM_BINS
        )
    else:
        # Fallback to canonical parametric fetal template
        param_f = (
            np.exp(-0.5 * ((lengths - 143.0) / 10.0) ** 2)
            * (1.0 + 0.15 * np.cos(2.0 * np.pi * (lengths - 143.0) / 10.4))
            + 0.15 * np.exp(-0.5 * ((lengths - 133.0) / 8.0) ** 2)
            + 0.05 * np.exp(-0.5 * ((lengths - 123.0) / 6.0) ** 2)
        )
        param_f = np.maximum(param_f, 0.0)
        p_fetal = param_f / np.sum(param_f)

    # 2. Pure Maternal Density p_M(L)
    # Autosomal normalized densities per sample
    row_sums = np.sum(auto_arr, axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    norm_auto = auto_arr / row_sums

    if len(ff) >= 5 and np.ptp(ff) > 0.03:
        # Fit OLS across samples for each bin to find intercept at f=0
        # norm_auto ~ beta_1 * ff + beta_0
        ff_centered = ff - np.mean(ff)
        var_ff = float(np.sum(ff_centered**2))
        if var_ff > 1e-6:
            slope = np.sum(norm_auto * ff_centered[:, np.newaxis], axis=0) / var_ff
            intercept = np.mean(norm_auto, axis=0) - slope * np.mean(ff)
            p_mat_raw = np.maximum(intercept, 0.0)
        else:
            p_mat_raw = np.median(norm_auto, axis=0)
    else:
        p_mat_raw = (
            np.median(norm_auto, axis=0)
            if len(auto_arr) > 0
            else np.zeros(FRAG_NUM_BINS)
        )

    if np.sum(p_mat_raw) <= 0.0:
        param_m = np.exp(-0.5 * ((lengths - 166.0) / 12.0) ** 2)
        p_mat_raw = param_m

    smoothed_m = ndimage.gaussian_filter1d(p_mat_raw, sigma=1.0)
    sum_m = np.sum(smoothed_m)
    p_maternal = (
        smoothed_m / sum_m if sum_m > 0.0 else np.ones(FRAG_NUM_BINS) / FRAG_NUM_BINS
    )

    return p_fetal, p_maternal


def fisher_optimal_weights(
    p_fetal: NDArray[np.number],
    p_maternal: NDArray[np.number],
    target_ff: float = 0.10,
    *,
    f0: float | None = None,
    clip_bounds: tuple[float, float] = (0.25, 3.0),
) -> NDArray[np.float64]:
    """Calculate parameter-free Fisher-score optimal coverage weights w*(L; f0)."""
    effective_f0 = f0 if f0 is not None else target_ff
    pf = np.asarray(p_fetal, dtype=np.float64)
    pm = np.asarray(p_maternal, dtype=np.float64)

    p_mix = effective_f0 * pf + (1.0 - effective_f0) * pm
    p_mix_safe = np.maximum(p_mix, 1e-12)

    raw_weights = pf / p_mix_safe
    c_min, c_max = clip_bounds

    # Solve for scaling factor alpha such that E_{p_mix}[clip(alpha * raw_weights)] == 1.0
    low, high = 0.05, 20.0
    for _ in range(25):
        mid = 0.5 * (low + high)
        val = float(np.sum(p_mix * np.clip(mid * raw_weights, c_min, c_max)))
        if val < 1.0:
            low = mid
        else:
            high = mid

    alpha = 0.5 * (low + high)
    return np.clip(alpha * raw_weights, c_min, c_max)


def build_fragmentomic_feature_vector(
    meta: FragmentomicsMetadata,
) -> NDArray[np.float64]:
    """Extract a 36-dimensional feature vector for the regularized FF_size ensemble."""
    auto = np.asarray(meta.hist_autosomal, dtype=np.float64)
    total = np.sum(auto)
    if total <= 0.0:
        return np.zeros(36, dtype=np.float64)

    p = auto / total
    # Align histogram by sizing shift
    p_aligned = align_histogram(p, meta.sizing_shift_bp)

    features: list[float] = []

    # 1. 28 mode-aligned 5 bp log-density bands in [80, 220] bp
    # Canonical chromatosome ref density around 166 bp ([164, 168] bp)
    idx_164 = 164 - FRAG_MIN_LEN
    idx_168 = 168 - FRAG_MIN_LEN
    ref_density = float(np.mean(p_aligned[idx_164 : idx_168 + 1]))
    ref_safe = max(ref_density, 1e-8)

    for start_bp in range(80, 220, 5):
        i_s = start_bp - FRAG_MIN_LEN
        i_e = i_s + 5
        band_density = float(np.mean(p_aligned[i_s:i_e]))
        features.append(math.log(max(band_density, 1e-8) / ref_safe))

    # 2. Self-referenced peak ratio
    features.append(float(meta.self_referenced_peak_ratio))

    # 3. Sub-nucleosomal peak ratios: densities at 112, 122, 133, 143 vs 166 bp (4 features)
    for bp in (112, 122, 133, 143):
        idx = bp - FRAG_MIN_LEN
        features.append(float(p_aligned[idx] / ref_safe))

    # 4. Helical periodicity amplitude (1 feature)
    features.append(float(meta.helical_10bp_amplitude))

    # 5. Motif Diversity Score (1 feature)
    features.append(float(meta.motif_diversity_score))

    # 6. Top cleavage 4-mer motif frequencies (CCCA, CCAG, CCTG) (1 feature)
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

    # Logit transform of target
    y_clipped = np.clip(y, 0.005, 0.85)
    u = np.log(y_clipped / (1.0 - y_clipped))
    ridge_alphas = np.logspace(-2, 4, 20)
    enet_alphas = np.logspace(-3, 2, 20)

    def _fit_ensemble_pair(
        z_mat: NDArray[np.float64], u_vec: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], float]:
        """Fit 50/50 RidgeCV + ElasticNetCV logit ensemble on standardized features."""
        r_est = RidgeCV(alphas=ridge_alphas)
        r_est.fit(z_mat, u_vec)
        e_est = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            alphas=enet_alphas,
            cv=min(5, z_mat.shape[0]),
            random_state=42,
            max_iter=2000,
        )
        e_est.fit(z_mat, u_vec)
        return (
            0.5 * r_est.coef_ + 0.5 * e_est.coef_,
            float(0.5 * r_est.intercept_ + 0.5 * e_est.intercept_),
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)

        weights, bias = _fit_ensemble_pair(Z, u)
        oof_pred = np.empty(n_samples, dtype=np.float64)
        kfold = KFold(n_splits=min(5, n_samples), shuffle=True, random_state=42)
        for train_idx, val_idx in kfold.split(X):
            f_mean = np.mean(X[train_idx], axis=0)
            f_scale = np.std(X[train_idx], axis=0)
            f_scale[f_scale < 1e-6] = 1.0
            Z_tr = (X[train_idx] - f_mean) / f_scale
            Z_va = (X[val_idx] - f_mean) / f_scale
            w_fold, b_fold = _fit_ensemble_pair(Z_tr, u[train_idx])
            oof_pred[val_idx] = 1.0 / (1.0 + np.exp(-(Z_va @ w_fold + b_fold)))

    loocv_std = float(max(np.std(y - oof_pred), 0.015))

    return weights, bias, mean, scale, loocv_std


def predict_ff_fragmentomics(
    meta: FragmentomicsMetadata,
    model: PrenatalModelMetadata,
) -> float | None:
    """Predict fetal fraction from fragmentomics metadata via logit ensemble."""
    if not meta.paired_end or model.ff_frag_weights is None:
        return None
    if model.ff_frag_feature_mean is None or model.ff_frag_feature_scale is None:
        return None

    feat = build_fragmentomic_feature_vector(meta)
    mean = np.asarray(model.ff_frag_feature_mean, dtype=np.float64)
    scale = np.asarray(model.ff_frag_feature_scale, dtype=np.float64)
    weights = np.asarray(model.ff_frag_weights, dtype=np.float64)
    bias = float(model.ff_frag_bias or 0.0)

    z = (feat - mean) / scale
    u = float(np.dot(z, weights) + bias)
    ff = 1.0 / (1.0 + math.exp(-u))
    return float(np.clip(ff, 0.005, 0.50))


def segment_fragment_llr(
    lengths: NDArray[np.number],
    shift_bp: float,
    p_fetal: NDArray[np.number],
    p_maternal: NDArray[np.number],
    active_ff: float,
    is_gain: bool,
) -> tuple[float, float]:
    """Calculate segment short-fragment ratio and exact per-fragment log-likelihood ratio."""
    if len(lengths) == 0:
        return 0.0, 0.0

    lens = np.asarray(lengths, dtype=np.float64)
    lens = lens[lens > 0.0]
    if lens.size == 0:
        return 0.0, 0.0
    short_ratio = float(np.mean(lens <= 150.0))

    # Shift fragment lengths onto canonical frame
    shifted_lens = np.clip(
        np.round(lens - shift_bp), FRAG_MIN_LEN, FRAG_MAX_LEN
    ).astype(int)
    indices = shifted_lens - FRAG_MIN_LEN

    pf = np.asarray(p_fetal, dtype=np.float64)
    pm = np.asarray(p_maternal, dtype=np.float64)

    f = float(np.clip(active_ff, 0.01, 0.40))
    if is_gain:
        f_fetal_cnv = (1.5 * f) / (1.0 + 0.5 * f)
        f_mat_cnv = f / (1.0 + 0.5 * (1.0 - f))
    else:
        f_fetal_cnv = (0.5 * f) / (1.0 - 0.5 * f)
        f_mat_cnv = f / (1.0 - 0.5 * (1.0 - f))

    p_f_eval = f_fetal_cnv * pf[indices] + (1.0 - f_fetal_cnv) * pm[indices]
    p_m_eval = f_mat_cnv * pf[indices] + (1.0 - f_mat_cnv) * pm[indices]

    llr = float(
        np.sum(np.log(np.maximum(p_f_eval, 1e-12) / np.maximum(p_m_eval, 1e-12)))
    )
    return short_ratio, llr


__all__ = [
    "CANONICAL_CHROMATOSOME_BP",
    "FRAG_MAX_LEN",
    "FRAG_MIN_LEN",
    "FRAG_NUM_BINS",
    "align_histogram",
    "build_fragmentomic_feature_vector",
    "compute_helical_periodicity",
    "compute_motif_diversity_score",
    "deconvolve_fetal_maternal_densities",
    "encode_kmer_4",
    "estimate_sample_landmarks",
    "fisher_optimal_weights",
    "fit_ff_fragmentomics_ensemble",
    "predict_ff_fragmentomics",
    "segment_fragment_llr",
]
