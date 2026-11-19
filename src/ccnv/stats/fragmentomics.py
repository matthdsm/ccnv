"""Stateless mathematical and machine learning kernels for cfDNA fragmentomics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import scipy.special
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage, optimize, signal

from ccnv._core import FRAG_BAND_EDGES
from ccnv.models.exceptions import InvalidInputError
from ccnv.stats.regression import LogitEnsembleFit, fit_logit_linear_ensemble

__all__ = [
    "CANONICAL_CHROMATOSOME_BP",
    "FRAG_MAX_LEN",
    "FRAG_MIN_LEN",
    "FRAG_NUM_BINS",
    "FragmentomicLandmarks",
    "align_histogram",
    "build_fragmentomic_feature_vector",
    "compute_band_fisher_weights",
    "compute_fragment_length_llr",
    "compute_helical_periodicity",
    "compute_motif_diversity_score",
    "deconvolve_fetal_maternal_densities",
    "encode_kmer_4",
    "estimate_sample_landmarks",
    "evaluate_band_weighted_counts",
    "fit_ff_fragmentomics_ensemble",
    "predict_ff_fragmentomics",
    "predict_ff_kl_deconvolution",
]

FRAG_MIN_LEN: Final[int] = 30
FRAG_MAX_LEN: Final[int] = 600
FRAG_NUM_BINS: Final[int] = 571
CANONICAL_CHROMATOSOME_BP: Final[float] = 166.0

_TOP_MOTIF_INDICES: Final[tuple[int, int, int]] = (84, 82, 94)
_SUBNUC_PEAK_OFFSETS: Final[tuple[int, int, int, int]] = (
    112 - FRAG_MIN_LEN,
    122 - FRAG_MIN_LEN,
    133 - FRAG_MIN_LEN,
    143 - FRAG_MIN_LEN,
)

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


@dataclass(frozen=True, slots=True)
class FragmentomicLandmarks:
    """Pure scalar sample-level cfDNA fragmentomic landmarks and QC metrics."""

    paired_end: bool
    total_paired_fragments: int
    chromatosome_mode_bp: float
    core_shoulder_bp: float
    sizing_shift_bp: float
    self_referenced_peak_ratio: float
    dinucleosome_fraction: float
    high_mw_fraction: float
    helical_10bp_amplitude: float
    helical_10bp_phase: float
    motif_diversity_score: float
    fetal_motif_ratio: float


def encode_kmer_4(seq: str | bytes) -> int | None:
    """Encode a 4-mer nucleotide sequence into a base-4 integer in [0, 255]."""
    if len(seq) != 4:
        return None

    if isinstance(seq, str):
        b0 = _BASE_MAP.get(ord(seq[0]))
        b1 = _BASE_MAP.get(ord(seq[1]))
        b2 = _BASE_MAP.get(ord(seq[2]))
        b3 = _BASE_MAP.get(ord(seq[3]))
    else:
        b0 = _BASE_MAP.get(seq[0])
        b1 = _BASE_MAP.get(seq[1])
        b2 = _BASE_MAP.get(seq[2])
        b3 = _BASE_MAP.get(seq[3])
    if b0 is None or b1 is None or b2 is None or b3 is None:
        return None
    return (b0 << 6) | (b1 << 4) | (b2 << 2) | b3


def compute_motif_diversity_score(motif_counts: ArrayLike) -> float:
    """Compute the normalized Shannon entropy (MDS) over 256 5' end motifs in [0, 1]."""
    counts = np.asarray(motif_counts, dtype=np.float64)
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0

    p = counts / total
    nonzero = p > 0.0
    if not np.any(nonzero):
        return 0.0

    entropy = -float(np.sum(p[nonzero] * np.log(p[nonzero])))
    return float(np.clip(entropy / math.log(256.0), 0.0, 1.0))


def _compute_fetal_motif_ratio(motif_counts: NDArray[np.float64]) -> float:
    """Compute pooled frequency of top fetal-associated 5' 4-mer motifs (CCCA, CCAG, CCTG)."""
    if motif_counts.size <= max(_TOP_MOTIF_INDICES):
        return 0.0
    total = float(np.sum(motif_counts))
    if total <= 0.0:
        return 0.0
    return float(np.sum(motif_counts[list(_TOP_MOTIF_INDICES)]) / total)


def compute_helical_periodicity(hist: ArrayLike) -> tuple[float, float]:
    """Evaluate 10.4 bp helical cleavage periodicity via DFT in the sub-nucleosomal band."""
    h = np.asarray(hist, dtype=np.float64)
    idx_start = 80 - FRAG_MIN_LEN
    idx_end = 155 - FRAG_MIN_LEN
    if h.size < idx_end + 1:
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


def align_histogram(hist: ArrayLike, shift_bp: float) -> NDArray[np.float64]:
    """Sub-pixel linear shift of a 1-bp histogram onto the canonical frame."""
    h = np.asarray(hist, dtype=np.float64).ravel()
    if abs(shift_bp) < 1e-6 or float(np.sum(h)) <= 0.0:
        return h.copy()

    n = h.size
    x = np.arange(n, dtype=np.float64)
    aligned = np.interp(x + shift_bp, x, h, left=0.0, right=0.0)

    total_orig = float(np.sum(h))
    total_new = float(np.sum(aligned))
    if total_new > 0.0 and total_orig > 0.0:
        aligned *= total_orig / total_new

    return aligned


def estimate_sample_landmarks(
    hist_autosomal: ArrayLike,
    end_motif_counts: ArrayLike,
    *,
    paired_end: bool = True,
) -> FragmentomicLandmarks:
    """Extract pure scalar structural landmarks, helical periodicity, and motif ratios."""
    auto = np.asarray(hist_autosomal, dtype=np.int64).ravel()
    motifs = np.asarray(end_motif_counts, dtype=np.float64).ravel()
    mds = compute_motif_diversity_score(motifs)
    fetal_motif = _compute_fetal_motif_ratio(motifs)

    total_paired = int(np.sum(auto))
    if not paired_end or total_paired == 0 or auto.size < FRAG_NUM_BINS:
        return FragmentomicLandmarks(
            paired_end=bool(paired_end),
            total_paired_fragments=total_paired,
            chromatosome_mode_bp=CANONICAL_CHROMATOSOME_BP,
            core_shoulder_bp=143.0,
            sizing_shift_bp=0.0,
            self_referenced_peak_ratio=0.0,
            dinucleosome_fraction=0.0,
            high_mw_fraction=0.0,
            helical_10bp_amplitude=0.0,
            helical_10bp_phase=0.0,
            motif_diversity_score=mds,
            fetal_motif_ratio=fetal_motif,
        )

    smoothed = ndimage.gaussian_filter1d(auto.astype(np.float64), sigma=2.0)

    idx_155 = 155 - FRAG_MIN_LEN
    idx_180 = 180 - FRAG_MIN_LEN
    local_argmax = int(np.argmax(smoothed[idx_155 : idx_180 + 1]))
    peak_idx = idx_155 + local_argmax

    if 0 < peak_idx < smoothed.size - 1:
        y0 = float(smoothed[peak_idx - 1])
        y1 = float(smoothed[peak_idx])
        y2 = float(smoothed[peak_idx + 1])
        denom = 2.0 * (y0 - 2.0 * y1 + y2)
        if denom < -1e-9:
            delta = (y0 - y2) / denom
            sub_bin = float(peak_idx) + float(np.clip(delta, -1.0, 1.0))
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

    aligned_auto = align_histogram(auto, sizing_shift)
    idx_160 = 160 - FRAG_MIN_LEN
    idx_175 = 175 - FRAG_MIN_LEN
    sum_core = float(np.sum(aligned_auto[idx_135 : idx_150 + 1]))
    sum_chrom = float(np.sum(aligned_auto[idx_160 : idx_175 + 1]))
    self_ref_ratio = float(sum_core / max(sum_chrom, 1.0))

    idx_250 = 250 - FRAG_MIN_LEN
    high_mw = float(np.sum(auto[idx_250:]) / max(total_paired, 1))

    idx_280 = 280 - FRAG_MIN_LEN
    idx_380 = 380 - FRAG_MIN_LEN
    dinucl = float(np.sum(auto[idx_280 : idx_380 + 1]) / max(total_paired, 1))

    helical_amp, helical_phase = compute_helical_periodicity(aligned_auto)

    return FragmentomicLandmarks(
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
        fetal_motif_ratio=fetal_motif,
    )


def deconvolve_fetal_maternal_densities(
    hist_autosomal_matrix: ArrayLike,
    hist_chry_matrix: ArrayLike,
    ff: ArrayLike,
    is_male: ArrayLike,
    *,
    smooth_sigma: float = 1.2,
    em_steps: int = 8,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cohort EM deconvolution of pure fetal and maternal cfDNA fragment length densities."""
    lengths = np.arange(FRAG_MIN_LEN, FRAG_MAX_LEN + 1, dtype=np.float64)
    auto = np.asarray(hist_autosomal_matrix, dtype=np.float64)
    chry = np.asarray(hist_chry_matrix, dtype=np.float64)
    ff_arr = np.asarray(ff, dtype=np.float64).ravel()
    male_arr = np.asarray(is_male, dtype=np.bool_).ravel()
    female_arr = ~male_arr

    param_f = np.exp(-0.5 * ((lengths - 143.0) / 10.0) ** 2)
    param_f /= np.sum(param_f)
    param_m = np.exp(-0.5 * ((lengths - 166.0) / 12.0) ** 2)
    param_m /= np.sum(param_m)

    if auto.ndim != 2 or chry.ndim != 2 or auto.shape != chry.shape:
        raise InvalidInputError(
            "hist_autosomal_matrix and hist_chry_matrix must be 2D arrays of identical shape"
        )
    n_samples, n_bins = auto.shape
    if n_samples == 0:
        return param_f, param_m
    if n_bins != FRAG_NUM_BINS:
        raise InvalidInputError(
            f"histogram matrices must have {FRAG_NUM_BINS} columns, got {n_bins}"
        )
    if ff_arr.size != n_samples or male_arr.size != n_samples:
        raise InvalidInputError(
            f"ff and is_male lengths must match sample count {n_samples}"
        )

    n_reads = np.sum(auto, axis=1)
    eligible_males = male_arr & np.isfinite(ff_arr) & (ff_arr >= 0.08)
    if not np.any(eligible_males):
        eligible_males = male_arr & np.isfinite(ff_arr) & (ff_arr >= 0.02)

    h_y_fetal = np.zeros(FRAG_NUM_BINS, dtype=np.float64)
    has_y_anchor = False
    if np.any(eligible_males) and float(np.sum(chry[eligible_males])) > 0.0:
        h_male_y = np.sum(chry[eligible_males], axis=0)
        h_female_y = (
            np.sum(chry[female_arr], axis=0)
            if np.any(female_arr)
            else np.zeros(FRAG_NUM_BINS, dtype=np.float64)
        )
        sum_male_n = float(np.sum(n_reads[eligible_males]))
        sum_female_n = float(np.sum(n_reads[female_arr])) if np.any(female_arr) else 0.0
        bg_scale = sum_male_n / max(sum_female_n, 1.0) if sum_female_n > 0.0 else 0.0

        y_anchor_males = male_arr & (
            (np.isfinite(ff_arr) & (ff_arr >= 0.08)) | ~np.isfinite(ff_arr)
        )
        if np.any(y_anchor_males) and float(np.sum(chry[y_anchor_males])) > float(
            np.sum(chry[eligible_males])
        ):
            h_male_y = np.sum(chry[y_anchor_males], axis=0)
            sum_male_n = float(np.sum(n_reads[y_anchor_males]))
            bg_scale = (
                sum_male_n / max(sum_female_n, 1.0) if sum_female_n > 0.0 else 0.0
            )

        h_y_fetal = np.maximum(h_male_y - bg_scale * h_female_y, 0.0)
        s_male_y = ndimage.gaussian_filter1d(h_male_y, sigma=smooth_sigma)
        s_female_y = ndimage.gaussian_filter1d(h_female_y, sigma=smooth_sigma)
        h_f0 = np.maximum(s_male_y - bg_scale * s_female_y, 0.0)
        sum_f0 = float(np.sum(h_f0))
        if sum_f0 > 0.0:
            p_f = h_f0 / sum_f0
            has_y_anchor = float(np.sum(h_y_fetal)) > 100.0
        else:
            p_f = param_f.copy()
    else:
        p_f = param_f.copy()
    # Preserve smoothed chrY anchor so autosomal soft-EM refines without drowning chrY.
    p_y_anchor = p_f.copy()

    h_m0 = np.zeros(FRAG_NUM_BINS, dtype=np.float64)
    for i in np.flatnonzero(eligible_males):
        f_s = float(np.clip(ff_arr[i], 0.002, 0.55))
        n_s = float(n_reads[i])
        if n_s > 0.0:
            p_s = auto[i] / n_s
            h_m0 += n_s * np.maximum(p_s - f_s * p_f, 0.0)

    sum_m0 = float(np.sum(h_m0))
    p_m = (h_m0 / sum_m0) if sum_m0 > 0.0 else param_m.copy()

    f_curr = np.where(
        eligible_males,
        np.clip(np.where(np.isfinite(ff_arr), ff_arr, 0.05), 0.002, 0.55),
        0.05,
    )

    for _ in range(max(1, em_steps)):
        p_f_ff = 0.5 * (p_f + p_y_anchor) if has_y_anchor else p_f
        for i in range(n_samples):
            if eligible_males[i] or n_reads[i] <= 0.0:
                continue
            h_i = auto[i]

            def _neg_ll(
                f_val: float,
                h_target: NDArray[np.float64] = h_i,
                pf_step: NDArray[np.float64] = p_f_ff,
                pm_step: NDArray[np.float64] = p_m,
            ) -> float:
                p_mix = f_val * pf_step + (1.0 - f_val) * pm_step
                return -float(np.sum(h_target * np.log(np.maximum(p_mix, 1e-12))))

            res = optimize.minimize_scalar(
                _neg_ll, bounds=(0.002, 0.55), method="bounded"
            )
            f_curr[i] = float(np.clip(res.x, 0.002, 0.55))

        all_mix = (
            f_curr[:, np.newaxis] * p_f[np.newaxis, :]
            + (1.0 - f_curr[:, np.newaxis]) * p_m[np.newaxis, :]
        )
        gamma_f = (f_curr[:, np.newaxis] * p_f[np.newaxis, :]) / np.maximum(
            all_mix, 1e-12
        )
        if has_y_anchor:
            auto_f_acc = np.sum(auto[eligible_males] * gamma_f[eligible_males], axis=0)
            anchor_mass = max(float(np.sum(h_y_fetal)), float(np.sum(auto_f_acc)))
            h_f_acc = auto_f_acc + anchor_mass * p_y_anchor
        else:
            h_f_acc = np.sum(auto * gamma_f, axis=0) + h_y_fetal
        h_m_acc = np.sum(auto * (1.0 - gamma_f), axis=0)

        sum_f_acc = float(np.sum(h_f_acc))
        if sum_f_acc > 0.0:
            p_f = np.maximum(h_f_acc, 0.0) / sum_f_acc
        sum_m_acc = float(np.sum(h_m_acc))
        if sum_m_acc > 0.0:
            p_m = np.maximum(h_m_acc, 0.0) / sum_m_acc

    p_f = np.maximum(ndimage.gaussian_filter1d(p_f, sigma=1.0), 0.0)
    sum_pf_final = float(np.sum(p_f))
    p_f = (p_f / sum_pf_final) if sum_pf_final > 0.0 else param_f

    p_m = np.maximum(ndimage.gaussian_filter1d(p_m, sigma=1.0), 0.0)
    sum_pm_final = float(np.sum(p_m))
    p_m = (p_m / sum_pm_final) if sum_pm_final > 0.0 else param_m

    return p_f, p_m


def predict_ff_kl_deconvolution(
    hist_sample: ArrayLike,
    p_fetal: ArrayLike,
    p_maternal: ArrayLike,
) -> tuple[float, float]:
    """Zero-parameter maximum likelihood estimation of fetal fraction via KL deconvolution."""
    h = np.asarray(hist_sample, dtype=np.float64).ravel()
    pf = np.asarray(p_fetal, dtype=np.float64).ravel()
    pm = np.asarray(p_maternal, dtype=np.float64).ravel()

    total = float(np.sum(h))
    if (
        total <= 0.0
        or h.size != FRAG_NUM_BINS
        or pf.size != FRAG_NUM_BINS
        or pm.size != FRAG_NUM_BINS
    ):
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


def compute_fragment_length_llr(
    hist: ArrayLike,
    p_fetal: ArrayLike,
    p_maternal: ArrayLike,
    *,
    sizing_shift_bp: float = 0.0,
    baseline_ff: float | None = None,
    is_gain: bool = False,
) -> float:
    """Compute fragment length log-likelihood ratio over [50, 350] bp in nats.

    When ``baseline_ff`` is provided, evaluates the exact Kass-Raftery log-Bayes-factor
    between fetal-origin CNV mixture ``q_fetal`` and maternal-origin CNV mixture
    ``q_maternal`` (matching ``chrx_llr_fetal < -2.0`` in ``classify_sca_ray_4d``).
    Otherwise evaluates ``sum_L h_L * ln(p_F(L) / p_M(L))`` over ``[50, 350]`` bp.
    """
    h_raw = np.asarray(hist, dtype=np.float64).ravel()
    pf_raw = np.asarray(p_fetal, dtype=np.float64).ravel()
    pm_raw = np.asarray(p_maternal, dtype=np.float64).ravel()
    if (
        h_raw.size != FRAG_NUM_BINS
        or pf_raw.size != FRAG_NUM_BINS
        or pm_raw.size != FRAG_NUM_BINS
    ):
        raise InvalidInputError(
            f"hist, p_fetal, and p_maternal must have length {FRAG_NUM_BINS}"
        )

    aligned = align_histogram(h_raw, sizing_shift_bp)
    idx_lo = 50 - FRAG_MIN_LEN
    idx_hi = 350 - FRAG_MIN_LEN + 1
    h_sub = np.maximum(aligned[idx_lo:idx_hi], 0.0)
    total_h = float(np.sum(h_sub))
    if total_h <= 0.0:
        return 0.0

    pf_sub = np.maximum(pf_raw[idx_lo:idx_hi], 0.0)
    pm_sub = np.maximum(pm_raw[idx_lo:idx_hi], 0.0)
    sum_pf = float(np.sum(pf_sub))
    sum_pm = float(np.sum(pm_sub))
    if sum_pf <= 0.0 or sum_pm <= 0.0:
        return 0.0
    pf_norm = np.maximum(pf_sub / sum_pf, 1e-12)
    pf_norm /= np.sum(pf_norm)
    pm_norm = np.maximum(pm_sub / sum_pm, 1e-12)
    pm_norm /= np.sum(pm_norm)

    if baseline_ff is not None and math.isfinite(baseline_ff) and baseline_ff > 0.0:
        f0 = float(np.clip(baseline_ff, 0.005, 0.55))
        if is_gain:
            phi_fet = float(np.clip((1.5 * f0) / (1.0 + 0.5 * f0), 1e-6, 1.0 - 1e-6))
            phi_mat = float(np.clip(f0 / (1.0 + 0.5 * f0), 1e-6, 1.0 - 1e-6))
        else:
            denom_loss = max(1.0 - 0.5 * f0, 1e-6)
            phi_fet = float(np.clip((0.5 * f0) / denom_loss, 1e-6, 1.0 - 1e-6))
            phi_mat = float(np.clip(f0 / denom_loss, 1e-6, 1.0 - 1e-6))
        q1 = np.maximum(phi_fet * pf_norm + (1.0 - phi_fet) * pm_norm, 1e-12)
        q2 = np.maximum(phi_mat * pf_norm + (1.0 - phi_mat) * pm_norm, 1e-12)
    else:
        q1 = pf_norm
        q2 = pm_norm

    pos = h_sub > 0.0
    return float(np.sum(h_sub[pos] * (np.log(q1[pos]) - np.log(q2[pos]))))


def compute_band_fisher_weights(
    p_fetal_571: ArrayLike,
    p_maternal_571: ArrayLike,
    *,
    f0: float = 0.08,
) -> tuple[NDArray[np.float64], float, float]:
    """Derive Fisher-optimal read weights mapped to 24 discrete fragmentomic bands."""
    pf = np.asarray(p_fetal_571, dtype=np.float64).ravel()
    pm = np.asarray(p_maternal_571, dtype=np.float64).ravel()
    if pf.size != FRAG_NUM_BINS or pm.size != FRAG_NUM_BINS:
        raise InvalidInputError(
            f"p_fetal_571 and p_maternal_571 must have length {FRAG_NUM_BINS}"
        )

    offsets = np.asarray(FRAG_BAND_EDGES, dtype=np.int64) - FRAG_MIN_LEN
    end_idx = int(offsets[-1])
    band_starts = offsets[:-1]
    pf_24 = np.add.reduceat(pf[:end_idx], band_starts)
    pm_24 = np.add.reduceat(pm[:end_idx], band_starts)

    sum_pf = float(np.sum(pf_24))
    if sum_pf > 0.0:
        pf_24 = pf_24 / sum_pf
    sum_pm = float(np.sum(pm_24))
    if sum_pm > 0.0:
        pm_24 = pm_24 / sum_pm

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
    frag_bands: ArrayLike,
    raw_counts: ArrayLike,
    paired_counts: ArrayLike,
    band_weights_24: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
    """Evaluate Fisher-weighted coverage and compound-Poisson variance dispersion factor from 24 frag_bands."""
    bands_f64 = np.asarray(frag_bands, dtype=np.float64)
    raw_f64 = np.asarray(raw_counts, dtype=np.float64)
    paired_f64 = np.asarray(paired_counts, dtype=np.float64)
    w_24 = np.asarray(band_weights_24, dtype=np.float64)

    w_paired = bands_f64 @ w_24
    raw_unpaired = np.maximum(raw_f64 - paired_f64, 0.0)
    w_counts = raw_unpaired + w_paired

    tot_w_sq = raw_unpaired + (bands_f64 @ np.square(w_24))
    has_reads = (w_counts > 0.0) & (paired_f64 > 0.0)

    kappa = np.ones(raw_f64.shape, dtype=np.float32)
    kappa[has_reads] = np.maximum(
        tot_w_sq[has_reads] / np.maximum(w_counts[has_reads], 1e-12),
        1e-6,
    ).astype(np.float32)

    return w_counts, kappa


def build_fragmentomic_feature_vector(
    hist_autosomal: ArrayLike,
    motif_counts: ArrayLike,
    chromatosome_mode_bp: float = CANONICAL_CHROMATOSOME_BP,
    sizing_shift_bp: float = 0.0,
    helical_10bp_amp: float | None = None,
    motif_mds: float | None = None,
    fetal_motif_ratio: float | None = None,
    reference_maternal_571: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Extract a 36-dimensional mode-aligned feature vector for the FF_size ensemble."""
    aligned = align_histogram(hist_autosomal, sizing_shift_bp)
    total = float(np.sum(aligned))
    if total <= 0.0 or aligned.size < FRAG_NUM_BINS:
        return np.zeros(36, dtype=np.float64)

    motifs = np.asarray(motif_counts, dtype=np.float64).ravel()
    p = aligned / total

    idx_164 = 164 - FRAG_MIN_LEN
    idx_168 = 168 - FRAG_MIN_LEN
    ref_safe = max(float(np.mean(p[idx_164 : idx_168 + 1])), 1e-8)

    idx_85 = 85 - FRAG_MIN_LEN
    idx_220 = 220 - FRAG_MIN_LEN
    band_means = p[idx_85:idx_220].reshape(27, 5).mean(axis=1)
    sample_band_ratios = np.maximum(band_means, 1e-8) / ref_safe

    if reference_maternal_571 is not None:
        pm_raw = np.asarray(reference_maternal_571, dtype=np.float64).ravel()
        pm_sum = float(np.sum(pm_raw))
        if pm_raw.size == FRAG_NUM_BINS and pm_sum > 0.0:
            pm = pm_raw / pm_sum
            pm_ref_safe = max(float(np.mean(pm[idx_164 : idx_168 + 1])), 1e-8)
            pm_band_means = pm[idx_85:idx_220].reshape(27, 5).mean(axis=1)
            pm_band_ratios = np.maximum(pm_band_means, 1e-8) / pm_ref_safe
            log_bands = np.log(sample_band_ratios / np.maximum(pm_band_ratios, 1e-8))
        else:
            log_bands = np.log(sample_band_ratios)
    else:
        log_bands = np.log(sample_band_ratios)

    idx_135 = 135 - FRAG_MIN_LEN
    idx_150 = 150 - FRAG_MIN_LEN
    idx_160 = 160 - FRAG_MIN_LEN
    idx_175 = 175 - FRAG_MIN_LEN
    self_ref_ratio = float(
        np.sum(aligned[idx_135 : idx_150 + 1])
        / max(float(np.sum(aligned[idx_160 : idx_175 + 1])), 1.0)
    )

    shoulder_ratios = p[list(_SUBNUC_PEAK_OFFSETS)] / ref_safe
    eff_helical = (
        float(helical_10bp_amp)
        if helical_10bp_amp is not None
        else compute_helical_periodicity(aligned)[0]
    )
    eff_mds = (
        float(motif_mds)
        if motif_mds is not None
        else compute_motif_diversity_score(motifs)
    )
    eff_fetal_motif = (
        float(fetal_motif_ratio)
        if fetal_motif_ratio is not None
        else _compute_fetal_motif_ratio(motifs)
    )

    out = np.empty(36, dtype=np.float64)
    out[0] = float(chromatosome_mode_bp)
    out[1:28] = log_bands
    out[28] = self_ref_ratio
    out[29:33] = shoulder_ratios
    out[33] = eff_helical
    out[34] = eff_mds
    out[35] = eff_fetal_motif
    return out


def fit_ff_fragmentomics_ensemble(
    features: ArrayLike,
    ff_targets: ArrayLike,
    *,
    seed: int = 42,
) -> LogitEnsembleFit:
    """Train a standardized convex logit-link RidgeCV + ElasticNetCV ensemble for FF_size."""
    x_arr = np.asarray(features, dtype=np.float64)
    if x_arr.ndim != 2 or x_arr.shape[0] < 5:
        n_rows = x_arr.shape[0] if x_arr.ndim >= 1 else 0
        raise InvalidInputError(
            f"fit_ff_fragmentomics_ensemble requires at least 5 samples, got {n_rows}"
        )

    fit = fit_logit_linear_ensemble(
        x_arr,
        ff_targets,
        standardize=True,
        l1_ratios=(0.1, 0.5, 0.9),
        ridge_alphas=np.logspace(-2, 4, 20),
        enet_alphas=np.logspace(-3, 2, 20),
        seed=seed,
        max_iter=2000,
    )
    if fit.oof_rmse < 0.015:
        return LogitEnsembleFit(
            weights=fit.weights,
            intercept=fit.intercept,
            feat_mean=fit.feat_mean,
            feat_std=fit.feat_std,
            oof_rmse=0.015,
        )
    return fit


def predict_ff_fragmentomics(
    hist_autosomal: ArrayLike,
    motif_counts: ArrayLike,
    *,
    paired_end: bool = True,
    chromatosome_mode_bp: float = CANONICAL_CHROMATOSOME_BP,
    sizing_shift_bp: float = 0.0,
    helical_10bp_amp: float | None = None,
    motif_mds: float | None = None,
    fetal_motif_ratio: float | None = None,
    p_fetal_density: ArrayLike | None = None,
    p_maternal_density: ArrayLike | None = None,
    ff_frag_weights: ArrayLike | None = None,
    ff_frag_bias: float = 0.0,
    ff_frag_feature_mean: ArrayLike | None = None,
    ff_frag_feature_scale: ArrayLike | None = None,
) -> float | None:
    """Predict fetal fraction from pure fragmentomic arrays via mode-aligned KL MLE + logit ensemble."""
    if not paired_end:
        return None

    raw_auto = np.asarray(hist_autosomal, dtype=np.float64).ravel()
    if raw_auto.size != FRAG_NUM_BINS or float(np.sum(raw_auto)) <= 0.0:
        return None

    aligned_auto = align_histogram(raw_auto, sizing_shift_bp)

    f_kl: float | None = None
    if p_fetal_density is not None and p_maternal_density is not None:
        pf = np.asarray(p_fetal_density, dtype=np.float64).ravel()
        pm = np.asarray(p_maternal_density, dtype=np.float64).ravel()
        if pf.size == FRAG_NUM_BINS and pm.size == FRAG_NUM_BINS:
            f_val, _ = predict_ff_kl_deconvolution(aligned_auto, pf, pm)
            f_kl = f_val

    f_ens: float | None = None
    if (
        ff_frag_weights is not None
        and ff_frag_feature_mean is not None
        and ff_frag_feature_scale is not None
    ):
        feat = build_fragmentomic_feature_vector(
            aligned_auto,
            motif_counts,
            chromatosome_mode_bp=chromatosome_mode_bp,
            sizing_shift_bp=0.0,
            helical_10bp_amp=helical_10bp_amp,
            motif_mds=motif_mds,
            fetal_motif_ratio=fetal_motif_ratio,
            reference_maternal_571=p_maternal_density,
        )
        mean = np.asarray(ff_frag_feature_mean, dtype=np.float64).ravel()
        scale = np.asarray(ff_frag_feature_scale, dtype=np.float64).ravel()
        weights = np.asarray(ff_frag_weights, dtype=np.float64).ravel()
        if (
            mean.size == feat.size
            and scale.size == feat.size
            and weights.size == feat.size
        ):
            safe_scale = np.where(np.abs(scale) < 1e-6, 1.0, scale)
            z = (feat - mean) / safe_scale
            u = float(np.dot(z, weights) + float(ff_frag_bias))
            f_ens = float(scipy.special.expit(u))

    if f_kl is not None and f_ens is not None:
        f_final = 0.5 * f_kl + 0.5 * f_ens
    elif f_kl is not None:
        f_final = f_kl
    elif f_ens is not None:
        f_final = f_ens
    else:
        return None

    return float(np.clip(f_final, 0.005, 0.50))
