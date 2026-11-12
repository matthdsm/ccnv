"""NumPy, SciPy, and scikit-learn numerical kernels for prenatal analysis."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import scipy.special
import scipy.stats
from numpy.typing import ArrayLike, NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor

from ccnv.models.exceptions import InvalidInputError
from ccnv.models.results import ActiveFFSource, SCACall
from ccnv.stats.regression import mad_scale

__all__ = [
    "MSYSexModel",
    "PoissonFFYModel",
    "SCARayClassification",
    "classify_sca_ray_4d",
    "compute_autosomal_density",
    "evaluate_poisson_msy_ffy",
    "fit_forward_ffx_model",
    "fit_msy_sex_mixture",
    "fit_poisson_msy_ffy",
    "fuse_active_fetal_fraction_student_t4",
]

_SCA_RAY_DIRECTIONS: Final[dict[SCACall, tuple[float, float, float, float]]] = {
    "XX": (0.0, 0.0, 1.0, 1.0),
    "XY": (-0.5, 1.0, 1.0, 1.0),
    "XO": (-0.5, 0.0, 1.0, 1.0),
    "XXX": (0.5, 0.0, 1.0, 1.0),
    "XXY": (0.0, 1.0, 1.0, 1.0),
    "XYY": (-0.5, 2.0, 1.0, 1.0),
}


@dataclass(frozen=True, slots=True)
class MSYSexModel:
    """Estimated prenatal sex mixture parameters and posteriors."""

    p_male: NDArray[np.float64]
    bg_rate: float
    bimodal: bool


@dataclass(frozen=True, slots=True)
class PoissonFFYModel:
    """Empirical clean MSY bins, female background rates, and calibration slope."""

    clean_bin_indices: NDArray[np.int64]
    lambda_bg: NDArray[np.float64]
    kappa_cal: float
    pooled_bg_rate: float


@dataclass(frozen=True, slots=True)
class SCARayClassification:
    """4D erfc ray-marginalized Sex Chromosome Aneuploidy classification and fused FF."""

    call: SCACall
    posterior: float
    mahalanobis_d2: float
    posteriors: dict[SCACall, float]
    active_ff: float
    active_ff_source: ActiveFFSource
    active_ff_se: float = 0.05


def _poisson_logpmf(
    c: NDArray[np.float64],
    mu: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate Poisson log probability mass function ln P(c | mu)."""
    mu_safe = np.maximum(mu, 1e-12)
    c_safe = np.maximum(c, 0.0)
    term1 = np.where(c_safe > 0.0, c_safe * np.log(mu_safe), 0.0)
    return term1 - mu_safe - scipy.special.gammaln(c_safe + 1.0)


def compute_autosomal_density(
    counts: ArrayLike,
    contigs: ArrayLike,
    exposure: ArrayLike,
    mappability: ArrayLike,
    gc_trend: ArrayLike,
    autosomal_mask: ArrayLike,
) -> float:
    """Compute all-bin chromosome-median ratio-of-sums autosomal read density."""
    c_arr = np.asarray(counts, dtype=np.float64).ravel()
    chr_arr = np.asarray(contigs, dtype=np.int64).ravel()
    exp_arr = np.asarray(exposure, dtype=np.float64).ravel()
    map_arr = np.asarray(mappability, dtype=np.float64).ravel()
    gc_arr = np.asarray(gc_trend, dtype=np.float64).ravel()
    mask_arr = np.asarray(autosomal_mask, dtype=np.bool_).ravel()

    n = c_arr.size
    if not (
        chr_arr.size == n
        and exp_arr.size == n
        and map_arr.size == n
        and gc_arr.size == n
        and mask_arr.size == n
    ):
        raise InvalidInputError(
            "counts, contigs, exposure, mappability, gc_trend, and autosomal_mask must have equal length"
        )

    expected_denom = exp_arr * map_arr * gc_arr
    usable = (
        mask_arr
        & np.isfinite(c_arr)
        & (c_arr >= 0.0)
        & np.isfinite(expected_denom)
        & (expected_denom > 0.0)
    )
    if not np.any(usable):
        return 0.001

    _, inv = np.unique(chr_arr[usable], return_inverse=True)
    sum_c = np.bincount(inv, weights=c_arr[usable])
    sum_denom = np.bincount(inv, weights=expected_denom[usable])
    med_density = float(np.median(sum_c / sum_denom))
    return med_density if med_density > 0.0 else 0.001


def fit_msy_sex_mixture(
    msy_counts: ArrayLike,
    msy_exposure: ArrayLike,
    *,
    bic_delta_min: float = 6.0,
) -> MSYSexModel:
    """Fit 2-component Poisson EM sex mixture model on cohort MSY read densities."""
    c = np.asarray(msy_counts, dtype=np.float64).ravel()
    e_raw = np.asarray(msy_exposure, dtype=np.float64).ravel()
    if c.size != e_raw.size:
        raise InvalidInputError(
            "msy_counts and msy_exposure must have identical length"
        )
    e = np.maximum(e_raw, 1e-9)
    s_size = int(c.size)

    if s_size < 2:
        rate = float(np.sum(c) / max(float(np.sum(e)), 1e-9))
        return MSYSexModel(
            p_male=np.zeros(s_size, dtype=np.float64),
            bg_rate=rate,
            bimodal=False,
        )

    lambda_bg_1 = float(np.sum(c) / np.sum(e))
    mu_1 = lambda_bg_1 * e
    l1 = float(np.sum(_poisson_logpmf(c, mu_1)))

    r = c / e
    med_r = float(np.median(r))
    lower_r = r[r <= med_r]
    lambda_bg = float(max(1e-6, np.median(lower_r) if lower_r.size > 0 else med_r))
    upper_r = r[r > med_r]
    male_target = float(np.median(upper_r) if upper_r.size > 0 else med_r + 0.02)
    delta_male = float(max(0.01, male_target - lambda_bg))
    pi_male = 0.5
    pi_female = 0.5

    p_male = np.zeros(s_size, dtype=np.float64)

    for _ in range(50):
        mu_fem = lambda_bg * e
        mu_male = (lambda_bg + delta_male) * e

        ll_fem = _poisson_logpmf(c, mu_fem)
        ll_male = _poisson_logpmf(c, mu_male)

        t_fem = math.log(max(pi_female, 1e-12)) + ll_fem
        t_male = math.log(max(pi_male, 1e-12)) + ll_male

        max_t = np.maximum(t_fem, t_male)
        log_sum = max_t + np.log(np.exp(t_fem - max_t) + np.exp(t_male - max_t))

        p_male = np.exp(t_male - log_sum)
        p_fem = 1.0 - p_male

        new_pi_male = float(np.clip(np.mean(p_male), 0.01, 0.99))
        new_pi_fem = 1.0 - new_pi_male

        w_fem = float(np.sum(p_fem * e))
        w_male = float(np.sum(p_male * e))

        new_lambda_bg = float(max(1e-6, np.sum(p_fem * c) / max(w_fem, 1e-9)))
        new_lambda_male = float(np.sum(p_male * c) / max(w_male, 1e-9))
        new_delta_male = float(max(1e-4, new_lambda_male - new_lambda_bg))

        diff = (
            abs(new_pi_male - pi_male)
            + abs(new_lambda_bg - lambda_bg)
            + abs(new_delta_male - delta_male)
        )
        pi_male = new_pi_male
        pi_female = new_pi_fem
        lambda_bg = new_lambda_bg
        delta_male = new_delta_male
        if diff < 1e-6:
            break

    mu_fem = lambda_bg * e
    mu_male = (lambda_bg + delta_male) * e
    ll_fem = _poisson_logpmf(c, mu_fem)
    ll_male = _poisson_logpmf(c, mu_male)
    t_fem = math.log(max(pi_female, 1e-12)) + ll_fem
    t_male = math.log(max(pi_male, 1e-12)) + ll_male
    max_t = np.maximum(t_fem, t_male)
    log_sum = max_t + np.log(np.exp(t_fem - max_t) + np.exp(t_male - max_t))
    l2 = float(np.sum(log_sum))

    delta_bic = 2.0 * (l2 - l1) - 2.0 * math.log(float(s_size))
    bimodal = (
        delta_bic > bic_delta_min and delta_male > 0.01 and (0.05 <= pi_male <= 0.95)
    )

    if bimodal:
        return MSYSexModel(
            p_male=p_male,
            bg_rate=lambda_bg,
            bimodal=True,
        )
    return MSYSexModel(
        p_male=np.zeros(s_size, dtype=np.float64),
        bg_rate=lambda_bg_1,
        bimodal=False,
    )


def fit_poisson_msy_ffy(
    msy_bin_counts: ArrayLike,
    msy_bin_exposure: ArrayLike,
    mappability: ArrayLike,
    female_mask: ArrayLike,
    male_mask: ArrayLike,
    labeled_ff: ArrayLike | None = None,
) -> PoissonFFYModel:
    """Identify clean single-copy MSY bins and fit closed-form Poisson FFY model."""
    c_mat = np.asarray(msy_bin_counts, dtype=np.float64)
    e_mat = np.asarray(msy_bin_exposure, dtype=np.float64)
    if c_mat.ndim != 2 or e_mat.ndim != 2 or c_mat.shape != e_mat.shape:
        raise InvalidInputError(
            "msy_bin_counts and msy_bin_exposure must be 2D (n_bins, n_samples) matrices of identical shape"
        )
    n_bins, n_samples = c_mat.shape

    map_arr = np.asarray(mappability, dtype=np.float32).ravel()
    fem_arr = np.asarray(female_mask, dtype=np.bool_).ravel()
    male_arr = np.asarray(male_mask, dtype=np.bool_).ravel()
    if map_arr.size != n_bins:
        raise InvalidInputError(
            f"mappability length {map_arr.size} must match n_bins {n_bins}"
        )
    if fem_arr.size != n_samples or male_arr.size != n_samples:
        raise InvalidInputError(
            f"female_mask and male_mask lengths must match n_samples {n_samples}"
        )

    has_females = bool(np.any(fem_arr))
    if has_females:
        fem_c_sum = np.sum(c_mat[:, fem_arr], axis=1)
        fem_e_sum = np.sum(e_mat[:, fem_arr], axis=1)
        lambda_bg_all = (0.5 + fem_c_sum) / (1.0 + fem_e_sum)
    else:
        lambda_bg_all = np.zeros(n_bins, dtype=np.float64)

    clean_mask = (map_arr >= 0.80) & (lambda_bg_all <= 0.05)
    if not np.any(clean_mask):
        clean_mask = map_arr >= 0.50
    if not np.any(clean_mask):
        clean_mask = np.ones(n_bins, dtype=np.bool_)

    clean_indices = np.flatnonzero(clean_mask).astype(np.int64)
    lambda_bg = lambda_bg_all[clean_indices]

    if has_females and clean_indices.size > 0:
        clean_fem_c = float(
            np.sum(c_mat[np.ix_(clean_indices, np.flatnonzero(fem_arr))])
        )
        clean_fem_e = float(
            np.sum(e_mat[np.ix_(clean_indices, np.flatnonzero(fem_arr))])
        )
        pooled_bg_rate = float(clean_fem_c / max(clean_fem_e, 1e-9))
    else:
        pooled_bg_rate = 0.0

    kappa_cal = 1.0
    if labeled_ff is not None and np.any(male_arr) and clean_indices.size > 0:
        lab_arr = np.asarray(labeled_ff, dtype=np.float64).ravel()
        if lab_arr.size != n_samples:
            raise InvalidInputError(
                f"labeled_ff length {lab_arr.size} must match n_samples {n_samples}"
            )
        valid_cal = male_arr & np.isfinite(lab_arr) & (lab_arr > 0.0)
        cal_idx = np.flatnonzero(valid_cal)
        if cal_idx.size >= 2:
            c_sub = c_mat[np.ix_(clean_indices, cal_idx)]
            e_sub = e_mat[np.ix_(clean_indices, cal_idx)]
            c_sums = np.sum(c_sub, axis=0)
            e_sums = np.sum(e_sub, axis=0)
            e_bg_sums = lambda_bg @ e_sub
            raw_vec = np.maximum(
                0.0, 2.0 * (c_sums - e_bg_sums) / np.maximum(e_sums, 1e-9)
            )
            lab_vec = lab_arr[cal_idx]
            denom_xx = float(np.dot(raw_vec, raw_vec))
            if denom_xx > 1e-8:
                slope = float(np.dot(raw_vec, lab_vec) / denom_xx)
                if math.isfinite(slope) and 0.5 <= slope <= 2.0:
                    kappa_cal = slope

    return PoissonFFYModel(
        clean_bin_indices=clean_indices,
        lambda_bg=lambda_bg,
        kappa_cal=kappa_cal,
        pooled_bg_rate=pooled_bg_rate,
    )


def evaluate_poisson_msy_ffy(
    sample_msy_counts: ArrayLike,
    sample_msy_exposure: ArrayLike,
    model: PoissonFFYModel,
) -> tuple[float, float]:
    """Evaluate closed-form Poisson MLE FFY and depth-propagated standard error."""
    c_arr = np.asarray(sample_msy_counts, dtype=np.float64)
    e_arr = np.asarray(sample_msy_exposure, dtype=np.float64)
    if c_arr.ndim != 1 or e_arr.ndim != 1 or c_arr.shape != e_arr.shape:
        raise InvalidInputError(
            "sample_msy_counts and sample_msy_exposure must be 1D full-contig vectors of identical shape"
        )

    idx = np.asarray(model.clean_bin_indices, dtype=np.int64)
    if idx.ndim != 1:
        raise InvalidInputError("model.clean_bin_indices must be a 1D integer array")
    if idx.size == 0:
        return 0.0, 0.05
    if int(np.min(idx)) < 0 or int(np.max(idx)) >= c_arr.size:
        raise InvalidInputError(
            f"model.clean_bin_indices out of bounds for contig vector of length {c_arr.size}"
        )

    c_clean = c_arr[idx]
    e_clean = e_arr[idx]
    sum_c = float(np.sum(c_clean))
    sum_e = float(np.sum(e_clean))
    denom = max(sum_e, 1e-9)

    lambda_bg = np.asarray(model.lambda_bg, dtype=np.float64)
    if lambda_bg.size != c_clean.size:
        lambda_bg = np.full(c_clean.size, model.pooled_bg_rate, dtype=np.float64)

    sum_bg = float(np.sum(e_clean * lambda_bg))
    raw_mle = max(0.0, 2.0 * (sum_c - sum_bg) / denom)
    ffy_hat = float(raw_mle * model.kappa_cal)
    ffy_se = float((2.0 * math.sqrt(max(sum_c, 1.0)) / denom) * model.kappa_cal)
    return ffy_hat, ffy_se


def fit_forward_ffx_model(
    chrx_ratios: ArrayLike,
    f_targets: ArrayLike,
    valid_mask: ArrayLike,
) -> tuple[float, float, float]:
    """Fit forward linear model R_X = alpha_0 + alpha_1 * f with shrinkage."""
    r_arr = np.asarray(chrx_ratios, dtype=np.float64).ravel()
    f_arr = np.asarray(f_targets, dtype=np.float64).ravel()
    v_arr = np.asarray(valid_mask, dtype=np.bool_).ravel()
    if not (r_arr.size == f_arr.size == v_arr.size):
        raise InvalidInputError(
            "chrx_ratios, f_targets, and valid_mask must have equal length"
        )

    clean = v_arr & np.isfinite(r_arr) & np.isfinite(f_arr)
    if int(np.count_nonzero(clean)) < 2:
        return 1.0, -0.5, 0.01

    r_clean = r_arr[clean]
    f_clean = f_arr[clean]

    xx_mask = clean & (f_arr == 0.0)
    alpha_0_prior = (
        float(np.median(r_arr[xx_mask]))
        if np.any(xx_mask)
        else float(np.median(r_clean))
    )

    male_f = f_clean[f_clean > 0.0]
    male_spread = float(np.ptp(male_f)) if male_f.size >= 2 else 0.0

    if male_spread < 1e-4 or male_f.size < 2:
        alpha_0 = alpha_0_prior
        alpha_1 = -0.5 * alpha_0
    else:
        huber = HuberRegressor(alpha=0.0)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                huber.fit(f_clean.reshape(-1, 1), r_clean)
            fit_a0 = float(huber.intercept_)
            fit_a1 = float(huber.coef_[0])
            if 0.5 <= fit_a0 <= 1.5 and fit_a1 < 0.0:
                alpha_0 = fit_a0
                alpha_1 = fit_a1
            else:
                alpha_0 = alpha_0_prior
                alpha_1 = -0.5 * alpha_0
        except ValueError:
            alpha_0 = alpha_0_prior
            alpha_1 = -0.5 * alpha_0

    residuals = r_clean - (alpha_0 + alpha_1 * f_clean)
    mad_sigma = mad_scale(residuals)
    sigma_sys = (
        max(0.005, mad_sigma)
        if mad_sigma > 0.0
        else max(0.005, float(np.std(residuals)))
    )

    return alpha_0, alpha_1, sigma_sys


def _log_normal_interval_prob(z_lo: float, z_hi: float) -> float:
    """Evaluate log(Phi(z_hi) - Phi(z_lo)) without float64 tail cancellation."""
    if z_hi <= z_lo:
        return -700.0
    if z_lo < 0.0 < z_hi:
        prob = 0.5 * float(
            scipy.special.erf(z_hi / math.sqrt(2.0))
            - scipy.special.erf(z_lo / math.sqrt(2.0))
        )
        return math.log(max(prob, 1e-300))
    a, b = (-z_hi, -z_lo) if z_lo >= 0.0 else (z_lo, z_hi)
    log_b = float(scipy.special.log_ndtr(b))
    log_a = float(scipy.special.log_ndtr(a))
    diff = log_a - log_b
    if diff < -40.0:
        return log_b
    return log_b + math.log(max(-math.expm1(min(diff, -1e-16)), 1e-300))


def fuse_active_fetal_fraction_student_t4(
    estimates: Sequence[float],
    standard_errors: Sequence[float],
    *,
    df: float = 4.0,
    max_iter: int = 5,
    tol: float = 1e-4,
) -> tuple[float, float]:
    """Fuse multi-modal fetal fraction estimates via Student-t4 IRLS."""
    valid_pairs = [
        (float(est), float(se))
        for est, se in zip(estimates, standard_errors, strict=True)
        if est is not None and math.isfinite(est) and math.isfinite(se) and se > 1e-9
    ]
    if not valid_pairs:
        return 0.0, 0.05
    if len(valid_pairs) == 1:
        est, se = valid_pairs[0]
        return max(0.0, est), se

    est_arr = np.array([p[0] for p in valid_pairs], dtype=np.float64)
    se_arr = np.array([p[1] for p in valid_pairs], dtype=np.float64)
    nu = max(float(df), 1.0)
    inv_var = 1.0 / np.square(se_arr)

    current_f = float(np.sum(inv_var * est_arr) / np.sum(inv_var))

    for _ in range(max(1, max_iter)):
        r = est_arr - current_f
        w = inv_var / (1.0 + np.square(r / se_arr) / nu)
        sum_w = float(np.sum(w))
        if sum_w <= 1e-12:
            break
        next_f = float(np.sum(w * est_arr) / sum_w)
        if abs(next_f - current_f) < tol:
            current_f = next_f
            break
        current_f = next_f

    r_final = est_arr - current_f
    w_final = inv_var / (1.0 + np.square(r_final / se_arr) / nu)
    final_se = float(1.0 / math.sqrt(max(float(np.sum(w_final)), 1e-12)))
    return max(0.0, current_f), max(final_se, 1e-4)


def classify_sca_ray_4d(
    delta_x: float,
    sigma_x: float,
    ffy: float | None,
    sigma_ffy: float,
    ffa: float | None,
    sigma_ffa: float,
    ff_size: float | None = None,
    sigma_size: float = 0.018,
    *,
    chrx_llr_fetal: float | None = None,
    clinical_priors: Mapping[str, float] | None = None,
) -> SCARayClassification:
    """Classify Sex Chromosome Aneuploidies via 4D closed-form erfc ray marginalization."""
    cands: list[tuple[int, float | None, float]] = [
        (0, delta_x, max(float(sigma_x), 1e-4)),
        (1, ffy, max(float(sigma_ffy), 1e-4)),
        (2, ffa, max(float(sigma_ffa), 1e-4)),
        (3, ff_size, max(float(sigma_size), 1e-4)),
    ]
    active = [
        (idx, float(val), se)
        for idx, val, se in cands
        if val is not None and math.isfinite(val) and math.isfinite(se) and se > 1e-9
    ]
    if not active:
        empty_post: dict[SCACall, float] = {k: 0.0 for k in _SCA_RAY_DIRECTIONS}
        empty_post["DISCORDANT_MIXED"] = 1.0
        return SCARayClassification(
            call="DISCORDANT_MIXED",
            posterior=1.0,
            mahalanobis_d2=0.0,
            posteriors=empty_post,
            active_ff=0.0,
            active_ff_source="FFA",
            active_ff_se=0.05,
        )

    indices = [a[0] for a in active]
    y = np.array([a[1] for a in active], dtype=np.float64)
    sig = np.array([a[2] for a in active], dtype=np.float64)
    dim = len(indices)

    inv_var = 1.0 / np.square(sig)
    log_det_sigma = float(np.sum(np.log(np.square(sig))))
    y_norm_sq = float(np.sum(inv_var * np.square(y)))

    log_joints: dict[SCACall, float] = {}
    d2_dict: dict[SCACall, float] = {}
    f_max = 0.60

    for k, direction in _SCA_RAY_DIRECTIONS.items():
        ak = np.array([direction[i] for i in indices], dtype=np.float64)
        vk = max(1e-12, float(np.sum(inv_var * np.square(ak))))
        mk = float(np.sum(inv_var * ak * y) / vk)
        d2_line = max(0.0, y_norm_sq - vk * (mk**2))
        f_star = min(max(mk, 0.0), f_max)
        d2_dict[k] = d2_line + vk * ((mk - f_star) ** 2)

        sqrt_vk = math.sqrt(vk)
        z_lo = -sqrt_vk * mk
        z_hi = sqrt_vk * (f_max - mk)
        log_integral = 0.5 * (
            math.log(2.0 * math.pi) - math.log(vk)
        ) + _log_normal_interval_prob(z_lo, z_hi)
        log_lik = (
            -0.5 * dim * math.log(2.0 * math.pi)
            - 0.5 * log_det_sigma
            - 0.5 * d2_line
            + log_integral
        )
        default_prior = 0.4955 if k in ("XX", "XY") else 0.0018
        prior_val = (
            float(clinical_priors.get(k, 1e-9))
            if clinical_priors is not None
            else default_prior
        )
        log_joints[k] = log_lik + math.log(max(prior_val, 1e-12))

    max_log = max(log_joints.values())
    unnorm = {k: math.exp(v - max_log) for k, v in log_joints.items()}
    sum_unnorm = sum(unnorm.values())
    posteriors: dict[SCACall, float] = {k: unnorm[k] / sum_unnorm for k in unnorm}

    best_k = max(posteriors, key=lambda k: posteriors[k])
    best_d2 = float(d2_dict[best_k])

    deg_f = max(dim - 1, 1)
    chi2_crit = float(scipy.stats.chi2.ppf(0.999, df=deg_f))

    call: SCACall = "DISCORDANT_MIXED" if best_d2 > chi2_crit else best_k
    if call == "XO" and chrx_llr_fetal is not None and chrx_llr_fetal < -2.0:
        call = "DISCORDANT_MIXED"

    if call == "DISCORDANT_MIXED":
        p_mixed = float(np.clip(scipy.stats.chi2.cdf(best_d2, df=deg_f), 0.51, 1.0))
        scale_rem = 1.0 - p_mixed
        for ray_k in list(posteriors.keys()):
            posteriors[ray_k] *= scale_rem
        posteriors["DISCORDANT_MIXED"] = p_mixed
    else:
        posteriors["DISCORDANT_MIXED"] = 0.0

    ref_dir = _SCA_RAY_DIRECTIONS[best_k]
    ff_cands: list[float] = []
    se_cands: list[float] = []
    source_names: list[ActiveFFSource] = []

    if call != "DISCORDANT_MIXED":
        if ref_dir[1] > 0.0 and ffy is not None and math.isfinite(ffy):
            ff_cands.append(max(0.0, float(ffy) / ref_dir[1]))
            se_cands.append(max(float(sigma_ffy), 1e-4) / ref_dir[1])
            source_names.append("FFY")
        if abs(ref_dir[0]) > 1e-6 and math.isfinite(delta_x):
            ff_cands.append(max(0.0, float(delta_x) / ref_dir[0]))
            se_cands.append(max(float(sigma_x), 1e-4) / abs(ref_dir[0]))
            source_names.append("FFX")
    if ffa is not None and math.isfinite(ffa):
        ff_cands.append(max(0.0, float(ffa)))
        se_cands.append(max(float(sigma_ffa), 1e-4))
        source_names.append("FFA")
    if ff_size is not None and math.isfinite(ff_size):
        ff_cands.append(max(0.0, float(ff_size)))
        se_cands.append(max(float(sigma_size), 1e-4))
        source_names.append("FF_SIZE")

    act_ff, act_se = fuse_active_fetal_fraction_student_t4(ff_cands, se_cands)
    act_source: ActiveFFSource = (
        "FUSED"
        if len(source_names) > 1
        else (source_names[0] if source_names else "FFA")
    )

    return SCARayClassification(
        call=call,
        posterior=float(posteriors[call]),
        mahalanobis_d2=best_d2,
        posteriors=posteriors,
        active_ff=act_ff,
        active_ff_source=act_source,
        active_ff_se=act_se,
    )
