"""NumPy, SciPy, and scikit-learn numerical kernels for prenatal analysis."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import scipy.special
import scipy.stats
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import KFold

from ccnv.models.results import ActiveFFSource, SCACall
from ccnv.stats.regression import fit_logit_linear_ensemble

__all__ = [
    "MSYSexModel",
    "PoissonFFYModel",
    "classify_sca_ray_4d",
    "compute_autosomal_density",
    "evaluate_poisson_msy_ffy",
    "fit_ffa_logit_ensemble",
    "fit_forward_ffx_model",
    "fit_msy_sex_mixture",
    "fit_poisson_msy_ffy",
    "fuse_active_fetal_fraction_student_t4",
    "ols_fit",
]


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


def _poisson_logpmf(
    c: NDArray[np.float64], mu: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Evaluate Poisson log probability mass function ln P(c | mu)."""
    mu_safe = np.maximum(mu, 1e-12)
    c_safe = np.maximum(c, 0.0)
    term1 = np.where(c_safe > 0.0, c_safe * np.log(mu_safe), 0.0)
    return term1 - mu_safe - scipy.special.gammaln(c_safe + 1.0)


def compute_autosomal_density(
    counts: NDArray[np.floating] | np.ndarray,
    contigs: NDArray[np.integer] | np.ndarray,
    exposure: NDArray[np.floating] | np.ndarray,
    mappability: NDArray[np.floating] | np.ndarray,
    gc_trend: NDArray[np.floating] | np.ndarray,
    autosomal_mask: NDArray[np.bool_] | np.ndarray,
) -> float:
    """Compute all-bin chromosome-median ratio-of-sums autosomal read density."""
    c_arr = np.asarray(counts, dtype=np.float64).ravel()
    chr_arr = np.asarray(contigs, dtype=np.int16).ravel()
    exp_arr = np.asarray(exposure, dtype=np.float64).ravel()
    map_arr = np.asarray(mappability, dtype=np.float64).ravel()
    gc_arr = np.asarray(gc_trend, dtype=np.float64).ravel()
    mask_arr = np.asarray(autosomal_mask, dtype=np.bool_).ravel()

    expected_denom = exp_arr * map_arr * gc_arr
    usable = (
        mask_arr
        & np.isfinite(c_arr)
        & np.isfinite(expected_denom)
        & (expected_denom > 0.0)
    )
    if not np.any(usable):
        return 0.001

    unique_chrs = np.unique(chr_arr[usable])
    ratios: list[float] = []
    for chrom in unique_chrs:
        c_mask = usable & (chr_arr == chrom)
        sum_c = float(np.sum(c_arr[c_mask]))
        sum_denom = float(np.sum(expected_denom[c_mask]))
        if sum_denom > 0.0:
            ratios.append(sum_c / sum_denom)

    if len(ratios) > 0:
        return float(np.median(ratios))

    sum_all_c = float(np.sum(c_arr[usable]))
    sum_all_denom = float(np.sum(expected_denom[usable]))
    return float(sum_all_c / max(sum_all_denom, 1.0))


def fit_msy_sex_mixture(
    msy_counts: NDArray[np.floating] | np.ndarray,
    msy_exposure: NDArray[np.floating] | np.ndarray,
    *,
    bic_delta_min: float = 6.0,
) -> MSYSexModel:
    """Fit 2-component Poisson EM sex mixture model on cohort MSY read densities."""
    c = np.asarray(msy_counts, dtype=np.float64).ravel()
    e = np.maximum(np.asarray(msy_exposure, dtype=np.float64).ravel(), 1e-9)
    s_size = c.size

    if s_size < 2:
        rate = float(np.sum(c) / max(np.sum(e), 1e-9))
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

        log_pi_fem = math.log(max(pi_female, 1e-12))
        log_pi_male = math.log(max(pi_male, 1e-12))

        t_fem = log_pi_fem + ll_fem
        t_male = log_pi_male + ll_male

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

    delta_bic = 2.0 * (l2 - l1) - 3.0 * math.log(float(s_size))
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
    msy_bin_counts: NDArray[np.floating] | np.ndarray,
    msy_bin_exposure: NDArray[np.floating] | np.ndarray,
    mappability: NDArray[np.floating] | np.ndarray,
    female_mask: NDArray[np.bool_] | np.ndarray,
    male_mask: NDArray[np.bool_] | np.ndarray,
    labeled_ff: NDArray[np.floating] | np.ndarray | None = None,
) -> PoissonFFYModel:
    """Identify clean single-copy MSY bins and fit closed-form Poisson FFY model."""
    c_mat = np.asarray(msy_bin_counts, dtype=np.float64)
    if c_mat.ndim == 1:
        c_mat = c_mat.reshape(-1, 1)
    n_bins, n_samples = c_mat.shape

    e_mat = np.asarray(msy_bin_exposure, dtype=np.float64)
    if e_mat.ndim == 1:
        if e_mat.size == n_bins:
            e_mat = np.repeat(e_mat.reshape(-1, 1), n_samples, axis=1)
        else:
            e_mat = np.repeat(e_mat.reshape(1, -1), n_bins, axis=0)

    map_arr = np.asarray(mappability, dtype=np.float32).ravel()
    fem_arr = np.asarray(female_mask, dtype=np.bool_).ravel()
    male_arr = np.asarray(male_mask, dtype=np.bool_).ravel()

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

    if has_females:
        clean_fem_c = float(np.sum(c_mat[clean_indices, :][:, fem_arr]))
        clean_fem_e = float(np.sum(e_mat[clean_indices, :][:, fem_arr]))
        pooled_bg_rate = float(clean_fem_c / max(clean_fem_e, 1e-9))
    else:
        pooled_bg_rate = 0.0

    kappa_cal = 1.0
    if labeled_ff is not None and np.any(male_arr):
        lab_arr = np.asarray(labeled_ff, dtype=np.float64).ravel()
        valid_cal = male_arr & np.isfinite(lab_arr) & (lab_arr > 0.0)
        if np.sum(valid_cal) >= 2:
            raw_ff_list: list[float] = []
            lab_list: list[float] = []
            for s in np.flatnonzero(valid_cal):
                c_s = float(np.sum(c_mat[clean_indices, s]))
                e_s = float(np.sum(e_mat[clean_indices, s]))
                e_bg_s = float(np.sum(e_mat[clean_indices, s] * lambda_bg))
                raw_f = max(0.0, 2.0 * (c_s - e_bg_s) / max(e_s, 1e-9))
                raw_ff_list.append(raw_f)
                lab_list.append(float(lab_arr[s]))
            raw_vec = np.array(raw_ff_list, dtype=np.float64)
            lab_vec = np.array(lab_list, dtype=np.float64)
            if float(np.ptp(raw_vec)) > 1e-4:
                _, slope = ols_fit(raw_vec, lab_vec)
                if 0.5 <= slope <= 2.0:
                    kappa_cal = float(slope)

    return PoissonFFYModel(
        clean_bin_indices=clean_indices,
        lambda_bg=lambda_bg,
        kappa_cal=kappa_cal,
        pooled_bg_rate=pooled_bg_rate,
    )


def evaluate_poisson_msy_ffy(
    sample_msy_counts: NDArray[np.floating] | np.ndarray,
    sample_msy_exposure: NDArray[np.floating] | np.ndarray,
    model: PoissonFFYModel,
) -> tuple[float, float]:
    """Evaluate closed-form Poisson MLE FFY and depth-propagated standard error."""
    c_arr = np.asarray(sample_msy_counts, dtype=np.float64).ravel()
    e_arr = np.asarray(sample_msy_exposure, dtype=np.float64).ravel()

    if c_arr.size == model.clean_bin_indices.size:
        c_clean = c_arr
        e_clean = e_arr
    elif c_arr.size > model.clean_bin_indices.size and (
        model.clean_bin_indices.size == 0
        or int(np.max(model.clean_bin_indices)) < c_arr.size
    ):
        c_clean = c_arr[model.clean_bin_indices]
        e_clean = e_arr[model.clean_bin_indices]
    else:
        c_clean = c_arr
        e_clean = e_arr

    sum_c = float(np.sum(c_clean))
    sum_e = float(np.sum(e_clean))
    denom = max(sum_e, 1e-9)

    lambda_bg = model.lambda_bg
    if lambda_bg.size != c_clean.size:
        lambda_bg = np.full(c_clean.size, model.pooled_bg_rate, dtype=np.float64)

    sum_bg = float(np.sum(e_clean * lambda_bg))
    raw_mle = max(0.0, 2.0 * (sum_c - sum_bg) / denom)
    ffy_hat = float(raw_mle * model.kappa_cal)
    ffy_se = float((2.0 * math.sqrt(max(sum_c, 1.0)) / denom) * model.kappa_cal)
    return ffy_hat, ffy_se


def fit_forward_ffx_model(
    chrx_ratios: NDArray[np.floating] | np.ndarray,
    f_targets: NDArray[np.floating] | np.ndarray,
    valid_mask: NDArray[np.bool_] | np.ndarray,
) -> tuple[float, float, float]:
    """Fit forward linear model R_X = alpha_0 + alpha_1 * f with shrinkage."""
    r_arr = np.asarray(chrx_ratios, dtype=np.float64).ravel()
    f_arr = np.asarray(f_targets, dtype=np.float64).ravel()
    v_arr = np.asarray(valid_mask, dtype=np.bool_).ravel()

    clean = v_arr & np.isfinite(r_arr) & np.isfinite(f_arr)
    if np.sum(clean) < 2:
        return 1.0, -0.5, 0.01

    r_clean = r_arr[clean]
    f_clean = f_arr[clean]

    xx_mask = clean & (f_arr == 0.0)
    if np.any(xx_mask):
        alpha_0_prior = float(np.median(r_arr[xx_mask]))
    else:
        alpha_0_prior = float(np.median(r_clean))

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
    med_res = float(np.median(residuals))
    mad_res = float(np.median(np.abs(residuals - med_res)))
    sigma_sys = (
        max(0.005, 1.4826 * mad_res)
        if mad_res > 0.0
        else max(0.005, float(np.std(residuals)))
    )

    return alpha_0, alpha_1, sigma_sys


def fit_ffa_logit_ensemble(
    X: NDArray[np.floating] | np.ndarray,
    y: NDArray[np.floating] | np.ndarray,
    seed: int = 42,
) -> tuple[NDArray[np.float64], float, float]:
    """Fit convex logit-link RidgeCV and ElasticNetCV ensemble for FFA."""
    x_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n_samples, n_features = x_arr.shape

    if n_samples < 2 or n_features == 0:
        mean_y = float(np.mean(y_arr)) if n_samples > 0 else 0.10
        b_fallback = float(scipy.special.logit(np.clip(mean_y, 0.005, 0.85)))
        return np.zeros(n_features, dtype=np.float64), b_fallback, 0.02

    y_clip = np.clip(y_arr, 0.005, 0.85)
    u = scipy.special.logit(y_clip)
    ridge_alphas = np.logspace(-2, 4, 40)

    w_ens, b_ens = fit_logit_linear_ensemble(
        x_arr,
        u,
        ridge_alphas=ridge_alphas,
        l1_ratio=(0.1, 0.5),
        seed=seed,
        max_iter=3000,
    )

    if n_samples >= 6:
        cv_outer = min(5, n_samples)
        oof_u = np.empty(n_samples, dtype=np.float64)
        cv = KFold(n_splits=cv_outer, shuffle=True, random_state=seed)
        for tr_idx, va_idx in cv.split(x_arr):
            w_fold, b_fold = fit_logit_linear_ensemble(
                x_arr[tr_idx],
                u[tr_idx],
                ridge_alphas=ridge_alphas,
                l1_ratio=(0.1, 0.5),
                seed=seed,
                max_iter=3000,
            )
            oof_u[va_idx] = x_arr[va_idx] @ w_fold + b_fold
        ens_pred = scipy.special.expit(oof_u)
        sigma_ffa = max(float(np.sqrt(np.mean((ens_pred - y_clip) ** 2))), 0.005)
    else:
        y_pred = scipy.special.expit(x_arr @ w_ens + b_ens)
        sigma_ffa = max(float(np.sqrt(np.mean((y_pred - y_clip) ** 2))), 0.005)

    return w_ens, b_ens, sigma_ffa


def ols_fit(
    x: NDArray[np.floating] | np.ndarray,
    y: NDArray[np.floating] | np.ndarray,
) -> tuple[float, float]:
    """Fit ordinary least squares returning (intercept, slope) with fallback."""
    xv = np.asarray(x, dtype=np.float64).ravel()
    yv = np.asarray(y, dtype=np.float64).ravel()
    if xv.size < 2 or float(np.var(xv)) < 1e-12:
        return 0.0, 1.0

    res = scipy.stats.linregress(xv, yv)
    return float(res.intercept), float(res.slope)


_SCA_RAY_DIRECTIONS: Final[dict[str, tuple[float, float, float, float]]] = {
    "XX": (0.0, 0.0, 1.0, 1.0),
    "XY": (-0.5, 1.0, 1.0, 1.0),
    "XO": (-0.5, 0.0, 1.0, 1.0),
    "XXX": (0.5, 0.0, 1.0, 1.0),
    "XXY": (0.0, 1.0, 1.0, 1.0),
    "XYY": (-0.5, 2.0, 1.0, 1.0),
}

_SCA_CLINICAL_PRIORS: Final[dict[str, float]] = {
    "XX": 0.4955,
    "XY": 0.4955,
    "XO": 0.0018,
    "XXX": 0.0018,
    "XXY": 0.0018,
    "XYY": 0.0018,
}


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
        for est, se in zip(estimates, standard_errors, strict=False)
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

    w = 1.0 / (se_arr**2)
    current_f = float(np.sum(w * est_arr) / np.sum(w))

    for _ in range(max(1, max_iter)):
        r = est_arr - current_f
        w = ((nu + 1.0) / (nu + (r / se_arr) ** 2)) * (1.0 / (se_arr**2))
        sum_w = float(np.sum(w))
        if sum_w <= 1e-12:
            break
        next_f = float(np.sum(w * est_arr) / sum_w)
        if abs(next_f - current_f) < tol:
            current_f = next_f
            break
        current_f = next_f

    final_se = float(1.0 / np.sqrt(max(float(np.sum(w)), 1e-12)))
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
    corr_matrix: NDArray[np.float64] | None = None,
    clinical_priors: dict[str, float] | None = None,
) -> tuple[SCACall, dict[str, float], float, float, ActiveFFSource]:
    """Classify Sex Chromosome Aneuploidies via 4D closed-form erfc ray marginalization."""
    priors_dict = (
        clinical_priors if clinical_priors is not None else _SCA_CLINICAL_PRIORS
    )
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
        equal_priors = {k: 1.0 / len(_SCA_RAY_DIRECTIONS) for k in _SCA_RAY_DIRECTIONS}
        return "DISCORDANT_MIXED", equal_priors, 0.0, 0.05, "FFA"

    indices = [a[0] for a in active]
    y = np.array([a[1] for a in active], dtype=np.float64)
    sig = np.array([a[2] for a in active], dtype=np.float64)
    dim = len(indices)

    if corr_matrix is None:
        omega = np.diag(1.0 / (sig**2))
        log_det_sigma = float(np.sum(np.log(sig**2)))
    else:
        r_sub = np.asarray(corr_matrix, dtype=np.float64)[np.ix_(indices, indices)]
        sigma_mat = np.diag(sig) @ r_sub @ np.diag(sig)
        omega = np.linalg.inv(sigma_mat)
        log_det_sigma = float(np.linalg.slogdet(sigma_mat)[1])

    log_joints: dict[str, float] = {}
    d2_dict: dict[str, float] = {}
    f_max = 0.60

    for k, direction in _SCA_RAY_DIRECTIONS.items():
        ak = np.array([direction[i] for i in indices], dtype=np.float64)
        vk = float(ak.T @ omega @ ak)
        vk = max(1e-12, vk)
        mk = float((ak.T @ omega @ y) / vk)
        d2 = max(0.0, float(y.T @ omega @ y - vk * (mk**2)))
        d2_dict[k] = d2

        arg_high = math.sqrt(0.5 * vk) * (f_max - mk)
        arg_low = -math.sqrt(0.5 * vk) * mk
        delta_erf = scipy.special.erf(arg_high) - scipy.special.erf(arg_low)
        delta_erf = max(delta_erf, 1e-300)

        log_integral = 0.5 * (math.log(math.pi) - math.log(2.0 * vk)) + math.log(
            delta_erf
        )
        log_lik = (
            -0.5 * dim * math.log(2.0 * math.pi)
            - 0.5 * log_det_sigma
            - 0.5 * d2
            + log_integral
        )
        prior_val = priors_dict.get(k, 1e-9)
        log_joints[k] = log_lik + math.log(max(prior_val, 1e-12))

    max_log = max(log_joints.values())
    unnorm = {k: math.exp(v - max_log) for k, v in log_joints.items()}
    sum_unnorm = sum(unnorm.values())
    posteriors = {k: unnorm[k] / sum_unnorm for k in unnorm}

    best_k = max(posteriors.keys(), key=lambda k: posteriors[k])
    best_d2 = d2_dict[best_k]

    deg_f = max(dim - 1, 1)
    chi2_crit = float(scipy.stats.chi2.ppf(0.999, df=deg_f))

    call: SCACall
    if best_d2 > chi2_crit:
        call = "DISCORDANT_MIXED"
    elif best_k in ("XX", "XY", "XO", "XXX", "XXY", "XYY"):
        call = best_k
    else:
        call = "DISCORDANT_MIXED"

    if call == "XO" and chrx_llr_fetal is not None and chrx_llr_fetal < -2.0:
        call = "DISCORDANT_MIXED"

    if call == "DISCORDANT_MIXED":
        p_mixed = float(np.clip(scipy.stats.chi2.cdf(best_d2, df=deg_f), 0.51, 1.0))
        posteriors["DISCORDANT_MIXED"] = p_mixed
    else:
        posteriors["DISCORDANT_MIXED"] = 0.0

    ref_dir = _SCA_RAY_DIRECTIONS[best_k]
    ff_cands: list[float] = []
    se_cands: list[float] = []
    source_names: list[ActiveFFSource] = []

    if call != "DISCORDANT_MIXED":
        if ref_dir[1] > 0 and ffy is not None and math.isfinite(ffy):
            ff_cands.append(ffy / ref_dir[1])
            se_cands.append(sigma_ffy / ref_dir[1])
            source_names.append("FFY")
        if abs(ref_dir[0]) > 1e-6 and math.isfinite(delta_x):
            ff_cands.append(delta_x / ref_dir[0])
            se_cands.append(sigma_x / abs(ref_dir[0]))
            source_names.append("FFX")
    if ffa is not None and math.isfinite(ffa):
        ff_cands.append(ffa)
        se_cands.append(sigma_ffa)
        source_names.append("FFA")
    if ff_size is not None and math.isfinite(ff_size):
        ff_cands.append(ff_size)
        se_cands.append(sigma_size)
        source_names.append("FF_SIZE")

    act_ff, act_se = fuse_active_fetal_fraction_student_t4(ff_cands, se_cands)
    act_source: ActiveFFSource
    if "FF_SIZE" in source_names:
        act_source = "FUSED"
    elif "FFY" in source_names:
        act_source = "FFY"
    elif len(source_names) > 1:
        act_source = "FUSED"
    else:
        act_source = source_names[0] if source_names else "FFA"

    return call, posteriors, act_ff, act_se, act_source
