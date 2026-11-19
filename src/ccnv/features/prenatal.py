"""Prenatal model training, inference, and 4D ray SCA classification."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import scipy.special
from numpy.typing import NDArray
from sklearn.model_selection import KFold

from ccnv.features.binning import (
    align_grid_indices,
    compute_bin_exposure,
)
from ccnv.features.cohort import Cohort, rebin_cohort, validate_cohort
from ccnv.features.fragmentomics import (
    extract_sample_fragmentomics,
    fit_cohort_fragmentomics_model,
    resolve_fisher_moment_weights,
)
from ccnv.features.normalization import (
    apply_lower_tail_deviance_floor,
    estimate_adjacent_dispersion_tau,
    shrink_kish_kappa,
)
from ccnv.features.prenatal_grid import (
    col_f64,
    col_i64,
    compute_5k_msy_exposure,
    compute_100k_profiles,
    compute_covariate_chrx_ratio,
    extract_ref_contig_roles,
    fit_sample_gc_exponent,
    prepare_5k_msy_sample,
    prepare_covariate_multipliers,
    rebin_track_1d,
    resolve_chry_anchor_ff,
    resolve_contig_masks,
    sample_grid,
)
from ccnv.features.rebinning import rebin_sample_buffers
from ccnv.features.sex import SexLabel
from ccnv.models import (
    BRANCH_PRENATAL_AUTOSOMAL,
    BRANCH_PRENATAL_FFY,
    DEFAULT_MIN_MAPPABILITY,
    ActiveFFSource,
    BinResult,
    BranchFitArrays,
    FetalFractionResult,
    FetalSex,
    InvalidInputError,
    PredictError,
    PrenatalModelMetadata,
    SampleBuffers,
    SamplesheetEntry,
    SCAResult,
    SchemaError,
    decode_dictionary_contigs,
    decode_metadata,
    fixed_size_list_to_numpy,
    normalize_chr,
)
from ccnv.stats.fragmentomics import (
    compute_band_fisher_weights,
    compute_fragment_length_llr,
    predict_ff_fragmentomics,
)
from ccnv.stats.neighbors import (
    LN2_SQ,
    exact_poisson_log2_jensen_bias,
    exact_poisson_log_variance,
)
from ccnv.stats.pca import pca_irlba_numpy, project_huber_pca_wave
from ccnv.stats.prenatal_math import (
    PoissonFFYModel,
    classify_sca_ray_4d,
    classify_twin_pregnancy,
    compute_autosomal_density,
    effective_ffa_sigma,
    evaluate_poisson_msy_ffy,
    fit_forward_ffx_model,
    fit_msy_sex_mixture,
    fit_poisson_msy_ffy,
)
from ccnv.stats.regression import fit_logit_linear_ensemble

_REQUIRED_CHRY_REF_COLUMNS: Final[frozenset[str]] = frozenset(
    ("branch", "contig", "start", "end", "masked", "gc", "mappability", "pca_mean")
)
_REQUIRED_REF_COLUMNS: Final[frozenset[str]] = _REQUIRED_CHRY_REF_COLUMNS | frozenset(
    ("pca_loadings", "null_mean")
)
_BASE_SCA_PRIORS: Final[dict[str, float]] = dict.fromkeys(
    ("XX", "XY", "XO", "XXX", "XXY", "XYY"), 1e-6
)
_FEMALE_SCA_PRIORS: Final[dict[str, float]] = _BASE_SCA_PRIORS | {
    "XX": 0.90,
    "XO": 0.05,
    "XXX": 0.05,
}
_MALE_SCA_PRIORS: Final[dict[str, float]] = _BASE_SCA_PRIORS | {
    "XY": 0.90,
    "XXY": 0.05,
    "XYY": 0.05,
}

__all__ = ["fit_prenatal_models", "normalize_prenatal_chry", "predict_prenatal"]


def _evaluate_sca_and_active_ff(
    fetal_sex: FetalSex,
    ffy: float | None,
    ffy_se: float | None,
    ffx: float,
    ffa: float,
    chrx_ratio: float,
    meta: PrenatalModelMetadata,
    *,
    ff_size: float | None = None,
    chrx_llr_fetal: float | None = None,
) -> tuple[SCAResult, float, float, ActiveFFSource]:
    """Evaluate 4D ray SCA classification and Student-t4 fused active FF."""
    mu_xf = float(meta.female_chrx_ratio_mean)
    std_xf = max(float(meta.female_chrx_ratio_std), 1e-4)
    abs_a1 = max(abs(float(meta.alpha_1)), 1e-6)
    f_y = float(ffy) if (fetal_sex == "M" and ffy is not None) else 0.0
    se_y = float(ffy_se) if (fetal_sex == "M" and ffy_se is not None) else 0.0
    has_alt = ff_size is not None or (fetal_sex == "M" and ffy is not None)
    sigma_ffa_eff = effective_ffa_sigma(
        float(meta.ffa_std), meta.olm_weights, has_alt_ff=has_alt
    )
    priors = _FEMALE_SCA_PRIORS if fetal_sex == "F" else _MALE_SCA_PRIORS
    cls_res = classify_sca_ray_4d(
        delta_x=float(0.5 * (float(chrx_ratio) - mu_xf) / abs_a1),
        sigma_x=max(0.5 * std_xf / abs_a1, 0.005),
        ffy=float(ffy) if (fetal_sex == "M" and ffy is not None) else None,
        sigma_ffy=max(math.hypot(se_y, float(meta.ffy_std)), 0.005),
        ffa=ffa,
        sigma_ffa=sigma_ffa_eff,
        ff_size=ff_size,
        sigma_size=max(float(meta.ff_size_std), 0.005),
        chrx_llr_fetal=chrx_llr_fetal,
        clinical_priors=priors,
    )
    sca = SCAResult(
        call=cls_res.call,
        is_aneuploid=cls_res.call not in ("XX", "XY"),
        confidence=float(np.clip(cls_res.posterior, 0.5, 1.0)),
        chrx_zscore=float((chrx_ratio - mu_xf) / std_xf),
        ff_delta=float(ffx - f_y) if fetal_sex == "M" else None,
    )
    return sca, cls_res.active_ff, cls_res.active_ff_se, cls_res.active_ff_source


def _compute_effective_ff(
    active_ff: float, bufs: SampleBuffers, meta: PrenatalModelMetadata
) -> float:
    """Compute Fisher-weighted effective fetal fraction when paired-end band weights apply."""
    mu_f, mu_m = resolve_fisher_moment_weights(bufs, meta)
    if mu_f == 1.0 and mu_m == 1.0:
        return float(active_ff)
    f_val = float(np.clip(active_ff, 0.0, 1.0))
    denom = f_val * mu_f + (1.0 - f_val) * mu_m
    return float(f_val * mu_f / denom) if denom > 0.0 else float(active_ff)


def fit_prenatal_models(
    sample_tables: Cohort | Sequence[SampleBuffers | pa.Table],
    entries: Sequence[SamplesheetEntry],
    *,
    prenatal_bin_size: int = 100_000,
    cohort_gc: NDArray[np.floating] | None = None,
    cohort_mappability: NDArray[np.floating] | None = None,
    cohort_n_mask: NDArray[np.bool_] | None = None,
    min_mappability: float = DEFAULT_MIN_MAPPABILITY,
    autosomes: Sequence[str] | None = None,
    sex_contigs: Sequence[str] | None = None,
    sex_signal_contig: str | None = None,
    seed: int = 42,
) -> tuple[PrenatalModelMetadata, tuple[BranchFitArrays, ...], tuple[SexLabel, ...]]:
    """Fit FFY, FFA, FFX, and fragmentomic models across a prenatal cohort."""
    if isinstance(sample_tables, Cohort):
        cohort_5k = sample_tables
    elif len(sample_tables) == 0:
        raise InvalidInputError("sample_tables must contain at least one sample table")
    else:
        cohort_5k = validate_cohort(sample_tables)

    n_samples = int(cohort_5k.counts.shape[1])
    if n_samples == 0:
        raise InvalidInputError("sample_tables must contain at least one sample table")
    if len(entries) != n_samples:
        raise InvalidInputError(
            f"sample_tables ({n_samples}) != entries ({len(entries)})"
        )
    if prenatal_bin_size <= 0:
        raise InvalidInputError("prenatal_bin_size must be a positive integer")

    c_5k = list(cohort_5k.contigs)
    s_5k, e_5k, counts_5k = cohort_5k.starts, cohort_5k.ends, cohort_5k.counts
    auto_mask_5k, _, chry_mask_5k = resolve_contig_masks(
        c_5k,
        autosomes=autosomes,
        sex_contigs=sex_contigs,
        sex_signal_contig=sex_signal_contig,
    )
    n_chry_bins = int(np.count_nonzero(chry_mask_5k))
    _, c_ids_5k = np.unique([normalize_chr(c) for c in c_5k], return_inverse=True)
    c_ids_i16 = c_ids_5k.astype(np.int16)
    exp_5k = compute_bin_exposure(s_5k, e_5k, cohort_5k.bin_size)
    eff_n_mask = cohort_n_mask
    if eff_n_mask is None and cohort_gc is not None:
        raw_gc_arr = np.asarray(cohort_gc, dtype=np.float64)
        if np.any(np.isfinite(raw_gc_arr) & (raw_gc_arr >= 0.0) & (raw_gc_arr <= 1.0)):
            eff_n_mask = ~np.isfinite(raw_gc_arr)
    gc_5k, map_5k = prepare_covariate_multipliers(
        cohort_gc, cohort_mappability, eff_n_mask, min_mappability, len(c_5k)
    )
    r_auto_5k = np.zeros(n_samples, dtype=np.float64)
    sample_gc_betas = np.ones(n_samples, dtype=np.float64)
    for i in range(n_samples):
        col_i = counts_5k[:, i].astype(np.float64)
        beta_i = fit_sample_gc_exponent(col_i, exp_5k, map_5k, gc_5k, auto_mask_5k)
        sample_gc_betas[i] = beta_i
        gc_i = (
            np.power(np.maximum(gc_5k, 1e-3), beta_i).astype(np.float32)
            if beta_i != 1.0
            else gc_5k
        )
        r_auto_5k[i] = compute_autosomal_density(
            col_i, c_ids_i16, exp_5k, map_5k, gc_i, auto_mask_5k
        )
    labeled_ff = np.array(
        [
            float(e.ff) if (e.ff is not None and math.isfinite(e.ff)) else np.nan
            for e in entries
        ],
        dtype=np.float64,
    )

    ffy_null_mean = np.zeros(n_chry_bins, dtype=np.float64)
    if n_chry_bins > 0:
        msy_counts = counts_5k[chry_mask_5k, :].astype(np.float64)
        map_y = map_5k[chry_mask_5k]
        gc_y_base = np.maximum(gc_5k[chry_mask_5k].astype(np.float64), 1e-3)
        msy_exposure = (
            (exp_5k[chry_mask_5k] * map_y).astype(np.float64)[:, np.newaxis]
            * np.power(gc_y_base[:, np.newaxis], sample_gc_betas[np.newaxis, :])
            * r_auto_5k[np.newaxis, :]
        )
        q25_rate = np.quantile(
            msy_counts / np.maximum(msy_exposure, 1e-6), 0.25, axis=1
        )
        mix_bins = (map_y >= 0.80) & (q25_rate <= 0.15)
        if not np.any(mix_bins):
            mix_bins = (
                (map_y >= 0.50)
                if np.any(map_y >= 0.50)
                else np.ones(n_chry_bins, np.bool_)
            )
        sex_mix = fit_msy_sex_mixture(
            np.sum(msy_counts[mix_bins], axis=0),
            np.sum(msy_exposure[mix_bins], axis=0),
            bic_delta_min=6.0,
        )
        sex_codes = np.array(
            [1 if e.sex == "M" else (0 if e.sex == "F" else -1) for e in entries]
        )
        is_male_arr = np.where(
            sex_codes >= 0, sex_codes == 1, sex_mix.bimodal & (sex_mix.p_male > 0.5)
        )
        quarantined = (
            (sex_mix.p_male > 0.02) & (sex_mix.p_male < 0.98)
            if sex_mix.bimodal
            else np.zeros(n_samples, dtype=np.bool_)
        )
        clean_f = (~is_male_arr) & (~quarantined)
        clean_m = is_male_arr & (~quarantined)
        clean_f = clean_f if np.any(clean_f) else ~is_male_arr
        clean_m = clean_m if np.any(clean_m) else is_male_arr

        poisson_model = fit_poisson_msy_ffy(
            msy_counts, msy_exposure, map_5k[chry_mask_5k], clean_f, clean_m, labeled_ff
        )
        final_ffy = np.zeros(n_samples, dtype=np.float64)
        male_ffy_se = np.zeros(n_samples, dtype=np.float64)
        for idx in np.flatnonzero(is_male_arr):
            final_ffy[idx], male_ffy_se[idx] = evaluate_poisson_msy_ffy(
                msy_counts[:, int(idx)], msy_exposure[:, int(idx)], poisson_model
            )

        cal_m = clean_m & np.isfinite(labeled_ff)
        if int(np.count_nonzero(cal_m)) >= 2:
            resid = final_ffy[cal_m] - labeled_ff[cal_m]
            ffy_std = max(float(np.sqrt(np.mean(resid * resid))), 0.005)
        else:
            ffy_std = (
                max(float(np.median(male_ffy_se[clean_m])), 0.005)
                if np.any(clean_m)
                else 0.01
            )

        clean_idx = poisson_model.clean_bin_indices
        ffy_slope = float(poisson_model.kappa_cal)
        if clean_idx.size > 0 and np.any(clean_m):
            m_cols = np.flatnonzero(clean_m)
            c_m_sum = np.sum(msy_counts[np.ix_(clean_idx, m_cols)], axis=1)
            e_m_sub = msy_exposure[np.ix_(clean_idx, m_cols)]
            e_bg_sum = poisson_model.lambda_bg * np.sum(e_m_sub, axis=1)
            f_m_pos = np.maximum(final_ffy[m_cols], 0.0)
            e_f1_sum = e_m_sub @ (0.5 * f_m_pos / max(ffy_slope, 1e-6))
            eta_raw = (np.maximum(c_m_sum - e_bg_sum, 0.0) + 3.0) / (e_f1_sum + 3.0)
            keep_m = ~((e_f1_sum >= 15.0) & ((eta_raw > 3.5) | (eta_raw < 0.15)))
            if np.any(keep_m) and not np.all(keep_m):
                clean_idx = clean_idx[keep_m]
                poisson_model = PoissonFFYModel(
                    clean_idx,
                    poisson_model.lambda_bg[keep_m],
                    poisson_model.kappa_cal,
                    poisson_model.pooled_bg_rate,
                )
                c_m_sum = c_m_sum[keep_m]
                e_bg_sum, e_f1_sum = e_bg_sum[keep_m], e_f1_sum[keep_m]
                eta_raw = eta_raw[keep_m]
                for idx in np.flatnonzero(is_male_arr):
                    i_m = int(idx)
                    final_ffy[idx], male_ffy_se[idx] = evaluate_poisson_msy_ffy(
                        msy_counts[:, i_m], msy_exposure[:, i_m], poisson_model
                    )
            exp_m_tot = e_bg_sum + e_f1_sum
            k_win = np.ones(min(3, c_m_sum.size), dtype=np.float64)
            c_win = np.convolve(c_m_sum, k_win, mode="same")
            e_win = np.convolve(exp_m_tot, k_win, mode="same")
            sig_shift = np.abs(c_win - e_win) > 3.5 * np.sqrt(np.maximum(e_win, 1.0))
            base_eta = (
                float(np.median(eta_raw[~sig_shift])) if np.any(~sig_shift) else 1.0
            )
            eta_shrunk = np.where(sig_shift, eta_raw, base_eta)
            eta_mean = float(np.mean(eta_shrunk)) if eta_shrunk.size > 0 else 1.0
            eta_norm = eta_shrunk / max(eta_mean, 1e-6)
            ffy_null_mean[clean_idx] = np.log2(np.clip(eta_norm, 0.1, 8.0))

        informative_mask = np.zeros(n_chry_bins, dtype=np.bool_)
        informative_mask[clean_idx] = True
        ffy_weights = np.zeros(n_chry_bins, dtype=np.float64)
        ffy_weights[clean_idx] = poisson_model.lambda_bg

        msy_sums = np.sum(
            msy_counts[clean_idx] if clean_idx.size else msy_counts, axis=0
        )
        yfracs = msy_sums / np.maximum(np.sum(counts_5k, axis=0, dtype=np.float64), 1.0)
        y_f, y_m = yfracs[clean_f], yfracs[clean_m]
        if y_f.size > 0 and y_m.size > 0:
            max_f, min_m = float(np.max(y_f)), float(np.min(y_m))
            if max_f < min_m:
                std_f = float(np.std(y_f)) if y_f.size >= 2 else 0.0
                fem_upper = max(
                    max_f * 2.5 + 2e-4, float(np.median(y_f)) + 6.0 * std_f + 2e-4
                )
                yfrac_threshold = min(0.5 * (max_f + min_m), fem_upper)
            else:
                yfrac_threshold = 0.5 * float(np.median(y_f) + np.median(y_m))
        elif y_f.size > 0:
            yfrac_threshold = max(float(np.max(y_f)) * 2.0 + 1e-4, 0.002)
        else:
            yfrac_threshold = 0.5 * float(np.min(y_m)) if y_m.size > 0 else 0.005
    else:
        informative_mask = np.zeros(0, dtype=np.bool_)
        ffy_weights = np.zeros(0, dtype=np.float64)
        ffy_slope, ffy_std, yfrac_threshold = 1.0, 0.01, 0.005
        final_ffy = np.zeros(n_samples, dtype=np.float64)
        is_male_arr = np.array([e.sex == "M" for e in entries], dtype=np.bool_)
        quarantined = np.zeros(n_samples, dtype=np.bool_)
        poisson_model = PoissonFFYModel(
            np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), 1.0, 0.0
        )

    unlabeled_m = np.where(is_male_arr, np.maximum(0.0, final_ffy), np.nan)
    ff_targets = np.where(
        quarantined, np.nan, np.where(np.isfinite(labeled_ff), labeled_ff, unlabeled_m)
    )

    cohort_100k = rebin_cohort(cohort_5k, prenatal_bin_size)
    c_100k, s_100k, e_100k = (
        list(cohort_100k.contigs),
        cohort_100k.starts,
        cohort_100k.ends,
    )
    auto_m_100k, chrx_m_100k, _ = resolve_contig_masks(
        c_100k,
        autosomes=autosomes,
        sex_contigs=sex_contigs,
        sex_signal_contig=sex_signal_contig,
    )
    full_log2_100k, chrx_ratios = compute_100k_profiles(
        cohort_100k.counts,
        c_100k,
        s_100k,
        e_100k,
        prenatal_bin_size,
        auto_m_100k,
        chrx_m_100k,
    )
    log2_ratios_100k = full_log2_100k[:, auto_m_100k]

    n_auto_100k = int(np.count_nonzero(auto_m_100k))
    clean_rows = (
        ~quarantined if np.any(~quarantined) else np.ones(n_samples, dtype=np.bool_)
    )
    auto_masked = np.mean(np.isfinite(log2_ratios_100k[clean_rows]), axis=0) < 0.99
    if np.all(auto_masked):
        auto_masked[:] = False

    raw_ratios = np.where(np.isfinite(log2_ratios_100k), log2_ratios_100k, np.nan)
    with np.errstate(all="ignore"):
        mean_features = np.nan_to_num(
            np.nanmean(raw_ratios[clean_rows], axis=0), nan=0.0
        )
    x_autosomal = np.where(
        np.isnan(raw_ratios), mean_features[np.newaxis, :], raw_ratios
    )

    unmasked_cols = ~auto_masked
    n_unmasked = int(np.count_nonzero(unmasked_cols))
    n_feat = min(50, max(1, n_samples - 1), max(1, n_unmasked))
    x_sub = x_autosomal[:, unmasked_cols]
    has_pca_var = n_unmasked > 0 and float(np.var(x_sub)) >= 1e-12
    pca_center = np.zeros(n_auto_100k, dtype=np.float64)
    if not has_pca_var:
        if n_unmasked > 0:
            pca_center[unmasked_cols] = np.mean(x_sub, axis=0)
        scores = np.zeros((n_samples, n_feat), dtype=np.float64)
        pca_rotation = np.zeros((n_auto_100k, n_feat), dtype=np.float64)
    else:
        scores, sub_center, sub_rot = pca_irlba_numpy(
            x_sub, n_components=n_feat, seed=seed
        )
        pca_center[unmasked_cols] = sub_center
        pca_rotation = np.zeros((n_auto_100k, sub_rot.shape[1]), dtype=np.float64)
        pca_rotation[unmasked_cols, :] = sub_rot

    ffa_valid = (~quarantined) & np.isfinite(ff_targets)
    if int(np.count_nonzero(ffa_valid)) >= 3:
        v_idx = np.flatnonzero(ffa_valid)
        ens_fit = fit_logit_linear_ensemble(
            scores[v_idx], ff_targets[v_idx], l1_ratios=(0.1, 0.5), seed=seed
        )
        w_ens, b_ens, sigma_ffa = ens_fit.weights, ens_fit.intercept, ens_fit.oof_rmse
        y_cv = np.clip(ff_targets[v_idx], 0.005, 0.85)
        if has_pca_var and float(np.ptp(y_cv)) >= 1e-4:
            oof_p = np.empty(v_idx.size, dtype=np.float64)
            kf = KFold(n_splits=min(5, v_idx.size), shuffle=True, random_state=seed)
            for tr_r, va_r in kf.split(v_idx):
                tr_a, va_a = v_idx[tr_r], v_idx[va_r]
                p_m = np.ones(n_samples, dtype=np.bool_)
                p_m[va_a] = False
                f_k = min(n_feat, max(1, int(np.count_nonzero(p_m)) - 1), n_unmasked)
                _, fc, fr = pca_irlba_numpy(x_sub[p_m], n_components=f_k, seed=seed)
                s_tr, s_va = (x_sub[tr_a] - fc) @ fr, (x_sub[va_a] - fc) @ fr
                ffit = fit_logit_linear_ensemble(
                    s_tr, ff_targets[tr_a], l1_ratios=(0.1, 0.5), seed=seed
                )
                oof_p[va_r] = scipy.special.expit(s_va @ ffit.weights + ffit.intercept)
            sigma_ffa = max(float(np.sqrt(np.mean(np.square(oof_p - y_cv)))), 0.005)
    else:
        w_ens = np.zeros(scores.shape[1], dtype=np.float64)
        mean_ff = float(np.mean(ff_targets[ffa_valid])) if np.any(ffa_valid) else 0.05
        b_ens = float(scipy.special.logit(float(np.clip(mean_ff, 0.005, 0.5))))
        sigma_ffa = 0.25

    f_rx = np.where(is_male_arr, ff_targets, 0.0)
    alpha_0, raw_alpha_1, sigma_x_sys = fit_forward_ffx_model(
        chrx_ratios, f_rx, (~quarantined) & np.isfinite(chrx_ratios) & np.isfinite(f_rx)
    )
    alpha_1 = float(raw_alpha_1) if raw_alpha_1 < -1e-6 else -0.5

    raw_metas = (
        [
            t.meta.fragmentomics
            if isinstance(t, SampleBuffers)
            else extract_sample_fragmentomics(decode_metadata(t.schema.metadata))
            for t in sample_tables
        ]
        if isinstance(sample_tables, Sequence)
        else [extract_sample_fragmentomics(m) for m in cohort_5k.sample_metadata]
    )
    frag_fit = fit_cohort_fragmentomics_model(
        raw_metas, is_male_arr, ff_targets, quarantined=quarantined, seed=seed
    )
    p_fet = p_mat = w_frag = b_frag = m_frag = s_frag = band_w_list = mu_f_val = (
        mu_m_val
    ) = None
    std_size = 0.018
    if frag_fit is not None:
        p_fet, p_mat = frag_fit.p_fetal_density, frag_fit.p_maternal_density
        w_frag, b_frag = frag_fit.ff_frag_weights, frag_fit.ff_frag_bias
        m_frag, s_frag = frag_fit.ff_frag_feature_mean, frag_fit.ff_frag_feature_scale
        std_size = float(frag_fit.ff_size_std)
        bw, mf, mm = compute_band_fisher_weights(p_fet, p_mat, f0=0.08)
        band_w_list = [float(x) for x in bw]
        mu_f_val, mu_m_val = float(mf), float(mm)

    q_list = (
        [int(x) for x in np.flatnonzero(quarantined)] if np.any(quarantined) else None
    )
    prenatal_meta = PrenatalModelMetadata(
        prenatal_bin_size=prenatal_bin_size,
        yfrac_threshold=float(yfrac_threshold),
        ffy_slope=float(ffy_slope),
        olm_weights=[float(x) for x in w_ens],
        olm_bias=float(b_ens),
        female_chrx_ratio_mean=float(alpha_0),
        female_chrx_ratio_std=float(sigma_x_sys),
        alpha_1=alpha_1,
        ffy_std=float(ffy_std),
        ffa_std=float(sigma_ffa),
        ffa_logit_mode=True,
        p_fetal_density=p_fet,
        p_maternal_density=p_mat,
        ff_frag_weights=w_frag,
        ff_frag_bias=b_frag,
        ff_frag_feature_mean=m_frag,
        ff_frag_feature_scale=s_frag,
        ff_size_std=std_size,
        msy_bg_rate=float(poisson_model.pooled_bg_rate),
        clean_msy_bins=[int(x) for x in poisson_model.clean_bin_indices],
        quarantined_samples=q_list,
        band_fisher_weights=band_w_list,
        mu_fetal_w=mu_f_val,
        mu_maternal_w=mu_m_val,
    )
    auto_idx = np.flatnonzero(auto_m_100k)
    prenatal_auto_fit = BranchFitArrays(
        branch=BRANCH_PRENATAL_AUTOSOMAL,
        contigs=tuple(c_100k[int(i)] for i in auto_idx),
        starts=s_100k[auto_idx],
        ends=e_100k[auto_idx],
        masked=auto_masked,
        pca_mean=pca_center,
        pca_loadings=np.asarray(pca_rotation, dtype=np.float32),
        null_mean=mean_features,
    )
    chry_idx = np.flatnonzero(chry_mask_5k)
    prenatal_ffy_fit = BranchFitArrays(
        branch=BRANCH_PRENATAL_FFY,
        contigs=tuple(c_5k[int(i)] for i in chry_idx),
        starts=s_5k[chry_idx],
        ends=e_5k[chry_idx],
        masked=~informative_mask,
        pca_mean=ffy_weights,
        null_mean=ffy_null_mean,
        gc=cohort_gc[chry_idx] if cohort_gc is not None else None,
        mappability=(
            cohort_mappability[chry_idx] if cohort_mappability is not None else None
        ),
    )
    sex_labels: tuple[SexLabel, ...] = tuple(
        "male" if m else "female" for m in is_male_arr.tolist()
    )
    return prenatal_meta, (prenatal_auto_fit, prenatal_ffy_fit), sex_labels


def predict_prenatal(
    sample: SampleBuffers,
    reference_table: pa.Table,
    prenatal_meta: PrenatalModelMetadata,
    *,
    min_ff_warning: float = 0.04,
    sample_gc_override: NDArray[np.floating] | None = None,
    sample_mappability_override: NDArray[np.floating] | None = None,
    sample_n_mask_override: NDArray[np.bool_] | None = None,
    min_mappability: float = DEFAULT_MIN_MAPPABILITY,
    twins: bool = False,
) -> FetalFractionResult:
    """Predict fetal sex, FFY, FFA, FFX, SCA status, and active fetal fraction."""
    pm, rt = prenatal_meta, reference_table
    missing_cols = sorted(_REQUIRED_REF_COLUMNS - set(rt.column_names))
    if missing_cols:
        raise PredictError(
            f"reference table is missing required columns: {', '.join(missing_cols)}"
        )

    branch_col = rt.column("branch")
    ref_branches = {str(v) for v in pc.unique(branch_col).to_pylist() if v is not None}
    for req in (BRANCH_PRENATAL_AUTOSOMAL, BRANCH_PRENATAL_FFY):
        if req not in ref_branches:
            raise PredictError(f"reference table is missing '{req}' branch")

    ffy_ref = rt.filter(pc.equal(branch_col, BRANCH_PRENATAL_FFY))
    auto_ref = rt.filter(pc.equal(branch_col, BRANCH_PRENATAL_AUTOSOMAL))
    roles = extract_ref_contig_roles(rt, ffy_ref)

    msy_prep = prepare_5k_msy_sample(sample, rt, ffy_ref, pm.clean_msy_bins)
    bufs, bufs_5k, ffy_bin = msy_prep[0], msy_prep[1], msy_prep[2]
    ffy_align, sample_chry_c, clean_idx = msy_prep[6], msy_prep[7], msy_prep[9]
    counts_5k = np.asarray(bufs_5k.count, dtype=np.float64)
    hit_y = ffy_align >= 0
    chry_reads = float(
        np.sum(sample_chry_c[clean_idx] if clean_idx.size else sample_chry_c)
    )
    chry_score = chry_reads / max(float(np.sum(counts_5k)), 1.0)
    fetal_sex: FetalSex = "M" if chry_score >= pm.yfrac_threshold else "F"

    ffy: float | None = None
    ffy_se: float | None = None
    ovr = (sample_gc_override, sample_mappability_override, sample_n_mask_override)
    if fetal_sex == "M" and np.any(hit_y):
        sample_chry_exp, _ = compute_5k_msy_exposure(
            bufs, bufs_5k, rt, ffy_ref, ffy_bin, ffy_align, roles, ovr, min_mappability
        )
        ref_w = col_f64(ffy_ref, "pca_mean")
        lam_bg = ref_w[clean_idx] if clean_idx.size else np.empty(0, dtype=np.float64)
        p_model = PoissonFFYModel(
            clean_idx, lam_bg, pm.ffy_slope, float(pm.msy_bg_rate or 0.0)
        )
        ffy_raw, ffy_se_val = evaluate_poisson_msy_ffy(
            sample_chry_c, sample_chry_exp, p_model
        )
        ffy, ffy_se = max(0.0, float(ffy_raw)), float(ffy_se_val)

    p_bin = pm.prenatal_bin_size
    bufs_100k = (
        bufs if bufs.meta.bin_size == p_bin else rebin_sample_buffers(bufs, p_bin)
    )
    counts_100k = np.asarray(bufs_100k.count, dtype=np.float64)
    c_100k, s_100k, e_100k = sample_grid(bufs_100k.meta)
    auto_m, chrx_m, _ = resolve_contig_masks(
        c_100k, autosomes=roles[0], sex_contigs=roles[1], sex_signal_contig=roles[2]
    )
    full_log2, chrx_arr = compute_100k_profiles(
        counts_100k[:, None], c_100k, s_100k, e_100k, p_bin, auto_m, chrx_m
    )
    cov_chrx = compute_covariate_chrx_ratio(
        bufs, rt, pm, overrides=ovr, min_mappability=min_mappability
    )
    chrx_ratio = float(cov_chrx) if cov_chrx is not None else float(chrx_arr[0])

    ref_null_mean = col_f64(auto_ref, "null_mean")
    ref_pca_center = col_f64(auto_ref, "pca_mean")
    ld_col = auto_ref.column("pca_loadings").combine_chunks()
    if not isinstance(ld_col.type, pa.FixedSizeListType):
        raise SchemaError(
            f"expected FixedSizeListType for 'pca_loadings', found {ld_col.type}"
        )
    n_olm = len(pm.olm_weights)
    ref_rot = np.zeros((auto_ref.num_rows, n_olm), dtype=np.float64)
    if ld_col.type.list_size > 0 and auto_ref.num_rows > 0 and n_olm > 0:
        full_ld = fixed_size_list_to_numpy(ld_col, ld_col.type.list_size, np.float32)
        k_copy = min(n_olm, ld_col.type.list_size)
        ref_rot[:, :k_copy] = np.nan_to_num(
            full_ld[:, :k_copy].astype(np.float64), nan=0.0
        )

    ref_auto_c = decode_dictionary_contigs(auto_ref.column("contig"))
    ref_auto_s, ref_auto_e = col_i64(auto_ref, "start"), col_i64(auto_ref, "end")
    auto_c_100k = tuple(c_100k[int(i)] for i in np.flatnonzero(auto_m))
    if (
        auto_c_100k == ref_auto_c
        and np.array_equal(s_100k[auto_m], ref_auto_s)
        and np.array_equal(e_100k[auto_m], ref_auto_e)
    ):
        log2_ratios = full_log2[0, auto_m]
    else:
        aligned = align_grid_indices(
            c_100k, s_100k, e_100k, ref_auto_c, ref_auto_s, ref_auto_e, allow_alias=True
        )
        log2_ratios = np.full(ref_null_mean.size, np.nan, dtype=np.float64)
        hit_a = aligned >= 0
        log2_ratios[hit_a] = full_log2[0, aligned[hit_a]]

    centered = (
        np.where(np.isnan(log2_ratios), ref_null_mean, log2_ratios) - ref_pca_center
    )
    ref_masked = np.asarray(
        auto_ref.column("masked").to_numpy(zero_copy_only=False), np.bool_
    )
    usable_pca = (~ref_masked) & np.isfinite(centered)
    if n_olm > 0 and int(np.count_nonzero(usable_pca)) > n_olm:
        feat, _ = project_huber_pca_wave(
            centered[usable_pca], ref_rot[usable_pca].T, alpha=1e-4, clip=None
        )
    elif n_olm > 0 and np.any(usable_pca):
        feat = centered[usable_pca] @ ref_rot[usable_pca]
    else:
        feat = np.zeros(n_olm, dtype=np.float64)

    raw_ffa = float(np.dot(feat, np.asarray(pm.olm_weights))) + pm.olm_bias
    ffa = (
        float(scipy.special.expit(raw_ffa))
        if pm.ffa_logit_mode
        else float(np.clip(raw_ffa, 0.0, 1.0))
    )
    ffx = float((chrx_ratio - pm.female_chrx_ratio_mean) / pm.alpha_1)

    frag_meta = bufs.meta.fragmentomics
    ff_size: float | None = None
    chrx_llr_fetal: float | None = None
    if frag_meta is not None:
        ff_size = predict_ff_fragmentomics(
            frag_meta.hist_autosomal,
            frag_meta.end_motif_counts,
            paired_end=frag_meta.paired_end,
            chromatosome_mode_bp=frag_meta.chromatosome_mode_bp,
            sizing_shift_bp=frag_meta.sizing_shift_bp,
            helical_10bp_amp=frag_meta.helical_10bp_amplitude,
            motif_mds=frag_meta.motif_diversity_score,
            p_fetal_density=pm.p_fetal_density,
            p_maternal_density=pm.p_maternal_density,
            ff_frag_weights=pm.ff_frag_weights,
            ff_frag_bias=pm.ff_frag_bias or 0.0,
            ff_frag_feature_mean=pm.ff_frag_feature_mean,
            ff_frag_feature_scale=pm.ff_frag_feature_scale,
        )
        if (
            frag_meta.paired_end
            and int(np.sum(frag_meta.hist_chrx)) >= 50
            and pm.p_fetal_density is not None
            and pm.p_maternal_density is not None
        ):
            chrx_llr_fetal = compute_fragment_length_llr(
                frag_meta.hist_chrx,
                pm.p_fetal_density,
                pm.p_maternal_density,
                sizing_shift_bp=frag_meta.sizing_shift_bp,
                baseline_ff=ffa,
            )

    sca_res, active_ff, active_ff_se, active_ff_source = _evaluate_sca_and_active_ff(
        fetal_sex,
        ffy,
        ffy_se,
        ffx,
        ffa,
        chrx_ratio,
        pm,
        ff_size=ff_size,
        chrx_llr_fetal=chrx_llr_fetal,
    )
    abs_a1 = max(abs(float(pm.alpha_1)), 1e-6)
    delta_x_val = float(0.5 * (chrx_ratio - float(pm.female_chrx_ratio_mean)) / abs_a1)
    sigma_x_val = max(0.5 * max(float(pm.female_chrx_ratio_std), 1e-4) / abs_a1, 0.005)
    sigma_y_val = max(math.hypot(float(ffy_se or 0.0), float(pm.ffy_std)), 0.005)
    has_alt = ff_size is not None or (fetal_sex == "M" and ffy is not None)
    olm_norm = (
        float(np.linalg.norm(np.asarray(pm.olm_weights, dtype=np.float64)))
        if pm.olm_weights
        else 0.0
    )
    twin_cls = classify_twin_pregnancy(
        fetal_sex=fetal_sex,
        chry_score=chry_score,
        yfrac_threshold=pm.yfrac_threshold,
        ffy=ffy,
        sigma_ffy=sigma_y_val,
        delta_x=delta_x_val,
        sigma_x=sigma_x_val,
        ffa=ffa,
        sigma_ffa=effective_ffa_sigma(
            float(pm.ffa_std), pm.olm_weights, has_alt_ff=has_alt
        ),
        ff_size=ff_size,
        sigma_size=max(float(pm.ff_size_std), 0.005),
        active_ff=active_ff,
        active_ff_se=active_ff_se,
        active_ff_source=active_ff_source,
        twins=twins,
        olm_norm=olm_norm,
    )
    if (
        twin_cls.gestation == "twin"
        and twin_cls.fetal_sexes == ("F", "M")
        and sca_res.call == "DISCORDANT_MIXED"
    ):
        sca_2d = classify_sca_ray_4d(
            delta_x_val,
            sigma_x_val,
            float(ffy or 0.0),
            sigma_y_val,
            None,
            sigma_y_val,
            None,
            sigma_y_val,
            chrx_llr_fetal=chrx_llr_fetal,
            clinical_priors=_MALE_SCA_PRIORS,
        )
        sca_res = SCAResult(
            sca_2d.call,
            sca_2d.call not in ("XX", "XY"),
            float(np.clip(sca_2d.posterior, 0.5, 1.0)),
            sca_res.chrx_zscore,
            sca_res.ff_delta,
        )
    active_ff, active_ff_se, active_ff_source = (
        twin_cls.active_ff,
        twin_cls.active_ff_se,
        twin_cls.active_ff_source,
    )
    effective_ff = _compute_effective_ff(active_ff, bufs, pm)
    low_ff_warning = bool(active_ff < min_ff_warning)
    warnings_list: list[str] = []
    if low_ff_warning:
        warnings_list.append(
            f"fetal fraction below warning threshold ({min_ff_warning:.2f})"
        )
    if sca_res.is_aneuploid:
        warnings_list.append(f"sex chromosome aneuploidy detected: {sca_res.call}")
    if not twins and twin_cls.gestation == "twin" and twin_cls.fetal_sexes is not None:
        warnings_list.append(
            f"suspected twin pregnancy detected: {''.join(twin_cls.fetal_sexes)}"
        )

    return FetalFractionResult(
        fetal_sex=fetal_sex,
        active_ff=active_ff,
        active_ff_source=active_ff_source,
        ffy=ffy,
        ffx=ffx,
        ffa=ffa,
        chrx_ratio=chrx_ratio,
        chry_score=chry_score,
        yfrac_threshold=pm.yfrac_threshold,
        sca=sca_res,
        ff_size=ff_size,
        low_ff_warning=low_ff_warning,
        warnings=tuple(warnings_list),
        active_ff_se=active_ff_se,
        effective_ff=effective_ff,
        twin=twin_cls.twin,
    )


def normalize_prenatal_chry(
    sample: SampleBuffers,
    reference_table: pa.Table,
    prenatal_meta: PrenatalModelMetadata,
    prenatal_result: FetalFractionResult,
    *,
    sample_gc_override: NDArray[np.floating] | None = None,
    sample_mappability_override: NDArray[np.floating] | None = None,
    sample_n_mask_override: NDArray[np.bool_] | None = None,
    min_mappability: float = DEFAULT_MIN_MAPPABILITY,
) -> BinResult | None:
    """Normalize 5 kb clean MSY chrY bins against the FF-calibrated 1-copy Poisson baseline."""
    if prenatal_result.fetal_sex != "M":
        return None
    pm, rt = prenatal_meta, reference_table
    missing_cols = sorted(_REQUIRED_CHRY_REF_COLUMNS - set(rt.column_names))
    if missing_cols:
        raise PredictError(
            f"reference table is missing required columns: {', '.join(missing_cols)}"
        )

    ffy_ref = rt.filter(pc.equal(rt.column("branch"), BRANCH_PRENATAL_FFY))
    n_y = ffy_ref.num_rows
    if n_y == 0:
        return None

    (
        bufs,
        bufs_5k,
        ffy_bin,
        ffy_ref_c,
        ffy_ref_s,
        ffy_ref_e,
        ffy_align,
        sample_chry_c,
        ref_ffy_masked,
        clean_idx,
    ) = prepare_5k_msy_sample(sample, rt, ffy_ref, pm.clean_msy_bins)
    hit_y = ffy_align >= 0
    if not np.any(hit_y):
        return None

    roles = extract_ref_contig_roles(rt, ffy_ref)
    ovr = (sample_gc_override, sample_mappability_override, sample_n_mask_override)
    e_b, n_auto_usable = compute_5k_msy_exposure(
        bufs, bufs_5k, rt, ffy_ref, ffy_bin, ffy_align, roles, ovr, min_mappability
    )
    clean_mask = np.zeros(n_y, dtype=np.bool_)
    clean_mask[clean_idx] = True

    usable = (
        clean_mask
        & (~ref_ffy_masked)
        & hit_y
        & np.isfinite(e_b)
        & (e_b > 0.0)
        & np.isfinite(sample_chry_c)
        & (sample_chry_c >= 0.0)
    )
    ratios = np.full(n_y, np.nan, dtype=np.float64)
    zscores = np.full(n_y, np.nan, dtype=np.float64)
    weights = np.zeros(n_y, dtype=np.float64)
    expected_out = np.zeros(n_y, dtype=np.float64)
    bg_out = np.zeros(n_y, dtype=np.float64)
    ref_sizes = np.zeros(n_y, dtype=np.int64)
    baseline_log2_var = 0.0

    u_idx = np.flatnonzero(usable)
    if u_idx.size > 0:
        c_u, u_sample_idx, e_u = sample_chry_c[u_idx], ffy_align[u_idx], e_b[u_idx]
        eta_u = (
            np.power(
                2.0,
                np.where(
                    np.isfinite(nm := col_f64(ffy_ref, "null_mean")[u_idx]), nm, 0.0
                ),
            )
            if "null_mean" in ffy_ref.column_names
            else np.ones(u_idx.size, dtype=np.float64)
        )
        kappa_cal = max(float(pm.ffy_slope), 1e-6)
        raw_bg_u = np.maximum(col_f64(ffy_ref, "pca_mean")[u_idx], 0.0)
        f_anchor, se_anchor = resolve_chry_anchor_ff(
            c_u, e_u, raw_bg_u, eta_u, kappa_cal, pm, prenatal_result
        )
        if bufs.kappa_frag is None:
            f_eff, bg_scale = f_anchor, 1.0
        else:
            mu_f, mu_m = resolve_fisher_moment_weights(bufs, pm)
            is_fm = prenatal_result.fetal_sexes == ("F", "M")
            f_tot = float(
                np.clip(prenatal_result.active_ff if is_fm else f_anchor, 0.0, 1.0)
            )
            w_mix = (1.0 - f_tot) * mu_m + f_tot * mu_f
            f_eff = max((f_anchor * mu_f / w_mix) if w_mix > 0.0 else f_anchor, 1e-4)
            bg_scale = (mu_m / w_mix) if w_mix > 0.0 else 1.0

        sigma_f = math.sqrt(
            se_anchor**2
            + (max(f_anchor, 0.02) * min(float(pm.ffy_std) / 0.10, 0.12)) ** 2
        )
        baseline_log2_var = (sigma_f / (max(f_anchor, 0.02) * math.log(2.0))) ** 2

        bg_counts_u = e_u * bg_scale * raw_bg_u
        lam_u = bg_counts_u + e_u * eta_u * (0.5 * f_eff / kappa_cal)
        r_u = np.log2((c_u + 0.5) / (lam_u + 0.5)) - exact_poisson_log2_jensen_bias(
            lam_u
        )

        raw_kappa_5k = (
            bufs_5k.kappa_frag
            if bufs_5k.kappa_frag is not None
            else rebin_track_1d(bufs.kappa_frag, bufs.meta, bufs_5k.meta)
        )
        kap_sub = (
            np.asarray(raw_kappa_5k, dtype=np.float64)[u_sample_idx]
            if raw_kappa_5k is not None
            else None
        )
        kap_u = shrink_kish_kappa(c_u, kap_sub)
        shot_var_u = kap_u * exact_poisson_log_variance(lam_u) / LN2_SQ
        bio_var_u = np.zeros(u_idx.size, dtype=np.float64)
        if "bio_dispersion" in ffy_ref.column_names:
            bd = col_f64(ffy_ref, "bio_dispersion")[u_idx]
            bio_var_u = np.where(np.isfinite(bd) & (bd > 0.0), bd, 0.0) / LN2_SQ
        sig0_sq_u = np.maximum(shot_var_u + bio_var_u, 1e-12)
        tau_y = estimate_adjacent_dispersion_tau(
            u_idx,
            np.asarray(ffy_ref_c, dtype=str)[u_idx],
            r_u,
            sig0_sq_u,
            tau_min=1.0,
            tau_max=2.5,
        )
        sig_u = apply_lower_tail_deviance_floor(
            c_u, lam_u, r_u, kap_u, shot_var_u, bio_var_u, tau_y
        )
        ratios[u_idx], zscores[u_idx] = r_u, r_u / sig_u
        weights[u_idx], expected_out[u_idx] = 1.0 / np.square(sig_u), lam_u
        bg_out[u_idx] = bg_counts_u
        ref_sizes[u_idx] = max(n_auto_usable, int(u_idx.size))

    return BinResult(
        branch=BRANCH_PRENATAL_FFY,
        contigs=ffy_ref_c,
        starts=ffy_ref_s,
        ends=ffy_ref_e,
        ratios=ratios,
        zscores=zscores,
        weights=weights,
        reference_sizes=ref_sizes,
        masked=~usable,
        counts=sample_chry_c,
        expected_counts=expected_out,
        background_counts=bg_out,
        baseline_log2_var=baseline_log2_var,
    )
