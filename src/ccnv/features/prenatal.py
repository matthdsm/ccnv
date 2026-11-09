"""Prenatal model training, inference, SCA calling, and segment annotation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import Final

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import scipy.special
from numpy.typing import NDArray
from sklearn.model_selection import KFold

from ccnv.features.cohort import (
    Cohort,
    rebin_cohort,
    rebin_sample_table,
    validate_cohort,
)
from ccnv.models import (
    BRANCH_PRENATAL_AUTOSOMAL,
    BRANCH_PRENATAL_FFY,
    REFERENCE_SCHEMA,
    ActiveFFSource,
    FetalFractionResult,
    FetalSex,
    PrenatalModelMetadata,
    SamplesheetEntry,
    SCACall,
    SCAResult,
    SegmentResult,
    is_autosome_name,
    normalize_chr,
)
from ccnv.stats.pca import pca_irlba_numpy
from ccnv.stats.prenatal_math import (
    fit_lasso_fista,
    fit_rprop_mlp,
    infer_sex_gmm_1d,
    ols_fit,
    rlm_huber,
    train_olm,
)

_PAR1_START: Final[int] = 2_781_479
_YQ12_END: Final[int] = 28_800_000
_EXCLUDED_PRENATAL_AUTOSOMES: Final[frozenset[int]] = frozenset({13, 18, 21})

__all__ = [
    "annotate_prenatal_segments",
    "classify_sca",
    "fit_prenatal_models",
    "predict_prenatal",
]


def _is_preface_autosome(norm_contig: str) -> bool:
    """Return True if contig is a canonical autosome excluding 13, 18, 21."""
    return (
        is_autosome_name(norm_contig)
        and int(norm_contig) not in _EXCLUDED_PRENATAL_AUTOSOMES
    )


def _resolve_autosome_mask(norm_contigs: Sequence[str]) -> NDArray[np.bool_]:
    """Resolve boolean autosome mask for PREFACE, falling back to all autosomes."""
    mask = np.array([_is_preface_autosome(c) for c in norm_contigs], dtype=np.bool_)
    if not np.any(mask):
        mask = np.array([is_autosome_name(c) for c in norm_contigs], dtype=np.bool_)
    return mask


def _build_prenatal_branch_table(
    *,
    branch: str,
    contigs: Sequence[str],
    starts: NDArray[np.int64],
    ends: NDArray[np.int64],
    masked: NDArray[np.bool_],
    pca_mean: NDArray[np.float64],
    pca_loadings: list[list[float]] | None = None,
    null_mean: NDArray[np.float64] | None = None,
) -> pa.Table:
    n_bins = len(contigs)
    return pa.table(
        {
            "branch": pa.array([branch] * n_bins, type=pa.string()),
            "contig": pa.array(contigs, type=pa.string()),
            "start": pa.array(starts, type=pa.int64()),
            "end": pa.array(ends, type=pa.int64()),
            "masked": pa.array(masked, type=pa.bool_()),
            "gc": pa.array(np.zeros(n_bins, dtype=np.float32), type=pa.float32()),
            "mappability": pa.array(
                np.zeros(n_bins, dtype=np.float32), type=pa.float32()
            ),
            "pca_mean": pa.array(pca_mean, type=pa.float64()),
            "pca_loadings": pa.array(
                pca_loadings
                if pca_loadings is not None
                else [[] for _ in range(n_bins)],
                type=pa.list_(pa.float64()),
            ),
            "nearest_ref_indices": pa.array(
                [[] for _ in range(n_bins)], type=pa.list_(pa.int32())
            ),
            "mean_distance": pa.array(
                np.zeros(n_bins, dtype=np.float32), type=pa.float32()
            ),
            "null_mean": pa.array(
                null_mean
                if null_mean is not None
                else np.zeros(n_bins, dtype=np.float64),
                type=pa.float64(),
            ),
            "null_std": pa.array(np.zeros(n_bins, dtype=np.float64), type=pa.float64()),
        },
        schema=REFERENCE_SCHEMA,
    )


def _align_sample_to_ref(
    sample_contigs: Sequence[str],
    sample_starts: NDArray[np.int64],
    sample_ends: NDArray[np.int64],
    ref_table: pa.Table,
) -> NDArray[np.int32]:
    """Map reference rows to matching sample bin indices, or -1 if missing."""
    sample_index = {
        (normalize_chr(c), int(s), int(e)): idx
        for idx, (c, s, e) in enumerate(
            zip(sample_contigs, sample_starts, sample_ends, strict=True)
        )
    }
    ref_c = [normalize_chr(str(c)) for c in ref_table.column("contig").to_pylist()]
    ref_s = ref_table.column("start").to_numpy(zero_copy_only=False)
    ref_e = ref_table.column("end").to_numpy(zero_copy_only=False)
    aligned = np.full(ref_table.num_rows, -1, dtype=np.int32)
    for j, key in enumerate(zip(ref_c, ref_s, ref_e, strict=True)):
        idx = sample_index.get(key)
        if idx is not None:
            aligned[j] = idx
    return aligned


def _autosomal_density(
    counts: np.ndarray,
    widths: np.ndarray,
    auto_mask: np.ndarray,
    fallback_bin_size: int,
) -> float:
    """Calculate mean mappable autosomal read density (reads per bp)."""
    sub_counts = counts[auto_mask]
    sub_widths = widths[auto_mask]
    mappable = sub_counts > 0
    if np.any(mappable):
        return float(np.mean(sub_counts[mappable] / sub_widths[mappable]))
    return 1.0 / float(max(fallback_bin_size, 1))


def classify_sca(
    fetal_sex: FetalSex,
    ffy: float | None,
    ffx: float,
    preface: float,
    chrx_ratio: float,
    prenatal_meta: PrenatalModelMetadata,
    z_cut: float = 3.0,
) -> SCAResult:
    """Classify Sex Chromosome Aneuploidy (SCA) across seven phenotypes."""
    mu_f = float(prenatal_meta.female_chrx_ratio_mean)
    std_f = max(float(prenatal_meta.female_chrx_ratio_std), 1e-4)
    chrx_zscore = float((chrx_ratio - mu_f) / std_f)

    if abs(prenatal_meta.ffx_slope) > 1e-12:
        mu_ffx_f = prenatal_meta.ffx_intercept + prenatal_meta.ffx_slope * mu_f
        std_ffx_f = max(abs(prenatal_meta.ffx_slope) * std_f, 0.005)
        z_ffx = float((ffx - mu_ffx_f) / std_ffx_f)
    else:
        z_ffx = -chrx_zscore

    if fetal_sex == "F":
        if z_ffx >= z_cut:
            call: SCACall = "XO"
            is_aneuploid = True
            excess = abs(z_ffx) - z_cut
            confidence = min(1.0, max(0.5, 0.5 + 0.5 * min(excess / z_cut, 1.0)))
        elif z_ffx <= -z_cut:
            call = "XXX"
            is_aneuploid = True
            excess = abs(z_ffx) - z_cut
            confidence = min(1.0, max(0.5, 0.5 + 0.5 * min(excess / z_cut, 1.0)))
        else:
            call = "XX"
            is_aneuploid = False
            margin = z_cut - abs(z_ffx)
            confidence = min(1.0, max(0.5, 0.5 + 0.5 * (margin / z_cut)))

        return SCAResult(
            call=call,
            is_aneuploid=is_aneuploid,
            confidence=confidence,
            chrx_zscore=chrx_zscore,
            ff_delta=None,
        )

    # fetal_sex == "M"
    ffy_val = float(ffy) if ffy is not None else 0.0
    ff_delta = float(ffx - ffy_val)

    sigma_xy = max(abs(prenatal_meta.ffx_slope) * std_f, 0.015)
    sigma_pref = max(std_f, 0.015)
    z_xy = float((ffx - ffy_val) / sigma_xy)
    z_y_pref = float((ffy_val - preface) / sigma_pref)

    # 1. XYY: chrY doubled relative to autosomal/chrX
    if z_xy <= -z_cut and z_y_pref >= z_cut and (ffy_val / max(ffx, 0.01)) > 1.5:
        call = "XYY"
        is_aneuploid = True
        excess = min(abs(z_xy), abs(z_y_pref)) - z_cut
        confidence = min(1.0, max(0.5, 0.5 + 0.5 * min(excess / z_cut, 1.0)))
    # 2. XXY: male chrY with female diploid chrX baseline
    elif z_xy <= -z_cut and abs(z_ffx) < z_cut:
        call = "XXY"
        is_aneuploid = True
        excess = abs(z_xy) - z_cut
        confidence = min(1.0, max(0.5, 0.5 + 0.5 * min(excess / z_cut, 1.0)))
    # 3. XY: normal concordant male
    elif abs(z_xy) < z_cut and abs(z_y_pref) < z_cut:
        call = "XY"
        is_aneuploid = False
        margin = min(z_cut - abs(z_xy), z_cut - abs(z_y_pref))
        confidence = min(1.0, max(0.5, 0.5 + 0.5 * (margin / z_cut)))
    # 4. DISCORDANT_MIXED: otherwise
    else:
        call = "DISCORDANT_MIXED"
        is_aneuploid = True
        excess = max(abs(z_xy), abs(z_y_pref)) - z_cut
        confidence = min(1.0, max(0.5, 0.5 + 0.5 * min(excess / z_cut, 1.0)))

    return SCAResult(
        call=call,
        is_aneuploid=is_aneuploid,
        confidence=confidence,
        chrx_zscore=chrx_zscore,
        ff_delta=ff_delta,
    )


def fit_prenatal_models(
    sample_tables: Sequence[pa.Table] | Cohort,
    entries: Sequence[SamplesheetEntry],
    bin_size: int,
    prenatal_bin_size: int,
    seed: int = 42,
) -> tuple[pa.Table, pa.Table, PrenatalModelMetadata, tuple[bool, ...]]:
    """Fit FFY, PREFACE, and FFX models on cohort samples."""
    # 1. Extract 5 kb coordinates, chrY fractions, and sex classification
    if not isinstance(sample_tables, Sequence):
        cohort_5k = sample_tables
    else:
        if len(sample_tables) == 0:
            raise ValueError("sample_tables must contain at least one sample table")
        cohort_5k = validate_cohort(sample_tables, target_bin_size=bin_size)

    n_samples = cohort_5k.counts.shape[1]
    if n_samples == 0:
        raise ValueError("sample_tables must contain at least one sample table")
    if len(entries) != n_samples:
        raise ValueError(
            f"sample_tables ({n_samples}) and entries ({len(entries)}) "
            f"must have the same length"
        )

    c_5k = list(cohort_5k.contigs)
    s_5k = cohort_5k.starts
    e_5k = cohort_5k.ends
    w_5k = e_5k - s_5k
    counts_5k_matrix = cohort_5k.counts

    norm_c_5k = np.array([normalize_chr(c) for c in c_5k], dtype=object)
    chry_mask_5k = norm_c_5k == "Y"
    auto_mask_5k = _resolve_autosome_mask(norm_c_5k)

    n_chry_bins = int(np.sum(chry_mask_5k))
    total_reads = np.sum(counts_5k_matrix, axis=0)
    chry_reads = (
        np.sum(counts_5k_matrix[chry_mask_5k, :], axis=0)
        if n_chry_bins > 0
        else np.zeros(n_samples, dtype=np.float64)
    )
    yfracs = chry_reads / np.maximum(total_reads, 1.0)

    y_norm_all = np.zeros((n_samples, n_chry_bins), dtype=np.float64)
    if n_chry_bins > 0:
        for i in range(n_samples):
            r_auto = _autosomal_density(
                counts_5k_matrix[:, i], w_5k, auto_mask_5k, bin_size
            )
            y_norm_all[i, :] = counts_5k_matrix[chry_mask_5k, i] / (
                r_auto * w_5k[chry_mask_5k]
            )

    is_male_gmm, yfrac_threshold = infer_sex_gmm_1d(yfracs, seed=seed)
    is_male_resolved: list[bool] = []
    for i, entry in enumerate(entries):
        if entry.sex == "M":
            is_male_resolved.append(True)
        elif entry.sex == "F":
            is_male_resolved.append(False)
        else:
            is_male_resolved.append(bool(is_male_gmm[i]))

    is_male_arr = np.array(is_male_resolved, dtype=np.bool_)

    # 2. Informative chrY mask and FFY fitting
    if n_chry_bins > 0:
        chry_starts = s_5k[chry_mask_5k]
        chry_ends = e_5k[chry_mask_5k]
        coord_mask = (chry_starts >= _PAR1_START) & (chry_ends <= _YQ12_END)
        if not np.any(coord_mask):
            coord_mask = np.ones(n_chry_bins, dtype=np.bool_)

        has_males = bool(np.any(is_male_arr))
        has_females = bool(np.any(~is_male_arr))

        if has_males and has_females:
            mean_male = np.mean(y_norm_all[is_male_arr, :], axis=0)
            med_female = np.median(y_norm_all[~is_male_arr, :], axis=0)
            info_mask = (
                coord_mask & (mean_male > med_female + 1e-4) & (med_female < 0.15)
            )
            if not np.any(info_mask):
                info_mask = coord_mask
        else:
            info_mask = coord_mask

        y_scores = np.mean(y_norm_all[:, info_mask], axis=1)
        if has_females:
            female_baseline = float(np.median(y_scores[~is_male_arr]))
        else:
            female_baseline = 0.0
        male_100_baseline = female_baseline + 0.5

        denom = max(male_100_baseline - female_baseline, 1e-6)
        prior_ffy = np.clip((y_scores - female_baseline) / denom, 0.0, 1.0)
        prior_ffy[~is_male_arr] = 0.0

        target_ffy = prior_ffy.copy()
        for i, entry in enumerate(entries):
            if entry.ff is not None:
                target_ffy[i] = entry.ff

        X_info = y_norm_all[:, info_mask]
        w_info = fit_lasso_fista(X_info, target_ffy, alpha=1e-4)
        ffy_weights = np.zeros(n_chry_bins, dtype=np.float64)
        ffy_weights[info_mask] = w_info

        raw_ffy_preds = X_info @ w_info
        if has_males:
            ffy_intercept, ffy_slope = ols_fit(
                raw_ffy_preds[is_male_arr], target_ffy[is_male_arr]
            )
        else:
            ffy_intercept, ffy_slope = ols_fit(raw_ffy_preds, target_ffy)

        final_ffy = ffy_intercept + ffy_slope * raw_ffy_preds
        informative_mask = info_mask
    else:
        informative_mask = np.zeros(0, dtype=np.bool_)
        ffy_weights = np.zeros(0, dtype=np.float64)
        ffy_intercept, ffy_slope = 0.0, 1.0
        final_ffy = np.zeros(n_samples, dtype=np.float64)

    # 3. Rebin to prenatal_bin_size (100 kb) for PREFACE and FFX
    cohort_100k = rebin_cohort(cohort_5k, prenatal_bin_size)
    c_100k = list(cohort_100k.contigs)
    s_100k = cohort_100k.starts
    e_100k = cohort_100k.ends
    counts_100k_matrix = cohort_100k.counts

    w_100k = e_100k - s_100k
    norm_c_100k = np.array([normalize_chr(c) for c in c_100k], dtype=object)
    auto_mask_100k = _resolve_autosome_mask(norm_c_100k)
    chrx_mask_100k = norm_c_100k == "X"

    n_auto_100k = int(np.sum(auto_mask_100k))
    log2_ratios_100k = np.zeros((n_samples, n_auto_100k), dtype=np.float64)
    chrx_ratios = np.ones(n_samples, dtype=np.float64)

    for i in range(n_samples):
        counts_100k = counts_100k_matrix[:, i]
        r_auto_100k = _autosomal_density(
            counts_100k, w_100k, auto_mask_100k, prenatal_bin_size
        )
        exp_auto = r_auto_100k * w_100k[auto_mask_100k]
        ratio_auto = counts_100k[auto_mask_100k] / exp_auto
        log2_ratios_100k[i, :] = np.log2(np.where(ratio_auto > 0, ratio_auto, np.nan))

        if np.any(chrx_mask_100k):
            exp_chrx = r_auto_100k * w_100k[chrx_mask_100k]
            ratio_chrx = counts_100k[chrx_mask_100k] / exp_chrx
            log2_chrx = np.log2(np.where(ratio_chrx > 0, ratio_chrx, np.nan))
            if np.any(np.isfinite(log2_chrx)):
                chrx_ratios[i] = float(2.0 ** np.nanmean(log2_chrx))

    # Mask low-coverage/variable 100 kb autosomal bins
    valid_bins = np.isfinite(log2_ratios_100k)
    frac_valid = np.mean(valid_bins, axis=0)
    auto_masked = frac_valid < 0.99
    if np.all(auto_masked):
        auto_masked[:] = False

    raw_ratios = np.where(valid_bins, log2_ratios_100k, np.nan)
    with np.errstate(all="ignore"):
        mean_features = np.nanmean(raw_ratios, axis=0)
    mean_features = np.nan_to_num(mean_features, nan=0.0)

    nan_r, nan_c = np.where(np.isnan(raw_ratios))
    X_autosomal = raw_ratios.copy()
    X_autosomal[nan_r, nan_c] = np.take(mean_features, nan_c)
    X_autosomal = np.nan_to_num(X_autosomal, nan=0.0)

    # Truncated PCA
    n_feat = min(50, max(1, n_samples - 1))
    if float(np.var(X_autosomal)) < 1e-12:
        pca_center = np.mean(X_autosomal, axis=0)
        scores = np.zeros((n_samples, n_feat), dtype=np.float64)
        pca_rotation = np.zeros((n_auto_100k, n_feat), dtype=np.float64)
    else:
        scores, pca_center, pca_rotation = pca_irlba_numpy(
            X_autosomal, n_components=n_feat, seed=seed
        )

    # 4. PREFACE Training targets (entry.ff if given, else FFY on male fetuses)
    target_preface = np.full(n_samples, np.nan, dtype=np.float64)
    for i, entry in enumerate(entries):
        if entry.ff is not None:
            target_preface[i] = entry.ff
        elif is_male_arr[i]:
            target_preface[i] = max(0.0, float(final_ffy[i]))

    labeled_mask = np.isfinite(target_preface)
    if not np.any(labeled_mask):
        labeled_mask[:] = True
        target_preface[:] = 0.05

    labeled_idx = np.flatnonzero(labeled_mask)
    X_labeled = scores[labeled_idx, :]
    y_labeled = target_preface[labeled_idx]

    # K-fold cross validation for PREFACE skew correction
    k_folds = min(10, labeled_idx.size)
    if k_folds >= 2:
        cv_mlp = np.zeros(labeled_idx.size, dtype=np.float64)
        cv_olm = np.zeros(labeled_idx.size, dtype=np.float64)
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

        for fold_idx, (train_part, test_part) in enumerate(kf.split(X_labeled)):
            X_tr, y_tr = X_labeled[train_part], y_labeled[train_part]
            X_te = X_labeled[test_part]

            w1_k, b1_k, w2_k, b2_k = fit_rprop_mlp(
                X_tr,
                y_tr,
                hidden_dim=2,
                n_inits=2,
                max_epochs=400,
                seed=seed + fold_idx,
            )
            h_te = scipy.special.expit(X_te @ w1_k + b1_k)
            cv_mlp[test_part] = scipy.special.expit(h_te @ w2_k + b2_k).ravel()

            olm_w_k, olm_b_k = train_olm(X_tr, y_tr)
            cv_olm[test_part] = (X_te @ olm_w_k + olm_b_k).ravel()

        mlp_skew_int, mlp_skew_slope = ols_fit(cv_mlp, y_labeled)
        olm_skew_int, olm_skew_slope = ols_fit(cv_olm, y_labeled)
    else:
        mlp_skew_int, mlp_skew_slope = 0.0, 1.0
        olm_skew_int, olm_skew_slope = 0.0, 1.0

    # Train final full-cohort MLP and OLM
    mlp_w1, mlp_b1, mlp_w2, mlp_b2 = fit_rprop_mlp(
        X_labeled, y_labeled, hidden_dim=2, n_inits=5, seed=seed
    )
    olm_w, olm_bias = train_olm(X_labeled, y_labeled)

    # 5. chrX Diploid Female Baseline and FFX Huber Regression
    has_females = bool(np.any(~is_male_arr))
    if has_females:
        fem_ratios = chrx_ratios[~is_male_arr]
        fem_chrx_mean = float(np.mean(fem_ratios))
        fem_chrx_std = max(float(np.std(fem_ratios)), 0.005)
    else:
        fem_chrx_mean = 1.0
        fem_chrx_std = 0.01

    male_mask = is_male_arr
    if np.sum(male_mask) >= 2:
        ffx_int, ffx_slope = rlm_huber(chrx_ratios[male_mask], final_ffy[male_mask])
    else:
        ffx_int, ffx_slope = 2.0, -2.0

    # 6. Build PrenatalModelMetadata
    prenatal_meta = PrenatalModelMetadata(
        prenatal_bin_size=prenatal_bin_size,
        yfrac_threshold=float(yfrac_threshold),
        ffy_intercept=float(ffy_intercept),
        ffy_slope=float(ffy_slope),
        mlp_w1=[[float(x) for x in row] for row in mlp_w1],
        mlp_b1=[float(x) for x in mlp_b1],
        mlp_w2=[[float(x) for x in row] for row in mlp_w2],
        mlp_b2=[float(x) for x in mlp_b2],
        mlp_skew_intercept=float(mlp_skew_int),
        mlp_skew_slope=float(mlp_skew_slope),
        olm_weights=[float(x) for x in olm_w],
        olm_bias=float(olm_bias),
        olm_skew_intercept=float(olm_skew_int),
        olm_skew_slope=float(olm_skew_slope),
        ffx_intercept=float(ffx_int),
        ffx_slope=float(ffx_slope),
        female_chrx_ratio_mean=float(fem_chrx_mean),
        female_chrx_ratio_std=float(fem_chrx_std),
    )

    # 7. Construct Arrow Reference Tables
    auto_indices = np.flatnonzero(auto_mask_100k)
    prenatal_auto_table = _build_prenatal_branch_table(
        branch=BRANCH_PRENATAL_AUTOSOMAL,
        contigs=[c_100k[i] for i in auto_indices],
        starts=s_100k[auto_indices],
        ends=e_100k[auto_indices],
        masked=auto_masked,
        pca_mean=pca_center,
        pca_loadings=[[float(v) for v in row] for row in pca_rotation],
        null_mean=mean_features,
    )

    chry_indices = np.flatnonzero(chry_mask_5k)
    prenatal_ffy_table = _build_prenatal_branch_table(
        branch=BRANCH_PRENATAL_FFY,
        contigs=[c_5k[i] for i in chry_indices],
        starts=s_5k[chry_indices],
        ends=e_5k[chry_indices],
        masked=~informative_mask,
        pca_mean=ffy_weights,
    )

    return (
        prenatal_auto_table,
        prenatal_ffy_table,
        prenatal_meta,
        tuple(is_male_resolved),
    )


def predict_prenatal(
    sample_table: pa.Table,
    reference_table: pa.Table,
    prenatal_meta: PrenatalModelMetadata,
    min_ff_warning: float = 0.04,
) -> FetalFractionResult:
    """Predict fetal sex, FFY, PREFACE, FFX, SCA status, and active fetal fraction."""
    # 1. Extract reference branches
    ref_branches = reference_table.column("branch").to_pylist()
    if BRANCH_PRENATAL_FFY not in ref_branches:
        raise ValueError(f"reference table is missing '{BRANCH_PRENATAL_FFY}' branch")
    if BRANCH_PRENATAL_AUTOSOMAL not in ref_branches:
        raise ValueError(
            f"reference table is missing '{BRANCH_PRENATAL_AUTOSOMAL}' branch"
        )

    ffy_ref = reference_table.filter(
        pc.equal(reference_table.column("branch"), BRANCH_PRENATAL_FFY)
    )
    auto_ref = reference_table.filter(
        pc.equal(reference_table.column("branch"), BRANCH_PRENATAL_AUTOSOMAL)
    )

    # 2. Evaluate chry_score and fetal sex from 5 kb sample table
    sample_c = [str(x) for x in sample_table.column("contig").to_pylist()]
    sample_s = np.asarray(
        sample_table.column("start").to_numpy(zero_copy_only=False), dtype=np.int64
    )
    sample_e = np.asarray(
        sample_table.column("end").to_numpy(zero_copy_only=False), dtype=np.int64
    )
    sample_w = sample_e - sample_s
    sample_counts = np.asarray(
        sample_table.column("count").to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )
    norm_sample_c = np.array([normalize_chr(c) for c in sample_c], dtype=object)

    chry_mask = norm_sample_c == "Y"
    total_reads = float(np.sum(sample_counts))
    chry_reads = float(np.sum(sample_counts[chry_mask])) if np.any(chry_mask) else 0.0
    chry_score = chry_reads / max(total_reads, 1.0)
    fetal_sex: FetalSex = "M" if chry_score >= prenatal_meta.yfrac_threshold else "F"

    # 3. Compute FFY if male
    if fetal_sex == "M" and np.any(chry_mask) and ffy_ref.num_rows > 0:
        auto_mask_5k = _resolve_autosome_mask(norm_sample_c)
        r_auto_5k = _autosomal_density(sample_counts, sample_w, auto_mask_5k, 5000)
        y_norm = sample_counts[chry_mask] / (r_auto_5k * sample_w[chry_mask])

        ref_ffy_weights = np.asarray(
            ffy_ref.column("pca_mean").to_numpy(zero_copy_only=False),
            dtype=np.float64,
        )
        ref_ffy_masked = np.asarray(
            ffy_ref.column("masked").to_numpy(zero_copy_only=False), dtype=np.bool_
        )
        weights = np.where(~ref_ffy_masked, ref_ffy_weights, 0.0)

        # Coordinate matching
        ref_ffy_s = np.asarray(
            ffy_ref.column("start").to_numpy(zero_copy_only=False),
            dtype=np.int64,
        )
        matched_coords = y_norm.size == weights.size and np.array_equal(
            sample_s[chry_mask], ref_ffy_s
        )
        if matched_coords:
            raw_ffy = float(np.sum(y_norm * weights))
        else:
            aligned = _align_sample_to_ref(sample_c, sample_s, sample_e, ffy_ref)
            raw_ffy = 0.0
            for j, s_idx in enumerate(aligned):
                if s_idx >= 0 and weights[j] != 0.0:
                    s_w = sample_w[s_idx]
                    raw_ffy += float(
                        (sample_counts[s_idx] / (r_auto_5k * s_w)) * weights[j]
                    )

        ffy_computed = float(
            prenatal_meta.ffy_intercept + prenatal_meta.ffy_slope * raw_ffy
        )
        ffy: float | None = max(0.0, ffy_computed)
    else:
        ffy = None

    # 4. Rebin sample to 100 kb for PREFACE and FFX
    sample_100k = rebin_sample_table(sample_table, prenatal_meta.prenatal_bin_size)
    c_100k = [str(x) for x in sample_100k.column("contig").to_pylist()]
    s_100k = np.asarray(
        sample_100k.column("start").to_numpy(zero_copy_only=False), dtype=np.int64
    )
    e_100k = np.asarray(
        sample_100k.column("end").to_numpy(zero_copy_only=False), dtype=np.int64
    )
    w_100k = e_100k - s_100k
    counts_100k = np.asarray(
        sample_100k.column("count").to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )
    norm_c_100k = np.array([normalize_chr(c) for c in c_100k], dtype=object)
    auto_mask_100k = _resolve_autosome_mask(norm_c_100k)
    chrx_mask_100k = norm_c_100k == "X"

    r_auto_100k = _autosomal_density(
        counts_100k, w_100k, auto_mask_100k, prenatal_meta.prenatal_bin_size
    )

    # Autosomal log2 ratios on auto_ref bins
    exp_counts = r_auto_100k * w_100k[auto_mask_100k]
    ratio_auto = counts_100k[auto_mask_100k] / exp_counts
    log2_ratios = np.log2(np.where(ratio_auto > 0, ratio_auto, np.nan))

    ref_null_mean = np.asarray(
        auto_ref.column("null_mean").to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )
    ref_pca_center = np.asarray(
        auto_ref.column("pca_mean").to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )
    ref_pca_rotation = np.array(
        auto_ref.column("pca_loadings").to_pylist(), dtype=np.float64
    )

    ref_auto_s = np.asarray(
        auto_ref.column("start").to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    auto_s_100k = s_100k[auto_mask_100k]
    matched_auto = log2_ratios.size == ref_null_mean.size and np.array_equal(
        auto_s_100k, ref_auto_s
    )
    if not matched_auto:
        aligned = _align_sample_to_ref(c_100k, s_100k, e_100k, auto_ref)
        aligned_ratios = np.full(ref_null_mean.size, np.nan, dtype=np.float64)
        for j, s_idx in enumerate(aligned):
            if s_idx >= 0:
                exp_count = r_auto_100k * w_100k[s_idx]
                if exp_count > 0 and counts_100k[s_idx] > 0:
                    aligned_ratios[j] = math.log2(counts_100k[s_idx] / exp_count)
        log2_ratios = aligned_ratios

    imputed = np.where(np.isnan(log2_ratios), ref_null_mean, log2_ratios)
    centered = imputed - ref_pca_center
    features = centered @ ref_pca_rotation

    # Evaluate MLP or fallback to OLM
    w1 = np.asarray(prenatal_meta.mlp_w1, dtype=np.float64)
    b1 = np.asarray(prenatal_meta.mlp_b1, dtype=np.float64)
    w2 = np.asarray(prenatal_meta.mlp_w2, dtype=np.float64)
    b2 = np.asarray(prenatal_meta.mlp_b2, dtype=np.float64)

    h = scipy.special.expit(features @ w1 + b1)
    raw_mlp = float(scipy.special.expit(h @ w2 + b2)[0])

    if 0.01 <= raw_mlp <= 0.50:
        preface = float(
            prenatal_meta.mlp_skew_intercept + prenatal_meta.mlp_skew_slope * raw_mlp
        )
    else:
        olm_w = np.asarray(prenatal_meta.olm_weights, dtype=np.float64)
        olm_bias = float(prenatal_meta.olm_bias)
        raw_olm = float(features @ olm_w + olm_bias)
        preface = float(
            prenatal_meta.olm_skew_intercept + prenatal_meta.olm_skew_slope * raw_olm
        )

    # 5. Evaluate chrX ratio and FFX
    if np.any(chrx_mask_100k):
        exp_chrx = r_auto_100k * w_100k[chrx_mask_100k]
        ratio_chrx = counts_100k[chrx_mask_100k] / exp_chrx
        log2_chrx = np.log2(np.where(ratio_chrx > 0, ratio_chrx, np.nan))
        chrx_ratio = (
            float(2.0 ** np.nanmean(log2_chrx))
            if np.any(np.isfinite(log2_chrx))
            else 1.0
        )
    else:
        chrx_ratio = 1.0

    ffx = float(prenatal_meta.ffx_intercept + prenatal_meta.ffx_slope * chrx_ratio)

    # 6. Classify SCA
    sca_result = classify_sca(
        fetal_sex=fetal_sex,
        ffy=ffy,
        ffx=ffx,
        preface=preface,
        chrx_ratio=chrx_ratio,
        prenatal_meta=prenatal_meta,
    )

    # 7. Active Fetal Fraction selection
    if fetal_sex == "M" and sca_result.call in ("XY", "XXY"):
        active_ff = max(0.0, float(ffy)) if ffy is not None else max(0.0, preface)
        active_ff_source: ActiveFFSource = "FFY"
    else:
        active_ff = max(0.0, float(preface))
        active_ff_source = "PREFACE"

    low_ff_warning = bool(active_ff < min_ff_warning)
    warnings: list[str] = []
    if low_ff_warning:
        warnings.append(
            f"fetal fraction below warning threshold ({min_ff_warning:.2f})"
        )
    if sca_result.is_aneuploid:
        warnings.append(f"sex chromosome aneuploidy detected: {sca_result.call}")

    return FetalFractionResult(
        fetal_sex=fetal_sex,
        active_ff=active_ff,
        active_ff_source=active_ff_source,
        ffy=ffy,
        ffx=ffx,
        preface=preface,
        chrx_ratio=chrx_ratio,
        chry_score=chry_score,
        yfrac_threshold=prenatal_meta.yfrac_threshold,
        sca=sca_result,
        low_ff_warning=low_ff_warning,
        warnings=tuple(warnings),
    )


def annotate_prenatal_segments(
    segments: Sequence[SegmentResult],
    prenatal_result: FetalFractionResult,
    z_threshold: float = 3.0,
) -> list[SegmentResult]:
    """Annotate segments with expected ratio, implied FF, and discordance status."""
    active_ff = max(0.0, float(prenatal_result.active_ff))
    annotated: list[SegmentResult] = []

    for seg in segments:
        if active_ff <= 0.0:
            expected_ratio = 0.0
        elif seg.ratio >= 0.0:
            expected_ratio = math.log2(1.0 + 0.5 * active_ff)
        else:
            arg = max(1e-6, 1.0 - 0.5 * active_ff)
            expected_ratio = math.log2(arg)

        implied_ff = 2.0 * abs(2.0**seg.ratio - 1.0)

        norm_contig = normalize_chr(seg.contig)
        is_gonosome = norm_contig in {"X", "Y"}
        has_call = (seg.call is not None) or (abs(seg.zscore) >= z_threshold)

        discordant = False
        reason: str | None = None

        if is_gonosome and (prenatal_result.sca.is_aneuploid or has_call):
            discordant = True
            reason = "SCA_DISCORDANT"
        elif has_call and abs(seg.ratio) < 0.5 * abs(expected_ratio):
            discordant = True
            reason = "SUB_FETAL_RATIO"
        elif has_call and implied_ff > max(2.5 * active_ff, active_ff + 0.15):
            discordant = True
            reason = "SUPER_FETAL_RATIO"

        annotated.append(
            replace(
                seg,
                expected_ratio=float(expected_ratio),
                implied_ff=float(implied_ff),
                discordant=discordant,
                discordance_reason=reason,
            )
        )

    return annotated
