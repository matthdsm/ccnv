"""Prenatal model training, inference, SCA calling, and segment annotation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import Final

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import scipy.special
import scipy.stats
from numpy.typing import NDArray

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
    fit_elasticnet_ffy,
    fit_ffa_logit_ensemble,
    infer_sex_gmm_1d,
    rlm_huber,
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


def _is_ffa_autosome(norm_contig: str) -> bool:
    """Return True if contig is a canonical autosome excluding 13, 18, 21."""
    return (
        is_autosome_name(norm_contig)
        and int(norm_contig) not in _EXCLUDED_PRENATAL_AUTOSOMES
    )


_is_preface_autosome = _is_ffa_autosome


def _resolve_autosome_mask(norm_contigs: Sequence[str]) -> NDArray[np.bool_]:
    """Resolve boolean autosome mask for FFA, falling back to all autosomes."""
    mask = np.array([_is_ffa_autosome(c) for c in norm_contigs], dtype=np.bool_)
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
    ref_n = ref_table.num_rows
    if ref_n == 0:
        return np.empty(0, dtype=np.int32)
    if len(sample_contigs) == 0:
        return np.full(ref_n, -1, dtype=np.int32)

    ref_c = [normalize_chr(str(c)) for c in ref_table.column("contig").to_pylist()]
    ref_s = np.asarray(
        ref_table.column("start").to_numpy(zero_copy_only=False), dtype=np.int64
    )
    ref_e = np.asarray(
        ref_table.column("end").to_numpy(zero_copy_only=False), dtype=np.int64
    )
    sample_norm_c = [normalize_chr(str(c)) for c in sample_contigs]

    ref_df = pl.DataFrame(
        {
            "contig_norm": ref_c,
            "start": ref_s,
            "end": ref_e,
            "ref_idx": np.arange(ref_n, dtype=np.int32),
        }
    )
    sample_df = pl.DataFrame(
        {
            "contig_norm": sample_norm_c,
            "start": np.asarray(sample_starts, dtype=np.int64),
            "end": np.asarray(sample_ends, dtype=np.int64),
            "sample_idx": np.arange(len(sample_contigs), dtype=np.int32),
        }
    )

    joined = ref_df.join(
        sample_df, on=["contig_norm", "start", "end"], how="left"
    ).sort("ref_idx")
    return joined["sample_idx"].fill_null(-1).to_numpy().astype(np.int32)


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
    ffa: float = 0.0,
    chrx_ratio: float = 1.0,
    prenatal_meta: PrenatalModelMetadata | None = None,
    z_cut: float = 3.0,
    *,
    preface: float | None = None,
) -> SCAResult:
    """Classify Sex Chromosome Aneuploidy (SCA) via 2D Stoichiometric Gaussian Bayes."""
    if preface is not None:
        ffa = preface
    if prenatal_meta is None:
        raise ValueError("prenatal_meta is required for classify_sca")
    mu_f = float(prenatal_meta.female_chrx_ratio_mean)
    std_f = max(float(prenatal_meta.female_chrx_ratio_std), 1e-4)
    chrx_zscore = float((chrx_ratio - mu_f) / std_f)

    rx = float(chrx_ratio)
    delta_x = (rx - 1.0) if abs(rx - 1.0) > 1e-6 else -0.5 * ffx
    f_y = float(ffy) if (fetal_sex == "M" and ffy is not None) else 0.0
    ff_delta = float(ffx - f_y) if fetal_sex == "M" else None

    # Reference fetal fraction f_0
    cands: list[float] = []
    if fetal_sex == "M" and ffy is not None and ffy > 0.001:
        cands.append(float(ffy))
    if ffa > 0.001:
        cands.append(float(ffa))
    if abs(ffx) > 0.001:
        cands.append(abs(float(ffx)))
    if not cands:
        cands = [max(float(ffa), 0.05)]
    f0 = max(float(np.median(cands)), 0.01)

    means: dict[SCACall, tuple[float, float]] = {
        "XX": (0.0, 0.0),
        "XY": (-0.5 * f0, f0),
        "XO": (-0.5 * f0, 0.0),
        "XXX": (0.5 * f0, 0.0),
        "XXY": (0.0, f0),
        "XYY": (-0.5 * f0, 2.0 * f0),
    }

    sigma_x = max(float(prenatal_meta.female_chrx_ratio_std), 0.005)
    sigma_y = max(float(prenatal_meta.ffy_std), 0.005)

    if fetal_sex == "F":
        priors: dict[SCACall, float] = {
            "XX": 0.90,
            "XO": 0.05,
            "XXX": 0.05,
            "XY": 1e-6,
            "XXY": 1e-6,
            "XYY": 1e-6,
        }
    else:
        priors = {
            "XY": 0.90,
            "XXY": 0.05,
            "XYY": 0.05,
            "XX": 1e-6,
            "XO": 1e-6,
            "XXX": 1e-6,
        }

    karyotypes = list(means.keys())
    d2_list: list[float] = []
    log_joints: list[float] = []
    for k in karyotypes:
        mx, my = means[k]
        d2 = ((delta_x - mx) / sigma_x) ** 2 + ((f_y - my) / sigma_y) ** 2
        d2_list.append(float(d2))
        log_lik = -0.5 * d2
        log_joints.append(float(log_lik + np.log(priors[k])))

    best_idx = int(np.argmax(log_joints))
    best_k = karyotypes[best_idx]
    best_d2 = d2_list[best_idx]

    log_norm = float(scipy.special.logsumexp(log_joints))
    best_post = float(np.exp(log_joints[best_idx] - log_norm))

    is_mixed = best_d2 > 13.8155 or (
        fetal_sex == "F"
        and abs(delta_x) > 0.02
        and abs(ffx) > 0.02
        and (delta_x * ffx > 0)
    )

    if is_mixed:
        call: SCACall = "DISCORDANT_MIXED"
        is_aneuploid = True
        confidence = float(np.clip(scipy.stats.chi2.cdf(best_d2, df=2), 0.5, 1.0))
    else:
        call = best_k
        is_aneuploid = call not in ("XX", "XY")
        confidence = float(np.clip(best_post, 0.5, 1.0))

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
    """Fit FFY, FFA, and FFX models on cohort samples."""
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
        w_info, ffy_intercept = fit_elasticnet_ffy(X_info, target_ffy, seed=seed)
        ffy_weights = np.zeros(n_chry_bins, dtype=np.float64)
        ffy_weights[info_mask] = w_info
        ffy_slope = 1.0

        final_ffy = ffy_intercept + X_info @ w_info
        informative_mask = info_mask
    else:
        informative_mask = np.zeros(0, dtype=np.bool_)
        ffy_weights = np.zeros(0, dtype=np.float64)
        ffy_intercept, ffy_slope = 0.0, 1.0
        final_ffy = np.zeros(n_samples, dtype=np.float64)

    # 3. Rebin to prenatal_bin_size (100 kb) for FFA and FFX
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

    # 4. FFA Training targets (entry.ff if given, else FFY on male fetuses)
    target_ffa = np.full(n_samples, np.nan, dtype=np.float64)
    for i, entry in enumerate(entries):
        if entry.ff is not None:
            target_ffa[i] = entry.ff
        elif is_male_arr[i]:
            target_ffa[i] = max(0.0, float(final_ffy[i]))

    labeled_mask = np.isfinite(target_ffa)
    if not np.any(labeled_mask):
        labeled_mask[:] = True
        target_ffa[:] = 0.05

    labeled_idx = np.flatnonzero(labeled_mask)
    X_labeled = scores[labeled_idx, :]
    y_labeled = target_ffa[labeled_idx]

    # Fit convex logit-link RidgeCV and ElasticNetCV ensemble
    w_ens, b_ens, sigma_ffa = fit_ffa_logit_ensemble(X_labeled, y_labeled, seed=seed)

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
        mlp_w1=[[0.0, 0.0] for _ in range(n_feat)],
        mlp_b1=[0.0, 0.0],
        mlp_w2=[[0.0], [0.0]],
        mlp_b2=[0.0],
        mlp_skew_intercept=0.0,
        mlp_skew_slope=1.0,
        olm_weights=[float(x) for x in w_ens],
        olm_bias=float(b_ens),
        olm_skew_intercept=0.0,
        olm_skew_slope=1.0,
        ffx_intercept=float(ffx_int),
        ffx_slope=float(ffx_slope),
        female_chrx_ratio_mean=float(fem_chrx_mean),
        female_chrx_ratio_std=float(fem_chrx_std),
        ffy_std=0.01,
        ffa_std=float(sigma_ffa),
        ffx_std=0.025,
        ffa_logit_mode=True,
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
    """Predict fetal sex, FFY, FFA, FFX, SCA status, and active fetal fraction."""
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

    # 4. Rebin sample to 100 kb for FFA and FFX
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

    # Evaluate FFA via convex logit ensemble or legacy MLP/OLM fallback
    if prenatal_meta.ffa_logit_mode:
        features_flat = features.ravel() if features.ndim > 1 else features
        raw_logit = float(
            np.dot(
                features_flat, np.asarray(prenatal_meta.olm_weights, dtype=np.float64)
            )
            + prenatal_meta.olm_bias
        )
        ffa = float(scipy.special.expit(raw_logit))
    else:
        # Legacy MLP or fallback to OLM
        w1 = np.asarray(prenatal_meta.mlp_w1, dtype=np.float64)
        b1 = np.asarray(prenatal_meta.mlp_b1, dtype=np.float64)
        w2 = np.asarray(prenatal_meta.mlp_w2, dtype=np.float64)
        b2 = np.asarray(prenatal_meta.mlp_b2, dtype=np.float64)

        h = scipy.special.expit(features @ w1 + b1)
        raw_mlp = float(scipy.special.expit(h @ w2 + b2)[0])

        if 0.01 <= raw_mlp <= 0.50:
            ffa = float(
                prenatal_meta.mlp_skew_intercept
                + prenatal_meta.mlp_skew_slope * raw_mlp
            )
        else:
            olm_w = np.asarray(prenatal_meta.olm_weights, dtype=np.float64)
            olm_bias = float(prenatal_meta.olm_bias)
            raw_olm = float(features @ olm_w + olm_bias)
            ffa = float(
                prenatal_meta.olm_skew_intercept
                + prenatal_meta.olm_skew_slope * raw_olm
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
        ffa=ffa,
        chrx_ratio=chrx_ratio,
        prenatal_meta=prenatal_meta,
    )

    # 7. Active Fetal Fraction selection
    tolerance = 0.05
    if fetal_sex == "M" and sca_result.call in ("XY", "XXY"):
        if ffy is not None:
            if abs(ffy - ffa) <= tolerance:
                # Precision fusion via inverse-variance weighting
                var_ffy = max(float(prenatal_meta.ffy_std), 1e-4) ** 2
                var_ffa = max(float(prenatal_meta.ffa_std), 1e-4) ** 2
                w_ffy = 1.0 / var_ffy
                w_ffa = 1.0 / var_ffa
                sum_w = w_ffy + w_ffa
                sum_prod = w_ffy * ffy + w_ffa * ffa
                if ffx > 0 and abs(ffx - ffy) <= tolerance:
                    var_ffx = max(float(prenatal_meta.ffx_std), 1e-4) ** 2
                    w_ffx = 1.0 / var_ffx
                    sum_w += w_ffx
                    sum_prod += w_ffx * ffx
                active_ff = float(max(0.0, sum_prod / sum_w))
            else:
                active_ff = float(max(0.0, ffy))
            active_ff_source: ActiveFFSource = "FFY"
        else:
            active_ff = float(max(0.0, ffa))
            active_ff_source = "FFA"
    else:
        active_ff = float(max(0.0, ffa))
        active_ff_source = "FFA"

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
        ffa=ffa,
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

        # Standard error sigma_m and fetal-fraction LLR
        if abs(seg.zscore) > 0.1:
            sigma_m = abs(seg.ratio) / max(abs(seg.zscore), 1e-6)
        else:
            sigma_m = 0.20 / math.sqrt(max(seg.bin_count, 1))
        sigma_m = max(sigma_m, 1e-6)

        mu_1 = expected_ratio
        llr_fetal = (seg.ratio**2 - (seg.ratio - mu_1) ** 2) / (2.0 * sigma_m**2)

        norm_contig = normalize_chr(seg.contig)
        is_gonosome = norm_contig in {"X", "Y"}
        has_call = (seg.call is not None) or (abs(seg.zscore) >= z_threshold)

        discordant = False
        reason: str | None = None

        if is_gonosome and (prenatal_result.sca.is_aneuploid or has_call):
            discordant = True
            reason = "SCA_DISCORDANT"
        elif has_call and (abs(seg.ratio) < 0.5 * abs(expected_ratio) or llr_fetal < 0):
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
