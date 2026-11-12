"""Prenatal model training, inference, SCA calling, and segment annotation."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Final

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import scipy.special
import scipy.stats
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor

from ccnv.features.cohort import (
    Cohort,
    rebin_cohort,
    rebin_sample_table,
    validate_cohort,
)
from ccnv.features.fragmentomics import (
    extract_sample_fragmentomics,
    fit_cohort_fragmentomics_model,
)
from ccnv.features.prenatal_origin import annotate_prenatal_segments
from ccnv.features.rebinning import rebin_sample_buffers
from ccnv.io.parquet import extract_sample_meta_and_counts
from ccnv.models import (
    BRANCH_PRENATAL_AUTOSOMAL,
    BRANCH_PRENATAL_FFY,
    NUM_GC_QUANTILES,
    NUM_PROTOTYPES_PER_BIN,
    ActiveFFSource,
    FetalFractionResult,
    FetalSex,
    PrenatalModelMetadata,
    SampleBuffers,
    SamplesheetEntry,
    SCAResult,
    SchemaError,
    build_reference_schema,
    decode_metadata,
    fixed_size_list_to_numpy,
    is_autosome_name,
    normalize_chr,
    reconstruct_sample_grid,
)
from ccnv.stats.fragmentomics import (
    compute_band_fisher_weights,
    predict_ff_fragmentomics,
)
from ccnv.stats.pca import pca_irlba_numpy
from ccnv.stats.prenatal_math import (
    PoissonFFYModel,
    classify_sca_ray_4d,
    compute_autosomal_density,
    evaluate_poisson_msy_ffy,
    fit_ffa_logit_ensemble,
    fit_forward_ffx_model,
    fit_msy_sex_mixture,
    fit_poisson_msy_ffy,
)

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
    pca_loadings: list[list[float]] | NDArray[np.floating] | None = None,
    null_mean: NDArray[np.float64] | None = None,
) -> pa.Table:
    """Build a Schema v2.0.0 prenatal branch table."""
    n_bins = len(contigs)
    if pca_loadings is not None:
        ld_arr = np.asarray(pca_loadings, dtype=np.float32)
        if ld_arr.ndim == 1:
            ld_arr = ld_arr[:, np.newaxis]
        pca_rank = int(ld_arr.shape[1]) if ld_arr.size > 0 else 0
    else:
        ld_arr = np.empty((n_bins, 0), dtype=np.float32)
        pca_rank = 0

    schema = build_reference_schema(pca_rank)
    uniq_c = list(dict.fromkeys(contigs))
    c_map = {name: idx for idx, name in enumerate(uniq_c)}
    c_codes = np.array([c_map[c] for c in contigs], dtype=np.int16)

    branch_col = pa.DictionaryArray.from_arrays(
        pa.array(np.zeros(n_bins, dtype=np.int8), type=pa.int8()),
        pa.array([branch], type=pa.string()),
    )
    contig_col = pa.DictionaryArray.from_arrays(
        pa.array(c_codes, type=pa.int16()),
        pa.array(uniq_c, type=pa.string())
        if uniq_c
        else pa.array([], type=pa.string()),
    )
    loadings_col: pa.Array = (
        pa.nulls(n_bins, type=pa.list_(pa.float32(), 0))
        if pca_rank == 0
        else pa.FixedSizeListArray.from_arrays(
            pa.array(ld_arr.ravel(), type=pa.float32()), pca_rank
        )
    )
    proto_ids_col = pa.FixedSizeListArray.from_arrays(
        pa.array(np.zeros(n_bins * NUM_PROTOTYPES_PER_BIN, dtype=np.uint16)),
        NUM_PROTOTYPES_PER_BIN,
    )
    proto_w_col = pa.FixedSizeListArray.from_arrays(
        pa.array(np.zeros(n_bins * NUM_PROTOTYPES_PER_BIN, dtype=np.float16)),
        NUM_PROTOTYPES_PER_BIN,
    )

    nm = (
        np.asarray(null_mean, dtype=np.float32)
        if null_mean is not None
        else np.zeros(n_bins, dtype=np.float32)
    )
    return pa.table(
        {
            "branch": branch_col,
            "contig": contig_col,
            "start": np.asarray(starts, dtype=np.uint32),
            "end": np.asarray(ends, dtype=np.uint32),
            "masked": np.asarray(masked, dtype=np.bool_),
            "gc": np.zeros(n_bins, dtype=np.float32),
            "mappability": np.zeros(n_bins, dtype=np.float32),
            "gc_quantiles": pa.nulls(
                n_bins, type=pa.list_(pa.float32(), NUM_GC_QUANTILES)
            ),
            "ref_frag_len": np.full(n_bins, -1.0, dtype=np.float32),
            "ref_ep_rate": np.full(n_bins, -1.0, dtype=np.float32),
            "pca_mean": np.asarray(pca_mean, dtype=np.float32),
            "pca_loadings": loadings_col,
            "proto_ids": proto_ids_col,
            "proto_weights": proto_w_col,
            "mean_distance": np.zeros(n_bins, dtype=np.float32),
            "null_mean": nm,
            "bio_dispersion": np.zeros(n_bins, dtype=np.float32),
        },
        schema=schema,
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

    sample_lut = {
        (normalize_chr(str(c)), int(s), int(e)): idx
        for idx, (c, s, e) in enumerate(
            zip(sample_contigs, sample_starts, sample_ends, strict=True)
        )
    }
    ref_c = [normalize_chr(str(c)) for c in ref_table.column("contig").to_pylist()]
    ref_s = ref_table.column("start").to_numpy(zero_copy_only=False)
    ref_e = ref_table.column("end").to_numpy(zero_copy_only=False)
    return np.array(
        [
            sample_lut.get((ref_c[i], int(ref_s[i]), int(ref_e[i])), -1)
            for i in range(ref_n)
        ],
        dtype=np.int32,
    )


def _autosomal_density(
    counts: np.ndarray,
    widths: np.ndarray,
    auto_mask: np.ndarray,
    fallback_bin_size: int,
    contigs: np.ndarray | None = None,
    mappability: np.ndarray | None = None,
    gc_trend: np.ndarray | None = None,
) -> float:
    """Calculate ratio-of-sums autosomal read density across chromosomes."""
    c_arr = np.asarray(counts, dtype=np.float64)
    w_arr = np.asarray(widths, dtype=np.float32)
    m_arr = (
        np.asarray(mappability, dtype=np.float32)
        if mappability is not None
        else np.ones_like(w_arr)
    )
    g_arr = (
        np.asarray(gc_trend, dtype=np.float32)
        if gc_trend is not None
        else np.ones_like(w_arr)
    )
    chr_arr = (
        np.asarray(contigs, dtype=np.int16)
        if contigs is not None
        else np.zeros(len(c_arr), dtype=np.int16)
    )
    exposure = w_arr / max(float(fallback_bin_size), 1.0)
    return compute_autosomal_density(c_arr, chr_arr, exposure, m_arr, g_arr, auto_mask)


_FEMALE_SCA_PRIORS: Final[dict[str, float]] = {
    "XX": 0.90,
    "XO": 0.05,
    "XXX": 0.05,
    "XY": 1e-6,
    "XXY": 1e-6,
    "XYY": 1e-6,
}
_MALE_SCA_PRIORS: Final[dict[str, float]] = {
    "XY": 0.90,
    "XXY": 0.05,
    "XYY": 0.05,
    "XX": 1e-6,
    "XO": 1e-6,
    "XXX": 1e-6,
}


def _evaluate_sca_and_active_ff(
    fetal_sex: FetalSex,
    ffy: float | None,
    ffx: float,
    ffa: float,
    chrx_ratio: float,
    prenatal_meta: PrenatalModelMetadata,
    *,
    ff_size: float | None = None,
    chrx_llr_fetal: float | None = None,
) -> tuple[SCAResult, float, float, ActiveFFSource]:
    """Evaluate 4D ray SCA classification and Student-t4 fused active FF."""
    mu_f = float(prenatal_meta.female_chrx_ratio_mean)
    std_f = max(float(prenatal_meta.female_chrx_ratio_std), 1e-4)
    chrx_zscore = float((chrx_ratio - mu_f) / std_f)

    rx = float(chrx_ratio)
    alpha_0 = (
        float(prenatal_meta.alpha_0) if prenatal_meta.alpha_0 is not None else mu_f
    )
    delta_x = float(rx / max(alpha_0, 1e-6) - 1.0)
    f_y = float(ffy) if (fetal_sex == "M" and ffy is not None) else 0.0
    ff_delta = float(ffx - f_y) if fetal_sex == "M" else None

    call, posteriors, active_ff, active_ff_se, active_ff_source = classify_sca_ray_4d(
        delta_x=delta_x,
        sigma_x=max(float(prenatal_meta.female_chrx_ratio_std), 0.005),
        ffy=f_y,
        sigma_ffy=max(float(prenatal_meta.ffy_std), 0.005),
        ffa=ffa,
        sigma_ffa=max(float(prenatal_meta.ffa_std), 0.005),
        ff_size=ff_size,
        sigma_size=max(float(prenatal_meta.ff_size_std), 0.005),
        chrx_llr_fetal=chrx_llr_fetal,
        clinical_priors=_FEMALE_SCA_PRIORS if fetal_sex == "F" else _MALE_SCA_PRIORS,
    )
    sca = SCAResult(
        call=call,
        is_aneuploid=call not in ("XX", "XY"),
        confidence=float(np.clip(posteriors.get(call, 0.5), 0.5, 1.0)),
        chrx_zscore=chrx_zscore,
        ff_delta=ff_delta,
    )
    return sca, active_ff, active_ff_se, active_ff_source


def classify_sca(
    fetal_sex: FetalSex,
    ffy: float | None,
    ffx: float,
    ffa: float = 0.0,
    chrx_ratio: float = 1.0,
    prenatal_meta: PrenatalModelMetadata | None = None,
    *,
    preface: float | None = None,
    ff_size: float | None = None,
    chrx_llr_fetal: float | None = None,
) -> SCAResult:
    """Classify Sex Chromosome Aneuploidy (SCA) via 4D erfc ray marginalization."""
    if preface is not None:
        ffa = preface
    if prenatal_meta is None:
        raise ValueError("prenatal_meta is required for classify_sca")
    sca, _, _, _ = _evaluate_sca_and_active_ff(
        fetal_sex=fetal_sex,
        ffy=ffy,
        ffx=ffx,
        ffa=ffa,
        chrx_ratio=chrx_ratio,
        prenatal_meta=prenatal_meta,
        ff_size=ff_size,
        chrx_llr_fetal=chrx_llr_fetal,
    )
    return sca


def fit_prenatal_models(
    sample_tables: Sequence[pa.Table] | Cohort,
    entries: Sequence[SamplesheetEntry],
    bin_size: int,
    prenatal_bin_size: int,
    seed: int = 42,
    cohort_gc: NDArray[np.float32] | None = None,
    cohort_mappability: NDArray[np.float32] | None = None,
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
    _, contig_ids_5k = np.unique(norm_c_5k, return_inverse=True)
    contig_ids_i16 = contig_ids_5k.astype(np.int16)
    exp_5k = (w_5k / max(float(bin_size), 1.0)).astype(np.float32)
    map_5k = (
        cohort_mappability
        if cohort_mappability is not None
        else np.ones(len(c_5k), dtype=np.float32)
    )
    gc_5k = cohort_gc if cohort_gc is not None else np.ones(len(c_5k), dtype=np.float32)

    r_auto_5k = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        r_auto_5k[i] = compute_autosomal_density(
            counts=counts_5k_matrix[:, i].astype(np.float64),
            contigs=contig_ids_i16,
            exposure=exp_5k,
            mappability=map_5k,
            gc_trend=gc_5k,
            autosomal_mask=auto_mask_5k,
        )

    # 2. Informative chrY mask and FFY fitting
    if n_chry_bins > 0:
        msy_counts = counts_5k_matrix[chry_mask_5k, :].astype(np.float64)
        msy_exposure = np.zeros((n_chry_bins, n_samples), dtype=np.float64)
        for i in range(n_samples):
            msy_exposure[:, i] = (
                r_auto_5k[i]
                * exp_5k[chry_mask_5k]
                * map_5k[chry_mask_5k]
                * gc_5k[chry_mask_5k]
            )

        msy_sample_c = np.sum(msy_counts, axis=0)
        msy_sample_e = np.sum(msy_exposure, axis=0)
        sex_mixture = fit_msy_sex_mixture(msy_sample_c, msy_sample_e, bic_delta_min=6.0)

        is_male_resolved: list[bool] = []
        for i, entry in enumerate(entries):
            if entry.sex == "M":
                is_male_resolved.append(True)
            elif entry.sex == "F":
                is_male_resolved.append(False)
            else:
                if sex_mixture.bimodal:
                    is_male_resolved.append(bool(sex_mixture.p_male[i] > 0.5))
                else:
                    is_male_resolved.append(False)

        is_male_arr = np.array(is_male_resolved, dtype=np.bool_)

        quarantined = (
            (sex_mixture.p_male > 0.02) & (sex_mixture.p_male < 0.98)
            if sex_mixture.bimodal
            else np.zeros(n_samples, dtype=np.bool_)
        )
        clean_females = (~is_male_arr) & (~quarantined)
        clean_males = is_male_arr & (~quarantined)
        if not np.any(clean_females):
            clean_females = ~is_male_arr
        if not np.any(clean_males):
            clean_males = is_male_arr

        labeled_ff = np.array(
            [entry.ff if entry.ff is not None else np.nan for entry in entries],
            dtype=np.float64,
        )

        poisson_model = fit_poisson_msy_ffy(
            msy_bin_counts=msy_counts,
            msy_bin_exposure=msy_exposure,
            mappability=map_5k[chry_mask_5k],
            female_mask=clean_females,
            male_mask=clean_males,
            labeled_ff=labeled_ff,
        )

        final_ffy = np.zeros(n_samples, dtype=np.float64)
        for i in range(n_samples):
            if is_male_arr[i]:
                f_val, _ = evaluate_poisson_msy_ffy(
                    msy_counts[:, i], msy_exposure[:, i], poisson_model
                )
                final_ffy[i] = f_val

        clean_indices = poisson_model.clean_bin_indices
        clean_set = set(clean_indices.tolist())
        informative_mask = np.array(
            [idx in clean_set for idx in range(n_chry_bins)], dtype=np.bool_
        )
        ffy_weights = np.zeros(n_chry_bins, dtype=np.float64)
        ffy_weights[clean_indices] = poisson_model.lambda_bg
        ffy_intercept = 0.0
        ffy_slope = float(poisson_model.kappa_cal)

        total_reads_5k = np.sum(counts_5k_matrix, axis=0).astype(np.float64)
        chry_reads_5k = np.sum(counts_5k_matrix[chry_mask_5k, :], axis=0).astype(
            np.float64
        )
        yfracs = chry_reads_5k / np.maximum(total_reads_5k, 1.0)
        if sex_mixture.bimodal:
            y_m = yfracs[is_male_arr]
            y_f = yfracs[~is_male_arr]
            if y_m.size > 0 and y_f.size > 0:
                yfrac_threshold = float(0.5 * (np.max(y_f) + np.min(y_m)))
            else:
                yfrac_threshold = float(np.median(yfracs))
        else:
            yfrac_threshold = float(np.max(yfracs) + 0.005)
    else:
        informative_mask = np.zeros(0, dtype=np.bool_)
        ffy_weights = np.zeros(0, dtype=np.float64)
        ffy_intercept, ffy_slope = 0.0, 1.0
        final_ffy = np.zeros(n_samples, dtype=np.float64)
        is_male_resolved = [entry.sex == "M" for entry in entries]
        is_male_arr = np.array(is_male_resolved, dtype=np.bool_)
        yfrac_threshold = 0.005
        quarantined = np.zeros(n_samples, dtype=np.bool_)
        poisson_model = PoissonFFYModel(
            clean_bin_indices=np.empty(0, dtype=np.int64),
            lambda_bg=np.empty(0, dtype=np.float64),
            kappa_cal=1.0,
            pooled_bg_rate=0.0,
        )

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

    # 5. Forward-calibrated chrX and FFX modeling
    f_targets_cohort = np.where(is_male_arr, final_ffy, 0.0)
    valid_rx_mask = (
        (~quarantined) & np.isfinite(chrx_ratios) & np.isfinite(f_targets_cohort)
    )
    alpha_0, alpha_1, sigma_x_sys = fit_forward_ffx_model(
        chrx_ratios, f_targets_cohort, valid_rx_mask
    )
    ffx_int = float(alpha_0)
    ffx_slope = float(alpha_1)
    fem_chrx_mean = float(alpha_0)
    fem_chrx_std = float(sigma_x_sys)

    if isinstance(sample_tables, Sequence):
        frag_metas = [
            extract_sample_fragmentomics(decode_metadata(t.schema.metadata))
            for t in sample_tables
        ]
    else:
        frag_metas = [
            extract_sample_fragmentomics(m) for m in cohort_5k.sample_metadata
        ]

    frag_fit = fit_cohort_fragmentomics_model(
        frag_metas, is_male_arr, final_ffy, labeled_ff
    )
    if frag_fit is not None:
        p_fet, p_mat, w_frag, b_frag, m_frag, s_frag, std_size = frag_fit
    else:
        p_fet = p_mat = w_frag = b_frag = m_frag = s_frag = None
        std_size = 0.018

    if p_fet is not None and p_mat is not None:
        band_w, mu_f_w, mu_m_w = compute_band_fisher_weights(
            np.asarray(p_fet, dtype=np.float64),
            np.asarray(p_mat, dtype=np.float64),
            f0=0.08,
        )
        band_weights_list: list[float] | None = [float(x) for x in band_w]
        mu_f_val: float | None = float(mu_f_w)
        mu_m_val: float | None = float(mu_m_w)
    else:
        band_weights_list = None
        mu_f_val = None
        mu_m_val = None

    # 6. Build PrenatalModelMetadata
    prenatal_meta = PrenatalModelMetadata(
        prenatal_bin_size=prenatal_bin_size,
        yfrac_threshold=float(yfrac_threshold),
        ffy_intercept=float(ffy_intercept),
        ffy_slope=float(ffy_slope),
        olm_weights=[float(x) for x in w_ens],
        olm_bias=float(b_ens),
        ffx_intercept=float(ffx_int),
        ffx_slope=float(ffx_slope),
        female_chrx_ratio_mean=float(fem_chrx_mean),
        female_chrx_ratio_std=float(fem_chrx_std),
        ffy_std=0.01,
        ffa_std=float(sigma_ffa),
        ffx_std=0.025,
        ffa_logit_mode=True,
        p_fetal_density=p_fet,
        p_maternal_density=p_mat,
        ff_frag_weights=w_frag,
        ff_frag_bias=b_frag,
        ff_frag_feature_mean=m_frag,
        ff_frag_feature_scale=s_frag,
        ff_size_std=float(std_size),
        msy_bg_rate=float(poisson_model.pooled_bg_rate),
        clean_msy_bins=[int(x) for x in poisson_model.clean_bin_indices],
        alpha_0=float(alpha_0),
        alpha_1=float(alpha_1),
        sigma_x_sys=float(sigma_x_sys),
        quarantined_samples=(
            [int(x) for x in np.flatnonzero(quarantined)]
            if np.any(quarantined)
            else None
        ),
        band_fisher_weights=band_weights_list,
        mu_fetal_w=mu_f_val,
        mu_maternal_w=mu_m_val,
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

    auto_rank = int(pca_rotation.shape[1]) if pca_rotation.ndim == 2 else 0
    chry_indices = np.flatnonzero(chry_mask_5k)
    prenatal_ffy_table = _build_prenatal_branch_table(
        branch=BRANCH_PRENATAL_FFY,
        contigs=[c_5k[i] for i in chry_indices],
        starts=s_5k[chry_indices],
        ends=e_5k[chry_indices],
        masked=~informative_mask,
        pca_mean=ffy_weights,
        pca_loadings=np.zeros((len(chry_indices), auto_rank), dtype=np.float32),
    )

    return (
        prenatal_auto_table,
        prenatal_ffy_table,
        prenatal_meta,
        tuple(is_male_resolved),
    )


def predict_prenatal(
    sample_table: pa.Table | SampleBuffers,
    reference_table: pa.Table,
    prenatal_meta: PrenatalModelMetadata,
    min_ff_warning: float = 0.04,
) -> FetalFractionResult:
    """Predict fetal sex, FFY, FFA, FFX, SCA status, and active fetal fraction."""
    # 1. Extract reference branches
    ref_branches = {
        str(v)
        for v in pc.unique(reference_table.column("branch")).to_pylist()
        if v is not None
    }
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

    # 2. Evaluate chry_score and fetal sex from sample sketch grid
    s_meta, sample_counts = extract_sample_meta_and_counts(sample_table)
    s_codes, s_names, s_starts_u32, s_ends_u32 = reconstruct_sample_grid(s_meta)
    sample_c = [s_names[int(c)] for c in s_codes]
    sample_s = s_starts_u32.astype(np.int64)
    sample_e = s_ends_u32.astype(np.int64)
    sample_w = sample_e - sample_s
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
        sample_chry_c = sample_counts[chry_mask].astype(np.float64)
        sample_chry_exp = (r_auto_5k * (sample_w[chry_mask] / 5000.0)).astype(
            np.float64
        )

        ref_ffy_weights = np.asarray(
            ffy_ref.column("pca_mean").to_numpy(zero_copy_only=False),
            dtype=np.float64,
        )
        ref_ffy_masked = np.asarray(
            ffy_ref.column("masked").to_numpy(zero_copy_only=False), dtype=np.bool_
        )
        clean_mask = ~ref_ffy_masked

        if prenatal_meta.clean_msy_bins is not None:
            clean_indices = np.asarray(prenatal_meta.clean_msy_bins, dtype=np.int64)
            clean_bg = (
                ref_ffy_weights[clean_indices]
                if int(np.max(clean_indices)) < ref_ffy_weights.size
                else np.zeros(clean_indices.size, dtype=np.float64)
            )
            poisson_model = PoissonFFYModel(
                clean_bin_indices=clean_indices,
                lambda_bg=clean_bg,
                kappa_cal=prenatal_meta.ffy_slope,
                pooled_bg_rate=float(prenatal_meta.msy_bg_rate or 0.0),
            )
            ffy_computed, _ = evaluate_poisson_msy_ffy(
                sample_chry_c, sample_chry_exp, poisson_model
            )
            ffy: float | None = max(0.0, float(ffy_computed))
        else:
            y_norm = sample_chry_c / np.maximum(sample_chry_exp, 1e-9)
            weights = np.where(clean_mask, ref_ffy_weights, 0.0)
            raw_ffy = float(np.sum(y_norm * weights))
            ffy = max(
                0.0,
                float(prenatal_meta.ffy_intercept + prenatal_meta.ffy_slope * raw_ffy),
            )
    else:
        ffy = None

    # 4. Rebin sample to 100 kb for FFA and FFX
    sample_100k: pa.Table | SampleBuffers = (
        rebin_sample_buffers(sample_table, prenatal_meta.prenatal_bin_size)
        if isinstance(sample_table, SampleBuffers)
        else rebin_sample_table(sample_table, prenatal_meta.prenatal_bin_size)
    )
    s100_meta, counts_100k = extract_sample_meta_and_counts(sample_100k)
    c100_codes, c100_names, s100_u32, e100_u32 = reconstruct_sample_grid(s100_meta)
    c_100k = [c100_names[int(c)] for c in c100_codes]
    s_100k = s100_u32.astype(np.int64)
    e_100k = e100_u32.astype(np.int64)
    w_100k = e_100k - s_100k
    norm_c_100k = np.array([normalize_chr(c) for c in c_100k], dtype=object)
    auto_mask_100k = _resolve_autosome_mask(norm_c_100k)
    chrx_mask_100k = norm_c_100k == "X"

    r_auto_100k = _autosomal_density(
        counts_100k, w_100k, auto_mask_100k, prenatal_meta.prenatal_bin_size
    )

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
    ld_col = auto_ref.column("pca_loadings").combine_chunks()
    ld_type = ld_col.type
    if not isinstance(ld_type, pa.FixedSizeListType):
        raise SchemaError(
            f"expected FixedSizeListType for 'pca_loadings', found {ld_type}"
        )
    n_olm = len(prenatal_meta.olm_weights)
    ref_pca_rotation = np.zeros((auto_ref.num_rows, n_olm), dtype=np.float64)
    if ld_type.list_size > 0 and auto_ref.num_rows > 0 and n_olm > 0:
        full_ld = fixed_size_list_to_numpy(ld_col, ld_type.list_size, np.float32)
        k_copy = min(n_olm, ld_type.list_size)
        ref_pca_rotation[:, :k_copy] = np.nan_to_num(
            full_ld[:, :k_copy].astype(np.float64), nan=0.0
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

    ref_masked = np.asarray(
        auto_ref.column("masked").to_numpy(zero_copy_only=False), dtype=np.bool_
    )
    usable_pca = (~ref_masked) & np.isfinite(centered)
    n_pcs = ref_pca_rotation.shape[1] if ref_pca_rotation.ndim == 2 else 0
    if n_pcs > 0 and int(np.count_nonzero(usable_pca)) >= n_pcs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                huber = HuberRegressor(fit_intercept=False, max_iter=200)
                huber.fit(ref_pca_rotation[usable_pca], centered[usable_pca])
                features_flat = huber.coef_.astype(np.float64)
        except ValueError, TypeError, np.linalg.LinAlgError:
            features = centered @ ref_pca_rotation
            features_flat = features.ravel() if features.ndim > 1 else features
    else:
        features = centered @ ref_pca_rotation
        features_flat = features.ravel() if features.ndim > 1 else features

    # Evaluate FFA via convex logit ensemble or linear OLM fallback
    raw_linear = float(
        np.dot(features_flat, np.asarray(prenatal_meta.olm_weights, dtype=np.float64))
        + prenatal_meta.olm_bias
    )
    ffa = (
        float(scipy.special.expit(raw_linear))
        if prenatal_meta.ffa_logit_mode
        else raw_linear
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

    alpha_0 = (
        prenatal_meta.alpha_0
        if prenatal_meta.alpha_0 is not None
        else prenatal_meta.female_chrx_ratio_mean
    )
    alpha_1 = (
        prenatal_meta.alpha_1
        if prenatal_meta.alpha_1 is not None
        else (
            prenatal_meta.ffx_slope
            if prenatal_meta.ffx_slope != 0.0
            else -0.5 * alpha_0
        )
    )
    if alpha_1 != 0.0:
        ffx = float((chrx_ratio - alpha_0) / alpha_1)
    else:
        ffx = float(prenatal_meta.ffx_intercept + prenatal_meta.ffx_slope * chrx_ratio)

    # 6. Fragmentomics FF size prediction
    sample_frag_meta = (
        sample_table.meta.fragmentomics
        if isinstance(sample_table, SampleBuffers)
        else extract_sample_fragmentomics(decode_metadata(sample_table.schema.metadata))
    )
    ff_size = (
        predict_ff_fragmentomics(sample_frag_meta, prenatal_meta)
        if sample_frag_meta is not None
        else None
    )

    # 7. Classify SCA and Fuse Active Fetal Fraction via 4D Ray Marginalization
    sca_result, active_ff, active_ff_se, active_ff_source = _evaluate_sca_and_active_ff(
        fetal_sex=fetal_sex,
        ffy=ffy,
        ffx=ffx,
        ffa=ffa,
        chrx_ratio=chrx_ratio,
        prenatal_meta=prenatal_meta,
        ff_size=ff_size,
    )

    low_ff_warning = bool(active_ff < min_ff_warning)
    result_warnings: list[str] = []
    if low_ff_warning:
        result_warnings.append(
            f"fetal fraction below warning threshold ({min_ff_warning:.2f})"
        )
    if sca_result.is_aneuploid:
        result_warnings.append(f"sex chromosome aneuploidy detected: {sca_result.call}")

    return FetalFractionResult(
        fetal_sex=fetal_sex,
        active_ff=active_ff,
        active_ff_source=active_ff_source,
        ffy=ffy,
        ffx=ffx,
        ffa=ffa,
        ff_size=ff_size,
        chrx_ratio=chrx_ratio,
        chry_score=chry_score,
        yfrac_threshold=prenatal_meta.yfrac_threshold,
        sca=sca_result,
        low_ff_warning=low_ff_warning,
        warnings=tuple(result_warnings),
        active_ff_se=active_ff_se,
    )
