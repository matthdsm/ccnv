"""Zero-inclusive per-branch sample normalization, PCA projection, and variance synthesis."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Final

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor

from ccnv.features.cohort import compute_bin_exposure
from ccnv.features.covariates import mask_covariates
from ccnv.features.fragmentomics import (
    extract_bin_fragmentomic_covariates,
    resolve_fragmentomics_compatibility,
)
from ccnv.io.parquet import (
    align_sample_grid_indices,
    extract_reference_branch_arrays,
    extract_sample_meta_and_counts,
)
from ccnv.io.tracks import (
    load_fasta_gc_quantiles,
    load_gc_track,
    load_mappability_track,
)
from ccnv.models import (
    DEFAULT_MIN_MAPPABILITY,
    NUM_PROTOTYPES_PER_BIN,
    BinResult,
    CovariateError,
    PredictConfig,
    PredictError,
    ReferenceMetadata,
    SampleBuffers,
    SchemaError,
)
from ccnv.stats.neighbors import (
    compute_kish_loco_inflation,
    evaluate_loco_prototype_baseline,
    exact_poisson_log2_jensen_bias,
)
from ccnv.stats.regression import (
    build_exogenous_covariate_basis,
    fit_poisson_covariate_trends,
)

logger = logging.getLogger(__name__)

_LN2_SQ: Final[float] = float(np.log(2.0) ** 2)
_COARSE_MASK_BLOCK_BINS: Final[int] = 100
_COARSE_BLOCK_Z_THRESHOLD: Final[float] = 3.5
_LOG_WAVE_CLIP: Final[float] = 4.0
_TAU_MIN: Final[float] = 0.7
_TAU_MAX: Final[float] = 2.5

__all__ = [
    "BranchModel",
    "BranchNormalizationResult",
    "ProjectedCoverage",
    "build_branch_model",
    "build_branch_model_from_tables",
    "half_sample_mode",
    "normalize_branch",
    "normalize_sample_branch",
    "project_coverage",
]


@dataclass(frozen=True, slots=True)
class BranchModel:
    """In-memory reference model and aligned sample counts for one branch."""

    branch: str
    contigs: tuple[str, ...]
    starts: NDArray[np.int64]
    ends: NDArray[np.int64]
    counts: NDArray[np.float64]
    masked: NDArray[np.bool_]
    pca_mean: NDArray[np.float64]
    pca_loadings: NDArray[np.float32]
    proto_ids: NDArray[np.uint16]
    proto_weights: NDArray[np.float16]
    mean_distance: NDArray[np.float32]
    null_mean: NDArray[np.float64]
    null_std: NDArray[np.float64]
    gc: NDArray[np.floating] | None = None
    mappability: NDArray[np.floating] | None = None
    mean_frag_len: NDArray[np.floating] | None = None
    endpoint_ratio: NDArray[np.floating] | None = None
    gc_quantiles: NDArray[np.float32] | None = None
    ref_frag_len: NDArray[np.float32] | None = None
    ref_ep_rate: NDArray[np.float32] | None = None
    bio_dispersion: NDArray[np.float64] | None = None
    bin_size: int | None = None
    is_flat: bool = False
    kappa_frag: NDArray[np.float32] | None = None


@dataclass(frozen=True, slots=True)
class ProjectedCoverage:
    """Multiplicatively projected sample coverage and expected counts."""

    normalized: NDArray[np.float64]
    corrected: NDArray[np.float64]
    usable: NDArray[np.bool_]
    expected_counts: NDArray[np.float64] | None = None


@dataclass(frozen=True, slots=True)
class BranchNormalizationResult:
    """Full per-bin output of zero-inclusive v2 branch normalization."""

    counts: NDArray[np.float64]
    expected_counts: NDArray[np.float64]
    bio_dispersion: NDArray[np.float64]
    ratios: NDArray[np.float64]
    zscores: NDArray[np.float64]
    weights: NDArray[np.float64]
    sigma_test: NDArray[np.float64]
    usable: NDArray[np.bool_]
    reference_sizes: NDArray[np.int64]


def half_sample_mode(values: NDArray[np.floating]) -> float:
    """Compute Robertson-Cryer O(N log N) iterative Half-Sample Mode."""
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    s = np.sort(finite)
    while s.size > 3:
        n = int(s.size)
        half = (n + 1) // 2
        widths = s[half - 1 :] - s[: n - half + 1]
        min_w = float(np.min(widths))
        ties = np.flatnonzero(widths <= min_w + 1e-15)
        best = int(ties[ties.size // 2])
        s = s[best : best + half]
    if s.size == 1:
        return float(s[0])
    if s.size == 2:
        return 0.5 * float(s[0] + s[1])
    d0 = float(s[1] - s[0])
    d1 = float(s[2] - s[1])
    if d0 < d1:
        return 0.5 * float(s[0] + s[1])
    if d1 < d0:
        return 0.5 * float(s[1] + s[2])
    return float(s[1])


def _hsm_biweight_mode(
    values: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
) -> float:
    """Refine Half-Sample Mode with 2-step precision-weighted Tukey biweight."""
    values = np.asarray(values, dtype=np.float64)
    finite_mask = np.isfinite(values)
    if weights is not None:
        w_arr = np.asarray(weights, dtype=np.float64)
        if w_arr.shape == finite_mask.shape:
            weights = w_arr[finite_mask]
    values = values[finite_mask]
    if values.size == 0:
        return 0.0
    m = half_sample_mode(values)
    if values.size < 6:
        return m
    mad = 1.4826 * float(np.median(np.abs(values - m)))
    if mad <= 1e-8:
        return m
    c = 3.5 * mad
    base_w = np.ones_like(values) if weights is None else np.maximum(weights, 0.0)
    for _ in range(2):
        u = (values - m) / c
        inlier = np.abs(u) < 1.0
        if not np.any(inlier):
            return m
        w = np.where(inlier, base_w * np.square(1.0 - np.square(u)), 0.0)
        w_sum = float(np.sum(w))
        if w_sum <= 0.0:
            return m
        m = float(np.sum(w * values)) / w_sum
    return m


def build_branch_model_from_tables(
    ref_table: pa.Table,
    branch: str,
    *,
    sample_data: pa.Table | SampleBuffers | None = None,
    ref_meta: ReferenceMetadata | None = None,
) -> BranchModel:
    """Construct a BranchModel from a Reference Parquet table and optional sample."""
    resolved_meta = (
        ref_meta
        if ref_meta is not None
        else ReferenceMetadata.from_schema(ref_table.schema)
    )
    arrs = extract_reference_branch_arrays(ref_table, branch, resolved_meta)
    if sample_data is not None:
        try:
            s_meta, s_counts = extract_sample_meta_and_counts(
                sample_data, validate=True
            )
        except SchemaError as err:
            raise PredictError(str(err)) from err
        s_idx = align_sample_grid_indices(
            s_meta, arrs.contigs, arrs.starts, arrs.ends, branch
        )
        counts = s_counts[s_idx]
        if resolve_fragmentomics_compatibility(sample_data, ref_table):
            f_len, e_rat = extract_bin_fragmentomic_covariates(sample_data)
            mean_frag_len: NDArray[np.floating] | None = f_len[s_idx].astype(np.float64)
            endpoint_ratio: NDArray[np.floating] | None = e_rat[s_idx].astype(
                np.float64
            )
        else:
            mean_frag_len, endpoint_ratio = None, None

        if (
            isinstance(sample_data, SampleBuffers)
            and sample_data.kappa_frag is not None
        ):
            kappa_frag: NDArray[np.float32] | None = sample_data.kappa_frag[s_idx]
        elif (
            isinstance(sample_data, pa.Table)
            and "kappa_frag" in sample_data.column_names
        ):
            kappa_frag = (
                sample_data.column("kappa_frag")
                .to_numpy(zero_copy_only=False)[s_idx]
                .astype(np.float32)
            )
        else:
            kappa_frag = None
    else:
        counts = np.zeros(arrs.starts.size, dtype=np.float64)
        mean_frag_len, endpoint_ratio = None, None
        kappa_frag = None

    return BranchModel(
        branch=branch,
        contigs=arrs.contigs,
        starts=arrs.starts,
        ends=arrs.ends,
        counts=counts,
        masked=arrs.masked,
        pca_mean=arrs.pca_mean,
        pca_loadings=arrs.pca_loadings,
        proto_ids=arrs.proto_ids,
        proto_weights=arrs.proto_weights,
        mean_distance=arrs.mean_distance,
        null_mean=arrs.null_mean,
        null_std=arrs.null_std,
        gc=arrs.gc,
        mappability=arrs.mappability,
        mean_frag_len=mean_frag_len,
        endpoint_ratio=endpoint_ratio,
        gc_quantiles=arrs.gc_quantiles,
        ref_frag_len=arrs.ref_frag_len,
        ref_ep_rate=arrs.ref_ep_rate,
        bio_dispersion=arrs.bio_dispersion,
        bin_size=arrs.bin_size,
        is_flat=arrs.is_flat,
        kappa_frag=kappa_frag,
    )


def build_branch_model(
    inputs: pa.Table,
    branch: str,
    ref_meta: ReferenceMetadata | None = None,
) -> BranchModel:
    """Build dense NumPy arrays for one reference branch from a Reference Parquet table."""
    return build_branch_model_from_tables(inputs, branch, ref_meta=ref_meta)


def _infer_bin_size(model: BranchModel) -> int:
    """Resolve branch nominal bin size in bp."""
    if model.bin_size is not None and model.bin_size > 0:
        return int(model.bin_size)
    spans = model.ends - model.starts
    return int(np.max(spans)) if spans.size > 0 else 5000


def _expand_branch_field(
    values: NDArray[np.floating],
    unmasked: NDArray[np.bool_],
    n_bins: int,
    *,
    name: str,
    fill: float = 0.0,
) -> NDArray[np.float64]:
    """Expand unmasked-indexed or full-length 1D array onto full branch length B."""
    arr = np.asarray(values, dtype=np.float64)
    u_count = int(np.count_nonzero(unmasked))
    if arr.ndim == 1 and arr.size == u_count:
        out = np.full(n_bins, fill, dtype=np.float64)
        out[unmasked] = arr
        return out
    if arr.ndim == 1 and arr.size == n_bins:
        return arr.copy()
    raise PredictError(f"{name} does not align with unmasked sample bins")


def _partition_contiguous_blocks(
    contig_ids: NDArray[np.int64],
    block_bins: int,
) -> list[tuple[int, int]]:
    """Partition ordered contig codes into contiguous [lo, hi) index slices."""
    n = int(contig_ids.size)
    if n == 0:
        return []
    k = max(1, block_bins)
    half_k = k // 2
    cuts = np.flatnonzero(contig_ids[1:] != contig_ids[:-1]) + 1
    c_starts = [0, *cuts.tolist()]
    c_ends = [*cuts.tolist(), n]
    blocks: list[tuple[int, int]] = []
    for r0, r1 in zip(c_starts, c_ends, strict=True):
        if r1 - r0 <= k:
            blocks.append((r0, r1))
            continue
        pts = list(range(r0, r1, k))
        if len(pts) > 1 and (r1 - pts[-1]) < half_k:
            pts.pop()
        pts.append(r1)
        for j in range(len(pts) - 1):
            blocks.append((pts[j], pts[j + 1]))
    return blocks


def _estimate_test_superblock_scale(
    counts_u: NDArray[np.float64],
    base_u: NDArray[np.float64],
    contig_ids_u: NDArray[np.int64],
    bin_size: int,
) -> float:
    """Estimate test-sample global depth factor s_test over 1 Mb super-blocks."""
    n_u = int(counts_u.size)
    if n_u == 0:
        return 1.0
    target_bp = max(1_000_000, 10 * max(bin_size, 1))
    bins_per_block = max(10, round(target_bp / max(bin_size, 1)))
    blocks = _partition_contiguous_blocks(contig_ids_u, bins_per_block)
    if len(blocks) < 4:
        blocks = _partition_contiguous_blocks(contig_ids_u, max(1, n_u // 8))

    ratios: list[float] = []
    for lo, hi in blocks:
        denom = float(np.sum(base_u[lo:hi]))
        num = float(np.sum(counts_u[lo:hi]))
        if denom > 0.0 and num > 0.0 and np.isfinite(denom) and np.isfinite(num):
            ratios.append(num / denom)

    if len(ratios) >= 2:
        med = float(np.median(np.asarray(ratios, dtype=np.float64)))
        if med > 0.0 and np.isfinite(med):
            return med

    tot_denom = float(np.sum(base_u))
    tot_num = float(np.sum(counts_u))
    if tot_denom > 0.0 and tot_num > 0.0:
        return tot_num / tot_denom
    return 1.0


def _project_expected_counts(
    counts: NDArray[np.float64],
    exposure: NDArray[np.floating],
    model: BranchModel,
    *,
    fit_mask: NDArray[np.bool_] | None = None,
    precomputed_design: NDArray[np.float64] | None = None,
    gc_override: NDArray[np.floating] | None = None,
    gc_quantiles_override: NDArray[np.float32] | None = None,
    mappability_override: NDArray[np.floating] | None = None,
    n_mask_override: NDArray[np.bool_] | None = None,
    min_mappability: float = DEFAULT_MIN_MAPPABILITY,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
    NDArray[np.float64],
]:
    """Fit exogenous Poisson covariate trend, super-block scale, and Huber PCA wave."""
    n_bins = int(counts.size)
    unmasked = ~np.asarray(model.masked, dtype=np.bool_)
    mu_full = _expand_branch_field(
        model.pca_mean, unmasked, n_bins, name="PCA mean", fill=0.0
    )

    gc = model.gc if gc_override is None else gc_override
    gc_q = (
        model.gc_quantiles if gc_quantiles_override is None else gc_quantiles_override
    )
    map_ = model.mappability if mappability_override is None else mappability_override
    cov_usable = mask_covariates(
        unmasked,
        gc=gc,
        mappability=map_,
        n_mask=n_mask_override,
        min_mappability=min_mappability,
    )
    finite_nonneg = np.isfinite(counts) & (counts >= 0.0)
    active_cov = cov_usable & finite_nonneg
    if float(np.sum(counts[active_cov])) <= 0.0:
        raise PredictError(
            f"branch {model.branch!r} has zero sample coverage on usable bins"
        )

    bin_size = _infer_bin_size(model)
    design_u = (
        precomputed_design
        if precomputed_design is not None
        else build_exogenous_covariate_basis(
            active_cov,
            bin_size=bin_size,
            gc=gc,
            gc_quantiles=gc_q,
            mappability=map_,
            ref_frag_len=model.ref_frag_len,
            ref_ep_rate=model.ref_ep_rate,
        )
    )
    mu_eff = np.where(mu_full > 0.0, mu_full, 1.0)
    eff_exp = np.asarray(exposure, dtype=np.float64) * mu_eff
    t_mu = fit_poisson_covariate_trends(
        counts,
        np.array([1.0], dtype=np.float64),
        eff_exp,
        active_cov,
        design_u,
        fit_mask=fit_mask,
    )[:, 0].astype(np.float64)

    fit_cov = (
        active_cov & np.asarray(fit_mask, dtype=np.bool_)
        if fit_mask is not None
        else active_cov
    )
    if float(np.sum(counts[fit_cov])) <= 0.0:
        fit_cov = active_cov
    _, c_ids_fit = np.unique(
        np.asarray(model.contigs, dtype=str)[fit_cov], return_inverse=True
    )
    s_test = _estimate_test_superblock_scale(
        counts[fit_cov],
        t_mu[fit_cov],
        c_ids_fit,
        bin_size,
    )
    e0 = np.zeros(n_bins, dtype=np.float64)
    e0[active_cov] = np.maximum(s_test * t_mu[active_cov], 1e-12)
    hat_lambda = e0.copy()

    u_idx = np.flatnonzero(unmasked)
    u_count = int(u_idx.size)
    ld_raw = np.asarray(model.pca_loadings, dtype=np.float64)
    if ld_raw.ndim == 2 and ld_raw.shape[0] > 0 and u_count > 0:
        if ld_raw.shape[1] == u_count:
            v_u = ld_raw
        elif ld_raw.shape[1] == n_bins:
            v_u = ld_raw[:, unmasked]
        else:
            raise PredictError("PCA loadings do not align with unmasked sample bins")

        act_in_u = active_cov[unmasked] & (e0[unmasked] > 0.0)
        fit_in_u = fit_cov[unmasked] & (e0[unmasked] > 0.0)
        if int(np.count_nonzero(fit_in_u)) < 2:
            fit_in_u = act_in_u
        n_fit_u = int(np.count_nonzero(fit_in_u))
        if n_fit_u >= 2:
            v_fit = v_u[:, fit_in_u]
            gram_v = 0.5 * (v_fit @ v_fit.T + (v_fit @ v_fit.T).T)
            evals_v, evecs_v = np.linalg.eigh(gram_v)
            tol_v = 1e-8 * max(float(np.max(evals_v)), 1.0)
            keep_v = evals_v > tol_v
            rank_eff = int(np.count_nonzero(keep_v))
            if n_fit_u > rank_eff > 0:
                proj_basis = evecs_v[:, keep_v].T
                v_red_fit = proj_basis @ v_fit
                v_red_u = proj_basis @ v_u
                e0_fit = e0[u_idx[fit_in_u]]
                c_fit = counts[u_idx[fit_in_u]]
                w_fit = np.sqrt(e0_fit + 0.5)
                y_fit = np.log((c_fit + 0.5) / (e0_fit + 0.5))
                x_w = (v_red_fit * w_fit[np.newaxis, :]).T
                y_w = y_fit * w_fit

                coef: NDArray[np.float64] | None = None
                if n_fit_u >= max(rank_eff + 2, 4) and float(np.std(y_w)) > 1e-12:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", ConvergenceWarning)
                            huber = HuberRegressor(
                                fit_intercept=False,
                                alpha=1e-4,
                                epsilon=1.35,
                                max_iter=100,
                            )
                            huber.fit(x_w, y_w)
                            coef = np.asarray(huber.coef_, dtype=np.float64)
                    except ValueError:
                        coef = None
                if coef is None or not bool(np.all(np.isfinite(coef))):
                    coef = np.linalg.lstsq(x_w, y_w, rcond=None)[0]
                eta_u = np.clip(v_red_u.T @ coef, -_LOG_WAVE_CLIP, _LOG_WAVE_CLIP)
                hat_lambda[u_idx] = np.where(
                    act_in_u,
                    np.maximum(e0[u_idx] * np.exp(eta_u), 1e-12),
                    0.0,
                )

    usable = active_cov & np.isfinite(hat_lambda) & (hat_lambda > 0.0)
    return e0, hat_lambda, usable, design_u


def project_coverage(
    model: BranchModel,
    *,
    gc_override: NDArray[np.floating] | None = None,
    mappability_override: NDArray[np.floating] | None = None,
    n_mask_override: NDArray[np.bool_] | None = None,
    min_mappability: float = DEFAULT_MIN_MAPPABILITY,
) -> ProjectedCoverage:
    """Project sample counts onto exogenous covariate trends and PCA subspace."""
    counts = np.asarray(model.counts, dtype=np.float64)
    exposure = compute_bin_exposure(model.starts, model.ends, _infer_bin_size(model))
    e0, hat_lambda, usable, _ = _project_expected_counts(
        counts,
        exposure,
        model,
        gc_override=gc_override,
        mappability_override=mappability_override,
        n_mask_override=n_mask_override,
        min_mappability=min_mappability,
    )
    normalized = np.full(counts.shape, np.nan, dtype=np.float64)
    corrected = np.full(counts.shape, np.nan, dtype=np.float64)
    normalized[usable] = (counts[usable] + 0.5) / (e0[usable] + 0.5)
    corrected[usable] = (counts[usable] + 0.5) / (hat_lambda[usable] + 0.5)
    return ProjectedCoverage(
        normalized=normalized,
        corrected=corrected,
        usable=usable,
        expected_counts=hat_lambda,
    )


def _calibrated_loco_baseline(
    q_u: NDArray[np.float64],
    donor_elig: NDArray[np.bool_],
    c_ids_u: NDArray[np.int64],
    pids_u: NDArray[np.uint16],
    pw_u: NDArray[np.float16],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Evaluate LOCO prototype baseline and center its mode to 1.0 across donors."""
    baselines, sup_u = evaluate_loco_prototype_baseline(
        np.where(donor_elig, q_u, np.nan), c_ids_u, pids_u, pw_u
    )
    b_pos = donor_elig & np.isfinite(baselines) & (baselines > 0.0)
    if q_u.size >= 20 and int(np.count_nonzero(b_pos)) >= 8:
        b_mode = half_sample_mode(baselines[b_pos])
        if b_mode > 0.0:
            baselines = baselines / b_mode
    return baselines, sup_u


def _empty_norm_result(
    counts_arr: NDArray[np.float64],
    bio_disp_full: NDArray[np.float64],
    reference_sizes: NDArray[np.int64],
) -> BranchNormalizationResult:
    """Return an empty BranchNormalizationResult when zero bins are usable."""
    n_bins = int(counts_arr.size)
    return BranchNormalizationResult(
        counts=counts_arr,
        expected_counts=np.zeros(n_bins, dtype=np.float64),
        bio_dispersion=bio_disp_full,
        ratios=np.full(n_bins, np.nan, dtype=np.float64),
        zscores=np.full(n_bins, np.nan, dtype=np.float64),
        weights=np.zeros(n_bins, dtype=np.float64),
        sigma_test=np.full(n_bins, np.nan, dtype=np.float64),
        usable=np.zeros(n_bins, dtype=np.bool_),
        reference_sizes=reference_sizes,
    )


def normalize_sample_branch(
    counts: NDArray[np.float64],
    exposure: NDArray[np.float32],
    model: BranchModel,
    *,
    mask_repeats: int = 3,
    kappa_frag: NDArray[np.float32] | None = None,
    min_reference_bins: int = 1,
    gc_override: NDArray[np.floating] | None = None,
    gc_quantiles_override: NDArray[np.float32] | None = None,
    mappability_override: NDArray[np.floating] | None = None,
    n_mask_override: NDArray[np.bool_] | None = None,
    min_mappability: float = DEFAULT_MIN_MAPPABILITY,
) -> BranchNormalizationResult:
    """Run zero-inclusive v2 normalization, coarse CNV donor masking, HSM, and variance synthesis."""
    counts_arr = np.asarray(counts, dtype=np.float64)
    n_bins = int(counts_arr.size)
    unmasked = ~np.asarray(model.masked, dtype=np.bool_)
    null_mean_full = _expand_branch_field(
        model.null_mean, unmasked, n_bins, name="null_mean", fill=0.0
    )
    dist_full = _expand_branch_field(
        model.mean_distance, unmasked, n_bins, name="mean_distance", fill=0.0
    )
    if model.bio_dispersion is not None:
        bio_disp_full = np.maximum(
            _expand_branch_field(
                model.bio_dispersion,
                unmasked,
                n_bins,
                name="bio_dispersion",
                fill=0.0,
            ),
            0.0,
        )
    else:
        null_std_full = _expand_branch_field(
            model.null_std, unmasked, n_bins, name="null_std", fill=0.0
        )
        bio_disp_full = _LN2_SQ * np.square(np.maximum(null_std_full, 0.0))

    _, hat_lambda, usable_pre, design_u = _project_expected_counts(
        counts_arr,
        exposure,
        model,
        gc_override=gc_override,
        gc_quantiles_override=gc_quantiles_override,
        mappability_override=mappability_override,
        n_mask_override=n_mask_override,
        min_mappability=min_mappability,
    )

    kappa_full = (
        np.ones(n_bins, dtype=np.float64)
        if kappa_frag is None
        else np.maximum(np.asarray(kappa_frag, dtype=np.float64), 1e-6)
    )
    reference_sizes = np.zeros(n_bins, dtype=np.int64)
    u_idx = np.flatnonzero(unmasked)
    u_count = int(u_idx.size)
    if u_count == 0:
        return _empty_norm_result(counts_arr, bio_disp_full, reference_sizes)

    contig_str = np.asarray(model.contigs, dtype=str)
    _, c_ids_u = np.unique(contig_str[u_idx], return_inverse=True)
    pids_u = (
        model.proto_ids[u_idx]
        if model.proto_ids.shape[0] == n_bins
        else model.proto_ids
    )
    pw_u = (
        model.proto_weights[u_idx]
        if model.proto_weights.shape[0] == n_bins
        else model.proto_weights
    )
    if pids_u.size == 0:
        pids_u = np.zeros((u_count, NUM_PROTOTYPES_PER_BIN), dtype=np.uint16)
        pw_u = np.zeros((u_count, NUM_PROTOTYPES_PER_BIN), dtype=np.float16)

    c_u = counts_arr[u_idx]
    lam_u = hat_lambda[u_idx]
    phi_u = bio_disp_full[u_idx]
    kap_u = kappa_full[u_idx]
    mu0_u = null_mean_full[u_idx]
    pre_u = usable_pre[u_idx]
    q_u = np.where(pre_u, (c_u + 0.5) / (lam_u + 0.5), np.nan)

    mask_block_len = (
        min(_COARSE_MASK_BLOCK_BINS, max(4, u_count // 16)) if u_count >= 16 else 1
    )
    coarse_blocks = _partition_contiguous_blocks(c_ids_u, mask_block_len)
    donor_elig = pre_u.copy()
    b_cal, sup_u = _calibrated_loco_baseline(q_u, donor_elig, c_ids_u, pids_u, pw_u)

    n_iters = max(1, mask_repeats)
    for it_idx in range(n_iters):
        ok_iter = (
            pre_u & (sup_u >= min_reference_bins) & np.isfinite(b_cal) & (b_cal > 0.0)
        )
        if not np.any(ok_iter):
            logger.debug(
                "branch %s within-sample iteration %d/%d: 0 eligible bins",
                model.branch,
                it_idx + 1,
                n_iters,
            )
            break
        eff_lam_it = np.where(ok_iter, lam_u * b_cal, 1.0)
        r_it = np.full(u_count, np.nan, dtype=np.float64)
        r_it[ok_iter] = np.log2(
            (c_u[ok_iter] + 0.5) / (eff_lam_it[ok_iter] + 0.5)
        ) - exact_poisson_log2_jensen_bias(eff_lam_it[ok_iter])
        f_loco_it = compute_kish_loco_inflation(
            c_ids_u, pids_u, pw_u, sup_u, eligible=donor_elig
        )
        sig0_it = np.sqrt(
            np.maximum(
                ((kap_u / (_LN2_SQ * (eff_lam_it + 0.5))) + (phi_u / _LN2_SQ))
                * f_loco_it,
                1e-12,
            )
        )
        prelim_shift = half_sample_mode(r_it[ok_iter])
        z_it = (r_it - prelim_shift - mu0_u) / sig0_it

        new_elig = pre_u.copy()
        for b_lo, b_hi in coarse_blocks:
            sub_ok = ok_iter[b_lo:b_hi]
            n_b = int(np.count_nonzero(sub_ok))
            if n_b > 0:
                z_blk = float(np.sum(z_it[b_lo:b_hi][sub_ok])) / float(np.sqrt(n_b))
                if abs(z_blk) > _COARSE_BLOCK_Z_THRESHOLD:
                    new_elig[b_lo:b_hi] = False

        logger.debug(
            "branch %s within-sample iteration %d/%d: %d eligible bins",
            model.branch,
            it_idx + 1,
            n_iters,
            int(np.count_nonzero(new_elig)),
        )
        if int(np.count_nonzero(new_elig)) < max(min_reference_bins, 2):
            break
        if np.array_equal(new_elig, donor_elig):
            break
        _, cand_sup = _calibrated_loco_baseline(q_u, new_elig, c_ids_u, pids_u, pw_u)
        if int(np.count_nonzero(pre_u & (cand_sup >= min_reference_bins))) < int(
            np.count_nonzero(ok_iter)
        ):
            break
        donor_elig = new_elig
        donor_full = np.zeros(n_bins, dtype=np.bool_)
        donor_full[u_idx] = donor_elig
        _, hat_lambda, usable_pre, _ = _project_expected_counts(
            counts_arr,
            exposure,
            model,
            fit_mask=donor_full,
            precomputed_design=design_u,
            gc_override=gc_override,
            gc_quantiles_override=gc_quantiles_override,
            mappability_override=mappability_override,
            n_mask_override=n_mask_override,
            min_mappability=min_mappability,
        )
        lam_u = hat_lambda[u_idx]
        pre_u = usable_pre[u_idx]
        q_u = np.where(pre_u, (c_u + 0.5) / (lam_u + 0.5), np.nan)
        b_cal, sup_u = _calibrated_loco_baseline(q_u, donor_elig, c_ids_u, pids_u, pw_u)

    reference_sizes[u_idx] = sup_u
    valid_u = pre_u & (sup_u >= min_reference_bins) & np.isfinite(b_cal) & (b_cal > 0.0)
    if not np.any(valid_u):
        return _empty_norm_result(counts_arr, bio_disp_full, reference_sizes)

    expected_out = np.zeros(n_bins, dtype=np.float64)
    ratios = np.full(n_bins, np.nan, dtype=np.float64)
    zscores = np.full(n_bins, np.nan, dtype=np.float64)
    weights = np.zeros(n_bins, dtype=np.float64)
    sigma_test = np.full(n_bins, np.nan, dtype=np.float64)

    f_loco_u = compute_kish_loco_inflation(
        c_ids_u, pids_u, pw_u, sup_u, eligible=donor_elig
    )
    eff_lam_v = lam_u[valid_u] * b_cal[valid_u]
    sig0_sq_v = np.maximum(
        ((kap_u[valid_u] / (_LN2_SQ * (eff_lam_v + 0.5))) + (phi_u[valid_u] / _LN2_SQ))
        * f_loco_u[valid_u],
        1e-12,
    )
    r_raw_u = np.full(u_count, np.nan, dtype=np.float64)
    r_raw_u[valid_u] = np.log2(
        (c_u[valid_u] + 0.5) / (eff_lam_v + 0.5)
    ) - exact_poisson_log2_jensen_bias(eff_lam_v)
    w0_u = np.zeros(u_count, dtype=np.float64)
    w0_u[valid_u] = 1.0 / sig0_sq_v

    hsm_block_len = min(50, max(12, u_count // 48)) if u_count >= 24 else 1
    hsm_blocks = _partition_contiguous_blocks(c_ids_u, hsm_block_len)
    blk_means_list: list[float] = []
    blk_weights_list: list[float] = []
    for lo, hi in hsm_blocks:
        w_blk = float(np.sum(w0_u[lo:hi]))
        if w_blk > 0.0:
            r_sub = np.where(valid_u[lo:hi], r_raw_u[lo:hi], 0.0)
            blk_means_list.append(float(np.sum(w0_u[lo:hi] * r_sub)) / w_blk)
            blk_weights_list.append(w_blk)
    hsm_shift = (
        _hsm_biweight_mode(
            np.asarray(blk_means_list, dtype=np.float64),
            np.asarray(blk_weights_list, dtype=np.float64),
        )
        if blk_means_list
        else 0.0
    )
    r_tilde_v = r_raw_u[valid_u] - hsm_shift

    usable_final = np.zeros(n_bins, dtype=np.bool_)
    valid_global = u_idx[valid_u]
    usable_final[valid_global] = True
    expected_out[valid_global] = eff_lam_v
    ratios[valid_global] = r_tilde_v

    sig0_sq_full = np.zeros(n_bins, dtype=np.float64)
    sig0_sq_full[valid_global] = sig0_sq_v
    tau_sample = 1.0
    if valid_global.size >= 5:
        same_adj = (valid_global[1:] == valid_global[:-1] + 1) & (
            contig_str[valid_global[1:]] == contig_str[valid_global[:-1]]
        )
        if int(np.count_nonzero(same_adj)) >= 4:
            i0 = valid_global[:-1][same_adj]
            i1 = valid_global[1:][same_adj]
            resid0 = ratios[i0] - null_mean_full[i0]
            resid1 = ratios[i1] - null_mean_full[i1]
            d_adj = (resid1 - resid0) / np.sqrt(
                np.maximum(sig0_sq_full[i0] + sig0_sq_full[i1], 1e-12)
            )
            mad_d = 1.4826 * float(np.median(np.abs(d_adj - float(np.median(d_adj)))))
            scale_d = mad_d if mad_d > 1e-8 else float(np.std(d_adj))
            if scale_d > 1e-8:
                tau_sample = float(np.clip(scale_d, _TAU_MIN, _TAU_MAX))

    sig_v = tau_sample * np.sqrt(sig0_sq_v)
    sigma_test[valid_global] = sig_v
    zscores[valid_global] = (r_tilde_v - null_mean_full[valid_global]) / sig_v
    weights_v = 1.0 / np.square(sig_v)

    if model.is_flat:
        d_v = dist_full[valid_global]
        pos_d = d_v[np.isfinite(d_v) & (d_v > 0.0)]
        if pos_d.size > 0:
            d_med = float(np.median(pos_d))
            weights_v = weights_v * np.clip(
                d_med / np.maximum(d_v, d_med / 20.0), 0.05, 20.0
            )
    weights[valid_global] = weights_v
    logger.debug(
        "branch %s: %d usable bins, median shift=%.4f (tau=%.3f)",
        model.branch,
        int(valid_global.size),
        hsm_shift,
        tau_sample,
    )

    return BranchNormalizationResult(
        counts=counts_arr,
        expected_counts=expected_out,
        bio_dispersion=bio_disp_full,
        ratios=ratios,
        zscores=zscores,
        weights=weights,
        sigma_test=sigma_test,
        usable=usable_final,
        reference_sizes=reference_sizes,
    )


def normalize_branch(model: BranchModel, config: PredictConfig) -> BinResult:
    """Normalize one branch according to PredictConfig and return BinResult."""
    logger.debug("normalizing branch %s (%d bins)", model.branch, model.counts.size)
    try:
        gc_override, n_mask_override = (
            load_gc_track(config.gc, model.contigs, model.starts, model.ends)
            if config.gc
            else (None, None)
        )
        gc_q_override = (
            load_fasta_gc_quantiles(config.gc, model.contigs, model.starts, model.ends)
            if config.gc
            else None
        )
        mappability_override = (
            load_mappability_track(
                config.mappability, model.contigs, model.starts, model.ends
            )
            if config.mappability
            else None
        )
    except CovariateError as err:
        raise PredictError(str(err)) from err

    exposure = compute_bin_exposure(model.starts, model.ends, _infer_bin_size(model))
    norm = normalize_sample_branch(
        model.counts,
        exposure,
        model,
        mask_repeats=config.mask_repeats,
        kappa_frag=model.kappa_frag,
        min_reference_bins=config.min_reference_bins,
        gc_override=gc_override,
        gc_quantiles_override=gc_q_override,
        mappability_override=mappability_override,
        n_mask_override=n_mask_override,
        min_mappability=config.min_mappability,
    )
    return BinResult(
        branch=model.branch,
        contigs=model.contigs,
        starts=model.starts,
        ends=model.ends,
        ratios=norm.ratios,
        zscores=norm.zscores,
        weights=norm.weights,
        reference_sizes=norm.reference_sizes,
        masked=~norm.usable,
        counts=norm.counts,
        expected_counts=norm.expected_counts,
    )
