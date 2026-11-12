"""Zero-inclusive per-branch sample normalization, PCA projection, and variance synthesis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from numpy.typing import NDArray

from ccnv.features.binning import align_sample_to_grid_indices, compute_bin_exposure
from ccnv.features.covariates import mask_covariates
from ccnv.models import (
    DEFAULT_MIN_MAPPABILITY,
    NUM_GC_QUANTILES,
    NUM_PROTOTYPES_PER_BIN,
    ArtifactKind,
    BinResult,
    CovariateError,
    PredictConfig,
    PredictError,
    ReferenceMetadata,
    SampleBuffers,
    SchemaError,
    decode_dictionary_contigs,
    fixed_size_list_to_numpy,
    validate_schema,
)
from ccnv.stats import (
    LN2_SQ,
    LOG_WAVE_CLIP,
    build_exogenous_covariate_basis,
    compute_kish_loco_inflation,
    estimate_superblock_scales,
    evaluate_loco_prototype_baseline,
    exact_poisson_log2_jensen_bias,
    exact_poisson_log_variance,
    fit_poisson_covariate_trends,
    half_sample_mode,
    hsm_biweight_mode,
    mad_scale,
    partition_contiguous_superblocks,
    project_huber_pca_wave,
)

logger = logging.getLogger(__name__)

_COARSE_MASK_BLOCK_BINS: Final[int] = 100
_COARSE_BLOCK_Z_THRESHOLD: Final[float] = 3.5
_TAU_MIN: Final[float] = 0.7
_TAU_MAX: Final[float] = 2.5

__all__ = [
    "BranchModel",
    "BranchNormalizationResult",
    "align_reference_unmasked_bins",
    "build_branch_model_from_tables",
    "normalize_branch",
    "normalize_sample_branch",
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
    bio_dispersion: NDArray[np.float64]
    gc: NDArray[np.floating] | None = None
    mappability: NDArray[np.floating] | None = None
    gc_quantiles: NDArray[np.float32] | None = None
    ref_frag_len: NDArray[np.float32] | None = None
    ref_ep_rate: NDArray[np.float32] | None = None
    bin_size: int | None = None
    is_flat: bool = False
    kappa_frag: NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        """Enforce Schema v2.0.0 full-length per-bin array invariants."""
        n_bins = len(self.contigs)
        for name in (
            "starts",
            "ends",
            "counts",
            "masked",
            "pca_mean",
            "mean_distance",
            "null_mean",
            "bio_dispersion",
            "gc",
            "mappability",
            "ref_frag_len",
            "ref_ep_rate",
            "kappa_frag",
        ):
            arr = getattr(self, name)
            if arr is not None and arr.shape != (n_bins,):
                raise PredictError(
                    f"BranchModel.{name} shape {arr.shape} != ({n_bins},)"
                )
        if self.gc_quantiles is not None and self.gc_quantiles.shape != (
            n_bins,
            NUM_GC_QUANTILES,
        ):
            raise PredictError(
                f"BranchModel.gc_quantiles shape {self.gc_quantiles.shape} != ({n_bins}, {NUM_GC_QUANTILES})"
            )
        if self.pca_loadings.ndim != 2 or self.pca_loadings.shape[1] != n_bins:
            raise PredictError(
                f"BranchModel.pca_loadings shape {self.pca_loadings.shape} must be 2D with width {n_bins}"
            )
        if (
            self.proto_ids.ndim != 2
            or self.proto_ids.shape[0] != n_bins
            or self.proto_weights.shape != self.proto_ids.shape
            or (n_bins > 0 and self.proto_ids.shape[1] < 1)
        ):
            raise PredictError(
                f"BranchModel prototype arrays {self.proto_ids.shape}/{self.proto_weights.shape} invalid for {n_bins} bins"
            )


@dataclass(frozen=True, slots=True)
class BranchNormalizationResult:
    """Full per-bin output of zero-inclusive v2 branch normalization."""

    counts: NDArray[np.float64]
    expected_counts: NDArray[np.float64]
    bio_dispersion: NDArray[np.float64]
    ratios: NDArray[np.float64]
    zscores: NDArray[np.float64]
    weights: NDArray[np.float64]
    usable: NDArray[np.bool_]
    reference_sizes: NDArray[np.int64]


def _col_np[ScalarT: np.generic](
    sub: pa.Table, name: str, dtype: type[ScalarT]
) -> NDArray[ScalarT]:
    return np.asarray(
        sub.column(name).combine_chunks().to_numpy(zero_copy_only=False), dtype=dtype
    )


def _opt_sentinel_track[ScalarT: np.floating](
    sub: pa.Table, name: str, dtype: type[ScalarT]
) -> NDArray[ScalarT] | None:
    raw = _col_np(sub, name, dtype)
    valid = np.isfinite(raw) & (raw >= 0.0)
    return (
        np.where(valid, raw, np.nan).astype(dtype, copy=False)
        if np.any(valid)
        else None
    )


def align_reference_unmasked_bins(
    sample: SampleBuffers, reference: pa.Table
) -> NDArray[np.bool_]:
    """Project unmasked reference table intervals onto the sample bin grid."""
    unmasked_sample = np.zeros(sample.num_rows, dtype=np.bool_)
    if reference.num_rows == 0:
        return unmasked_sample
    ref_c = decode_dictionary_contigs(reference.column("contig"))
    ref_s = _col_np(reference, "start", np.int64)
    ref_e = _col_np(reference, "end", np.int64)
    ref_unmasked = ~_col_np(reference, "masked", np.bool_)
    s_idx = align_sample_to_grid_indices(sample.meta, ref_c, ref_s, ref_e)
    hit = (s_idx >= 0) & ref_unmasked
    if np.any(hit):
        unmasked_sample[s_idx[hit]] = True
    return unmasked_sample


def build_branch_model_from_tables(
    ref_table: pa.Table,
    branch: str,
    *,
    sample: SampleBuffers | None = None,
    ref_meta: ReferenceMetadata | None = None,
) -> BranchModel:
    """Construct a BranchModel from a Reference Parquet table and optional SampleBuffers."""
    try:
        validate_schema(ref_table.schema, ArtifactKind.REFERENCE)
        meta = ref_meta or ReferenceMetadata.from_schema(ref_table.schema)
        sub = ref_table.filter(pc.equal(ref_table.column("branch"), branch))
        if sub.num_rows == 0:
            raise PredictError(f"reference has no rows for branch {branch!r}")

        n_bins = sub.num_rows
        contigs = decode_dictionary_contigs(sub.column("contig"))
        starts, ends = _col_np(sub, "start", np.int64), _col_np(sub, "end", np.int64)
        masked = _col_np(sub, "masked", np.bool_)
        pca_mean = _col_np(sub, "pca_mean", np.float64)
        mean_distance = _col_np(sub, "mean_distance", np.float32)
        null_mean = _col_np(sub, "null_mean", np.float64)
        bio_dispersion = np.maximum(_col_np(sub, "bio_dispersion", np.float64), 0.0)

        ld_col = sub.column("pca_loadings").combine_chunks()
        if not isinstance(ld_col.type, pa.FixedSizeListType):
            raise SchemaError(
                f"expected FixedSizeListType for 'pca_loadings', found {ld_col.type}"
            )
        list_rank = ld_col.type.list_size
        k_branch = (
            meta.branch_pca_ranks.get(branch, list_rank)
            if meta.branch_pca_ranks is not None
            else list_rank
        )
        k_eff = max(0, min(int(k_branch), list_rank))
        pca_loadings = (
            np.empty((0, n_bins), dtype=np.float32)
            if k_eff == 0 or list_rank == 0
            else np.ascontiguousarray(
                fixed_size_list_to_numpy(ld_col, list_rank, np.float32)[:, :k_eff].T,
                dtype=np.float32,
            )
        )
        proto_ids = fixed_size_list_to_numpy(
            sub.column("proto_ids"), NUM_PROTOTYPES_PER_BIN, np.uint16
        )
        proto_weights = fixed_size_list_to_numpy(
            sub.column("proto_weights"), NUM_PROTOTYPES_PER_BIN, np.float16
        )
        gc = _opt_sentinel_track(sub, "gc", np.float64)
        mappability = _opt_sentinel_track(sub, "mappability", np.float64)
        ref_frag_len = _opt_sentinel_track(sub, "ref_frag_len", np.float32)
        ref_ep_rate = _opt_sentinel_track(sub, "ref_ep_rate", np.float32)

        gc_q_col = sub.column("gc_quantiles").combine_chunks()
        gc_quantiles: NDArray[np.float32] | None = None
        if gc_q_col.null_count < n_bins:
            gc_q_raw = fixed_size_list_to_numpy(gc_q_col, NUM_GC_QUANTILES, np.float32)
            if np.any(np.isfinite(gc_q_raw) & (gc_q_raw >= 0.0)):
                gc_quantiles = gc_q_raw
    except SchemaError as err:
        raise PredictError(str(err)) from err

    if sample is not None:
        if not isinstance(sample, SampleBuffers):
            raise PredictError("sample must be a SampleBuffers instance")
        s_idx = align_sample_to_grid_indices(
            sample.meta, contigs, starts, ends, strict_branch=branch
        )
        counts = np.asarray(sample.count, dtype=np.float64)[s_idx]
        kappa_frag = sample.kappa_frag[s_idx] if sample.kappa_frag is not None else None
    else:
        counts = np.zeros(starts.size, dtype=np.float64)
        kappa_frag = None

    return BranchModel(
        branch=branch,
        contigs=contigs,
        starts=starts,
        ends=ends,
        counts=counts,
        masked=masked,
        pca_mean=pca_mean,
        pca_loadings=pca_loadings,
        proto_ids=proto_ids,
        proto_weights=proto_weights,
        mean_distance=mean_distance,
        null_mean=null_mean,
        bio_dispersion=bio_dispersion,
        gc=gc,
        mappability=mappability,
        gc_quantiles=gc_quantiles,
        ref_frag_len=ref_frag_len,
        ref_ep_rate=ref_ep_rate,
        bin_size=meta.bin_size,
        is_flat=meta.flat,
        kappa_frag=kappa_frag,
    )


def _infer_bin_size(model: BranchModel) -> int:
    if model.bin_size is not None and model.bin_size > 0:
        return int(model.bin_size)
    spans = model.ends - model.starts
    return int(np.max(spans)) if spans.size > 0 else 5000


def _project_expected_counts(
    counts: NDArray[np.float64],
    exposure: NDArray[np.floating],
    model: BranchModel,
    contig_ids: NDArray[np.int64],
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
    mu_full = np.asarray(model.pca_mean, dtype=np.float64)

    gc = model.gc if gc_override is None else gc_override
    gc_q = (
        gc_quantiles_override
        if (gc_override is not None or gc_quantiles_override is not None)
        else model.gc_quantiles
    )
    map_ = model.mappability if mappability_override is None else mappability_override
    cov_usable = mask_covariates(
        unmasked,
        gc=gc,
        mappability=map_,
        n_mask=n_mask_override,
        min_mappability=min_mappability,
    )
    active_cov = cov_usable & np.isfinite(counts) & (counts >= 0.0)
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
    eff_exp = np.asarray(exposure, dtype=np.float64) * np.where(
        mu_full > 0.0, mu_full, 1.0
    )
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

    s_test = float(
        estimate_superblock_scales(
            counts, contig_ids, usable=fit_cov, bin_size=bin_size, reference=t_mu
        )[0]
    )
    e0 = np.zeros(n_bins, dtype=np.float64)
    e0[active_cov] = np.maximum(s_test * t_mu[active_cov], 1e-12)
    hat_lambda = e0.copy()

    u_idx = np.flatnonzero(unmasked)
    ld_raw = np.asarray(model.pca_loadings, dtype=np.float64)
    if ld_raw.shape[0] > 0 and u_idx.size > 0:
        v_u = ld_raw[:, u_idx]
        act_in_u = active_cov[u_idx] & (e0[u_idx] > 0.0)
        fit_in_u = fit_cov[u_idx] & (e0[u_idx] > 0.0)
        if int(np.count_nonzero(fit_in_u)) < 2:
            fit_in_u = act_in_u
        if int(np.count_nonzero(fit_in_u)) >= 2:
            fit_u = u_idx[fit_in_u]
            y_fit = np.log((counts[fit_u] + 0.5) / (e0[fit_u] + 0.5))
            _, eta_u = project_huber_pca_wave(
                y_fit,
                v_u[:, fit_in_u],
                whitening_scale=np.sqrt(e0[fit_u] + 0.5),
                eval_loadings=v_u,
                eval_whitening_scale=np.sqrt(e0[u_idx] + 0.5),
                clip=LOG_WAVE_CLIP,
            )
            hat_lambda[u_idx] = np.where(
                act_in_u, np.maximum(e0[u_idx] * np.exp(eta_u), 1e-12), 0.0
            )

    usable = active_cov & np.isfinite(hat_lambda) & (hat_lambda > 0.0)
    return e0, hat_lambda, usable, design_u


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


def _evaluate_loco_scores(
    c_u: NDArray[np.float64],
    lam_u: NDArray[np.float64],
    b_cal: NDArray[np.float64],
    sup_u: NDArray[np.int64],
    ok_u: NDArray[np.bool_],
    donor_elig: NDArray[np.bool_],
    c_ids_u: NDArray[np.int64],
    pids_u: NDArray[np.uint16],
    pw_u: NDArray[np.float16],
    kap_u: NDArray[np.float64],
    phi_u: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate effective expected counts, Jensen-debiased log2 ratios, and exact variances on ``ok_u``."""
    eff_lam_v = lam_u[ok_u] * b_cal[ok_u]
    r_raw_v = np.log2(
        (c_u[ok_u] + 0.5) / (eff_lam_v + 0.5)
    ) - exact_poisson_log2_jensen_bias(eff_lam_v)
    f_loco_u = compute_kish_loco_inflation(
        c_ids_u, pids_u, pw_u, sup_u, eligible=donor_elig
    )
    shot_var_v = kap_u[ok_u] * exact_poisson_log_variance(eff_lam_v) / LN2_SQ
    sig0_sq_v = np.maximum((shot_var_v + phi_u[ok_u] / LN2_SQ) * f_loco_u[ok_u], 1e-12)
    return eff_lam_v, r_raw_v, sig0_sq_v


def _empty_norm_result(
    counts_arr: NDArray[np.float64],
    bio_disp_full: NDArray[np.float64],
    reference_sizes: NDArray[np.int64],
) -> BranchNormalizationResult:
    n_bins = int(counts_arr.size)
    return BranchNormalizationResult(
        counts=counts_arr,
        expected_counts=np.zeros(n_bins, dtype=np.float64),
        bio_dispersion=bio_disp_full,
        ratios=np.full(n_bins, np.nan, dtype=np.float64),
        zscores=np.full(n_bins, np.nan, dtype=np.float64),
        weights=np.zeros(n_bins, dtype=np.float64),
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
    exp_arr = np.asarray(exposure, dtype=np.float64)
    n_bins = len(model.contigs)
    if counts_arr.shape != (n_bins,) or exp_arr.shape != (n_bins,):
        raise PredictError(f"counts/exposure must match model bin count ({n_bins},)")

    unmasked = ~np.asarray(model.masked, dtype=np.bool_)
    null_mean_full = np.asarray(model.null_mean, dtype=np.float64)
    dist_full = np.asarray(model.mean_distance, dtype=np.float64)
    bio_disp_full = np.maximum(np.asarray(model.bio_dispersion, dtype=np.float64), 0.0)
    contig_str = np.asarray(model.contigs, dtype=str)
    _, contig_ids_full = np.unique(contig_str, return_inverse=True)

    _, hat_lambda, usable_pre, design_u = _project_expected_counts(
        counts_arr,
        exp_arr,
        model,
        contig_ids_full,
        gc_override=gc_override,
        gc_quantiles_override=gc_quantiles_override,
        mappability_override=mappability_override,
        n_mask_override=n_mask_override,
        min_mappability=min_mappability,
    )

    eff_kappa = kappa_frag if kappa_frag is not None else model.kappa_frag
    if eff_kappa is None:
        kappa_full = np.ones(n_bins, dtype=np.float64)
    else:
        kap_raw = np.asarray(eff_kappa, dtype=np.float64)
        if kap_raw.shape != (n_bins,):
            raise PredictError(f"kappa_frag shape {kap_raw.shape} != ({n_bins},)")
        kap_clean = np.where(np.isfinite(kap_raw), np.maximum(kap_raw, 1.0), 1.0)
        elig_kap = unmasked & np.isfinite(counts_arr) & (counts_arr >= 5.0)
        if np.any(elig_kap):
            kap_prior = float(np.median(kap_clean[elig_kap]))
        elif np.any(unmasked):
            kap_prior = float(np.median(kap_clean[unmasked]))
        else:
            kap_prior = 1.0
        n_eff = np.where(np.isfinite(counts_arr), np.maximum(counts_arr, 0.0), 0.0)
        kappa_full = (n_eff * kap_clean + 10.0 * kap_prior) / (n_eff + 10.0)

    reference_sizes = np.zeros(n_bins, dtype=np.int64)
    u_idx = np.flatnonzero(unmasked)
    u_count = int(u_idx.size)
    if u_count == 0:
        return _empty_norm_result(counts_arr, bio_disp_full, reference_sizes)

    c_ids_u = contig_ids_full[u_idx]
    pids_u, pw_u = model.proto_ids[u_idx], model.proto_weights[u_idx]
    c_u, lam_u = counts_arr[u_idx], hat_lambda[u_idx]
    phi_u, kap_u, mu0_u = bio_disp_full[u_idx], kappa_full[u_idx], null_mean_full[u_idx]
    pre_u = usable_pre[u_idx]
    q_u = np.where(pre_u, (c_u + 0.5) / (lam_u + 0.5), np.nan)

    mask_block_len = (
        min(_COARSE_MASK_BLOCK_BINS, max(4, u_count // 16)) if u_count >= 16 else 1
    )
    coarse_starts, coarse_ends = partition_contiguous_superblocks(
        c_ids_u, block_bins=mask_block_len
    )
    donor_elig = pre_u.copy()
    b_cal, sup_u = _calibrated_loco_baseline(q_u, donor_elig, c_ids_u, pids_u, pw_u)

    n_iters = max(1, mask_repeats)
    for it_idx in range(n_iters):
        ok_iter = (
            pre_u & (sup_u >= min_reference_bins) & np.isfinite(b_cal) & (b_cal > 0.0)
        )
        if not np.any(ok_iter):
            break
        _, r_raw_it, sig0_sq_it = _evaluate_loco_scores(
            c_u,
            lam_u,
            b_cal,
            sup_u,
            ok_iter,
            donor_elig,
            c_ids_u,
            pids_u,
            pw_u,
            kap_u,
            phi_u,
        )
        prelim_shift = half_sample_mode(r_raw_it)
        z_u = np.zeros(u_count, dtype=np.float64)
        z_u[ok_iter] = (r_raw_it - prelim_shift - mu0_u[ok_iter]) / np.sqrt(sig0_sq_it)

        new_elig = pre_u.copy()
        n_blk = np.add.reduceat(ok_iter.astype(np.int64), coarse_starts)
        sum_z_blk = np.add.reduceat(z_u, coarse_starts)
        for b_idx in np.flatnonzero(
            (n_blk > 0)
            & (np.abs(sum_z_blk) > _COARSE_BLOCK_Z_THRESHOLD * np.sqrt(n_blk))
        ):
            new_elig[coarse_starts[b_idx] : coarse_ends[b_idx]] = False

        logger.debug(
            "branch %s within-sample iteration %d/%d: %d eligible bins",
            model.branch,
            it_idx + 1,
            n_iters,
            int(np.count_nonzero(new_elig)),
        )
        if int(np.count_nonzero(new_elig)) < max(
            min_reference_bins, 2
        ) or np.array_equal(new_elig, donor_elig):
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
            exp_arr,
            model,
            contig_ids_full,
            fit_mask=donor_full,
            precomputed_design=design_u,
            gc_override=gc_override,
            gc_quantiles_override=gc_quantiles_override,
            mappability_override=mappability_override,
            n_mask_override=n_mask_override,
            min_mappability=min_mappability,
        )
        lam_u, pre_u = hat_lambda[u_idx], usable_pre[u_idx]
        q_u = np.where(pre_u, (c_u + 0.5) / (lam_u + 0.5), np.nan)
        b_cal, sup_u = _calibrated_loco_baseline(q_u, donor_elig, c_ids_u, pids_u, pw_u)

    reference_sizes[u_idx] = sup_u
    valid_u = pre_u & (sup_u >= min_reference_bins) & np.isfinite(b_cal) & (b_cal > 0.0)
    if not np.any(valid_u):
        return _empty_norm_result(counts_arr, bio_disp_full, reference_sizes)

    eff_lam_v, r_raw_v, sig0_sq_v = _evaluate_loco_scores(
        c_u,
        lam_u,
        b_cal,
        sup_u,
        valid_u,
        donor_elig,
        c_ids_u,
        pids_u,
        pw_u,
        kap_u,
        phi_u,
    )
    r_raw_u = np.zeros(u_count, dtype=np.float64)
    w0_u = np.zeros(u_count, dtype=np.float64)
    r_raw_u[valid_u], w0_u[valid_u] = r_raw_v, 1.0 / sig0_sq_v

    hsm_block_len = min(50, max(12, u_count // 48)) if u_count >= 24 else 1
    hsm_starts, _ = partition_contiguous_superblocks(c_ids_u, block_bins=hsm_block_len)
    w_blk = np.add.reduceat(w0_u, hsm_starts)
    r_w_blk = np.add.reduceat(w0_u * r_raw_u, hsm_starts)
    pos_blk = w_blk > 0.0
    hsm_shift = (
        hsm_biweight_mode(r_w_blk[pos_blk] / w_blk[pos_blk], w_blk[pos_blk])
        if np.any(pos_blk)
        else 0.0
    )
    r_tilde_v = r_raw_v - hsm_shift

    expected_out = np.zeros(n_bins, dtype=np.float64)
    ratios = np.full(n_bins, np.nan, dtype=np.float64)
    zscores = np.full(n_bins, np.nan, dtype=np.float64)
    weights = np.zeros(n_bins, dtype=np.float64)
    usable_final = np.zeros(n_bins, dtype=np.bool_)

    valid_global = u_idx[valid_u]
    usable_final[valid_global] = True
    expected_out[valid_global] = eff_lam_v * float(2.0**hsm_shift)
    ratios[valid_global] = r_tilde_v

    sig0_sq_full = np.zeros(n_bins, dtype=np.float64)
    sig0_sq_full[valid_global] = sig0_sq_v
    tau_sample = 1.0
    if valid_global.size >= 5:
        same_adj = (valid_global[1:] == valid_global[:-1] + 1) & (
            contig_str[valid_global[1:]] == contig_str[valid_global[:-1]]
        )
        if int(np.count_nonzero(same_adj)) >= 4:
            i0, i1 = valid_global[:-1][same_adj], valid_global[1:][same_adj]
            d_adj = (
                (ratios[i1] - null_mean_full[i1]) - (ratios[i0] - null_mean_full[i0])
            ) / np.sqrt(np.maximum(sig0_sq_full[i0] + sig0_sq_full[i1], 1e-12))
            mad_d = mad_scale(d_adj)
            scale_d = mad_d if mad_d > 1e-8 else float(np.std(d_adj))
            if scale_d > 1e-8:
                tau_sample = float(np.clip(scale_d, _TAU_MIN, _TAU_MAX))

    sig_v = tau_sample * np.sqrt(sig0_sq_v)
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
        usable=usable_final,
        reference_sizes=reference_sizes,
    )


def normalize_branch(
    model: BranchModel,
    config: PredictConfig,
    *,
    gc_override: NDArray[np.floating] | None = None,
    gc_quantiles_override: NDArray[np.float32] | None = None,
    n_mask_override: NDArray[np.bool_] | None = None,
    mappability_override: NDArray[np.floating] | None = None,
) -> BinResult:
    """Normalize one branch according to PredictConfig and return BinResult."""
    logger.debug("normalizing branch %s (%d bins)", model.branch, model.counts.size)
    exposure = compute_bin_exposure(model.starts, model.ends, _infer_bin_size(model))
    try:
        norm = normalize_sample_branch(
            model.counts,
            exposure,
            model,
            mask_repeats=config.mask_repeats,
            kappa_frag=model.kappa_frag,
            min_reference_bins=config.min_reference_bins,
            gc_override=gc_override,
            gc_quantiles_override=gc_quantiles_override,
            mappability_override=mappability_override,
            n_mask_override=n_mask_override,
            min_mappability=config.min_mappability,
        )
    except CovariateError as err:
        raise PredictError(str(err)) from err

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
