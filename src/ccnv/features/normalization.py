"""Zero-inclusive per-branch sample normalization, PCA projection, and variance synthesis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from numpy.typing import NDArray

from ccnv.features.binning import compute_bin_exposure
from ccnv.features.covariates import mask_covariates
from ccnv.io.tracks import (
    load_gc_track,
    load_mappability_track,
)
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
    SampleMetadata,
    SchemaError,
    decode_dictionary_contigs,
    fixed_size_list_to_numpy,
    reconstruct_sample_grid,
    validate_schema,
)
from ccnv.stats.neighbors import (
    LN2_SQ,
    compute_kish_loco_inflation,
    evaluate_loco_prototype_baseline,
    exact_poisson_log2_jensen_bias,
)
from ccnv.stats.pca import LOG_WAVE_CLIP, project_huber_pca_wave
from ccnv.stats.regression import (
    build_exogenous_covariate_basis,
    fit_poisson_covariate_trends,
    half_sample_mode,
    hsm_biweight_mode,
)

logger = logging.getLogger(__name__)

_COARSE_MASK_BLOCK_BINS: Final[int] = 100
_COARSE_BLOCK_Z_THRESHOLD: Final[float] = 3.5
_TAU_MIN: Final[float] = 0.7
_TAU_MAX: Final[float] = 2.5

__all__ = [
    "BranchModel",
    "BranchNormalizationResult",
    "build_branch_model",
    "build_branch_model_from_tables",
    "half_sample_mode",
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


def _opt_sentinel_track[ScalarT: np.floating](
    col: pa.Array | pa.ChunkedArray,
    dtype: type[ScalarT],
) -> NDArray[ScalarT] | None:
    raw = np.asarray(col.combine_chunks().to_numpy(zero_copy_only=False), dtype=dtype)
    valid = np.isfinite(raw) & (raw >= 0.0)
    if not np.any(valid):
        return None
    return np.where(valid, raw, np.nan).astype(dtype, copy=False)


def _align_sample_grid_indices(
    s_meta: SampleMetadata,
    ref_contigs: tuple[str, ...],
    ref_starts: NDArray[np.int64],
    ref_ends: NDArray[np.int64],
    branch: str,
) -> NDArray[np.int64]:
    """Align reference branch bin coordinates to sample grid indices in O(B) time."""
    n_ref = len(ref_contigs)
    if n_ref == 0:
        return np.empty(0, dtype=np.int64)

    _, _, s_starts_u32, s_ends_u32 = reconstruct_sample_grid(s_meta)
    s_starts = s_starts_u32.astype(np.int64, copy=False)
    s_ends = s_ends_u32.astype(np.int64, copy=False)

    contig_slices: dict[str, tuple[int, int]] = {}
    offset = 0
    for c_name in s_meta.contig_order:
        c_bins = (
            s_meta.contig_lengths[c_name] + s_meta.bin_size - 1
        ) // s_meta.bin_size
        contig_slices[c_name] = (offset, offset + c_bins)
        offset += c_bins

    ref_c_arr = np.asarray(ref_contigs, dtype=object)
    out_idx = np.empty(n_ref, dtype=np.int64)
    for c_name in dict.fromkeys(ref_contigs):
        bounds = contig_slices.get(c_name)
        if bounds is None:
            raise PredictError(
                f"sample is missing reference bins for branch {branch!r} on contig {c_name!r}"
            )
        s_lo, s_hi = bounds
        s_s_sub = s_starts[s_lo:s_hi]
        s_e_sub = s_ends[s_lo:s_hi]
        r_mask = ref_c_arr == c_name
        r_s = ref_starts[r_mask]
        r_e = ref_ends[r_mask]
        if s_s_sub.size == 0:
            raise PredictError(
                f"sample has zero bins on contig {c_name!r} for branch {branch!r}"
            )
        rel_idx = np.searchsorted(s_s_sub, r_s)
        clipped = np.minimum(rel_idx, s_s_sub.size - 1)
        matched = (
            (rel_idx < s_s_sub.size)
            & (s_s_sub[clipped] == r_s)
            & (s_e_sub[clipped] == r_e)
        )
        if not np.all(matched):
            raise PredictError(
                f"sample bin coordinates do not match reference branch {branch!r} on contig {c_name!r}"
            )
        out_idx[r_mask] = s_lo + rel_idx
    return out_idx


def build_branch_model_from_tables(
    ref_table: pa.Table,
    branch: str,
    *,
    sample_data: pa.Table | SampleBuffers | None = None,
    ref_meta: ReferenceMetadata | None = None,
) -> BranchModel:
    """Construct a BranchModel from a Reference Parquet table and optional sample."""
    try:
        validate_schema(ref_table.schema, ArtifactKind.REFERENCE)
        resolved_meta = (
            ref_meta
            if ref_meta is not None
            else ReferenceMetadata.from_schema(ref_table.schema)
        )
        sub = ref_table.filter(pc.equal(ref_table.column("branch"), branch))
        if sub.num_rows == 0:
            raise PredictError(f"reference has no rows for branch {branch!r}")

        n_bins = sub.num_rows
        contigs = decode_dictionary_contigs(sub.column("contig"))
        starts = np.asarray(
            sub.column("start").combine_chunks().to_numpy(zero_copy_only=False),
            dtype=np.int64,
        )
        ends = np.asarray(
            sub.column("end").combine_chunks().to_numpy(zero_copy_only=False),
            dtype=np.int64,
        )
        masked = np.asarray(
            sub.column("masked").combine_chunks().to_numpy(zero_copy_only=False),
            dtype=np.bool_,
        )
        pca_mean = np.asarray(
            sub.column("pca_mean").combine_chunks().to_numpy(zero_copy_only=False),
            dtype=np.float64,
        )
        mean_distance = np.asarray(
            sub.column("mean_distance").combine_chunks().to_numpy(zero_copy_only=False),
            dtype=np.float32,
        )
        null_mean = np.asarray(
            sub.column("null_mean").combine_chunks().to_numpy(zero_copy_only=False),
            dtype=np.float64,
        )
        bio_dispersion = np.maximum(
            np.asarray(
                sub.column("bio_dispersion")
                .combine_chunks()
                .to_numpy(zero_copy_only=False),
                dtype=np.float64,
            ),
            0.0,
        )

        ld_col = sub.column("pca_loadings").combine_chunks()
        ld_type = ld_col.type
        if not isinstance(ld_type, pa.FixedSizeListType):
            raise SchemaError(
                f"expected FixedSizeListType for 'pca_loadings', found {ld_type}"
            )
        list_rank = ld_type.list_size
        k_branch = (
            resolved_meta.branch_pca_ranks.get(branch, list_rank)
            if resolved_meta.branch_pca_ranks is not None
            else list_rank
        )
        k_eff = max(0, min(int(k_branch), list_rank))
        if k_eff == 0 or list_rank == 0:
            pca_loadings = np.empty((0, n_bins), dtype=np.float32)
        else:
            ld_2d = fixed_size_list_to_numpy(ld_col, list_rank, np.float32)
            pca_loadings = np.ascontiguousarray(ld_2d[:, :k_eff].T, dtype=np.float32)

        proto_ids = fixed_size_list_to_numpy(
            sub.column("proto_ids"), NUM_PROTOTYPES_PER_BIN, np.uint16
        )
        proto_weights = fixed_size_list_to_numpy(
            sub.column("proto_weights"), NUM_PROTOTYPES_PER_BIN, np.float16
        )
        gc = _opt_sentinel_track(sub.column("gc"), np.float64)
        mappability = _opt_sentinel_track(sub.column("mappability"), np.float64)
        ref_frag_len = _opt_sentinel_track(sub.column("ref_frag_len"), np.float32)
        ref_ep_rate = _opt_sentinel_track(sub.column("ref_ep_rate"), np.float32)

        gc_q_col = sub.column("gc_quantiles").combine_chunks()
        if gc_q_col.null_count == n_bins:
            gc_quantiles: NDArray[np.float32] | None = None
        else:
            gc_q_raw = fixed_size_list_to_numpy(gc_q_col, NUM_GC_QUANTILES, np.float32)
            gc_quantiles = (
                gc_q_raw if np.any(np.isfinite(gc_q_raw) & (gc_q_raw >= 0.0)) else None
            )
    except SchemaError as err:
        raise PredictError(str(err)) from err

    if sample_data is not None:
        try:
            bufs = SampleBuffers.coerce(sample_data, validate=True)
        except SchemaError as err:
            raise PredictError(str(err)) from err
        s_idx = _align_sample_grid_indices(bufs.meta, contigs, starts, ends, branch)
        counts = np.asarray(bufs.count, dtype=np.float64)[s_idx]
        kappa_frag: NDArray[np.float32] | None = (
            bufs.kappa_frag[s_idx] if bufs.kappa_frag is not None else None
        )
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
        bin_size=resolved_meta.bin_size,
        is_flat=resolved_meta.flat,
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


def _align_branch_vector(
    values: NDArray[np.floating],
    unmasked: NDArray[np.bool_],
    n_bins: int,
    name: str,
) -> NDArray[np.float64]:
    """Return 1D float64 vector of length n_bins, expanding unmasked-only inputs if needed."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1 and arr.size == n_bins:
        return arr
    if arr.ndim == 1 and arr.size == int(np.count_nonzero(unmasked)):
        out = np.zeros(n_bins, dtype=np.float64)
        out[unmasked] = arr
        return out
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
    mu_full = _align_branch_vector(model.pca_mean, unmasked, n_bins, "PCA mean")

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
        if int(np.count_nonzero(fit_in_u)) >= 2:
            e0_fit = e0[u_idx[fit_in_u]]
            c_fit = counts[u_idx[fit_in_u]]
            w_fit = np.sqrt(e0_fit + 0.5)
            y_fit = np.log((c_fit + 0.5) / (e0_fit + 0.5))
            _, eta_u = project_huber_pca_wave(
                y_fit,
                v_u[:, fit_in_u],
                whitening_scale=w_fit,
                eval_loadings=v_u,
                clip=LOG_WAVE_CLIP,
            )
            hat_lambda[u_idx] = np.where(
                act_in_u,
                np.maximum(e0[u_idx] * np.exp(eta_u), 1e-12),
                0.0,
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
    null_mean_full = _align_branch_vector(
        model.null_mean, unmasked, n_bins, "null_mean"
    )
    dist_full = _align_branch_vector(
        model.mean_distance, unmasked, n_bins, "mean_distance"
    )
    bio_disp_full = np.maximum(
        _align_branch_vector(model.bio_dispersion, unmasked, n_bins, "bio_dispersion"),
        0.0,
    )

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
                ((kap_u / (LN2_SQ * (eff_lam_it + 0.5))) + (phi_u / LN2_SQ))
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

    f_loco_u = compute_kish_loco_inflation(
        c_ids_u, pids_u, pw_u, sup_u, eligible=donor_elig
    )
    eff_lam_v = lam_u[valid_u] * b_cal[valid_u]
    sig0_sq_v = np.maximum(
        ((kap_u[valid_u] / (LN2_SQ * (eff_lam_v + 0.5))) + (phi_u[valid_u] / LN2_SQ))
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
        hsm_biweight_mode(
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
    gc_override: NDArray[np.float64] | None = None,
    gc_quantiles_override: NDArray[np.float32] | None = None,
    n_mask_override: NDArray[np.bool_] | None = None,
    mappability_override: NDArray[np.float64] | None = None,
) -> BinResult:
    """Normalize one branch according to PredictConfig and return BinResult."""
    logger.debug("normalizing branch %s (%d bins)", model.branch, model.counts.size)
    try:
        if config.gc is not None and gc_override is None:
            gc_override, n_mask_override, gc_quantiles_override = load_gc_track(
                config.gc, model.contigs, model.starts, model.ends
            )
        if config.mappability is not None and mappability_override is None:
            mappability_override = load_mappability_track(
                config.mappability, model.contigs, model.starts, model.ends
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
        gc_quantiles_override=gc_quantiles_override,
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
