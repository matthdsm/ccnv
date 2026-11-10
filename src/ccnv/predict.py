"""Typed copy-number normalization, segmentation, and calling."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Literal

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import scipy.stats
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import HuberRegressor

from ccnv.features.cohort import rebin_sample_table
from ccnv.features.covariates import (
    correct_covariate_bias,
    mask_covariates,
    validate_min_mappability,
)
from ccnv.features.fragmentomics import (
    apply_prenatal_fisher_weighting,
)
from ccnv.features.prenatal import annotate_prenatal_segments, predict_prenatal
from ccnv.io.parquet import load_sample_or_extract, read_reference_parquet
from ccnv.io.tracks import load_cytobands_track, load_gc_track, load_mappability_track
from ccnv.models import (
    BRANCH_AUTOSOMAL,
    BRANCH_FEMALE_SEX,
    BRANCH_MALE_SEX,
    DEFAULT_MIN_MAPPABILITY,
    AnalysisMode,
    BinResult,
    CallType,
    CovariateError,
    FetalFractionResult,
    PredictConfig,
    PredictError,
    PredictionOutputs,
    PrenatalModelMetadata,
    SegmentResult,
    bin_coordinates,
    decode_metadata,
)
from ccnv.qc.predict import build_predict_qc_payload, write_predict_qc_json
from ccnv.report import write_outputs
from ccnv.stats.segmentation import segment_genome

Sex = Literal["female", "male"]
_MASK_Z_THRESHOLD: Final[float] = 2.326347874
logger = logging.getLogger(__name__)

__all__ = [
    "BranchModel",
    "PredictError",
    "PredictionInput",
    "ProjectedCoverage",
    "call_segment",
    "load_inputs",
    "normalize_branch",
    "project_coverage",
    "run",
    "run_predict",
    "segment_and_call",
]


@dataclass(frozen=True, slots=True)
class PredictionInput:
    sample: pa.Table
    reference: pa.Table
    contig_lengths: dict[str, int]
    sex: Sex | None
    branches: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BranchModel:
    branch: str
    contigs: tuple[str, ...]
    starts: NDArray[np.int64]
    ends: NDArray[np.int64]
    counts: NDArray[np.float64]
    masked: NDArray[np.bool_]
    pca_mean: NDArray[np.float64]
    pca_loadings: NDArray[np.float64]
    nearest_indices: tuple[NDArray[np.int32] | NDArray[np.int64], ...]
    mean_distance: NDArray[np.float32]
    null_mean: NDArray[np.float64]
    null_std: NDArray[np.float64]
    gc: NDArray[np.floating] | None = None
    mappability: NDArray[np.floating] | None = None
    nearest_distances: tuple[NDArray[np.float32], ...] = ()


@dataclass(frozen=True, slots=True)
class ProjectedCoverage:
    normalized: NDArray[np.float64]
    corrected: NDArray[np.float64]
    usable: NDArray[np.bool_]


def run(config: PredictConfig) -> PredictionOutputs:
    """Run prediction and write bin, segment, call, and optional plot artifacts."""
    _validate_config(config)
    if config.cytobands is not None and not config.plot:
        logger.warning("--cytobands is ignored when --no-plot is set")
    logger.debug("loading sample %s and reference %s", config.sample, config.reference)
    inputs = load_inputs(
        config.sample,
        config.reference,
        sex=config.sex,
        mode=config.mode,
        fasta=config.fasta,
    )
    prenatal_meta: PrenatalModelMetadata | None = None
    prenatal_result: FetalFractionResult | None = None
    if config.mode == "prenatal":
        prenatal_meta = PrenatalModelMetadata.from_table(inputs.reference)
        if prenatal_meta is None:
            raise PredictError("reference is missing prenatal metadata (ccnv.prenatal)")
        prenatal_result = predict_prenatal(
            inputs.sample,
            inputs.reference,
            prenatal_meta,
            min_ff_warning=config.min_fetal_fraction_warning,
        )
        inputs = replace(
            inputs, sample=apply_prenatal_fisher_weighting(inputs.sample, prenatal_meta)
        )

    bin_results = tuple(
        normalize_branch(build_branch_model(inputs, branch), config)
        for branch in inputs.branches
    )
    segments = segment_and_call(
        bin_results,
        inputs.contig_lengths,
        penalty=config.penalty,
        z_threshold=config.z_threshold,
        ratio_threshold=config.ratio_threshold,
        max_blacklist_gap=config.max_blacklist_gap,
    )
    if config.mode == "prenatal" and prenatal_result is not None:
        segments = tuple(
            annotate_prenatal_segments(
                segments,
                prenatal_result,
                z_threshold=config.z_threshold,
                sample_table=inputs.sample,
                prenatal_meta=prenatal_meta,
            )
        )
    qc_json_path = Path(f"{config.prefix}_qc.json")
    write_predict_qc_json(
        qc_json_path,
        build_predict_qc_payload(
            mode=config.mode,
            sample_table=inputs.sample,
            segments=segments,
            prenatal_result=prenatal_result,
        ),
    )
    cytobands_track = (
        load_cytobands_track(config.cytobands, inputs.contig_lengths)
        if (config.plot and config.cytobands is not None)
        else None
    )
    return write_outputs(
        prefix=config.prefix,
        bins=bin_results,
        segments=segments,
        contig_lengths=inputs.contig_lengths,
        plot=config.plot,
        vcf=config.vcf,
        cytobands=cytobands_track,
        z_threshold=config.z_threshold,
        ratio_threshold=config.ratio_threshold,
        prenatal_result=prenatal_result,
        mode=config.mode,
        qc_json_path=qc_json_path,
    )


def load_inputs(
    sample_path: Path,
    reference_path: Path,
    *,
    sex: Literal["autosomal", "female", "male"] = "autosomal",
    mode: AnalysisMode = "germline",
    fasta: Path | None = None,
) -> PredictionInput:
    """Load and validate sample and reference tables against each other."""
    try:
        reference = read_reference_parquet(reference_path)
    except Exception as err:
        raise PredictError(f"failed to read reference {reference_path}: {err}") from err

    ref_meta = decode_metadata(reference.schema.metadata)
    ref_model = ref_meta.get("model", {})
    ref_bin_size = (
        int(ref_model.get("bin_size", 5000)) if isinstance(ref_model, Mapping) else 5000
    )
    try:
        sample = load_sample_or_extract(sample_path, fasta=fasta, bin_size=ref_bin_size)
    except Exception as err:
        raise PredictError(f"failed to read sample {sample_path}: {err}") from err

    sample_meta = decode_metadata(sample.schema.metadata)
    ref_model = ref_meta.get("model", {})
    ref_mode = (
        ref_model.get("mode", "germline")
        if isinstance(ref_model, Mapping)
        else "germline"
    )
    if mode == "prenatal" and ref_mode != "prenatal":
        msg = (
            "reference was built in germline mode and lacks prenatal models; "
            "rebuild reference with --mode prenatal or run predict with --mode germline"
        )
        raise PredictError(msg)
    if mode == "germline" and ref_mode == "prenatal":
        logger.warning(
            "reference was built in prenatal mode; "
            "prenatal models are ignored in germline mode"
        )
    sample_lens, ref_lens = _contig_lengths(sample_meta), _contig_lengths(ref_meta)
    branch_vals = set(reference.column("branch").unique().to_pylist())
    if BRANCH_AUTOSOMAL not in branch_vals:
        raise PredictError("reference has no autosomal branch")
    if mode == "prenatal":
        branches = [BRANCH_AUTOSOMAL] + (
            [BRANCH_FEMALE_SEX] if BRANCH_FEMALE_SEX in branch_vals else []
        )
        chosen_sex = "female" if BRANCH_FEMALE_SEX in branch_vals else None
    else:
        chosen_sex = _resolve_sample_sex(sex, sample, ref_meta, branch_vals)
        branches = [BRANCH_AUTOSOMAL]
        if chosen_sex is not None:
            sex_branch = (
                BRANCH_FEMALE_SEX if chosen_sex == "female" else BRANCH_MALE_SEX
            )
            if sex_branch not in branch_vals and sex != "autosomal":
                raise PredictError(f"reference has no {sex_branch!r} branch")
            if sex_branch in branch_vals:
                branches.append(sex_branch)
    reference_bin_size = (
        ref_model.get("bin_size") if isinstance(ref_model, dict) else None
    )
    sample_bin_size = sample_meta.get("bin_size")
    if (
        isinstance(reference_bin_size, int)
        and isinstance(sample_bin_size, int)
        and sample_bin_size != reference_bin_size
    ):
        logger.debug(
            "rebinning sample from %d bp to reference bin size %d bp",
            sample_bin_size,
            reference_bin_size,
        )
        sample = rebin_sample_table(sample, reference_bin_size)
    active_ref = reference.filter(
        pc.is_in(reference.column("branch"), pa.array(branches))
    )
    for contig_name in set(active_ref.column("contig").unique().to_pylist()):
        if sample_lens.get(contig_name) != ref_lens.get(contig_name):
            raise PredictError(f"contig length mismatch for {contig_name!r}")
    sample_active = sample.filter(
        pc.is_in(
            sample.column("contig"),
            pa.array(list(set(active_ref.column("contig").unique().to_pylist()))),
        )
    )
    if sample_active.num_rows != active_ref.num_rows and (
        missing := sorted(
            set(bin_coordinates(active_ref)) - set(bin_coordinates(sample))
        )
    ):
        contigs = ", ".join(dict.fromkeys(item[0] for item in missing))
        raise PredictError(f"sample is missing reference bins on contig(s): {contigs}")
    return PredictionInput(
        sample=sample,
        reference=reference,
        contig_lengths=ref_lens,
        sex=chosen_sex,
        branches=tuple(branches),
    )


def build_branch_model(inputs: PredictionInput, branch: str) -> BranchModel:
    """Build dense NumPy arrays for one reference branch."""
    ref_b = inputs.reference.filter(pc.equal(inputs.reference.column("branch"), branch))
    if ref_b.num_rows == 0:
        raise PredictError(f"branch {branch!r} has no reference rows")

    ref_pl = pl.from_arrow(ref_b)
    sm_pl = pl.from_arrow(inputs.sample)
    joined = ref_pl.select(["contig", "start", "end"]).join(
        sm_pl.select(["contig", "start", "end", "count"]),
        on=["contig", "start", "end"],
        how="left",
        maintain_order="left",
    )
    if joined["count"].null_count() != 0 or joined.height != ref_pl.height:
        raise PredictError(f"branch {branch!r} bin count mismatch")

    c_vals = ref_pl["contig"].to_list()
    starts = ref_pl["start"].to_numpy().astype(np.int64)
    ends = ref_pl["end"].to_numpy().astype(np.int64)
    counts = joined["count"].to_numpy().astype(np.float64)
    masked = ref_pl["masked"].to_numpy().astype(np.bool_)
    gc_col = ref_pl["gc"].to_numpy().astype(np.float32)
    map_col = ref_pl["mappability"].to_numpy().astype(np.float32)
    gc = gc_col if np.any(gc_col >= 0) else None
    map_ = map_col if np.any(map_col >= 0) else None
    unmasked = ~masked
    pca_m = ref_pl["pca_mean"].to_numpy().astype(np.float64)[unmasked]
    raw_ld = [
        item
        for item, keep in zip(
            ref_b.column("pca_loadings").to_pylist(), unmasked, strict=True
        )
        if keep
    ]
    pca_ld = (
        np.asarray(raw_ld, dtype=np.float64).T
        if raw_ld and raw_ld[0]
        else np.empty((0, int(np.count_nonzero(unmasked))), dtype=np.float64)
    )
    raw_nbrs = [
        item
        for item, keep in zip(
            ref_b.column("nearest_ref_indices").to_pylist(), unmasked, strict=True
        )
        if keep
    ]
    nearest_idx = tuple(np.asarray(nbrs, dtype=np.int32) for nbrs in raw_nbrs)
    mean_dist = ref_pl["mean_distance"].to_numpy().astype(np.float32)[unmasked]
    nearest_dst = tuple(
        np.full(nbrs.shape, dist, dtype=np.float32)
        for nbrs, dist in zip(nearest_idx, mean_dist, strict=True)
    )
    null_m = ref_pl["null_mean"].to_numpy().astype(np.float64)[unmasked]
    null_s = ref_pl["null_std"].to_numpy().astype(np.float64)[unmasked]
    return BranchModel(
        branch=branch,
        contigs=tuple(c_vals),
        starts=starts,
        ends=ends,
        counts=counts,
        masked=masked,
        pca_mean=pca_m,
        pca_loadings=pca_ld,
        nearest_indices=nearest_idx,
        mean_distance=mean_dist,
        null_mean=null_m,
        null_std=null_s,
        gc=gc,
        mappability=map_,
        nearest_distances=nearest_dst,
    )


def project_coverage(
    model: BranchModel,
    *,
    gc_override: NDArray[np.floating] | None = None,
    mappability_override: NDArray[np.floating] | None = None,
    n_mask_override: NDArray[np.bool_] | None = None,
    min_mappability: float = DEFAULT_MIN_MAPPABILITY,
) -> ProjectedCoverage:
    """Mask covariates, depth-normalize sample counts, and apply PCA projection."""
    gc = model.gc if gc_override is None else gc_override
    mappability = (
        model.mappability if mappability_override is None else mappability_override
    )
    unmasked_reference_bins = ~model.masked
    covariate_usable = mask_covariates(
        unmasked_reference_bins,
        gc=gc,
        mappability=mappability,
        n_mask=n_mask_override,
        min_mappability=min_mappability,
    )
    usable_total = float(np.sum(model.counts[covariate_usable]))
    if usable_total <= 0:
        raise PredictError(
            f"branch {model.branch!r} has zero sample coverage on usable bins"
        )

    active_unmasked = covariate_usable[unmasked_reference_bins]
    pos_mu = active_unmasked & (model.pca_mean > 0)
    scale = usable_total
    if np.count_nonzero(pos_mu) >= 2:
        ratios = model.counts[unmasked_reference_bins][pos_mu] / model.pca_mean[pos_mu]
        valid_ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
        if valid_ratios.size >= 2:
            s_hat = float(np.median(valid_ratios))
            if s_hat > 0 and np.isfinite(s_hat):
                scale = s_hat

    normalized = correct_covariate_bias(
        model.counts / scale, covariate_usable, gc=gc, mappability=mappability
    )
    selected = normalized[unmasked_reference_bins]
    if selected.shape != model.pca_mean.shape:
        raise PredictError("PCA mean does not align with unmasked sample bins")
    active_subset = covariate_usable[unmasked_reference_bins] & np.isfinite(selected)
    reconstruction = model.pca_mean.copy()
    n_active = int(np.count_nonzero(active_subset))
    rank = (
        int(np.linalg.matrix_rank(model.pca_loadings[:, active_subset].T))
        if model.pca_loadings.size > 0
        else 0
    )
    if n_active > rank > 0:
        subset_centered = selected[active_subset] - model.pca_mean[active_subset]
        X = model.pca_loadings[:, active_subset].T
        coeffs = None
        if n_active >= max(rank + 2, 4) and float(np.std(subset_centered)) > 1e-12:
            try:
                huber = HuberRegressor(
                    fit_intercept=False,
                    alpha=0.0,
                    epsilon=1.35,
                    max_iter=100,
                )
                huber.fit(X, subset_centered)
                coeffs = huber.coef_
            except ValueError:
                coeffs = None
        if coeffs is None:
            coeffs, *_ = np.linalg.lstsq(X, subset_centered, rcond=None)
        reconstruction = (coeffs @ model.pca_loadings) + model.pca_mean

    corrected = np.full(normalized.shape, np.nan, dtype=np.float64)
    valid = active_subset & np.isfinite(reconstruction) & (reconstruction > 0)
    corrected_values = np.full(selected.shape, np.nan, dtype=np.float64)
    corrected_values[valid] = selected[valid] / reconstruction[valid]
    corrected[unmasked_reference_bins] = corrected_values
    return ProjectedCoverage(
        normalized,
        corrected,
        covariate_usable & np.isfinite(corrected) & (corrected > 0),
    )


def normalize_branch(model: BranchModel, config: PredictConfig) -> BinResult:
    """Calculate within-sample log2 ratios, null z-scores, and segment weights."""
    logger.debug("normalizing branch %s (%d bins)", model.branch, model.counts.size)
    try:
        gc_override, n_mask_override = (
            load_gc_track(config.gc, model.contigs, model.starts, model.ends)
            if config.gc
            else (None, None)
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
    projected = project_coverage(
        model,
        gc_override=gc_override,
        mappability_override=mappability_override,
        n_mask_override=n_mask_override,
        min_mappability=config.min_mappability,
    )
    unmasked_indices = np.flatnonzero(~model.masked)
    U = unmasked_indices.size
    corrected = projected.corrected[unmasked_indices]
    ratios = np.full(model.counts.shape, np.nan)
    zscores = np.full(model.counts.shape, np.nan)
    weights = np.zeros(model.counts.shape)
    reference_sizes = np.zeros(model.counts.shape, dtype=np.int64)

    filtered_neighbors = tuple(
        neighbors[neighbors != i] for i, neighbors in enumerate(model.nearest_indices)
    )
    max_k = max((nbrs.size for nbrs in filtered_neighbors), default=0)

    if max_k == 0:
        neighbor_matrix = np.full((U, 0), U, dtype=np.int32)
    else:
        neighbor_matrix = np.full((U, max_k), U, dtype=np.int32)
        for i, nbrs in enumerate(filtered_neighbors):
            if nbrs.size > 0:
                neighbor_matrix[i, : nbrs.size] = nbrs

    eligible = np.isfinite(corrected) & (corrected > 0)
    local_ratios = np.full(corrected.shape, np.nan)
    local_zscores = np.full(corrected.shape, np.nan)
    CHUNK_SIZE = 25_000

    for iteration in range(config.mask_repeats):
        active_padded = np.full(U + 1, np.nan, dtype=np.float64)
        active_padded[:U] = np.where(eligible, corrected, np.nan)

        if max_k > 0:
            usable_nbr_counts = np.zeros(U, dtype=np.int64)
            baselines = np.full(U, np.nan, dtype=np.float64)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for start_idx in range(0, U, CHUNK_SIZE):
                    end_idx = min(start_idx + CHUNK_SIZE, U)
                    chunk_vals = active_padded[neighbor_matrix[start_idx:end_idx]]
                    usable_nbr_counts[start_idx:end_idx] = np.sum(
                        np.isfinite(chunk_vals), axis=1
                    )
                    baselines[start_idx:end_idx] = np.nanmedian(chunk_vals, axis=1)
        else:
            usable_nbr_counts = np.zeros(U, dtype=np.int64)
            baselines = np.full(U, np.nan, dtype=np.float64)

        reference_sizes[unmasked_indices] = usable_nbr_counts

        valid_bins = (
            np.isfinite(corrected)
            & (corrected > 0)
            & (usable_nbr_counts >= config.min_reference_bins)
            & np.isfinite(baselines)
            & (baselines > 0)
        )
        local_ratios[:] = np.nan
        local_zscores[:] = np.nan

        local_ratios[valid_bins] = np.log2(
            corrected[valid_bins] / baselines[valid_bins]
        )

        has_std = valid_bins & (model.null_std > 1e-8)
        local_zscores[has_std] = (
            local_ratios[has_std] - model.null_mean[has_std]
        ) / model.null_std[has_std]

        missing_zscores = np.isfinite(local_ratios) & ~np.isfinite(local_zscores)
        if np.any(missing_zscores):
            finite_ratios_sub = local_ratios[np.isfinite(local_ratios)]
            median_ratio = float(np.median(finite_ratios_sub))
            empirical_std = 1.4826 * float(
                scipy.stats.median_abs_deviation(finite_ratios_sub, scale=1.0)
            )
            if empirical_std <= 1e-8:
                empirical_std = float(np.std(finite_ratios_sub))
            local_zscores[missing_zscores] = (
                (local_ratios[missing_zscores] - median_ratio) / empirical_std
                if empirical_std > 1e-8
                else 0.0
            )
        eligible = (
            np.isfinite(corrected)
            & np.isfinite(local_zscores)
            & (np.abs(local_zscores) < _MASK_Z_THRESHOLD)
        )
        logger.debug(
            "branch %s within-sample iteration %d/%d: %d eligible bins",
            model.branch,
            iteration + 1,
            config.mask_repeats,
            int(np.count_nonzero(eligible)),
        )

    finite_ratios = np.isfinite(local_ratios)
    median_shift = 0.0
    if np.any(finite_ratios):
        median_shift = float(np.median(local_ratios[finite_ratios]))
        local_ratios[finite_ratios] -= median_shift

    finite_zscores = np.isfinite(local_zscores)
    if np.any(finite_zscores):
        zscore_shift = float(np.median(local_zscores[finite_zscores]))
        local_zscores[finite_zscores] -= zscore_shift

    logger.debug(
        "branch %s: %d finite ratios, median shift=%.4f (mean=%.4f, std=%.4f)",
        model.branch,
        int(np.count_nonzero(finite_ratios)),
        median_shift,
        float(np.mean(local_ratios[finite_ratios])) if np.any(finite_ratios) else 0.0,
        float(np.std(local_ratios[finite_ratios])) if np.any(finite_ratios) else 0.0,
    )
    ratios[unmasked_indices] = local_ratios
    zscores[unmasked_indices] = local_zscores

    pos_null = model.null_std > 0
    s0_sq = float(np.median(model.null_std[pos_null] ** 2)) if np.any(pos_null) else 1.0
    eb_weights = np.clip(s0_sq / np.maximum(model.null_std**2, 1e-8), 0.05, 20.0)
    tier2_weights = np.where(
        model.mean_distance > 0,
        1.0 / np.maximum(model.mean_distance, 1e-12),
        1.0,
    )
    weights[unmasked_indices] = np.where(pos_null, eb_weights, tier2_weights)

    valid = (
        projected.usable
        & np.isfinite(ratios)
        & np.isfinite(zscores)
        & (reference_sizes >= config.min_reference_bins)
    )
    ratios[~valid], zscores[~valid], weights[~valid] = np.nan, np.nan, 0.0
    _require_aligned(model.counts.size, ratios, zscores, weights, reference_sizes)
    return BinResult(
        branch=model.branch,
        contigs=model.contigs,
        starts=model.starts,
        ends=model.ends,
        ratios=ratios,
        zscores=zscores,
        weights=weights,
        reference_sizes=reference_sizes,
        masked=~valid,
    )


def call_segment(
    ratio: float,
    zscore: float,
    *,
    z_threshold: float,
    ratio_threshold: float | None = None,
) -> CallType | None:
    """Classify a segment using a symmetric ratio or z-score threshold."""
    val, thresh = (
        (ratio, ratio_threshold)
        if ratio_threshold is not None
        else (zscore, z_threshold)
    )
    if not np.isfinite(val):
        return None
    if val > thresh:
        return "gain"
    if val < -thresh:
        return "loss"
    return None


def _compute_contig_rho1(z_contig: NDArray[np.float64]) -> float:
    """Compute per-contig lag-1 autocorrelation on finite diploid-clipped z-scores."""
    mask = np.isfinite(z_contig) & (np.abs(z_contig) <= 2.5)
    z_clip = z_contig[mask]
    if z_clip.size < 3:
        return 0.0
    x = z_clip[:-1]
    y = z_clip[1:]
    std_x = float(np.std(x))
    std_y = float(np.std(y))
    if std_x < 1e-12 or std_y < 1e-12:
        return 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return float(np.clip(corr, 0.0, 0.95))


def _evaluate_stouffer_segment_scores(
    bins_df: pl.DataFrame,
    segs_df: pl.DataFrame,
) -> pl.DataFrame:
    """Evaluate autocorrelation-corrected Stouffer Z-scores via asof interval join."""
    bins_sorted = bins_df.sort(["contig", "start"])
    segs_sorted = segs_df.sort(["contig", "seg_start"])

    joined = bins_sorted.join_asof(
        segs_sorted.select(["seg_id", "contig", "seg_start", "seg_end"]),
        left_on="start",
        right_on="seg_start",
        by="contig",
        strategy="backward",
        check_sortedness=False,
    ).filter(pl.col("end") <= pl.col("seg_end"))

    usable = joined.filter(pl.col("zscore").is_finite() & (pl.col("weight") > 0))

    if usable.height == 0:
        return segs_df.with_columns(
            [
                pl.lit(0).alias("n_usable"),
                pl.lit(0.0).alias("zscore"),
            ]
        ).sort("seg_id")

    stats = (
        usable.group_by("seg_id")
        .agg(
            [
                pl.len().alias("n_usable"),
                (pl.col("weight") * pl.col("zscore")).sum().alias("sum_w_z"),
                (pl.col("weight") * pl.col("weight")).sum().alias("sum_w_sq"),
                pl.col("rho1").first().alias("rho1"),
            ]
        )
        .with_columns(
            (
                pl.col("sum_w_sq")
                * (
                    1.0
                    + pl.when(pl.col("n_usable") > 1)
                    .then(
                        2.0
                        * (pl.col("n_usable") - 1)
                        / pl.col("n_usable")
                        * pl.col("rho1")
                    )
                    .otherwise(0.0)
                )
            )
            .sqrt()
            .alias("denom")
        )
        .with_columns(
            pl.when(pl.col("denom") > 1e-12)
            .then(pl.col("sum_w_z") / pl.col("denom"))
            .otherwise(0.0)
            .alias("zscore")
        )
    )
    return (
        segs_df.join(
            stats.select(["seg_id", "n_usable", "zscore"]),
            on="seg_id",
            how="left",
        )
        .with_columns(
            [
                pl.col("n_usable").fill_null(0),
                pl.col("zscore").fill_null(0.0),
            ]
        )
        .sort("seg_id")
    )


def segment_and_call(
    bins: tuple[BinResult, ...],
    contig_lengths: dict[str, int],
    *,
    penalty: float,
    z_threshold: float,
    ratio_threshold: float | None = None,
    max_blacklist_gap: int = 2_000_000,
) -> tuple[SegmentResult, ...]:
    """Segment all selected branches and classify with AR(1)-corrected Stouffer Z-scores."""
    if not bins:
        return ()

    by_contig: dict[
        str,
        tuple[
            NDArray[np.int64],
            NDArray[np.int64],
            NDArray[np.float64],
            NDArray[np.float64],
            int,
        ],
    ] = {}

    all_contigs: list[str] = []
    all_starts_list: list[NDArray[np.int64]] = []
    all_ends_list: list[NDArray[np.int64]] = []
    all_zscores_list: list[NDArray[np.float64]] = []
    all_weights_list: list[NDArray[np.float64]] = []
    all_rho1_list: list[NDArray[np.float64]] = []

    for result in bins:
        contig_array = np.asarray(result.contigs, dtype=object)
        for contig_name in dict.fromkeys(result.contigs):
            selected = contig_array == contig_name
            c_starts = result.starts[selected]
            c_ends = result.ends[selected]
            c_ratios = result.ratios[selected]
            c_weights = result.weights[selected]
            c_zscores = result.zscores[selected]

            by_contig[contig_name] = (
                c_starts,
                c_ends,
                c_ratios,
                c_weights,
                contig_lengths[contig_name],
            )
            rho1 = _compute_contig_rho1(c_zscores)

            all_contigs.extend([contig_name] * int(np.count_nonzero(selected)))
            all_starts_list.append(c_starts)
            all_ends_list.append(c_ends)
            all_zscores_list.append(c_zscores)
            all_weights_list.append(c_weights)
            all_rho1_list.append(np.full(c_starts.size, rho1, dtype=np.float64))

    raw_segments = segment_genome(
        by_contig,
        penalty=penalty,
        algorithm="binseg",
        max_blacklist_gap=max_blacklist_gap,
    )
    if not raw_segments:
        logger.debug("called 0 segments: 0 gains, 0 losses, 0 neutral")
        return ()

    all_starts = (
        np.concatenate(all_starts_list)
        if all_starts_list
        else np.empty(0, dtype=np.int64)
    )
    all_ends = (
        np.concatenate(all_ends_list) if all_ends_list else np.empty(0, dtype=np.int64)
    )
    all_zscores = (
        np.concatenate(all_zscores_list)
        if all_zscores_list
        else np.empty(0, dtype=np.float64)
    )
    all_weights = (
        np.concatenate(all_weights_list)
        if all_weights_list
        else np.empty(0, dtype=np.float64)
    )
    all_rho1 = (
        np.concatenate(all_rho1_list)
        if all_rho1_list
        else np.empty(0, dtype=np.float64)
    )

    bins_df = pl.DataFrame(
        {
            "contig": all_contigs,
            "start": all_starts,
            "end": all_ends,
            "zscore": all_zscores,
            "weight": all_weights,
            "rho1": all_rho1,
        }
    )

    segs_df = pl.DataFrame(
        {
            "seg_id": np.arange(len(raw_segments), dtype=np.int64),
            "contig": [s.contig for s in raw_segments],
            "seg_start": np.array([s.start for s in raw_segments], dtype=np.int64),
            "seg_end": np.array([s.end for s in raw_segments], dtype=np.int64),
            "mean": np.array([s.mean for s in raw_segments], dtype=np.float64),
            "weight": np.array([s.weight for s in raw_segments], dtype=np.float64),
            "bin_count": np.array([s.bin_count for s in raw_segments], dtype=np.int64),
        }
    )

    final_segs = _evaluate_stouffer_segment_scores(bins_df, segs_df)

    output: list[SegmentResult] = []
    for row in final_segs.iter_rows(named=True):
        contig = row["contig"]
        start = row["seg_start"]
        end = min(row["seg_end"], contig_lengths[contig])
        ratio = row["mean"]
        seg_zscore = row["zscore"]
        weight = row["weight"]
        bin_count = row["bin_count"]
        n_usable = row["n_usable"]
        call = (
            call_segment(
                ratio,
                seg_zscore,
                z_threshold=z_threshold,
                ratio_threshold=ratio_threshold,
            )
            if n_usable > 0
            else None
        )
        output.append(
            SegmentResult(
                contig=contig,
                start=start,
                end=end,
                ratio=ratio,
                zscore=seg_zscore,
                weight=weight,
                bin_count=bin_count,
                call=call,
            )
        )

    gains = sum(1 for s in output if s.call == "gain")
    losses = sum(1 for s in output if s.call == "loss")
    neutral = sum(1 for s in output if s.call is None)
    logger.debug(
        "called %d segments: %d gains, %d losses, %d neutral",
        len(output),
        gains,
        losses,
        neutral,
    )
    return tuple(output)


def _resolve_sample_sex(
    requested: Literal["autosomal", "female", "male"],
    sample: pa.Table,
    metadata: dict[str, object],
    branches: set[object],
) -> Sex | None:
    """Infer sample genetic sex from sex-signal contig coverage against reference cutoff."""
    if requested != "autosomal":
        return requested
    if not ({BRANCH_FEMALE_SEX, BRANCH_MALE_SEX} & branches):
        return None
    model = metadata.get("model")
    if (
        not isinstance(model, dict)
        or not isinstance(model.get("sex_inference"), dict)
        or not isinstance(model.get("sex_signal_contig"), str)
    ):
        raise PredictError("cannot infer sample sex; pass --sex female or --sex male")
    cutoff = model["sex_inference"].get("cutoff")
    if not isinstance(cutoff, int | float):
        raise PredictError("cannot infer sample sex; pass --sex female or --sex male")
    contigs, counts = (
        np.asarray(sample.column("contig").to_pylist(), dtype=object),
        np.asarray(
            sample.column("count").to_numpy(zero_copy_only=False), dtype=np.float64
        ),
    )
    total = float(np.sum(counts))
    if total <= 0:
        raise PredictError("sample has zero total coverage")
    signal = str(model["sex_signal_contig"])
    fraction = float(np.sum(counts[contigs == signal]) / total)
    sex = "male" if fraction > float(cutoff) else "female"
    logger.debug(
        "inferred sample sex as %s (signal %s fraction=%.4f vs cutoff=%.4f)",
        sex,
        signal,
        fraction,
        float(cutoff),
    )
    return sex


def _contig_lengths(metadata: dict[str, object]) -> dict[str, int]:
    """Extract contig lengths dictionary from reference metadata."""
    val = metadata.get("contig_lengths")
    if not isinstance(val, dict):
        raise PredictError("artifact metadata has no contig lengths")
    return {str(c): int(length) for c, length in val.items()}


def _validate_config(config: PredictConfig) -> None:
    """Validate prediction configuration parameters."""
    if (
        not np.isfinite(config.penalty)
        or config.penalty <= 0
        or not np.isfinite(config.z_threshold)
        or config.z_threshold <= 0
    ):
        raise PredictError("penalty and z_threshold must be finite and positive")
    if config.ratio_threshold is not None and (
        not np.isfinite(config.ratio_threshold) or config.ratio_threshold <= 0
    ):
        raise PredictError("ratio_threshold must be finite and positive")
    if (
        config.min_reference_bins < 1
        or config.mask_repeats < 1
        or config.max_blacklist_gap < 0
    ):
        msg = (
            "min_reference_bins and mask_repeats must be positive, "
            "max_blacklist_gap non-negative"
        )
        raise PredictError(msg)
    try:
        validate_min_mappability(config.min_mappability)
    except CovariateError as err:
        raise PredictError(str(err)) from err


def _require_aligned(length: int, *arrays: ArrayLike) -> None:
    """Ensure all provided arrays have 1D shape matching expected length."""
    if any(np.asarray(a).shape != (length,) for a in arrays):
        raise PredictError(
            "prediction result arrays must be aligned and one-dimensional"
        )


run_predict: Final = run
