"""Typed copy-number normalization, segmentation, and calling."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pyarrow as pa
import scipy.stats
from numpy.typing import ArrayLike, NDArray

from ccnv.features.cohort import rebin_sample_table
from ccnv.features.covariates import (
    correct_covariate_bias,
    mask_covariates,
    validate_min_mappability,
)
from ccnv.features.prenatal import annotate_prenatal_segments, predict_prenatal
from ccnv.io.parquet import read_reference_parquet, read_sample_parquet
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
        config.sample, config.reference, sex=config.sex, mode=config.mode
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
    prenatal_result: FetalFractionResult | None = None
    if config.mode == "prenatal":
        metadata = PrenatalModelMetadata.from_table(inputs.reference)
        if metadata is None:
            raise PredictError("reference is missing prenatal metadata (ccnv.prenatal)")
        prenatal_result = predict_prenatal(
            inputs.sample,
            inputs.reference,
            metadata,
            min_ff_warning=config.min_fetal_fraction_warning,
        )
        segments = tuple(
            annotate_prenatal_segments(
                segments, prenatal_result, z_threshold=config.z_threshold
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
) -> PredictionInput:
    """Load and validate sample and reference tables against each other."""
    try:
        sample = read_sample_parquet(sample_path)
    except Exception as err:
        raise PredictError(f"failed to read sample {sample_path}: {err}") from err
    try:
        reference = read_reference_parquet(reference_path)
    except Exception as err:
        raise PredictError(f"failed to read reference {reference_path}: {err}") from err

    sample_meta, ref_meta = (
        decode_metadata(sample.schema.metadata),
        decode_metadata(reference.schema.metadata),
    )
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
        pa.compute.is_in(reference.column("branch"), pa.array(branches))
    )
    for contig_name in set(active_ref.column("contig").unique().to_pylist()):
        if sample_lens.get(contig_name) != ref_lens.get(contig_name):
            raise PredictError(f"contig length mismatch for {contig_name!r}")
    sample_active = sample.filter(
        pa.compute.is_in(
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
    ref_b = inputs.reference.filter(
        pa.compute.equal(inputs.reference.column("branch"), branch)
    )
    if ref_b.num_rows == 0:
        raise PredictError(f"branch {branch!r} has no reference rows")
    ref_coords = set(bin_coordinates(ref_b))
    sm_b = inputs.sample.filter(
        pa.compute.is_in(
            inputs.sample.column("contig"),
            pa.array(list(set(ref_b.column("contig").unique().to_pylist()))),
        )
    )
    c_m = [
        item in ref_coords
        for item in zip(
            sm_b.column("contig").to_pylist(),
            sm_b.column("start").to_pylist(),
            sm_b.column("end").to_pylist(),
            strict=True,
        )
    ]
    sample_branch = sm_b.filter(pa.array(c_m, type=pa.bool_()))
    if sample_branch.num_rows != ref_b.num_rows:
        raise PredictError(f"branch {branch!r} bin count mismatch")

    c_vals = ref_b.column("contig").to_pylist()
    starts = np.asarray(
        ref_b.column("start").to_numpy(zero_copy_only=False), dtype=np.int64
    )
    ends = np.asarray(
        ref_b.column("end").to_numpy(zero_copy_only=False), dtype=np.int64
    )
    counts = np.asarray(
        sample_branch.column("count").to_numpy(zero_copy_only=False), dtype=np.float64
    )
    masked = np.asarray(
        ref_b.column("masked").to_numpy(zero_copy_only=False), dtype=np.bool_
    )
    gc_col = np.asarray(
        ref_b.column("gc").to_numpy(zero_copy_only=False), dtype=np.float32
    )
    map_col = np.asarray(
        ref_b.column("mappability").to_numpy(zero_copy_only=False), dtype=np.float32
    )
    gc = gc_col if np.any(gc_col >= 0) else None
    map_ = map_col if np.any(map_col >= 0) else None
    unmasked = ~masked
    pca_m = np.asarray(
        ref_b.column("pca_mean").to_numpy(zero_copy_only=False), dtype=np.float64
    )[unmasked]
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
        else np.empty((0, np.count_nonzero(unmasked)), dtype=np.float64)
    )
    raw_nbrs = [
        item
        for item, keep in zip(
            ref_b.column("nearest_ref_indices").to_pylist(), unmasked, strict=True
        )
        if keep
    ]
    nearest_idx = tuple(np.asarray(nbrs, dtype=np.int32) for nbrs in raw_nbrs)
    mean_dist = np.asarray(
        ref_b.column("mean_distance").to_numpy(zero_copy_only=False), dtype=np.float32
    )[unmasked]
    nearest_dst = tuple(
        np.full(nbrs.shape, dist, dtype=np.float32)
        for nbrs, dist in zip(nearest_idx, mean_dist, strict=True)
    )
    null_m = np.asarray(
        ref_b.column("null_mean").to_numpy(zero_copy_only=False), dtype=np.float64
    )[unmasked]
    null_s = np.asarray(
        ref_b.column("null_std").to_numpy(zero_copy_only=False), dtype=np.float64
    )[unmasked]
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
    normalized = correct_covariate_bias(
        model.counts / usable_total, covariate_usable, gc=gc, mappability=mappability
    )
    selected = normalized[unmasked_reference_bins]
    if selected.shape != model.pca_mean.shape:
        raise PredictError("PCA mean does not align with unmasked sample bins")
    active_subset = covariate_usable[unmasked_reference_bins] & np.isfinite(selected)
    reconstruction = model.pca_mean.copy()
    if np.count_nonzero(active_subset) > int(
        np.linalg.matrix_rank(model.pca_loadings[:, active_subset].T)
    ):
        subset_centered = selected[active_subset] - model.pca_mean[active_subset]
        coeffs, *_ = np.linalg.lstsq(
            model.pca_loadings[:, active_subset].T, subset_centered, rcond=None
        )
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
    corrected = projected.corrected[unmasked_indices]
    ratios = np.full(model.counts.shape, np.nan)
    zscores = np.full(model.counts.shape, np.nan)
    weights = np.zeros(model.counts.shape)
    reference_sizes = np.zeros(model.counts.shape, dtype=np.int64)
    eligible = np.isfinite(corrected) & (corrected > 0)
    local_ratios = np.full(corrected.shape, np.nan)
    local_zscores = np.full(corrected.shape, np.nan)
    filtered_neighbors = tuple(
        neighbors[neighbors != i] for i, neighbors in enumerate(model.nearest_indices)
    )
    for iteration in range(config.mask_repeats):
        for idx in range(unmasked_indices.size):
            val, neighbors = corrected[idx], filtered_neighbors[idx]
            selected_neighbors = neighbors[eligible[neighbors]]
            reference_sizes[unmasked_indices[idx]] = selected_neighbors.size
            if (
                not np.isfinite(val)
                or val <= 0
                or selected_neighbors.size < config.min_reference_bins
            ):
                local_ratios[idx], local_zscores[idx] = np.nan, np.nan
                continue
            baseline = float(np.median(corrected[selected_neighbors]))
            if not np.isfinite(baseline) or baseline <= 0:
                continue
            ratio_val = float(np.log2(val / baseline))
            local_ratios[idx] = ratio_val
            null_std = model.null_std[idx]
            local_zscores[idx] = (
                (ratio_val - model.null_mean[idx]) / null_std
                if null_std > 0
                else np.nan
            )
        missing_zscores = np.isfinite(local_ratios) & ~np.isfinite(local_zscores)
        if np.any(missing_zscores):
            finite_ratios_sub = local_ratios[np.isfinite(local_ratios)]
            median_ratio = float(np.median(finite_ratios_sub))
            empirical_std = 1.4826 * float(
                scipy.stats.median_abs_deviation(finite_ratios_sub, scale=1.0)
            )
            local_zscores[missing_zscores] = (
                (local_ratios[missing_zscores] - median_ratio) / empirical_std
                if empirical_std > 1e-12
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
    valid_distances = (model.mean_distance > 0) & np.isfinite(model.mean_distance)
    weights[unmasked_indices] = np.where(
        valid_distances,
        1.0 / np.where(valid_distances, model.mean_distance, 1.0),
        1.0,
    )
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


def segment_and_call(
    bins: tuple[BinResult, ...],
    contig_lengths: dict[str, int],
    *,
    penalty: float,
    z_threshold: float,
    ratio_threshold: float | None = None,
    max_blacklist_gap: int = 2_000_000,
) -> tuple[SegmentResult, ...]:
    """Segment all selected branches and classify their weighted means."""
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
    evidence: dict[
        str,
        tuple[
            NDArray[np.int64],
            NDArray[np.int64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ] = {}
    for result in bins:
        contig_array = np.asarray(result.contigs, dtype=object)
        for contig_name in dict.fromkeys(result.contigs):
            selected = contig_array == contig_name
            contig_starts = result.starts[selected]
            contig_ends = result.ends[selected]
            contig_weights = result.weights[selected]
            by_contig[contig_name] = (
                contig_starts,
                contig_ends,
                result.ratios[selected],
                contig_weights,
                contig_lengths[contig_name],
            )
            evidence[contig_name] = (
                contig_starts,
                contig_ends,
                result.zscores[selected],
                contig_weights,
            )
    raw_segments = segment_genome(
        by_contig,
        penalty=penalty,
        algorithm="binseg",
        max_blacklist_gap=max_blacklist_gap,
    )
    output: list[SegmentResult] = []
    for segment in raw_segments:
        contig_starts, contig_ends, contig_zscores, contig_weights = evidence[
            segment.contig
        ]
        overlapping = (
            (contig_starts < segment.end)
            & (contig_ends > segment.start)
            & np.isfinite(contig_zscores)
            & (contig_weights > 0)
        )
        segment_zscore = (
            float(
                np.average(
                    contig_zscores[overlapping], weights=contig_weights[overlapping]
                )
            )
            if np.any(overlapping)
            else np.nan
        )
        output.append(
            SegmentResult(
                contig=segment.contig,
                start=segment.start,
                end=min(segment.end, contig_lengths[segment.contig]),
                ratio=segment.mean,
                zscore=segment_zscore,
                weight=segment.weight,
                bin_count=segment.bin_count,
                call=call_segment(
                    segment.mean,
                    segment_zscore,
                    z_threshold=z_threshold,
                    ratio_threshold=ratio_threshold,
                ),
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
    val = metadata.get("contig_lengths")
    if not isinstance(val, dict):
        raise PredictError("artifact metadata has no contig lengths")
    return {str(c): int(length) for c, length in val.items()}


def _validate_config(config: PredictConfig) -> None:
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
    if any(np.asarray(a).shape != (length,) for a in arrays):
        raise PredictError(
            "prediction result arrays must be aligned and one-dimensional"
        )


run_predict: Final = run
