"""Typed copy-number normalization, segmentation, and calling."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import ArrayLike, NDArray

from ccnv.convert import read_sample_parquet, rebin_sample_table
from ccnv.models import PredictConfig
from ccnv.reference import (
    BRANCH_AUTOSOMAL,
    BRANCH_FEMALE_SEX,
    BRANCH_MALE_SEX,
)
from ccnv.report import write_outputs
from ccnv.results import BinResult, CallType, PredictionOutputs, SegmentResult
from ccnv.schemas import (
    ArtifactKind,
    SchemaError,
    bin_coordinates,
    decode_metadata,
    validate_table,
)
from ccnv.segment import segment_genome

__all__ = (
    "BranchModel",
    "PredictError",
    "PredictionInput",
    "ProjectedCoverage",
    "call_segment",
    "load_inputs",
    "normalize_branch",
    "run",
    "segment_and_call",
)

logger = logging.getLogger(__name__)

Sex = Literal["female", "male"]
_MASK_Z_THRESHOLD: Final[float] = 2.326347874


class PredictError(ValueError):
    """Raised when prediction inputs or intermediate values are invalid."""


@dataclass(frozen=True, slots=True)
class PredictionInput:
    """Validated sample and reference artifacts."""

    sample: pa.Table
    reference: pa.Table
    contig_lengths: dict[str, int]
    sex: Sex | None
    branches: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BranchModel:
    """Dense arrays for one selected reference branch."""

    branch: str
    contigs: tuple[str, ...]
    starts: NDArray[np.int64]
    ends: NDArray[np.int64]
    counts: NDArray[np.float64]
    masked: NDArray[np.bool_]
    pca_mean: NDArray[np.float64]
    pca_loadings: NDArray[np.float64]
    nearest_indices: tuple[NDArray[np.int64], ...]
    mean_distance: NDArray[np.float32]
    null_mean: NDArray[np.float64]
    null_std: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ProjectedCoverage:
    """Coverage after depth normalization, masking, and PCA projection."""

    normalized: NDArray[np.float64]
    corrected: NDArray[np.float64]
    usable: NDArray[np.bool_]


def run(config: PredictConfig) -> PredictionOutputs:
    """Run prediction and write bin, segment, call, and optional plot artifacts."""

    _validate_config(config)
    logger.debug("loading sample %s and reference %s", config.sample, config.reference)
    inputs = load_inputs(config.sample, config.reference, sex=config.sex)
    logger.debug("using branches %s (sex=%s)", inputs.branches, inputs.sex)
    bin_results = tuple(
        normalize_branch(build_branch_model(inputs, branch), config)
        for branch in inputs.branches
    )
    logger.debug(
        "segmenting %d branch result(s) with penalty=%s",
        len(bin_results),
        config.penalty,
    )
    segments = segment_and_call(
        bin_results,
        inputs.contig_lengths,
        penalty=config.penalty,
        z_threshold=config.z_threshold,
        ratio_threshold=config.ratio_threshold,
        max_blacklist_gap=config.max_blacklist_gap,
    )
    return write_outputs(
        config.prefix,
        bin_results,
        segments,
        inputs.contig_lengths,
        config.plot,
    )


def load_inputs(
    sample_path: Path,
    reference_path: Path,
    *,
    sex: Literal["autosomal", "female", "male"] = "autosomal",
) -> PredictionInput:
    """Read artifacts through their contracts and select model branches."""

    try:
        sample = read_sample_parquet(sample_path)
    except (FileNotFoundError, OSError, pa.ArrowInvalid, SchemaError) as error:
        raise PredictError(
            f"{sample_path}: cannot read sample Parquet ({error})"
        ) from error
    try:
        reference = pq.read_table(reference_path)
        validate_table(reference, ArtifactKind.REFERENCE)
    except (FileNotFoundError, OSError, pa.ArrowInvalid, SchemaError) as error:
        raise PredictError(
            f"{reference_path}: cannot read reference Parquet ({error})"
        ) from error

    sample_meta = decode_metadata(sample.schema.metadata)
    reference_meta = decode_metadata(reference.schema.metadata)
    sample_lengths = _contig_lengths(sample_meta)
    reference_lengths = _contig_lengths(reference_meta)
    branch_values = set(reference.column("branch").unique().to_pylist())
    chosen_sex = _resolve_sample_sex(sex, sample, reference_meta, branch_values)
    branches = [BRANCH_AUTOSOMAL]
    if BRANCH_AUTOSOMAL not in branch_values:
        raise PredictError("reference has no autosomal branch")
    if chosen_sex is not None:
        sex_branch = BRANCH_FEMALE_SEX if chosen_sex == "female" else BRANCH_MALE_SEX
        if sex_branch not in branch_values:
            if sex != "autosomal":
                raise PredictError(f"reference has no {sex_branch!r} branch")
        else:
            branches.append(sex_branch)

    ref_model = reference_meta.get("model")
    ref_bin_size = ref_model.get("bin_size") if isinstance(ref_model, dict) else None
    sample_bin_size = sample_meta.get("bin_size")
    if (
        isinstance(ref_bin_size, int)
        and isinstance(sample_bin_size, int)
        and sample_bin_size != ref_bin_size
    ):
        logger.debug(
            "rebinning sample from %d bp to reference bin size %d bp",
            sample_bin_size,
            ref_bin_size,
        )
        sample = rebin_sample_table(sample, ref_bin_size)

    ref_branch_arr = np.asarray(
        reference.column("branch").to_numpy(zero_copy_only=False)
    )
    branch_mask = np.isin(ref_branch_arr, branches)
    active_ref = reference.filter(pa.array(branch_mask))

    ref_active_contigs = set(active_ref.column("contig").unique().to_pylist())
    for contig in ref_active_contigs:
        if sample_lengths.get(contig) != reference_lengths.get(contig):
            raise PredictError(f"contig length mismatch for {contig!r}")

    sample_contig_arr = np.asarray(
        sample.column("contig").to_numpy(zero_copy_only=False)
    )
    sample_active_mask = np.isin(sample_contig_arr, list(ref_active_contigs))
    sample_active = sample.filter(pa.array(sample_active_mask))
    if sample_active.num_rows != active_ref.num_rows:
        sample_bins = set(bin_coordinates(sample))
        required = set(bin_coordinates(active_ref))
        missing = sorted(required - sample_bins)
        if missing:
            contigs = ", ".join(dict.fromkeys(item[0] for item in missing))
            raise PredictError(
                f"sample is missing reference bins on contig(s): {contigs}"
            )

    return PredictionInput(
        sample=sample,
        reference=reference,
        contig_lengths=reference_lengths,
        sex=chosen_sex,
        branches=tuple(branches),
    )


def build_branch_model(inputs: PredictionInput, branch: str) -> BranchModel:
    """Align sample counts to a branch and materialize its nested model arrays."""

    branch_col = inputs.reference.column("branch")
    branch_arr = np.asarray(branch_col.to_numpy(zero_copy_only=False))
    row_indexes = np.flatnonzero(branch_arr == branch)
    if row_indexes.size == 0:
        raise PredictError(f"reference branch {branch!r} is empty")

    start_row = int(row_indexes[0])
    num_rows = int(row_indexes.size)
    if int(row_indexes[-1]) == start_row + num_rows - 1:
        branch_table = inputs.reference.slice(start_row, num_rows)
    else:
        branch_table = inputs.reference.take(pa.array(row_indexes))

    branch_contigs = tuple(branch_table.column("contig").to_pylist())
    branch_starts = branch_table.column("start").to_numpy(zero_copy_only=False)
    branch_ends = branch_table.column("end").to_numpy(zero_copy_only=False)

    sample_contig_arr = np.asarray(
        inputs.sample.column("contig").to_numpy(zero_copy_only=False)
    )
    branch_contig_set = set(branch_contigs)
    sample_branch_mask = np.isin(sample_contig_arr, list(branch_contig_set))
    sample_branch_table = inputs.sample.filter(pa.array(sample_branch_mask))
    if (
        sample_branch_table.num_rows == branch_table.num_rows
        and sample_branch_table.column("start").equals(branch_table.column("start"))
        and sample_branch_table.column("end").equals(branch_table.column("end"))
        and sample_branch_table.column("contig").equals(branch_table.column("contig"))
    ):
        counts = (
            sample_branch_table.column("count")
            .to_numpy(zero_copy_only=False)
            .astype(np.float64)
        )
    else:
        sample_counts = {
            coordinate: float(count)
            for coordinate, count in zip(
                bin_coordinates(inputs.sample),
                inputs.sample.column("count").to_pylist(),
                strict=True,
            )
        }
        coordinates = bin_coordinates(branch_table)
        counts = np.asarray(
            [sample_counts[item] for item in coordinates], dtype=np.float64
        )

    masked = (
        branch_table.column("masked").to_numpy(zero_copy_only=False).astype(np.bool_)
    )
    unmasked_rows = np.flatnonzero(~masked)
    unmasked_count = int(unmasked_rows.size)

    pca_mean_col = branch_table.column("pca_mean").to_numpy(zero_copy_only=False)
    pca_mean = np.nan_to_num(pca_mean_col[unmasked_rows], nan=0.0).astype(np.float64)

    loading_chunk = branch_table.column("pca_loadings").combine_chunks()
    loading_offsets = loading_chunk.offsets.to_numpy()
    comp_sizes = loading_offsets[unmasked_rows + 1] - loading_offsets[unmasked_rows]
    unique_comp_sizes = np.unique(comp_sizes)
    if len(unique_comp_sizes) != 1:
        raise PredictError(f"branch {branch!r} has inconsistent PCA loading shapes")
    n_components = int(unique_comp_sizes[0])
    loadings_flat = loading_chunk.values.to_numpy()
    if unmasked_count == branch_table.num_rows:
        loadings = loadings_flat.reshape(unmasked_count, n_components).T.astype(
            np.float64
        )
    else:
        loadings = np.empty((n_components, unmasked_count), dtype=np.float64)
        for i, row in enumerate(unmasked_rows):
            start = loading_offsets[row]
            loadings[:, i] = loadings_flat[start : start + n_components]

    nearest_chunk = branch_table.column("nearest_ref_indices").combine_chunks()
    nearest_offsets = nearest_chunk.offsets.to_numpy()
    nearest_flat = nearest_chunk.values.to_numpy()
    nearest = tuple(
        nearest_flat[nearest_offsets[row] : nearest_offsets[row + 1]].astype(np.int64)
        for row in unmasked_rows
    )

    mean_distance = branch_table.column("mean_distance").to_numpy(zero_copy_only=False)[
        unmasked_rows
    ]
    null_mean = branch_table.column("null_mean").to_numpy(zero_copy_only=False)[
        unmasked_rows
    ]
    null_std = branch_table.column("null_std").to_numpy(zero_copy_only=False)[
        unmasked_rows
    ]
    for indexes in nearest:
        if np.any(indexes < 0) or np.any(indexes >= unmasked_count):
            raise PredictError(f"branch {branch!r} has invalid nearest references")

    return BranchModel(
        branch=branch,
        contigs=branch_contigs,
        starts=branch_starts,
        ends=branch_ends,
        counts=counts,
        masked=masked,
        pca_mean=pca_mean,
        pca_loadings=loadings,
        nearest_indices=nearest,
        mean_distance=mean_distance,
        null_mean=null_mean,
        null_std=null_std,
    )


def project_coverage(model: BranchModel) -> ProjectedCoverage:
    """Normalize branch depth and remove the fitted PCA coverage trend."""

    if model.counts.ndim != 1 or model.counts.size != model.masked.size:
        raise PredictError("counts and mask must be aligned one-dimensional arrays")
    if np.any(~np.isfinite(model.counts)) or np.any(model.counts < 0):
        raise PredictError("sample counts must be finite and non-negative")
    total = float(np.sum(model.counts))
    if total <= 0:
        raise PredictError(f"branch {model.branch!r} has zero sample coverage")
    normalized = model.counts / total
    usable = ~model.masked
    selected = normalized[usable]
    if selected.shape != model.pca_mean.shape:
        raise PredictError("PCA mean does not align with unmasked sample bins")
    centered = selected - model.pca_mean
    reconstruction = (
        centered @ model.pca_loadings.T
    ) @ model.pca_loadings + model.pca_mean
    corrected = np.full(normalized.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(reconstruction) & (reconstruction > 0)
    corrected_values = np.full(selected.shape, np.nan, dtype=np.float64)
    corrected_values[valid] = selected[valid] / reconstruction[valid]
    corrected[usable] = corrected_values
    usable &= np.isfinite(corrected) & (corrected > 0)
    return ProjectedCoverage(normalized, corrected, usable)


def normalize_branch(model: BranchModel, config: PredictConfig) -> BinResult:
    """Calculate within-sample log2 ratios, null z-scores, and segment weights."""

    logger.debug("normalizing branch %s (%d bins)", model.branch, model.counts.size)
    projected = project_coverage(model)
    unmasked_positions = np.flatnonzero(~model.masked)
    corrected = projected.corrected[unmasked_positions]
    n_unmasked = unmasked_positions.size

    ratios = np.full(model.counts.shape, np.nan, dtype=np.float64)
    zscores = np.full(model.counts.shape, np.nan, dtype=np.float64)
    weights = np.zeros(model.counts.shape, dtype=np.float64)
    reference_sizes = np.zeros(model.counts.shape, dtype=np.int64)

    eligible = np.isfinite(corrected) & (corrected > 0)
    local_ratios = np.full(corrected.shape, np.nan, dtype=np.float64)
    local_zscores = np.full(corrected.shape, np.nan, dtype=np.float64)

    filtered_neighbors = model.nearest_indices
    null_mean = model.null_mean
    null_std = model.null_std

    min_ref_bins = config.min_reference_bins

    for iteration in range(config.mask_repeats):
        for index in range(n_unmasked):
            val = corrected[index]
            neighbors = filtered_neighbors[index]
            mask = eligible[neighbors] & (neighbors != index)
            selected = neighbors[mask]
            n_sel = int(selected.size)
            reference_sizes[unmasked_positions[index]] = n_sel
            if not np.isfinite(val) or val <= 0 or n_sel < min_ref_bins:
                local_ratios[index] = np.nan
                local_zscores[index] = np.nan
                continue
            baseline = float(np.median(corrected[selected]))
            if not np.isfinite(baseline) or baseline <= 0:
                continue
            ratio = float(np.log2(val / baseline))
            local_ratios[index] = ratio
            s = null_std[index]
            local_zscores[index] = (ratio - null_mean[index]) / s if s > 0 else 0.0

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
    ratios[unmasked_positions] = local_ratios
    zscores[unmasked_positions] = local_zscores

    valid_dist = (model.mean_distance > 0) & np.isfinite(model.mean_distance)
    unmasked_weights = np.zeros(n_unmasked, dtype=np.float64)
    unmasked_weights[valid_dist] = 1.0 / model.mean_distance[valid_dist]
    unmasked_weights[~valid_dist] = 1.0
    weights[unmasked_positions] = unmasked_weights
    valid = (
        projected.usable
        & np.isfinite(ratios)
        & np.isfinite(zscores)
        & (reference_sizes >= config.min_reference_bins)
    )
    ratios[~valid] = np.nan
    zscores[~valid] = np.nan
    weights[~valid] = 0.0
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
    ratio_threshold: float | None,
) -> CallType | None:
    """Classify a segment using a symmetric ratio or z-score threshold."""

    value = ratio if ratio_threshold is not None else zscore
    threshold = ratio_threshold if ratio_threshold is not None else z_threshold
    if not np.isfinite(value):
        return None
    if value > threshold:
        return "gain"
    if value < -threshold:
        return "loss"
    return None


def segment_and_call(
    bins: tuple[BinResult, ...],
    contig_lengths: dict[str, int],
    *,
    penalty: float,
    z_threshold: float,
    ratio_threshold: float | None,
    max_blacklist_gap: int,
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
    evidence: dict[str, tuple[BinResult, NDArray[np.bool_]]] = {}
    for result in bins:
        contig_array = np.asarray(result.contigs, dtype=object)
        for contig in dict.fromkeys(result.contigs):
            selected = contig_array == contig
            by_contig[contig] = (
                result.starts[selected],
                result.ends[selected],
                result.ratios[selected],
                result.weights[selected],
                contig_lengths[contig],
            )
            evidence[contig] = (result, selected)
    raw_segments = segment_genome(
        by_contig,
        penalty=penalty,
        algorithm="binseg",
        max_blacklist_gap=max_blacklist_gap,
    )
    output: list[SegmentResult] = []
    for segment in raw_segments:
        result, contig_mask = evidence[segment.contig]
        overlap = (
            contig_mask
            & (result.starts < segment.end)
            & (result.ends > segment.start)
            & np.isfinite(result.zscores)
            & (result.weights > 0)
        )
        segment_z = (
            float(np.average(result.zscores[overlap], weights=result.weights[overlap]))
            if np.any(overlap)
            else np.nan
        )
        output.append(
            SegmentResult(
                contig=segment.contig,
                start=segment.start,
                end=min(segment.end, contig_lengths[segment.contig]),
                ratio=segment.mean,
                zscore=segment_z,
                weight=segment.weight,
                bin_count=segment.bin_count,
                call=call_segment(
                    segment.mean,
                    segment_z,
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
    for s in output:
        if s.call is not None:
            logger.debug(
                "call %s: %s:%d-%d (%d bins, ratio=%.3f, z=%.2f)",
                s.call.upper(),
                s.contig,
                s.start,
                s.end,
                s.bin_count,
                s.ratio,
                s.zscore,
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
    sex_branches = {BRANCH_FEMALE_SEX, BRANCH_MALE_SEX} & branches
    if not sex_branches:
        return None
    model = metadata.get("model")
    if not isinstance(model, dict):
        raise PredictError("cannot infer sample sex; pass --sex female or --sex male")
    inference = model.get("sex_inference")
    signal = model.get("sex_signal_contig")
    if not isinstance(inference, dict) or not isinstance(signal, str):
        raise PredictError("cannot infer sample sex; pass --sex female or --sex male")
    cutoff = inference.get("cutoff")
    if not isinstance(cutoff, int | float):
        raise PredictError("cannot infer sample sex; pass --sex female or --sex male")
    contigs = np.asarray(sample.column("contig").to_pylist(), dtype=object)
    counts = np.asarray(sample.column("count").to_pylist(), dtype=np.float64)
    total = float(np.sum(counts))
    if total <= 0:
        raise PredictError("sample has zero total coverage")
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
    values = metadata.get("contig_lengths")
    if not isinstance(values, dict):
        raise PredictError("artifact metadata has no contig lengths")
    return {str(contig): int(length) for contig, length in values.items()}


def _validate_config(config: PredictConfig) -> None:
    if not np.isfinite(config.penalty) or config.penalty <= 0:
        raise PredictError("penalty must be finite and positive")
    if not np.isfinite(config.z_threshold) or config.z_threshold <= 0:
        raise PredictError("z_threshold must be finite and positive")
    if config.ratio_threshold is not None and (
        not np.isfinite(config.ratio_threshold) or config.ratio_threshold <= 0
    ):
        raise PredictError("ratio_threshold must be finite and positive")
    if config.min_reference_bins < 1:
        raise PredictError("min_reference_bins must be positive")
    if config.mask_repeats < 1:
        raise PredictError("mask_repeats must be positive")
    if config.max_blacklist_gap < 0:
        raise PredictError("max_blacklist_gap must be non-negative")


def _require_aligned(length: int, *arrays: ArrayLike) -> None:
    if any(np.asarray(array).shape != (length,) for array in arrays):
        raise PredictError(
            "prediction result arrays must be aligned and one-dimensional"
        )
