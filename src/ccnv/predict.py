"""Typed copy-number normalization, segmentation, calling, and output stages."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import ArrayLike, NDArray

from ccnv.convert import read_sample_parquet
from ccnv.models import PredictConfig
from ccnv.reference import (
    BRANCH_AUTOSOMAL,
    BRANCH_FEMALE_SEX,
    BRANCH_MALE_SEX,
)
from ccnv.schemas import (
    ArtifactKind,
    SchemaError,
    decode_metadata,
    validate_table,
)
from ccnv.segment import segment_genome

logger = logging.getLogger(__name__)

Sex = Literal["female", "male"]
CallType = Literal["gain", "loss"]

BINS_BED_HEADER = (
    "contig",
    "start",
    "end",
    "id",
    "ratio",
    "zscore",
    "weight",
    "reference_size",
    "masked",
    "branch",
)
SEGMENTS_BED_HEADER = (
    "contig",
    "start",
    "end",
    "ratio",
    "zscore",
    "weight",
    "bin_count",
    "call",
)
CALLS_BED_HEADER = ("contig", "start", "end", "ratio", "zscore", "type")


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
    nearest_distances: tuple[NDArray[np.float64], ...]
    null_ratios: tuple[NDArray[np.float64], ...]


@dataclass(frozen=True, slots=True)
class ProjectedCoverage:
    """Coverage after depth normalization, masking, and PCA projection."""

    normalized: NDArray[np.float64]
    corrected: NDArray[np.float64]
    usable: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class BinResult:
    """Per-bin normalized copy-number evidence."""

    branch: str
    contigs: tuple[str, ...]
    starts: NDArray[np.int64]
    ends: NDArray[np.int64]
    ratios: NDArray[np.float64]
    zscores: NDArray[np.float64]
    weights: NDArray[np.float64]
    reference_sizes: NDArray[np.int64]
    masked: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class SegmentResult:
    """One segmented interval with aggregate evidence and an optional call."""

    contig: str
    start: int
    end: int
    ratio: float
    zscore: float
    weight: float
    bin_count: int
    call: CallType | None


@dataclass(frozen=True, slots=True)
class PredictionOutputs:
    """Paths written by :func:`run`."""

    bins_bed: Path
    segments_bed: Path
    calls_bed: Path
    plot_svg: Path | None
    bins_parquet: Path
    segments_parquet: Path
    bins: tuple[BinResult, ...]
    segments: tuple[SegmentResult, ...]


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
    logger.debug(
        "produced %d segments (%d called)",
        len(segments),
        sum(1 for item in segments if item.call is not None),
    )
    return write_outputs(config.prefix, bin_results, segments, inputs, config.plot)


def load_inputs(
    sample_path: Path,
    reference_path: Path,
    *,
    sex: Literal["auto", "female", "male"] = "auto",
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
    sample_bins = set(_coordinates(sample))
    branch_values = set(reference.column("branch").to_pylist())
    chosen_sex = _resolve_sample_sex(sex, sample, reference_meta, branch_values)
    branches = [BRANCH_AUTOSOMAL]
    if BRANCH_AUTOSOMAL not in branch_values:
        raise PredictError("reference has no autosomal branch")
    if chosen_sex is not None:
        sex_branch = BRANCH_FEMALE_SEX if chosen_sex == "female" else BRANCH_MALE_SEX
        if sex_branch not in branch_values:
            if sex != "auto":
                raise PredictError(f"reference has no {sex_branch!r} branch")
        else:
            branches.append(sex_branch)

    required = {
        (contig, start, end)
        for branch, contig, start, end in zip(
            reference.column("branch").to_pylist(),
            reference.column("contig").to_pylist(),
            reference.column("start").to_pylist(),
            reference.column("end").to_pylist(),
            strict=True,
        )
        if branch in branches
    }
    missing = sorted(required - sample_bins)
    if missing:
        contigs = ", ".join(dict.fromkeys(item[0] for item in missing))
        raise PredictError(f"sample is missing reference bins on contig(s): {contigs}")
    for contig in {item[0] for item in required}:
        if sample_lengths.get(contig) != reference_lengths.get(contig):
            raise PredictError(f"contig length mismatch for {contig!r}")

    return PredictionInput(
        sample=sample,
        reference=reference,
        contig_lengths=reference_lengths,
        sex=chosen_sex,
        branches=tuple(branches),
    )


def build_branch_model(inputs: PredictionInput, branch: str) -> BranchModel:
    """Align sample counts to a branch and materialize its nested model arrays."""

    reference_branches = np.asarray(
        inputs.reference.column("branch").to_pylist(), dtype=object
    )
    row_indexes = np.flatnonzero(reference_branches == branch)
    if row_indexes.size == 0:
        raise PredictError(f"reference branch {branch!r} is empty")
    branch_table = inputs.reference.take(pa.array(row_indexes))
    sample_counts = {
        coordinate: float(count)
        for coordinate, count in zip(
            _coordinates(inputs.sample),
            inputs.sample.column("count").to_pylist(),
            strict=True,
        )
    }
    coordinates = _coordinates(branch_table)
    counts = np.asarray([sample_counts[item] for item in coordinates], dtype=np.float64)
    masked = np.asarray(branch_table.column("masked").to_pylist(), dtype=np.bool_)
    unmasked_count = int(np.count_nonzero(~masked))
    loading_rows = branch_table.column("pca_loadings").to_pylist()
    component_counts = {len(loading_rows[index]) for index in np.flatnonzero(~masked)}
    if len(component_counts) != 1:
        raise PredictError(f"branch {branch!r} has inconsistent PCA loading shapes")
    n_components = component_counts.pop()
    loadings = np.asarray(
        [loading_rows[index] for index in np.flatnonzero(~masked)],
        dtype=np.float64,
    ).T
    if loadings.shape != (n_components, unmasked_count):
        raise PredictError(f"branch {branch!r} has invalid PCA loading shape")

    unmasked_rows = np.flatnonzero(~masked)
    nearest_values = branch_table.column("nearest_ref_indices").to_pylist()
    distance_values = branch_table.column("nearest_ref_distances").to_pylist()
    null_values = branch_table.column("null_ratios").to_pylist()
    nearest = tuple(
        np.asarray(nearest_values[index], dtype=np.int64) for index in unmasked_rows
    )
    distances = tuple(
        np.asarray(distance_values[index], dtype=np.float64) for index in unmasked_rows
    )
    null_ratios = tuple(
        np.asarray(null_values[index], dtype=np.float64) for index in unmasked_rows
    )
    if not (len(nearest) == len(distances) == len(null_ratios) == unmasked_count):
        raise PredictError(
            f"branch {branch!r} nested arrays do not align with its mask"
        )
    for indexes, values in zip(nearest, distances, strict=True):
        if (
            indexes.size != values.size
            or np.any(indexes < 0)
            or np.any(indexes >= unmasked_count)
        ):
            raise PredictError(f"branch {branch!r} has invalid nearest references")

    return BranchModel(
        branch=branch,
        contigs=tuple(branch_table.column("contig").to_pylist()),
        starts=np.asarray(branch_table.column("start").to_pylist(), dtype=np.int64),
        ends=np.asarray(branch_table.column("end").to_pylist(), dtype=np.int64),
        counts=counts,
        masked=masked,
        pca_mean=np.asarray(
            branch_table.filter(pa.array(~masked)).column("pca_mean").to_pylist(),
            dtype=np.float64,
        ),
        pca_loadings=loadings,
        nearest_indices=nearest,
        nearest_distances=distances,
        null_ratios=null_ratios,
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
    ratios = np.full(model.counts.shape, np.nan, dtype=np.float64)
    zscores = np.full(model.counts.shape, np.nan, dtype=np.float64)
    weights = np.zeros(model.counts.shape, dtype=np.float64)
    reference_sizes = np.zeros(model.counts.shape, dtype=np.int64)
    cutoff = distance_cutoff(model.nearest_distances, config.mask_repeats)
    eligible = np.isfinite(corrected) & (corrected > 0)
    local_ratios = np.full(corrected.shape, np.nan, dtype=np.float64)
    local_zscores = np.full(corrected.shape, np.nan, dtype=np.float64)

    for _ in range(config.mask_repeats):
        for index, (neighbors, distances, null) in enumerate(
            zip(
                model.nearest_indices,
                model.nearest_distances,
                model.null_ratios,
                strict=True,
            )
        ):
            selected = neighbors[
                (distances <= cutoff) & eligible[neighbors] & (neighbors != index)
            ]
            reference_sizes[unmasked_positions[index]] = selected.size
            if (
                not np.isfinite(corrected[index])
                or corrected[index] <= 0
                or selected.size < config.min_reference_bins
                or null.size < 2
            ):
                local_ratios[index] = np.nan
                local_zscores[index] = np.nan
                continue
            baseline = float(np.median(corrected[selected]))
            if not np.isfinite(baseline) or baseline <= 0:
                continue
            ratio = float(np.log2(corrected[index] / baseline))
            null_finite = null[np.isfinite(null)]
            null_std = float(np.std(null_finite, ddof=1))
            local_ratios[index] = ratio
            local_zscores[index] = (
                (ratio - float(np.mean(null_finite))) / null_std
                if null_std > 0
                else 0.0
            )
        eligible = (
            np.isfinite(corrected)
            & np.isfinite(local_zscores)
            & (np.abs(local_zscores) < 2.326347874)
        )

    finite_ratios = np.isfinite(local_ratios)
    if np.any(finite_ratios):
        local_ratios[finite_ratios] -= float(np.median(local_ratios[finite_ratios]))
    ratios[unmasked_positions] = local_ratios
    zscores[unmasked_positions] = local_zscores
    for local, distances in enumerate(model.nearest_distances):
        finite = distances[np.isfinite(distances) & (distances <= cutoff)]
        if finite.size:
            mean_distance = float(np.mean(np.sqrt(np.maximum(finite, 0.0))))
            weights[unmasked_positions[local]] = (
                1.0 / mean_distance if mean_distance > 0 else 1.0
            )
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


def distance_cutoff(distances: tuple[NDArray[np.float64], ...], repeats: int) -> float:
    """Return an iterative three-sigma cutoff for nearest-bin distances."""

    values = np.concatenate(distances) if distances else np.empty(0)
    values = values[np.isfinite(values) & (values >= 0)]
    if values.size == 0:
        return np.inf
    cutoff = np.inf
    for _ in range(repeats):
        selected = values[values < cutoff]
        if selected.size == 0:
            break
        cutoff = float(np.mean(selected) + 3.0 * np.std(selected))
    return cutoff


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
    return tuple(output)


def write_outputs(
    prefix: Path,
    bins: tuple[BinResult, ...],
    segments: tuple[SegmentResult, ...],
    inputs: PredictionInput,
    plot: bool,
) -> PredictionOutputs:
    """Write 0-based half-open BED evidence, optional Parquet, and an SVG plot."""

    prefix.parent.mkdir(parents=True, exist_ok=True)
    bins_bed = Path(f"{prefix}_bins.bed")
    segments_bed = Path(f"{prefix}_segments.bed")
    calls_bed = Path(f"{prefix}_calls.bed")
    bins_parquet = Path(f"{prefix}_bins.parquet")
    segments_parquet = Path(f"{prefix}_segments.parquet")
    plot_path = Path(f"{prefix}_plot.svg") if plot else None
    logger.debug("writing prediction outputs with prefix %s", prefix)
    _write_bed(bins_bed, BINS_BED_HEADER, _bin_bed_rows(bins))
    _write_bed(segments_bed, SEGMENTS_BED_HEADER, _segment_bed_rows(segments))
    _write_bed(
        calls_bed,
        CALLS_BED_HEADER,
        (
            (
                item.contig,
                item.start,
                item.end,
                _format_bed_float(item.ratio),
                _format_bed_float(item.zscore),
                item.call,
            )
            for item in segments
            if item.call is not None
        ),
    )
    pq.write_table(_bins_table(bins), bins_parquet, compression="zstd")
    pq.write_table(_segments_table(segments), segments_parquet, compression="zstd")
    if plot_path is not None:
        write_genome_plot(plot_path, bins, segments, inputs.contig_lengths)
    return PredictionOutputs(
        bins_bed,
        segments_bed,
        calls_bed,
        plot_path,
        bins_parquet,
        segments_parquet,
        bins,
        segments,
    )


def write_genome_plot(
    path: Path,
    bins: tuple[BinResult, ...],
    segments: tuple[SegmentResult, ...],
    contig_lengths: dict[str, int],
) -> None:
    """Write a dependency-free SVG with true contig-length genome offsets."""

    ordered = tuple(
        dict.fromkeys(contig for result in bins for contig in result.contigs)
    )
    offsets: dict[str, int] = {}
    total = 0
    for contig in ordered:
        offsets[contig] = total
        total += contig_lengths[contig]
    width, height = 1200, 360
    left, right, top, bottom = 55, 20, 20, 45
    plot_width, plot_height = width - left - right, height - top - bottom

    def x(contig: str, position: float) -> float:
        return left + plot_width * (offsets[contig] + position) / max(total, 1)

    def y(value: float) -> float:
        clipped = min(max(value, -2.0), 2.0)
        return top + plot_height * (2.0 - clipped) / 4.0

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{y(0):.2f}" x2="{width - right}" '
        f'y2="{y(0):.2f}" stroke="#999"/>',
    ]
    for result in bins:
        for contig, start, end, ratio, masked in zip(
            result.contigs,
            result.starts,
            result.ends,
            result.ratios,
            result.masked,
            strict=True,
        ):
            if masked or not np.isfinite(ratio):
                continue
            lines.append(
                f'<circle cx="{x(contig, (start + end) / 2):.2f}" '
                f'cy="{y(float(ratio)):.2f}" r="1.2" fill="#4776b4"/>'
            )
    for item in segments:
        color = (
            "#c43c39"
            if item.call == "gain"
            else ("#3c8c5a" if item.call == "loss" else "#222")
        )
        lines.append(
            f'<line x1="{x(item.contig, item.start):.2f}" '
            f'y1="{y(item.ratio):.2f}" x2="{x(item.contig, item.end):.2f}" '
            f'y2="{y(item.ratio):.2f}" stroke="{color}" stroke-width="2"/>'
        )
    for contig in ordered:
        midpoint = contig_lengths[contig] / 2
        lines.append(
            f'<text x="{x(contig, midpoint):.2f}" y="{height - 15}" '
            f'text-anchor="middle" font-size="10">{_xml_escape(contig)}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_bed(
    path: Path, header: tuple[str, ...], rows: Iterable[tuple[object, ...]]
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write("\t".join(str(value) for value in row) + "\n")


def _format_bed_float(value: float) -> str:
    if not np.isfinite(value):
        return "NaN"
    return f"{value:.8g}"


def _bin_bed_rows(results: tuple[BinResult, ...]) -> tuple[tuple[object, ...], ...]:
    rows: list[tuple[object, ...]] = []
    for result in results:
        for index in range(len(result.contigs)):
            contig = result.contigs[index]
            start = int(result.starts[index])
            end = int(result.ends[index])
            rows.append(
                (
                    contig,
                    start,
                    end,
                    f"{contig}:{start}-{end}",
                    _format_bed_float(float(result.ratios[index])),
                    _format_bed_float(float(result.zscores[index])),
                    _format_bed_float(float(result.weights[index])),
                    int(result.reference_sizes[index]),
                    "true" if bool(result.masked[index]) else "false",
                    result.branch,
                )
            )
    return tuple(rows)


def _segment_bed_rows(
    results: tuple[SegmentResult, ...],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            item.contig,
            item.start,
            item.end,
            _format_bed_float(item.ratio),
            _format_bed_float(item.zscore),
            _format_bed_float(item.weight),
            item.bin_count,
            item.call or "",
        )
        for item in results
    )


def _bins_table(results: tuple[BinResult, ...]) -> pa.Table:
    rows = [
        (result, index) for result in results for index in range(len(result.contigs))
    ]
    return pa.table(
        {
            "branch": [result.branch for result, _ in rows],
            "contig": [result.contigs[index] for result, index in rows],
            "start": pa.array(
                [int(result.starts[index]) for result, index in rows], type=pa.int64()
            ),
            "end": pa.array(
                [int(result.ends[index]) for result, index in rows], type=pa.int64()
            ),
            "ratio": pa.array(
                [float(result.ratios[index]) for result, index in rows],
                type=pa.float64(),
                from_pandas=True,
            ),
            "zscore": pa.array(
                [float(result.zscores[index]) for result, index in rows],
                type=pa.float64(),
                from_pandas=True,
            ),
            "weight": pa.array(
                [float(result.weights[index]) for result, index in rows],
                type=pa.float64(),
            ),
            "reference_size": pa.array(
                [int(result.reference_sizes[index]) for result, index in rows],
                type=pa.int64(),
            ),
            "masked": [bool(result.masked[index]) for result, index in rows],
        }
    )


def _segments_table(results: tuple[SegmentResult, ...]) -> pa.Table:
    return pa.table(
        {
            "contig": [item.contig for item in results],
            "start": pa.array([item.start for item in results], type=pa.int64()),
            "end": pa.array([item.end for item in results], type=pa.int64()),
            "ratio": pa.array([item.ratio for item in results], type=pa.float64()),
            "zscore": pa.array(
                [item.zscore for item in results],
                type=pa.float64(),
                from_pandas=True,
            ),
            "weight": pa.array([item.weight for item in results], type=pa.float64()),
            "bin_count": pa.array(
                [item.bin_count for item in results], type=pa.int64()
            ),
            "call": pa.array([item.call for item in results], type=pa.string()),
        }
    )


def _resolve_sample_sex(
    requested: Literal["auto", "female", "male"],
    sample: pa.Table,
    metadata: dict[str, object],
    branches: set[object],
) -> Sex | None:
    if requested != "auto":
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
    return "male" if fraction > float(cutoff) else "female"


def _coordinates(table: pa.Table) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        zip(
            table.column("contig").to_pylist(),
            table.column("start").to_pylist(),
            table.column("end").to_pylist(),
            strict=True,
        )
    )


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


def _xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
