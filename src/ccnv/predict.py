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

from ccnv.convert import rebin_sample_table
from ccnv.covariates import (
    CovariateError,
    correct_covariate_bias,
    load_gc_track,
    load_mappability_track,
    mask_covariates,
    validate_min_mappability,
)
from ccnv.io.parquet import read_reference_parquet, read_sample_parquet
from ccnv.io.tracks import load_cytobands_track
from ccnv.models import (
    DEFAULT_MIN_MAPPABILITY,
    AnalysisMode,
    PredictConfig,
    PredictError,
)
from ccnv.predict_qc import build_predict_qc_payload, write_predict_qc_json
from ccnv.prenatal import annotate_prenatal_segments, predict_prenatal
from ccnv.report import write_outputs
from ccnv.results import (
    BinResult,
    CallType,
    FetalFractionResult,
    PredictionOutputs,
    SegmentResult,
)
from ccnv.schemas import (
    BRANCH_AUTOSOMAL,
    BRANCH_FEMALE_SEX,
    BRANCH_MALE_SEX,
    PrenatalModelMetadata,
    bin_coordinates,
    decode_metadata,
)
from ccnv.segment import segment_genome

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
    nearest_indices: tuple[NDArray[np.int64], ...]
    mean_distance: NDArray[np.float32]
    null_mean: NDArray[np.float64]
    null_std: NDArray[np.float64]
    gc: NDArray[np.float64] | None = None
    mappability: NDArray[np.float64] | None = None


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
        normalize_branch(build_branch_model(inputs, b), config) for b in inputs.branches
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
        meta = PrenatalModelMetadata.from_table(inputs.reference)
        if meta is None:
            raise PredictError("reference is missing prenatal metadata (ccnv.prenatal)")
        prenatal_result = predict_prenatal(
            inputs.sample,
            inputs.reference,
            meta,
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
        config.prefix,
        bin_results,
        segments,
        inputs.contig_lengths,
        config.plot,
        config.vcf,
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
    """Read artifacts through their contracts and select model branches."""
    try:
        sample, reference = (
            read_sample_parquet(sample_path),
            read_reference_parquet(reference_path),
        )
    except Exception as err:
        raise PredictError(f"cannot read Parquet artifacts: {err}") from err
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
        branches, chosen_sex = (
            [BRANCH_AUTOSOMAL]
            + ([BRANCH_FEMALE_SEX] if BRANCH_FEMALE_SEX in branch_vals else []),
            ("female" if BRANCH_FEMALE_SEX in branch_vals else None),
        )
    else:
        chosen_sex = _resolve_sample_sex(sex, sample, ref_meta, branch_vals)
        branches = [BRANCH_AUTOSOMAL]
        if chosen_sex is not None:
            sb = BRANCH_FEMALE_SEX if chosen_sex == "female" else BRANCH_MALE_SEX
            if sb not in branch_vals and sex != "autosomal":
                raise PredictError(f"reference has no {sb!r} branch")
            if sb in branch_vals:
                branches.append(sb)
    ref_bs, sample_bs = (
        (ref_model.get("bin_size") if isinstance(ref_model, dict) else None),
        sample_meta.get("bin_size"),
    )
    if isinstance(ref_bs, int) and isinstance(sample_bs, int) and sample_bs != ref_bs:
        logger.debug(
            "rebinning sample from %d bp to reference bin size %d bp", sample_bs, ref_bs
        )
        sample = rebin_sample_table(sample, ref_bs)
    active_ref = reference.filter(
        pa.compute.is_in(reference.column("branch"), pa.array(branches))
    )
    for c in set(active_ref.column("contig").unique().to_pylist()):
        if sample_lens.get(c) != ref_lens.get(c):
            raise PredictError(f"contig length mismatch for {c!r}")
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
    """Align sample counts to a branch and materialize its nested model arrays."""
    rows = np.flatnonzero(
        np.asarray(inputs.reference.column("branch").to_numpy(zero_copy_only=False))
        == branch
    )
    if rows.size == 0:
        raise PredictError(f"reference branch {branch!r} is empty")
    s_row, n_rows = int(rows[0]), int(rows.size)
    b_tbl = (
        inputs.reference.slice(s_row, n_rows)
        if int(rows[-1]) == s_row + n_rows - 1
        else inputs.reference.take(pa.array(rows))
    )
    b_contigs = tuple(b_tbl.column("contig").to_pylist())
    s_branch = inputs.sample.filter(
        pa.compute.is_in(inputs.sample.column("contig"), pa.array(list(set(b_contigs))))
    )
    if (
        s_branch.num_rows == b_tbl.num_rows
        and s_branch.column("start").equals(b_tbl.column("start"))
        and s_branch.column("end").equals(b_tbl.column("end"))
        and s_branch.column("contig").equals(b_tbl.column("contig"))
    ):
        counts = (
            s_branch.column("count").to_numpy(zero_copy_only=False).astype(np.float64)
        )
    else:
        s_counts = {
            c: float(cnt)
            for c, cnt in zip(
                bin_coordinates(inputs.sample),
                inputs.sample.column("count").to_pylist(),
                strict=True,
            )
        }
        counts = np.asarray(
            [s_counts[c] for c in bin_coordinates(b_tbl)], dtype=np.float64
        )
    masked = b_tbl.column("masked").to_numpy(zero_copy_only=False).astype(np.bool_)
    unmasked = np.flatnonzero(~masked)
    pca_mean = np.nan_to_num(
        b_tbl.column("pca_mean").to_numpy(zero_copy_only=False)[unmasked], nan=0.0
    ).astype(np.float64)
    l_chunk = b_tbl.column("pca_loadings").combine_chunks()
    l_offs, l_flat = l_chunk.offsets.to_numpy(), l_chunk.values.to_numpy()
    if len(np.unique(l_offs[unmasked + 1] - l_offs[unmasked])) != 1:
        raise PredictError(f"branch {branch!r} has inconsistent PCA loading shapes")
    n_comp = int((l_offs[unmasked + 1] - l_offs[unmasked])[0])
    loadings = (
        l_flat.reshape(unmasked.size, n_comp).T.astype(np.float64)
        if unmasked.size == b_tbl.num_rows
        else np.column_stack(
            [l_flat[l_offs[r] : l_offs[r] + n_comp] for r in unmasked]
        ).astype(np.float64)
    )
    n_chunk = b_tbl.column("nearest_ref_indices").combine_chunks()
    n_offs, n_flat = n_chunk.offsets.to_numpy(), n_chunk.values.to_numpy()
    nearest = tuple(
        n_flat[n_offs[r] : n_offs[r + 1]].astype(np.int64) for r in unmasked
    )
    if any(np.any(idxs < 0) or np.any(idxs >= unmasked.size) for idxs in nearest):
        raise PredictError(f"branch {branch!r} has invalid nearest references")
    return BranchModel(
        branch=branch,
        contigs=b_contigs,
        starts=b_tbl.column("start").to_numpy(zero_copy_only=False),
        ends=b_tbl.column("end").to_numpy(zero_copy_only=False),
        counts=counts,
        masked=masked,
        pca_mean=pca_mean,
        pca_loadings=loadings,
        nearest_indices=nearest,
        mean_distance=b_tbl.column("mean_distance").to_numpy(zero_copy_only=False)[
            unmasked
        ],
        null_mean=b_tbl.column("null_mean").to_numpy(zero_copy_only=False)[unmasked],
        null_std=b_tbl.column("null_std").to_numpy(zero_copy_only=False)[unmasked],
        gc=_extract_covariate_col(b_tbl, "gc"),
        mappability=_extract_covariate_col(b_tbl, "mappability"),
    )


def _extract_covariate_col(table: pa.Table, name: str) -> NDArray[np.float64] | None:
    if name not in table.schema.names:
        return None
    raw = table.column(name).to_numpy(zero_copy_only=False).astype(np.float64)
    raw[raw < 0.0] = np.nan
    return raw if np.any(np.isfinite(raw)) else None


def project_coverage(
    model: BranchModel,
    *,
    gc_override: NDArray[np.float64] | None = None,
    mappability_override: NDArray[np.float64] | None = None,
    n_mask_override: NDArray[np.bool_] | None = None,
    min_mappability: float = DEFAULT_MIN_MAPPABILITY,
) -> ProjectedCoverage:
    """Normalize branch depth, remove covariate bias, and subtract PCA trend."""
    if model.counts.ndim != 1 or model.counts.size != model.masked.size:
        raise PredictError("counts and mask must be aligned one-dimensional arrays")
    if (
        np.any(~np.isfinite(model.counts))
        or np.any(model.counts < 0)
        or float(np.sum(model.counts)) <= 0
    ):
        raise PredictError("sample counts must be finite, non-negative, and non-zero")
    ref_unm = ~model.masked
    gc, map_ = (
        (gc_override if gc_override is not None else model.gc),
        (
            mappability_override
            if mappability_override is not None
            else model.mappability
        ),
    )
    cov_usable = mask_covariates(
        ref_unm,
        gc=gc,
        mappability=map_,
        n_mask=n_mask_override,
        min_mappability=min_mappability,
    )
    usable_total = float(np.sum(model.counts[cov_usable]))
    if usable_total <= 0:
        raise PredictError(
            f"branch {model.branch!r} has zero sample coverage on usable bins"
        )
    normalized = correct_covariate_bias(
        model.counts / usable_total, cov_usable, gc=gc, mappability=map_
    )
    selected = normalized[ref_unm]
    if selected.shape != model.pca_mean.shape:
        raise PredictError("PCA mean does not align with unmasked sample bins")
    active_sub = cov_usable[ref_unm] & np.isfinite(selected)
    reconstruction = model.pca_mean.copy()
    if np.count_nonzero(active_sub) > int(
        np.linalg.matrix_rank(model.pca_loadings[:, active_sub].T)
    ):
        sub_c = selected[active_sub] - model.pca_mean[active_sub]
        coeffs, *_ = np.linalg.lstsq(
            model.pca_loadings[:, active_sub].T, sub_c, rcond=None
        )
        reconstruction = (coeffs @ model.pca_loadings) + model.pca_mean
    corrected = np.full(normalized.shape, np.nan, dtype=np.float64)
    valid = active_sub & np.isfinite(reconstruction) & (reconstruction > 0)
    corrected_values = np.full(selected.shape, np.nan, dtype=np.float64)
    corrected_values[valid] = selected[valid] / reconstruction[valid]
    corrected[ref_unm] = corrected_values
    return ProjectedCoverage(
        normalized, corrected, cov_usable & np.isfinite(corrected) & (corrected > 0)
    )


def normalize_branch(model: BranchModel, config: PredictConfig) -> BinResult:
    """Calculate within-sample log2 ratios, null z-scores, and segment weights."""
    logger.debug("normalizing branch %s (%d bins)", model.branch, model.counts.size)
    try:
        gc_ov, n_ov = (
            load_gc_track(config.gc, model.contigs, model.starts, model.ends)
            if config.gc
            else (None, None)
        )
        map_ov = (
            load_mappability_track(
                config.mappability, model.contigs, model.starts, model.ends
            )
            if config.mappability
            else None
        )
    except CovariateError as err:
        raise PredictError(str(err)) from err
    proj = project_coverage(
        model,
        gc_override=gc_ov,
        mappability_override=map_ov,
        n_mask_override=n_ov,
        min_mappability=config.min_mappability,
    )
    unm = np.flatnonzero(~model.masked)
    corrected = proj.corrected[unm]
    ratios, zscores, weights = (
        np.full(model.counts.shape, np.nan),
        np.full(model.counts.shape, np.nan),
        np.zeros(model.counts.shape),
    )
    ref_sizes = np.zeros(model.counts.shape, dtype=np.int64)
    eligible = np.isfinite(corrected) & (corrected > 0)
    loc_r, loc_z = np.full(corrected.shape, np.nan), np.full(corrected.shape, np.nan)
    flt_nbrs = tuple(nbrs[nbrs != i] for i, nbrs in enumerate(model.nearest_indices))
    for iteration in range(config.mask_repeats):
        for idx in range(unm.size):
            val, nbrs = corrected[idx], flt_nbrs[idx]
            sel = nbrs[eligible[nbrs]]
            ref_sizes[unm[idx]] = sel.size
            if not np.isfinite(val) or val <= 0 or sel.size < config.min_reference_bins:
                loc_r[idx], loc_z[idx] = np.nan, np.nan
                continue
            base = float(np.median(corrected[sel]))
            if not np.isfinite(base) or base <= 0:
                continue
            r = float(np.log2(val / base))
            loc_r[idx], s = r, model.null_std[idx]
            loc_z[idx] = (r - model.null_mean[idx]) / s if s > 0 else np.nan
        miss_z = np.isfinite(loc_r) & ~np.isfinite(loc_z)
        if np.any(miss_z):
            fin_r = loc_r[np.isfinite(loc_r)]
            med_r, emp_std = (
                float(np.median(fin_r)),
                1.4826 * float(scipy.stats.median_abs_deviation(fin_r, scale=1.0)),
            )
            loc_z[miss_z] = (
                (loc_r[miss_z] - med_r) / emp_std if emp_std > 1e-12 else 0.0
            )
        eligible = (
            np.isfinite(corrected)
            & np.isfinite(loc_z)
            & (np.abs(loc_z) < _MASK_Z_THRESHOLD)
        )
        logger.debug(
            "branch %s within-sample iteration %d/%d: %d eligible bins",
            model.branch,
            iteration + 1,
            config.mask_repeats,
            int(np.count_nonzero(eligible)),
        )
    finite_ratios = np.isfinite(loc_r)
    median_shift = 0.0
    if np.any(finite_ratios):
        median_shift = float(np.median(loc_r[finite_ratios]))
        loc_r[finite_ratios] -= median_shift
    logger.debug(
        "branch %s: %d finite ratios, median shift=%.4f (mean=%.4f, std=%.4f)",
        model.branch,
        int(np.count_nonzero(finite_ratios)),
        median_shift,
        float(np.mean(loc_r[finite_ratios])) if np.any(finite_ratios) else 0.0,
        float(np.std(loc_r[finite_ratios])) if np.any(finite_ratios) else 0.0,
    )
    ratios[unm], zscores[unm] = loc_r, loc_z
    valid_d = (model.mean_distance > 0) & np.isfinite(model.mean_distance)
    weights[unm] = np.where(
        valid_d, 1.0 / np.where(valid_d, model.mean_distance, 1.0), 1.0
    )
    valid = (
        proj.usable
        & np.isfinite(ratios)
        & np.isfinite(zscores)
        & (ref_sizes >= config.min_reference_bins)
    )
    ratios[~valid], zscores[~valid], weights[~valid] = np.nan, np.nan, 0.0
    _require_aligned(model.counts.size, ratios, zscores, weights, ref_sizes)
    return BinResult(
        branch=model.branch,
        contigs=model.contigs,
        starts=model.starts,
        ends=model.ends,
        ratios=ratios,
        zscores=zscores,
        weights=weights,
        reference_sizes=ref_sizes,
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
    by_contig, evidence = {}, {}
    for res in bins:
        c_arr = np.asarray(res.contigs, dtype=object)
        for c in dict.fromkeys(res.contigs):
            sel = c_arr == c
            c_s, c_e, c_w = res.starts[sel], res.ends[sel], res.weights[sel]
            by_contig[c] = (c_s, c_e, res.ratios[sel], c_w, contig_lengths[c])
            evidence[c] = (c_s, c_e, res.zscores[sel], c_w)
    raw = segment_genome(
        by_contig,
        penalty=penalty,
        algorithm="binseg",
        max_blacklist_gap=max_blacklist_gap,
    )
    output: list[SegmentResult] = []
    for seg in raw:
        c_s, c_e, c_z, c_w = evidence[seg.contig]
        ov = (c_s < seg.end) & (c_e > seg.start) & np.isfinite(c_z) & (c_w > 0)
        seg_z = float(np.average(c_z[ov], weights=c_w[ov])) if np.any(ov) else np.nan
        output.append(
            SegmentResult(
                contig=seg.contig,
                start=seg.start,
                end=min(seg.end, contig_lengths[seg.contig]),
                ratio=seg.mean,
                zscore=seg_z,
                weight=seg.weight,
                bin_count=seg.bin_count,
                call=call_segment(
                    seg.mean,
                    seg_z,
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
        np.asarray(sample.column("count").to_pylist(), dtype=np.float64),
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
