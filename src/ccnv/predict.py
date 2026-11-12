"""Typed copy-number normalization, segmentation, and calling."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Literal, overload

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from numpy.typing import ArrayLike, NDArray

from ccnv.extract import extract_for_predict, extracted_to_sample_buffers
from ccnv.features.covariates import validate_min_mappability
from ccnv.features.fragmentomics import (
    apply_prenatal_fisher_weighting,
    resolve_fragmentomics_compatibility,
)
from ccnv.features.normalization import (
    BranchModel,
    BranchNormalizationResult,
    ProjectedCoverage,
    build_branch_model_from_tables,
    normalize_branch,
    normalize_sample_branch,
    project_coverage,
)
from ccnv.features.prenatal import predict_prenatal
from ccnv.features.prenatal_origin import annotate_prenatal_origin_and_lod
from ccnv.features.rebinning import rebin_sample_buffers, rebin_sample_sketch
from ccnv.features.sex import infer_single_sample_sex
from ccnv.io.parquet import (
    read_reference_parquet,
    read_sample_parquet,
    unique_dictionary_contigs,
)
from ccnv.io.tracks import load_cytobands_track
from ccnv.models import (
    BRANCH_AUTOSOMAL,
    BRANCH_FEMALE_SEX,
    BRANCH_MALE_SEX,
    BRANCH_PRENATAL_AUTOSOMAL,
    BRANCH_PRENATAL_FFY,
    AnalysisMode,
    ArtifactKind,
    BinCompatibilityError,
    BinResult,
    CallType,
    CovariateError,
    FetalFractionResult,
    PredictConfig,
    PredictError,
    PredictionOutputs,
    PrenatalModelMetadata,
    ReferenceMetadata,
    SampleBuffers,
    SegmentResult,
    decode_metadata,
    validate_sample_table,
    validate_schema,
)
from ccnv.qc.predict import build_predict_qc_payload, write_predict_qc_json
from ccnv.report import write_outputs
from ccnv.stats.segmentation import (
    compute_contig_rho1,
    evaluate_stouffer_segment_scores,
    segment_and_score_genome,
)

Sex = Literal["female", "male"]
logger = logging.getLogger(__name__)

__all__ = [
    "BranchModel",
    "BranchNormalizationResult",
    "PredictError",
    "PredictionInput",
    "ProjectedCoverage",
    "SampleBuffers",
    "build_branch_model",
    "call_segment",
    "extract_for_predict",
    "load_inputs",
    "normalize_branch",
    "normalize_sample_branch",
    "project_coverage",
    "run",
    "run_predict",
    "segment_and_call",
]


@dataclass(frozen=True, slots=True)
class PredictionInput:
    sample: pa.Table | SampleBuffers
    reference: pa.Table
    contig_lengths: dict[str, int]
    sex: Sex | None
    branches: tuple[str, ...]


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
        threads=config.threads,
    )
    raw_sample = inputs.sample
    prenatal_meta: PrenatalModelMetadata | None = None
    prenatal_result: FetalFractionResult | None = None
    if config.mode == "prenatal":
        prenatal_meta = PrenatalModelMetadata.from_table(inputs.reference)
        if prenatal_meta is None:
            raise PredictError("reference is missing prenatal metadata (ccnv.prenatal)")
        prenatal_result = predict_prenatal(
            raw_sample,
            inputs.reference,
            prenatal_meta,
            min_ff_warning=config.min_fetal_fraction_warning,
        )
        weighted_sample = apply_prenatal_fisher_weighting(raw_sample, prenatal_meta)
        if weighted_sample is not raw_sample:
            inputs = replace(inputs, sample=weighted_sample)

    bin_results = tuple(
        normalize_branch(build_branch_model(inputs, branch), config)
        for branch in inputs.branches
    )
    _require_aligned_results(bin_results)
    segments = segment_and_call(
        bin_results,
        inputs.contig_lengths,
        penalty=config.penalty,
        z_threshold=config.z_threshold,
        ratio_threshold=config.ratio_threshold,
        max_blacklist_gap=config.max_blacklist_gap,
        threads=config.threads,
        mode=config.mode,
    )
    if config.mode == "prenatal" and prenatal_result is not None:
        annotated_segs, lod_metrics = annotate_prenatal_origin_and_lod(
            segments,
            sample_table=raw_sample,
            prenatal_meta=prenatal_meta,
            prenatal_result=prenatal_result,
            z_threshold=config.z_threshold,
        )
        segments = tuple(annotated_segs)
        prenatal_result = replace(prenatal_result, lod_metrics=lod_metrics)
    qc_json_path = Path(f"{config.prefix}_qc.json")
    write_predict_qc_json(
        qc_json_path,
        build_predict_qc_payload(
            mode=config.mode,
            sample_table=raw_sample,
            segments=segments,
            prenatal_result=prenatal_result,
            bin_results=bin_results,
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
        bins_vcf=config.bins_vcf,
        threads=config.threads,
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
    threads: int = 1,
) -> PredictionInput:
    """Load and validate sample and selective reference branches against each other."""
    try:
        ref_schema = pq.read_schema(reference_path)
        validate_schema(ref_schema, ArtifactKind.REFERENCE)
        ref_meta_obj = ReferenceMetadata.from_schema(ref_schema)
        raw_ref_meta = decode_metadata(ref_schema.metadata)
    except Exception as err:
        raise PredictError(f"failed to read reference {reference_path}: {err}") from err

    if sample_path.suffix.lower() in {".bam", ".cram"}:
        extracted = extract_for_predict(
            sample_path,
            ref_meta_obj,
            threads=threads,
            fasta=fasta,
            collect_bands=(mode == "prenatal"),
        )
        sample: pa.Table | SampleBuffers = extracted_to_sample_buffers(extracted)
    else:
        try:
            sample = read_sample_parquet(sample_path)
        except Exception as err:
            raise PredictError(f"failed to read sample {sample_path}: {err}") from err

    if mode == "prenatal" and ref_meta_obj.mode != "prenatal":
        msg = (
            "reference was built in germline mode and lacks prenatal models; "
            "rebuild reference with --mode prenatal or run predict with --mode germline"
        )
        raise PredictError(msg)
    if mode == "germline" and ref_meta_obj.mode == "prenatal":
        logger.warning(
            "reference was built in prenatal mode; "
            "prenatal models are ignored in germline mode"
        )

    branch_vals: set[object] = set(ref_meta_obj.branches)
    if not branch_vals:
        try:
            b_tbl = pq.read_table(reference_path, columns=["branch"])
            branch_vals = {
                str(v)
                for v in pc.unique(b_tbl.column("branch")).to_pylist()
                if v is not None
            }
        except Exception as err:
            raise PredictError(
                f"failed to read reference {reference_path}: {err}"
            ) from err

    if BRANCH_AUTOSOMAL not in branch_vals:
        raise PredictError("reference has no autosomal branch")

    if mode == "prenatal":
        branches = [BRANCH_AUTOSOMAL] + (
            [BRANCH_FEMALE_SEX] if BRANCH_FEMALE_SEX in branch_vals else []
        )
        chosen_sex: Sex | None = "female" if BRANCH_FEMALE_SEX in branch_vals else None
        read_branches = [
            b
            for b in (*branches, BRANCH_PRENATAL_AUTOSOMAL, BRANCH_PRENATAL_FFY)
            if b in branch_vals
        ]
    else:
        chosen_sex = _resolve_sample_sex(sex, sample, raw_ref_meta, branch_vals)
        branches = [BRANCH_AUTOSOMAL]
        if chosen_sex is not None:
            sex_branch = (
                BRANCH_FEMALE_SEX if chosen_sex == "female" else BRANCH_MALE_SEX
            )
            if sex_branch not in branch_vals and sex != "autosomal":
                raise PredictError(f"reference has no {sex_branch!r} branch")
            if sex_branch in branch_vals:
                branches.append(sex_branch)
        read_branches = list(branches)

    sample_meta_obj = (
        sample.meta
        if isinstance(sample, SampleBuffers)
        else validate_sample_table(sample)
    )
    if sample_meta_obj.bin_size != ref_meta_obj.bin_size:
        logger.debug(
            "rebinning sample from %d bp to reference bin size %d bp",
            sample_meta_obj.bin_size,
            ref_meta_obj.bin_size,
        )
        try:
            if isinstance(sample, SampleBuffers):
                sample = rebin_sample_buffers(sample, ref_meta_obj.bin_size)
                sample_meta_obj = sample.meta
            else:
                sample = rebin_sample_sketch(sample, ref_meta_obj.bin_size)
                sample_meta_obj = validate_sample_table(sample)
        except BinCompatibilityError as err:
            raise PredictError(str(err)) from err

    try:
        reference, _ = read_reference_parquet(
            reference_path, branches=read_branches, check_rows=False
        )
    except Exception as err:
        raise PredictError(f"failed to read reference {reference_path}: {err}") from err

    resolve_fragmentomics_compatibility(sample, reference)
    active_ref = reference.filter(
        pc.is_in(reference.column("branch"), pa.array(branches, type=pa.string()))
    )
    active_contigs = unique_dictionary_contigs(active_ref.column("contig"))
    sample_lens = sample_meta_obj.contig_lengths
    ref_lens = dict(ref_meta_obj.contig_lengths)
    missing_contigs = [c for c in active_contigs if c not in sample_lens]
    if missing_contigs:
        raise PredictError(
            f"sample is missing reference bins on contig(s): {', '.join(missing_contigs)}"
        )
    for contig_name in active_contigs:
        if sample_lens.get(contig_name) != ref_lens.get(contig_name):
            raise PredictError(f"contig length mismatch for {contig_name!r}")

    return PredictionInput(
        sample=sample,
        reference=reference,
        contig_lengths=ref_lens,
        sex=chosen_sex,
        branches=tuple(branches),
    )


@overload
def build_branch_model(inputs: PredictionInput, branch: str) -> BranchModel: ...


@overload
def build_branch_model(
    inputs: pa.Table,
    branch: str,
    ref_meta: ReferenceMetadata | None = None,
) -> BranchModel: ...


def build_branch_model(
    inputs: PredictionInput | pa.Table,
    branch: str,
    ref_meta: ReferenceMetadata | None = None,
) -> BranchModel:
    """Build dense NumPy arrays for one reference branch."""
    if isinstance(inputs, PredictionInput):
        return build_branch_model_from_tables(
            inputs.reference, branch, sample_data=inputs.sample
        )
    return build_branch_model_from_tables(inputs, branch, ref_meta=ref_meta)


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


_compute_contig_rho1 = compute_contig_rho1
_evaluate_stouffer_segment_scores = evaluate_stouffer_segment_scores


def segment_and_call(
    bins: tuple[BinResult, ...],
    contig_lengths: dict[str, int],
    *,
    penalty: float,
    z_threshold: float,
    ratio_threshold: float | None = None,
    max_blacklist_gap: int = 2_000_000,
    threads: int = 1,
    mode: AnalysisMode = "germline",
) -> tuple[SegmentResult, ...]:
    """Segment all selected branches and classify with calibrated dyadic BLUE scores."""
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
            NDArray[np.float64],
            NDArray[np.float64] | None,
            NDArray[np.float64] | None,
        ],
    ] = {}

    for result in bins:
        contig_array = np.asarray(result.contigs, dtype=object)
        for contig_name in dict.fromkeys(result.contigs):
            selected = contig_array == contig_name
            c_starts = result.starts[selected]
            c_ends = result.ends[selected]
            c_ratios = result.ratios[selected]
            c_weights = result.weights[selected]
            c_zscores = result.zscores[selected]
            c_counts = result.counts[selected] if result.counts is not None else None
            c_expected = (
                result.expected_counts[selected]
                if result.expected_counts is not None
                else None
            )

            by_contig[contig_name] = (
                c_starts,
                c_ends,
                c_ratios,
                c_weights,
                contig_lengths[contig_name],
                c_zscores,
                c_counts,
                c_expected,
            )

    scored_segments = segment_and_score_genome(
        by_contig,
        penalty=None,
        sensitivity=penalty,
        algorithm="pelt",
        z_threshold=z_threshold,
        ratio_threshold=ratio_threshold,
        max_blacklist_gap=max_blacklist_gap,
        threads=threads,
        mode=mode,
    )

    output: list[SegmentResult] = []
    for s in scored_segments:
        contig = s.contig
        start = s.start
        end = min(s.end, contig_lengths[contig])
        ratio = s.mean
        seg_zscore = s.zscore
        weight = s.weight
        bin_count = s.bin_count
        call = s.call.lower() if s.call in ("gain", "loss", "GAIN", "LOSS") else None
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
                copy_number=s.copy_number,
                log_bf=s.log_bf,
                q_value=s.q_value,
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
    sample: pa.Table | SampleBuffers,
    metadata: dict[str, object],
    branches: set[object],
) -> Sex | None:
    """Infer sample genetic sex from non-PAR sex-signal coverage against reference cutoff."""
    if requested != "autosomal":
        return requested
    return infer_single_sample_sex(sample, metadata, branches)


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


def _require_aligned_results(results: tuple[BinResult, ...]) -> None:
    """Ensure all per-branch BinResult arrays are 1D aligned."""
    for res in results:
        _require_aligned(
            res.starts.size, res.ratios, res.zscores, res.weights, res.reference_sizes
        )


run_predict: Final = run
