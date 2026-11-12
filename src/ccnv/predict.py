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

from ccnv.extract import extract_for_predict
from ccnv.features.covariates import validate_min_mappability
from ccnv.features.fragmentomics import (
    apply_prenatal_fisher_weighting,
    resolve_fragmentomics_compatibility,
)
from ccnv.features.normalization import (
    BranchModel,
    BranchNormalizationResult,
    build_branch_model_from_tables,
    normalize_branch,
    normalize_sample_branch,
)
from ccnv.features.prenatal import predict_prenatal
from ccnv.features.prenatal_origin import annotate_sample_prenatal_segments
from ccnv.features.rebinning import rebin_sample_buffers
from ccnv.features.sex import infer_single_sample_sex
from ccnv.io.parquet import (
    read_reference_parquet,
    read_sample_buffers,
)
from ccnv.io.tracks import (
    load_cytobands_track,
    load_gc_track,
    load_mappability_track,
)
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
    normalize_chr,
    reconstruct_sample_grid,
    unique_dictionary_contigs,
    validate_schema,
)
from ccnv.qc.predict import build_predict_qc_payload, write_predict_qc_json
from ccnv.report import write_outputs
from ccnv.stats.segmentation import (
    ContigSegmentationInput,
    segment_and_score_genome,
)

Sex = Literal["female", "male"]
logger = logging.getLogger(__name__)

__all__ = [
    "BranchModel",
    "BranchNormalizationResult",
    "PredictError",
    "PredictionInput",
    "SampleBuffers",
    "build_branch_model",
    "extract_for_predict",
    "load_inputs",
    "normalize_branch",
    "normalize_sample_branch",
    "run",
    "run_predict",
    "segment_and_call",
]


@dataclass(frozen=True, slots=True)
class PredictionInput:
    """Loaded sample buffers and reference branches ready for normalization."""

    sample: SampleBuffers
    reference: pa.Table
    contig_lengths: dict[str, int]
    sex: Sex | None
    branches: tuple[str, ...]


def _normalize_branches_once(
    inputs: PredictionInput,
    config: PredictConfig,
) -> tuple[BinResult, ...]:
    models = tuple(build_branch_model(inputs, branch) for branch in inputs.branches)
    if not models or (config.gc is None and config.mappability is None):
        return tuple(normalize_branch(model, config) for model in models)

    all_contigs: list[str] = []
    offsets = [0]
    for model in models:
        all_contigs.extend(model.contigs)
        offsets.append(len(all_contigs))
    all_starts = (
        np.concatenate([m.starts for m in models])
        if all_contigs
        else np.empty(0, dtype=np.int64)
    )
    all_ends = (
        np.concatenate([m.ends for m in models])
        if all_contigs
        else np.empty(0, dtype=np.int64)
    )

    try:
        gc_all, n_mask_all, gc_q_all = (
            load_gc_track(config.gc, all_contigs, all_starts, all_ends)
            if config.gc is not None
            else (None, None, None)
        )
        map_all = (
            load_mappability_track(
                config.mappability, all_contigs, all_starts, all_ends
            )
            if config.mappability is not None
            else None
        )
    except CovariateError as err:
        raise PredictError(str(err)) from err

    results: list[BinResult] = []
    for idx, model in enumerate(models):
        sl = slice(offsets[idx], offsets[idx + 1])
        results.append(
            normalize_branch(
                model,
                config,
                gc_override=gc_all[sl] if gc_all is not None else None,
                gc_quantiles_override=gc_q_all[sl] if gc_q_all is not None else None,
                n_mask_override=n_mask_all[sl] if n_mask_all is not None else None,
                mappability_override=map_all[sl] if map_all is not None else None,
            )
        )
    return tuple(results)


def _sample_usable_from_bin_results(
    raw_sample: SampleBuffers,
    bin_results: tuple[BinResult, ...],
) -> NDArray[np.bool_] | None:
    if raw_sample.num_rows == 0 or not bin_results:
        return None
    codes, names, starts_u32, _ = reconstruct_sample_grid(raw_sample.meta)
    usable_by_contig: dict[str, tuple[int, NDArray[np.bool_]]] = {}
    for br in bin_results:
        if br.starts.size == 0:
            continue
        c_arr = np.asarray(br.contigs, dtype=object)
        for c_name in dict.fromkeys(br.contigs):
            mask = c_arr == c_name
            b_starts = br.starts[mask]
            b_ends = br.ends[mask]
            ref_bs = max(1, int(np.max(b_ends - b_starts)))
            b_idx = b_starts // ref_bs
            u_grid = np.zeros(int(np.max(b_idx)) + 1, dtype=np.bool_)
            u_grid[b_idx] = ~br.masked[mask]
            usable_by_contig[normalize_chr(str(c_name))] = (ref_bs, u_grid)

    out = np.zeros(starts_u32.size, dtype=np.bool_)
    starts_i64 = starts_u32.astype(np.int64, copy=False)
    for c_code, c_name in enumerate(names):
        entry = usable_by_contig.get(normalize_chr(c_name))
        if entry is None:
            continue
        ref_bs, u_grid = entry
        s_mask = codes == c_code
        idx = starts_i64[s_mask] // ref_bs
        in_bounds = (idx >= 0) & (idx < u_grid.size)
        out[s_mask] = in_bounds & u_grid[np.clip(idx, 0, max(u_grid.size - 1, 0))]
    return out


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

    bin_results = _normalize_branches_once(inputs, config)
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
        sample_usable = _sample_usable_from_bin_results(raw_sample, bin_results)
        segments, lod_metrics = annotate_sample_prenatal_segments(
            segments,
            raw_sample,
            bin_usable=sample_usable,
            prenatal_meta=prenatal_meta,
            prenatal_result=prenatal_result,
            z_threshold=config.z_threshold,
        )
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
    """Load and validate sample buffers and selective reference branches."""
    try:
        ref_schema = pq.read_schema(reference_path)
        validate_schema(ref_schema, ArtifactKind.REFERENCE)
        ref_meta_obj = ReferenceMetadata.from_schema(ref_schema)
        raw_ref_meta = decode_metadata(ref_schema.metadata)
    except Exception as err:
        raise PredictError(f"failed to read reference {reference_path}: {err}") from err

    if sample_path.suffix.lower() in {".bam", ".cram"}:
        sample = extract_for_predict(
            sample_path,
            ref_meta_obj,
            threads=threads,
            fasta=fasta,
            collect_bands=(mode == "prenatal"),
        )
    else:
        try:
            sample = read_sample_buffers(sample_path)
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

    if sample.meta.bin_size != ref_meta_obj.bin_size:
        logger.debug(
            "rebinning sample from %d bp to reference bin size %d bp",
            sample.meta.bin_size,
            ref_meta_obj.bin_size,
        )
        try:
            sample = rebin_sample_buffers(sample, ref_meta_obj.bin_size)
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
    sample_lens = sample.meta.contig_lengths
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

    by_contig: dict[str, ContigSegmentationInput] = {}
    for result in bins:
        contig_array = np.asarray(result.contigs, dtype=object)
        for contig_name in dict.fromkeys(result.contigs):
            selected = contig_array == contig_name
            by_contig[contig_name] = ContigSegmentationInput(
                starts=result.starts[selected],
                ends=result.ends[selected],
                ratios=result.ratios[selected],
                weights=result.weights[selected],
                contig_length=contig_lengths[contig_name],
                zscores=result.zscores[selected],
                counts=result.counts[selected] if result.counts is not None else None,
                expected=(
                    result.expected_counts[selected]
                    if result.expected_counts is not None
                    else None
                ),
            )

    scored_segments = segment_and_score_genome(
        by_contig,
        penalty=None,
        sensitivity=penalty,
        z_threshold=z_threshold,
        ratio_threshold=ratio_threshold,
        max_blacklist_gap=max_blacklist_gap,
        threads=threads,
        mode=mode,
    )

    output = tuple(
        SegmentResult(
            contig=s.contig,
            start=s.start,
            end=min(s.end, contig_lengths[s.contig]),
            ratio=s.mean,
            zscore=s.zscore,
            weight=s.weight,
            bin_count=s.bin_count,
            call=s.call,
            copy_number=s.copy_number,
            log_bf=s.log_bf,
            q_value=s.q_value,
        )
        for s in scored_segments
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
    return output


def _resolve_sample_sex(
    requested: Literal["autosomal", "female", "male"],
    sample: SampleBuffers | pa.Table,
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
