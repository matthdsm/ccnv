"""Typed copy-number normalization, segmentation, and calling."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Literal, overload

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from ccnv.extract import extract_for_predict
from ccnv.features.binning import align_sample_to_grid_indices
from ccnv.features.covariates import validate_min_mappability
from ccnv.features.fragmentomics import (
    apply_prenatal_fisher_weighting,
    resolve_fragmentomics_compatibility,
)
from ccnv.features.normalization import (
    BranchModel,
    BranchNormalizationResult,
    align_reference_unmasked_bins,
    build_branch_model_from_tables,
    normalize_branch,
    normalize_sample_branch,
)
from ccnv.features.prenatal import normalize_prenatal_chry, predict_prenatal
from ccnv.features.prenatal_origin import annotate_sample_prenatal_segments
from ccnv.features.rebinning import rebin_sample_buffers
from ccnv.features.sex import infer_single_sample_sex
from ccnv.io.parquet import (
    filter_reference_branches,
    read_reference_parquet,
    read_sample_buffers,
)
from ccnv.io.tracks import (
    load_cytobands_track,
    load_default_cytobands_track,
    load_gc_track,
    load_mappability_track,
)
from ccnv.models import (
    BRANCH_AUTOSOMAL,
    BRANCH_FEMALE_SEX,
    BRANCH_MALE_SEX,
    BRANCH_PRENATAL_AUTOSOMAL,
    BRANCH_PRENATAL_FFY,
    NUM_GC_QUANTILES,
    AnalysisMode,
    BinCompatibilityError,
    BinResult,
    CcnvError,
    ContigLookup,
    CovariateError,
    Cytoband,
    FetalFractionResult,
    InvalidInputError,
    PredictConfig,
    PredictError,
    PredictionOutputs,
    PrenatalModelMetadata,
    ReferenceMetadata,
    SampleBuffers,
    SchemaError,
    SegmentResult,
    reconstruct_sample_grid,
    unique_dictionary_contigs,
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
    ref_meta: ReferenceMetadata | None = None


def _preload_sample_track_overrides(
    sample: SampleBuffers,
    reference: pa.Table,
    branches: tuple[str, ...],
    *,
    gc_path: Path | None,
    mappability_path: Path | None,
    mode: AnalysisMode,
) -> tuple[
    NDArray[np.float64] | None,
    NDArray[np.float32] | None,
    NDArray[np.bool_] | None,
    NDArray[np.float64] | None,
]:
    """Load optional --gc and --mappability tracks once onto the sample grid."""
    if gc_path is None and mappability_path is None:
        return None, None, None, None

    track_branches = (
        (*branches, BRANCH_PRENATAL_FFY) if mode == "prenatal" else branches
    )
    sub_ref = filter_reference_branches(reference, track_branches)
    ref_contigs = unique_dictionary_contigs(sub_ref.column("contig"))
    lookup = ContigLookup.from_names(sample.meta.contig_order)
    target_contigs: set[str] = set()
    for c_name in ref_contigs:
        if c_name in sample.meta.contig_lengths:
            target_contigs.add(c_name)
        elif mode == "prenatal":
            resolved = lookup.resolve(c_name)
            if resolved is not None:
                target_contigs.add(resolved)

    codes, names, starts_u32, ends_u32 = reconstruct_sample_grid(sample.meta)
    keep_codes = [i for i, name in enumerate(names) if name in target_contigs]
    q_mask = (
        np.isin(codes, keep_codes)
        if keep_codes
        else np.zeros(sample.num_rows, dtype=np.bool_)
    )
    q_contigs = [names[int(c)] for c in codes[q_mask]]
    q_starts = starts_u32[q_mask].astype(np.int64, copy=False)
    q_ends = ends_u32[q_mask].astype(np.int64, copy=False)

    n_rows = sample.num_rows
    sample_gc: NDArray[np.float64] | None = None
    sample_gc_q: NDArray[np.float32] | None = None
    sample_n_mask: NDArray[np.bool_] | None = None
    sample_map: NDArray[np.float64] | None = None
    try:
        if gc_path is not None:
            gc_sub, n_sub, gc_q_sub = load_gc_track(
                gc_path, q_contigs, q_starts, q_ends
            )
            sample_gc = np.full(n_rows, np.nan, dtype=np.float64)
            sample_gc[q_mask] = gc_sub
            sample_n_mask = np.ones(n_rows, dtype=np.bool_)
            sample_n_mask[q_mask] = n_sub
            if gc_q_sub is not None:
                sample_gc_q = np.zeros((n_rows, NUM_GC_QUANTILES), dtype=np.float32)
                sample_gc_q[q_mask] = gc_q_sub
        if mappability_path is not None:
            map_sub = load_mappability_track(
                mappability_path, q_contigs, q_starts, q_ends
            )
            sample_map = np.zeros(n_rows, dtype=np.float64)
            sample_map[q_mask] = map_sub
    except CovariateError as err:
        raise PredictError(str(err)) from err
    return sample_gc, sample_gc_q, sample_n_mask, sample_map


def _normalize_branches_once(
    inputs: PredictionInput,
    config: PredictConfig,
    *,
    sample_gc: NDArray[np.float64] | None = None,
    sample_gc_q: NDArray[np.float32] | None = None,
    sample_n_mask: NDArray[np.bool_] | None = None,
    sample_map: NDArray[np.float64] | None = None,
    tracks_loaded: bool = False,
) -> tuple[BinResult, ...]:
    models = tuple(build_branch_model(inputs, branch) for branch in inputs.branches)
    if not models:
        return ()
    if not tracks_loaded and (config.gc is not None or config.mappability is not None):
        sample_gc, sample_gc_q, sample_n_mask, sample_map = (
            _preload_sample_track_overrides(
                inputs.sample,
                inputs.reference,
                inputs.branches,
                gc_path=config.gc,
                mappability_path=config.mappability,
                mode=config.mode,
            )
        )
    if sample_gc is None and sample_map is None:
        return tuple(normalize_branch(model, config) for model in models)

    results: list[BinResult] = []
    for model in models:
        s_idx = align_sample_to_grid_indices(
            inputs.sample.meta,
            model.contigs,
            model.starts,
            model.ends,
            strict_branch=model.branch,
        )
        results.append(
            normalize_branch(
                model,
                config,
                gc_override=sample_gc[s_idx] if sample_gc is not None else None,
                gc_quantiles_override=(
                    sample_gc_q[s_idx] if sample_gc_q is not None else None
                ),
                n_mask_override=(
                    sample_n_mask[s_idx] if sample_n_mask is not None else None
                ),
                mappability_override=(
                    sample_map[s_idx] if sample_map is not None else None
                ),
            )
        )
    return tuple(results)


def _sample_usable_from_bin_results(
    raw_sample: SampleBuffers,
    bin_results: tuple[BinResult, ...],
) -> NDArray[np.bool_] | None:
    if raw_sample.num_rows == 0 or not bin_results:
        return None
    usable = np.zeros(raw_sample.num_rows, dtype=np.bool_)
    for br in bin_results:
        if br.starts.size == 0:
            continue
        s_idx = align_sample_to_grid_indices(
            raw_sample.meta, br.contigs, br.starts, br.ends, allow_alias=True
        )
        hit = (s_idx >= 0) & ~br.masked
        usable[s_idx[hit]] = True
    return usable


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
    cytobands_track: tuple[Cytoband, ...] | None = None
    if config.plot:
        try:
            cytobands_track = (
                load_cytobands_track(config.cytobands, inputs.contig_lengths)
                if config.cytobands is not None
                else load_default_cytobands_track(inputs.contig_lengths)
            )
        except CovariateError as err:
            raise PredictError(str(err)) from err

    raw_sample = inputs.sample
    sample_gc, sample_gc_q, sample_n_mask, sample_map = _preload_sample_track_overrides(
        raw_sample,
        inputs.reference,
        inputs.branches,
        gc_path=config.gc,
        mappability_path=config.mappability,
        mode=config.mode,
    )

    prenatal_meta: PrenatalModelMetadata | None = None
    prenatal_result: FetalFractionResult | None = None
    if config.mode == "prenatal":
        try:
            prenatal_meta = (
                inputs.ref_meta.prenatal
                if inputs.ref_meta is not None and inputs.ref_meta.prenatal is not None
                else PrenatalModelMetadata.from_table(inputs.reference)
            )
            if prenatal_meta is None:
                raise PredictError(
                    "reference is missing prenatal metadata (ccnv.prenatal)"
                )
            prenatal_result = predict_prenatal(
                raw_sample,
                inputs.reference,
                prenatal_meta,
                min_ff_warning=config.min_fetal_fraction_warning,
                sample_gc_override=sample_gc,
                sample_mappability_override=sample_map,
                sample_n_mask_override=sample_n_mask,
                min_mappability=config.min_mappability,
                twins=config.twins,
            )
            weighted_sample = apply_prenatal_fisher_weighting(raw_sample, prenatal_meta)
        except (SchemaError, ValueError) as err:
            raise PredictError(str(err)) from err
        fetal_sex: Sex = "male" if prenatal_result.fetal_sex == "M" else "female"
        inputs = replace(inputs, sample=weighted_sample, sex=fetal_sex)

    bin_results = _normalize_branches_once(
        inputs,
        config,
        sample_gc=sample_gc,
        sample_gc_q=sample_gc_q,
        sample_n_mask=sample_n_mask,
        sample_map=sample_map,
        tracks_loaded=True,
    )
    if (
        config.mode == "prenatal"
        and prenatal_meta is not None
        and prenatal_result is not None
        and prenatal_result.fetal_sex == "M"
    ):
        try:
            chry_bins = normalize_prenatal_chry(
                raw_sample,
                inputs.reference,
                prenatal_meta,
                prenatal_result,
                sample_gc_override=sample_gc,
                sample_mappability_override=sample_map,
                sample_n_mask_override=sample_n_mask,
                min_mappability=config.min_mappability,
            )
        except (SchemaError, ValueError) as err:
            raise PredictError(str(err)) from err
        if chry_bins is not None:
            bin_results = (*bin_results, chry_bins)
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
        try:
            segments, lod_metrics = annotate_sample_prenatal_segments(
                segments,
                raw_sample,
                bin_usable=sample_usable,
                prenatal_meta=prenatal_meta,
                prenatal_result=prenatal_result,
                z_threshold=config.z_threshold,
            )
        except (SchemaError, ValueError) as err:
            raise PredictError(str(err)) from err
        prenatal_result = replace(prenatal_result, lod_metrics=lod_metrics)

    qc_json_path = Path(f"{config.prefix}_qc.json")
    try:
        write_predict_qc_json(
            qc_json_path,
            build_predict_qc_payload(
                mode=config.mode,
                sample_table=raw_sample,
                segments=segments,
                prenatal_result=prenatal_result,
                bin_results=bin_results,
                sex=inputs.sex,
            ),
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
    except OSError as err:
        raise PredictError(
            f"failed to write prediction outputs for {config.prefix}: {err}"
        ) from err


def _validate_branch_contigs(
    reference: pa.Table,
    branches: list[str],
    sample_lens: dict[str, int],
    ref_lens: dict[str, int],
    *,
    allow_alias: bool = False,
) -> None:
    active_ref = filter_reference_branches(reference, branches)
    active_contigs = unique_dictionary_contigs(active_ref.column("contig"))
    lookup = ContigLookup.from_names(sample_lens.keys()) if allow_alias else None
    missing_contigs: list[str] = []
    resolved_pairs: list[tuple[str, str]] = []
    for c_name in active_contigs:
        if c_name in sample_lens:
            resolved_pairs.append((c_name, c_name))
        elif lookup is not None and (alias := lookup.resolve(c_name)) is not None:
            resolved_pairs.append((c_name, alias))
        else:
            missing_contigs.append(c_name)
    if missing_contigs:
        raise PredictError(
            f"sample is missing reference bins on contig(s): {', '.join(missing_contigs)}"
        )
    for ref_name, sample_name in resolved_pairs:
        if sample_lens.get(sample_name) != ref_lens.get(ref_name):
            raise PredictError(f"contig length mismatch for {ref_name!r}")


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
        _, ref_meta_obj = read_reference_parquet(
            reference_path, branches=(), check_rows=False
        )
    except (OSError, pa.ArrowException, CcnvError, ValueError) as err:
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
        except (OSError, pa.ArrowException, CcnvError, ValueError) as err:
            raise PredictError(f"failed to read sample {sample_path}: {err}") from err

    if mode == "prenatal" and ref_meta_obj.mode != "prenatal":
        raise PredictError(
            "reference was built in germline mode and lacks prenatal models; "
            "rebuild reference with --mode prenatal or run predict with --mode germline"
        )
    if mode == "germline" and ref_meta_obj.mode == "prenatal":
        logger.warning(
            "reference was built in prenatal mode; "
            "prenatal models are ignored in germline mode"
        )

    branch_vals = set(ref_meta_obj.branches)
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
    elif sex != "autosomal":
        chosen_sex = infer_single_sample_sex(sample, ref_meta_obj, sex=sex)
        sex_branch = BRANCH_FEMALE_SEX if chosen_sex == "female" else BRANCH_MALE_SEX
        if sex_branch not in branch_vals:
            raise PredictError(f"reference has no {sex_branch!r} branch")
        branches = [BRANCH_AUTOSOMAL, sex_branch]
        read_branches = list(branches)
    else:
        chosen_sex = None
        branches = [BRANCH_AUTOSOMAL]
        read_branches = [
            b
            for b in (BRANCH_AUTOSOMAL, BRANCH_FEMALE_SEX, BRANCH_MALE_SEX)
            if b in branch_vals
        ]

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
    except (OSError, pa.ArrowException, CcnvError, ValueError) as err:
        raise PredictError(f"failed to read reference {reference_path}: {err}") from err

    resolve_fragmentomics_compatibility(sample, ref_meta_obj)
    sample_lens = dict(sample.meta.contig_lengths)
    ref_lens = dict(ref_meta_obj.contig_lengths)

    if mode == "germline" and sex == "autosomal":
        _validate_branch_contigs(reference, [BRANCH_AUTOSOMAL], sample_lens, ref_lens)
        for c_name in unique_dictionary_contigs(reference.column("contig")):
            if c_name in sample_lens and sample_lens[c_name] != ref_lens.get(c_name):
                raise PredictError(f"contig length mismatch for {c_name!r}")

        unmasked_sample_bins = align_reference_unmasked_bins(sample, reference)

        chosen_sex = infer_single_sample_sex(
            sample,
            ref_meta_obj,
            sex=None,
            unmasked_sample_bins=unmasked_sample_bins,
        )
        if chosen_sex is not None:
            sex_branch = (
                BRANCH_FEMALE_SEX if chosen_sex == "female" else BRANCH_MALE_SEX
            )
            if sex_branch in branch_vals:
                branches.append(sex_branch)
        if set(branches) != set(read_branches):
            reference = filter_reference_branches(reference, branches)

    _validate_branch_contigs(reference, branches, sample_lens, ref_lens)
    if mode == "prenatal":
        _validate_branch_contigs(
            reference, read_branches, sample_lens, ref_lens, allow_alias=True
        )

    return PredictionInput(
        sample=sample,
        reference=reference,
        contig_lengths=ref_lens,
        sex=chosen_sex,
        branches=tuple(branches),
        ref_meta=ref_meta_obj,
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
            inputs.reference,
            branch,
            sample=inputs.sample,
            ref_meta=inputs.ref_meta,
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
        is_ffy = result.branch == BRANCH_PRENATAL_FFY
        ploidy = 1 if (result.branch == BRANCH_MALE_SEX or is_ffy) else 2
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
                background_expected=(
                    result.background_counts[selected]
                    if result.background_counts is not None
                    else None
                ),
                baseline_ploidy=ploidy,
                baseline_log2_var=result.baseline_log2_var,
                poisson_pooled=is_ffy,
            )

    try:
        output = segment_and_score_genome(
            by_contig,
            penalty=None,
            sensitivity=penalty,
            z_threshold=z_threshold,
            ratio_threshold=ratio_threshold,
            max_blacklist_gap=max_blacklist_gap,
            threads=threads,
            mode=mode,
        )
    except InvalidInputError as err:
        raise PredictError(str(err)) from err

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
        raise PredictError(
            "min_reference_bins and mask_repeats must be positive, "
            "max_blacklist_gap non-negative"
        )
    try:
        validate_min_mappability(config.min_mappability)
    except CovariateError as err:
        raise PredictError(str(err)) from err


run_predict: Final = run
