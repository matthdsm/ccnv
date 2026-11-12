"""Feature transformation kernels for cell-free DNA fragmentomic signals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from ccnv.models import (
    FRAGMENTOMICS_METADATA_KEY,
    FragmentomicsMetadata,
    PredictError,
    PrenatalModelMetadata,
    ReferenceMetadata,
    SampleBuffers,
    SchemaError,
    decode_metadata,
)
from ccnv.stats.fragmentomics import (
    FRAG_NUM_BINS,
    align_histogram,
    build_fragmentomic_feature_vector,
    compute_band_fisher_weights,
    deconvolve_fetal_maternal_densities,
    evaluate_band_weighted_counts,
    fit_ff_fragmentomics_ensemble,
)

__all__ = [
    "CohortFragmentomicsFit",
    "apply_prenatal_fisher_weighting",
    "extract_sample_fragmentomics",
    "fit_cohort_fragmentomics_model",
    "has_paired_fragmentomics",
    "resolve_fisher_moment_weights",
    "resolve_fragmentomics_compatibility",
]


@dataclass(frozen=True, slots=True)
class CohortFragmentomicsFit:
    """Fitted cohort fetal/maternal fragment densities and FF_size logit ensemble."""

    p_fetal_density: list[float]
    p_maternal_density: list[float]
    ff_frag_weights: list[float] | None
    ff_frag_bias: float | None
    ff_frag_feature_mean: list[float] | None
    ff_frag_feature_scale: list[float] | None
    ff_size_std: float


def extract_sample_fragmentomics(
    metadata: Mapping[str, object] | Mapping[bytes, bytes] | None,
) -> FragmentomicsMetadata | None:
    """Parse validated FragmentomicsMetadata from decoded or Arrow schema metadata."""
    if not metadata:
        return None
    if FRAGMENTOMICS_METADATA_KEY in metadata:
        raw_bytes = {
            k: v
            for k, v in metadata.items()
            if isinstance(k, bytes) and isinstance(v, bytes)
        }
        val: object = decode_metadata(raw_bytes).get("fragmentomics")
    else:
        val = metadata.get("fragmentomics")
    if val is None:
        return None
    if isinstance(val, FragmentomicsMetadata):
        return val
    if not isinstance(val, Mapping):
        raise SchemaError("ccnv.fragmentomics metadata must be a mapping")
    return FragmentomicsMetadata.from_dict({str(k): v for k, v in val.items()})


def fit_cohort_fragmentomics_model(
    frag_metas: Sequence[FragmentomicsMetadata | None],
    is_male_arr: NDArray[np.bool_],
    ff_targets: NDArray[np.floating],
    *,
    quarantined: NDArray[np.bool_] | None = None,
    seed: int = 42,
) -> CohortFragmentomicsFit | None:
    """Deconvolve cohort fragment length densities and fit FF_size regularized ensemble."""
    n_samples = len(frag_metas)
    if n_samples == 0:
        return None

    male_mask = np.asarray(is_male_arr, dtype=np.bool_)
    targets = np.asarray(ff_targets, dtype=np.float64)
    clean_mask = (
        ~np.asarray(quarantined, dtype=np.bool_)
        if quarantined is not None
        else np.ones(n_samples, dtype=np.bool_)
    )
    clean_idx = np.flatnonzero(clean_mask)
    if clean_idx.size == 0:
        return None

    clean_metas: list[FragmentomicsMetadata] = []
    for idx in clean_idx:
        meta = frag_metas[int(idx)]
        if meta is None or not meta.paired_end:
            return None
        clean_metas.append(meta)

    n_clean = len(clean_metas)
    auto_aligned = np.zeros((n_clean, FRAG_NUM_BINS), dtype=np.float64)
    chry_aligned = np.zeros((n_clean, FRAG_NUM_BINS), dtype=np.float64)
    for i, meta in enumerate(clean_metas):
        auto_aligned[i] = align_histogram(
            np.asarray(meta.hist_autosomal, dtype=np.float64),
            meta.sizing_shift_bp,
        )
        chry_aligned[i] = align_histogram(
            np.asarray(meta.hist_chry, dtype=np.float64),
            meta.sizing_shift_bp,
        )

    targets_clean = targets[clean_idx]
    male_clean = male_mask[clean_idx]
    p_fetal, p_maternal = deconvolve_fetal_maternal_densities(
        auto_aligned,
        chry_aligned,
        targets_clean,
        male_clean,
    )

    x_frag = np.zeros((n_clean, 36), dtype=np.float64)
    for i, meta in enumerate(clean_metas):
        x_frag[i] = build_fragmentomic_feature_vector(
            auto_aligned[i],
            meta.end_motif_counts,
            chromatosome_mode_bp=meta.chromatosome_mode_bp,
            sizing_shift_bp=0.0,
            helical_10bp_amp=meta.helical_10bp_amplitude,
            motif_mds=meta.motif_diversity_score,
            reference_maternal_571=p_maternal,
        )

    finite_mask = np.isfinite(targets_clean)
    p_fet_list = [float(x) for x in p_fetal]
    p_mat_list = [float(x) for x in p_maternal]
    if int(np.count_nonzero(finite_mask)) < 5:
        return CohortFragmentomicsFit(
            p_fetal_density=p_fet_list,
            p_maternal_density=p_mat_list,
            ff_frag_weights=None,
            ff_frag_bias=None,
            ff_frag_feature_mean=None,
            ff_frag_feature_scale=None,
            ff_size_std=0.018,
        )

    ens = fit_ff_fragmentomics_ensemble(
        x_frag[finite_mask],
        targets_clean[finite_mask],
        seed=seed,
    )
    return CohortFragmentomicsFit(
        p_fetal_density=p_fet_list,
        p_maternal_density=p_mat_list,
        ff_frag_weights=[float(x) for x in ens.weights],
        ff_frag_bias=float(ens.intercept),
        ff_frag_feature_mean=[float(x) for x in ens.feat_mean],
        ff_frag_feature_scale=[float(x) for x in ens.feat_std],
        ff_size_std=float(ens.oof_rmse),
    )


def apply_prenatal_fisher_weighting(
    sample: SampleBuffers,
    prenatal_meta: PrenatalModelMetadata | None,
) -> SampleBuffers:
    """Evaluate 24-band Fisher-optimal weighted coverage and attach per-bin Kish factor."""
    bufs = SampleBuffers.coerce(sample)
    if (
        prenatal_meta is None
        or prenatal_meta.band_fisher_weights is None
        or bufs.num_rows == 0
        or bufs.frag_bands is None
        or not np.any(bufs.paired_count > 0)
    ):
        return bufs

    w_24 = np.asarray(prenatal_meta.band_fisher_weights, dtype=np.float64)
    w_counts, kappa = evaluate_band_weighted_counts(
        bufs.frag_bands, bufs.count, bufs.paired_count, w_24
    )
    return replace(bufs, count=w_counts, kappa_frag=kappa)


def resolve_fisher_moment_weights(
    sample: SampleBuffers,
    prenatal_meta: PrenatalModelMetadata | None,
) -> tuple[float, float]:
    """Resolve (mu_fetal_w, mu_maternal_w) matching apply_prenatal_fisher_weighting eligibility."""
    bufs = SampleBuffers.coerce(sample)
    if (
        prenatal_meta is None
        or prenatal_meta.band_fisher_weights is None
        or bufs.num_rows == 0
        or bufs.frag_bands is None
        or not np.any(bufs.paired_count > 0)
    ):
        return 1.0, 1.0
    if prenatal_meta.mu_fetal_w is not None and prenatal_meta.mu_maternal_w is not None:
        return (
            max(float(prenatal_meta.mu_fetal_w), 1e-6),
            max(float(prenatal_meta.mu_maternal_w), 1e-6),
        )
    p_fet = prenatal_meta.p_fetal_density
    p_mat = prenatal_meta.p_maternal_density
    if (
        p_fet is not None
        and p_mat is not None
        and len(p_fet) == FRAG_NUM_BINS == len(p_mat)
    ):
        _, mf, mm = compute_band_fisher_weights(p_fet, p_mat)
        return max(float(mf), 1e-6), max(float(mm), 1e-6)
    return 1.0, 1.0


def has_paired_fragmentomics(sample: SampleBuffers) -> bool:
    """Return whether a sample sketch carries paired-end fragmentomic observations."""
    bufs = SampleBuffers.coerce(sample)
    if bufs.meta.fragmentomics is not None:
        return bool(bufs.meta.fragmentomics.paired_end)
    return bufs.frag_bands is not None and (
        bool(np.any(bufs.paired_count > 0)) or bool(np.any(bufs.frag_bands > 0))
    )


def resolve_fragmentomics_compatibility(
    sample: SampleBuffers,
    reference: ReferenceMetadata | pa.Table,
) -> bool:
    """Validate sample vs reference fragmentomic correction settings."""
    bufs = SampleBuffers.coerce(sample)
    sample_has_frag = has_paired_fragmentomics(bufs)

    ref_meta = (
        reference
        if isinstance(reference, ReferenceMetadata)
        else ReferenceMetadata.from_table(reference)
    )
    if ref_meta.flat:
        return sample_has_frag
    if ref_meta.fragmentomics_corrected != sample_has_frag:
        raise PredictError(
            f"sample fragmentomics ({'paired-end' if sample_has_frag else 'single-end'}) "
            f"does not match cohort reference fragmentomics_corrected="
            f"{ref_meta.fragmentomics_corrected}; "
            "use matching library types or rebuild the reference"
        )
    return ref_meta.fragmentomics_corrected
