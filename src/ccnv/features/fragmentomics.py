"""Feature transformation kernels for cell-free DNA fragmentomic signals."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from ccnv.models import SampleBuffers
from ccnv.models.exceptions import PredictError, SchemaError
from ccnv.models.schemas import (
    FRAGMENTOMICS_METADATA_KEY,
    FragmentomicsMetadata,
    PrenatalModelMetadata,
    decode_metadata,
)
from ccnv.stats.fragmentomics import (
    CANONICAL_CHROMATOSOME_BP,
    FRAG_NUM_BINS,
    align_histogram,
    build_fragmentomic_feature_vector,
    compute_band_fisher_weights,
    deconvolve_fetal_maternal_densities,
    evaluate_band_weighted_counts,
    fit_ff_fragmentomics_ensemble,
)

__all__ = [
    "apply_prenatal_fisher_weighting",
    "extract_bin_fragmentomic_covariates",
    "extract_sample_fragmentomics",
    "fit_cohort_fragmentomics_model",
    "is_paired_sample_metadata",
    "resolve_fragmentomics_compatibility",
]


def extract_bin_fragmentomic_covariates(
    sample: SampleBuffers | pa.Table,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute per-bin mean fragment length and cleavage endpoint protection ratio."""
    bufs = SampleBuffers.coerce(sample)
    counts = np.asarray(bufs.count, dtype=np.float64)
    paired_cnt = np.asarray(bufs.paired_count, dtype=np.float64)
    flen_sum = np.asarray(bufs.frag_len_sum, dtype=np.float64)
    ep_counts = np.asarray(bufs.endpoint_count, dtype=np.float64)

    n_rows = int(counts.shape[0])
    mean_lens = np.full(n_rows, CANONICAL_CHROMATOSOME_BP, dtype=np.float64)
    if n_rows > 0:
        has_reads = paired_cnt > 0.0
        if np.any(has_reads):
            computed = flen_sum[has_reads] / paired_cnt[has_reads]
            mean_lens[has_reads] = computed
            mean_lens[~has_reads] = float(np.median(computed))
    ep_ratios = ep_counts / np.maximum(counts, 1.0)
    return mean_lens, ep_ratios


def _lookup_frag_entry(metadata: Mapping[object, object]) -> object:
    for key in ("fragmentomics", "ccnv.fragmentomics", FRAGMENTOMICS_METADATA_KEY):
        val = metadata.get(key)
        if val is None and isinstance(key, str):
            val = metadata.get(key.encode())
        elif val is None and isinstance(key, bytes):
            val = metadata.get(key.decode())
        if val is not None:
            return val
    return None


def extract_sample_fragmentomics(
    metadata: Mapping[object, object] | None,
) -> FragmentomicsMetadata | None:
    """Extract FragmentomicsMetadata from decoded metadata mapping or schema metadata."""
    if not metadata:
        return None
    val = _lookup_frag_entry(metadata)
    if val is None:
        return None
    if isinstance(val, FragmentomicsMetadata):
        return val
    if isinstance(val, (bytes, str)):
        try:
            val = json.loads(val)
        except json.JSONDecodeError, UnicodeDecodeError:
            return None
    if isinstance(val, Mapping):
        try:
            return FragmentomicsMetadata.from_dict(val)
        except TypeError, ValueError, KeyError, SchemaError:
            return None
    return None


def fit_cohort_fragmentomics_model(
    frag_metas: Sequence[FragmentomicsMetadata | None],
    is_male_arr: NDArray[np.bool_],
    final_ffy: NDArray[np.floating],
    target_ffa: NDArray[np.floating],
) -> (
    tuple[
        list[float],
        list[float],
        list[float] | None,
        float | None,
        list[float] | None,
        list[float] | None,
        float,
    ]
    | None
):
    """Deconvolve cohort fragment length densities and fit FF_size regularized ensemble."""
    n_samples = len(frag_metas)
    if n_samples == 0:
        return None
    valid_metas = [m for m in frag_metas if m is not None and m.paired_end]
    if len(valid_metas) != n_samples:
        return None

    auto_aligned = np.zeros((n_samples, FRAG_NUM_BINS), dtype=np.float64)
    chry_aligned = np.zeros((n_samples, FRAG_NUM_BINS), dtype=np.float64)
    x_frag = np.zeros((n_samples, 36), dtype=np.float64)

    for i, meta in enumerate(valid_metas):
        auto_aligned[i] = align_histogram(
            np.asarray(meta.hist_autosomal, dtype=np.float64),
            meta.sizing_shift_bp,
        )
        chry_aligned[i] = align_histogram(
            np.asarray(meta.hist_chry, dtype=np.float64),
            meta.sizing_shift_bp,
        )
        x_frag[i] = build_fragmentomic_feature_vector(meta)

    p_fetal, p_maternal = deconvolve_fetal_maternal_densities(
        auto_aligned,
        chry_aligned,
        male_mask=is_male_arr,
        ffy_values=final_ffy,
    )

    valid_mask = np.isfinite(target_ffa)
    if int(np.sum(valid_mask)) < 5:
        return (
            [float(x) for x in p_fetal],
            [float(x) for x in p_maternal],
            None,
            None,
            None,
            None,
            0.018,
        )

    w_frag, b_frag, m_frag, s_frag, loocv_std = fit_ff_fragmentomics_ensemble(
        x_frag[valid_mask],
        target_ffa[valid_mask],
    )

    return (
        [float(x) for x in p_fetal],
        [float(x) for x in p_maternal],
        [float(x) for x in w_frag],
        float(b_frag),
        [float(x) for x in m_frag],
        [float(x) for x in s_frag],
        float(loocv_std),
    )


def apply_prenatal_fisher_weighting(
    sample: SampleBuffers | pa.Table,
    prenatal_meta: PrenatalModelMetadata | None,
) -> SampleBuffers:
    """Evaluate 24-band Fisher-optimal weighted coverage and attach per-bin Kish factor."""
    bufs = SampleBuffers.coerce(sample)
    if prenatal_meta is None or bufs.num_rows == 0 or bufs.frag_bands is None:
        return bufs
    if not np.any(bufs.paired_count > 0):
        return bufs

    if prenatal_meta.band_fisher_weights is not None:
        w_24 = np.asarray(prenatal_meta.band_fisher_weights, dtype=np.float64)
    elif (
        prenatal_meta.p_fetal_density is not None
        and prenatal_meta.p_maternal_density is not None
    ):
        w_24, _, _ = compute_band_fisher_weights(
            np.asarray(prenatal_meta.p_fetal_density, dtype=np.float64),
            np.asarray(prenatal_meta.p_maternal_density, dtype=np.float64),
        )
    else:
        return bufs

    w_counts, kappa = evaluate_band_weighted_counts(
        bufs.frag_bands, bufs.count, bufs.paired_count, w_24
    )
    return replace(bufs, count=w_counts, kappa_frag=kappa)


def is_paired_sample_metadata(metadata: Mapping[object, object] | None) -> bool:
    """Return whether decoded or raw Arrow sample metadata indicates paired-end reads."""
    if not metadata:
        return False
    frag_meta = extract_sample_fragmentomics(metadata)
    if frag_meta is not None:
        return bool(frag_meta.paired_end)
    val = _lookup_frag_entry(metadata)
    if isinstance(val, (bytes, str)):
        try:
            val = json.loads(val)
        except json.JSONDecodeError, UnicodeDecodeError:
            return False
    if isinstance(val, Mapping):
        return bool(val.get("paired_end", False))
    return False


def resolve_fragmentomics_compatibility(
    sample: SampleBuffers | pa.Table,
    reference: pa.Table,
) -> bool:
    """Validate sample vs reference fragmentomic correction settings."""
    bufs = SampleBuffers.coerce(sample)
    sample_has_frag = (
        bool(bufs.meta.fragmentomics.paired_end)
        if bufs.meta.fragmentomics is not None
        else (
            bufs.frag_bands is not None
            and (
                bool(np.any(bufs.paired_count > 0)) or bool(np.any(bufs.frag_bands > 0))
            )
        )
    )

    ref_meta = decode_metadata(reference.schema.metadata)
    ref_model = ref_meta.get("model", {})
    if not isinstance(ref_model, Mapping):
        ref_model = {}

    if bool(ref_model.get("flat", False)):
        return sample_has_frag

    if "fragmentomics_corrected" in ref_model:
        ref_frag = bool(ref_model["fragmentomics_corrected"])
        if ref_frag != sample_has_frag:
            raise PredictError(
                f"sample fragmentomics ({'paired-end' if sample_has_frag else 'single-end'}) "
                f"does not match cohort reference fragmentomics_corrected={ref_frag}; "
                "use matching library types or rebuild the reference"
            )
        return ref_frag

    if sample_has_frag:
        raise PredictError(
            "reference lacks fragmentomics_corrected metadata (uncorrected legacy cohort reference) "
            "and cannot be projected against a paired-end sample; rebuild the reference"
        )
    return False
