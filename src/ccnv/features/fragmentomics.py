"""Feature transformation kernels for cell-free DNA fragmentomic signals."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
from numpy.typing import NDArray

from ccnv.models.results import ActiveFFSource
from ccnv.models.schemas import (
    FRAGMENTOMICS_METADATA_KEY,
    FragmentomicsMetadata,
    PrenatalModelMetadata,
    decode_metadata,
)
from ccnv.stats.fragmentomics import (
    CANONICAL_CHROMATOSOME_BP,
    FRAG_MAX_LEN,
    FRAG_MIN_LEN,
    FRAG_NUM_BINS,
    align_histogram,
    build_fragmentomic_feature_vector,
    deconvolve_fetal_maternal_densities,
    fisher_optimal_weights,
    fit_ff_fragmentomics_ensemble,
    segment_fragment_llr,
)

__all__ = [
    "align_reference_to_weighted_sample",
    "apply_prenatal_fisher_weighting",
    "evaluate_weighted_coverage",
    "extract_bin_fragmentomic_covariates",
    "extract_sample_fragmentomics",
    "extract_segment_fragment_lengths",
    "fit_cohort_fragmentomics_model",
    "fuse_active_fetal_fraction",
    "score_segment_origin",
]


def evaluate_weighted_coverage(
    table: pa.Table,
    weights_571: NDArray[np.floating],
    shift_bp: float = 0.0,
) -> NDArray[np.float64]:
    """Map fragment lengths through a 65,536-entry lookup table in <20 ms.

    Falls back to unweighted table counts when fragment_lengths is absent or empty.
    """
    counts = np.asarray(
        table.column("count").to_numpy(zero_copy_only=False), dtype=np.float64
    )
    if "fragment_lengths" not in table.column_names:
        return counts

    frag_col = table.column("fragment_lengths").combine_chunks()
    if not isinstance(frag_col, pa.ListArray):
        return counts

    values = np.asarray(frag_col.values)
    if values.size == 0:
        return counts

    offsets = np.asarray(frag_col.offsets)
    std_grid = np.arange(FRAG_MIN_LEN, FRAG_MAX_LEN + 1, dtype=np.float64)
    all_l = np.arange(65536, dtype=np.float64)
    w_lut = np.ones(65536, dtype=np.float32)

    valid = (all_l - shift_bp >= FRAG_MIN_LEN) & (all_l - shift_bp <= FRAG_MAX_LEN)
    w_lut[valid] = np.interp(all_l[valid] - shift_bp, std_grid, weights_571).astype(
        np.float32
    )

    cum = np.empty(values.size + 1, dtype=np.float64)
    cum[0] = 0.0
    np.cumsum(w_lut[values], out=cum[1:])
    return cum[offsets[1:]] - cum[offsets[:-1]]


def extract_bin_fragmentomic_covariates(
    table: pa.Table,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute per-bin mean fragment length and cleavage endpoint protection ratio.

    Imputes zero-coverage bins with the sample-wide median fragment length.
    """
    n_rows = table.num_rows
    counts = (
        np.asarray(
            table.column("count").to_numpy(zero_copy_only=False), dtype=np.float64
        )
        if "count" in table.column_names
        else np.zeros(n_rows, dtype=np.float64)
    )

    # 1. Per-bin mean fragment length
    mean_lens = np.full(n_rows, CANONICAL_CHROMATOSOME_BP, dtype=np.float64)
    if "fragment_lengths" in table.column_names:
        frag_col = table.column("fragment_lengths").combine_chunks()
        if isinstance(frag_col, pa.ListArray):
            values = np.asarray(frag_col.values, dtype=np.float64)
            if values.size > 0:
                offsets = np.asarray(frag_col.offsets)
                cum = np.empty(values.size + 1, dtype=np.float64)
                cum[0] = 0.0
                np.cumsum(values, out=cum[1:])
                sum_lens = cum[offsets[1:]] - cum[offsets[:-1]]
                bin_counts = offsets[1:] - offsets[:-1]
                has_reads = bin_counts > 0
                if np.any(has_reads):
                    computed = sum_lens[has_reads] / bin_counts[has_reads]
                    mean_lens[has_reads] = computed
                    sample_median = float(np.median(computed))
                    mean_lens[~has_reads] = sample_median

    # 2. Cleavage endpoint ratio: E_i / max(C_i, 1)
    if "endpoint_count" in table.column_names:
        ep_counts = np.asarray(
            table.column("endpoint_count").to_numpy(zero_copy_only=False),
            dtype=np.float64,
        )
        ep_ratios = ep_counts / np.maximum(counts, 1.0)
    else:
        ep_ratios = np.zeros(n_rows, dtype=np.float64)

    return mean_lens, ep_ratios


def extract_sample_fragmentomics(
    metadata: Mapping[object, object] | None,
) -> FragmentomicsMetadata | None:
    """Extract FragmentomicsMetadata from decoded metadata mapping or schema metadata."""
    if not metadata:
        return None
    for k in ("fragmentomics", "ccnv.fragmentomics", FRAGMENTOMICS_METADATA_KEY):
        val = metadata.get(k)
        if val is None and isinstance(k, str) and k.encode() in metadata:
            val = metadata[k.encode()]
        elif val is None and isinstance(k, bytes) and k.decode() in metadata:
            val = metadata[k.decode()]
        if val is not None:
            if isinstance(val, FragmentomicsMetadata):
                return val
            if isinstance(val, (bytes, str)):
                try:
                    loaded = json.loads(val)
                    if isinstance(loaded, dict):
                        return FragmentomicsMetadata(**loaded)
                except json.JSONDecodeError, UnicodeDecodeError, TypeError:
                    continue
            if isinstance(val, dict):
                try:
                    return FragmentomicsMetadata(**val)
                except TypeError:
                    continue
    return None


def extract_segment_fragment_lengths(
    table: pa.Table,
    contig: str,
    start: int,
    end: int,
) -> NDArray[np.uint16]:
    """Extract and concatenate fragment lengths within [start, end) on contig."""
    if "fragment_lengths" not in table.column_names:
        return np.empty(0, dtype=np.uint16)
    c_arr = table.column("contig")
    s_arr = table.column("start")
    e_arr = table.column("end")
    mask = pc.and_(
        pc.equal(c_arr, contig),
        pc.and_(pc.greater(e_arr, start), pc.less(s_arr, end)),
    )
    sub_table = table.filter(mask)
    if sub_table.num_rows == 0:
        return np.empty(0, dtype=np.uint16)
    frag_col = sub_table.column("fragment_lengths").combine_chunks()
    if not isinstance(frag_col, pa.ListArray):
        return np.empty(0, dtype=np.uint16)
    return np.asarray(frag_col.values, dtype=np.uint16)


def fuse_active_fetal_fraction(
    fetal_sex: str,
    sca_call: str,
    ffy: float | None,
    ffa: float,
    ffx: float,
    ff_size: float | None,
    prenatal_meta: PrenatalModelMetadata,
    tolerance: float = 0.05,
) -> tuple[float, ActiveFFSource]:
    """Fuse FFY, FFA, FFX, and FF_size via Bates-Granger precision pooling."""
    if fetal_sex == "M" and sca_call in ("XY", "XXY"):
        if ffy is not None:
            var_ffy = max(float(prenatal_meta.ffy_std), 1e-4) ** 2
            candidates: list[tuple[float, float, ActiveFFSource]] = [
                (ffy, var_ffy, "FFY")
            ]

            if abs(ffa - ffy) <= tolerance:
                var_ffa = max(float(prenatal_meta.ffa_std), 1e-4) ** 2
                candidates.append((ffa, var_ffa, "FFA"))

            if ffx > 0 and abs(ffx - ffy) <= tolerance:
                var_ffx = max(float(prenatal_meta.ffx_std), 1e-4) ** 2
                candidates.append((ffx, var_ffx, "FFX"))

            has_size = False
            if ff_size is not None and abs(ff_size - ffy) <= tolerance:
                var_size = max(float(prenatal_meta.ff_size_std), 1e-4) ** 2
                candidates.append((ff_size, var_size, "FF_SIZE"))
                has_size = True

            sum_w = sum(1.0 / v for _, v, _ in candidates)
            sum_p = sum(val / v for val, v, _ in candidates)
            active_ff = float(max(0.0, sum_p / sum_w))
            source: ActiveFFSource = "FUSED" if has_size else "FFY"
            return active_ff, source
        return float(max(0.0, ffa)), "FFA"

    # Female (XX) or other SCA
    if ff_size is not None and abs(ffa - ff_size) <= tolerance:
        var_ffa = max(float(prenatal_meta.ffa_std), 1e-4) ** 2
        var_size = max(float(prenatal_meta.ff_size_std), 1e-4) ** 2
        w_ffa = 1.0 / var_ffa
        w_size = 1.0 / var_size
        fused = (w_ffa * ffa + w_size * ff_size) / (w_ffa + w_size)
        return float(max(0.0, fused)), "FUSED"

    return float(max(0.0, ffa)), "FFA"


def score_segment_origin(
    lengths: Sequence[int] | NDArray[np.integer],
    shift_bp: float,
    p_fetal: Sequence[float] | NDArray[np.floating],
    p_maternal: Sequence[float] | NDArray[np.floating],
    active_ff: float,
    is_gain: bool,
    has_call: bool,
) -> tuple[float | None, float | None, str | None]:
    """Calculate segment short-fragment ratio, LLR_size, and origin classification."""
    if len(lengths) == 0:
        return None, None, None
    sr, llr = segment_fragment_llr(
        lengths,
        shift_bp,
        p_fetal=p_fetal,
        p_maternal=p_maternal,
        active_ff=active_ff,
        is_gain=is_gain,
    )
    if has_call:
        if llr > 2.0:
            origin = "FETAL_CONFIRMED"
        elif llr < -2.0:
            origin = "MATERNAL_CONFIRMED"
        else:
            origin = "INDETERMINATE"
    else:
        origin = None
    return float(sr), float(llr), origin


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
    if n_samples == 0 or not all(m is not None and m.paired_end for m in frag_metas):
        return None

    auto_aligned = np.zeros((n_samples, FRAG_NUM_BINS), dtype=np.float64)
    chry_aligned = np.zeros((n_samples, FRAG_NUM_BINS), dtype=np.float64)
    X_frag = np.zeros((n_samples, 36), dtype=np.float64)

    for i, meta in enumerate(frag_metas):
        assert meta is not None
        auto_aligned[i] = align_histogram(
            np.asarray(meta.hist_autosomal, dtype=np.float64),
            meta.sizing_shift_bp,
        )
        chry_aligned[i] = align_histogram(
            np.asarray(meta.hist_chry, dtype=np.float64),
            meta.sizing_shift_bp,
        )
        X_frag[i] = build_fragmentomic_feature_vector(meta)

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
        X_frag[valid_mask],
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


def align_reference_to_weighted_sample(
    reference: pa.Table,
    sample: pa.Table,
    weighted_sample: pa.Table,
) -> pa.Table:
    """Scale reference pca_mean and null_mean to match active Fisher-weighted sample counts."""
    u_counts = np.asarray(
        sample.column("count").to_numpy(zero_copy_only=False), dtype=np.float64
    )
    w_counts = np.asarray(
        weighted_sample.column("count").to_numpy(zero_copy_only=False), dtype=np.float64
    )
    scale_factors = np.where(u_counts > 0, w_counts / np.maximum(u_counts, 1e-6), 1.0)
    sm_coords = pl.DataFrame(
        {
            "contig": sample.column("contig").to_pylist(),
            "start": sample.column("start").to_numpy(),
            "end": sample.column("end").to_numpy(),
            "scale_factor": scale_factors,
        }
    )
    ref_df = pl.from_arrow(reference)
    joined_ref = ref_df.join(sm_coords, on=["contig", "start", "end"], how="left")
    ref_scales = joined_ref["scale_factor"].fill_null(1.0).to_numpy()
    new_pca_mean = (
        np.asarray(reference.column("pca_mean").to_numpy(), dtype=np.float64)
        * ref_scales
    )
    pca_idx = reference.schema.get_field_index("pca_mean")
    out_ref = reference.set_column(
        pca_idx, "pca_mean", pa.array(new_pca_mean, type=pa.float64())
    )
    if "null_mean" in reference.schema.names:
        new_null_mean = (
            np.asarray(reference.column("null_mean").to_numpy(), dtype=np.float64)
            * ref_scales
        )
        null_idx = reference.schema.get_field_index("null_mean")
        out_ref = out_ref.set_column(
            null_idx, "null_mean", pa.array(new_null_mean, type=pa.float64())
        )
    return out_ref


def apply_prenatal_fisher_weighting(
    sample: pa.Table,
    prenatal_meta: PrenatalModelMetadata | None,
) -> pa.Table:
    """Evaluate Fisher-optimal weighted coverage and replace the count column."""
    if (
        prenatal_meta is None
        or prenatal_meta.p_fetal_density is None
        or prenatal_meta.p_maternal_density is None
        or "fragment_lengths" not in sample.column_names
    ):
        return sample

    frag_col = sample.column("fragment_lengths").combine_chunks()
    if isinstance(frag_col, pa.ListArray) and len(frag_col.values) == 0:
        return sample

    frag_meta = extract_sample_fragmentomics(decode_metadata(sample.schema.metadata))
    shift_bp = frag_meta.sizing_shift_bp if frag_meta is not None else 0.0
    weights = fisher_optimal_weights(
        prenatal_meta.p_fetal_density,
        prenatal_meta.p_maternal_density,
        target_ff=0.10,
    )
    w_counts = evaluate_weighted_coverage(sample, weights, shift_bp=shift_bp)
    count_idx = sample.schema.get_field_index("count")
    return sample.set_column(count_idx, "count", pa.array(w_counts, type=pa.float64()))
