"""Feature transformation kernels for cell-free DNA fragmentomic signals."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from numpy.typing import NDArray

from ccnv._core import FRAG_BAND_EDGES
from ccnv.models import (
    NUM_FRAG_BANDS,
    SampleBuffers,
    SampleMetadata,
    fixed_size_list_to_numpy,
    reconstruct_sample_grid,
)
from ccnv.models.exceptions import PredictError
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
    compute_band_fisher_weights,
    deconvolve_fetal_maternal_densities,
    evaluate_band_weighted_counts,
    fit_ff_fragmentomics_ensemble,
    segment_fragment_llr,
)

__all__ = [
    "apply_prenatal_fisher_weighting",
    "evaluate_weighted_coverage",
    "extract_bin_fragmentomic_covariates",
    "extract_sample_fragmentomics",
    "extract_segment_fragment_lengths",
    "fit_cohort_fragmentomics_model",
    "fuse_active_fetal_fraction",
    "is_paired_sample_metadata",
    "resolve_fragmentomics_compatibility",
    "score_segment_origin",
    "score_segment_table_origin",
]


def _band_weights_from_curve(
    weights_571: NDArray[np.floating],
    shift_bp: float,
) -> NDArray[np.float64]:
    std_grid = np.arange(FRAG_MIN_LEN, FRAG_MAX_LEN + 1, dtype=np.float64)
    edges = np.asarray(FRAG_BAND_EDGES, dtype=np.int64)
    band_w = np.ones(NUM_FRAG_BANDS, dtype=np.float64)
    for k in range(NUM_FRAG_BANDS):
        lengths = np.arange(edges[k], edges[k + 1], dtype=np.float64) - shift_bp
        clipped = np.clip(lengths, FRAG_MIN_LEN, FRAG_MAX_LEN)
        band_w[k] = float(np.mean(np.interp(clipped, std_grid, weights_571)))
    return band_w


def evaluate_weighted_coverage(
    table: pa.Table | SampleBuffers,
    weights_571: NDArray[np.floating],
    shift_bp: float = 0.0,
) -> NDArray[np.float64]:
    """Evaluate Fisher-weighted per-bin coverage from Schema v2.0.0 frag_bands."""
    if isinstance(table, SampleBuffers):
        counts = np.asarray(table.count, dtype=np.float64)
        if table.num_rows == 0 or table.frag_bands is None:
            return counts
        band_w = _band_weights_from_curve(weights_571, shift_bp)
        bands_2d = table.frag_bands.astype(np.float64)
        paired_in_range = np.sum(bands_2d, axis=1)
        if not np.any(paired_in_range > 0):
            return counts
        return (bands_2d @ band_w) + np.maximum(counts - paired_in_range, 0.0)

    counts = np.asarray(
        table.column("count").to_numpy(zero_copy_only=False), dtype=np.float64
    )
    if table.num_rows == 0 or "frag_bands" not in table.column_names:
        return counts

    bands_col = table.column("frag_bands").combine_chunks()
    if bands_col.null_count == table.num_rows:
        return counts

    band_w = _band_weights_from_curve(weights_571, shift_bp)
    bands_2d = fixed_size_list_to_numpy(bands_col, NUM_FRAG_BANDS, np.uint32).astype(
        np.float64
    )
    if bands_col.null_count == 0:
        paired_in_range = np.sum(bands_2d, axis=1)
        if not np.any(paired_in_range > 0):
            return counts
        return (bands_2d @ band_w) + np.maximum(counts - paired_in_range, 0.0)

    valid_mask = ~pc.is_null(bands_col).to_numpy(zero_copy_only=False)
    valid_2d = bands_2d[valid_mask]
    out = counts.copy()
    paired_sub = np.sum(valid_2d, axis=1)
    if np.any(paired_sub > 0):
        out[valid_mask] = (valid_2d @ band_w) + np.maximum(
            counts[valid_mask] - paired_sub, 0.0
        )
    return out


def _compute_bin_frag_covariates(
    counts: NDArray[np.float64],
    paired_cnt: NDArray[np.float64] | None,
    flen_sum: NDArray[np.float64] | None,
    ep_counts: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n_rows = int(counts.shape[0])
    mean_lens = np.full(n_rows, CANONICAL_CHROMATOSOME_BP, dtype=np.float64)
    if paired_cnt is not None and flen_sum is not None and n_rows > 0:
        has_reads = paired_cnt > 0.0
        if np.any(has_reads):
            computed = flen_sum[has_reads] / paired_cnt[has_reads]
            mean_lens[has_reads] = computed
            mean_lens[~has_reads] = float(np.median(computed))
    ep_ratios = (
        ep_counts / np.maximum(counts, 1.0)
        if ep_counts is not None
        else np.zeros(n_rows, dtype=np.float64)
    )
    return mean_lens, ep_ratios


def extract_bin_fragmentomic_covariates(
    table: pa.Table | SampleBuffers,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute per-bin mean fragment length and cleavage endpoint protection ratio."""
    if isinstance(table, SampleBuffers):
        return _compute_bin_frag_covariates(
            np.asarray(table.count, dtype=np.float64),
            np.asarray(table.paired_count, dtype=np.float64),
            np.asarray(table.frag_len_sum, dtype=np.float64),
            np.asarray(table.endpoint_count, dtype=np.float64),
        )

    n_rows = table.num_rows
    counts = (
        np.asarray(
            table.column("count").to_numpy(zero_copy_only=False), dtype=np.float64
        )
        if "count" in table.column_names
        else np.zeros(n_rows, dtype=np.float64)
    )
    paired_cnt = (
        np.asarray(
            table.column("paired_count").to_numpy(zero_copy_only=False),
            dtype=np.float64,
        )
        if "paired_count" in table.column_names
        else None
    )
    flen_sum = (
        np.asarray(
            table.column("frag_len_sum").to_numpy(zero_copy_only=False),
            dtype=np.float64,
        )
        if "frag_len_sum" in table.column_names
        else None
    )
    ep_counts = (
        np.asarray(
            table.column("endpoint_count").to_numpy(zero_copy_only=False),
            dtype=np.float64,
        )
        if "endpoint_count" in table.column_names
        else None
    )
    return _compute_bin_frag_covariates(counts, paired_cnt, flen_sum, ep_counts)


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
    """Reconstruct representative fragment lengths within [start, end) from frag_bands."""
    if table.num_rows == 0 or "frag_bands" not in table.column_names:
        return np.empty(0, dtype=np.uint16)
    bands_col = table.column("frag_bands").combine_chunks()
    if bands_col.null_count == table.num_rows:
        return np.empty(0, dtype=np.uint16)

    sample_meta = SampleMetadata.from_table(table)
    codes, names, starts_u32, ends_u32 = reconstruct_sample_grid(sample_meta)
    try:
        c_code = names.index(contig)
    except ValueError:
        return np.empty(0, dtype=np.uint16)

    mask = (codes == c_code) & (ends_u32 > start) & (starts_u32 < end)
    if not np.any(mask):
        return np.empty(0, dtype=np.uint16)

    sub_col = bands_col.filter(pa.array(mask, type=pa.bool_()))
    if len(sub_col) == 0 or sub_col.null_count == len(sub_col):
        return np.empty(0, dtype=np.uint16)

    bands_2d = fixed_size_list_to_numpy(sub_col, NUM_FRAG_BANDS, np.uint32)
    band_counts = np.sum(bands_2d, axis=0, dtype=np.int64)
    if not np.any(band_counts > 0):
        return np.empty(0, dtype=np.uint16)

    edges = np.asarray(FRAG_BAND_EDGES, dtype=np.int64)
    midpoints = ((edges[:-1] + edges[1:] - 1) // 2).astype(np.uint16)
    return np.repeat(midpoints, band_counts)


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
    known_lens = np.asarray(lengths)
    known_lens = known_lens[known_lens > 0]
    if known_lens.size == 0:
        return None, None, None
    sr, llr = segment_fragment_llr(
        known_lens,
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


def score_segment_table_origin(
    table: pa.Table | SampleBuffers,
    contig: str,
    start: int,
    end: int,
    shift_bp: float,
    p_fetal: Sequence[float] | NDArray[np.floating],
    p_maternal: Sequence[float] | NDArray[np.floating],
    active_ff: float,
    is_gain: bool,
    has_call: bool,
) -> tuple[float | None, float | None, str | None]:
    """Evaluate segment short-fragment ratio, multinomial band LLR, and origin from frag_bands."""
    if isinstance(table, SampleBuffers):
        if table.num_rows == 0 or table.frag_bands is None:
            return None, None, None
        codes, names, starts_u32, ends_u32 = reconstruct_sample_grid(table.meta)
        if contig not in names:
            return None, None, None
        mask = (codes == names.index(contig)) & (ends_u32 > start) & (starts_u32 < end)
        if not np.any(mask):
            return None, None, None
        band_counts = table.frag_bands[mask].sum(axis=0, dtype=np.float64)
    else:
        if table.num_rows == 0 or "frag_bands" not in table.column_names:
            return None, None, None
        bands_col = table.column("frag_bands").combine_chunks()
        if bands_col.null_count == table.num_rows:
            return None, None, None
        sample_meta = SampleMetadata.from_table(table)
        codes, names, starts_u32, ends_u32 = reconstruct_sample_grid(sample_meta)
        if contig not in names:
            return None, None, None
        mask = (codes == names.index(contig)) & (ends_u32 > start) & (starts_u32 < end)
        if not np.any(mask):
            return None, None, None
        sub_col = bands_col.filter(pa.array(mask, type=pa.bool_()))
        if len(sub_col) == 0 or sub_col.null_count == len(sub_col):
            return None, None, None
        band_counts = fixed_size_list_to_numpy(sub_col, NUM_FRAG_BANDS, np.uint32).sum(
            axis=0, dtype=np.float64
        )
    total = float(np.sum(band_counts))
    if total <= 0.0:
        return None, None, None

    edges = np.asarray(FRAG_BAND_EDGES, dtype=np.int64)
    short_bands = edges[1:] <= 151
    sr = float(np.sum(band_counts[short_bands]) / total)

    pf = np.asarray(p_fetal, dtype=np.float64)
    pm = np.asarray(p_maternal, dtype=np.float64)
    pf_band = np.zeros(NUM_FRAG_BANDS, dtype=np.float64)
    pm_band = np.zeros(NUM_FRAG_BANDS, dtype=np.float64)
    for k in range(NUM_FRAG_BANDS):
        bp = np.arange(int(edges[k]), int(edges[k + 1]), dtype=np.float64)
        idx = (
            np.clip(np.round(bp - shift_bp), FRAG_MIN_LEN, FRAG_MAX_LEN).astype(int)
            - FRAG_MIN_LEN
        )
        pf_band[k] = float(np.sum(pf[idx]))
        pm_band[k] = float(np.sum(pm[idx]))

    f = float(np.clip(active_ff, 0.01, 0.40))
    if is_gain:
        f_fet, f_mat = (1.5 * f) / (1.0 + 0.5 * f), f / (1.0 + 0.5 * (1.0 - f))
    else:
        f_fet, f_mat = (0.5 * f) / (1.0 - 0.5 * f), f / (1.0 - 0.5 * (1.0 - f))

    p_f_eval = f_fet * pf_band + (1.0 - f_fet) * pm_band
    p_m_eval = f_mat * pf_band + (1.0 - f_mat) * pm_band
    llr = float(
        np.sum(
            band_counts
            * np.log(np.maximum(p_f_eval, 1e-12) / np.maximum(p_m_eval, 1e-12))
        )
    )
    origin = (
        (
            "FETAL_CONFIRMED"
            if llr > 2.0
            else "MATERNAL_CONFIRMED"
            if llr < -2.0
            else "INDETERMINATE"
        )
        if has_call
        else None
    )
    return sr, llr, origin


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


def apply_prenatal_fisher_weighting[SampleT: (pa.Table, SampleBuffers)](
    sample: SampleT,
    prenatal_meta: PrenatalModelMetadata | None,
) -> SampleT:
    """Evaluate 24-band Fisher-optimal weighted coverage and attach per-bin Kish factor."""
    if (
        prenatal_meta is None
        or (
            prenatal_meta.band_fisher_weights is None
            and (
                prenatal_meta.p_fetal_density is None
                or prenatal_meta.p_maternal_density is None
            )
        )
        or sample.num_rows == 0
    ):
        return sample

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
        return sample

    if isinstance(sample, SampleBuffers):
        if sample.frag_bands is None or not np.any(sample.paired_count > 0):
            return sample
        w_counts, kappa = evaluate_band_weighted_counts(
            sample.frag_bands, sample.count, sample.paired_count, w_24
        )
        return replace(sample, count=w_counts, kappa_frag=kappa)

    if "frag_bands" not in sample.column_names:
        return sample
    bands_col = sample.column("frag_bands").combine_chunks()
    if bands_col.null_count == sample.num_rows:
        return sample
    if "paired_count" in sample.column_names:
        total_paired = int(pc.sum(sample.column("paired_count")).as_py() or 0)
        if total_paired == 0:
            return sample

    bands_2d = fixed_size_list_to_numpy(bands_col, NUM_FRAG_BANDS, np.uint32)
    counts_raw = np.asarray(
        sample.column("count").to_numpy(zero_copy_only=False), dtype=np.float64
    )
    paired_raw = (
        np.asarray(
            sample.column("paired_count").to_numpy(zero_copy_only=False),
            dtype=np.float64,
        )
        if "paired_count" in sample.column_names
        else counts_raw
    )
    w_counts, kappa = evaluate_band_weighted_counts(
        bands_2d, counts_raw, paired_raw, w_24
    )
    count_idx = sample.schema.get_field_index("count")
    out = sample.set_column(count_idx, "count", pa.array(w_counts, type=pa.float64()))
    if "kappa_frag" in out.column_names:
        k_idx = out.schema.get_field_index("kappa_frag")
        out = out.set_column(k_idx, "kappa_frag", pa.array(kappa, type=pa.float32()))
    else:
        out = out.append_column("kappa_frag", pa.array(kappa, type=pa.float32()))
    return out


def is_paired_sample_metadata(metadata: Mapping[object, object] | None) -> bool:
    """Return whether decoded or raw Arrow sample metadata indicates paired-end reads."""
    if not metadata:
        return False
    frag_meta = extract_sample_fragmentomics(metadata)
    if frag_meta is not None:
        return bool(frag_meta.paired_end)
    for k in ("fragmentomics", "ccnv.fragmentomics", FRAGMENTOMICS_METADATA_KEY):
        val = metadata.get(k)
        if val is None and isinstance(k, str) and k.encode() in metadata:
            val = metadata[k.encode()]
        elif val is None and isinstance(k, bytes) and k.decode() in metadata:
            val = metadata[k.decode()]
        if isinstance(val, Mapping):
            return bool(val.get("paired_end", False))
        if isinstance(val, (bytes, str)):
            try:
                loaded = json.loads(val)
                if isinstance(loaded, Mapping):
                    return bool(loaded.get("paired_end", False))
            except json.JSONDecodeError, UnicodeDecodeError, TypeError:
                continue
    return False


def resolve_fragmentomics_compatibility(
    sample: pa.Table | SampleBuffers,
    reference: pa.Table,
) -> bool:
    """Validate sample vs reference fragmentomic correction settings."""
    if isinstance(sample, SampleBuffers):
        if sample.meta.fragmentomics is not None:
            sample_has_frag = bool(sample.meta.fragmentomics.paired_end)
        else:
            sample_has_frag = sample.frag_bands is not None
    else:
        sample_meta = decode_metadata(sample.schema.metadata)
        frag_meta = extract_sample_fragmentomics(sample_meta)
        if frag_meta is not None:
            sample_has_frag = bool(frag_meta.paired_end)
        elif is_paired_sample_metadata(sample_meta):
            sample_has_frag = True
        elif "frag_bands" in sample.column_names and sample.num_rows > 0:
            sample_has_frag = sample.column("frag_bands").null_count < sample.num_rows
        else:
            sample_has_frag = False

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
