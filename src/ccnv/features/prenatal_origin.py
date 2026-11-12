"""Prenatal segment origin scoring and Limit-of-Detection calculation.

Implements the 5-State Self-Referenced (p0, pF) Coupled Depth and Dirichlet-Multinomial
origin engine, 16-node Gauss-Legendre mosaic quadrature, O(log U) 24-band prefix sums,
and analytical Limit-of-Detection (LoD) calculations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import Final

import numpy as np
import pyarrow as pa
import scipy.special
from numpy.typing import NDArray

from ccnv._core import FRAG_BAND_EDGES
from ccnv.io.parquet import fixed_size_list_to_numpy
from ccnv.models.domain import normalize_chr
from ccnv.models.metadata import PrenatalModelMetadata
from ccnv.models.results import (
    FetalFractionResult,
    PrenatalLoDMetrics,
    SCACall,
    SegmentResult,
)
from ccnv.models.schemas import (
    NUM_FRAG_BANDS,
    SampleBuffers,
    SampleMetadata,
    reconstruct_sample_grid,
)
from ccnv.stats.fragmentomics import FRAG_MIN_LEN

__all__ = [
    "BandPrefixSums",
    "PrenatalLoDMetrics",
    "annotate_prenatal_origin_and_lod",
    "annotate_prenatal_segments",
    "classify_segment_origin_5state",
]

_QUAD_NODES_16: Final[tuple[NDArray[np.float64], NDArray[np.float64]]] = (
    np.polynomial.legendre.leggauss(16)
)


class BandPrefixSums:
    """O(log U) 24-band cumulative prefix sum query engine across genomic bins."""

    __slots__ = ("_contig_data",)

    def __init__(
        self,
        sample_bands: NDArray[np.uint32],
        bin_contigs: NDArray[np.int16],
        contig_names: Sequence[str],
        bin_starts: NDArray[np.int64],
        bin_ends: NDArray[np.int64],
        bin_usable: NDArray[np.bool_] | None = None,
    ) -> None:
        self._contig_data: dict[
            str, tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]
        ] = {}

        bands_f64 = np.asarray(sample_bands, dtype=np.float64)
        if bin_usable is not None:
            usable_mask = np.asarray(bin_usable, dtype=np.bool_)
            bands_f64 = np.where(usable_mask[:, None], bands_f64, 0.0)

        for c_idx, c_name in enumerate(contig_names):
            norm_name = normalize_chr(c_name)
            idx = np.where(bin_contigs == c_idx)[0]
            if idx.size == 0:
                continue
            starts_c = np.asarray(bin_starts[idx], dtype=np.int64)
            ends_c = np.asarray(bin_ends[idx], dtype=np.int64)
            bands_c = bands_f64[idx]

            cum_c = np.zeros((len(idx) + 1, 24), dtype=np.float64)
            cum_c[1:] = np.cumsum(bands_c, axis=0)
            self._contig_data[norm_name] = (starts_c, ends_c, cum_c)

    @classmethod
    def from_sample_table(
        cls,
        table: pa.Table | SampleBuffers,
        bin_usable: NDArray[np.bool_] | None = None,
    ) -> BandPrefixSums | None:
        """Construct BandPrefixSums engine from SampleBuffers or pa.Table."""
        if isinstance(table, SampleBuffers):
            if table.num_rows == 0 or table.frag_bands is None:
                return None
            codes, names, starts_u32, ends_u32 = reconstruct_sample_grid(table.meta)
            return cls(
                sample_bands=table.frag_bands,
                bin_contigs=np.asarray(codes, dtype=np.int16),
                contig_names=names,
                bin_starts=np.asarray(starts_u32, dtype=np.int64),
                bin_ends=np.asarray(ends_u32, dtype=np.int64),
                bin_usable=bin_usable,
            )

        if table.num_rows == 0 or "frag_bands" not in table.column_names:
            return None
        bands_col = table.column("frag_bands").combine_chunks()
        if bands_col.null_count == table.num_rows:
            return None
        sample_meta = SampleMetadata.from_table(table)
        codes, names, starts_u32, ends_u32 = reconstruct_sample_grid(sample_meta)
        bands_np = fixed_size_list_to_numpy(bands_col, NUM_FRAG_BANDS, np.uint32)
        return cls(
            sample_bands=bands_np,
            bin_contigs=np.asarray(codes, dtype=np.int16),
            contig_names=names,
            bin_starts=np.asarray(starts_u32, dtype=np.int64),
            bin_ends=np.asarray(ends_u32, dtype=np.int64),
            bin_usable=bin_usable,
        )

    def query(self, contig: str, start: int, end: int) -> NDArray[np.float64]:
        """Query 24-band cumulative count vector in [start, end) on contig in O(log U)."""
        norm_name = normalize_chr(contig)
        data = self._contig_data.get(norm_name)
        if data is None:
            return np.zeros(24, dtype=np.float64)

        starts_c, ends_c, cum_c = data
        i_start = int(np.searchsorted(ends_c, start, side="right"))
        i_end = int(np.searchsorted(starts_c, end, side="left"))

        if i_start >= i_end:
            return np.zeros(24, dtype=np.float64)

        diff = cum_c[i_end] - cum_c[i_start]
        return np.asarray(diff, dtype=np.float64)


def _dirichlet_multinomial_logpmf(
    k_vec: NDArray[np.float64],
    alpha_vec: NDArray[np.float64],
) -> float:
    """Evaluate Dirichlet-Multinomial log PMF via gammaln."""
    total_n = float(np.sum(k_vec))
    if total_n <= 0.0:
        return 0.0
    alpha_sum = float(np.sum(alpha_vec))
    gammaln = scipy.special.gammaln
    pos = k_vec > 0
    k_pos = k_vec[pos]
    alpha_pos = np.maximum(alpha_vec[pos], 1e-12)
    ll = (
        gammaln(alpha_sum)
        - gammaln(total_n + alpha_sum)
        + float(np.sum(gammaln(k_pos + alpha_pos) - gammaln(alpha_pos)))
    )
    return float(ll)


def _gaussian_depth_loglik(
    r_obs: float,
    r_exp: float,
    sigma_r: float,
    sigma_prop: float,
) -> float:
    """Evaluate Gaussian log-likelihood for observed vs expected depth ratio."""
    sig2 = max(sigma_r**2 + sigma_prop**2, 1e-8)
    return -0.5 * math.log(2.0 * math.pi * sig2) - 0.5 * ((r_obs - r_exp) ** 2) / sig2


def _convert_571_to_24_bands(
    p_571: Sequence[float] | NDArray[np.floating],
) -> NDArray[np.float64]:
    """Convert 571-bin fragment length density into canonical 24 bands."""
    p_arr = np.asarray(p_571, dtype=np.float64)
    if p_arr.size == 24:
        res = np.clip(p_arr, 1e-8, None)
        return res / np.sum(res)

    edges = np.asarray(FRAG_BAND_EDGES, dtype=np.int64)
    n_bands = len(edges) - 1
    p_24 = np.zeros(n_bands, dtype=np.float64)
    for k in range(n_bands):
        s = int(edges[k] - FRAG_MIN_LEN)
        e = int(edges[k + 1] - FRAG_MIN_LEN)
        p_24[k] = float(np.sum(p_arr[s:e]))
    p_24 = np.clip(p_24, 1e-8, None)
    return p_24 / np.sum(p_24)


def classify_segment_origin_5state(
    r_obs: float,
    k_band_counts: NDArray[np.float64],
    p_fetal_24: NDArray[np.float64],
    p_maternal_24: NDArray[np.float64],
    p_plasma_0: NDArray[np.float64],
    active_ff: float,
    active_ff_se: float,
    is_gain: bool,
    *,
    alpha_dm: float = 10000.0,
    sigma_r: float = 0.005,
) -> tuple[str, dict[str, float], float | None, tuple[float, float] | None]:
    """Classify segment biological origin across 5 coupled Depth + DM hypotheses."""
    cn_f = 3.0 if is_gain else 1.0
    cn_m = 3.0 if is_gain else 1.0
    f_eff = float(np.clip(active_ff, 0.001, 0.60))
    f_se = max(float(active_ff_se), 0.001)

    nodes_x, weights_w = _QUAD_NODES_16
    m_cpm_nodes = 0.5 * (0.95 - 0.05) * nodes_x + 0.5 * (0.95 + 0.05)
    m_cpm_weights = 0.5 * (0.95 - 0.05) * weights_w

    m_mat_nodes = 0.5 * (0.95 - 0.01) * nodes_x + 0.5 * (0.95 + 0.01)
    m_mat_weights = 0.5 * (0.95 - 0.01) * weights_w

    # 1. FETAL_CONSTITUTIONAL (CN_F in {1, 3}, CN_M = 2)
    r_exp_fc = 1.0 + (cn_f / 2.0 - 1.0) * f_eff
    phi_fc = (cn_f / 2.0 * f_eff) / (1.0 - f_eff + cn_f / 2.0 * f_eff)
    p_fc = phi_fc * p_fetal_24 + (1.0 - phi_fc) * p_maternal_24
    ll_fc = _gaussian_depth_loglik(
        r_obs, r_exp_fc, sigma_r, abs(cn_f / 2.0 - 1.0) * f_se
    ) + _dirichlet_multinomial_logpmf(k_band_counts, alpha_dm * p_fc)

    # 2. PLACENTAL_MOSAIC (CPM, m in [0.05, 0.95])
    ll_cpm_nodes: list[float] = []
    jac_cpm = abs(cn_f / 2.0 - 1.0) * f_eff
    for m in m_cpm_nodes:
        r_exp_m = 1.0 + (cn_f / 2.0 - 1.0) * m * f_eff
        phi_m = (
            (1.0 + (cn_f / 2.0 - 1.0) * m)
            * f_eff
            / (1.0 - f_eff + (1.0 + (cn_f / 2.0 - 1.0) * m) * f_eff)
        )
        p_m_st = phi_m * p_fetal_24 + (1.0 - phi_m) * p_maternal_24
        ll_node = _gaussian_depth_loglik(
            r_obs, r_exp_m, sigma_r, abs(cn_f / 2.0 - 1.0) * m * f_se
        ) + _dirichlet_multinomial_logpmf(k_band_counts, alpha_dm * p_m_st)
        ll_cpm_nodes.append(ll_node)
    max_cpm = max(ll_cpm_nodes)
    unnorm_cpm = np.exp(np.array(ll_cpm_nodes) - max_cpm)
    int_cpm = float(np.sum(m_cpm_weights * unnorm_cpm))
    ll_cpm = max_cpm + math.log(max(int_cpm, 1e-300)) + math.log(max(jac_cpm, 1e-6))
    m_hat_cpm = float(
        np.sum(m_cpm_weights * m_cpm_nodes * unnorm_cpm) / max(int_cpm, 1e-300)
    )
    var_cpm = float(
        np.sum(m_cpm_weights * ((m_cpm_nodes - m_hat_cpm) ** 2) * unnorm_cpm)
        / max(int_cpm, 1e-300)
    )

    # 3. MATERNAL_UNINHERITED (CN_M in {1, 3}, CN_F = 2)
    r_exp_mu = 1.0 + (cn_m / 2.0 - 1.0) * (1.0 - f_eff)
    phi_mu = f_eff / (cn_m / 2.0 * (1.0 - f_eff) + f_eff)
    p_mu = phi_mu * p_fetal_24 + (1.0 - phi_mu) * p_maternal_24
    ll_mu = _gaussian_depth_loglik(
        r_obs, r_exp_mu, sigma_r, abs(cn_m / 2.0 - 1.0) * f_se
    ) + _dirichlet_multinomial_logpmf(k_band_counts, alpha_dm * p_mu)

    # 4. MATERNAL_AND_FETAL_INHERITED (Mendelian, CN_M = CN_F in {1, 3})
    r_exp_mfi = cn_m / 2.0
    p_mfi = p_plasma_0
    ll_mfi = _gaussian_depth_loglik(
        r_obs, r_exp_mfi, sigma_r, 0.0
    ) + _dirichlet_multinomial_logpmf(k_band_counts, alpha_dm * p_mfi)

    # 5. MATERNAL_MOSAIC (m in [0.01, 0.95])
    ll_mat_nodes: list[float] = []
    jac_mat = abs(cn_m / 2.0 - 1.0) * (1.0 - f_eff)
    for m in m_mat_nodes:
        r_exp_m = 1.0 + (cn_m / 2.0 - 1.0) * m * (1.0 - f_eff)
        phi_m = f_eff / ((1.0 + (cn_m / 2.0 - 1.0) * m) * (1.0 - f_eff) + f_eff)
        p_m_st = phi_m * p_fetal_24 + (1.0 - phi_m) * p_maternal_24
        ll_node = _gaussian_depth_loglik(
            r_obs, r_exp_m, sigma_r, abs(cn_m / 2.0 - 1.0) * m * f_se
        ) + _dirichlet_multinomial_logpmf(k_band_counts, alpha_dm * p_m_st)
        ll_mat_nodes.append(ll_node)
    max_mat = max(ll_mat_nodes)
    unnorm_mat = np.exp(np.array(ll_mat_nodes) - max_mat)
    int_mat = float(np.sum(m_mat_weights * unnorm_mat))
    ll_mm = max_mat + math.log(max(int_mat, 1e-300)) + math.log(max(jac_mat, 1e-6))
    m_hat_mat = float(
        np.sum(m_mat_weights * m_mat_nodes * unnorm_mat) / max(int_mat, 1e-300)
    )
    var_mat = float(
        np.sum(m_mat_weights * ((m_mat_nodes - m_hat_mat) ** 2) * unnorm_mat)
        / max(int_mat, 1e-300)
    )

    priors: dict[str, float] = {
        "FETAL_CONSTITUTIONAL": 0.20,
        "PLACENTAL_MOSAIC": 0.20,
        "MATERNAL_UNINHERITED": 0.20,
        "MATERNAL_AND_FETAL_INHERITED": 0.20,
        "MATERNAL_MOSAIC": 0.20,
    }

    log_joints: dict[str, float] = {
        "FETAL_CONSTITUTIONAL": ll_fc + math.log(priors["FETAL_CONSTITUTIONAL"]),
        "PLACENTAL_MOSAIC": ll_cpm + math.log(priors["PLACENTAL_MOSAIC"]),
        "MATERNAL_UNINHERITED": ll_mu + math.log(priors["MATERNAL_UNINHERITED"]),
        "MATERNAL_AND_FETAL_INHERITED": ll_mfi
        + math.log(priors["MATERNAL_AND_FETAL_INHERITED"]),
        "MATERNAL_MOSAIC": ll_mm + math.log(priors["MATERNAL_MOSAIC"]),
    }
    max_lj = max(log_joints.values())
    unnorm_post = {k: math.exp(v - max_lj) for k, v in log_joints.items()}
    sum_post = sum(unnorm_post.values())
    posteriors = {k: unnorm_post[k] / sum_post for k in unnorm_post}

    best_state = max(posteriors.keys(), key=lambda k: posteriors[k])
    call: str
    if posteriors[best_state] < 0.80:
        call = "INDETERMINATE"
    else:
        call = best_state

    mosaic_frac: float | None = None
    mosaic_ci: tuple[float, float] | None = None
    if call == "PLACENTAL_MOSAIC":
        mosaic_frac = round(float(np.clip(m_hat_cpm, 0.05, 0.95)), 4)
        se_cpm = math.sqrt(max(var_cpm, 1e-6))
        mosaic_ci = (
            round(float(max(0.0, m_hat_cpm - 1.96 * se_cpm)), 4),
            round(float(min(1.0, m_hat_cpm + 1.96 * se_cpm)), 4),
        )
    elif call == "MATERNAL_MOSAIC":
        mosaic_frac = round(float(np.clip(m_hat_mat, 0.01, 0.95)), 4)
        se_mat = math.sqrt(max(var_mat, 1e-6))
        mosaic_ci = (
            round(float(max(0.0, m_hat_mat - 1.96 * se_mat)), 4),
            round(float(min(1.0, m_hat_mat + 1.96 * se_mat)), 4),
        )

    return call, posteriors, mosaic_frac, mosaic_ci


def annotate_prenatal_origin_and_lod(
    segments: Sequence[SegmentResult],
    sample_bands: NDArray[np.uint32] | None = None,
    bin_contigs: NDArray[np.int16] | None = None,
    contig_names: Sequence[str] | None = None,
    bin_usable: NDArray[np.bool_] | None = None,
    p_fetal_24: NDArray[np.float64] | None = None,
    mu_fetal_w: float = 1.0,
    mu_maternal_w: float = 1.0,
    active_ff: float = 0.0,
    active_ff_se: float = 0.01,
    sca_call: SCACall = "XX",
    *,
    sample_depth_per_mb: float = 10000.0,
    sys_var_per_mb: float = 0.0001,
    z_threshold: float = 3.0,
    bin_starts: NDArray[np.int64] | None = None,
    bin_ends: NDArray[np.int64] | None = None,
    p_maternal_24: NDArray[np.float64] | None = None,
    prefix_sums: BandPrefixSums | None = None,
    sample_table: pa.Table | SampleBuffers | None = None,
    prenatal_meta: PrenatalModelMetadata | None = None,
    prenatal_result: FetalFractionResult | None = None,
) -> tuple[tuple[SegmentResult, ...], PrenatalLoDMetrics]:
    """Annotate segments with expected ratio, implied FF, discordance, origin, and LoD."""
    if prenatal_result is not None:
        active_ff = prenatal_result.active_ff
        active_ff_se = (
            prenatal_result.active_ff_se
            if prenatal_result.active_ff_se is not None
            else active_ff_se
        )
        sca_call = prenatal_result.sca.call

    if prenatal_meta is not None:
        if p_fetal_24 is None and prenatal_meta.p_fetal_density is not None:
            p_fetal_24 = _convert_571_to_24_bands(prenatal_meta.p_fetal_density)
        if p_maternal_24 is None and prenatal_meta.p_maternal_density is not None:
            p_maternal_24 = _convert_571_to_24_bands(prenatal_meta.p_maternal_density)
        if prenatal_meta.mu_fetal_w is not None:
            mu_fetal_w = float(prenatal_meta.mu_fetal_w)
        if prenatal_meta.mu_maternal_w is not None:
            mu_maternal_w = float(prenatal_meta.mu_maternal_w)

    if prefix_sums is None and sample_table is not None:
        prefix_sums = BandPrefixSums.from_sample_table(sample_table)

    # 1. Analytical Limit of Detection (LoD) minimum fetal fraction thresholds
    z_target = 3.0 + 2.3263479
    d_mb = max(float(sample_depth_per_mb), 1.0)
    v_sys = max(float(sys_var_per_mb), 0.0)

    sig_35mb = math.sqrt(1.0 / (d_mb * 35.0) + v_sys / 35.0)
    sig_3mb = math.sqrt(1.0 / (d_mb * 3.0) + v_sys / 3.0)
    lod_ff_35mb = float(np.clip(2.0 * z_target * sig_35mb, 0.005, 0.49))
    lod_ff_3mb = float(np.clip(2.0 * z_target * sig_3mb, lod_ff_35mb + 0.001, 0.499))
    lod_metrics = PrenatalLoDMetrics(lod_ff_35mb=lod_ff_35mb, lod_ff_3mb=lod_ff_3mb)

    # 2. Build PrefixSums if needed
    if (
        prefix_sums is None
        and sample_bands is not None
        and bin_contigs is not None
        and contig_names is not None
        and bin_starts is not None
        and bin_ends is not None
    ):
        prefix_sums = BandPrefixSums(
            sample_bands=sample_bands,
            bin_contigs=bin_contigs,
            contig_names=contig_names,
            bin_starts=bin_starts,
            bin_ends=bin_ends,
            bin_usable=bin_usable,
        )

    # 3. Reference and self-referenced plasma densities
    p_f24 = (
        _convert_571_to_24_bands(p_fetal_24)
        if p_fetal_24 is not None
        else np.ones(24, dtype=np.float64) / 24.0
    )

    if p_maternal_24 is not None:
        p_m24 = _convert_571_to_24_bands(p_maternal_24)
        p0_plasma = active_ff * p_f24 + (1.0 - active_ff) * p_m24
    elif sample_bands is not None and sample_bands.size > 0:
        usable_idx = (
            np.where(bin_usable)[0]
            if bin_usable is not None and np.any(bin_usable)
            else np.arange(sample_bands.shape[0])
        )
        sum_bands = np.sum(sample_bands[usable_idx], axis=0, dtype=np.float64)
        total_k = float(np.sum(sum_bands))
        p0_plasma = sum_bands / max(total_k, 1.0) if total_k > 0 else np.ones(24) / 24.0
        p_m_raw = np.clip(p0_plasma - active_ff * p_f24, 1e-8, None)
        p_m24 = p_m_raw / np.sum(p_m_raw)
    else:
        p0_plasma = np.ones(24, dtype=np.float64) / 24.0
        p_m24 = np.ones(24, dtype=np.float64) / 24.0

    f_act = max(float(active_ff), 0.0)
    mu_f = max(float(mu_fetal_w), 1e-4)
    mu_m = max(float(mu_maternal_w), 1e-4)
    f_eff = float(
        np.clip((f_act * mu_f) / ((1.0 - f_act) * mu_m + f_act * mu_f), 0.0, 0.60)
    )

    annotated: list[SegmentResult] = []
    is_aneuploid = sca_call not in ("XX", "XY")

    for seg in segments:
        if f_eff <= 0.0:
            expected_ratio = 0.0
        elif seg.ratio >= 0.0:
            expected_ratio = math.log2(1.0 + 0.5 * f_eff)
        else:
            arg = max(1e-6, 1.0 - 0.5 * f_eff)
            expected_ratio = math.log2(arg)

        r_obs = 2.0**seg.ratio
        implied_ff_raw = 2.0 * abs(r_obs - 1.0)
        implied_ff = float(implied_ff_raw)

        norm_contig = normalize_chr(seg.contig)
        is_gonosome = norm_contig in {"X", "Y"}
        has_call = (seg.call is not None) or (abs(seg.zscore) >= z_threshold)

        discordant = False
        reason: str | None = None

        if is_gonosome and (is_aneuploid or has_call):
            discordant = True
            reason = "SCA_DISCORDANT"
        elif has_call and abs(seg.ratio) < 0.5 * abs(expected_ratio):
            discordant = True
            reason = "SUB_FETAL_RATIO"
        elif has_call and implied_ff > max(2.5 * f_act, f_act + 0.15):
            discordant = True
            reason = "SUPER_FETAL_RATIO"

        frag_short_ratio: float | None = None
        frag_llr_size: float | None = None
        origin_call: str | None = None
        mosaic_frac: float | None = None
        mosaic_ci: tuple[float, float] | None = None

        if prefix_sums is not None:
            k_s = prefix_sums.query(seg.contig, seg.start, seg.end)
            total_seg_k = float(np.sum(k_s))
            if total_seg_k > 0:
                frag_short_ratio = float(np.sum(k_s[:6]) / total_seg_k)
                log_ratio_fm = np.log(np.maximum(p_f24, 1e-12)) - np.log(
                    np.maximum(p_m24, 1e-12)
                )
                frag_llr_size = float(np.sum(k_s * log_ratio_fm))

                if has_call:
                    is_gain = (
                        seg.call.lower() == "gain"
                        if seg.call is not None
                        else (seg.ratio >= 0.0)
                    )
                    origin_call, _, mosaic_frac, mosaic_ci = (
                        classify_segment_origin_5state(
                            r_obs=r_obs,
                            k_band_counts=k_s,
                            p_fetal_24=p_f24,
                            p_maternal_24=p_m24,
                            p_plasma_0=p0_plasma,
                            active_ff=f_eff,
                            active_ff_se=active_ff_se,
                            is_gain=is_gain,
                        )
                    )

        annotated.append(
            replace(
                seg,
                expected_ratio=float(expected_ratio),
                implied_ff=float(implied_ff),
                discordant=discordant,
                discordance_reason=reason,
                frag_short_ratio=frag_short_ratio,
                frag_llr_size=frag_llr_size,
                origin_call=origin_call,
                mosaic_fraction=mosaic_frac,
                mosaic_ci=mosaic_ci,
            )
        )

    return tuple(annotated), lod_metrics


def annotate_prenatal_segments(
    segments: Sequence[SegmentResult],
    prenatal_result: FetalFractionResult,
    z_threshold: float = 3.0,
    sample_table: pa.Table | SampleBuffers | None = None,
    prenatal_meta: PrenatalModelMetadata | None = None,
    *,
    sample_depth_per_mb: float = 10000.0,
    sys_var_per_mb: float = 0.0001,
) -> list[SegmentResult]:
    """Backwards-compatible wrapper annotating segments and attaching origin and LoD."""
    annotated, lod = annotate_prenatal_origin_and_lod(
        segments=segments,
        z_threshold=z_threshold,
        sample_table=sample_table,
        prenatal_meta=prenatal_meta,
        prenatal_result=prenatal_result,
        sample_depth_per_mb=sample_depth_per_mb,
        sys_var_per_mb=sys_var_per_mb,
    )
    if isinstance(prenatal_result, FetalFractionResult):
        object.__setattr__(prenatal_result, "lod_metrics", lod)
    return list(annotated)
