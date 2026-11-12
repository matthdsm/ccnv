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
    reconstruct_sample_grid,
)
from ccnv.stats.fragmentomics import FRAG_MIN_LEN

__all__ = [
    "BandPrefixSums",
    "PrenatalLoDMetrics",
    "annotate_prenatal_origin_and_lod",
    "annotate_sample_prenatal_segments",
    "classify_segment_origin_5state",
]

_QUAD_NODES_16: Final[tuple[NDArray[np.float64], NDArray[np.float64]]] = (
    np.polynomial.legendre.leggauss(16)
)
_STATE_LOG_PRIOR: Final[float] = math.log(0.20)


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
            idx = np.flatnonzero(bin_contigs == c_idx)
            if idx.size == 0:
                continue
            starts_c = np.asarray(bin_starts[idx], dtype=np.int64)
            ends_c = np.asarray(bin_ends[idx], dtype=np.int64)
            cum_c = np.zeros((idx.size + 1, NUM_FRAG_BANDS), dtype=np.float64)
            cum_c[1:] = np.cumsum(bands_f64[idx], axis=0)
            self._contig_data[norm_name] = (starts_c, ends_c, cum_c)

    @classmethod
    def from_sample_table(
        cls,
        sample: SampleBuffers | pa.Table,
        bin_usable: NDArray[np.bool_] | None = None,
    ) -> BandPrefixSums | None:
        """Construct BandPrefixSums engine from SampleBuffers or pa.Table."""
        bufs = SampleBuffers.coerce(sample)
        if bufs.num_rows == 0 or bufs.frag_bands is None:
            return None
        codes, names, starts_u32, ends_u32 = reconstruct_sample_grid(bufs.meta)
        return cls(
            sample_bands=bufs.frag_bands,
            bin_contigs=np.asarray(codes, dtype=np.int16),
            contig_names=names,
            bin_starts=np.asarray(starts_u32, dtype=np.int64),
            bin_ends=np.asarray(ends_u32, dtype=np.int64),
            bin_usable=bin_usable,
        )

    def query(self, contig: str, start: int, end: int) -> NDArray[np.float64]:
        """Query 24-band cumulative count vector in [start, end) on contig in O(log U)."""
        data = self._contig_data.get(normalize_chr(contig))
        if data is None:
            return np.zeros(NUM_FRAG_BANDS, dtype=np.float64)

        starts_c, ends_c, cum_c = data
        i_start = int(np.searchsorted(ends_c, start, side="right"))
        i_end = int(np.searchsorted(starts_c, end, side="left"))
        if i_start >= i_end:
            return np.zeros(NUM_FRAG_BANDS, dtype=np.float64)
        return np.asarray(cum_c[i_end] - cum_c[i_start], dtype=np.float64)


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
    return float(
        gammaln(alpha_sum)
        - gammaln(total_n + alpha_sum)
        + float(np.sum(gammaln(k_pos + alpha_pos) - gammaln(alpha_pos)))
    )


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
    if p_arr.size == NUM_FRAG_BANDS:
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


def _integrate_mosaic_state(
    m_low: float,
    m_high: float,
    delta_cn: float,
    depth_scale: float,
    f_bio: float,
    f_se: float,
    *,
    is_fetal: bool,
    r_obs: float,
    sigma_r: float,
    k_band_counts: NDArray[np.float64],
    p_fetal_24: NDArray[np.float64],
    p_maternal_24: NDArray[np.float64],
    alpha_dm: float,
) -> tuple[float, float, float]:
    """Integrate coupled depth + DM mosaic likelihood over [m_low, m_high] via 16-node Gauss-Legendre quadrature."""
    nodes_x, weights_w = _QUAD_NODES_16
    half_span = 0.5 * (m_high - m_low)
    m_nodes = half_span * nodes_x + 0.5 * (m_high + m_low)
    m_weights = half_span * weights_w
    jacobian = abs(delta_cn) * depth_scale

    ll_nodes = np.empty(m_nodes.size, dtype=np.float64)
    for idx, m in enumerate(m_nodes):
        r_exp_m = 1.0 + delta_cn * m * depth_scale
        if is_fetal:
            f_mult = (1.0 + delta_cn * m) * f_bio
            phi_m = f_mult / (1.0 - f_bio + f_mult)
        else:
            phi_m = f_bio / ((1.0 + delta_cn * m) * (1.0 - f_bio) + f_bio)
        p_m_st = phi_m * p_fetal_24 + (1.0 - phi_m) * p_maternal_24
        ll_nodes[idx] = _gaussian_depth_loglik(
            r_obs, r_exp_m, sigma_r, abs(delta_cn) * m * f_se
        ) + _dirichlet_multinomial_logpmf(k_band_counts, alpha_dm * p_m_st)

    max_ll = float(np.max(ll_nodes))
    unnorm = np.exp(ll_nodes - max_ll)
    integral = float(np.sum(m_weights * unnorm))
    log_ev = max_ll + math.log(max(integral, 1e-300)) + math.log(max(jacobian, 1e-6))
    m_hat = float(np.sum(m_weights * m_nodes * unnorm) / max(integral, 1e-300))
    var_m = float(
        np.sum(m_weights * np.square(m_nodes - m_hat) * unnorm) / max(integral, 1e-300)
    )
    return log_ev, m_hat, var_m


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
    depth_ff: float | None = None,
    alpha_dm: float = 10000.0,
    sigma_r: float = 0.005,
) -> tuple[str, dict[str, float], float | None, tuple[float, float] | None]:
    """Classify segment biological origin across 5 coupled Depth + DM hypotheses."""
    cn_alt = 3.0 if is_gain else 1.0
    delta_cn = cn_alt / 2.0 - 1.0
    f_bio = float(np.clip(active_ff, 0.001, 0.60))
    f_dep = float(np.clip(depth_ff if depth_ff is not None else active_ff, 0.001, 0.60))
    f_se = max(float(active_ff_se) * (f_dep / f_bio), 0.001)

    r_exp_fc = 1.0 + delta_cn * f_dep
    phi_fc = (cn_alt / 2.0 * f_bio) / (1.0 - f_bio + cn_alt / 2.0 * f_bio)
    p_fc = phi_fc * p_fetal_24 + (1.0 - phi_fc) * p_maternal_24
    ll_fc = _gaussian_depth_loglik(
        r_obs, r_exp_fc, sigma_r, abs(delta_cn) * f_se
    ) + _dirichlet_multinomial_logpmf(k_band_counts, alpha_dm * p_fc)

    ll_cpm, m_hat_cpm, var_cpm = _integrate_mosaic_state(
        0.05,
        0.95,
        delta_cn,
        f_dep,
        f_bio,
        f_se,
        is_fetal=True,
        r_obs=r_obs,
        sigma_r=sigma_r,
        k_band_counts=k_band_counts,
        p_fetal_24=p_fetal_24,
        p_maternal_24=p_maternal_24,
        alpha_dm=alpha_dm,
    )

    r_exp_mu = 1.0 + delta_cn * (1.0 - f_dep)
    phi_mu = f_bio / (cn_alt / 2.0 * (1.0 - f_bio) + f_bio)
    p_mu = phi_mu * p_fetal_24 + (1.0 - phi_mu) * p_maternal_24
    ll_mu = _gaussian_depth_loglik(
        r_obs, r_exp_mu, sigma_r, abs(delta_cn) * f_se
    ) + _dirichlet_multinomial_logpmf(k_band_counts, alpha_dm * p_mu)

    ll_mfi = _gaussian_depth_loglik(
        r_obs, cn_alt / 2.0, sigma_r, 0.0
    ) + _dirichlet_multinomial_logpmf(k_band_counts, alpha_dm * p_plasma_0)

    ll_mm, m_hat_mat, var_mat = _integrate_mosaic_state(
        0.01,
        0.95,
        delta_cn,
        1.0 - f_dep,
        f_bio,
        f_se,
        is_fetal=False,
        r_obs=r_obs,
        sigma_r=sigma_r,
        k_band_counts=k_band_counts,
        p_fetal_24=p_fetal_24,
        p_maternal_24=p_maternal_24,
        alpha_dm=alpha_dm,
    )

    log_joints: dict[str, float] = {
        "FETAL_CONSTITUTIONAL": ll_fc + _STATE_LOG_PRIOR,
        "PLACENTAL_MOSAIC": ll_cpm + _STATE_LOG_PRIOR,
        "MATERNAL_UNINHERITED": ll_mu + _STATE_LOG_PRIOR,
        "MATERNAL_AND_FETAL_INHERITED": ll_mfi + _STATE_LOG_PRIOR,
        "MATERNAL_MOSAIC": ll_mm + _STATE_LOG_PRIOR,
    }
    max_lj = max(log_joints.values())
    unnorm_post = {k: math.exp(v - max_lj) for k, v in log_joints.items()}
    sum_post = sum(unnorm_post.values())
    posteriors = {k: unnorm_post[k] / sum_post for k in unnorm_post}

    best_state = max(posteriors.keys(), key=lambda k: posteriors[k])
    call = "INDETERMINATE" if posteriors[best_state] < 0.80 else best_state

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
    *,
    prefix_sums: BandPrefixSums | None = None,
    sample_bands: NDArray[np.uint32] | None = None,
    bin_usable: NDArray[np.bool_] | None = None,
    p_fetal_24: Sequence[float] | NDArray[np.floating] | None = None,
    p_maternal_24: Sequence[float] | NDArray[np.floating] | None = None,
    mu_fetal_w: float = 1.0,
    mu_maternal_w: float = 1.0,
    active_ff: float = 0.0,
    active_ff_se: float = 0.01,
    sca_call: SCACall = "XX",
    sample_depth_per_mb: float = 10000.0,
    sys_var_per_mb: float = 0.0001,
    z_threshold: float = 3.0,
) -> tuple[tuple[SegmentResult, ...], PrenatalLoDMetrics]:
    """Annotate segments with expected ratio, implied FF, discordance, 5-state origin, and LoD."""
    z_target = max(float(z_threshold), 0.1) + 2.3263479
    d_mb = max(float(sample_depth_per_mb), 1.0)
    v_sys = max(float(sys_var_per_mb), 0.0)
    sig_35mb = math.sqrt(1.0 / (d_mb * 35.0) + v_sys / 35.0)
    sig_3mb = math.sqrt(1.0 / (d_mb * 3.0) + v_sys / 3.0)
    lod_ff_35mb = float(np.clip(2.0 * z_target * sig_35mb, 0.005, 0.49))
    lod_ff_3mb = float(np.clip(2.0 * z_target * sig_3mb, lod_ff_35mb + 0.001, 0.499))
    lod_metrics = PrenatalLoDMetrics(lod_ff_35mb=lod_ff_35mb, lod_ff_3mb=lod_ff_3mb)

    uniform_24 = np.full(NUM_FRAG_BANDS, 1.0 / NUM_FRAG_BANDS, dtype=np.float64)
    p_f24 = (
        _convert_571_to_24_bands(p_fetal_24) if p_fetal_24 is not None else uniform_24
    )
    if p_maternal_24 is not None:
        p_m24 = _convert_571_to_24_bands(p_maternal_24)
        p0_plasma = active_ff * p_f24 + (1.0 - active_ff) * p_m24
    elif sample_bands is not None and sample_bands.size > 0:
        usable_idx = (
            np.flatnonzero(bin_usable)
            if bin_usable is not None and np.any(bin_usable)
            else np.arange(sample_bands.shape[0])
        )
        sum_bands = np.sum(sample_bands[usable_idx], axis=0, dtype=np.float64)
        total_k = float(np.sum(sum_bands))
        p0_plasma = sum_bands / max(total_k, 1.0) if total_k > 0 else uniform_24
        p_m_raw = np.clip(p0_plasma - active_ff * p_f24, 1e-8, None)
        p_m24 = p_m_raw / np.sum(p_m_raw)
    else:
        p0_plasma = uniform_24
        p_m24 = uniform_24

    f_act = max(float(active_ff), 0.0)
    mu_f = max(float(mu_fetal_w), 1e-4)
    mu_m = max(float(mu_maternal_w), 1e-4)
    f_eff = float(
        np.clip((f_act * mu_f) / ((1.0 - f_act) * mu_m + f_act * mu_f), 0.0, 0.60)
    )
    log_ratio_fm = np.log(np.maximum(p_f24, 1e-12)) - np.log(np.maximum(p_m24, 1e-12))

    annotated: list[SegmentResult] = []
    is_aneuploid = sca_call not in ("XX", "XY")

    for seg in segments:
        if f_eff <= 0.0:
            expected_ratio = 0.0
        elif seg.ratio >= 0.0:
            expected_ratio = math.log2(1.0 + 0.5 * f_eff)
        else:
            expected_ratio = math.log2(max(1e-6, 1.0 - 0.5 * f_eff))

        r_obs = 2.0**seg.ratio
        implied_eff = 2.0 * abs(r_obs - 1.0)
        if implied_eff < 0.9999 and abs(mu_f - mu_m) > 1e-9:
            implied_ff = float(
                (implied_eff * mu_m)
                / max((1.0 - implied_eff) * mu_f + implied_eff * mu_m, 1e-12)
            )
        else:
            implied_ff = float(implied_eff)

        is_gonosome = normalize_chr(seg.contig) in {"X", "Y"}
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
                frag_llr_size = float(np.sum(k_s * log_ratio_fm))
                if has_call:
                    is_gain = (
                        seg.call == "gain"
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
                            active_ff=f_act,
                            active_ff_se=active_ff_se,
                            is_gain=is_gain,
                            depth_ff=f_eff,
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


def annotate_sample_prenatal_segments(
    segments: Sequence[SegmentResult],
    sample: SampleBuffers | pa.Table,
    *,
    bin_usable: NDArray[np.bool_] | None = None,
    prenatal_meta: PrenatalModelMetadata | None = None,
    prenatal_result: FetalFractionResult,
    z_threshold: float = 3.0,
) -> tuple[tuple[SegmentResult, ...], PrenatalLoDMetrics]:
    """Annotate sample segments using prenatal model metadata and sample band prefix sums."""
    bufs = SampleBuffers.coerce(sample)
    prefix_sums = BandPrefixSums.from_sample_table(bufs, bin_usable=bin_usable)
    total_bp = sum(bufs.meta.contig_lengths.values())
    depth_per_mb = float(np.sum(bufs.count)) / max(total_bp / 1e6, 1e-6)
    return annotate_prenatal_origin_and_lod(
        segments,
        prefix_sums=prefix_sums,
        sample_bands=bufs.frag_bands,
        bin_usable=bin_usable,
        p_fetal_24=prenatal_meta.p_fetal_density if prenatal_meta is not None else None,
        p_maternal_24=(
            prenatal_meta.p_maternal_density if prenatal_meta is not None else None
        ),
        mu_fetal_w=(
            float(prenatal_meta.mu_fetal_w)
            if prenatal_meta is not None and prenatal_meta.mu_fetal_w is not None
            else 1.0
        ),
        mu_maternal_w=(
            float(prenatal_meta.mu_maternal_w)
            if prenatal_meta is not None and prenatal_meta.mu_maternal_w is not None
            else 1.0
        ),
        active_ff=prenatal_result.active_ff,
        active_ff_se=(
            prenatal_result.active_ff_se
            if prenatal_result.active_ff_se is not None
            else 0.01
        ),
        sca_call=prenatal_result.sca.call,
        sample_depth_per_mb=depth_per_mb,
        z_threshold=z_threshold,
    )
