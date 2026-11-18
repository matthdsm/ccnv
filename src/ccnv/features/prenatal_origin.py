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
from ccnv.features.fragmentomics import resolve_fisher_moment_weights
from ccnv.models import (
    NUM_FRAG_BANDS,
    FetalFractionResult,
    PrenatalLoDMetrics,
    PrenatalModelMetadata,
    SampleBuffers,
    SCACall,
    SegmentResult,
    normalize_chr,
    reconstruct_sample_grid,
)
from ccnv.stats.fragmentomics import FRAG_MIN_LEN, FRAG_NUM_BINS

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
_SHORT_FRAG_BAND_STOP: Final[int] = int(
    np.searchsorted(FRAG_BAND_EDGES, 150, side="right") - 1
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


def _to_24_band_density(
    density: Sequence[float] | NDArray[np.floating],
) -> NDArray[np.float64]:
    """Convert a 24-band or 571-bin fragment length density into canonical 24 bands."""
    p_arr = np.asarray(density, dtype=np.float64).ravel()
    if p_arr.size == NUM_FRAG_BANDS:
        res = np.clip(p_arr, 1e-8, None)
        return res / float(np.sum(res))
    if p_arr.size != FRAG_NUM_BINS:
        raise ValueError(
            f"fragment density must contain {NUM_FRAG_BANDS} or {FRAG_NUM_BINS} "
            f"elements, got {p_arr.size}"
        )

    offsets = np.asarray(FRAG_BAND_EDGES, dtype=np.int64) - FRAG_MIN_LEN
    p_24 = np.add.reduceat(p_arr[: int(offsets[-1])], offsets[:-1])
    p_24 = np.clip(p_24, 1e-8, None)
    return p_24 / float(np.sum(p_24))


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
    jacobian = max(abs(delta_cn) * depth_scale, 1e-6)
    denom = delta_cn * depth_scale
    m_star = (
        float(np.clip((r_obs - 1.0) / denom, m_low, m_high))
        if abs(denom) > 1e-9
        else 0.5 * (m_low + m_high)
    )
    r_star = 1.0 + denom * m_star
    sig_eff = math.hypot(sigma_r, abs(delta_cn) * m_star * f_se)
    z_boundary = abs(r_obs - r_star) / max(sig_eff, 1e-6)
    if z_boundary > 1.0:
        width = max(12.0 * (sig_eff / jacobian) / z_boundary, 1e-6)
    else:
        sig_hi = math.hypot(
            sigma_r,
            abs(delta_cn) * min(m_high, m_star + 6.0 * (sig_eff / jacobian)) * f_se,
        )
        width = 6.0 * (sig_hi / jacobian)
    a = max(m_low, m_star - width)
    b = min(m_high, m_star + width)
    if b <= a:
        a, b = m_low, m_high
    half_span = 0.5 * (b - a)
    m_nodes = half_span * nodes_x + 0.5 * (b + a)
    m_weights = half_span * weights_w

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
    p_fetal_24: Sequence[float] | NDArray[np.floating],
    p_maternal_24: Sequence[float] | NDArray[np.floating],
    p_plasma_0: Sequence[float] | NDArray[np.floating],
    active_ff: float,
    active_ff_se: float,
    is_gain: bool,
    *,
    depth_ff: float | None = None,
    twin_min_ff: float | None = None,
    alpha_dm: float = 10000.0,
    sigma_r: float = 0.005,
) -> tuple[str, dict[str, float], float | None, tuple[float, float] | None]:
    """Classify segment biological origin across 5 coupled Depth + DM hypotheses."""
    k_arr = np.asarray(k_band_counts, dtype=np.float64).ravel()
    if k_arr.size != NUM_FRAG_BANDS:
        raise ValueError(
            f"k_band_counts must contain {NUM_FRAG_BANDS} elements, got {k_arr.size}"
        )
    pf_24 = _to_24_band_density(p_fetal_24)
    pm_24 = _to_24_band_density(p_maternal_24)
    p0_24 = _to_24_band_density(p_plasma_0)

    cn_alt = 3.0 if is_gain else 1.0
    delta_cn = cn_alt / 2.0 - 1.0
    f_bio = float(np.clip(active_ff, 0.001, 0.60))
    f_dep = float(np.clip(depth_ff if depth_ff is not None else active_ff, 0.001, 0.60))
    f_se = max(float(active_ff_se) * (f_dep / f_bio), 0.001)

    cand_ffs: tuple[float, ...] = (f_bio,)
    if twin_min_ff is not None and twin_min_ff > 0.0:
        f_1 = float(np.clip(twin_min_ff, 0.001, f_bio))
        f_2 = float(np.clip(f_bio - f_1, 0.001, f_bio))
        cand_ffs = (f_bio, f_1, f_2)

    ll_fc = -math.inf
    for f_aff in cand_ffs:
        f_aff_dep = f_dep * (f_aff / f_bio)
        r_exp_aff = 1.0 + delta_cn * f_aff_dep
        phi_aff = float(
            np.clip(
                (f_bio + delta_cn * f_aff) / max(1.0 + delta_cn * f_aff, 1e-6),
                1e-6,
                1.0 - 1e-6,
            )
        )
        p_aff = phi_aff * pf_24 + (1.0 - phi_aff) * pm_24
        se_aff = (
            abs(delta_cn) * f_se
            if f_aff == f_bio
            else abs(delta_cn) * math.hypot(f_se * (f_aff / f_bio), 0.04 * f_dep)
        )
        ll_cand = _gaussian_depth_loglik(
            r_obs, r_exp_aff, sigma_r, se_aff
        ) + _dirichlet_multinomial_logpmf(k_arr, alpha_dm * p_aff)
        ll_fc = max(ll_fc, ll_cand)

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
        k_band_counts=k_arr,
        p_fetal_24=pf_24,
        p_maternal_24=pm_24,
        alpha_dm=alpha_dm,
    )

    r_exp_mu = 1.0 + delta_cn * (1.0 - f_dep)
    phi_mu = f_bio / (cn_alt / 2.0 * (1.0 - f_bio) + f_bio)
    p_mu = phi_mu * pf_24 + (1.0 - phi_mu) * pm_24
    ll_mu = _gaussian_depth_loglik(
        r_obs, r_exp_mu, sigma_r, abs(delta_cn) * f_se
    ) + _dirichlet_multinomial_logpmf(k_arr, alpha_dm * p_mu)

    ll_mfi = _gaussian_depth_loglik(
        r_obs, cn_alt / 2.0, sigma_r, 0.0
    ) + _dirichlet_multinomial_logpmf(k_arr, alpha_dm * p0_24)

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
        k_band_counts=k_arr,
        p_fetal_24=pf_24,
        p_maternal_24=pm_24,
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
    p_fetal_density: Sequence[float] | NDArray[np.floating] | None = None,
    p_maternal_density: Sequence[float] | NDArray[np.floating] | None = None,
    mu_fetal_w: float = 1.0,
    mu_maternal_w: float = 1.0,
    active_ff: float = 0.0,
    active_ff_se: float = 0.01,
    sca_call: SCACall = "XX",
    sample_depth_per_mb: float = 10000.0,
    sys_var_per_mb: float = 0.0001,
    z_threshold: float = 3.0,
    twin_min_ff: float | None = None,
) -> tuple[tuple[SegmentResult, ...], PrenatalLoDMetrics]:
    """Annotate segments with expected ratio, implied FF, discordance, 5-state origin, and LoD."""
    f_act = float(np.clip(active_ff, 0.0, 0.60))
    lod_scale = (
        max(f_act / max(float(twin_min_ff), 1e-4), 1.0)
        if (twin_min_ff is not None and f_act > 0.0)
        else 1.0
    )
    z_target = max(float(z_threshold), 0.1) + 2.3263479
    d_mb = max(float(sample_depth_per_mb), 1.0)
    v_sys = max(float(sys_var_per_mb), 0.0)
    sig_35mb = math.sqrt(1.0 / (d_mb * 35.0) + v_sys / 35.0)
    sig_3mb = math.sqrt(1.0 / (d_mb * 3.0) + v_sys / 3.0)
    lod_ff_35mb = float(np.clip(2.0 * z_target * sig_35mb * lod_scale, 0.005, 0.49))
    lod_ff_3mb = float(
        np.clip(2.0 * z_target * sig_3mb * lod_scale, lod_ff_35mb + 0.001, 0.499)
    )
    lod_metrics = PrenatalLoDMetrics(lod_ff_35mb=lod_ff_35mb, lod_ff_3mb=lod_ff_3mb)

    uniform_24 = np.full(NUM_FRAG_BANDS, 1.0 / NUM_FRAG_BANDS, dtype=np.float64)
    p_f24 = (
        _to_24_band_density(p_fetal_density)
        if p_fetal_density is not None
        else uniform_24
    )
    if p_maternal_density is not None:
        p_m24 = _to_24_band_density(p_maternal_density)
        p0_plasma = f_act * p_f24 + (1.0 - f_act) * p_m24
    elif sample_bands is not None and sample_bands.size > 0:
        usable_idx = (
            np.flatnonzero(bin_usable)
            if bin_usable is not None
            else np.arange(sample_bands.shape[0])
        )
        sum_bands = np.sum(sample_bands[usable_idx], axis=0, dtype=np.float64)
        total_k = float(np.sum(sum_bands))
        p0_plasma = sum_bands / max(total_k, 1.0) if total_k > 0 else uniform_24
        p_m_raw = np.clip(p0_plasma - f_act * p_f24, 1e-8, None)
        p_m24 = p_m_raw / float(np.sum(p_m_raw))
    else:
        p0_plasma = uniform_24
        p_m24 = uniform_24

    mu_f = max(float(mu_fetal_w), 1e-4)
    mu_m = max(float(mu_maternal_w), 1e-4)
    f_eff = float(
        np.clip((f_act * mu_f) / ((1.0 - f_act) * mu_m + f_act * mu_f), 0.0, 0.60)
    )
    f_min_eff = (
        float(np.clip(f_eff * (float(twin_min_ff) / max(f_act, 1e-6)), 0.0, f_eff))
        if (twin_min_ff is not None and f_act > 0.0)
        else f_eff
    )
    log_ratio_fm = np.log(np.maximum(p_f24, 1e-12)) - np.log(np.maximum(p_m24, 1e-12))

    annotated: list[SegmentResult] = []
    is_aneuploid = sca_call not in ("XX", "XY")
    total_y_bins = sum(
        max(0, s.bin_count) for s in segments if normalize_chr(s.contig) == "Y"
    )

    for seg in segments:
        norm_contig = normalize_chr(seg.contig)
        is_y = norm_contig == "Y"
        is_gonosome = norm_contig in {"X", "Y"}
        has_call = (seg.call is not None) or (abs(seg.zscore) >= z_threshold)

        expected_ratio: float | None = None
        implied_ff: float | None = None
        discordant = False
        reason: str | None = None

        if is_y:
            whole_y_call = (
                seg.call is not None
                and total_y_bins > 0
                and seg.bin_count >= 0.75 * total_y_bins
            )
            if (
                sca_call in ("XYY", "DISCORDANT_MIXED")
                or (sca_call in ("XX", "XO", "XXX") and has_call)
                or whole_y_call
            ):
                discordant = True
                reason = "SCA_DISCORDANT"
        else:
            if f_min_eff <= 0.0:
                exp_val = 0.0
            elif seg.ratio >= 0.0:
                exp_val = math.log2(1.0 + 0.5 * f_min_eff)
            else:
                exp_val = math.log2(max(1e-6, 1.0 - 0.5 * f_min_eff))
            expected_ratio = float(exp_val)

            r_obs = 2.0**seg.ratio
            implied_eff = 2.0 * abs(r_obs - 1.0)
            if implied_eff < 0.9999 and abs(mu_f - mu_m) > 1e-9:
                imp_val = float(
                    (implied_eff * mu_m)
                    / max((1.0 - implied_eff) * mu_f + implied_eff * mu_m, 1e-12)
                )
            else:
                imp_val = float(implied_eff)
            implied_ff = imp_val

            if is_gonosome and (is_aneuploid or has_call):
                discordant = True
                reason = "SCA_DISCORDANT"
            elif has_call and abs(seg.ratio) < 0.5 * abs(exp_val):
                discordant = True
                reason = "SUB_FETAL_RATIO"
            elif has_call and imp_val > max(2.5 * f_act, f_act + 0.15):
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
                frag_short_ratio = float(
                    np.sum(k_s[:_SHORT_FRAG_BAND_STOP]) / total_seg_k
                )
                frag_llr_size = float(np.sum(k_s * log_ratio_fm))
                if has_call and not is_y:
                    r_obs = 2.0**seg.ratio
                    is_gain = (
                        seg.call == "gain"
                        if seg.call is not None
                        else (seg.ratio >= 0.0)
                    )
                    sigma_log2r = (
                        abs(seg.ratio / seg.zscore)
                        if abs(seg.zscore) > 1e-6
                        else 1.0 / math.sqrt(max(seg.weight, 1e-6))
                    )
                    sigma_r = max(math.log(2.0) * r_obs * sigma_log2r, 0.002)
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
                            twin_min_ff=twin_min_ff,
                            sigma_r=sigma_r,
                        )
                    )

        annotated.append(
            replace(
                seg,
                expected_ratio=expected_ratio,
                implied_ff=implied_ff,
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
    sample: SampleBuffers,
    *,
    bin_usable: NDArray[np.bool_] | None = None,
    prenatal_meta: PrenatalModelMetadata | None = None,
    prenatal_result: FetalFractionResult,
    z_threshold: float = 3.0,
) -> tuple[tuple[SegmentResult, ...], PrenatalLoDMetrics | None]:
    """Annotate sample segments using prenatal model metadata and sample band prefix sums."""
    bufs = SampleBuffers.coerce(sample)
    prefix_sums = BandPrefixSums.from_sample_table(bufs, bin_usable=bin_usable)
    total_bp = sum(bufs.meta.contig_lengths.values())
    depth_per_mb = float(np.sum(bufs.count)) / max(total_bp / 1e6, 1e-6)
    mu_f, mu_m = resolve_fisher_moment_weights(bufs, prenatal_meta)
    f_twin_min: float | None = None
    if prenatal_result.gestation == "twin":
        if (
            prenatal_result.fetal_sexes == ("F", "M")
            and prenatal_result.male_ff is not None
            and prenatal_result.female_ff is not None
        ):
            f_twin_min = max(
                float(min(prenatal_result.male_ff, prenatal_result.female_ff)),
                min(0.01, 0.5 * float(prenatal_result.active_ff)),
            )
        else:
            f_twin_min = 0.5 * float(prenatal_result.active_ff)
    return annotate_prenatal_origin_and_lod(
        segments,
        prefix_sums=prefix_sums,
        sample_bands=bufs.frag_bands,
        bin_usable=bin_usable,
        p_fetal_density=(
            prenatal_meta.p_fetal_density if prenatal_meta is not None else None
        ),
        p_maternal_density=(
            prenatal_meta.p_maternal_density if prenatal_meta is not None else None
        ),
        mu_fetal_w=mu_f,
        mu_maternal_w=mu_m,
        active_ff=prenatal_result.active_ff,
        active_ff_se=(
            prenatal_result.active_ff_se
            if prenatal_result.active_ff_se is not None
            else 0.01
        ),
        sca_call=prenatal_result.sca.call,
        sample_depth_per_mb=depth_per_mb,
        z_threshold=z_threshold,
        twin_min_ff=f_twin_min,
    )
