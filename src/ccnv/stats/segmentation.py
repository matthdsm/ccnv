"""Python-native weighted copy-number segmentation and segment scoring.

The ``penalty`` is measured in weighted squared-error units. PELT minimizes
``sum(weighted squared error) + penalty * number of breakpoints``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import erfc

from ccnv._core import pelt_weighted_l2
from ccnv.models import AnalysisMode, CallType
from ccnv.models.exceptions import InvalidInputError
from ccnv.models.results import SegmentResult
from ccnv.stats.neighbors import (
    LN2_SQ,
    exact_poisson_log2_jensen_bias,
    exact_poisson_log_variance,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_BLACKLIST_GAP: Final[int] = 2_000_000
_DEFAULT_DYADIC_SCALES: Final[tuple[int, ...]] = (1, 4, 16, 64, 256, 1024)
_DEFAULT_LOG2_SCALES: Final[NDArray[np.float64]] = np.log2(
    np.asarray(_DEFAULT_DYADIC_SCALES, dtype=np.float64)
)
_DIPLOID_FALLBACK_LOG2: Final[dict[int, float]] = {
    0: -4.5,
    1: -1.0,
    3: math.log2(1.5),
    4: 1.0,
}
_HAPLOID_FALLBACK_LOG2: Final[dict[int, float]] = {
    0: -4.5,
    2: 1.0,
    3: math.log2(3.0),
}

__all__ = [
    "DEFAULT_MAX_BLACKLIST_GAP",
    "ContigSegmentationInput",
    "WeightedL2Cost",
    "estimate_dyadic_inflation",
    "interpolate_dyadic_inflation",
    "segment_and_score_genome",
    "segment_contig",
]


@dataclass(frozen=True, slots=True)
class ContigSegmentationInput:
    """Typed per-contig array inputs for genome segmentation and BLUE scoring."""

    starts: ArrayLike
    ends: ArrayLike
    ratios: ArrayLike
    weights: ArrayLike
    contig_length: int
    zscores: ArrayLike | None = None
    counts: ArrayLike | None = None
    expected: ArrayLike | None = None
    background_expected: ArrayLike | None = None
    baseline_ploidy: int = 2
    baseline_log2_var: float = 0.0
    poisson_pooled: bool = False

    def __post_init__(self) -> None:
        """Validate baseline ploidy and baseline variance state."""
        if isinstance(self.baseline_ploidy, bool) or self.baseline_ploidy not in (1, 2):
            raise InvalidInputError(
                f"baseline_ploidy must be 1 or 2, got {self.baseline_ploidy!r}"
            )
        if not math.isfinite(self.baseline_log2_var) or self.baseline_log2_var < 0.0:
            raise InvalidInputError("baseline_log2_var must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class _PreScoredSegment:
    """Internal per-contig segment metrics prior to genome-wide BH-FDR adjustment."""

    contig: str
    start: int
    end: int
    ratio: float
    weight: float
    n_usable: int
    zscore: float
    baseline_ploidy: int
    cand_cn: int | None
    log_bf: float | None


class WeightedL2Cost:
    """Weighted residual sum-of-squares segment cost."""

    model = "weighted_l2"
    min_size = 1

    def __init__(self) -> None:
        """Initialize empty weighted L2 cost accumulator."""
        self.signal = np.empty((0, 2), dtype=np.float64)
        self._n = 0
        self._weight_sum: NDArray[np.float64] = np.zeros(1, dtype=np.float64)
        self._weighted_value_sum: NDArray[np.float64] = np.zeros(1, dtype=np.float64)
        self._weighted_square_sum: NDArray[np.float64] = np.zeros(1, dtype=np.float64)
        self._w_list: list[float] = [0.0]
        self._wv_list: list[float] = [0.0]
        self._wsq_list: list[float] = [0.0]

    def fit(self, signal: ArrayLike) -> WeightedL2Cost:
        """Fit contiguous prefix sums from a 2D ``(N, 2)`` ``(values, weights)`` matrix."""
        array = np.asarray(signal, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 2:
            raise InvalidInputError(
                "signal must be a 2D array of shape (N, 2) containing (values, weights)"
            )
        values = array[:, 0]
        weight_array = array[:, 1]
        if np.any(np.isfinite(weight_array) & (weight_array < 0.0)):
            raise InvalidInputError("weights must be non-negative")

        usable = np.isfinite(values) & np.isfinite(weight_array) & (weight_array > 0.0)
        clean_weights = np.where(usable, weight_array, 0.0)
        clean_values = np.where(usable, values, 0.0)

        total_w = float(np.sum(clean_weights))
        mu_0 = (
            float(np.sum(clean_weights * clean_values) / total_w)
            if total_w > 0.0
            else 0.0
        )
        shifted = np.where(usable, clean_values - mu_0, 0.0)

        self.signal = np.column_stack((clean_values, clean_weights))
        self._n = int(clean_values.size)
        self._weight_sum = _cumulative(clean_weights)
        self._weighted_value_sum = _cumulative(clean_weights * shifted)
        self._weighted_square_sum = _cumulative(clean_weights * np.square(shifted))
        self._w_list = self._weight_sum.tolist()
        self._wv_list = self._weighted_value_sum.tolist()
        self._wsq_list = self._weighted_square_sum.tolist()
        return self

    def error(self, start: int, end: int) -> float:
        """Return weighted residual sum of squares for ``[start, end)``."""
        if start < 0 or end > self._n or start >= end:
            raise InvalidInputError("cost interval must be a non-empty in-bounds slice")
        weight = self._w_list[end] - self._w_list[start]
        if weight <= 0.0:
            return 0.0
        weighted_val = self._wv_list[end] - self._wv_list[start]
        weighted_sq = self._wsq_list[end] - self._wsq_list[start]
        rss = weighted_sq - (weighted_val * weighted_val) / weight
        return max(0.0, rss)

    def sum_of_costs(self, bkps: Sequence[int]) -> float:
        """Return sum of weighted segment errors across breakpoint end indices."""
        total = 0.0
        start = 0
        for end in bkps:
            total += self.error(start, end)
            start = end
        return total


def estimate_dyadic_inflation(
    zscores: NDArray[np.float64],
    usable: NDArray[np.bool_],
    *,
    scales: Sequence[int] = _DEFAULT_DYADIC_SCALES,
) -> NDArray[np.float64]:
    """Estimate scale-dependent variance inflation kappa(L) over unmasked runs."""
    z = np.ravel(np.asarray(zscores, dtype=np.float64))
    u = np.ravel(np.asarray(usable, dtype=np.bool_))
    n_scales = len(scales)
    if z.size != u.size or z.size == 0 or n_scales == 0:
        return np.ones(n_scales, dtype=np.float64)

    valid_mask = u & np.isfinite(z)
    z_eval = np.where(valid_mask, np.clip(z, -2.5, 2.5), 0.0)
    padded = np.concatenate(([False], valid_mask, [False]))
    diff = np.diff(padded.astype(np.int8))
    run_starts = np.flatnonzero(diff == 1)
    run_ends = np.flatnonzero(diff == -1)
    run_lengths = run_ends - run_starts

    cumsum_z = _cumulative(z_eval)
    inflations = np.empty(n_scales, dtype=np.float64)
    prev_kappa = 1.0

    for s_idx, raw_scale in enumerate(scales):
        scale = int(raw_scale)
        if scale <= 0:
            raise InvalidInputError("dyadic scales must be positive integers")
        keep_runs = np.flatnonzero(run_lengths >= scale)
        if keep_runs.size > 0:
            slices = [
                np.arange(
                    int(run_starts[r_i]),
                    int(run_starts[r_i]) + int(run_lengths[r_i] // scale) * scale,
                    scale,
                    dtype=np.int64,
                )
                for r_i in keep_runs
            ]
            b_starts = np.concatenate(slices)
            if b_starts.size >= 5:
                block_vals = (
                    cumsum_z[b_starts + scale] - cumsum_z[b_starts]
                ) / math.sqrt(float(scale))
                kappa_l = max(1.0, float(np.var(block_vals, ddof=1)))
            else:
                kappa_l = prev_kappa
        else:
            kappa_l = prev_kappa

        if s_idx > 0 and kappa_l < inflations[s_idx - 1]:
            kappa_l = float(inflations[s_idx - 1])
        inflations[s_idx] = kappa_l
        prev_kappa = kappa_l

    return inflations


def interpolate_dyadic_inflation(
    n_bins: float,
    inflation_values: Sequence[float] | NDArray[np.float64],
    *,
    scales: Sequence[int] = _DEFAULT_DYADIC_SCALES,
) -> float:
    """Interpolate kappa(N) monotonically in log2 scale space."""
    infl_arr = np.asarray(inflation_values, dtype=np.float64)
    if infl_arr.size == 0:
        return 1.0
    n = max(1.0, float(n_bins))
    if n <= scales[0]:
        return max(1.0, float(infl_arr[0]))
    if n >= scales[-1]:
        return max(1.0, float(infl_arr[-1]))
    log_scales = (
        _DEFAULT_LOG2_SCALES
        if tuple(scales) == _DEFAULT_DYADIC_SCALES
        else np.log2(np.asarray(scales, dtype=np.float64))
    )
    val = float(np.interp(math.log2(n), log_scales, infl_arr))
    return max(1.0, val)


def _segment_bin_spans(
    start_array: NDArray[np.int64],
    end_array: NDArray[np.int64],
    ratio_array: NDArray[np.float64],
    weight_array: NDArray[np.float64],
    usable: NDArray[np.bool_],
    *,
    penalty: float | None,
    sensitivity: float,
    kappa_local: float,
    min_size: int,
    jump: int,
    max_blacklist_gap: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Run euchromatic-block PELT and return ``(s_bins, e_bins)`` half-open bin index arrays."""
    if ratio_array.size == 0 or not np.any(usable):
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    blocks = _split_blacklist_gaps(
        0,
        int(ratio_array.size),
        starts=start_array,
        ends=end_array,
        usable=usable,
        max_gap=max_blacklist_gap,
    )

    s_bins: list[int] = []
    e_bins: list[int] = []
    for block_left, block_right in blocks:
        if block_right <= block_left:
            continue
        valid_in_block = np.flatnonzero(usable[block_left:block_right])
        if valid_in_block.size == 0:
            continue
        valid_global = block_left + valid_in_block
        c_signal = np.column_stack(
            (ratio_array[valid_global], weight_array[valid_global])
        )

        block_penalty = (
            penalty
            if penalty is not None
            else float(
                sensitivity
                * 2.0
                * max(kappa_local, 1.0)
                * math.log(max(int(valid_global.size), 3))
            )
        )

        compact_bps = _find_breakpoints(
            signal=c_signal,
            penalty=block_penalty,
            min_size=min_size,
            jump=jump,
        )

        cur_compact = 0
        for bp in compact_bps:
            if bp <= cur_compact:
                continue
            b_bounded = min(int(bp), int(valid_global.size))
            s_bins.append(int(valid_global[cur_compact]))
            e_bins.append(int(valid_global[b_bounded - 1]) + 1)
            cur_compact = b_bounded

    return np.asarray(s_bins, dtype=np.int64), np.asarray(e_bins, dtype=np.int64)


def _reduce_segment_means(
    ratios: NDArray[np.float64],
    weights: NDArray[np.float64],
    usable: NDArray[np.bool_],
    s_bins: NDArray[np.int64],
    e_bins: NDArray[np.int64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
]:
    """Compute clean weights and per-segment weight, usable-bin count, and weighted mean."""
    clean_w = np.where(usable, weights, 0.0)
    clean_r = np.where(usable, ratios, 0.0)
    cumsum_w = _cumulative(clean_w)
    cumsum_wr = _cumulative(clean_w * clean_r)
    cumsum_u = _cumulative(usable.astype(np.float64))
    seg_w = cumsum_w[e_bins] - cumsum_w[s_bins]
    seg_wr = cumsum_wr[e_bins] - cumsum_wr[s_bins]
    seg_u = np.rint(cumsum_u[e_bins] - cumsum_u[s_bins]).astype(np.int64)
    seg_means = np.where(seg_w > 0.0, seg_wr / np.maximum(seg_w, 1e-12), 0.0)
    return clean_w, seg_w, seg_u, seg_means


def segment_contig(
    *,
    contig: str,
    starts: ArrayLike,
    ends: ArrayLike,
    ratios: ArrayLike,
    weights: ArrayLike,
    contig_length: int,
    penalty: float | None = None,
    sensitivity: float = 1.0,
    kappa_local: float = 1.0,
    min_size: int = 2,
    jump: int = 1,
    max_blacklist_gap: int = DEFAULT_MAX_BLACKLIST_GAP,
) -> tuple[SegmentResult, ...]:
    """Segment one contig using weighted L2 PELT with usable-bin compaction."""
    start_array, end_array, ratio_array, weight_array = _validate_inputs(
        contig=contig,
        starts=starts,
        ends=ends,
        ratios=ratios,
        weights=weights,
        contig_length=contig_length,
        penalty=penalty,
        sensitivity=sensitivity,
        kappa_local=kappa_local,
        min_size=min_size,
        jump=jump,
        max_blacklist_gap=max_blacklist_gap,
    )
    usable = np.isfinite(ratio_array) & np.isfinite(weight_array) & (weight_array > 0.0)
    s_bins, e_bins = _segment_bin_spans(
        start_array,
        end_array,
        ratio_array,
        weight_array,
        usable,
        penalty=penalty,
        sensitivity=sensitivity,
        kappa_local=kappa_local,
        min_size=min_size,
        jump=jump,
        max_blacklist_gap=max_blacklist_gap,
    )
    if s_bins.size == 0:
        return ()

    _, seg_w, seg_u, seg_means = _reduce_segment_means(
        ratio_array, weight_array, usable, s_bins, e_bins
    )

    return tuple(
        SegmentResult(
            contig=contig,
            start=int(start_array[s_i]),
            end=int(end_array[e_i - 1]),
            ratio=float(mean_i),
            zscore=0.0,
            weight=float(w_i),
            bin_count=int(u_i),
            call=None,
        )
        for s_i, e_i, mean_i, w_i, u_i in zip(
            s_bins, e_bins, seg_means, seg_w, seg_u, strict=True
        )
        if u_i > 0
    )


def _evaluate_integer_cn_bf(
    *,
    seg_ratio: float,
    sigma_s_sq: float,
    baseline_ploidy: int,
    has_counts: bool,
    c_s: float,
    lam_s: float,
    bg_s: float = 0.0,
    kappa_s: float = 1.0,
) -> tuple[int, float]:
    """Compute best integer CN candidate and log-Bayes-factor strictly from one likelihood."""
    alt_states = (0, 1, 3, 4) if baseline_ploidy == 2 else (0, 2, 3)
    best_alt_k = baseline_ploidy
    best_raw_diff = -math.inf

    if has_counts and lam_s > 0.0:
        k_eff = max(1.0, float(kappa_s))
        c_eff = c_s / k_eff
        lam_eff = lam_s / k_eff
        bg_eff = bg_s / k_eff
        nb_alpha = max(0.0, LN2_SQ * sigma_s_sq - k_eff / lam_s)
        bg_clean = min(max(0.0, bg_eff), 0.99 * lam_eff)
        fetal_lam = max(lam_eff - bg_clean, 0.0)
        for k in alt_states:
            if baseline_ploidy == 1 and bg_clean > 0.0:
                mu_k = max(
                    0.001,
                    max(bg_clean, 0.01 * lam_eff)
                    if k == 0
                    else bg_clean + float(k) * fetal_lam,
                )
            else:
                scale_k = (
                    0.01
                    if k == 0
                    else (0.5 * float(k) if baseline_ploidy == 2 else float(k))
                )
                mu_k = max(0.001, scale_k * lam_eff)
            if nb_alpha <= 1e-12:
                term_c = c_eff * math.log(mu_k / lam_eff) if c_eff > 0.0 else 0.0
                ll_diff = term_c - (mu_k - lam_eff)
            else:
                r_nb = 1.0 / nb_alpha
                term_c = (
                    c_eff
                    * math.log((mu_k * (lam_eff + r_nb)) / (lam_eff * (mu_k + r_nb)))
                    if c_eff > 0.0
                    else 0.0
                )
                ll_diff = term_c + r_nb * math.log((lam_eff + r_nb) / (mu_k + r_nb))
            if ll_diff > best_raw_diff:
                best_raw_diff = ll_diff
                best_alt_k = k
    else:
        fallback_table = (
            _DIPLOID_FALLBACK_LOG2 if baseline_ploidy == 2 else _HAPLOID_FALLBACK_LOG2
        )
        denom = max(2.0 * sigma_s_sq, 1e-12)
        for k in alt_states:
            mu_log2_k = fallback_table[k]
            ll_diff = (seg_ratio * seg_ratio - (seg_ratio - mu_log2_k) ** 2) / denom
            if ll_diff > best_raw_diff:
                best_raw_diff = ll_diff
                best_alt_k = k

    if best_raw_diff > 0.0:
        return best_alt_k, float(best_raw_diff)
    return baseline_ploidy, 0.0


def _as_optional_1d_f64(
    val: ArrayLike | None,
    expected_len: int,
    name: str,
) -> NDArray[np.float64] | None:
    if val is None:
        return None
    arr = np.ravel(np.asarray(val, dtype=np.float64))
    if arr.size != expected_len:
        raise InvalidInputError(
            f"{name} length {arr.size} must match bin count {expected_len}"
        )
    return arr


def _segment_and_prescore_contig(
    contig_name: str,
    item: ContigSegmentationInput,
    *,
    penalty: float | None,
    sensitivity: float,
    mode: AnalysisMode,
    min_size: int,
    jump: int,
    max_blacklist_gap: int,
) -> list[_PreScoredSegment]:
    """Validate, segment, and evaluate prefix-sum BLUE Z + CN Bayes factors on one contig."""
    c_starts, c_ends, c_ratios, c_weights = _validate_inputs(
        contig=contig_name,
        starts=item.starts,
        ends=item.ends,
        ratios=item.ratios,
        weights=item.weights,
        contig_length=int(item.contig_length),
        penalty=penalty,
        sensitivity=sensitivity,
        kappa_local=1.0,
        min_size=min_size,
        jump=jump,
        max_blacklist_gap=max_blacklist_gap,
    )
    n_bins = int(c_ratios.size)
    if n_bins == 0:
        return []

    c_z_raw = _as_optional_1d_f64(item.zscores, n_bins, "zscores")
    c_counts = _as_optional_1d_f64(item.counts, n_bins, "counts")
    c_exp = _as_optional_1d_f64(item.expected, n_bins, "expected")
    c_bg = _as_optional_1d_f64(item.background_expected, n_bins, "background_expected")

    u = np.isfinite(c_ratios) & np.isfinite(c_weights) & (c_weights > 0.0)
    if c_z_raw is None:
        c_z = np.where(u, c_ratios * np.sqrt(np.maximum(c_weights, 0.0)), 0.0)
    else:
        u = u & np.isfinite(c_z_raw)
        c_z = np.where(u, c_z_raw, 0.0)

    if not np.any(u):
        return []

    kappa_local = float(estimate_dyadic_inflation(c_z, u, scales=(1,))[0])
    s_bins, e_bins = _segment_bin_spans(
        c_starts,
        c_ends,
        c_ratios,
        c_weights,
        u,
        penalty=penalty,
        sensitivity=sensitivity,
        kappa_local=kappa_local,
        min_size=min_size,
        jump=jump,
        max_blacklist_gap=max_blacklist_gap,
    )
    if s_bins.size == 0:
        return []

    clean_w, seg_w, seg_u, seg_ratios = _reduce_segment_means(
        c_ratios, c_weights, u, s_bins, e_bins
    )
    if s_bins.size > 1:
        c_z_resid = c_z.copy()
        cumsum_z_raw = _cumulative(np.where(u, c_z, 0.0))
        seg_z_sum = cumsum_z_raw[e_bins] - cumsum_z_raw[s_bins]
        has_step = False
        for k_idx in range(s_bins.size):
            n_k = int(seg_u[k_idx])
            if n_k >= 2 and abs(float(seg_z_sum[k_idx])) >= 5.0 * math.sqrt(float(n_k)):
                sl = slice(int(s_bins[k_idx]), int(e_bins[k_idx]))
                u_sl = u[sl]
                c_z_resid[sl] = np.where(
                    u_sl, c_z_resid[sl] - float(seg_z_sum[k_idx]) / float(n_k), 0.0
                )
                has_step = True
        infl = estimate_dyadic_inflation(c_z_resid if has_step else c_z, u)
    else:
        infl = estimate_dyadic_inflation(c_z, u)

    cumsum_wz = _cumulative(np.sqrt(np.maximum(clean_w, 0.0)) * c_z)
    seg_wz = cumsum_wz[e_bins] - cumsum_wz[s_bins]

    has_counts = c_counts is not None and c_exp is not None
    if c_counts is not None and c_exp is not None:
        u_cnt = (
            u
            & np.isfinite(c_counts)
            & (c_counts >= 0.0)
            & np.isfinite(c_exp)
            & (c_exp >= 0.0)
        )
        cumsum_c = _cumulative(np.where(u_cnt, c_counts, 0.0))
        cumsum_e = _cumulative(np.where(u_cnt, c_exp, 0.0))
        seg_c = cumsum_c[e_bins] - cumsum_c[s_bins]
        seg_e = cumsum_e[e_bins] - cumsum_e[s_bins]
        seg_bg = (
            (
                _cumulative(
                    np.where(u_cnt & np.isfinite(c_bg), np.maximum(c_bg, 0.0), 0.0)
                )[e_bins]
                - _cumulative(
                    np.where(u_cnt & np.isfinite(c_bg), np.maximum(c_bg, 0.0), 0.0)
                )[s_bins]
            )
            if c_bg is not None
            else np.zeros(s_bins.size, dtype=np.float64)
        )
        if item.poisson_pooled:
            w0_bins = np.where(
                u_cnt,
                LN2_SQ / np.maximum(exact_poisson_log_variance(c_exp), 1e-12),
                0.0,
            )
            cumsum_w0 = _cumulative(w0_bins)
            seg_w0 = cumsum_w0[e_bins] - cumsum_w0[s_bins]
        else:
            seg_w0 = seg_w
    else:
        seg_c = np.zeros(s_bins.size, dtype=np.float64)
        seg_e = np.zeros(s_bins.size, dtype=np.float64)
        seg_bg = np.zeros(s_bins.size, dtype=np.float64)
        seg_w0 = seg_w

    ploidy = int(item.baseline_ploidy)
    base_var = float(item.baseline_log2_var)
    use_cn_bf = mode == "germline" or ploidy == 1
    out: list[_PreScoredSegment] = []
    for idx in range(s_bins.size):
        n_u = int(seg_u[idx])
        if n_u <= 0:
            continue
        s_i = int(s_bins[idx])
        e_i = int(e_bins[idx])
        w_tot = float(seg_w[idx])
        ratio_val = float(seg_ratios[idx])
        kappa_s = interpolate_dyadic_inflation(float(n_u), infl)
        c_s_val = float(seg_c[idx])
        lam_s_val = float(seg_e[idx])

        if item.poisson_pooled and has_counts and lam_s_val > 0.0:
            lam_1d = np.array([lam_s_val], dtype=np.float64)
            ratio_val = math.log2((c_s_val + 0.5) / (lam_s_val + 0.5)) - float(
                exact_poisson_log2_jensen_bias(lam_1d)[0]
            )
            v_inf = max(1.0, float(seg_w0[idx]) / max(w_tot, 1e-12))
            shot_var = (
                kappa_s * v_inf * float(exact_poisson_log_variance(lam_1d)[0]) / LN2_SQ
            )
            j_s = lam_s_val / (lam_s_val + 0.5)
            sigma_s_sq = max(shot_var + j_s * j_s * base_var, 1e-12)
            fetal_s = max(lam_s_val - float(seg_bg[idx]), 0.0)
            var_c = (
                kappa_s * v_inf * max(lam_s_val, 0.25)
                + (fetal_s * fetal_s) * LN2_SQ * base_var
            )
            z_blue = (c_s_val - lam_s_val) / math.sqrt(max(var_c, 1e-12))
            w_tot = 1.0 / sigma_s_sq
            kappa_bf = kappa_s * v_inf
        else:
            sigma_s_sq = (kappa_s / max(w_tot, 1e-12)) + base_var
            z_blue = (
                (float(seg_wz[idx]) / max(w_tot, 1e-12)) / math.sqrt(sigma_s_sq)
                if w_tot > 1e-12
                else 0.0
            )
            kappa_bf = kappa_s

        cand_cn: int | None = None
        log_bf: float | None = None
        if use_cn_bf:
            cand_cn, log_bf = _evaluate_integer_cn_bf(
                seg_ratio=ratio_val,
                sigma_s_sq=max(sigma_s_sq, 1e-4),
                baseline_ploidy=ploidy,
                has_counts=has_counts,
                c_s=c_s_val,
                lam_s=lam_s_val,
                bg_s=float(seg_bg[idx]),
                kappa_s=kappa_bf,
            )

        out.append(
            _PreScoredSegment(
                contig=contig_name,
                start=int(c_starts[s_i]),
                end=int(c_ends[e_i - 1]),
                ratio=ratio_val,
                weight=w_tot,
                n_usable=n_u,
                zscore=z_blue,
                baseline_ploidy=ploidy,
                cand_cn=cand_cn,
                log_bf=log_bf,
            )
        )
    return out


def segment_and_score_genome(
    contigs: Mapping[str, ContigSegmentationInput],
    *,
    penalty: float | None = None,
    sensitivity: float = 1.0,
    mode: AnalysisMode = "germline",
    z_threshold: float = 3.0,
    ratio_threshold: float | None = None,
    min_size: int = 2,
    jump: int = 1,
    max_blacklist_gap: int = DEFAULT_MAX_BLACKLIST_GAP,
    threads: int = 1,
) -> tuple[SegmentResult, ...]:
    """Segment genome and score with dyadic Gauss-Markov BLUE Z and integer CN Bayes factors."""
    if not math.isfinite(z_threshold) or z_threshold < 0.0:
        raise InvalidInputError("z_threshold must be finite and non-negative")
    if ratio_threshold is not None and (
        not math.isfinite(ratio_threshold) or ratio_threshold < 0.0
    ):
        raise InvalidInputError("ratio_threshold must be finite and non-negative")
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise InvalidInputError("threads must be a positive integer")
    if penalty is None and (not math.isfinite(sensitivity) or sensitivity <= 0.0):
        raise InvalidInputError("sensitivity must be finite and positive")
    started = perf_counter()
    logger.debug(
        "segmenting and scoring %d contigs with PELT (mode=%s, z_thresh=%.1f, threads=%d)",
        len(contigs),
        mode,
        z_threshold,
        threads,
    )
    if not contigs:
        return ()

    def _run_one(name: str) -> list[_PreScoredSegment]:
        return _segment_and_prescore_contig(
            name,
            contigs[name],
            penalty=penalty,
            sensitivity=sensitivity,
            mode=mode,
            min_size=min_size,
            jump=jump,
            max_blacklist_gap=max_blacklist_gap,
        )

    pre_segments: list[_PreScoredSegment] = []
    if threads > 1 and len(contigs) > 1:
        with ThreadPoolExecutor(max_workers=min(threads, len(contigs))) as executor:
            for seg_batch in executor.map(_run_one, list(contigs.keys())):
                pre_segments.extend(seg_batch)
    else:
        for contig_name in contigs:
            pre_segments.extend(_run_one(contig_name))

    n_segs = len(pre_segments)
    if n_segs == 0:
        return ()

    z_arr = np.fromiter(
        (seg.zscore for seg in pre_segments), dtype=np.float64, count=n_segs
    )
    p_vals = erfc(np.abs(z_arr) / math.sqrt(2.0))
    order = np.argsort(p_vals)
    ranks = np.arange(1, n_segs + 1, dtype=np.float64)
    bh = p_vals[order] * float(n_segs) / ranks
    cummin_bh = np.clip(np.minimum.accumulate(bh[::-1])[::-1], 0.0, 1.0)
    q_vals = np.empty(n_segs, dtype=np.float64)
    q_vals[order] = cummin_bh

    results: list[SegmentResult] = []
    for seg, q_raw in zip(pre_segments, q_vals, strict=True):
        q_val = float(q_raw)
        passes_thresh = (
            abs(seg.ratio) >= ratio_threshold
            if ratio_threshold is not None
            else abs(seg.zscore) >= z_threshold
        )
        call: CallType | None
        final_cn: int | None
        final_bf: float | None
        if mode == "germline" or seg.baseline_ploidy == 1:
            neutral_cn = seg.baseline_ploidy
            bf = seg.log_bf
            k_cand = seg.cand_cn
            concordant_loss = (
                k_cand is not None
                and k_cand < neutral_cn
                and seg.zscore < 0.0
                and seg.ratio < 0.0
            )
            concordant_gain = (
                k_cand is not None
                and k_cand > neutral_cn
                and seg.zscore > 0.0
                and seg.ratio > 0.0
            )
            if (
                passes_thresh
                and q_val <= 0.05
                and bf is not None
                and bf > 2.0
                and k_cand is not None
                and (concordant_loss or concordant_gain)
            ):
                call = "loss" if concordant_loss else "gain"
                final_cn = k_cand
            else:
                call = None
                final_cn = neutral_cn
            final_bf = bf
        else:
            if passes_thresh and q_val <= 0.05 and seg.zscore < 0.0 and seg.ratio < 0.0:
                call = "loss"
            elif (
                passes_thresh and q_val <= 0.05 and seg.zscore > 0.0 and seg.ratio > 0.0
            ):
                call = "gain"
            else:
                call = None
            final_cn = None
            final_bf = None

        results.append(
            SegmentResult(
                contig=seg.contig,
                start=seg.start,
                end=seg.end,
                ratio=seg.ratio,
                zscore=seg.zscore,
                weight=seg.weight,
                bin_count=seg.n_usable,
                call=call,
                copy_number=final_cn,
                log_bf=final_bf,
                q_value=q_val,
            )
        )

    elapsed = perf_counter() - started
    logger.debug(
        "segment and score finished: %d segments across %d contigs in %.3f s",
        len(results),
        len(contigs),
        elapsed,
    )
    return tuple(results)


def _cumulative(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Prefix cumulative sum with leading zero as contiguous float64 array."""
    out = np.empty(values.size + 1, dtype=np.float64)
    out[0] = 0.0
    np.cumsum(values, out=out[1:])
    return out


def _find_breakpoints(
    *,
    signal: NDArray[np.float64],
    penalty: float,
    min_size: int,
    jump: int,
) -> list[int]:
    """Detect change-point indices in a euchromatic block via compiled Rust PELT."""
    return pelt_weighted_l2(
        np.ascontiguousarray(signal, dtype=np.float64),
        float(penalty),
        int(min_size),
        int(jump),
    )


def _validate_inputs(
    *,
    contig: str,
    starts: ArrayLike,
    ends: ArrayLike,
    ratios: ArrayLike,
    weights: ArrayLike,
    contig_length: int,
    penalty: float | None,
    sensitivity: float,
    kappa_local: float,
    min_size: int,
    jump: int,
    max_blacklist_gap: int,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Validate input interval coordinates, numerical arrays, and segmentation parameters."""
    if not contig:
        raise InvalidInputError("contig must not be empty")
    if contig_length <= 0:
        raise InvalidInputError("contig_length must be positive")
    if penalty is not None:
        if not np.isfinite(penalty) or penalty <= 0.0:
            raise InvalidInputError("penalty must be finite and positive")
    else:
        if not math.isfinite(sensitivity) or sensitivity <= 0.0:
            raise InvalidInputError("sensitivity must be finite and positive")
        if not math.isfinite(kappa_local):
            raise InvalidInputError("kappa_local must be finite")
    if min_size < 1 or jump < 1 or max_blacklist_gap < 0:
        raise InvalidInputError("min_size, jump, and max_blacklist_gap must be valid")

    start_array = np.ravel(np.asarray(starts, dtype=np.int64))
    end_array = np.ravel(np.asarray(ends, dtype=np.int64))
    ratio_array = np.ravel(np.asarray(ratios, dtype=np.float64))
    weight_array = np.ravel(np.asarray(weights, dtype=np.float64))
    if not (
        start_array.size == end_array.size == ratio_array.size == weight_array.size
    ):
        raise InvalidInputError(
            "starts, ends, ratios, and weights must have equal lengths"
        )
    if start_array.size == 0:
        return start_array, end_array, ratio_array, weight_array
    if np.any(start_array < 0) or np.any(end_array <= start_array):
        raise InvalidInputError("bins must be non-empty 0-based half-open intervals")
    if np.any(end_array > contig_length):
        raise InvalidInputError("bin end exceeds contig_length")
    if np.any(start_array[1:] < end_array[:-1]):
        raise InvalidInputError("bins must be sorted and non-overlapping")
    if np.any(np.isfinite(weight_array) & (weight_array < 0.0)):
        raise InvalidInputError("weights must be non-negative")
    return start_array, end_array, ratio_array, weight_array


def _split_blacklist_gaps(
    left: int,
    right: int,
    *,
    starts: NDArray[np.int64],
    ends: NDArray[np.int64],
    usable: NDArray[np.bool_],
    max_gap: int,
) -> list[tuple[int, int]]:
    """Partition interval range into contiguous euchromatic blocks split at large masked gaps."""
    splits: list[tuple[int, int]] = []
    cursor = left
    index = left
    while index < right:
        if usable[index]:
            index += 1
            continue
        gap_start = index
        while index < right and not usable[index]:
            index += 1
        gap_end = index
        gap_bases = int(ends[gap_end - 1] - starts[gap_start])
        if gap_bases > max_gap:
            splits.append((cursor, gap_start))
            cursor = gap_end
    splits.append((cursor, right))
    return splits
