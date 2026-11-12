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

import numpy as np
import ruptures as rpt
from numpy.typing import ArrayLike, NDArray
from ruptures.base import BaseCost
from scipy.special import erfc

from ccnv.models import AnalysisMode, CallType

logger = logging.getLogger(__name__)

DEFAULT_MAX_BLACKLIST_GAP = 2_000_000

__all__ = [
    "DEFAULT_MAX_BLACKLIST_GAP",
    "ContigSegmentationInput",
    "PreparedContigData",
    "Segment",
    "WeightedL2Cost",
    "estimate_dyadic_inflation",
    "evaluate_blue_scores_and_bayes_factors",
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


@dataclass(frozen=True, slots=True)
class PreparedContigData:
    """Validated per-contig arrays and usability mask for segment scoring."""

    starts: NDArray[np.int64]
    ends: NDArray[np.int64]
    ratios: NDArray[np.float64]
    weights: NDArray[np.float64]
    zscores: NDArray[np.float64]
    counts: NDArray[np.float64] | None
    expected: NDArray[np.float64] | None
    usable: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class Segment:
    """A weighted segment in canonical 0-based, half-open coordinates."""

    contig: str
    start: int
    end: int
    mean: float
    weight: float
    bin_count: int
    zscore: float = 0.0
    call: CallType | None = None
    copy_number: int | None = None
    log_bf: float | None = None
    q_value: float = 1.0
    n_usable: int = 0


class WeightedL2Cost(BaseCost):
    """Ruptures-compatible weighted residual sum-of-squares cost."""

    model = "weighted_l2"
    min_size = 1

    def __init__(self) -> None:
        """Initialize empty weighted L2 cost accumulator."""
        self.signal = np.empty((0, 2), dtype=np.float64)
        self._n = 0
        self._weight_sum: list[float] = [0.0]
        self._weighted_value_sum: list[float] = [0.0]
        self._weighted_square_sum: list[float] = [0.0]

    def fit(
        self, signal: ArrayLike, weights: ArrayLike | None = None
    ) -> WeightedL2Cost:
        """Fit cumulative sums from values and optional non-negative weights."""
        array = np.asarray(signal, dtype=np.float64)
        if weights is None:
            if array.ndim == 2 and array.shape[1] == 2:
                values, weight_array = array[:, 0], array[:, 1]
            else:
                values = np.ravel(array)
                weight_array = np.ones(values.size, dtype=np.float64)
        else:
            values = np.ravel(array)
            weight_array = np.ravel(np.asarray(weights, dtype=np.float64))

        if values.size != weight_array.size:
            raise ValueError("values and weights must have equal lengths")
        if np.any(np.isfinite(weight_array) & (weight_array < 0)):
            raise ValueError("weights must be non-negative")

        usable = np.isfinite(values) & np.isfinite(weight_array) & (weight_array > 0)
        clean_weights = np.where(usable, weight_array, 0.0)
        clean_values = np.where(usable, values, 0.0)

        total_w = float(np.sum(clean_weights))
        mu_0 = (
            float(np.sum(clean_weights * clean_values) / total_w)
            if total_w > 0
            else 0.0
        )
        shifted = np.where(usable, clean_values - mu_0, 0.0)

        self.signal = np.column_stack((clean_values, clean_weights))
        self._n = clean_values.size
        self._weight_sum = _cumulative(clean_weights).tolist()
        self._weighted_value_sum = _cumulative(clean_weights * shifted).tolist()
        self._weighted_square_sum = _cumulative(
            clean_weights * np.square(shifted)
        ).tolist()
        return self

    def error(self, start: int, end: int) -> float:
        """Return weighted residual sum of squares for ``[start, end)``."""
        if start < 0 or end > self._n or start >= end:
            raise ValueError("cost interval must be a non-empty in-bounds slice")
        weight = self._weight_sum[end] - self._weight_sum[start]
        if weight <= 0.0:
            return 0.0
        weighted_val = self._weighted_value_sum[end] - self._weighted_value_sum[start]
        weighted_sq = self._weighted_square_sum[end] - self._weighted_square_sum[start]
        rss = weighted_sq - (weighted_val * weighted_val) / weight
        return max(0.0, rss)


def estimate_dyadic_inflation(
    zscores: NDArray[np.float64],
    usable: NDArray[np.bool_],
    *,
    scales: Sequence[int] = (1, 4, 16, 64, 256, 1024),
) -> NDArray[np.float64]:
    """Estimate scale-dependent variance inflation kappa(L) over unmasked runs."""
    z = np.ravel(np.asarray(zscores, dtype=np.float64))
    u = np.ravel(np.asarray(usable, dtype=bool))
    if z.size != u.size or z.size == 0:
        return np.ones(len(scales), dtype=np.float64)

    valid_mask = u & np.isfinite(z)
    z_eval = np.where(valid_mask, np.clip(z, -2.5, 2.5), 0.0)
    padded = np.concatenate(([False], valid_mask, [False]))
    diff = np.diff(padded.astype(np.int8))
    run_starts = np.flatnonzero(diff == 1)
    run_ends = np.flatnonzero(diff == -1)

    inflations: list[float] = []
    prev_kappa = 1.0

    cumsum_z = _cumulative(z_eval)
    for s_idx, scale in enumerate(scales):
        block_vals: list[float] = []
        inv_sqrt_scale = 1.0 / np.sqrt(scale)
        for r_start, r_end in zip(run_starts, run_ends, strict=False):
            n_blocks = (r_end - r_start) // scale
            for k in range(n_blocks):
                b = r_start + k * scale
                block_vals.append(
                    float(cumsum_z[b + scale] - cumsum_z[b]) * inv_sqrt_scale
                )

        if len(block_vals) >= 5:
            kappa_l = max(1.0, float(np.var(block_vals, ddof=1)))
        else:
            kappa_l = prev_kappa

        if s_idx > 0:
            kappa_l = max(kappa_l, inflations[-1])

        inflations.append(kappa_l)
        prev_kappa = kappa_l

    return np.asarray(inflations, dtype=np.float64)


def interpolate_dyadic_inflation(
    n_bins: float,
    inflation_values: Sequence[float] | NDArray[np.float64],
    *,
    scales: Sequence[int] = (1, 4, 16, 64, 256, 1024),
) -> float:
    """Interpolate kappa(N) monotonically in log2 scale space."""
    infl_arr = np.asarray(inflation_values, dtype=np.float64)
    if infl_arr.size == 0:
        return 1.0
    n = max(1.0, float(n_bins))
    if n <= scales[0]:
        return float(infl_arr[0])
    if n >= scales[-1]:
        return float(infl_arr[-1])
    log_scales = np.log2(scales)
    log_n = np.log2(n)
    val = float(np.interp(log_n, log_scales, infl_arr))
    return max(1.0, val)


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
) -> tuple[Segment, ...]:
    """Segment one contig using weighted L2 PELT with usable-bin compaction."""
    start_array, end_array, ratio_array, weight_array = _validate_inputs(
        contig=contig,
        starts=starts,
        ends=ends,
        ratios=ratios,
        weights=weights,
        contig_length=contig_length,
        penalty=penalty,
        min_size=min_size,
        jump=jump,
        max_blacklist_gap=max_blacklist_gap,
    )
    if ratio_array.size == 0:
        return ()

    usable = np.isfinite(ratio_array) & np.isfinite(weight_array) & (weight_array > 0)
    if not np.any(usable):
        return ()

    blocks = _split_blacklist_gaps(
        0,
        ratio_array.size,
        starts=start_array,
        ends=end_array,
        usable=usable,
        max_gap=max_blacklist_gap,
    )

    segments: list[Segment] = []
    for block_left, block_right in blocks:
        if block_right <= block_left:
            continue
        valid_in_block = np.flatnonzero(usable[block_left:block_right])
        if valid_in_block.size == 0:
            continue
        valid_global = block_left + valid_in_block
        c_values = ratio_array[valid_global]
        c_weights = weight_array[valid_global]
        c_signal = np.column_stack((c_values, c_weights))

        block_penalty = (
            penalty
            if penalty is not None
            else float(
                sensitivity
                * 2.0
                * max(kappa_local, 1.0)
                * np.log(max(valid_global.size, 3))
            )
        )

        compact_bps = _find_breakpoints(
            signal=c_signal,
            block_usable=np.ones(valid_global.size, dtype=bool),
            penalty=block_penalty,
            min_size=min_size,
            jump=jump,
        )

        cur_compact = 0
        for b in compact_bps:
            if b <= cur_compact:
                continue
            b_bounded = min(b, valid_global.size)
            first_idx = int(valid_global[cur_compact])
            last_idx = int(valid_global[b_bounded - 1])
            usable_slice = valid_global[cur_compact:b_bounded]

            seg = _make_segment_from_indices(
                contig,
                first_idx,
                last_idx,
                usable_slice,
                start_array,
                end_array,
                ratio_array,
                weight_array,
            )
            if seg is not None:
                segments.append(seg)
            cur_compact = b_bounded

    return tuple(segments)


def evaluate_blue_scores_and_bayes_factors(
    segments: Sequence[Segment],
    contigs_data: Mapping[str, PreparedContigData],
    dyadic_inflations: Mapping[str, NDArray[np.float64]],
    *,
    mode: AnalysisMode = "germline",
    z_threshold: float = 3.0,
    ratio_threshold: float | None = None,
) -> tuple[Segment, ...]:
    """Score segments with Gauss-Markov BLUE Z-scores, BH-FDR, and CN Bayes factors."""
    if not segments:
        return ()

    prefix_map: dict[str, dict[str, NDArray[np.float64]]] = {}
    for c_name, data in contigs_data.items():
        clean_w = np.where(data.usable, data.weights, 0.0)
        clean_z = np.where(data.usable, data.zscores, 0.0)
        sqrt_w = np.sqrt(np.maximum(clean_w, 0.0))
        w_z = sqrt_w * clean_z

        acc: dict[str, NDArray[np.float64]] = {
            "cumsum_w_z": _cumulative(w_z),
            "cumsum_usable": _cumulative(data.usable.astype(np.float64)),
            "cumsum_w": _cumulative(clean_w),
        }
        if data.counts is not None:
            acc["cumsum_c"] = _cumulative(np.where(data.usable, data.counts, 0.0))
        if data.expected is not None:
            acc["cumsum_e"] = _cumulative(np.where(data.usable, data.expected, 0.0))

        prefix_map[c_name] = acc

    n_segs = len(segments)
    blue_zscores = np.zeros(n_segs, dtype=np.float64)
    usable_counts = np.zeros(n_segs, dtype=np.int64)
    log_bfs: list[float | None] = [None] * n_segs
    copy_numbers: list[int | None] = [None] * n_segs

    fallback_log_ratios = {0: -4.5, 1: -1.0, 2: 0.0, 3: math.log2(1.5), 4: 1.0}
    ln2_sq = math.log(2.0) ** 2

    for i, seg in enumerate(segments):
        c_name = seg.contig
        if c_name not in contigs_data or c_name not in prefix_map:
            continue
        c_data = contigs_data[c_name]
        acc = prefix_map[c_name]

        s_bin = int(np.searchsorted(c_data.starts, seg.start, side="left"))
        e_bin = int(np.searchsorted(c_data.ends, seg.end, side="right"))

        w_z = float(acc["cumsum_w_z"][e_bin] - acc["cumsum_w_z"][s_bin])
        n_u = int(acc["cumsum_usable"][e_bin] - acc["cumsum_usable"][s_bin])
        w_tot = float(acc["cumsum_w"][e_bin] - acc["cumsum_w"][s_bin])

        usable_counts[i] = n_u
        if n_u == 0:
            blue_zscores[i] = 0.0
            continue

        c_infl = dyadic_inflations.get(c_name, np.ones(1, dtype=np.float64))
        kappa_s = interpolate_dyadic_inflation(n_u, c_infl)
        denom = math.sqrt(kappa_s * max(w_tot, 1e-12))
        z_blue = w_z / denom if denom > 1e-12 else 0.0
        blue_zscores[i] = z_blue

        if mode == "germline":
            sigma_s_sq = kappa_s / max(w_tot, 1e-4)
            has_counts = "cumsum_c" in acc and "cumsum_e" in acc
            c_s = (
                float(acc["cumsum_c"][e_bin] - acc["cumsum_c"][s_bin])
                if has_counts
                else 0.0
            )
            lam_s = (
                float(acc["cumsum_e"][e_bin] - acc["cumsum_e"][s_bin])
                if has_counts
                else 0.0
            )
            lam_bar = lam_s / n_u if (has_counts and lam_s > 0.0) else 0.0
            nb_alpha = (
                max(0.0, ln2_sq * sigma_s_sq - 1.0 / lam_s)
                if (has_counts and lam_s > 0.0)
                else 0.0
            )

            best_k = 2
            best_bf = -math.inf
            for k in (0, 1, 3, 4):
                mu_log2_k = (
                    math.log2((0.5 * k * lam_bar + 0.5) / (lam_bar + 0.5))
                    if lam_bar > 0.0
                    else fallback_log_ratios[k]
                )
                d_sq = seg.mean**2 - (seg.mean - mu_log2_k) ** 2
                delta = d_sq / (2.0 * sigma_s_sq)
                if has_counts and lam_s > 0.0:
                    mu_k = max(0.001, 0.5 * k * lam_s)
                    if nb_alpha <= 1e-12:
                        delta += c_s * math.log(mu_k / lam_s) - (mu_k - lam_s)
                    else:
                        r_nb = 1.0 / nb_alpha
                        delta += c_s * math.log(
                            (mu_k * (lam_s + r_nb)) / (lam_s * (mu_k + r_nb))
                        ) + r_nb * math.log((lam_s + r_nb) / (mu_k + r_nb))
                if delta > best_bf:
                    best_bf = delta
                    best_k = k

            log_bfs[i] = float(best_bf)
            copy_numbers[i] = best_k

    p_vals = np.ones(n_segs, dtype=np.float64)
    usable_seg_idx = np.flatnonzero(usable_counts > 0)
    if usable_seg_idx.size > 0:
        p_vals[usable_seg_idx] = erfc(
            np.abs(blue_zscores[usable_seg_idx]) / np.sqrt(2.0)
        )

    q_vals = np.ones(n_segs, dtype=np.float64)
    m_tests = usable_seg_idx.size
    if m_tests > 0:
        sub_p = p_vals[usable_seg_idx]
        order = np.argsort(sub_p)
        sorted_p = sub_p[order]
        ranks = np.arange(1, m_tests + 1, dtype=np.float64)
        bh = sorted_p * float(m_tests) / ranks
        cummin_bh = np.minimum.accumulate(bh[::-1])[::-1]
        cummin_bh = np.clip(cummin_bh, 0.0, 1.0)
        orig_q = np.empty(m_tests, dtype=np.float64)
        orig_q[order] = cummin_bh
        q_vals[usable_seg_idx] = orig_q

    scored_segments: list[Segment] = []
    for i, seg in enumerate(segments):
        n_u = int(usable_counts[i])
        z_blue = float(blue_zscores[i])
        q_val = float(q_vals[i])

        call: CallType | None
        if n_u == 0:
            call = None
            final_cn = 2 if mode == "germline" else None
            final_bf = None
            z_blue = 0.0
            q_val = 1.0
        else:
            passes_thresh = (
                abs(seg.mean) >= ratio_threshold
                if ratio_threshold is not None
                else abs(z_blue) >= z_threshold
            )
            if mode == "germline":
                bf, k_cand = log_bfs[i], copy_numbers[i]
                if passes_thresh and q_val <= 0.05 and bf is not None and bf > 2.0:
                    call = "loss" if (k_cand is not None and k_cand < 2) else "gain"
                    final_cn = k_cand
                else:
                    call, final_cn = None, 2
                final_bf = bf
            else:
                call = (
                    ("loss" if z_blue < 0.0 else "gain")
                    if (passes_thresh and q_val <= 0.05)
                    else None
                )
                final_cn, final_bf = None, None

        scored_segments.append(
            Segment(
                contig=seg.contig,
                start=seg.start,
                end=seg.end,
                mean=seg.mean,
                weight=seg.weight,
                bin_count=seg.bin_count,
                zscore=z_blue,
                call=call,
                copy_number=final_cn,
                log_bf=final_bf,
                q_value=q_val,
                n_usable=n_u,
            )
        )

    return tuple(scored_segments)


def _as_optional_f64(val: ArrayLike | None) -> NDArray[np.float64] | None:
    return np.ravel(np.asarray(val, dtype=np.float64)) if val is not None else None


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
) -> tuple[Segment, ...]:
    """Segment genome and score with dyadic Gauss-Markov BLUE Z and integer CN Bayes factors."""
    started = perf_counter()
    logger.debug(
        "segmenting and scoring %d contigs with PELT (mode=%s, z_thresh=%.1f, threads=%d)",
        len(contigs),
        mode,
        z_threshold,
        threads,
    )

    contigs_data: dict[str, PreparedContigData] = {}
    contig_lengths: dict[str, int] = {}
    dyadic_inflations: dict[str, NDArray[np.float64]] = {}

    for c_name, item in contigs.items():
        c_starts = np.ravel(np.asarray(item.starts, dtype=np.int64))
        c_ends = np.ravel(np.asarray(item.ends, dtype=np.int64))
        c_ratios = np.ravel(np.asarray(item.ratios, dtype=np.float64))
        c_weights = np.ravel(np.asarray(item.weights, dtype=np.float64))
        c_len = int(item.contig_length)
        c_z_raw = _as_optional_f64(item.zscores)
        c_counts = _as_optional_f64(item.counts)
        c_exp = _as_optional_f64(item.expected)

        u = np.isfinite(c_ratios) & np.isfinite(c_weights) & (c_weights > 0)
        if c_z_raw is None:
            c_z = np.where(u, c_ratios * np.sqrt(c_weights), 0.0)
        else:
            u = u & np.isfinite(c_z_raw)
            c_z = np.where(u, c_z_raw, 0.0)

        infl = estimate_dyadic_inflation(c_z, u)
        dyadic_inflations[c_name] = infl
        contig_lengths[c_name] = c_len
        contigs_data[c_name] = PreparedContigData(
            starts=c_starts,
            ends=c_ends,
            ratios=c_ratios,
            weights=c_weights,
            zscores=c_z,
            counts=c_counts,
            expected=c_exp,
            usable=u,
        )

    def _segment_one(contig_name: str) -> tuple[Segment, ...]:
        c_data = contigs_data[contig_name]
        c_len = contig_lengths[contig_name]
        k_local = float(dyadic_inflations[contig_name][0])
        return segment_contig(
            contig=contig_name,
            starts=c_data.starts,
            ends=c_data.ends,
            ratios=c_data.ratios,
            weights=c_data.weights,
            contig_length=c_len,
            penalty=penalty,
            sensitivity=sensitivity,
            kappa_local=k_local,
            min_size=min_size,
            jump=jump,
            max_blacklist_gap=max_blacklist_gap,
        )

    all_raw_segments: list[Segment] = []
    if threads > 1 and len(contigs) > 1:
        with ThreadPoolExecutor(max_workers=threads) as executor:
            for seg_tuple in executor.map(_segment_one, list(contigs.keys())):
                all_raw_segments.extend(seg_tuple)
    else:
        for contig_name in contigs:
            all_raw_segments.extend(_segment_one(contig_name))

    final_segments = evaluate_blue_scores_and_bayes_factors(
        all_raw_segments,
        contigs_data,
        dyadic_inflations,
        mode=mode,
        z_threshold=z_threshold,
        ratio_threshold=ratio_threshold,
    )

    elapsed = perf_counter() - started
    logger.debug(
        "segment and score finished: %d segments across %d contigs in %.3f s",
        len(final_segments),
        len(contigs),
        elapsed,
    )
    return final_segments


def _cumulative(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Prefix cumulative sum with leading zero."""
    return np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(values)))


def _find_breakpoints(
    *,
    signal: NDArray[np.float64],
    block_usable: NDArray[np.bool_],
    penalty: float,
    min_size: int,
    jump: int,
) -> list[int]:
    """Detect change-point indices in a euchromatic block via calibrated PELT."""
    n_samples = signal.shape[0]
    if n_samples < 2 * min_size:
        return [n_samples]

    values = signal[:, 0]
    weights = signal[:, 1]

    if np.all(block_usable) and np.allclose(weights, weights[0]):
        pos_w = float(weights[0])
        kernel_cpd_cls = getattr(rpt, "KernelCPD", None)
        if kernel_cpd_cls is not None:
            detector = kernel_cpd_cls(kernel="linear", min_size=min_size, jump=jump)
            return list(
                detector.fit(values.reshape(-1, 1)).predict(pen=penalty / pos_w)
            )
    detector = rpt.Pelt(custom_cost=WeightedL2Cost(), min_size=min_size, jump=jump)
    return list(detector.fit(signal).predict(pen=penalty))


def _validate_inputs(
    *,
    contig: str,
    starts: ArrayLike,
    ends: ArrayLike,
    ratios: ArrayLike,
    weights: ArrayLike,
    contig_length: int,
    penalty: float | None,
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
        raise ValueError("contig must not be empty")
    if contig_length <= 0:
        raise ValueError("contig_length must be positive")
    if penalty is not None and (not np.isfinite(penalty) or penalty <= 0):
        raise ValueError("penalty must be finite and positive")
    if min_size < 1 or jump < 1 or max_blacklist_gap < 0:
        raise ValueError("min_size, jump, and max_blacklist_gap must be valid")

    start_array = np.ravel(np.asarray(starts, dtype=np.int64))
    end_array = np.ravel(np.asarray(ends, dtype=np.int64))
    ratio_array = np.ravel(np.asarray(ratios, dtype=np.float64))
    weight_array = np.ravel(np.asarray(weights, dtype=np.float64))
    if not (
        start_array.size == end_array.size == ratio_array.size == weight_array.size
    ):
        raise ValueError("starts, ends, ratios, and weights must have equal lengths")
    if start_array.size == 0:
        return start_array, end_array, ratio_array, weight_array
    if np.any(start_array < 0) or np.any(end_array <= start_array):
        raise ValueError("bins must be non-empty 0-based half-open intervals")
    if np.any(end_array > contig_length):
        raise ValueError("bin end exceeds contig_length")
    if np.any(start_array[1:] < end_array[:-1]):
        raise ValueError("bins must be sorted and non-overlapping")
    if np.any(np.isfinite(weight_array) & (weight_array < 0)):
        raise ValueError("weights must be non-negative")
    return start_array, end_array, ratio_array, weight_array


def _split_blacklist_gaps(
    left: int,
    right: int,
    *,
    starts: NDArray[np.int64],
    ends: NDArray[np.int64],
    usable: NDArray[np.bool_],
    max_gap: int,
) -> Sequence[tuple[int, int]]:
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


def _make_segment_from_indices(
    contig: str,
    first_idx: int,
    last_idx: int,
    valid_indexes: NDArray[np.int64],
    starts: NDArray[np.int64],
    ends: NDArray[np.int64],
    ratios: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> Segment | None:
    """Create a Segment from known usable bin indices."""
    if valid_indexes.size == 0:
        return None
    valid_weights = weights[valid_indexes]
    total_weight = float(np.sum(valid_weights))
    mean = (
        float(np.average(ratios[valid_indexes], weights=valid_weights))
        if total_weight > 0
        else float(np.mean(ratios[valid_indexes]))
    )
    n_usable = int(valid_indexes.size)
    return Segment(
        contig=contig,
        start=int(starts[first_idx]),
        end=int(ends[last_idx]),
        mean=mean,
        weight=total_weight,
        bin_count=n_usable,
        n_usable=n_usable,
    )
