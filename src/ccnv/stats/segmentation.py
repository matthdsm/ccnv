"""Python-native weighted copy-number segmentation.

The ``penalty`` is measured in weighted squared-error units. PELT minimizes
``sum(weighted squared error) + penalty * number of breakpoints``; Binseg
greedily accepts a breakpoint only when its weighted-error reduction exceeds
the penalty.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
import polars as pl
import ruptures as rpt
from numpy.typing import ArrayLike, NDArray
from ruptures.base import BaseCost

logger = logging.getLogger(__name__)

Algorithm = Literal["pelt", "binseg"]
DEFAULT_MAX_BLACKLIST_GAP = 2_000_000

__all__ = [
    "DEFAULT_MAX_BLACKLIST_GAP",
    "Algorithm",
    "Segment",
    "WeightedL2Cost",
    "compute_contig_rho1",
    "evaluate_stouffer_segment_scores",
    "segment_contig",
    "segment_genome",
]


@dataclass(frozen=True, slots=True)
class Segment:
    """A weighted segment in canonical 0-based, half-open coordinates."""

    contig: str
    start: int
    end: int
    mean: float
    weight: float
    bin_count: int


class WeightedL2Cost(BaseCost):
    """Ruptures-compatible weighted residual sum-of-squares cost."""

    model = "weighted_l2"
    min_size = 1

    def __init__(self) -> None:
        """Initialize empty weighted L2 cost accumulator."""
        self.signal = np.empty((0, 2), dtype=np.float64)
        self._weight_sum = np.zeros(1, dtype=np.float64)
        self._weighted_value_sum = np.zeros(1, dtype=np.float64)
        self._weighted_square_sum = np.zeros(1, dtype=np.float64)

    def fit(
        self, signal: ArrayLike, weights: ArrayLike | None = None
    ) -> WeightedL2Cost:
        """Fit cumulative sums from values and optional non-negative weights."""

        array = np.asarray(signal, dtype=np.float64)
        if weights is None:
            if array.ndim == 2 and array.shape[1] == 2:
                values = array[:, 0]
                weight_array = array[:, 1]
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
        self.signal = np.column_stack((clean_values, clean_weights))
        self._weight_sum = _cumulative(clean_weights)
        self._weighted_value_sum = _cumulative(clean_weights * clean_values)
        self._weighted_square_sum = _cumulative(clean_weights * np.square(clean_values))
        return self

    def error(self, start: int, end: int) -> float:
        """Return weighted residual sum of squares for ``[start, end)``."""

        if start < 0 or end > self.signal.shape[0] or start >= end:
            raise ValueError("cost interval must be a non-empty in-bounds slice")
        weight = self._weight_sum[end] - self._weight_sum[start]
        if weight == 0:
            return 0.0
        weighted_value = self._weighted_value_sum[end] - self._weighted_value_sum[start]
        weighted_square = (
            self._weighted_square_sum[end] - self._weighted_square_sum[start]
        )
        return max(0.0, float(weighted_square - weighted_value**2 / weight))


def segment_contig(
    *,
    contig: str,
    starts: ArrayLike,
    ends: ArrayLike,
    ratios: ArrayLike,
    weights: ArrayLike,
    contig_length: int,
    penalty: float,
    algorithm: Algorithm = "binseg",
    min_size: int = 2,
    jump: int = 1,
    max_blacklist_gap: int = DEFAULT_MAX_BLACKLIST_GAP,
) -> tuple[Segment, ...]:
    """Segment one contig using weighted L2 loss.

    Zero or non-finite weights and non-finite ratios are blacklisted. Long
    blacklisted runs split output segments; shorter runs may be bridged.
    """

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
        trim_left = block_left + int(valid_in_block[0])
        trim_right = block_left + int(valid_in_block[-1]) + 1

        block_usable = usable[trim_left:trim_right]
        clean_ratios = np.where(block_usable, ratio_array[trim_left:trim_right], 0.0)
        clean_weights = np.where(block_usable, weight_array[trim_left:trim_right], 0.0)
        signal = np.column_stack((clean_ratios, clean_weights))

        local_bps = _find_breakpoints(
            signal=signal,
            block_usable=block_usable,
            penalty=penalty,
            algorithm=algorithm,
            min_size=min_size,
            jump=jump,
        )

        cur_left = trim_left
        for bp in local_bps:
            bp_global = trim_left + bp
            segment = _make_segment(
                contig,
                cur_left,
                bp_global,
                start_array,
                end_array,
                ratio_array,
                weight_array,
                usable,
            )
            if segment is not None:
                segments.append(segment)
            cur_left = bp_global

    return tuple(segments)


def segment_genome(
    contigs: Mapping[str, tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, int]],
    *,
    penalty: float,
    algorithm: Algorithm = "binseg",
    min_size: int = 2,
    jump: int = 1,
    max_blacklist_gap: int = DEFAULT_MAX_BLACKLIST_GAP,
) -> tuple[Segment, ...]:
    """Segment ordered contig inputs without assumptions about contig names."""

    started = perf_counter()
    logger.debug(
        "segmenting %d contigs with %s (penalty=%.2f, max_blacklist_gap=%d bp)",
        len(contigs),
        algorithm,
        penalty,
        max_blacklist_gap,
    )
    all_segments: list[Segment] = []
    for contig, (starts, ends, ratios, weights, contig_length) in contigs.items():
        segments = segment_contig(
            contig=contig,
            starts=starts,
            ends=ends,
            ratios=ratios,
            weights=weights,
            contig_length=contig_length,
            penalty=penalty,
            algorithm=algorithm,
            min_size=min_size,
            jump=jump,
            max_blacklist_gap=max_blacklist_gap,
        )
        if segments:
            logger.debug(
                "contig %s: %d segments found across %d bp",
                contig,
                len(segments),
                contig_length,
            )
        all_segments.extend(segments)

    elapsed = perf_counter() - started
    logger.debug(
        "segmentation finished: %d segments across %d contigs in %.3f s",
        len(all_segments),
        len(contigs),
        elapsed,
    )
    return tuple(all_segments)


def _cumulative(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Prefix cumulative sum with leading zero."""
    return np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(values)))


def _find_breakpoints(
    *,
    signal: NDArray[np.float64],
    block_usable: NDArray[np.bool_],
    penalty: float,
    algorithm: Algorithm,
    min_size: int,
    jump: int,
) -> list[int]:
    """Detect change-point indices in a euchromatic block using ruptures."""
    n_samples = signal.shape[0]
    if n_samples < 2 * min_size:
        return [n_samples]

    values = signal[:, 0]
    weights = signal[:, 1]

    if algorithm == "pelt":
        if np.all(block_usable) and np.allclose(weights, weights[0]):
            pos_w = float(weights[0])
            if hasattr(rpt, "KernelCPD"):
                detector = rpt.KernelCPD(  # type: ignore[attr-defined]
                    kernel="linear", min_size=min_size, jump=jump
                )
                return detector.fit(values.reshape(-1, 1)).predict(pen=penalty / pos_w)
        detector = rpt.Pelt(custom_cost=WeightedL2Cost(), min_size=min_size, jump=jump)
        return detector.fit(signal).predict(pen=penalty)
    if algorithm == "binseg":
        detector = rpt.Binseg(
            custom_cost=WeightedL2Cost(), min_size=min_size, jump=jump
        )
        return detector.fit(signal).predict(pen=penalty)
    raise ValueError(f"unknown segmentation algorithm: {algorithm}")


def _validate_inputs(
    *,
    contig: str,
    starts: ArrayLike,
    ends: ArrayLike,
    ratios: ArrayLike,
    weights: ArrayLike,
    contig_length: int,
    penalty: float,
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
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("penalty must be finite and positive")
    if min_size < 1:
        raise ValueError("min_size must be positive")
    if jump < 1:
        raise ValueError("jump must be positive")
    if max_blacklist_gap < 0:
        raise ValueError("max_blacklist_gap must be non-negative")

    start_array = np.ravel(np.asarray(starts, dtype=np.int64))
    end_array = np.ravel(np.asarray(ends, dtype=np.int64))
    ratio_array = np.ravel(np.asarray(ratios, dtype=np.float64))
    weight_array = np.ravel(np.asarray(weights, dtype=np.float64))
    sizes = {
        start_array.size,
        end_array.size,
        ratio_array.size,
        weight_array.size,
    }
    if len(sizes) != 1:
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


def _make_segment(
    contig: str,
    left: int,
    right: int,
    starts: NDArray[np.int64],
    ends: NDArray[np.int64],
    ratios: NDArray[np.float64],
    weights: NDArray[np.float64],
    usable: NDArray[np.bool_],
) -> Segment | None:
    """Create a Segment object with weighted mean ratio and total weight."""
    valid_indexes = np.flatnonzero(usable[left:right]) + left
    if valid_indexes.size == 0:
        return None
    valid_weights = weights[valid_indexes]
    total_weight = float(np.sum(valid_weights))
    mean = float(np.average(ratios[valid_indexes], weights=valid_weights))
    first = int(valid_indexes[0])
    last = int(valid_indexes[-1])
    return Segment(
        contig=contig,
        start=int(starts[first]),
        end=int(ends[last]),
        mean=mean,
        weight=total_weight,
        bin_count=int(valid_indexes.size),
    )


def compute_contig_rho1(z_contig: NDArray[np.float64]) -> float:
    """Compute per-contig lag-1 autocorrelation on finite diploid-clipped z-scores."""
    mask = np.isfinite(z_contig) & (np.abs(z_contig) <= 2.5)
    z_clip = z_contig[mask]
    if z_clip.size < 3:
        return 0.0
    x = z_clip[:-1]
    y = z_clip[1:]
    std_x = float(np.std(x))
    std_y = float(np.std(y))
    if std_x < 1e-12 or std_y < 1e-12:
        return 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return float(np.clip(corr, 0.0, 0.95))


def evaluate_stouffer_segment_scores(
    bins_df: pl.DataFrame,
    segs_df: pl.DataFrame,
) -> pl.DataFrame:
    """Evaluate autocorrelation-corrected Stouffer Z-scores via asof interval join."""
    bins_sorted = bins_df.sort(["contig", "start"])
    segs_sorted = segs_df.sort(["contig", "seg_start"])

    joined = bins_sorted.join_asof(
        segs_sorted.select(["seg_id", "contig", "seg_start", "seg_end"]),
        left_on="start",
        right_on="seg_start",
        by="contig",
        strategy="backward",
        check_sortedness=False,
    ).filter(pl.col("end") <= pl.col("seg_end"))

    usable = joined.filter(pl.col("zscore").is_finite() & (pl.col("weight") > 0))

    if usable.height == 0:
        return segs_df.with_columns(
            [
                pl.lit(0).alias("n_usable"),
                pl.lit(0.0).alias("zscore"),
            ]
        ).sort("seg_id")

    stats = (
        usable.group_by("seg_id")
        .agg(
            [
                pl.len().alias("n_usable"),
                (pl.col("weight") * pl.col("zscore")).sum().alias("sum_w_z"),
                (pl.col("weight") * pl.col("weight")).sum().alias("sum_w_sq"),
                pl.col("rho1").first().alias("rho1"),
            ]
        )
        .with_columns(
            (
                pl.col("sum_w_sq")
                * (
                    1.0
                    + pl.when(pl.col("n_usable") > 1)
                    .then(
                        2.0
                        * (pl.col("n_usable") - 1)
                        / pl.col("n_usable")
                        * pl.col("rho1")
                    )
                    .otherwise(0.0)
                )
            )
            .sqrt()
            .alias("denom")
        )
        .with_columns(
            pl.when(pl.col("denom") > 1e-12)
            .then(pl.col("sum_w_z") / pl.col("denom"))
            .otherwise(0.0)
            .alias("zscore")
        )
    )
    return (
        segs_df.join(
            stats.select(["seg_id", "n_usable", "zscore"]),
            on="seg_id",
            how="left",
        )
        .with_columns(
            [
                pl.col("n_usable").fill_null(0),
                pl.col("zscore").fill_null(0.0),
            ]
        )
        .sort("seg_id")
    )
