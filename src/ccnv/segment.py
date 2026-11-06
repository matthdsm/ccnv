"""Python-native weighted copy-number segmentation.

The ``penalty`` is measured in weighted squared-error units. PELT minimizes
``sum(weighted squared error) + penalty * number of breakpoints``; Binseg
greedily accepts a breakpoint only when its weighted-error reduction exceeds
the penalty.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
import ruptures as rpt
from numpy.typing import ArrayLike, NDArray
from ruptures.base import BaseCost

logger = logging.getLogger(__name__)

Algorithm = Literal["pelt", "binseg"]
DEFAULT_MAX_BLACKLIST_GAP = 2_000_000


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

    signal = np.column_stack((np.where(usable, ratio_array, 0.0), weight_array))
    breakpoints = _find_breakpoints(
        signal=signal,
        penalty=penalty,
        algorithm=algorithm,
        min_size=min_size,
        jump=jump,
    )

    segments: list[Segment] = []
    left = 0
    for right in breakpoints:
        split_ranges = _split_blacklist_gaps(
            left,
            right,
            starts=start_array,
            ends=end_array,
            usable=usable,
            max_gap=max_blacklist_gap,
        )
        for split_left, split_right in split_ranges:
            segment = _make_segment(
                contig,
                split_left,
                split_right,
                start_array,
                end_array,
                ratio_array,
                weight_array,
                usable,
            )
            if segment is not None:
                segments.append(segment)
        left = right
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
    return np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(values)))


def _find_breakpoints(
    *,
    signal: NDArray[np.float64],
    penalty: float,
    algorithm: Algorithm,
    min_size: int,
    jump: int,
) -> list[int]:
    if signal.shape[0] < 2 * min_size:
        return [signal.shape[0]]
    cost = WeightedL2Cost()
    if algorithm == "pelt":
        detector = rpt.Pelt(custom_cost=cost, min_size=min_size, jump=jump)
    elif algorithm == "binseg":
        detector = rpt.Binseg(custom_cost=cost, min_size=min_size, jump=jump)
    else:
        raise ValueError(f"unknown segmentation algorithm: {algorithm}")
    return detector.fit(signal).predict(pen=penalty)


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
