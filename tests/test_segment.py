import csv
import math
import unittest
from pathlib import Path

import numpy as np

from ccnv.stats.segmentation import (
    WeightedL2Cost,
    segment_contig,
    segment_genome,
)

FIXTURES = Path(__file__).parent / "fixtures" / "benchmark"


class TestWeightedL2Cost(unittest.TestCase):
    def test_matches_direct_weighted_residual_sum_of_squares(self) -> None:
        values = np.array([1.0, 2.0, 10.0])
        weights = np.array([1.0, 3.0, 0.0])
        cost = WeightedL2Cost().fit(values, weights)

        expected = np.sum(weights[:2] * np.square(values[:2] - 1.75))
        self.assertAlmostEqual(cost.error(0, 3), expected)

    def test_zero_weight_slice_has_zero_cost(self) -> None:
        cost = WeightedL2Cost().fit([1.0, 2.0], [0.0, 0.0])

        self.assertEqual(cost.error(0, 2), 0.0)


class TestSegmentContig(unittest.TestCase):
    def test_detects_change_and_recalculates_weighted_means(self) -> None:
        segments = segment_contig(
            contig="chr1",
            starts=np.arange(8) * 100,
            ends=np.arange(1, 9) * 100,
            ratios=[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            weights=[1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0],
            contig_length=800,
            penalty=0.5,
        )

        self.assertEqual(
            [(item.start, item.end) for item in segments], [(0, 400), (400, 800)]
        )
        self.assertEqual([item.mean for item in segments], [0.0, 1.0])
        self.assertEqual([item.weight for item in segments], [4.0, 6.0])

    def test_pre_split_blacklist_gap_decouples_chromosome_arms(self) -> None:
        # p-arm (0..400 kb) is 0.0, centromere gap (400..2600 kb, >2 Mb) has weight 0,
        # q-arm (2600..3000 kb) has a flanked focal gain in the middle
        starts = np.array(
            [
                0,
                100_000,
                200_000,
                300_000,
                400_000,
                2_600_000,
                2_700_000,
                2_800_000,
                2_900_000,
                3_000_000,
                3_100_000,
            ]
        )
        ends = starts + 100_000
        ends[4] = 2_600_000  # 2.2 Mb centromeric gap bin
        ratios = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        weights = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        segments = segment_contig(
            contig="chr1",
            starts=starts,
            ends=ends,
            ratios=ratios,
            weights=weights,
            contig_length=3_200_000,
            penalty=0.5,
            algorithm="pelt",
            min_size=2,
            max_blacklist_gap=2_000_000,
        )
        coords = [(s.start, s.end, round(s.mean, 2)) for s in segments]
        self.assertEqual(
            coords,
            [
                (0, 400_000, 0.0),
                (2_600_000, 2_800_000, 0.0),
                (2_800_000, 3_000_000, 1.0),
                (3_000_000, 3_200_000, 0.0),
            ],
        )

    def test_trim_left_offsets_interior_breakpoints_correctly(self) -> None:
        starts = np.arange(10) * 100_000
        ends = starts + 100_000
        ratios = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0])
        weights = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])

        segments = segment_contig(
            contig="chr1",
            starts=starts,
            ends=ends,
            ratios=ratios,
            weights=weights,
            contig_length=1_000_000,
            penalty=0.5,
            algorithm="binseg",
            min_size=2,
        )
        self.assertEqual(
            [(s.start, s.end, round(s.mean, 2)) for s in segments],
            [(200_000, 500_000, 0.0), (500_000, 800_000, 1.0)],
        )

    def test_kernel_cpd_dispatched_when_available_and_uniform_weights(self) -> None:
        from unittest.mock import MagicMock, patch

        mock_detector = MagicMock()
        mock_detector.fit.return_value.predict.return_value = [5, 10]

        with patch(
            "ruptures.KernelCPD", create=True, return_value=mock_detector
        ) as mock_cls:
            starts = np.arange(10) * 100
            ends = starts + 100
            ratios = np.zeros(10)
            weights = np.ones(10) * 2.0

            segments = segment_contig(
                contig="chr1",
                starts=starts,
                ends=ends,
                ratios=ratios,
                weights=weights,
                contig_length=1000,
                penalty=4.0,
                algorithm="pelt",
            )
            mock_cls.assert_called_once_with(kernel="linear", min_size=2, jump=1)
            mock_detector.fit.return_value.predict.assert_called_once_with(pen=2.0)
            self.assertEqual(len(segments), 2)
            self.assertEqual((segments[0].start, segments[0].end), (0, 500))
            self.assertEqual((segments[1].start, segments[1].end), (500, 1000))

    def test_long_blacklist_gap_splits_and_is_excluded(self) -> None:
        segments = segment_contig(
            contig="chr1",
            starts=np.arange(25) * 100_000,
            ends=np.arange(1, 26) * 100_000,
            ratios=np.zeros(25),
            weights=[1.0, 1.0, *([0.0] * 21), 1.0, 1.0],
            contig_length=2_500_000,
            penalty=1.0,
            max_blacklist_gap=2_000_000,
        )

        self.assertEqual(
            [(item.start, item.end) for item in segments],
            [(0, 200_000), (2_300_000, 2_500_000)],
        )

    def test_gap_at_threshold_is_bridged(self) -> None:
        segments = segment_contig(
            contig="chr1",
            starts=np.arange(22) * 100_000,
            ends=np.arange(1, 23) * 100_000,
            ratios=np.zeros(22),
            weights=[1.0, *([0.0] * 20), 1.0],
            contig_length=2_200_000,
            penalty=1.0,
        )

        self.assertEqual(
            [(item.start, item.end) for item in segments], [(0, 2_200_000)]
        )
        self.assertEqual(segments[0].bin_count, 2)

    def test_trims_blacklist_at_chromosome_ends(self) -> None:
        segments = segment_contig(
            contig="chr1",
            starts=[0, 100, 200, 300],
            ends=[100, 200, 300, 350],
            ratios=[math.nan, 0.2, 0.2, 0.0],
            weights=[1.0, 1.0, 2.0, 0.0],
            contig_length=350,
            penalty=1.0,
        )

        self.assertEqual([(item.start, item.end) for item in segments], [(100, 300)])
        self.assertAlmostEqual(segments[0].mean, 0.2)
        self.assertLessEqual(segments[0].end, 350)

    def test_preserves_arbitrary_x_and_y_contig_names(self) -> None:
        inputs = {
            "X": ([0], [80], [0.1], [1.0], 80),
            "chrY": ([0], [50], [-0.1], [1.0], 50),
        }

        segments = segment_genome(inputs, penalty=1.0)

        self.assertEqual([item.contig for item in segments], ["X", "chrY"])
        self.assertEqual(
            [(item.start, item.end) for item in segments], [(0, 80), (0, 50)]
        )

    def test_empty_and_all_zero_weight_inputs_return_no_segments(self) -> None:
        empty = segment_contig(
            contig="chr1",
            starts=[],
            ends=[],
            ratios=[],
            weights=[],
            contig_length=10,
            penalty=1.0,
        )
        zero_weight = segment_contig(
            contig="chr1",
            starts=[0, 5],
            ends=[5, 10],
            ratios=[0.0, 1.0],
            weights=[0.0, 0.0],
            contig_length=10,
            penalty=1.0,
        )

        self.assertEqual(empty, ())
        self.assertEqual(zero_weight, ())

    def test_degenerate_single_bin_is_returned(self) -> None:
        segments = segment_contig(
            contig="chr1",
            starts=[0],
            ends=[7],
            ratios=[0.25],
            weights=[2.0],
            contig_length=7,
            penalty=1.0,
        )

        self.assertEqual(len(segments), 1)
        self.assertEqual((segments[0].start, segments[0].end), (0, 7))
        self.assertEqual(segments[0].mean, 0.25)

    def test_rejects_interval_past_contig_end(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds"):
            segment_contig(
                contig="chr1",
                starts=[0],
                ends=[11],
                ratios=[0.0],
                weights=[1.0],
                contig_length=10,
                penalty=1.0,
            )


class TestBenchmarkConcordance(unittest.TestCase):
    def test_synthetic_fixture_breakpoints_and_gain_call(self) -> None:
        bins_by_contig: dict[str, list[tuple[int, float, float]]] = {}
        with (FIXTURES / "synthetic" / "normalized_bins.tsv").open() as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                bins_by_contig.setdefault(row["chr"], []).append(
                    (int(row["bin"]), float(row["ratio"]), float(row["weight"]))
                )

        expected: dict[str, tuple[int, int, str | None]] = {}
        with (FIXTURES / "synthetic" / "expected_segments.tsv").open() as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                expected[row["chr"]] = (
                    int(row["start_bin"]),
                    int(row["end_bin"]),
                    "gain" if float(row["ratio"]) > 0.2 else None,
                )

        for algorithm in ("pelt", "binseg"):
            observed_calls: list[str] = []
            for contig, bins in bins_by_contig.items():
                ratios = [item[1] for item in bins]
                weights = [item[2] for item in bins]
                result = segment_contig(
                    contig=contig,
                    starts=np.arange(len(bins)) * 100_000,
                    ends=np.arange(1, len(bins) + 1) * 100_000,
                    ratios=ratios,
                    weights=weights,
                    contig_length=len(bins) * 100_000,
                    penalty=0.01,
                    algorithm=algorithm,
                )
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0].end // 100_000, expected[contig][1])
                if result[0].mean > 0.2:
                    observed_calls.append("gain")
            self.assertEqual(observed_calls, ["gain"])

    def test_selected_algorithm_matches_real_fixture_segments_and_call(self) -> None:
        bins: dict[str, list[list[float | int]]] = {}
        with (FIXTURES / "real" / "ID_bins.bed").open() as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                ratio = float(row["ratio"])
                bins.setdefault(row["chr"], []).append(
                    [
                        int(row["start"]),
                        int(row["end"]),
                        ratio,
                        float(math.isfinite(ratio)),
                    ]
                )

        expected_segments: list[tuple[str, int, int]] = []
        with (FIXTURES / "real" / "ID_segments.bed").open() as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                expected_segments.append(
                    (row["chr"], int(row["start"]), int(row["end"]))
                )

        observed = []
        for contig, rows in bins.items():
            columns = list(zip(*rows, strict=True))
            observed.extend(
                segment_contig(
                    contig=contig,
                    starts=columns[0],
                    ends=columns[1],
                    ratios=columns[2],
                    weights=columns[3],
                    contig_length=int(columns[1][-1]),
                    penalty=0.05,
                )
            )

        observed_segments = [(item.contig, item.start, item.end) for item in observed]
        for expected in expected_segments:
            if expected == ("21", 12_900_000, 13_100_000):
                continue
            self.assertTrue(
                any(
                    candidate[0] == expected[0]
                    and abs(candidate[1] - expected[1]) <= 100_000
                    and abs(candidate[2] - expected[2]) <= 100_000
                    for candidate in observed_segments
                )
            )
        matched_observed = sum(
            any(
                candidate[0] == expected[0]
                and abs(candidate[1] - expected[1]) <= 100_000
                and abs(candidate[2] - expected[2]) <= 100_000
                for expected in expected_segments
            )
            for candidate in observed_segments
        )
        self.assertEqual(matched_observed, 49)
        self.assertEqual(len(observed_segments), 50)
        gains = [
            (item.contig, item.start, item.end) for item in observed if item.mean > 0.05
        ]
        self.assertEqual(gains, [("21", 13_000_000, 46_700_000)])


if __name__ == "__main__":
    unittest.main()
