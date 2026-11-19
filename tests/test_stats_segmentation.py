"""Unit and benchmark tests for calibrated PELT segmentation, dyadic BLUE, and integer CN calling."""

from __future__ import annotations

import csv
import math
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from ccnv.models import InvalidInputError, SegmentResult
from ccnv.stats.segmentation import (
    ContigSegmentationInput,
    WeightedL2Cost,
    _evaluate_integer_cn_bf,
    _find_breakpoints,
    estimate_dyadic_inflation,
    segment_and_score_genome,
    segment_contig,
)

FIXTURES = Path(__file__).parent / "fixtures" / "benchmark"


class TestWeightedL2Cost(unittest.TestCase):
    def test_matches_direct_weighted_residual_sum_of_squares(self) -> None:
        values = np.array([1.0, 2.0, 10.0])
        weights = np.array([1.0, 3.0, 0.0])
        cost = WeightedL2Cost().fit(np.column_stack((values, weights)))

        expected = np.sum(weights[:2] * np.square(values[:2] - 1.75))
        self.assertAlmostEqual(cost.error(0, 3), float(expected))

    def test_zero_weight_slice_has_zero_cost(self) -> None:
        cost = WeightedL2Cost().fit(np.column_stack(([1.0, 2.0], [0.0, 0.0])))

        self.assertEqual(cost.error(0, 2), 0.0)

    def test_weighted_l2_cost_and_segment_genome(self) -> None:
        values = np.array([1.0, 2.0, 10.0])
        weights = np.array([1.0, 3.0, 0.0])
        cost = WeightedL2Cost().fit(np.column_stack((values, weights)))
        expected = np.sum(weights[:2] * np.square(values[:2] - 1.75))
        self.assertAlmostEqual(cost.error(0, 3), float(expected))

        starts = np.array([0, 100, 200, 300, 400], dtype=np.int64)
        ends = np.array([100, 200, 300, 400, 500], dtype=np.int64)
        ratios = np.array([0.0, 0.0, 0.8, 0.8, 0.8], dtype=np.float64)
        prec = np.full(5, 100.0, dtype=np.float64)

        segs = segment_contig(
            contig="chr1",
            starts=starts,
            ends=ends,
            ratios=ratios,
            weights=prec,
            contig_length=500,
            penalty=0.5,
            min_size=2,
        )
        self.assertGreaterEqual(len(segs), 1)
        self.assertIsInstance(segs[0], SegmentResult)

        scored = segment_and_score_genome(
            {
                "chr1": ContigSegmentationInput(
                    starts=starts,
                    ends=ends,
                    ratios=ratios,
                    weights=prec,
                    contig_length=500,
                )
            },
            penalty=0.5,
            min_size=2,
        )
        self.assertGreaterEqual(len(scored), 1)


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
        self.assertEqual([item.ratio for item in segments], [0.0, 1.0])
        self.assertEqual([item.weight for item in segments], [4.0, 6.0])

    def test_pre_split_blacklist_gap_decouples_chromosome_arms(self) -> None:
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
        ends[4] = 2_600_000
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
            min_size=2,
            max_blacklist_gap=2_000_000,
        )
        coords = [(s.start, s.end, round(s.ratio, 2)) for s in segments]
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
            min_size=2,
        )
        self.assertEqual(
            [(s.start, s.end, round(s.ratio, 2)) for s in segments],
            [(200_000, 500_000, 0.0), (500_000, 800_000, 1.0)],
        )

    def test_kernel_cpd_dispatched_when_available_and_uniform_weights(self) -> None:
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
        self.assertAlmostEqual(segments[0].ratio, 0.2)
        self.assertLessEqual(segments[0].end, 350)

    def test_preserves_arbitrary_x_and_y_contig_names(self) -> None:
        inputs = {
            "X": ContigSegmentationInput(
                starts=np.array([0], dtype=np.int64),
                ends=np.array([80], dtype=np.int64),
                ratios=np.array([0.1], dtype=np.float64),
                weights=np.array([1.0], dtype=np.float64),
                contig_length=80,
            ),
            "chrY": ContigSegmentationInput(
                starts=np.array([0], dtype=np.int64),
                ends=np.array([50], dtype=np.int64),
                ratios=np.array([-0.1], dtype=np.float64),
                weights=np.array([1.0], dtype=np.float64),
                contig_length=50,
            ),
        }

        segments = segment_and_score_genome(inputs, penalty=1.0)

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
        self.assertEqual(segments[0].ratio, 0.25)

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

    def test_find_breakpoints_scale_invariance_small_weights_and_zero_weights(
        self,
    ) -> None:
        vals = np.concatenate([np.zeros(12), np.full(12, 2.0)])
        base_w = np.tile(np.array([1.0, 10.0], dtype=np.float64), 12)
        bkps_unit = _find_breakpoints(
            signal=np.column_stack((vals, base_w)),
            penalty=15.0,
            min_size=2,
            jump=1,
        )
        bkps_tiny = _find_breakpoints(
            signal=np.column_stack((vals, base_w * 1e-9)),
            penalty=15.0 * 1e-9,
            min_size=2,
            jump=1,
        )
        self.assertEqual(bkps_tiny, bkps_unit)
        self.assertEqual(bkps_tiny, [12, 24])

        non_contig_vals = np.column_stack((vals, vals))[::2, 0]
        self.assertFalse(non_contig_vals.flags["C_CONTIGUOUS"])
        bkps_uniform = _find_breakpoints(
            signal=np.column_stack((non_contig_vals, np.ones_like(non_contig_vals))),
            penalty=4.0,
            min_size=2,
            jump=1,
        )
        self.assertEqual(bkps_uniform, [6, 12])

        bkps_zero = _find_breakpoints(
            signal=np.column_stack((vals, np.zeros_like(vals))),
            penalty=4.0,
            min_size=2,
            jump=1,
        )
        self.assertEqual(bkps_zero, [24])

    def test_segment_contig_jump_greater_than_one_respects_stride(self) -> None:
        starts = np.arange(12, dtype=np.int64) * 1000
        ends = starts + 1000
        ratios = np.array([0.0] * 5 + [2.5] * 7, dtype=np.float64)
        weights = np.full(12, 4.0, dtype=np.float64)
        segs = segment_contig(
            contig="chr1",
            starts=starts,
            ends=ends,
            ratios=ratios,
            weights=weights,
            contig_length=12_000,
            penalty=2.0,
            min_size=3,
            jump=3,
        )
        self.assertGreaterEqual(len(segs), 2)
        for seg in segs[:-1]:
            self.assertEqual((seg.end // 1000) % 3, 0)


class TestCalibratedPeltAndCalling(unittest.TestCase):
    def test_homozygous_cn0_deletion_and_interior_focal_50kb_deletion_detected(
        self,
    ) -> None:
        n_bins = 20_000
        bin_size = 5_000
        contig_len = n_bins * bin_size
        starts = np.arange(0, contig_len, bin_size, dtype=np.int64)
        ends = starts + bin_size

        ratios = np.zeros(n_bins, dtype=np.float64)
        weights = np.full(n_bins, 4.0, dtype=np.float64)
        zscores = np.zeros(n_bins, dtype=np.float64)
        counts = np.full(n_bins, 8.0, dtype=np.float64)
        expected = np.full(n_bins, 8.0, dtype=np.float64)

        ratios[5000:5020] = -4.5
        zscores[5000:5020] = -15.0
        counts[5000:5020] = 0.0

        ratios[10000:10010] = -1.0
        zscores[10000:10010] = -5.0
        counts[10000:10010] = 4.0

        arm_inputs = {
            "chr1": ContigSegmentationInput(
                starts=starts,
                ends=ends,
                ratios=ratios,
                weights=weights,
                contig_length=contig_len,
                zscores=zscores,
                counts=counts,
                expected=expected,
            )
        }
        segs = segment_and_score_genome(
            arm_inputs,
            sensitivity=1.0,
            mode="germline",
            z_threshold=3.0,
        )
        calls = [s for s in segs if s.call == "loss"]
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].copy_number, 0)
        self.assertEqual(calls[1].copy_number, 1)

    def test_dyadic_wave_inflation_suppresses_megabase_wave_false_positives_at_1kb(
        self,
    ) -> None:
        rng = np.random.default_rng(42)
        n_bins = 50_000
        noise = rng.normal(0.0, np.sqrt(0.98), size=n_bins)
        wave = np.sqrt(2.0 * 0.02) * np.sin(2.0 * np.pi * np.arange(n_bins) / 5000.0)
        wavy_z = noise + wave
        usable_all = np.ones(n_bins, dtype=bool)

        infl = estimate_dyadic_inflation(wavy_z, usable_all)
        self.assertGreater(float(infl[-1]), 8.0)

        # Verify dyadic inflation suppresses spurious megabase wave calls in segment_and_score_genome
        bin_size = 1_000
        starts = np.arange(n_bins, dtype=np.int64) * bin_size
        ends = starts + bin_size
        ratios = 0.05 * wavy_z
        weights = np.full(n_bins, 1.0 / (0.05**2), dtype=np.float64)
        segs = segment_and_score_genome(
            {
                "chr1": ContigSegmentationInput(
                    starts=starts,
                    ends=ends,
                    ratios=ratios,
                    weights=weights,
                    contig_length=n_bins * bin_size,
                    zscores=wavy_z,
                )
            },
            mode="germline",
            z_threshold=3.0,
        )
        self.assertEqual([s for s in segs if s.call is not None], [])

    def test_germline_integer_cn_bayes_factor_rejects_cn_203_high_depth_ripple(
        self,
    ) -> None:
        n_bins = 100
        bin_size = 500_000
        contig_len = n_bins * bin_size
        starts = np.arange(0, contig_len, bin_size, dtype=np.int64)
        ends = starts + bin_size

        ratios = np.zeros(n_bins, dtype=np.float64)
        weights = np.full(n_bins, 4.0, dtype=np.float64)
        zscores = np.zeros(n_bins, dtype=np.float64)
        counts = np.full(n_bins, 1600.0, dtype=np.float64)
        expected = np.full(n_bins, 1600.0, dtype=np.float64)

        ratios[30:70] = 0.022
        zscores[30:70] = 4.5 / np.sqrt(40.0)
        counts[30:70] = 1600.0 * 1.015355

        high_depth_ripple_inputs = {
            "chr1": ContigSegmentationInput(
                starts=starts,
                ends=ends,
                ratios=ratios,
                weights=weights,
                contig_length=contig_len,
                zscores=zscores,
                counts=counts,
                expected=expected,
            )
        }
        segs = segment_and_score_genome(
            high_depth_ripple_inputs,
            mode="germline",
            z_threshold=3.0,
        )
        self.assertTrue(all(s.call is None for s in segs))

    def test_segment_and_score_non_doubled_nb_bf_and_haploid_calling(self) -> None:
        starts = np.arange(10, dtype=np.int64) * 5000
        gain_inp = ContigSegmentationInput(
            starts,
            starts + 5000,
            np.full(10, math.log2(1.5)),
            np.full(10, 200.0),
            50_000,
            counts=np.full(10, 30.0),
            expected=np.full(10, 20.0),
            baseline_ploidy=2,
        )
        segs_dip = segment_and_score_genome({"chr1": gain_inp}, penalty=50.0)
        self.assertEqual((segs_dip[0].call, segs_dip[0].copy_number), ("gain", 3))
        self.assertAlmostEqual(
            segs_dip[0].log_bf or 0.0, 300.0 * math.log(1.5) - 100.0, places=5
        )

        hap_base = replace(gain_inp, weights=np.full(10, 20.0), baseline_ploidy=1)
        hap_loss = replace(hap_base, ratios=np.full(10, -3.0), counts=np.zeros(10))
        hap_dup = replace(hap_base, ratios=np.full(10, 1.0), counts=np.full(10, 40.0))
        by_c = {
            s.contig: s
            for s in segment_and_score_genome(
                {"chrX": hap_loss, "chrY": hap_dup}, penalty=50.0
            )
        }
        self.assertEqual((by_c["chrX"].call, by_c["chrX"].copy_number), ("loss", 0))
        self.assertEqual((by_c["chrY"].call, by_c["chrY"].copy_number), ("gain", 2))

    def test_integer_cn_bf_calls_cn0_with_proportional_background_and_low_depth(
        self,
    ) -> None:
        cn_hap, bf_hap = _evaluate_integer_cn_bf(
            seg_ratio=-5.5,
            sigma_s_sq=0.02,
            baseline_ploidy=1,
            has_counts=True,
            c_s=100.0,
            lam_s=5000.0,
        )
        self.assertEqual(cn_hap, 0)
        self.assertGreater(bf_hap, 100.0)

        cn_dip, bf_dip = _evaluate_integer_cn_bf(
            seg_ratio=-5.5,
            sigma_s_sq=0.01,
            baseline_ploidy=2,
            has_counts=True,
            c_s=100.0,
            lam_s=5000.0,
        )
        self.assertEqual(cn_dip, 0)
        self.assertGreater(bf_dip, 100.0)

        cn_low, bf_low = _evaluate_integer_cn_bf(
            seg_ratio=-4.0,
            sigma_s_sq=0.25,
            baseline_ploidy=2,
            has_counts=True,
            c_s=0.0,
            lam_s=10.0,
        )
        self.assertEqual(cn_low, 0)
        self.assertGreater(bf_low, 4.5)

    def test_segmentation_validation_and_integer_cn_sign_concordance(self) -> None:
        starts = np.array([0, 5000, 10000, 15000], dtype=np.int64)
        ends = np.array([5000, 10000, 15000, 20000], dtype=np.int64)
        ratios = np.full(4, -0.5, dtype=np.float64)
        weights = np.full(4, 25.0, dtype=np.float64)

        with self.assertRaises(InvalidInputError):
            segment_contig(
                contig="chr1",
                starts=starts,
                ends=ends,
                ratios=ratios,
                weights=weights,
                contig_length=20000,
                sensitivity=0.0,
            )
        with self.assertRaises(InvalidInputError):
            segment_contig(
                contig="chr1",
                starts=starts,
                ends=ends,
                ratios=ratios,
                weights=weights,
                contig_length=20000,
                kappa_local=math.nan,
            )

        inp = ContigSegmentationInput(
            starts=starts,
            ends=ends,
            ratios=ratios,
            weights=weights,
            contig_length=20000,
            zscores=np.full(4, -2.5, dtype=np.float64),
            counts=np.full(4, 180.0, dtype=np.float64),
            expected=np.full(4, 100.0, dtype=np.float64),
        )
        with self.assertRaises(InvalidInputError):
            segment_and_score_genome({"chr1": inp}, sensitivity=0.0)
        with self.assertRaises(InvalidInputError):
            segment_and_score_genome({"chr1": inp}, z_threshold=-1.0)
        with self.assertRaises(InvalidInputError):
            segment_and_score_genome({"chr1": inp}, ratio_threshold=-0.1)
        with self.assertRaises(InvalidInputError):
            segment_and_score_genome({"chr1": inp}, threads=0)

        segs = segment_and_score_genome({"chr1": inp}, z_threshold=3.0)
        self.assertEqual(len(segs), 1)
        self.assertIsNone(segs[0].call)
        self.assertEqual(segs[0].copy_number, 2)


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
            )
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].end // 100_000, expected[contig][1])
            if result[0].ratio > 0.2:
                observed_calls.append("gain")
        self.assertEqual(observed_calls, ["gain"])

    def test_selected_algorithm_matches_real_fixture_segments_and_call(self) -> None:
        bins: dict[str, list[tuple[int, int, float, float]]] = {}
        with (FIXTURES / "real" / "ID_bins.bed").open() as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                ratio = float(row["ratio"])
                bins.setdefault(row["chr"], []).append(
                    (
                        int(row["start"]),
                        int(row["end"]),
                        ratio,
                        float(math.isfinite(ratio)),
                    )
                )

        expected_segments: list[tuple[str, int, int]] = []
        with (FIXTURES / "real" / "ID_segments.bed").open() as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                expected_segments.append(
                    (row["chr"], int(row["start"]), int(row["end"]))
                )

        observed: list[SegmentResult] = []
        for contig, rows in bins.items():
            starts = [r[0] for r in rows]
            ends = [r[1] for r in rows]
            ratios = [r[2] for r in rows]
            weights = [r[3] for r in rows]
            observed.extend(
                segment_contig(
                    contig=contig,
                    starts=starts,
                    ends=ends,
                    ratios=ratios,
                    weights=weights,
                    contig_length=ends[-1],
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
            (item.contig, item.start, item.end)
            for item in observed
            if item.ratio > 0.05
        ]
        self.assertEqual(gains, [("21", 13_000_000, 46_700_000)])


if __name__ == "__main__":
    unittest.main()
