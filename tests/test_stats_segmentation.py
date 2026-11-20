"""Unit and benchmark tests for calibrated PELT segmentation, dyadic BLUE, and integer CN calling."""

from __future__ import annotations

import csv
import math
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import ruptures as rpt
from numpy.typing import NDArray
from ruptures.base import BaseCost

from ccnv._core import pelt_weighted_l2
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

BaseCost.register(WeightedL2Cost)

FIXTURES = Path(__file__).parent / "fixtures" / "benchmark"


class TestWeightedL2Cost(unittest.TestCase):
    def test_matches_direct_weighted_residual_sum_of_squares(self) -> None:
        values = np.array([1.0, 2.0, 10.0])
        weights = np.array([1.0, 3.0, 0.0])
        cost = WeightedL2Cost().fit(np.column_stack((values, weights)))

        expected = np.sum(weights[:2] * np.square(values[:2] - 1.75))
        self.assertAlmostEqual(cost.error(0, 3), float(expected))
        self.assertAlmostEqual(cost.sum_of_costs([2, 3]), float(expected))

    def test_zero_weight_slice_has_zero_cost(self) -> None:
        cost = WeightedL2Cost().fit(np.column_stack(([1.0, 2.0], [0.0, 0.0])))

        self.assertEqual(cost.error(0, 2), 0.0)
        self.assertEqual(cost.sum_of_costs([]), 0.0)
        self.assertEqual(cost.sum_of_costs([1, 2]), 0.0)
        with self.assertRaises(InvalidInputError):
            cost.sum_of_costs([2, 1])

    def test_weighted_l2_cost_and_segment_genome(self) -> None:
        values = np.array([1.0, 2.0, 10.0])
        weights = np.array([1.0, 3.0, 0.0])
        cost = WeightedL2Cost().fit(np.column_stack((values, weights)))
        expected = np.sum(weights[:2] * np.square(values[:2] - 1.75))
        self.assertAlmostEqual(cost.error(0, 3), float(expected))
        self.assertAlmostEqual(cost.sum_of_costs([2, 3]), float(expected))

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

    def test_uniform_weights_segment_contig_matches_rust_pelt(self) -> None:
        starts = np.arange(10, dtype=np.int64) * 100
        ends = starts + 100
        ratios = np.array([0.0] * 5 + [1.2] * 5, dtype=np.float64)
        weights = np.full(10, 2.0, dtype=np.float64)
        signal = np.column_stack((ratios, weights))

        segments = segment_contig(
            contig="chr1",
            starts=starts,
            ends=ends,
            ratios=ratios,
            weights=weights,
            contig_length=1000,
            penalty=2.0,
        )
        rust_bkps = pelt_weighted_l2(signal, 2.0, 2, 1)
        self.assertEqual(rust_bkps, [5, 10])
        self.assertEqual(
            [(seg.start, seg.end) for seg in segments],
            [(0, 500), (500, 1000)],
        )

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

    def test_integer_cn_bf_deflates_spatial_autocorrelation_and_avoids_nb_double_count(
        self,
    ) -> None:
        from ccnv.stats.neighbors import LN2_SQ

        n_wave, n_del = 256, 64
        s_wave = np.arange(n_wave, dtype=np.int64) * 5000
        # Smooth diploid wave on chr1 -> genome-wide dyadic autocorrelation kappa_s > 3.0
        wave_inp = ContigSegmentationInput(
            s_wave,
            s_wave + 5000,
            0.35 * np.sin(2.0 * np.pi * np.arange(n_wave) / 32.0),
            np.full(n_wave, 25.0),
            n_wave * 5000,
            counts=np.full(n_wave, 25.0),
            expected=np.full(n_wave, 25.0),
            baseline_ploidy=2,
        )
        s_del = np.arange(n_del, dtype=np.int64) * 5000
        del_inp = ContigSegmentationInput(
            s_del,
            s_del + 5000,
            np.full(n_del, -4.0),
            np.full(n_del, 2.0),
            n_del * 5000,
            counts=np.zeros(n_del),
            expected=np.full(n_del, 2.0 / n_del),
            baseline_ploidy=2,
            poisson_pooled=True,
        )
        segs_all = segment_and_score_genome(
            {"chr1": wave_inp, "chr2": del_inp}, penalty=500.0
        )
        chr2_segs = [s for s in segs_all if s.contig == "chr2"]
        # Raw lam_s=2.0 without kappa_s deflation would return bf ~ 1.98; deflated bf must be < 1.2
        self.assertEqual(len(chr2_segs), 1)
        self.assertLess(float(chr2_segs[0].log_bf or 0.0), 1.2)

        _, bf_uncorr = _evaluate_integer_cn_bf(
            seg_ratio=-4.0,
            sigma_s_sq=0.50,
            baseline_ploidy=2,
            has_counts=True,
            c_s=0.0,
            lam_s=2.0,
            kappa_s=1.0,
        )
        _, bf_corr = _evaluate_integer_cn_bf(
            seg_ratio=-4.0,
            sigma_s_sq=0.50,
            baseline_ploidy=2,
            has_counts=True,
            c_s=0.0,
            lam_s=2.0,
            kappa_s=4.0,
        )
        self.assertAlmostEqual(bf_uncorr, 1.98, places=2)
        self.assertAlmostEqual(bf_corr, 0.495, places=2)

        _, bf_mod_corr = _evaluate_integer_cn_bf(
            seg_ratio=-4.0,
            sigma_s_sq=4.0 / (LN2_SQ * 100.0),
            baseline_ploidy=2,
            has_counts=True,
            c_s=0.0,
            lam_s=100.0,
            kappa_s=4.0,
        )
        self.assertAlmostEqual(bf_mod_corr, 24.75, delta=0.5)


class TestRustPeltRupturesParity(unittest.TestCase):
    @staticmethod
    def _oracle_bkps(
        signal: NDArray[np.float64],
        penalty: float,
        min_size: int,
        jump: int,
    ) -> list[int]:
        detector = rpt.Pelt(
            custom_cost=WeightedL2Cost(), min_size=min_size, jump=jump
        ).fit(signal)
        return [int(bp) for bp in detector.predict(pen=penalty)]

    def _assert_parity(
        self,
        signal: NDArray[np.float64],
        penalty: float,
        min_size: int,
        jump: int,
    ) -> list[int]:
        expected = self._oracle_bkps(signal, penalty, min_size, jump)
        rust_bkps = pelt_weighted_l2(signal, penalty, min_size, jump)
        py_bkps = _find_breakpoints(
            signal=signal, penalty=penalty, min_size=min_size, jump=jump
        )
        self.assertEqual(
            rust_bkps,
            expected,
            f"Rust PELT mismatch (n={signal.shape[0]}, pen={penalty}, min_size={min_size}, jump={jump})",
        )
        self.assertEqual(py_bkps, expected)
        return rust_bkps

    def test_grid_min_size_jump_penalties_and_weight_regimes(self) -> None:
        rng = np.random.default_rng(2026)
        n = 48
        base_vals = np.concatenate(
            [
                rng.normal(0.0, 0.15, 13),
                rng.normal(1.2, 0.2, 11),
                rng.normal(-0.8, 0.15, 14),
                rng.normal(0.1, 0.1, 10),
            ]
        )
        alt_weights = np.tile(np.array([1e-9, 1e6], dtype=np.float64), n // 2)
        hetero_weights = rng.uniform(0.2, 25.0, size=n)
        uniform_weights = np.full(n, 3.5, dtype=np.float64)
        zero_slice_weights = hetero_weights.copy()
        zero_slice_weights[8:12] = 0.0
        zero_slice_weights[35:39] = 0.0

        nan_inf_vals = base_vals.copy()
        nan_inf_vals[[2, 19, 41]] = [math.nan, math.inf, -math.inf]
        nan_inf_weights = hetero_weights.copy()
        nan_inf_weights[[5, 24, 44]] = [math.nan, math.inf, -math.inf]

        signals = [
            np.column_stack((base_vals, uniform_weights)),
            np.column_stack((base_vals, hetero_weights)),
            np.column_stack((base_vals, alt_weights)),
            np.column_stack((base_vals, zero_slice_weights)),
            np.column_stack((nan_inf_vals, nan_inf_weights)),
        ]
        for sig in signals:
            for min_size in (1, 2, 3, 5, 10):
                for jump in (1, 2, 3, 5):
                    for penalty in (0.01, 2.5, 1e9):
                        self._assert_parity(sig, penalty, min_size, jump)

    def test_short_blocks_and_empty_inputs(self) -> None:
        empty = np.empty((0, 2), dtype=np.float64)
        self.assertEqual(pelt_weighted_l2(empty, 1.0, 2, 1), [])
        self.assertEqual(
            _find_breakpoints(signal=empty, penalty=1.0, min_size=2, jump=1), []
        )

        for n in (1, 2, 3, 7, 9):
            sig = np.column_stack(
                (np.arange(n, dtype=np.float64), np.ones(n, dtype=np.float64))
            )
            for min_size in (2, 5, 10):
                if n >= 2 * min_size:
                    continue
                for jump in (1, 2, 3):
                    self.assertEqual(pelt_weighted_l2(sig, 1.0, min_size, jump), [n])
                    self.assertEqual(
                        _find_breakpoints(
                            signal=sig, penalty=1.0, min_size=min_size, jump=jump
                        ),
                        [n],
                    )
                    if n >= min_size:
                        self.assertEqual(
                            self._oracle_bkps(sig, 1.0, min_size, jump), [n]
                        )

    def test_tie_breaking_and_strided_non_contiguous_arrays(self) -> None:
        tie_signals = [
            np.column_stack((np.zeros(12), np.ones(12))),
            np.column_stack((np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]), np.ones(6))),
            np.column_stack((np.array([0.0] * 3 + [2.0] * 3 + [0.0] * 3), np.ones(9))),
        ]
        for sig in tie_signals:
            for min_size in (1, 2, 3):
                for jump in (1, 2):
                    for penalty in (0.1, 1.0, 5.0):
                        self._assert_parity(sig, penalty, min_size, jump)

        raw = np.zeros((24, 4), dtype=np.float64)
        raw[::2, 0] = np.concatenate([np.zeros(6), np.full(6, 1.5)])
        raw[::2, 2] = np.linspace(1.0, 5.0, 12)
        strided = raw[::2, ::2]
        self.assertFalse(strided.flags["C_CONTIGUOUS"])
        self._assert_parity(strided, 1.0, 2, 1)

        fortran = np.asfortranarray(strided)
        self.assertFalse(fortran.flags["C_CONTIGUOUS"])
        self._assert_parity(fortran, 1.0, 2, 1)

        self._assert_parity(strided, 0.1, 2, 20)
        self._assert_parity(np.full((8, 2), math.nan), 1.0, 2, 1)
        single_w = np.column_stack(
            (np.arange(8.0), [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )
        self._assert_parity(single_w, 0.5, 2, 1)
        tiny = np.column_stack((strided[:, 0], np.full(12, 1e-12)))
        self._assert_parity(tiny, 1e-12, 2, 1)
        shifted = np.column_stack((strided[:, 0] + 1e6, strided[:, 1]))
        self.assertEqual(self._assert_parity(shifted, 1.0, 2, 1), [6, 12])
        for seed, ms, jp, pen in (
            (18, 2, 2, 0.1),
            (12, 1, 2, 1e-12),
            (12, 2, 2, 1e-12),
        ):
            rng_s = np.random.default_rng(seed)
            n_s = int(rng_s.integers(6, 55))
            v_s, w_s = rng_s.normal(0.0, 1.0, n_s), rng_s.exponential(2.0, n_s)
            v_s[rng_s.random(n_s) < 0.15] = math.nan
            w_s[rng_s.random(n_s) < 0.15] = 0.0
            self._assert_parity(np.column_stack((v_s, w_s)), pen, ms, jp)

    def test_benchmark_fixtures_exact_parity(self) -> None:
        synth_by_chr: dict[str, list[tuple[float, float]]] = {}
        with (FIXTURES / "synthetic" / "normalized_bins.tsv").open() as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                synth_by_chr.setdefault(row["chr"], []).append(
                    (float(row["ratio"]), float(row["weight"]))
                )
        for rows in synth_by_chr.values():
            sig = np.asarray(rows, dtype=np.float64)
            for min_size in (1, 2):
                for jump in (1, 2):
                    for penalty in (0.001, 0.01, 10.0):
                        self._assert_parity(sig, penalty, min_size, jump)

        real_by_chr: dict[str, list[tuple[float, float]]] = {}
        with (FIXTURES / "real" / "ID_bins.bed").open() as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                ratio = float(row["ratio"])
                real_by_chr.setdefault(row["chr"], []).append(
                    (ratio, float(math.isfinite(ratio)))
                )
        for contig, rows in real_by_chr.items():
            sig = np.asarray(rows, dtype=np.float64)
            self._assert_parity(sig, 0.05, 2, 1)
            if contig in ("1", "21"):
                for min_size in (2, 5):
                    for jump in (1, 3):
                        self._assert_parity(sig, 0.5, min_size, jump)

    def test_input_validation_raises_invalid_input_error(self) -> None:
        valid = np.column_stack((np.zeros(6), np.ones(6)))
        bad_cols_1 = np.zeros((6, 1), dtype=np.float64)
        bad_cols_3 = np.zeros((6, 3), dtype=np.float64)
        for bad_sig in (bad_cols_1, bad_cols_3):
            with self.assertRaises(InvalidInputError):
                pelt_weighted_l2(bad_sig, 1.0, 2, 1)

        neg_weight = np.column_stack((np.array([0.0, 1.0]), np.array([1.0, -0.5])))
        with self.assertRaises(InvalidInputError):
            pelt_weighted_l2(neg_weight, 1.0, 2, 1)

        for bad_min in (0, -1):
            with self.assertRaises(InvalidInputError):
                pelt_weighted_l2(valid, 1.0, bad_min, 1)

        for bad_jump in (0, -2):
            with self.assertRaises(InvalidInputError):
                pelt_weighted_l2(valid, 1.0, 2, bad_jump)

        for bad_pen in (0.0, -1.0, math.nan, math.inf, -math.inf):
            with self.assertRaises(InvalidInputError):
                pelt_weighted_l2(valid, bad_pen, 2, 1)


if __name__ == "__main__":
    unittest.main()
