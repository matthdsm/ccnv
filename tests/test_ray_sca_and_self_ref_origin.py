"""Contract tests for Plan 3 Task 3:

- 4D erfc Ray-Marginalized SCA Classifier (XX, XY, XO, XXX, XXY, XYY, DISCORDANT_MIXED)
- Student-t4 IRLS Active Fetal Fraction Fusion
- 5-State Self-Referenced (p0, pF) Coupled Depth + Dirichlet-Multinomial Origin/Mosaicism Engine
- Fast O(log U) 24-Band Cumulative Prefix Sum Queries
- Analytical Limit-of-Detection (LoD) QC Metrics
"""

from __future__ import annotations

import math
import unittest

import numpy as np
from scipy.stats import norm

from ccnv._core import FRAG_BAND_EDGES
from ccnv.features.prenatal_origin import (
    BandPrefixSums,
    PrenatalLoDMetrics,
    annotate_prenatal_origin_and_lod,
)
from ccnv.models.results import SegmentResult
from ccnv.stats.fragmentomics import FRAG_MAX_LEN, FRAG_MIN_LEN
from ccnv.stats.prenatal_math import (
    classify_sca_ray_4d,
    fuse_active_fetal_fraction_student_t4,
)


class TestRaySCAAndSelfRefOrigin(unittest.TestCase):
    def setUp(self) -> None:
        self.lengths = np.arange(FRAG_MIN_LEN, FRAG_MAX_LEN + 1, dtype=np.float64)
        p_f = norm.pdf(self.lengths, loc=143.0, scale=10.0)
        p_f[self.lengths < 155] *= 1.0 + 0.15 * np.cos(
            2.0 * np.pi * (self.lengths[self.lengths < 155] - 143.0) / 10.4
        )
        p_f = np.maximum(p_f, 0.0)
        p_f /= np.sum(p_f)

        p_m = norm.pdf(self.lengths, loc=166.0, scale=12.0)
        p_m /= np.sum(p_m)

        edges = np.asarray(FRAG_BAND_EDGES, dtype=np.int64)
        n_bands = len(edges) - 1
        p_f_24 = np.zeros(n_bands, dtype=np.float64)
        p_m_24 = np.zeros(n_bands, dtype=np.float64)
        for k in range(n_bands):
            s = int(edges[k] - FRAG_MIN_LEN)
            e = int(edges[k + 1] - FRAG_MIN_LEN)
            p_f_24[k] = np.sum(p_f[s:e])
            p_m_24[k] = np.sum(p_m[s:e])
        p_f_24 = np.clip(p_f_24, 1e-8, None)
        self.p_f_24 = p_f_24 / np.sum(p_f_24)
        p_m_24 = np.clip(p_m_24, 1e-8, None)
        self.p_m_24 = p_m_24 / np.sum(p_m_24)

    def test_4d_ray_sca_classifies_xyy_without_ffa_and_xx_at_3pct_ff(self) -> None:
        """4D ray SCA must classify XYY when FFA is missing and classify XX accurately at 3% FF."""
        # XYY at true f=0.12 -> delta_x=-0.06, ffy=0.24, ffa=None, ff_size=0.12
        call_xyy, post_xyy, f_act, f_se, src_xyy = classify_sca_ray_4d(
            -0.06, 0.01, 0.24, 0.012, None, 0.03, 0.12, 0.02
        )
        self.assertEqual(call_xyy, "XYY")
        self.assertGreater(post_xyy["XYY"], 0.90)
        self.assertLess(abs(f_act - 0.12), 0.015)
        self.assertLess(f_se, 0.02)
        self.assertIn(src_xyy, ("FUSED", "FFY"))

        # Euploid XX at f=0.03 -> delta_x=0.0, ffy=0.0, ffa=0.03, ff_size=0.032
        call_xx, post_xx, f_xx, se_xx, src_xx = classify_sca_ray_4d(
            0.0, 0.008, 0.0, 0.004, 0.03, 0.025, 0.032, 0.02
        )
        self.assertEqual(call_xx, "XX")
        self.assertGreater(post_xx["XX"], 0.95)
        self.assertLess(abs(f_xx - 0.031), 0.005)
        self.assertLess(se_xx, 0.03)
        self.assertEqual(src_xx, "FUSED")

    def test_4d_ray_sca_classifies_xxy_xo_xxx_and_discordant_mixed(self) -> None:
        """4D ray SCA covers XXY, XO, XXX, and flags DISCORDANT_MIXED on outliers or maternal mosaicism."""
        # XXY at f=0.10 -> delta_x=0.0, ffy=0.10, ffa=0.10, ff_size=0.10
        call_xxy, post_xxy, f_xxy, _, _ = classify_sca_ray_4d(
            0.0, 0.008, 0.10, 0.008, 0.10, 0.015, 0.10, 0.015
        )
        self.assertEqual(call_xxy, "XXY")
        self.assertGreater(post_xxy["XXY"], 0.85)
        self.assertLess(abs(f_xxy - 0.10), 0.015)

        # XO at f=0.10 -> delta_x=-0.05, ffy=0.0, ffa=0.10, ff_size=0.10
        call_xo, post_xo, _, _, _ = classify_sca_ray_4d(
            -0.05, 0.008, 0.0, 0.004, 0.10, 0.015, 0.10, 0.015
        )
        self.assertEqual(call_xo, "XO")
        self.assertGreater(post_xo["XO"], 0.85)

        # Maternal 45,X mosaicism fragment gate override
        call_maternal_xo, _, _, _, _ = classify_sca_ray_4d(
            -0.05, 0.008, 0.0, 0.004, 0.10, 0.015, 0.10, 0.015, chrx_llr_fetal=-2.5
        )
        self.assertEqual(call_maternal_xo, "DISCORDANT_MIXED")

        # XXX at f=0.10 -> delta_x=+0.05, ffy=0.0, ffa=0.10, ff_size=0.10
        call_xxx, post_xxx, _, _, _ = classify_sca_ray_4d(
            0.05, 0.008, 0.0, 0.004, 0.10, 0.015, 0.10, 0.015
        )
        self.assertEqual(call_xxx, "XXX")
        self.assertGreater(post_xxx["XXX"], 0.85)

        # Impossible discordant combination: delta_x=+0.15 (Triple X) with ffy=0.20 (Male)
        call_disc, _, _, _, _ = classify_sca_ray_4d(
            0.15, 0.008, 0.20, 0.008, 0.08, 0.015, 0.08, 0.015
        )
        self.assertEqual(call_disc, "DISCORDANT_MIXED")

    def test_student_t4_fusion_robust_against_single_modality_corruption(self) -> None:
        """Student-t4 IRLS must downweight corrupted estimator without pulling active FF."""
        # 3 modalities agree at 0.08, one modality corrupted by maternal CNV at 0.35
        estimates = [0.08, 0.082, 0.079, 0.350]
        ses = [0.010, 0.012, 0.015, 0.015]
        f_fused, se_fused = fuse_active_fetal_fraction_student_t4(estimates, ses)
        self.assertLess(abs(f_fused - 0.08), 0.006)
        self.assertGreater(se_fused, 0.0)

    def test_band_prefix_sums_matches_direct_slice_at_o_log_u(self) -> None:
        """BandPrefixSums range query must return exact counts matching direct slice in O(log U)."""
        rng = np.random.default_rng(123)
        n_bins = 200
        sample_bands = rng.integers(10, 100, size=(n_bins, 24), dtype=np.uint32)
        bin_contigs = np.array([0] * 100 + [1] * 100, dtype=np.int16)
        contig_names = ["chr1", "chr2"]
        bin_starts = np.concatenate([np.arange(100) * 5000, np.arange(100) * 5000])
        bin_ends = bin_starts + 5000
        bin_usable = np.ones(n_bins, dtype=np.bool_)

        engine = BandPrefixSums(
            sample_bands, bin_contigs, contig_names, bin_starts, bin_ends, bin_usable
        )

        # Query chr1: [15000, 45000) -> bins 3 to 9 (start index 3, end index 9)
        k_query = engine.query("chr1", 15000, 45000)
        expected = np.sum(sample_bands[3:9], axis=0, dtype=np.float64)
        np.testing.assert_array_equal(k_query, expected)

    def test_5state_self_referenced_origin_distinguishes_fetal_t21_inherited_dup_cpm_and_maternal_mosaic(
        self,
    ) -> None:
        """5-state origin engine must distinguish constitutional, inherited, CPM, and maternal mosaicism."""
        active_ff = 0.08
        active_ff_se = 0.005
        p0 = active_ff * self.p_f_24 + (1.0 - active_ff) * self.p_m_24

        n_bins = 400
        bin_contigs = np.array(
            [0] * 100 + [1] * 100 + [2] * 100 + [3] * 100, dtype=np.int16
        )
        contig_names = ["chr1", "chr2", "chr3", "chr21"]
        bin_starts = np.tile(np.arange(100, dtype=np.int64) * 20000, 4)
        bin_ends = bin_starts + 20000
        bin_usable = np.ones(n_bins, dtype=np.bool_)
        sample_bands = np.zeros((n_bins, 24), dtype=np.uint32)

        # Diploid background across all bins: 500 fragments per bin
        for b in range(n_bins):
            sample_bands[b] = np.round(500 * p0).astype(np.uint32)

        # Segment 0 (chr21): Fetal Constitutional Trisomy 21 (R ~ 1.04, short fragment enriched)
        # Expected ratio: 1 + 0.5 * 0.08 = 1.04 -> log2(1.04) ~ 0.05658
        phi_fc = (1.5 * active_ff) / (1.0 - active_ff + 1.5 * active_ff)
        p_fc = phi_fc * self.p_f_24 + (1.0 - phi_fc) * self.p_m_24
        for b in range(300, 350):
            sample_bands[b] = np.round(520 * p_fc).astype(np.uint32)
        seg0 = SegmentResult(
            contig="chr21",
            start=0,
            end=1000000,
            ratio=math.log2(1.04),
            zscore=4.5,
            weight=50.0,
            bin_count=50,
            call="gain",
        )

        # Segment 1 (chr1): Maternal+Fetal Inherited 2 Mb Duplication (R = 1.50, bands == p0)
        # Both mother and fetus have 3 copies -> depth 1.50, sizing identical to diploid baseline!
        for b in range(10, 60):
            sample_bands[b] = np.round(750 * p0).astype(np.uint32)
        seg1 = SegmentResult(
            contig="chr1",
            start=200000,
            end=1200000,
            ratio=math.log2(1.50),
            zscore=12.0,
            weight=50.0,
            bin_count=50,
            call="gain",
        )

        # Segment 2 (chr2): 40% CPM gain (R = 1.016, short-fragment enriched)
        r_cpm = 1.0 + 0.5 * 0.40 * active_ff
        phi_cpm = (1.0 + 0.5 * 0.40) * active_ff / r_cpm
        p_cpm = phi_cpm * self.p_f_24 + (1.0 - phi_cpm) * self.p_m_24
        for b in range(110, 160):
            sample_bands[b] = np.round(508 * p_cpm).astype(np.uint32)
        seg2 = SegmentResult(
            contig="chr2",
            start=200000,
            end=1200000,
            ratio=math.log2(r_cpm),
            zscore=3.5,
            weight=50.0,
            bin_count=50,
            call="gain",
        )

        # Segment 3 (chr3): 4.4% Maternal Somatic Mosaic gain (R = 1.020, short-fragment depleted)
        r_mat = 1.0 + 0.5 * 0.044 * (1.0 - active_ff)
        phi_mat = active_ff / r_mat
        p_mat = phi_mat * self.p_f_24 + (1.0 - phi_mat) * self.p_m_24
        for b in range(210, 260):
            sample_bands[b] = np.round(510 * p_mat).astype(np.uint32)
        seg3 = SegmentResult(
            contig="chr3",
            start=200000,
            end=1200000,
            ratio=math.log2(r_mat),
            zscore=3.5,
            weight=50.0,
            bin_count=50,
            call="gain",
        )

        prefix_sums = BandPrefixSums(
            sample_bands, bin_contigs, contig_names, bin_starts, bin_ends, bin_usable
        )
        annotated, lod = annotate_prenatal_origin_and_lod(
            segments=(seg0, seg1, seg2, seg3),
            sample_bands=sample_bands,
            prefix_sums=prefix_sums,
            bin_usable=bin_usable,
            p_fetal_24=self.p_f_24,
            mu_fetal_w=1.0,
            mu_maternal_w=1.0,
            active_ff=active_ff,
            active_ff_se=active_ff_se,
            sca_call="XX",
            sample_depth_per_mb=25000.0,
            sys_var_per_mb=0.0001,
            p_maternal_24=self.p_m_24,
        )

        self.assertEqual(len(annotated), 4)
        self.assertEqual(annotated[0].origin_call, "FETAL_CONSTITUTIONAL")
        self.assertEqual(annotated[1].origin_call, "MATERNAL_AND_FETAL_INHERITED")
        self.assertEqual(annotated[2].origin_call, "PLACENTAL_MOSAIC")
        self.assertIsNotNone(annotated[2].mosaic_fraction)
        self.assertGreater(float(annotated[2].mosaic_fraction or 0.0), 0.25)
        self.assertLess(float(annotated[2].mosaic_fraction or 0.0), 0.55)
        self.assertEqual(annotated[3].origin_call, "MATERNAL_MOSAIC")

        self.assertIsInstance(lod, PrenatalLoDMetrics)
        self.assertGreater(lod.lod_ff_35mb, 0.0)
        self.assertLess(lod.lod_ff_35mb, lod.lod_ff_3mb)
        self.assertLess(lod.lod_ff_3mb, 0.50)


if __name__ == "__main__":
    unittest.main()
