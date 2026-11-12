"""Unit and contract tests for prenatal segment annotation, 5-state origin, and LoD."""

from __future__ import annotations

import math
import unittest

import numpy as np
from helpers import make_sample_table
from numpy.typing import NDArray
from scipy.stats import norm

from ccnv._core import FRAG_BAND_EDGES
from ccnv.features.prenatal_origin import (
    BandPrefixSums,
    PrenatalLoDMetrics,
    annotate_prenatal_origin_and_lod,
    annotate_sample_prenatal_segments,
    classify_segment_origin_5state,
)
from ccnv.models import (
    NUM_FRAG_BANDS,
    FetalFractionResult,
    PrenatalModelMetadata,
    SampleBuffers,
    SampleMetadata,
    SCAResult,
    SegmentResult,
)
from ccnv.stats.fragmentomics import FRAG_MAX_LEN, FRAG_MIN_LEN


def _annotate_prenatal_segments(
    segments: list[SegmentResult],
    prenatal_result: FetalFractionResult,
    *,
    z_threshold: float = 3.0,
) -> list[SegmentResult]:
    eff_ff = (
        prenatal_result.effective_ff
        if prenatal_result.effective_ff is not None
        else prenatal_result.active_ff
    )
    annotated, _ = annotate_prenatal_origin_and_lod(
        segments,
        active_ff=eff_ff,
        sca_call=prenatal_result.sca.call,
        z_threshold=z_threshold,
    )
    return list(annotated)


def _build_24band_densities() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    lengths = np.arange(FRAG_MIN_LEN, FRAG_MAX_LEN + 1, dtype=np.float64)
    p_f = norm.pdf(lengths, loc=143.0, scale=10.0)
    p_f[lengths < 155] *= 1.0 + 0.15 * np.cos(
        2.0 * np.pi * (lengths[lengths < 155] - 143.0) / 10.4
    )
    p_f = np.maximum(p_f, 0.0)
    p_f /= np.sum(p_f)

    p_m = norm.pdf(lengths, loc=166.0, scale=12.0)
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
    p_f_24 /= np.sum(p_f_24)
    p_m_24 = np.clip(p_m_24, 1e-8, None)
    p_m_24 /= np.sum(p_m_24)
    return p_f_24, p_m_24


class TestAnnotatePrenatalSegments(unittest.TestCase):
    def setUp(self) -> None:
        sca_concordant = SCAResult("XY", False, 0.9, 0.1, ff_delta=0.0)
        self.prenatal_result = FetalFractionResult(
            fetal_sex="M",
            active_ff=0.10,
            active_ff_source="FFY",
            ffy=0.10,
            ffx=0.10,
            ffa=0.10,
            chrx_ratio=0.95,
            chry_score=0.02,
            yfrac_threshold=0.005,
            sca=sca_concordant,
            low_ff_warning=False,
            effective_ff=0.10,
        )

    def test_concordant_gain_and_loss(self) -> None:
        expected_gain = math.log2(1.05)
        expected_loss = math.log2(0.95)
        segs = [
            SegmentResult("chr1", 0, 100000, expected_gain, 4.5, 10.0, 20, "gain"),
            SegmentResult("chr2", 0, 100000, expected_loss, -4.5, 10.0, 20, "loss"),
        ]
        annotated = _annotate_prenatal_segments(segs, self.prenatal_result)
        self.assertEqual(len(annotated), 2)
        self.assertAlmostEqual(
            annotated[0].expected_ratio or 0.0, expected_gain, places=5
        )
        self.assertAlmostEqual(annotated[0].implied_ff or 0.0, 0.10, places=5)
        self.assertFalse(annotated[0].discordant)
        self.assertIsNone(annotated[0].discordance_reason)

        self.assertAlmostEqual(
            annotated[1].expected_ratio or 0.0, expected_loss, places=5
        )
        self.assertAlmostEqual(annotated[1].implied_ff or 0.0, 0.10, places=5)
        self.assertFalse(annotated[1].discordant)
        self.assertIsNone(annotated[1].discordance_reason)

    def test_sub_fetal_ratio(self) -> None:
        small_ratio = 0.3 * math.log2(1.05)
        seg = SegmentResult("chr1", 0, 100000, small_ratio, 3.5, 10.0, 20, "gain")
        annotated = _annotate_prenatal_segments([seg], self.prenatal_result)
        self.assertTrue(annotated[0].discordant)
        self.assertEqual(annotated[0].discordance_reason, "SUB_FETAL_RATIO")

    def test_super_fetal_ratio(self) -> None:
        seg = SegmentResult("chr1", 0, 100000, math.log2(1.5), 8.0, 10.0, 20, "gain")
        annotated = _annotate_prenatal_segments([seg], self.prenatal_result)
        self.assertTrue(annotated[0].discordant)
        self.assertEqual(annotated[0].discordance_reason, "SUPER_FETAL_RATIO")

    def test_sca_discordant(self) -> None:
        sca_aneuploid = SCAResult("XO", True, 0.95, -4.0, ff_delta=None)
        prenatal_xo = FetalFractionResult(
            fetal_sex="F",
            active_ff=0.10,
            active_ff_source="FFA",
            ffy=None,
            ffx=0.10,
            ffa=0.10,
            chrx_ratio=0.95,
            chry_score=0.001,
            yfrac_threshold=0.005,
            sca=sca_aneuploid,
            low_ff_warning=False,
            effective_ff=0.10,
        )
        seg_x = SegmentResult("chrX", 0, 100000, -0.05, -2.0, 10.0, 20, None)
        annotated = _annotate_prenatal_segments([seg_x], prenatal_xo)
        self.assertTrue(annotated[0].discordant)
        self.assertEqual(annotated[0].discordance_reason, "SCA_DISCORDANT")

    def test_annotate_prenatal_origin_and_lod(self) -> None:
        segments = [
            SegmentResult("chr1", 0, 1000, 0.07, 4.0, 10.0, 10, "gain"),
            SegmentResult("chr2", 0, 1000, 0.01, 3.5, 10.0, 10, "gain"),
        ]
        annotated, _ = annotate_prenatal_origin_and_lod(
            segments, active_ff=0.10, sca_call="XY"
        )
        self.assertEqual(len(annotated), 2)
        self.assertFalse(annotated[0].discordant)
        self.assertTrue(annotated[1].discordant)
        self.assertEqual(annotated[1].discordance_reason, "SUB_FETAL_RATIO")

    def test_annotate_prenatal_segments_origin_fetal_and_maternal(self) -> None:
        grid_len = 571
        p_fet = np.zeros(grid_len, dtype=np.float64)
        p_mat = np.zeros(grid_len, dtype=np.float64)
        p_fet[113] = 1.0
        p_mat[136] = 1.0

        prenatal_meta = PrenatalModelMetadata(
            prenatal_bin_size=100000,
            yfrac_threshold=0.001,
            ffy_slope=1.0,
            olm_weights=[],
            olm_bias=0.0,
            female_chrx_ratio_mean=1.0,
            female_chrx_ratio_std=0.01,
            ffy_std=0.01,
            ffa_std=0.02,
            ff_size_std=0.018,
            p_fetal_density=[float(x) for x in p_fet],
            p_maternal_density=[float(x) for x in p_mat],
        )
        prenatal_res = FetalFractionResult(
            fetal_sex="M",
            active_ff=0.10,
            active_ff_source="FFY",
            ffy=0.10,
            ffx=0.10,
            ffa=0.10,
            chrx_ratio=0.95,
            chry_score=0.01,
            yfrac_threshold=0.001,
            sca=SCAResult("XY", False, 1.0, 0.0),
        )
        sample = SampleBuffers.from_table(
            make_sample_table(
                counts=[100, 300],
                contigs=["chr1", "chr1"],
                starts=[0, 10000],
                ends=[10000, 20000],
                bin_size=10000,
                fragment_lengths=[[143] * 100, [166] * 300],
            )
        )
        segs = [
            SegmentResult("chr1", 0, 10000, 0.07, 4.5, 1.0, 1, "gain"),
            SegmentResult("chr1", 10000, 20000, 0.53605, 60.0, 100.0, 1, "gain"),
        ]
        annotated, _lod = annotate_sample_prenatal_segments(
            segs, sample, prenatal_meta=prenatal_meta, prenatal_result=prenatal_res
        )
        self.assertEqual(len(annotated), 2)
        if annotated[0].frag_llr_size is None or annotated[1].frag_llr_size is None:
            self.fail("expected non-None frag_llr_size")
        self.assertGreater(annotated[0].frag_llr_size, 2.0)
        self.assertEqual(annotated[0].origin_call, "FETAL_CONSTITUTIONAL")
        self.assertLess(annotated[1].frag_llr_size, -2.0)
        self.assertEqual(annotated[1].origin_call, "MATERNAL_UNINHERITED")


class TestPrenatalSelfReferencedOriginEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.p_f_24, self.p_m_24 = _build_24band_densities()

    def test_band_prefix_sums_matches_direct_slice_at_o_log_u(self) -> None:
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
        k_query = engine.query("chr1", 15000, 45000)
        expected = np.sum(sample_bands[3:9], axis=0, dtype=np.float64)
        np.testing.assert_array_equal(k_query, expected)

    def test_5state_self_referenced_origin_distinguishes_fetal_t21_inherited_dup_cpm_and_maternal_mosaic(
        self,
    ) -> None:
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

        for b in range(n_bins):
            sample_bands[b] = np.round(500 * p0).astype(np.uint32)

        phi_fc = (1.5 * active_ff) / (1.0 - active_ff + 1.5 * active_ff)
        p_fc = phi_fc * self.p_f_24 + (1.0 - phi_fc) * self.p_m_24
        for b in range(300, 350):
            sample_bands[b] = np.round(520 * p_fc).astype(np.uint32)
        seg0 = SegmentResult(
            "chr21", 0, 1000000, math.log2(1.04), 4.5, 50.0, 50, "gain"
        )

        for b in range(10, 60):
            sample_bands[b] = np.round(750 * p0).astype(np.uint32)
        seg1 = SegmentResult(
            "chr1", 200000, 1200000, math.log2(1.50), 12.0, 50.0, 50, "gain"
        )

        r_cpm = 1.0 + 0.5 * 0.40 * active_ff
        phi_cpm = (1.0 + 0.5 * 0.40) * active_ff / r_cpm
        p_cpm = phi_cpm * self.p_f_24 + (1.0 - phi_cpm) * self.p_m_24
        for b in range(110, 160):
            sample_bands[b] = np.round(508 * p_cpm).astype(np.uint32)
        seg2 = SegmentResult(
            "chr2", 200000, 1200000, math.log2(r_cpm), 3.5, 50.0, 50, "gain"
        )

        r_mat = 1.0 + 0.5 * 0.044 * (1.0 - active_ff)
        phi_mat = active_ff / r_mat
        p_mat = phi_mat * self.p_f_24 + (1.0 - phi_mat) * self.p_m_24
        for b in range(210, 260):
            sample_bands[b] = np.round(510 * p_mat).astype(np.uint32)
        seg3 = SegmentResult(
            "chr3", 200000, 1200000, math.log2(r_mat), 3.5, 50.0, 50, "gain"
        )

        prefix_sums = BandPrefixSums(
            sample_bands, bin_contigs, contig_names, bin_starts, bin_ends, bin_usable
        )
        annotated, lod = annotate_prenatal_origin_and_lod(
            segments=(seg0, seg1, seg2, seg3),
            sample_bands=sample_bands,
            prefix_sums=prefix_sums,
            bin_usable=bin_usable,
            p_fetal_density=self.p_f_24,
            mu_fetal_w=1.0,
            mu_maternal_w=1.0,
            active_ff=active_ff,
            active_ff_se=active_ff_se,
            sca_call="XX",
            sample_depth_per_mb=25000.0,
            sys_var_per_mb=0.0001,
            p_maternal_density=self.p_m_24,
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

    def test_prenatal_5state_origin_and_mosaicism_classification_recovery(self) -> None:
        p_f_24, p_m_24 = _build_24band_densities()
        active_ff = 0.08
        active_ff_se = 0.005
        p0 = active_ff * p_f_24 + (1.0 - active_ff) * p_m_24

        n_bins = 400
        bin_contigs = np.array(
            [0] * 100 + [1] * 100 + [2] * 100 + [3] * 100, dtype=np.int16
        )
        contig_names = ["chr1", "chr2", "chr3", "chr21"]
        bin_starts = np.tile(np.arange(100, dtype=np.int64) * 20_000, 4)
        bin_ends = bin_starts + 20_000
        bin_usable = np.ones(n_bins, dtype=np.bool_)
        sample_bands = np.zeros((n_bins, 24), dtype=np.uint32)

        for b in range(n_bins):
            sample_bands[b] = np.round(500 * p0).astype(np.uint32)

        phi_fc = (1.5 * active_ff) / (1.0 - active_ff + 1.5 * active_ff)
        p_fc = phi_fc * p_f_24 + (1.0 - phi_fc) * p_m_24
        for b in range(300, 350):
            sample_bands[b] = np.round(520 * p_fc).astype(np.uint32)
        seg0 = SegmentResult(
            "chr21", 0, 1_000_000, math.log2(1.04), 4.5, 50.0, 50, "gain"
        )

        for b in range(10, 60):
            sample_bands[b] = np.round(750 * p0).astype(np.uint32)
        seg1 = SegmentResult(
            "chr1", 200_000, 1_200_000, math.log2(1.50), 12.0, 50.0, 50, "gain"
        )

        r_cpm = 1.0 + 0.5 * 0.40 * active_ff
        phi_cpm = (1.0 + 0.5 * 0.40) * active_ff / r_cpm
        p_cpm = phi_cpm * p_f_24 + (1.0 - phi_cpm) * p_m_24
        for b in range(110, 160):
            sample_bands[b] = np.round(508 * p_cpm).astype(np.uint32)
        seg2 = SegmentResult(
            "chr2", 200_000, 1_200_000, math.log2(r_cpm), 3.5, 50.0, 50, "gain"
        )

        r_mat = 1.0 + 0.5 * 0.044 * (1.0 - active_ff)
        phi_mat = active_ff / r_mat
        p_mat = phi_mat * p_f_24 + (1.0 - phi_mat) * p_m_24
        for b in range(210, 260):
            sample_bands[b] = np.round(510 * p_mat).astype(np.uint32)
        seg3 = SegmentResult(
            "chr3", 200_000, 1_200_000, math.log2(r_mat), 3.5, 50.0, 50, "gain"
        )

        prefix_sums = BandPrefixSums(
            sample_bands, bin_contigs, contig_names, bin_starts, bin_ends, bin_usable
        )
        annotated, lod = annotate_prenatal_origin_and_lod(
            segments=(seg0, seg1, seg2, seg3),
            sample_bands=sample_bands,
            prefix_sums=prefix_sums,
            bin_usable=bin_usable,
            p_fetal_density=p_f_24,
            mu_fetal_w=1.0,
            mu_maternal_w=1.0,
            active_ff=active_ff,
            active_ff_se=active_ff_se,
            sca_call="XX",
            sample_depth_per_mb=25_000.0,
            sys_var_per_mb=0.0001,
            p_maternal_density=p_m_24,
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

    def test_defect7_band_prefix_sums_masking_z_lod_and_depth_ff_decoupling(
        self,
    ) -> None:
        contig_lengths = {"chr1": 20_000}
        bands = np.zeros((4, NUM_FRAG_BANDS), dtype=np.uint32)
        bands[0, 3] = 50
        bands[1, 20] = 9_999
        bands[2, 3] = 50
        bands[3, 3] = 50
        counts = np.sum(bands, axis=1, dtype=np.uint32)
        meta = SampleMetadata(
            bin_size=5000,
            contig_lengths=contig_lengths,
            contig_order=("chr1",),
            source_identity={},
            reference_identity={},
            qc={"effective_reads_kept": 10_149},
            filters={"min_mapping_quality": 1},
        )
        z_u32 = np.zeros(4, dtype=np.uint32)
        bufs = SampleBuffers(
            meta=meta,
            count=counts,
            paired_count=counts,
            frag_len_sum=np.zeros(4, dtype=np.uint64),
            short_frag_count=z_u32,
            endpoint_count=z_u32,
            frag_gc_sum=np.zeros(4, dtype=np.float32),
            frag_bands=bands,
        )
        usable = np.array([True, False, True, True], dtype=np.bool_)
        ps = BandPrefixSums.from_sample_table(bufs, bin_usable=usable)
        if ps is None:
            self.fail("expected non-None BandPrefixSums")
        seg_counts = ps.query("chr1", 0, 20_000)
        self.assertEqual(int(seg_counts[20]), 0)
        self.assertEqual(int(seg_counts[3]), 150)

        _, lod_low = annotate_prenatal_origin_and_lod([], z_threshold=2.0)
        _, lod_high = annotate_prenatal_origin_and_lod([], z_threshold=5.0)
        self.assertGreater(lod_high.lod_ff_35mb, lod_low.lod_ff_35mb)
        self.assertGreater(lod_high.lod_ff_3mb, lod_low.lod_ff_3mb)

        p_f24 = np.zeros(NUM_FRAG_BANDS, dtype=np.float64)
        p_f24[3:7] = [0.20, 0.35, 0.30, 0.15]
        p_m24 = np.zeros(NUM_FRAG_BANDS, dtype=np.float64)
        p_m24[8:12] = [0.20, 0.35, 0.30, 0.15]
        active_ff = 0.10
        depth_ff = 0.22
        p_plasma_0 = active_ff * p_f24 + (1.0 - active_ff) * p_m24
        r_unweighted = 1.0 + 0.5 * active_ff
        p_trisomy = (1.5 * active_ff * p_f24 + (1.0 - active_ff) * p_m24) / r_unweighted
        k_seg = np.round(20_000.0 * p_trisomy).astype(np.float64)
        r_obs_weighted = 1.0 + 0.5 * depth_ff
        call, posteriors, _, _ = classify_segment_origin_5state(
            r_obs=r_obs_weighted,
            k_band_counts=k_seg,
            p_fetal_24=p_f24,
            p_maternal_24=p_m24,
            p_plasma_0=p_plasma_0,
            active_ff=active_ff,
            active_ff_se=0.01,
            is_gain=True,
            depth_ff=depth_ff,
        )
        self.assertEqual(call, "FETAL_CONSTITUTIONAL")
        self.assertGreater(posteriors["FETAL_CONSTITUTIONAL"], 0.85)

        seg = SegmentResult(
            "chr1", 0, 20_000, math.log2(r_obs_weighted), 8.0, 500.0, 4, "gain"
        )
        ps_trisomy = BandPrefixSums(
            sample_bands=np.tile((k_seg / 4.0).astype(np.uint32), (4, 1)),
            bin_contigs=np.zeros(4, dtype=np.int16),
            contig_names=("chr1",),
            bin_starts=np.array([0, 5000, 10000, 15000], dtype=np.int64),
            bin_ends=np.array([5000, 10000, 15000, 20000], dtype=np.int64),
        )
        mu_f_w = (depth_ff * (1.0 - active_ff)) / ((1.0 - depth_ff) * active_ff)
        annotated, _ = annotate_prenatal_origin_and_lod(
            [seg],
            prefix_sums=ps_trisomy,
            p_fetal_density=p_f24,
            p_maternal_density=p_m24,
            mu_fetal_w=mu_f_w,
            mu_maternal_w=1.0,
            active_ff=active_ff,
            active_ff_se=0.01,
        )
        self.assertEqual(annotated[0].origin_call, "FETAL_CONSTITUTIONAL")

    def test_prenatal_origin_short_frag_ratio_spans_under_150bp_and_empty_usable(
        self,
    ) -> None:
        band_138_150 = int(np.searchsorted(FRAG_BAND_EDGES, 143, side="right") - 1)
        band_150_162 = int(np.searchsorted(FRAG_BAND_EDGES, 155, side="right") - 1)
        self.assertEqual(band_138_150, 9)
        self.assertEqual(band_150_162, 10)

        bands = np.zeros((2, NUM_FRAG_BANDS), dtype=np.uint32)
        bands[0, band_138_150] = 60
        bands[0, band_150_162] = 40
        bands[1, 0] = 500
        ps = BandPrefixSums(
            bands,
            np.array([0, 0], dtype=np.int16),
            ["chr1"],
            np.array([0, 5000], dtype=np.int64),
            np.array([5000, 10000], dtype=np.int64),
            bin_usable=np.array([True, False]),
        )
        seg = SegmentResult("chr1", 0, 5000, 0.08, 4.0, 20.0, 1, "gain")
        annotated, _ = annotate_prenatal_origin_and_lod(
            [seg],
            prefix_sums=ps,
            sample_bands=bands,
            bin_usable=np.array([False, False]),
            active_ff=0.10,
        )
        self.assertAlmostEqual(float(annotated[0].frag_short_ratio or 0.0), 0.60)

    def test_prenatal_origin_skips_diploid_math_on_chry_segments(self) -> None:
        seg_y = SegmentResult("chrY", 0, 50_000, 3.2, 0.5, 20.0, 10, None)
        bands = np.tile(np.ones(NUM_FRAG_BANDS, dtype=np.uint32), (10, 1))
        prefix = BandPrefixSums(
            sample_bands=bands,
            bin_contigs=np.zeros(10, dtype=np.int16),
            contig_names=("chrY",),
            bin_starts=np.arange(10, dtype=np.int64) * 5000,
            bin_ends=(np.arange(10, dtype=np.int64) + 1) * 5000,
        )
        p_f = [1.0 / 571.0] * 571
        p_m = [1.0 / 571.0] * 571

        ann_xy, _ = annotate_prenatal_origin_and_lod(
            [seg_y],
            active_ff=0.10,
            sca_call="XY",
            prefix_sums=prefix,
            p_fetal_density=p_f,
            p_maternal_density=p_m,
        )
        self.assertIsNone(ann_xy[0].expected_ratio)
        self.assertIsNone(ann_xy[0].implied_ff)
        self.assertFalse(ann_xy[0].discordant)
        self.assertIsNone(ann_xy[0].origin_call)
        self.assertIsNotNone(ann_xy[0].frag_short_ratio)

        ann_xyy, _ = annotate_prenatal_origin_and_lod(
            [seg_y],
            active_ff=0.10,
            sca_call="XYY",
            prefix_sums=prefix,
            p_fetal_density=p_f,
            p_maternal_density=p_m,
        )
        self.assertTrue(ann_xyy[0].discordant)
        self.assertEqual(ann_xyy[0].discordance_reason, "SCA_DISCORDANT")

        seg_y_gain = SegmentResult("chrY", 0, 50_000, 2.5, 6.0, 20.0, 10, "gain")
        ann_xx_gain, _ = annotate_prenatal_origin_and_lod(
            [seg_y_gain],
            active_ff=0.10,
            sca_call="XX",
            prefix_sums=prefix,
            p_fetal_density=p_f,
            p_maternal_density=p_m,
        )
        self.assertTrue(ann_xx_gain[0].discordant)
        self.assertEqual(ann_xx_gain[0].discordance_reason, "SCA_DISCORDANT")
        self.assertIsNone(ann_xx_gain[0].origin_call)


if __name__ == "__main__":
    unittest.main()
