"""Unit tests for ccnv.features subpackage."""

from __future__ import annotations

import unittest

import numpy as np
from helpers import make_sample_table

from ccnv.features.binning import (
    bin_counts,
    bin_intervals,
    rebin_counts,
)
from ccnv.features.cohort import (
    Cohort,
    infer_sample_sexes,
    mask_low_coverage,
    normalize_depth,
    rebin_sample_table,
    validate_cohort,
)
from ccnv.features.covariates import (
    DEFAULT_MIN_MAPPABILITY,
    correct_covariate_bias,
    covariate_features,
    mask_covariates,
    validate_min_mappability,
)
from ccnv.features.prenatal import (
    annotate_prenatal_segments,
    classify_sca,
)
from ccnv.models import (
    CovariateError,
    FetalFractionResult,
    PrenatalModelMetadata,
    ReferenceError,
    SCAResult,
    SegmentResult,
)


class TestFeaturesBinning(unittest.TestCase):
    def test_bin_intervals_exact_and_partial(self) -> None:
        starts, ends = bin_intervals(250, 100)
        np.testing.assert_array_equal(starts, [0, 100, 200])
        np.testing.assert_array_equal(ends, [100, 200, 250])

    def test_bin_intervals_invalid(self) -> None:
        with self.assertRaises(ValueError):
            bin_intervals(0, 100)
        with self.assertRaises(ValueError):
            bin_intervals(100, 0)

    def test_bin_counts(self) -> None:
        starts = np.array([5, 12, 105, 199, 250], dtype=np.int64)
        counts = bin_counts(starts, bin_size=100, n_bins=3)
        np.testing.assert_array_equal(counts, [2, 2, 1])

    def test_bin_counts_empty_and_invalid(self) -> None:
        counts = bin_counts(np.empty(0, dtype=np.int64), bin_size=100, n_bins=3)
        np.testing.assert_array_equal(counts, [0, 0, 0])
        with self.assertRaises(ValueError):
            bin_counts(np.array([1]), 100, 0)

    def test_rebin_counts_coarsening(self) -> None:
        src_s = np.array([0, 50, 100, 150], dtype=np.int64)
        src_e = np.array([50, 100, 150, 200], dtype=np.int64)
        src_c = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)

        tgt_s = np.array([0, 100], dtype=np.int64)
        tgt_e = np.array([100, 200], dtype=np.int64)

        res = rebin_counts(src_s, src_e, src_c, tgt_s, tgt_e)
        np.testing.assert_allclose(res, [30.0, 70.0])

    def test_rebin_counts_refining(self) -> None:
        src_s = np.array([0, 100], dtype=np.int64)
        src_e = np.array([100, 200], dtype=np.int64)
        src_c = np.array([40.0, 60.0], dtype=np.float64)

        tgt_s = np.array([0, 50, 100, 150], dtype=np.int64)
        tgt_e = np.array([50, 100, 150, 200], dtype=np.int64)

        res = rebin_counts(src_s, src_e, src_c, tgt_s, tgt_e)
        np.testing.assert_allclose(res, [20.0, 20.0, 30.0, 30.0])

    def test_rebin_counts_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            rebin_counts(
                np.array([0], dtype=np.int64),
                np.array([100], dtype=np.int64),
                np.array([1.0]),
                np.array([0], dtype=np.int64),
                np.array([150], dtype=np.int64),
            )


class TestFeaturesCohort(unittest.TestCase):
    def test_normalize_depth(self) -> None:
        counts = np.array([[10.0, 20.0], [30.0, 60.0]], dtype=np.float64)
        norm = normalize_depth(counts)
        np.testing.assert_allclose(norm[:, 0], [0.25, 0.75])
        np.testing.assert_allclose(norm[:, 1], [0.25, 0.75])

    def test_normalize_depth_zero_sum_fails(self) -> None:
        counts = np.array([[0.0, 10.0], [0.0, 20.0]])
        with self.assertRaises(ReferenceError):
            normalize_depth(counts)

    def test_mask_low_coverage(self) -> None:
        norm = np.array([[0.0], [0.01], [1.0], [1.1], [1.0]], dtype=np.float64)
        mask = mask_low_coverage(norm)
        self.assertFalse(mask[0])
        self.assertFalse(mask[1])
        self.assertTrue(mask[2])
        self.assertTrue(mask[3])
        self.assertTrue(mask[4])

    def test_infer_sample_sexes(self) -> None:
        fractions = np.array([0.001, 0.0012, 0.0009, 0.02, 0.021, 0.019])
        result = infer_sample_sexes(fractions, min_group_size=2)
        self.assertTrue(result.passed)
        self.assertEqual(
            result.labels, ("female", "female", "female", "male", "male", "male")
        )
        self.assertIsNotNone(result.cutoff)

    def test_infer_sample_sexes_cutoff(self) -> None:
        fractions = np.array([0.005, 0.025])
        result = infer_sample_sexes(fractions, min_group_size=1, cutoff=0.01)
        self.assertTrue(result.passed)
        self.assertEqual(result.labels, ("female", "male"))

    def test_validate_cohort_and_rebin(self) -> None:
        t1 = make_sample_table([10, 20], contig_lengths={"chr1": 200}, bin_size=100)
        t2 = make_sample_table(
            [5, 5, 10, 10], contig_lengths={"chr1": 200}, bin_size=50
        )
        cohort = validate_cohort((t1, t2), target_bin_size=100)
        self.assertIsInstance(cohort, Cohort)
        self.assertEqual(cohort.counts.shape, (2, 2))
        np.testing.assert_allclose(cohort.counts[:, 0], [10.0, 20.0])
        np.testing.assert_allclose(cohort.counts[:, 1], [10.0, 20.0])

    def test_rebin_sample_table(self) -> None:
        t = make_sample_table([10, 20], contig_lengths={"chr1": 200}, bin_size=100)
        rebinned = rebin_sample_table(t, target_bin_size=50)
        self.assertEqual(rebinned.num_rows, 4)
        np.testing.assert_allclose(
            rebinned.column("count").to_numpy(), [5.0, 5.0, 10.0, 10.0]
        )


class TestFeaturesCovariates(unittest.TestCase):
    def test_validate_min_mappability(self) -> None:
        validate_min_mappability(0.5)
        validate_min_mappability(0.0)
        validate_min_mappability(1.0)
        with self.assertRaises(CovariateError):
            validate_min_mappability(-0.1)
        with self.assertRaises(CovariateError):
            validate_min_mappability(1.1)
        with self.assertRaises(CovariateError):
            validate_min_mappability(True)

    def test_mask_covariates(self) -> None:
        usable = np.array([True, True, True, True])
        gc = np.array([0.4, 0.5, float("nan"), 0.6])
        mappability = np.array([0.9, 0.3, 0.9, 0.8])
        n_mask = np.array([False, False, False, True])

        masked = mask_covariates(
            usable,
            gc=gc,
            mappability=mappability,
            n_mask=n_mask,
            min_mappability=DEFAULT_MIN_MAPPABILITY,
        )
        self.assertTrue(masked[0])
        self.assertFalse(masked[1])
        self.assertFalse(masked[2])
        self.assertFalse(masked[3])

    def test_covariate_features(self) -> None:
        usable = np.array([True, True, True, True, False])
        gc = np.array([0.3, 0.4, 0.5, 0.6, 0.9])
        mappability = np.array([0.8, 0.85, 0.9, 0.95, 0.5])

        features = covariate_features(usable, gc=gc, mappability=mappability)
        self.assertIsNotNone(features)
        assert features is not None
        self.assertEqual(features.shape, (4, 2))
        np.testing.assert_allclose(np.mean(features, axis=0), [0.0, 0.0], atol=1e-10)

    def test_correct_covariate_bias(self) -> None:
        usable = np.ones(10, dtype=np.bool_)
        gc = np.linspace(0.3, 0.7, 10)
        counts = 100.0 * (1.0 + 2.0 * gc)
        corrected = correct_covariate_bias(counts, usable, gc=gc)
        self.assertEqual(corrected.shape, (10,))
        self.assertLess(np.std(corrected), np.std(counts))


class TestFeaturesPrenatal(unittest.TestCase):
    def setUp(self) -> None:
        self.meta = PrenatalModelMetadata(
            prenatal_bin_size=100_000,
            yfrac_threshold=0.002,
            ffy_intercept=0.0,
            ffy_slope=1.0,
            mlp_w1=[[0.1, 0.2]],
            mlp_b1=[0.0, 0.0],
            mlp_w2=[[0.3], [0.4]],
            mlp_b2=[0.0],
            mlp_skew_intercept=0.0,
            mlp_skew_slope=1.0,
            olm_weights=[0.1],
            olm_bias=0.0,
            olm_skew_intercept=0.0,
            olm_skew_slope=1.0,
            ffx_intercept=2.0,
            ffx_slope=-2.0,
            female_chrx_ratio_mean=1.0,
            female_chrx_ratio_std=0.02,
        )

    def test_classify_sca_female(self) -> None:
        res_xx = classify_sca(
            "F", None, ffx=0.0, ffa=0.1, chrx_ratio=1.0, prenatal_meta=self.meta
        )
        self.assertEqual(res_xx.call, "XX")
        self.assertFalse(res_xx.is_aneuploid)

        res_xo = classify_sca(
            "F", None, ffx=0.2, ffa=0.1, chrx_ratio=0.9, prenatal_meta=self.meta
        )
        self.assertEqual(res_xo.call, "XO")
        self.assertTrue(res_xo.is_aneuploid)

        res_xxx = classify_sca(
            "F", None, ffx=-0.2, ffa=0.1, chrx_ratio=1.1, prenatal_meta=self.meta
        )
        self.assertEqual(res_xxx.call, "XXX")
        self.assertTrue(res_xxx.is_aneuploid)

    def test_classify_sca_male(self) -> None:
        res_xy = classify_sca(
            "M", ffy=0.1, ffx=0.1, ffa=0.1, chrx_ratio=0.95, prenatal_meta=self.meta
        )
        self.assertEqual(res_xy.call, "XY")
        self.assertFalse(res_xy.is_aneuploid)

        res_xxy = classify_sca(
            "M", ffy=0.1, ffx=-0.1, ffa=0.1, chrx_ratio=1.0, prenatal_meta=self.meta
        )
        self.assertEqual(res_xxy.call, "XXY")
        self.assertTrue(res_xxy.is_aneuploid)

    def test_annotate_prenatal_segments(self) -> None:
        sca = SCAResult("XY", False, 0.9, 0.0, 0.0)
        ff_res = FetalFractionResult(
            fetal_sex="M",
            active_ff=0.10,
            active_ff_source="FFY",
            ffy=0.10,
            ffx=0.10,
            ffa=0.10,
            chrx_ratio=0.95,
            chry_score=0.01,
            yfrac_threshold=0.002,
            sca=sca,
            low_ff_warning=False,
            warnings=(),
        )

        segments = [
            SegmentResult(
                contig="chr1",
                start=0,
                end=1000,
                ratio=0.07,
                zscore=4.0,
                weight=10.0,
                bin_count=10,
                call="gain",
            ),
            SegmentResult(
                contig="chr2",
                start=0,
                end=1000,
                ratio=0.01,
                zscore=3.5,
                weight=10.0,
                bin_count=10,
                call="gain",
            ),
        ]

        annotated = annotate_prenatal_segments(segments, ff_res)
        self.assertEqual(len(annotated), 2)
        self.assertFalse(annotated[0].discordant)
        self.assertTrue(annotated[1].discordant)
        self.assertEqual(annotated[1].discordance_reason, "SUB_FETAL_RATIO")


if __name__ == "__main__":
    unittest.main()
