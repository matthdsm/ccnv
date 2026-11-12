"""Unit tests for ccnv.features subpackage."""

from __future__ import annotations

import unittest

import numpy as np
from helpers import make_sample_table

from ccnv.features.binning import (
    bin_intervals,
    compute_bin_exposure,
    rebin_counts,
)
from ccnv.features.cohort import (
    Cohort,
    estimate_superblock_size_factors,
    infer_sample_sexes,
    mask_low_coverage_per_contig,
    validate_cohort,
)
from ccnv.features.covariates import (
    DEFAULT_MIN_MAPPABILITY,
    build_exogenous_covariate_basis,
    covariate_features,
    fit_poisson_covariate_trends,
    mask_covariates,
    validate_min_mappability,
)
from ccnv.features.prenatal_origin import annotate_prenatal_origin_and_lod
from ccnv.features.rebinning import rebin_sample_buffers, rebin_sample_sketch
from ccnv.models import (
    BinCompatibilityError,
    CovariateError,
    ReferenceError,
    SampleBuffers,
    SegmentResult,
)
from ccnv.stats.prenatal_math import classify_sca_ray_4d


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

    def test_compute_bin_exposure(self) -> None:
        starts = np.array([0, 100, 200], dtype=np.int64)
        ends = np.array([100, 200, 250], dtype=np.int64)
        exp = compute_bin_exposure(starts, ends, 100)
        np.testing.assert_allclose(exp, [1.0, 1.0, 0.5])

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
    def test_superblock_size_factors(self) -> None:
        counts = np.array([[10.0, 20.0], [30.0, 60.0]], dtype=np.float32)
        contigs = ("chr1", "chr1")
        starts = np.array([0, 100], dtype=np.int64)
        ends = np.array([100, 200], dtype=np.int64)
        usable = np.ones(2, dtype=np.bool_)
        sf = estimate_superblock_size_factors(
            counts, contigs, starts, ends, usable, block_bp=100
        )
        np.testing.assert_allclose(sf[1] / sf[0], 2.0, rtol=1e-6)

    def test_superblock_size_factors_zero_sum_fails(self) -> None:
        counts = np.array([[0.0, 10.0], [0.0, 20.0]], dtype=np.float32)
        with self.assertRaises(ReferenceError):
            estimate_superblock_size_factors(
                counts,
                ("chr1", "chr1"),
                np.array([0, 100], dtype=np.int64),
                np.array([100, 200], dtype=np.int64),
                np.ones(2, dtype=np.bool_),
                block_bp=100,
            )

    def test_mask_low_coverage_per_contig(self) -> None:
        counts = np.array([[0.0], [0.01], [100.0], [110.0], [100.0]], dtype=np.float32)
        exp = np.ones(5, dtype=np.float32)
        mask = mask_low_coverage_per_contig(counts, ("chr1",) * 5, exp)
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

    def test_rebin_sample_sketch_and_buffers(self) -> None:
        t = make_sample_table([5, 5, 10, 10], contig_lengths={"chr1": 200}, bin_size=50)
        rebinned = rebin_sample_sketch(t, target_bin_size=100)
        self.assertEqual(rebinned.num_rows, 2)
        np.testing.assert_allclose(rebinned.column("count").to_numpy(), [10, 20])
        bufs = rebin_sample_buffers(SampleBuffers.from_table(t), 100)
        np.testing.assert_allclose(bufs.count, [10, 20])
        with self.assertRaises(BinCompatibilityError):
            rebin_sample_sketch(t, target_bin_size=25)


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

    def test_covariate_features_and_poisson_trend(self) -> None:
        usable = np.array([True, True, True, True, False])
        gc = np.array([0.3, 0.4, 0.5, 0.6, 0.9])
        mappability = np.array([0.8, 0.85, 0.9, 0.95, 0.5])

        features = covariate_features(usable, gc=gc, mappability=mappability)
        self.assertIsNotNone(features)
        assert features is not None
        self.assertEqual(features.shape, (4, 2))
        np.testing.assert_allclose(np.mean(features, axis=0), [0.0, 0.0], atol=1e-10)

        usable_10 = np.ones(10, dtype=np.bool_)
        gc_10 = np.linspace(0.3, 0.7, 10)
        counts_10 = (100.0 * (1.0 + 2.0 * gc_10)).astype(np.float64)
        design = build_exogenous_covariate_basis(usable_10, bin_size=5000, gc=gc_10)
        trends = fit_poisson_covariate_trends(
            counts_10,
            np.array([1.0], dtype=np.float64),
            np.ones(10, dtype=np.float64),
            usable_10,
            design,
        )[:, 0]
        self.assertEqual(trends.shape, (10,))
        self.assertLess(np.std(counts_10 / trends), np.std(counts_10))


class TestFeaturesPrenatal(unittest.TestCase):
    def test_classify_sca_ray_4d_female_and_male(self) -> None:
        fem_priors = {
            "XX": 0.90,
            "XO": 0.05,
            "XXX": 0.05,
            "XY": 1e-6,
            "XXY": 1e-6,
            "XYY": 1e-6,
        }
        male_priors = {
            "XY": 0.90,
            "XXY": 0.05,
            "XYY": 0.05,
            "XX": 1e-6,
            "XO": 1e-6,
            "XXX": 1e-6,
        }
        call_xx, *_ = classify_sca_ray_4d(
            delta_x=0.0,
            sigma_x=0.02,
            ffy=0.0,
            sigma_ffy=0.01,
            ffa=0.10,
            sigma_ffa=0.02,
            clinical_priors=fem_priors,
        )
        self.assertEqual(call_xx, "XX")

        call_xo, *_ = classify_sca_ray_4d(
            delta_x=-0.05,
            sigma_x=0.01,
            ffy=0.0,
            sigma_ffy=0.01,
            ffa=0.10,
            sigma_ffa=0.02,
            clinical_priors=fem_priors,
        )
        self.assertEqual(call_xo, "XO")

        call_xxx, *_ = classify_sca_ray_4d(
            delta_x=0.05,
            sigma_x=0.01,
            ffy=0.0,
            sigma_ffy=0.01,
            ffa=0.10,
            sigma_ffa=0.02,
            clinical_priors=fem_priors,
        )
        self.assertEqual(call_xxx, "XXX")

        call_xy, *_ = classify_sca_ray_4d(
            delta_x=-0.05,
            sigma_x=0.02,
            ffy=0.10,
            sigma_ffy=0.01,
            ffa=0.10,
            sigma_ffa=0.02,
            clinical_priors=male_priors,
        )
        self.assertEqual(call_xy, "XY")

        call_xxy, *_ = classify_sca_ray_4d(
            delta_x=0.0,
            sigma_x=0.01,
            ffy=0.10,
            sigma_ffy=0.01,
            ffa=0.10,
            sigma_ffa=0.02,
            clinical_priors=male_priors,
        )
        self.assertEqual(call_xxy, "XXY")

    def test_annotate_prenatal_origin_and_lod(self) -> None:
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

        annotated, _ = annotate_prenatal_origin_and_lod(
            segments,
            active_ff=0.10,
            sca_call="XY",
        )
        self.assertEqual(len(annotated), 2)
        self.assertFalse(annotated[0].discordant)
        self.assertTrue(annotated[1].discordant)
        self.assertEqual(annotated[1].discordance_reason, "SUB_FETAL_RATIO")


if __name__ == "__main__":
    unittest.main()
