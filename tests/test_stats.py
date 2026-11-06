"""Comprehensive unit tests for ccnv.stats subpackage."""

from __future__ import annotations

import unittest

import numpy as np

import ccnv.stats as stats
from ccnv.models.exceptions import ReferenceError
from ccnv.stats.neighbors import (
    compute_null_models,
    distance_cutoff,
    select_nearest_bins,
)
from ccnv.stats.pca import (
    fit_pca,
    pca_irlba_numpy,
)
from ccnv.stats.prenatal_math import (
    fit_lasso_fista,
    fit_rprop_mlp,
    infer_sex_gmm_1d,
    ols_fit,
    rlm_huber,
    train_olm,
)
from ccnv.stats.regression import (
    apply_1d_correction,
    build_1d_basis,
    fit_sample_trend,
)
from ccnv.stats.segmentation import (
    Segment,
    WeightedL2Cost,
    segment_contig,
    segment_genome,
)


class TestStatsPackageExports(unittest.TestCase):
    def test_all_symbols_exported(self) -> None:
        expected_symbols = [
            "DEFAULT_MAX_BLACKLIST_GAP",
            "Algorithm",
            "Segment",
            "WeightedL2Cost",
            "apply_1d_correction",
            "build_1d_basis",
            "compute_null_models",
            "distance_cutoff",
            "fit_lasso_fista",
            "fit_pca",
            "fit_rprop_mlp",
            "fit_sample_trend",
            "infer_sex_gmm_1d",
            "ols_fit",
            "pca_irlba_numpy",
            "rlm_huber",
            "segment_contig",
            "segment_genome",
            "select_nearest_bins",
            "train_olm",
        ]
        for symbol in expected_symbols:
            self.assertTrue(hasattr(stats, symbol), f"Symbol {symbol} missing in stats")
            self.assertIn(symbol, stats.__all__)


class TestPCA(unittest.TestCase):
    def test_fit_pca_reconstruction_and_correction(self) -> None:
        rng = np.random.default_rng(42)
        n_bins, n_samples = 50, 10
        # Rank-2 signal + small noise
        u = rng.normal(size=(n_bins, 2))
        v = rng.normal(size=(2, n_samples))
        raw = np.abs(u @ v) + 1.0

        corrected, mean, loadings = fit_pca(raw, n_components=2)
        self.assertEqual(corrected.shape, (n_bins, n_samples))
        self.assertEqual(mean.shape, (n_bins,))
        self.assertEqual(loadings.shape, (2, n_bins))
        self.assertTrue(np.all(np.isfinite(corrected)))

    def test_fit_pca_constant_data(self) -> None:
        data = np.ones((10, 5), dtype=np.float64) * 2.0
        corrected, mean, loadings = fit_pca(data, n_components=2)
        np.testing.assert_allclose(corrected, 1.0)
        np.testing.assert_allclose(mean, 2.0)
        np.testing.assert_allclose(loadings, 0.0)

    def test_fit_pca_invalid_shapes(self) -> None:
        with self.assertRaises(ReferenceError):
            fit_pca(np.empty((0, 0)), n_components=1)
        with self.assertRaises(ReferenceError):
            fit_pca(np.ones(10), n_components=1)
        with self.assertRaises(ReferenceError):
            fit_pca(np.ones((10, 5)), n_components=0)

    def test_pca_irlba_numpy_matches_expected_shapes(self) -> None:
        rng = np.random.default_rng(123)
        X = rng.normal(size=(20, 15))
        scores, center, rotation = pca_irlba_numpy(X, n_components=3, seed=42)
        self.assertEqual(scores.shape, (20, 3))
        self.assertEqual(center.shape, (15,))
        self.assertEqual(rotation.shape, (15, 3))

    def test_pca_irlba_numpy_empty_and_zero(self) -> None:
        empty = np.empty((0, 0), dtype=np.float64)
        scores, _center, _rotation = pca_irlba_numpy(empty, n_components=2)
        self.assertEqual(scores.shape, (0, 2))

        zeros = np.zeros((10, 5), dtype=np.float64)
        scores, _center, _rotation = pca_irlba_numpy(zeros, n_components=2)
        self.assertEqual(scores.shape, (10, 2))
        np.testing.assert_allclose(scores, 0.0)


class TestNeighbors(unittest.TestCase):
    def test_select_nearest_bins_basic(self) -> None:
        profiles = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.1],
                [0.1, 1.0],
            ]
        )
        contig_ids = np.array([0, 0, 1, 1])
        indices, _distances = select_nearest_bins(
            profiles, contig_ids, k=1, max_batch_memory_mb=1
        )
        self.assertEqual(indices[0], [2])
        self.assertEqual(indices[1], [3])
        self.assertEqual(indices[2], [0])
        self.assertEqual(indices[3], [1])

    def test_select_nearest_bins_empty_and_errors(self) -> None:
        indices, distances = select_nearest_bins(
            np.empty((0, 2)), np.array([], dtype=int), k=1, max_batch_memory_mb=1
        )
        self.assertEqual(indices, [])
        self.assertEqual(distances, [])

        with self.assertRaisesRegex(ValueError, "positive integer"):
            select_nearest_bins(
                np.ones((4, 2)), np.array([0, 0, 1, 1]), k=0, max_batch_memory_mb=1
            )

    def test_compute_null_models_basic(self) -> None:
        corrected = np.ones((5, 10), dtype=np.float64)
        nearest_indices = [[1, 2], [0, 2], [0, 1], [0, 1], [0, 1]]
        null_mean, null_std = compute_null_models(corrected, nearest_indices)
        self.assertEqual(null_mean.shape, (5,))
        self.assertEqual(null_std.shape, (5,))
        np.testing.assert_allclose(null_mean, 0.0)
        np.testing.assert_allclose(null_std, 0.0)

    def test_distance_cutoff(self) -> None:
        distances = [[1.0, 2.0], [2.0, 3.0], [1.5, 2.5]]
        cutoff = distance_cutoff(distances, repeats=3)
        self.assertTrue(np.isfinite(cutoff))
        self.assertGreater(cutoff, 3.0)

        # Empty / non-finite
        self.assertEqual(distance_cutoff([]), float("inf"))
        self.assertEqual(distance_cutoff([[float("nan")]]), float("inf"))


class TestSegmentation(unittest.TestCase):
    def test_weighted_l2_cost(self) -> None:
        values = np.array([1.0, 2.0, 10.0])
        weights = np.array([1.0, 3.0, 0.0])
        cost = WeightedL2Cost().fit(values, weights)
        expected = np.sum(weights[:2] * np.square(values[:2] - 1.75))
        self.assertAlmostEqual(cost.error(0, 3), float(expected))

    def test_segment_contig_and_genome(self) -> None:
        starts = [0, 100, 200, 300, 400]
        ends = [100, 200, 300, 400, 500]
        ratios = [1.0, 1.0, 2.0, 2.0, 2.0]
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]

        segments = segment_contig(
            contig="chr1",
            starts=starts,
            ends=ends,
            ratios=ratios,
            weights=weights,
            contig_length=500,
            penalty=0.5,
            min_size=2,
        )
        self.assertGreaterEqual(len(segments), 1)
        self.assertIsInstance(segments[0], Segment)

        genome_segments = segment_genome(
            {"chr1": (starts, ends, ratios, weights, 500)},
            penalty=0.5,
            min_size=2,
        )
        self.assertEqual(len(genome_segments), len(segments))


class TestRegression(unittest.TestCase):
    def test_build_1d_basis(self) -> None:
        x = np.linspace(0.1, 0.9, 50)
        basis = build_1d_basis(x, degree=2)
        self.assertIsNotNone(basis)
        assert basis is not None
        self.assertEqual(basis.shape, (50, 3))

        # Constant array -> None
        const_x = np.ones(50) * 0.5
        self.assertIsNone(build_1d_basis(const_x, degree=2))

    def test_fit_sample_trend_and_apply_1d_correction(self) -> None:
        x = np.linspace(0.2, 0.8, 100)
        basis = build_1d_basis(x, degree=2)
        assert basis is not None

        trend_true = 1.0 + 0.5 * (x - 0.5)
        raw_cov = 100.0 * trend_true
        fitted_trend = fit_sample_trend(basis, raw_cov)
        self.assertEqual(fitted_trend.shape, (100,))
        self.assertTrue(np.all(np.isfinite(fitted_trend)))

        corrected = apply_1d_correction(raw_cov, basis)
        self.assertAlmostEqual(
            float(np.sum(corrected)), float(np.sum(raw_cov)), places=4
        )


class TestPrenatalMath(unittest.TestCase):
    def test_infer_sex_gmm_1d(self) -> None:
        y_females = np.array([0.0001, 0.0002, 0.00015, 0.0003])
        y_males = np.array([0.015, 0.02, 0.018, 0.022])
        combined = np.concatenate([y_females, y_males])
        is_male, threshold = infer_sex_gmm_1d(combined)
        self.assertEqual(is_male.shape, (8,))
        self.assertTrue(np.all(~is_male[:4]))
        self.assertTrue(np.all(is_male[4:]))
        self.assertGreater(threshold, 0.0005)
        self.assertLess(threshold, 0.01)

    def test_fit_lasso_fista(self) -> None:
        rng = np.random.default_rng(42)
        X = np.abs(rng.normal(size=(20, 5)))
        true_w = np.array([1.0, 0.0, 2.0, 0.0, 0.5])
        y = X @ true_w
        w = fit_lasso_fista(X, y, alpha=0.01)
        self.assertEqual(w.shape, (5,))
        self.assertTrue(np.all(w >= 0.0))

    def test_fit_rprop_mlp(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.normal(size=(20, 4))
        y = rng.uniform(0.01, 0.25, size=20)
        w1, b1, w2, b2 = fit_rprop_mlp(X, y, hidden_dim=2, n_inits=2, max_epochs=50)
        self.assertEqual(w1.shape, (4, 2))
        self.assertEqual(b1.shape, (2,))
        self.assertEqual(w2.shape, (2, 1))
        self.assertEqual(b2.shape, (1,))

    def test_ols_and_rlm(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 * x + 1.0
        w, b = train_olm(x.reshape(-1, 1), y)
        self.assertAlmostEqual(float(w[0]), 2.0)
        self.assertAlmostEqual(b, 1.0)

        intercept, slope = ols_fit(x, y)
        self.assertAlmostEqual(intercept, 1.0)
        self.assertAlmostEqual(slope, 2.0)

        intercept_h, slope_h = rlm_huber(x, y)
        self.assertAlmostEqual(intercept_h, 1.0, places=4)
        self.assertAlmostEqual(slope_h, 2.0, places=4)


if __name__ == "__main__":
    unittest.main()
