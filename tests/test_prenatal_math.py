from __future__ import annotations

import math
import unittest
import unittest.mock

import numpy as np

from ccnv.stats.pca import pca_irlba_numpy
from ccnv.stats.prenatal_math import (
    infer_sex_gmm_1d,
    ols_fit,
    rlm_huber,
)


class TestInferSexGMM1D(unittest.TestCase):
    def test_bimodal_separation(self) -> None:
        rng = np.random.default_rng(123)
        females = rng.normal(loc=0.001, scale=0.0002, size=50)
        males = rng.normal(loc=0.020, scale=0.002, size=50)
        yfracs = np.concatenate([females, males])

        is_male, threshold = infer_sex_gmm_1d(yfracs)

        self.assertGreater(threshold, 0.002)
        self.assertLess(threshold, 0.015)
        np.testing.assert_array_equal(is_male[:50], False)
        np.testing.assert_array_equal(is_male[50:], True)

    def test_empty_and_single_input(self) -> None:
        empty_flags, empty_thresh = infer_sex_gmm_1d(np.array([]))
        self.assertEqual(len(empty_flags), 0)
        self.assertEqual(empty_thresh, 0.0)

        single_flags, single_thresh = infer_sex_gmm_1d(np.array([0.05]))
        self.assertEqual(len(single_flags), 1)
        self.assertFalse(single_flags[0])
        self.assertEqual(single_thresh, 0.05)

    def test_unimodal_identical_values(self) -> None:
        yfracs = np.full(20, 0.01)
        is_male, threshold = infer_sex_gmm_1d(yfracs)
        self.assertEqual(threshold, 0.01)
        self.assertTrue(np.all(is_male))

    def test_deterministic(self) -> None:
        rng = np.random.default_rng(42)
        yfracs = rng.uniform(0.0, 0.05, size=40)
        flags1, thresh1 = infer_sex_gmm_1d(yfracs)
        flags2, thresh2 = infer_sex_gmm_1d(yfracs)
        self.assertEqual(thresh1, thresh2)
        np.testing.assert_array_equal(flags1, flags2)


class TestFitElasticNetFFY(unittest.TestCase):
    def test_ffy_elasticnet_retains_correlated_chry_bins_when_p_gt_n(self) -> None:
        from ccnv.stats.prenatal_math import fit_elasticnet_ffy

        rng = np.random.default_rng(42)
        # S = 20 samples, B_Y = 200 correlated euchromatic chrY bins
        true_ff = np.concatenate([np.zeros(10), rng.uniform(0.04, 0.25, size=10)])
        X = np.outer(true_ff, np.ones(200)) + rng.normal(0.0, 0.005, size=(20, 200))
        X = np.clip(X, 0.0, None)

        weights, intercept = fit_elasticnet_ffy(X, true_ff, seed=42)
        self.assertTrue(np.all(weights >= 0.0))
        # Pure Lasso caps non-zeros at <= 20; ElasticNet with l1_ratio <= 0.1 retains > 80 bins
        self.assertGreater(int(np.count_nonzero(weights > 0)), 80)
        preds = intercept + X @ weights
        np.testing.assert_allclose(preds[10:], true_ff[10:], atol=0.02)

    def test_empty_or_zero_input(self) -> None:
        from ccnv.stats.prenatal_math import fit_elasticnet_ffy

        w, b = fit_elasticnet_ffy(np.empty((0, 5)), np.empty(0))
        self.assertEqual(w.shape, (5,))
        self.assertEqual(b, 0.0)


class TestFitFFALogitEnsemble(unittest.TestCase):
    def test_recovers_smooth_bounded_fetal_fraction(self) -> None:
        from ccnv.stats.prenatal_math import (
            fit_ffa_logit_ensemble,
            fit_preface_logit_ensemble,
        )

        rng = np.random.default_rng(42)
        n_samples = 30
        n_features = 10
        X = rng.standard_normal((n_samples, n_features))
        true_beta = rng.uniform(-0.5, 0.5, size=n_features)
        true_logits = X @ true_beta - 2.0
        from scipy.special import expit

        true_ff = expit(true_logits)

        weights, bias, sigma = fit_ffa_logit_ensemble(X, true_ff, seed=42)
        self.assertEqual(weights.shape, (n_features,))
        self.assertTrue(math.isfinite(bias))
        self.assertGreater(sigma, 0.0)
        self.assertLess(sigma, 0.10)

        preds = expit(X @ weights + bias)
        self.assertTrue(np.all((preds > 0.0) & (preds < 1.0)))
        np.testing.assert_allclose(preds, true_ff, atol=0.05)

        # Test alias compatibility
        w_pref, b_pref, s_pref = fit_preface_logit_ensemble(X, true_ff, seed=42)
        np.testing.assert_array_equal(weights, w_pref)
        self.assertEqual(bias, b_pref)
        self.assertEqual(sigma, s_pref)


class TestPCAIrlbaNumpy(unittest.TestCase):
    def test_orthogonality_and_centering_svd_branch(self) -> None:
        rng = np.random.default_rng(42)
        # n_samples > n_features
        x = rng.normal(loc=5.0, scale=2.0, size=(60, 10))
        scores, center, rotation = pca_irlba_numpy(x, n_components=4, seed=42)

        self.assertEqual(scores.shape, (60, 4))
        self.assertEqual(rotation.shape, (10, 4))
        np.testing.assert_allclose(center, np.mean(x, axis=0), atol=1e-12)

        # Orthogonality: V^T V = I
        ident = rotation.T @ rotation
        np.testing.assert_allclose(ident, np.eye(4), atol=1e-7)

        # Descending variance of components
        variances = np.var(scores, axis=0)
        self.assertTrue(np.all(np.diff(variances) <= 1e-12))

    def test_gram_matrix_branch_samples_le_features(self) -> None:
        rng = np.random.default_rng(101)
        # n_samples <= n_features
        x = rng.standard_normal(size=(8, 25))
        scores, center, rotation = pca_irlba_numpy(x, n_components=3, seed=101)

        self.assertEqual(scores.shape, (8, 3))
        self.assertEqual(rotation.shape, (25, 3))
        np.testing.assert_allclose(center, np.mean(x, axis=0), atol=1e-12)

        ident = rotation.T @ rotation
        np.testing.assert_allclose(ident, np.eye(3), atol=1e-7)

    def test_sign_canonicalization(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.standard_normal(size=(50, 12))
        _, _, rotation = pca_irlba_numpy(x, n_components=5, seed=7)

        for j in range(5):
            col = rotation[:, j]
            max_idx = int(np.argmax(np.abs(col)))
            self.assertGreaterEqual(col[max_idx], 0.0)

    def test_deterministic_seeding(self) -> None:
        rng = np.random.default_rng(1234)
        x = rng.standard_normal(size=(30, 8))
        sc1, c1, rot1 = pca_irlba_numpy(x, n_components=3, seed=999)
        sc2, c2, rot2 = pca_irlba_numpy(x, n_components=3, seed=999)

        np.testing.assert_array_equal(sc1, sc2)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(rot1, rot2)

    def test_components_exceeding_rank(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.standard_normal(size=(5, 10))
        scores, center, rotation = pca_irlba_numpy(x, n_components=8, seed=1)

        self.assertEqual(scores.shape, (5, 8))
        self.assertEqual(rotation.shape, (10, 8))
        np.testing.assert_allclose(center, np.mean(x, axis=0), atol=1e-12)

    def test_pca_irlba_numpy_all_nan_column_imputed_to_zero(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.normal(size=(15, 8))
        x[:, 3] = np.nan
        scores, center, rotation = pca_irlba_numpy(x, n_components=3, seed=42)
        self.assertEqual(scores.shape, (15, 3))
        self.assertEqual(center[3], 0.0)
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertTrue(np.all(np.isfinite(rotation)))


class TestOLSFit(unittest.TestCase):
    def test_recovers_exact_line(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.5 + 1.8 * x

        intercept, slope = ols_fit(x, y)
        self.assertAlmostEqual(intercept, 2.5, places=6)
        self.assertAlmostEqual(slope, 1.8, places=6)

    def test_zero_variance_fallback(self) -> None:
        x_const = np.array([3.0, 3.0, 3.0, 3.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])

        intercept, slope = ols_fit(x_const, y)
        self.assertEqual(intercept, 0.0)
        self.assertEqual(slope, 1.0)


class TestRLMHuber(unittest.TestCase):
    def test_outlier_robustness_compared_to_ols(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        x = np.linspace(-10.0, 10.0, n)
        true_intercept = 1.5
        true_slope = 2.0
        y = true_intercept + true_slope * x + rng.normal(0.0, 0.1, size=n)

        # Inject 10% severe outliers
        outlier_indices = rng.choice(n, size=10, replace=False)
        y[outlier_indices] += rng.choice([-1, 1], size=10) * 80.0

        ols_intercept, ols_slope = ols_fit(x, y)
        huber_intercept, huber_slope = rlm_huber(x, y)

        # Huber should remain close to the true slope
        self.assertLess(abs(huber_slope - true_slope), 0.15)
        self.assertLess(abs(huber_intercept - true_intercept), 0.5)

        # OLS error should be much larger than Huber error
        huber_slope_err = abs(huber_slope - true_slope)
        ols_slope_err = abs(ols_slope - true_slope)
        self.assertGreater(ols_slope_err, huber_slope_err * 2.0)

        huber_int_err = abs(huber_intercept - true_intercept)
        ols_int_err = abs(ols_intercept - true_intercept)
        self.assertGreater(ols_int_err, huber_int_err * 2.0)

    def test_zero_variance_fallback(self) -> None:
        x = np.array([4.0, 4.0, 4.0])
        y = np.array([1.0, 2.0, 3.0])
        intercept, slope = rlm_huber(x, y)
        self.assertEqual(intercept, 0.0)
        self.assertEqual(slope, 1.0)

    def test_clean_data_matches_ols(self) -> None:
        x = np.linspace(-5.0, 5.0, 40)
        y = -0.5 + 3.2 * x
        h_intercept, h_slope = rlm_huber(x, y)
        o_intercept, o_slope = ols_fit(x, y)
        self.assertAlmostEqual(h_intercept, o_intercept, places=5)
        self.assertAlmostEqual(h_slope, o_slope, places=5)

    def test_rlm_huber_matches_sklearn_huber_on_clean_and_outlier_data(self) -> None:
        from sklearn.linear_model import HuberRegressor

        x = np.linspace(0.0, 10.0, 50)
        y = 1.5 - 0.5 * x
        y[0] = 50.0
        huber = HuberRegressor(alpha=0.0, epsilon=1.345, max_iter=100).fit(
            x.reshape(-1, 1), y
        )
        intercept, slope = rlm_huber(x, y)
        self.assertAlmostEqual(intercept, float(huber.intercept_), places=6)
        self.assertAlmostEqual(slope, float(huber.coef_[0]), places=6)

    def test_rlm_huber_fallback_on_value_error(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([2.0, 4.0, 6.0, 8.0])
        with unittest.mock.patch(
            "ccnv.stats.prenatal_math.HuberRegressor.fit",
            side_effect=ValueError("convergence failed"),
        ):
            intercept, slope = rlm_huber(x, y)
        o_intercept, o_slope = ols_fit(x, y)
        self.assertAlmostEqual(intercept, o_intercept, places=9)
        self.assertAlmostEqual(slope, o_slope, places=9)
