from __future__ import annotations

import unittest

import numpy as np
import scipy.special

from ccnv.prenatal_math import (
    fit_lasso_fista,
    fit_rprop_mlp,
    infer_sex_gmm_1d,
    ols_fit,
    pca_irlba_numpy,
    rlm_huber,
    train_olm,
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


class TestFitLassoFISTA(unittest.TestCase):
    def test_sparse_recovery_and_non_negativity(self) -> None:
        rng = np.random.default_rng(42)
        n_samples = 120
        n_features = 8
        x = rng.standard_normal(size=(n_samples, n_features))
        true_w = np.array([2.5, 0.0, 1.2, 0.0, 0.0, 3.0, 0.0, 0.5])
        y = x @ true_w + rng.normal(0.0, 0.05, size=n_samples)

        weights = fit_lasso_fista(x, y, alpha=0.05, max_iter=2000, tol=1e-7)

        self.assertTrue(np.all(weights >= 0.0))
        # Active features should be positive
        for idx in (0, 2, 5, 7):
            self.assertGreater(weights[idx], 0.2)
        # Inactive features should be strongly shrunk toward zero
        for idx in (1, 3, 4, 6):
            self.assertLess(weights[idx], 0.05)

    def test_high_alpha_shrinks_to_zero(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.standard_normal(size=(50, 5))
        y = rng.standard_normal(size=50)

        weights = fit_lasso_fista(x, y, alpha=50.0)
        np.testing.assert_allclose(weights, 0.0, atol=1e-9)

    def test_zero_alpha_nnls(self) -> None:
        rng = np.random.default_rng(99)
        x = rng.uniform(1.0, 2.0, size=(40, 3))
        true_w = np.array([1.0, 2.0, 3.0])
        y = x @ true_w

        weights = fit_lasso_fista(x, y, alpha=0.0, max_iter=2500, tol=1e-8)
        self.assertTrue(np.all(weights >= 0.0))
        np.testing.assert_allclose(weights, true_w, rtol=1e-3, atol=1e-3)

    def test_empty_input(self) -> None:
        weights = fit_lasso_fista(np.empty((0, 5)), np.empty(0), alpha=0.1)
        self.assertEqual(weights.shape, (5,))
        np.testing.assert_array_equal(weights, 0.0)

    def test_fit_lasso_fista_all_zero_target_no_warning(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(0.0, 1.0, size=(20, 10))
        y = np.zeros(20, dtype=np.float64)
        w = fit_lasso_fista(x, y, alpha=1e-4)
        np.testing.assert_allclose(w, 0.0, atol=1e-12)


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


class TestFitRpropMLP(unittest.TestCase):
    def test_output_bounded_in_unit_interval(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.standard_normal(size=(30, 4))
        y = rng.uniform(0.1, 0.9, size=30)

        w1, b1, w2, b2 = fit_rprop_mlp(
            x, y, hidden_dim=2, n_inits=3, max_epochs=200, seed=42
        )

        # Test on unseen out-of-range inputs
        x_extreme = np.array([[-100.0, 50.0, 0.0, -20.0], [30.0, -40.0, 10.0, 100.0]])
        h = scipy.special.expit(x_extreme @ w1 + b1)
        pred = scipy.special.expit(h @ w2 + b2)

        self.assertTrue(np.all(pred > 0.0))
        self.assertTrue(np.all(pred < 1.0))

    def test_convergence_on_smooth_target(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(-2.0, 2.0, size=(50, 2))
        y = scipy.special.expit(0.8 * x[:, 0] - 0.5 * x[:, 1] + 0.2)

        w1, b1, w2, b2 = fit_rprop_mlp(
            x, y, hidden_dim=2, n_inits=3, max_epochs=500, seed=42
        )

        h = scipy.special.expit(x @ w1 + b1)
        pred = scipy.special.expit(h @ w2 + b2).ravel()
        mse = float(np.mean((pred - y) ** 2))
        self.assertLess(mse, 0.015)

    def test_deterministic_across_runs(self) -> None:
        rng = np.random.default_rng(77)
        x = rng.standard_normal(size=(20, 3))
        y = rng.uniform(0.2, 0.8, size=20)

        w1_a, b1_a, w2_a, b2_a = fit_rprop_mlp(
            x, y, hidden_dim=2, n_inits=2, max_epochs=100, seed=77
        )
        w1_b, b1_b, w2_b, b2_b = fit_rprop_mlp(
            x, y, hidden_dim=2, n_inits=2, max_epochs=100, seed=77
        )

        np.testing.assert_array_equal(w1_a, w1_b)
        np.testing.assert_array_equal(b1_a, b1_b)
        np.testing.assert_array_equal(w2_a, w2_b)
        np.testing.assert_array_equal(b2_a, b2_b)

    def test_insufficient_samples_guard(self) -> None:
        w1, b1, w2, b2 = fit_rprop_mlp(np.empty((0, 3)), np.empty(0))
        self.assertEqual(w1.shape, (3, 2))
        np.testing.assert_array_equal(w1, 0.0)
        np.testing.assert_array_equal(b1, 0.0)
        np.testing.assert_array_equal(w2, 0.0)
        np.testing.assert_array_equal(b2, 0.0)

        w1_s, b1_s, w2_s, b2_s = fit_rprop_mlp(np.ones((1, 3)), np.ones(1))
        self.assertEqual(w1_s.shape, (3, 2))
        np.testing.assert_array_equal(w1_s, 0.0)
        np.testing.assert_array_equal(b1_s, 0.0)
        np.testing.assert_array_equal(w2_s, 0.0)
        np.testing.assert_array_equal(b2_s, 0.0)


class TestTrainOLM(unittest.TestCase):
    def test_recovers_linear_parameters(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.standard_normal(size=(80, 3))
        true_w = np.array([2.0, -1.5, 0.75])
        true_bias = 3.25
        y = x @ true_w + true_bias

        weights, bias = train_olm(x, y)

        self.assertAlmostEqual(bias, true_bias, places=6)
        np.testing.assert_allclose(weights, true_w, rtol=1e-5, atol=1e-6)

    def test_empty_input(self) -> None:
        weights, bias = train_olm(np.empty((0, 4)), np.empty(0))
        self.assertEqual(weights.shape, (4,))
        self.assertEqual(bias, 0.0)

    def test_single_sample_guard(self) -> None:
        weights, bias = train_olm(np.ones((1, 4)), np.ones(1))
        self.assertEqual(weights.shape, (4,))
        self.assertEqual(bias, 0.0)
        np.testing.assert_array_equal(weights, 0.0)


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
        x = np.linspace(0.0, 10.0, 50)
        y = 1.5 - 0.5 * x
        y[0] = 50.0
        intercept, slope = rlm_huber(x, y)
        self.assertAlmostEqual(intercept, 1.5, delta=0.1)
        self.assertAlmostEqual(slope, -0.5, delta=0.05)
