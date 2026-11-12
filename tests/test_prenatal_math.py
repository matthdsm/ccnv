from __future__ import annotations

import math
import unittest

import numpy as np
from scipy.special import expit

from ccnv.stats.pca import pca_irlba_numpy
from ccnv.stats.prenatal_math import (
    evaluate_poisson_msy_ffy,
    fit_forward_ffx_model,
    fit_msy_sex_mixture,
    fit_poisson_msy_ffy,
)
from ccnv.stats.regression import fit_logit_linear_ensemble


class TestMSYSexMixtureAndPoissonFFY(unittest.TestCase):
    def test_fit_msy_sex_mixture_bimodal_and_unimodal(self) -> None:
        rng = np.random.default_rng(123)
        exposure = np.full(60, 1000.0, dtype=np.float64)
        fem_counts = rng.poisson(0.5, size=30).astype(np.float64)
        male_counts = rng.poisson(60.0, size=30).astype(np.float64)
        counts = np.concatenate([fem_counts, male_counts])

        res = fit_msy_sex_mixture(counts, exposure)
        self.assertTrue(res.bimodal)
        self.assertTrue(np.all(res.p_male[:30] < 0.05))
        self.assertTrue(np.all(res.p_male[30:] > 0.95))

        unimodal = fit_msy_sex_mixture(np.full(20, 2.0), np.full(20, 1000.0))
        self.assertFalse(unimodal.bimodal)
        self.assertEqual(unimodal.p_male.shape, (20,))
        self.assertTrue(np.all(unimodal.p_male == 0.0))

    def test_poisson_msy_ffy_recovers_calibrated_fetal_fraction(self) -> None:
        rng = np.random.default_rng(42)
        n_bins, n_fem, n_male = 40, 10, 10
        n_samples = n_fem + n_male
        mappability = np.ones(n_bins, dtype=np.float64)
        mappability[:5] = 0.20  # Low-mappability bins excluded from clean set
        exposure = np.full((n_bins, n_samples), 200.0, dtype=np.float64)
        fem_mask = np.array([True] * n_fem + [False] * n_male)
        male_mask = ~fem_mask

        true_ff = np.zeros(n_samples, dtype=np.float64)
        true_ff[n_fem:] = np.linspace(0.05, 0.22, n_male)
        counts = np.zeros((n_bins, n_samples), dtype=np.float64)
        # Simulate slight empirical capture scale 0.85 so kappa_cal ~= 1 / 0.85
        for s in range(n_samples):
            lam = exposure[:, s] * (0.001 + 0.5 * 0.85 * true_ff[s])
            lam[:5] = 25.0  # Noisy repetitive bins
            counts[:, s] = rng.poisson(lam)

        model = fit_poisson_msy_ffy(
            counts,
            exposure,
            mappability,
            fem_mask,
            male_mask,
            labeled_ff=true_ff,
        )
        self.assertEqual(model.clean_bin_indices.size, 35)
        self.assertGreater(model.kappa_cal, 1.05)
        self.assertLess(model.kappa_cal, 1.35)

        ffy_hat, ffy_se = evaluate_poisson_msy_ffy(
            counts[:, -1], exposure[:, -1], model
        )
        self.assertAlmostEqual(ffy_hat, float(true_ff[-1]), delta=0.025)
        self.assertGreater(ffy_se, 0.0)


class TestForwardFFXAndFFALogitEnsemble(unittest.TestCase):
    def test_fit_forward_ffx_model_recovers_intercept_and_negative_slope(self) -> None:
        f_targets = np.array([0.0, 0.0, 0.0, 0.06, 0.10, 0.16, 0.20])
        chrx_ratios = 1.0 - 0.5 * f_targets
        valid = np.ones_like(f_targets, dtype=np.bool_)
        alpha_0, alpha_1, sigma_sys = fit_forward_ffx_model(
            chrx_ratios, f_targets, valid
        )
        self.assertAlmostEqual(alpha_0, 1.0, places=4)
        self.assertAlmostEqual(alpha_1, -0.5, places=4)
        self.assertGreaterEqual(sigma_sys, 0.005)

    def test_recovers_smooth_bounded_fetal_fraction(self) -> None:
        rng = np.random.default_rng(42)
        n_samples = 30
        n_features = 10
        x_mat = rng.standard_normal((n_samples, n_features))
        true_beta = rng.uniform(-0.5, 0.5, size=n_features)
        true_ff = expit(x_mat @ true_beta - 2.0)

        fit = fit_logit_linear_ensemble(
            x_mat, true_ff, standardize=False, l1_ratios=(0.1, 0.5), seed=42
        )
        self.assertEqual(fit.weights.shape, (n_features,))
        self.assertTrue(math.isfinite(fit.intercept))
        self.assertGreater(fit.oof_rmse, 0.0)
        self.assertLess(fit.oof_rmse, 0.10)

        preds = expit(x_mat @ fit.weights + fit.intercept)
        self.assertTrue(np.all((preds > 0.0) & (preds < 1.0)))
        np.testing.assert_allclose(preds, true_ff, atol=0.05)


class TestPCAIrlbaNumpy(unittest.TestCase):
    def test_orthogonality_and_centering_svd_branch(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.normal(loc=5.0, scale=2.0, size=(60, 10))
        scores, center, rotation = pca_irlba_numpy(x, n_components=4, seed=42)

        self.assertEqual(scores.shape, (60, 4))
        self.assertEqual(rotation.shape, (10, 4))
        np.testing.assert_allclose(center, np.mean(x, axis=0), atol=1e-12)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(4), atol=1e-7)
        variances = np.var(scores, axis=0)
        self.assertTrue(np.all(np.diff(variances) <= 1e-12))

    def test_gram_matrix_branch_samples_le_features(self) -> None:
        rng = np.random.default_rng(101)
        x = rng.standard_normal(size=(8, 25))
        scores, center, rotation = pca_irlba_numpy(x, n_components=3, seed=101)

        self.assertEqual(scores.shape, (8, 3))
        self.assertEqual(rotation.shape, (25, 3))
        np.testing.assert_allclose(center, np.mean(x, axis=0), atol=1e-12)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)

    def test_sign_canonicalization_and_determinism(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.standard_normal(size=(50, 12))
        sc1, c1, rot1 = pca_irlba_numpy(x, n_components=5, seed=7)
        sc2, c2, rot2 = pca_irlba_numpy(x, n_components=5, seed=7)
        np.testing.assert_array_equal(sc1, sc2)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(rot1, rot2)

        for j in range(5):
            col = rot1[:, j]
            max_idx = int(np.argmax(np.abs(col)))
            self.assertGreaterEqual(col[max_idx], 0.0)

    def test_components_exceeding_rank_and_all_nan_column(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.standard_normal(size=(5, 10))
        x[:, 3] = np.nan
        scores, center, rotation = pca_irlba_numpy(x, n_components=8, seed=1)

        self.assertEqual(scores.shape, (5, 8))
        self.assertEqual(rotation.shape, (10, 8))
        self.assertEqual(center[3], 0.0)
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertTrue(np.all(np.isfinite(rotation)))


if __name__ == "__main__":
    unittest.main()
