"""Unit and contract tests for prenatal mathematical estimators, 4D ray SCA, and IRLS fusion."""

from __future__ import annotations

import math
import unittest

import numpy as np
from scipy.special import expit

from ccnv.features.prenatal_origin import _integrate_mosaic_state
from ccnv.models import NUM_FRAG_BANDS
from ccnv.stats.pca import pca_irlba_numpy
from ccnv.stats.prenatal_math import (
    classify_sca_ray_4d,
    classify_twin_pregnancy,
    compute_autosomal_density,
    effective_ffa_sigma,
    evaluate_poisson_msy_ffy,
    fit_forward_ffx_model,
    fit_msy_sex_mixture,
    fit_poisson_msy_ffy,
    fuse_active_fetal_fraction_student_t4,
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
        self.assertTrue(bool(np.all(res.p_male[:30] < 0.05)))
        self.assertTrue(bool(np.all(res.p_male[30:] > 0.95)))

        unimodal = fit_msy_sex_mixture(np.full(20, 2.0), np.full(20, 1000.0))
        self.assertFalse(unimodal.bimodal)
        self.assertEqual(unimodal.p_male.shape, (20,))
        self.assertTrue(bool(np.all(unimodal.p_male == 0.0)))

    def test_poisson_msy_ffy_recovers_calibrated_fetal_fraction(self) -> None:
        rng = np.random.default_rng(42)
        n_bins, n_fem, n_male = 40, 10, 10
        n_samples = n_fem + n_male
        mappability = np.ones(n_bins, dtype=np.float64)
        mappability[:5] = 0.20
        exposure = np.full((n_bins, n_samples), 200.0, dtype=np.float64)
        fem_mask = np.array([True] * n_fem + [False] * n_male)
        male_mask = ~fem_mask

        true_ff = np.zeros(n_samples, dtype=np.float64)
        true_ff[n_fem:] = np.linspace(0.05, 0.22, n_male)
        counts = np.zeros((n_bins, n_samples), dtype=np.float64)
        for s in range(n_samples):
            lam = exposure[:, s] * (0.001 + 0.5 * 0.85 * true_ff[s])
            lam[:5] = 25.0
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

    def test_msy_sex_mixture_succeeds_at_6pct_ff_and_does_not_split_all_female_cohort(
        self,
    ) -> None:
        rng = np.random.default_rng(42)
        n_samples = 20
        true_is_male = np.array([i % 2 == 1 for i in range(n_samples)], dtype=np.bool_)

        msy_exp_6pct = np.full(n_samples, 2000.0, dtype=np.float64)
        lambda_bg = 0.005
        male_rate = lambda_bg + 0.03

        rates_6pct = np.where(true_is_male, male_rate, lambda_bg)
        msy_counts_6pct = rng.poisson(rates_6pct * msy_exp_6pct).astype(np.float64)

        mix_6pct = fit_msy_sex_mixture(msy_counts_6pct, msy_exp_6pct)
        self.assertTrue(mix_6pct.bimodal)
        np.testing.assert_array_equal(mix_6pct.p_male > 0.5, true_is_male)
        self.assertAlmostEqual(mix_6pct.bg_rate, lambda_bg, delta=0.003)

        n_females = 15
        msy_exp_all_female = np.full(n_females, 2000.0, dtype=np.float64)
        msy_counts_all_female = rng.poisson(lambda_bg * msy_exp_all_female).astype(
            np.float64
        )

        mix_all_f = fit_msy_sex_mixture(msy_counts_all_female, msy_exp_all_female)
        self.assertFalse(mix_all_f.bimodal)
        self.assertTrue(bool(np.all(mix_all_f.p_male < 0.05)))

    def test_msy_sex_mixture_final_estep_p_male_consistent(self) -> None:
        # Overlapping rates force many EM iterations; verify returned p_male matches
        # the final E-step evaluated at the final M-step parameters (res.bg_rate).
        rng = np.random.default_rng(99)
        exp = np.full(24, 500.0, dtype=np.float64)
        cts = np.concatenate(
            [rng.poisson(1.5, size=12), rng.poisson(12.0, size=12)]
        ).astype(np.float64)
        res = fit_msy_sex_mixture(cts, exp)
        self.assertTrue(res.bimodal)
        self.assertEqual(res.p_male.shape, (24,))
        self.assertTrue(np.all(np.isfinite(res.p_male)))
        self.assertLess(float(np.max(res.p_male[:12])), 0.05)
        self.assertGreater(float(np.min(res.p_male[12:])), 0.95)
        # Re-evaluate female-group rate weighted by (1 - p_male) -> matches bg_rate closely.
        recomp_bg = float(
            np.sum((1.0 - res.p_male) * cts) / np.sum((1.0 - res.p_male) * exp)
        )
        self.assertAlmostEqual(res.bg_rate, recomp_bg, delta=1e-4)

    def test_poisson_msy_ffy_clean_bins_and_analytical_se(self) -> None:
        rng = np.random.default_rng(456)
        n_bins = 100
        n_samples = 20
        female_mask = np.array([i < 10 for i in range(n_samples)], dtype=np.bool_)
        male_mask = ~female_mask

        mapp = np.ones(n_bins, dtype=np.float32)
        mapp[:10] = 0.60

        exposure = np.full((n_bins, n_samples), 50.0, dtype=np.float64)
        counts = np.zeros((n_bins, n_samples), dtype=np.float64)

        clean_bg = 0.005
        noisy_bg = 0.15
        true_bg = np.full(n_bins, clean_bg, dtype=np.float64)
        true_bg[10:20] = noisy_bg

        for s in range(10):
            counts[:, s] = rng.poisson(true_bg * exposure[:, s])
        for s in range(10, 20):
            counts[:, s] = rng.poisson((true_bg + 0.05) * exposure[:, s])

        model = fit_poisson_msy_ffy(
            msy_bin_counts=counts,
            msy_bin_exposure=exposure,
            mappability=mapp,
            female_mask=female_mask,
            male_mask=male_mask,
        )

        clean_set = set(model.clean_bin_indices.tolist())
        for b in range(20):
            self.assertNotIn(b, clean_set)
        for b in range(20, n_bins):
            self.assertIn(b, clean_set)

        test_exp = np.full(n_bins, 50.0, dtype=np.float64)
        test_counts = rng.poisson((true_bg + 0.5 * 0.08) * test_exp).astype(np.float64)

        ffy_hat, ffy_se = evaluate_poisson_msy_ffy(test_counts, test_exp, model)
        self.assertAlmostEqual(ffy_hat, 0.08, delta=0.02)
        self.assertGreater(ffy_se, 0.0)
        self.assertLess(ffy_se, 0.03)

    def test_fit_poisson_msy_ffy_origin_constrained_calibration_slope(self) -> None:
        e_mat = np.full((4, 6), 1000.0, dtype=np.float64)
        c_mat = np.zeros((4, 6), dtype=np.float64)
        fem_mask = np.array([True, True, True, False, False, False])
        male_mask = ~fem_mask
        labeled_ff = np.array([np.nan, np.nan, np.nan, 0.098, 0.100, 0.102])
        raw_male_ff = np.array([0.092, 0.085, 0.079])
        for j, idx in enumerate([3, 4, 5]):
            c_mat[:, idx] = 0.5 * raw_male_ff[j] * e_mat[:, idx]

        model = fit_poisson_msy_ffy(
            c_mat,
            e_mat,
            np.ones(4, dtype=np.float64),
            fem_mask,
            male_mask,
            labeled_ff,
        )
        c_sub = c_mat[:, 3:]
        e_sub = e_mat[:, 3:]
        raw_post_bg = (
            2.0
            * (np.sum(c_sub, axis=0) - model.lambda_bg @ e_sub)
            / np.sum(e_sub, axis=0)
        )
        expected_slope = float(
            np.dot(labeled_ff[3:], labeled_ff[3:]) / np.dot(raw_post_bg, labeled_ff[3:])
        )
        self.assertAlmostEqual(model.kappa_cal, expected_slope, places=6)
        self.assertGreater(model.kappa_cal, 1.15)

        rng_att = np.random.default_rng(314)
        e_att = np.full((20, 12), 12.5, dtype=np.float64)
        fem_att = np.array([i < 4 for i in range(12)], dtype=np.bool_)
        male_att = ~fem_att
        lab_att = np.where(male_att, np.linspace(0.04, 0.16, 12), np.nan)
        kappas: list[float] = []
        for _ in range(100):
            rate_att = 0.002 + np.where(male_att, 0.5 * lab_att / 1.25, 0.0)
            c_att = rng_att.poisson(e_att * rate_att[None, :]).astype(np.float64)
            m_att = fit_poisson_msy_ffy(
                c_att, e_att, np.ones(20), fem_att, male_att, lab_att
            )
            kappas.append(m_att.kappa_cal)
        self.assertLess(abs(float(np.mean(kappas)) - 1.25), 0.04)

        unimodal = fit_msy_sex_mixture(
            np.array([10.0, 11.0, 9.0, 10.0, 12.0, 10.0]),
            np.full(6, 1000.0),
        )
        self.assertFalse(unimodal.bimodal)


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
        self.assertTrue(bool(np.all((preds > 0.0) & (preds < 1.0))))
        np.testing.assert_allclose(preds, true_ff, atol=0.05)

    def test_autosomal_density_and_forward_rx_unbiased_at_01x_with_094_chrx_baseline(
        self,
    ) -> None:
        rng = np.random.default_rng(123)
        n_autosomes = 22
        bins_per_chr = 1000
        n_total_bins = n_autosomes * bins_per_chr

        contigs = np.repeat(np.arange(1, n_autosomes + 1, dtype=np.int16), bins_per_chr)
        auto_mask = np.ones(n_total_bins, dtype=np.bool_)
        exposure = rng.uniform(0.95, 1.05, n_total_bins).astype(np.float32)
        mapp = rng.uniform(0.85, 1.0, n_total_bins).astype(np.float32)
        gc_trend = rng.uniform(0.90, 1.10, n_total_bins).astype(np.float32)

        true_rho = 0.10
        mean_counts = (true_rho * exposure * mapp * gc_trend).astype(np.float64)
        counts_01x = rng.poisson(mean_counts).astype(np.float64)

        zero_fraction = float(np.mean(counts_01x == 0.0))
        self.assertGreater(zero_fraction, 0.80)

        rho_hat = compute_autosomal_density(
            counts=counts_01x,
            contigs=contigs,
            exposure=exposure,
            mappability=mapp,
            gc_trend=gc_trend,
            autosomal_mask=auto_mask,
        )
        self.assertLess(abs(rho_hat - true_rho) / true_rho, 0.03)

        n_cohort = 24
        f_targets = np.array(
            [0.0] * 12 + [0.04 + 0.01 * i for i in range(12)], dtype=np.float64
        )
        valid_mask = np.ones(n_cohort, dtype=np.bool_)

        true_alpha_0 = 0.94
        true_alpha_1 = -0.5 * true_alpha_0
        noise = rng.normal(0.0, 0.003, n_cohort)
        rx_cohort = true_alpha_0 + true_alpha_1 * f_targets + noise

        alpha_0, alpha_1, sigma_sys = fit_forward_ffx_model(
            rx_cohort, f_targets, valid_mask
        )
        self.assertLess(abs(alpha_0 - 0.94), 0.01)
        self.assertLess(abs(alpha_1 - (-0.47)), 0.03)
        self.assertLess(sigma_sys, 0.015)

    def test_forward_rx_male_shrinkage_on_small_spread(self) -> None:
        n_samples = 10
        f_targets = np.zeros(n_samples, dtype=np.float64)
        rx_cohort = np.full(n_samples, 0.96, dtype=np.float64)
        valid_mask = np.ones(n_samples, dtype=np.bool_)

        alpha_0, alpha_1, _ = fit_forward_ffx_model(rx_cohort, f_targets, valid_mask)
        self.assertAlmostEqual(alpha_0, 0.96, delta=0.001)
        self.assertAlmostEqual(alpha_1, -0.48, delta=0.001)

    def test_compute_autosomal_density_zero_coverage_floor(self) -> None:
        counts = np.zeros(8, dtype=np.float64)
        cids = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int16)
        exp = np.ones(8, dtype=np.float32)
        mask = np.ones(8, dtype=np.bool_)
        density = compute_autosomal_density(counts, cids, exp, exp, exp, mask)
        self.assertAlmostEqual(density, 0.001, places=6)


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
        self.assertTrue(bool(np.all(np.diff(variances) <= 1e-12)))

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
        self.assertTrue(bool(np.all(np.isfinite(scores))))
        self.assertTrue(bool(np.all(np.isfinite(rotation))))


class TestSCARayAndActiveFFFusion(unittest.TestCase):
    def test_prenatal_msy_fusion_and_sca_ray_stats(self) -> None:
        sex_model = fit_msy_sex_mixture(
            np.array([2, 3, 1, 2, 80, 95, 110, 90], dtype=np.float64),
            np.full(8, 2_000.0),
        )
        self.assertTrue(sex_model.bimodal)
        self.assertGreater(sex_model.bg_rate, 0.0)

        fused_ff, fused_se = fuse_active_fetal_fraction_student_t4(
            [0.10, 0.105, 0.098], [0.008, 0.02, 0.018]
        )
        self.assertAlmostEqual(fused_ff, 0.10, delta=0.01)
        self.assertGreater(fused_se, 0.0)

        sca_res = classify_sca_ray_4d(
            delta_x=-0.05,
            sigma_x=0.005,
            ffy=0.10,
            sigma_ffy=0.008,
            ffa=0.10,
            sigma_ffa=0.02,
            ff_size=0.10,
            sigma_size=0.018,
        )
        self.assertEqual(sca_res.call, "XY")
        self.assertAlmostEqual(sca_res.active_ff, 0.10, delta=0.015)
        self.assertGreater(sca_res.active_ff_se, 0.0)
        self.assertIn("XY", sca_res.posteriors)

    def test_4d_ray_sca_classifies_xyy_without_ffa_and_xx_at_3pct_ff(self) -> None:
        res_xyy = classify_sca_ray_4d(-0.06, 0.01, 0.24, 0.012, None, 0.03, 0.12, 0.02)
        self.assertEqual(res_xyy.call, "XYY")
        self.assertGreater(res_xyy.posteriors["XYY"], 0.90)
        self.assertLess(abs(res_xyy.active_ff - 0.12), 0.015)
        self.assertLess(res_xyy.active_ff_se, 0.02)
        self.assertIn(res_xyy.active_ff_source, ("FUSED", "FFY"))

        res_xx = classify_sca_ray_4d(0.0, 0.008, 0.0, 0.004, 0.03, 0.025, 0.032, 0.02)
        self.assertEqual(res_xx.call, "XX")
        self.assertGreater(res_xx.posteriors["XX"], 0.95)
        self.assertLess(abs(res_xx.active_ff - 0.031), 0.005)
        self.assertLess(res_xx.active_ff_se, 0.03)
        self.assertEqual(res_xx.active_ff_source, "FUSED")

    def test_4d_ray_sca_classifies_xxy_xo_xxx_and_discordant_mixed(self) -> None:
        res_xxy = classify_sca_ray_4d(0.0, 0.008, 0.10, 0.008, 0.10, 0.015, 0.10, 0.015)
        self.assertEqual(res_xxy.call, "XXY")
        self.assertGreater(res_xxy.posteriors["XXY"], 0.85)
        self.assertLess(abs(res_xxy.active_ff - 0.10), 0.015)

        res_xo = classify_sca_ray_4d(-0.05, 0.008, 0.0, 0.004, 0.10, 0.015, 0.10, 0.015)
        self.assertEqual(res_xo.call, "XO")
        self.assertGreater(res_xo.posteriors["XO"], 0.85)

        res_maternal_xo = classify_sca_ray_4d(
            -0.05, 0.008, 0.0, 0.004, 0.10, 0.015, 0.10, 0.015, chrx_llr_fetal=-2.5
        )
        self.assertEqual(res_maternal_xo.call, "DISCORDANT_MIXED")

        res_xxx = classify_sca_ray_4d(0.05, 0.008, 0.0, 0.004, 0.10, 0.015, 0.10, 0.015)
        self.assertEqual(res_xxx.call, "XXX")
        self.assertGreater(res_xxx.posteriors["XXX"], 0.85)

        res_disc = classify_sca_ray_4d(
            0.15, 0.008, 0.20, 0.008, 0.08, 0.015, 0.08, 0.015
        )
        self.assertEqual(res_disc.call, "DISCORDANT_MIXED")

    def test_student_t4_fusion_robust_against_single_modality_corruption(self) -> None:
        estimates = [0.08, 0.082, 0.079, 0.350]
        ses = [0.010, 0.012, 0.015, 0.015]
        f_fused, se_fused = fuse_active_fetal_fraction_student_t4(estimates, ses)
        self.assertLess(abs(f_fused - 0.08), 0.006)
        self.assertGreater(se_fused, 0.0)

    def test_classify_sca_ray_4d_bounded_ray_d2_and_tail_log_ndtr(self) -> None:
        gain_res = classify_sca_ray_4d(
            delta_x=0.12,
            sigma_x=0.01,
            ffy=0.0,
            sigma_ffy=0.01,
            ffa=None,
            sigma_ffa=0.02,
            ff_size=None,
            clinical_priors={
                "XX": 0.90,
                "XO": 0.05,
                "XXX": 0.05,
                "XY": 1e-6,
                "XXY": 1e-6,
                "XYY": 1e-6,
            },
        )
        self.assertEqual(gain_res.call, "XXX")
        self.assertNotEqual(gain_res.call, "XO")

        extreme_gain = classify_sca_ray_4d(
            delta_x=0.85,
            sigma_x=0.01,
            ffy=0.0,
            sigma_ffy=0.01,
            ffa=None,
            sigma_ffa=0.02,
            ff_size=None,
        )
        self.assertEqual(extreme_gain.call, "DISCORDANT_MIXED")
        self.assertGreater(extreme_gain.mahalanobis_d2, 20.0)

    def test_student_t4_fusion_respects_cramer_rao_se_and_converged_weights(
        self,
    ) -> None:
        f_eq, se_eq = fuse_active_fetal_fraction_student_t4([0.10, 0.10], [0.01, 0.01])
        self.assertAlmostEqual(f_eq, 0.10, places=6)
        self.assertAlmostEqual(se_eq, 0.01 / math.sqrt(2.0), places=6)

        f_out, se_out = fuse_active_fetal_fraction_student_t4(
            [0.10, 0.10, 0.45], [0.01, 0.01, 0.01]
        )
        self.assertLess(abs(f_out - 0.10), 0.005)
        self.assertGreater(se_out, 0.01 / math.sqrt(3.0))
        self.assertLess(se_out, 0.012)

        f_conf, se_conf = fuse_active_fetal_fraction_student_t4(
            [0.07, 0.10], [0.01, 0.01]
        )
        self.assertAlmostEqual(f_conf, 0.085, places=4)
        self.assertGreater(se_conf, 0.012)

    def test_mosaic_quadrature_captures_narrow_interior_and_boundary_peaks(
        self,
    ) -> None:
        p_f = np.full(NUM_FRAG_BANDS, 1.0 / NUM_FRAG_BANDS, dtype=np.float64)
        p_m = p_f.copy()
        k_s = np.full(NUM_FRAG_BANDS, 10.0, dtype=np.float64)
        m_true = 0.317
        f_bio = 0.12
        r_obs = 1.0 + 0.5 * m_true * f_bio
        log_ev, m_hat, var_m = _integrate_mosaic_state(
            0.05,
            0.95,
            0.5,
            f_bio,
            f_bio,
            0.0002,
            is_fetal=True,
            r_obs=r_obs,
            sigma_r=0.0003,
            k_band_counts=k_s,
            p_fetal_24=p_f,
            p_maternal_24=p_m,
            alpha_dm=10000.0,
        )
        self.assertTrue(math.isfinite(log_ev))
        self.assertAlmostEqual(m_hat, m_true, places=3)
        self.assertGreater(var_m, 0.0)

        log_ev_b, m_hat_b, _ = _integrate_mosaic_state(
            0.05,
            0.95,
            0.5,
            f_bio,
            f_bio,
            0.0002,
            is_fetal=True,
            r_obs=1.15,
            sigma_r=0.0005,
            k_band_counts=k_s,
            p_fetal_24=p_f,
            p_maternal_24=p_m,
            alpha_dm=10000.0,
        )
        self.assertTrue(math.isfinite(log_ev_b))
        self.assertGreater(m_hat_b, 0.90)


class TestTwinStoichiometryAndDegenerateFFASigma(unittest.TestCase):
    def test_effective_ffa_sigma_inflates_only_when_shrunk_and_alt_channel_present(
        self,
    ) -> None:
        shrunk_weights = [0.01525, 0.0, 0.0]
        sig_inflated = effective_ffa_sigma(0.0375, shrunk_weights, has_alt_ff=True)
        self.assertGreater(sig_inflated, 0.50)
        self.assertAlmostEqual(sig_inflated, 0.25 * math.sqrt(0.10 / 0.01525), places=6)

        sig_no_alt = effective_ffa_sigma(0.0375, shrunk_weights, has_alt_ff=False)
        self.assertAlmostEqual(sig_no_alt, 0.0375, places=6)

        sig_none_alt = effective_ffa_sigma(0.0375, None, has_alt_ff=True)
        self.assertAlmostEqual(sig_none_alt, 0.25 * math.sqrt(0.10 / 0.01), places=6)

        unshrunk_weights = [0.20, 0.15]
        sig_unshrunk = effective_ffa_sigma(0.0375, unshrunk_weights, has_alt_ff=True)
        self.assertAlmostEqual(sig_unshrunk, 0.0375, places=6)

        sig_floor = effective_ffa_sigma(0.001, unshrunk_weights, has_alt_ff=False)
        self.assertAlmostEqual(sig_floor, 0.005, places=6)

    def test_classify_twin_pregnancy_and_sca_cohort_concordance(
        self,
    ) -> None:
        yfrac_thresh = 0.000043
        olm_norm = 0.01525
        sigma_ffa_eff = effective_ffa_sigma(0.0375, [olm_norm], has_alt_ff=True)
        sigma_size = 0.0247
        sigma_ffy = 0.0050
        sigma_x = 0.5 * 0.009393 / 0.398684

        empirical_twins = [
            ("sample1", "FM", "M", 0.154, 0.133, 0.102, 0.352, 0.351, 0.00025),
            ("sample2", "MM", "M", 0.173, 0.156, 0.102, 0.174, 0.171, 0.00028),
            ("sample3", "MM", "M", 0.244, 0.284, 0.102, 0.196, 0.177, 0.00039),
            ("sample4", "FF", "F", None, 0.044, 0.103, 0.305, 0.294, 0.00000),
            ("sample5", "MM", "M", 0.119, 0.085, 0.100, 0.084, 0.116, 0.00019),
            ("sample6", "FM", "M", 0.143, 0.154, 0.102, 0.218, 0.144, 0.00023),
            ("sample7", "FF", "F", None, -0.010, 0.101, 0.093, 0.096, 0.00000),
            ("sample8", "FM", "M", 0.094, 0.134, 0.097, 0.128, 0.096, 0.00015),
            ("sample9", "MM", "M", 0.127, 0.131, 0.100, 0.069, 0.126, 0.00021),
            ("sample10", "MM", "M", 0.176, 0.145, 0.103, 0.181, 0.174, 0.00028),
            ("sample11", "FF", "F", None, -0.015, 0.103, 0.082, 0.088, 0.00000),
            ("sample12", "FM", "M", 0.075, 0.073, 0.100, 0.177, 0.159, 0.00012),
        ]

        for (
            sid,
            truth_call,
            sex,
            ffy,
            ffx,
            ffa,
            ff_size,
            act_ff,
            chry_sc,
        ) in empirical_twins:
            truth_sexes = (truth_call[0], truth_call[1])
            delta_x = -0.5 * ffx
            cls = classify_twin_pregnancy(
                fetal_sex="F" if sex == "F" else "M",
                chry_score=chry_sc,
                yfrac_threshold=yfrac_thresh,
                ffy=ffy,
                sigma_ffy=sigma_ffy,
                delta_x=delta_x,
                sigma_x=sigma_x,
                ffa=ffa,
                sigma_ffa=sigma_ffa_eff,
                ff_size=ff_size,
                sigma_size=sigma_size,
                active_ff=act_ff,
                active_ff_se=0.015,
                active_ff_source="FUSED",
                twins=True,
                olm_norm=olm_norm,
            )
            self.assertEqual(cls.twin.call, truth_call, f"Failed on {sid}")
            self.assertEqual(cls.twin.fetal_sexes, truth_sexes, f"Failed on {sid}")
            self.assertTrue(cls.twin.is_twin, f"Failed on {sid}")
            self.assertEqual(cls.gestation, "twin", f"Failed on {sid}")
            self.assertEqual(cls.fetal_sexes, truth_sexes, f"Failed on {sid}")
            self.assertEqual(cls.twin.twin_probability, 1.0)
            self.assertAlmostEqual(
                sum(cls.twin.posteriors.values()), 1.0, places=5, msg=sid
            )
            self.assertIsNotNone(cls.twin_confidence)
            if truth_call == "FF":
                self.assertEqual(cls.male_ff, 0.0)
                self.assertGreater(cls.female_ff or 0.0, 0.0)
            elif truth_call == "FM":
                self.assertAlmostEqual(cls.male_ff or 0.0, ffy or 0.0, places=5)
                self.assertGreater(cls.female_ff or 0.0, 0.0)
                self.assertEqual(cls.active_ff_source, "FF_SIZE")
            else:
                self.assertGreater(cls.male_ff or 0.0, 0.0)
                self.assertEqual(cls.female_ff, 0.0)

        p0 = dict.fromkeys(("XX", "XY", "XO", "XXX", "XXY", "XYY"), 1e-6)
        fem_p = p0 | {"XX": 0.90, "XO": 0.05, "XXX": 0.05}
        male_p = p0 | {"XY": 0.90, "XXY": 0.05, "XYY": 0.05}
        empirical_scas = [
            ("sample13", "XXX", "F", None, -0.1407, 0.1009, 0.1286, 0.000004),
            ("sample14", "XXY", "M", 0.1322, 0.0130, 0.1002, 0.1186, 0.000220),
            ("sample15", "XO", "F", None, 0.0925, 0.0955, 0.0502, 0.000003),
        ]
        for sid, truth_sca, sex, ffy, ffx, ffa, ff_size, chry_sc in empirical_scas:
            fsex = "F" if sex == "F" else "M"
            delta_x = -0.5 * ffx
            sca_cls = classify_sca_ray_4d(
                delta_x=delta_x,
                sigma_x=sigma_x,
                ffy=ffy if fsex == "M" else 0.0,
                sigma_ffy=sigma_ffy,
                ffa=ffa,
                sigma_ffa=sigma_ffa_eff,
                ff_size=ff_size,
                sigma_size=sigma_size,
                clinical_priors=fem_p if fsex == "F" else male_p,
            )
            self.assertEqual(sca_cls.call, truth_sca, f"SCA failed on {sid}")
            twin_cls = classify_twin_pregnancy(
                fetal_sex=fsex,
                chry_score=chry_sc,
                yfrac_threshold=yfrac_thresh,
                ffy=ffy,
                sigma_ffy=sigma_ffy,
                delta_x=delta_x,
                sigma_x=sigma_x,
                ffa=ffa,
                sigma_ffa=sigma_ffa_eff,
                ff_size=ff_size,
                sigma_size=sigma_size,
                active_ff=sca_cls.active_ff,
                active_ff_se=sca_cls.active_ff_se,
                active_ff_source=sca_cls.active_ff_source,
                twins=False,
                olm_norm=olm_norm,
            )
            self.assertEqual(twin_cls.gestation, "singleton", f"Twin failed on {sid}")
            self.assertFalse(twin_cls.twin.is_twin, f"Twin failed on {sid}")

    def test_classify_twin_pregnancy_auto_mode_promotion_and_47xxy_guard(self) -> None:
        fm_auto = classify_twin_pregnancy(
            fetal_sex="M",
            chry_score=0.00023,
            yfrac_threshold=0.000043,
            ffy=0.09,
            sigma_ffy=0.005,
            delta_x=-0.045,
            sigma_x=0.008,
            ffa=0.22,
            sigma_ffa=0.02,
            ff_size=0.22,
            sigma_size=0.02,
            active_ff=0.14,
            active_ff_se=0.01,
            active_ff_source="FUSED",
            twins=False,
            olm_norm=0.50,
        )
        self.assertEqual(fm_auto.gestation, "twin")
        self.assertTrue(fm_auto.twin.is_twin)
        self.assertEqual(fm_auto.twin.call, "FM")
        self.assertEqual(fm_auto.fetal_sexes, ("F", "M"))
        self.assertGreater(fm_auto.twin.twin_probability, 0.80)
        self.assertAlmostEqual(fm_auto.active_ff, 0.22, places=3)
        self.assertEqual(fm_auto.active_ff_source, "FUSED")

        ff_high = classify_twin_pregnancy(
            fetal_sex="F",
            chry_score=0.0,
            yfrac_threshold=0.000043,
            ffy=None,
            sigma_ffy=0.005,
            delta_x=0.0,
            sigma_x=0.008,
            ffa=0.28,
            sigma_ffa=0.02,
            ff_size=0.29,
            sigma_size=0.02,
            active_ff=0.285,
            active_ff_se=0.014,
            active_ff_source="FUSED",
            twins=False,
        )
        self.assertEqual(ff_high.gestation, "singleton")
        self.assertFalse(ff_high.twin.is_twin)
        self.assertIsNone(ff_high.fetal_sexes)
        self.assertGreater(ff_high.twin.twin_probability, 0.80)

        xy_single = classify_twin_pregnancy(
            fetal_sex="M",
            chry_score=0.00018,
            yfrac_threshold=0.000043,
            ffy=0.11,
            sigma_ffy=0.005,
            delta_x=-0.055,
            sigma_x=0.008,
            ffa=0.11,
            sigma_ffa=0.02,
            ff_size=0.11,
            sigma_size=0.02,
            active_ff=0.11,
            active_ff_se=0.005,
            active_ff_source="FUSED",
            twins=False,
        )
        self.assertEqual(xy_single.gestation, "singleton")
        self.assertFalse(xy_single.twin.is_twin)
        self.assertIsNone(xy_single.fetal_sexes)
        self.assertIsNone(xy_single.male_ff)
        self.assertIsNone(xy_single.female_ff)
        self.assertIsNone(xy_single.twin_confidence)
        self.assertEqual(xy_single.twin.call, "MM")

        xxy_single = classify_twin_pregnancy(
            fetal_sex="M",
            chry_score=0.00020,
            yfrac_threshold=0.000043,
            ffy=0.10,
            sigma_ffy=0.005,
            delta_x=0.08,
            sigma_x=0.005,
            ffa=0.19,
            sigma_ffa=0.02,
            ff_size=0.19,
            sigma_size=0.02,
            active_ff=0.13,
            active_ff_se=0.01,
            active_ff_source="FUSED",
            twins=False,
        )
        self.assertEqual(xxy_single.gestation, "singleton")
        self.assertFalse(xxy_single.twin.is_twin)
        self.assertIsNone(xxy_single.fetal_sexes)
        self.assertLess(xxy_single.twin.posteriors["FM"], 0.01)

        xxy_high = classify_twin_pregnancy(
            fetal_sex="M",
            chry_score=0.00028,
            yfrac_threshold=0.000043,
            ffy=0.28,
            sigma_ffy=0.005,
            delta_x=0.08,
            sigma_x=0.005,
            ffa=0.28,
            sigma_ffa=0.02,
            ff_size=0.28,
            sigma_size=0.02,
            active_ff=0.28,
            active_ff_se=0.01,
            active_ff_source="FUSED",
            twins=False,
        )
        self.assertEqual(xxy_high.gestation, "singleton")
        self.assertFalse(xxy_high.twin.is_twin)

        no_size_male = classify_twin_pregnancy(
            fetal_sex="M",
            chry_score=0.00020,
            yfrac_threshold=0.000043,
            ffy=0.12,
            sigma_ffy=0.006,
            delta_x=-0.06,
            sigma_x=0.01,
            ffa=0.10,
            sigma_ffa=0.640,
            ff_size=None,
            sigma_size=0.02,
            active_ff=0.12,
            active_ff_se=0.006,
            active_ff_source="FFY",
            twins=False,
            olm_norm=0.01525,
        )
        self.assertEqual(no_size_male.gestation, "singleton")
        self.assertEqual(no_size_male.twin.call, "MM")
        self.assertAlmostEqual(no_size_male.active_ff_se, 0.006, places=5)

        vanishing_male = classify_twin_pregnancy(
            fetal_sex="M",
            chry_score=0.00008,
            yfrac_threshold=0.000043,
            ffy=0.03,
            sigma_ffy=0.005,
            delta_x=-0.015,
            sigma_x=0.01,
            ffa=0.14,
            sigma_ffa=0.02,
            ff_size=0.14,
            sigma_size=0.02,
            active_ff=0.10,
            active_ff_se=0.01,
            active_ff_source="FUSED",
            twins=False,
        )
        self.assertEqual(vanishing_male.gestation, "singleton")
        self.assertFalse(vanishing_male.twin.is_twin)

        missing_ffy_male = classify_twin_pregnancy(
            fetal_sex="M",
            chry_score=0.00020,
            yfrac_threshold=0.000043,
            ffy=None,
            sigma_ffy=0.005,
            delta_x=0.0,
            sigma_x=0.005,
            ffa=0.12,
            sigma_ffa=0.005,
            ff_size=None,
            sigma_size=0.02,
            active_ff=0.12,
            active_ff_se=0.005,
            active_ff_source="FFA",
            twins=False,
        )
        self.assertEqual(missing_ffy_male.gestation, "singleton")
        self.assertEqual(missing_ffy_male.twin.call, "MM")
        self.assertAlmostEqual(missing_ffy_male.twin.male_ff, 0.12, places=5)
        self.assertLess(missing_ffy_male.twin.confidence, 0.55)

        missing_ffy_fm_twin = classify_twin_pregnancy(
            fetal_sex="M",
            chry_score=0.00020,
            yfrac_threshold=0.000043,
            ffy=None,
            sigma_ffy=0.005,
            delta_x=-0.055,
            sigma_x=0.005,
            ffa=0.22,
            sigma_ffa=0.005,
            ff_size=None,
            sigma_size=0.02,
            active_ff=0.22,
            active_ff_se=0.005,
            active_ff_source="FFA",
            twins=True,
        )
        self.assertEqual(missing_ffy_fm_twin.gestation, "twin")
        self.assertEqual(missing_ffy_fm_twin.twin.call, "FM")
        self.assertAlmostEqual(missing_ffy_fm_twin.male_ff or 0.0, 0.11, places=3)
        self.assertAlmostEqual(missing_ffy_fm_twin.female_ff or 0.0, 0.11, places=3)

        unequal_fm = classify_twin_pregnancy(
            fetal_sex="M",
            chry_score=0.00025,
            yfrac_threshold=0.000043,
            ffy=0.16,
            sigma_ffy=0.005,
            delta_x=-0.08,
            sigma_x=0.008,
            ffa=None,
            sigma_ffa=0.02,
            ff_size=0.22,
            sigma_size=0.005,
            active_ff=0.16,
            active_ff_se=0.005,
            active_ff_source="FFY",
            twins=True,
            olm_norm=0.50,
        )
        self.assertEqual(unequal_fm.twin.call, "FM")
        self.assertEqual(unequal_fm.active_ff_source, "FF_SIZE")
        self.assertGreater(unequal_fm.twin.female_ff, 0.0)


if __name__ == "__main__":
    unittest.main()
