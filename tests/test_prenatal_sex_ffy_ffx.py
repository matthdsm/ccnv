"""Contract tests for Plan 3 Task 1: Prenatal sex mixture, Poisson FFY, and forward RX."""

from __future__ import annotations

import unittest

import numpy as np

from ccnv.stats.prenatal_math import (
    compute_autosomal_density,
    evaluate_poisson_msy_ffy,
    fit_forward_ffx_model,
    fit_msy_sex_mixture,
    fit_poisson_msy_ffy,
)


class TestPrenatalSexFFYFFX(unittest.TestCase):
    """Test prenatal sex mixture, Poisson MSY FFY, and forward RX calibration."""

    def test_msy_sex_mixture_succeeds_at_6pct_ff_and_does_not_split_all_female_cohort(
        self,
    ) -> None:
        """Test EM sex mixture resolves 6% FF males and does not split 100% female cohort."""
        rng = np.random.default_rng(42)
        n_samples = 20
        true_is_male = np.array([i % 2 == 1 for i in range(n_samples)], dtype=np.bool_)

        # Background cross-mapping rate: 0.005, exposure: 2000.0 per sample
        msy_exp_6pct = np.full(n_samples, 2000.0, dtype=np.float64)
        lambda_bg = 0.005
        # 6% FF male has delta = 0.5 * 0.06 = 0.03
        male_rate = lambda_bg + 0.03

        rates_6pct = np.where(true_is_male, male_rate, lambda_bg)
        msy_counts_6pct = rng.poisson(rates_6pct * msy_exp_6pct).astype(np.float64)

        mix_6pct = fit_msy_sex_mixture(msy_counts_6pct, msy_exp_6pct)
        self.assertTrue(mix_6pct.bimodal)
        np.testing.assert_array_equal(mix_6pct.p_male > 0.5, true_is_male)
        self.assertAlmostEqual(mix_6pct.bg_rate, lambda_bg, delta=0.003)

        # 100% female cohort of size 15
        n_females = 15
        msy_exp_all_female = np.full(n_females, 2000.0, dtype=np.float64)
        msy_counts_all_female = rng.poisson(lambda_bg * msy_exp_all_female).astype(
            np.float64
        )

        mix_all_f = fit_msy_sex_mixture(msy_counts_all_female, msy_exp_all_female)
        self.assertFalse(mix_all_f.bimodal)
        self.assertTrue(bool(np.all(mix_all_f.p_male < 0.05)))

    def test_autosomal_density_and_forward_rx_unbiased_at_01x_with_094_chrx_baseline(
        self,
    ) -> None:
        """Test all-bin ratio-of-sums at 0.1x WGS and forward RX with 0.94 baseline."""
        rng = np.random.default_rng(123)

        # 22 autosomes, 1000 bins each = 22,000 bins
        n_autosomes = 22
        bins_per_chr = 1000
        n_total_bins = n_autosomes * bins_per_chr

        contigs = np.repeat(np.arange(1, n_autosomes + 1, dtype=np.int16), bins_per_chr)
        auto_mask = np.ones(n_total_bins, dtype=np.bool_)
        exposure = rng.uniform(0.95, 1.05, n_total_bins).astype(np.float32)
        mapp = rng.uniform(0.85, 1.0, n_total_bins).astype(np.float32)
        gc_trend = rng.uniform(0.90, 1.10, n_total_bins).astype(np.float32)

        # True density at 0.1x WGS (~0.1 reads per bin)
        true_rho = 0.10
        mean_counts = (true_rho * exposure * mapp * gc_trend).astype(np.float64)
        counts_01x = rng.poisson(mean_counts).astype(np.float64)

        # Verify many bins are zero (~90%)
        zero_fraction = float(np.mean(counts_01x == 0.0))
        self.assertGreater(zero_fraction, 0.80)

        # Autosomal density must include C=0 and have < 3% error
        rho_hat = compute_autosomal_density(
            counts=counts_01x,
            contigs=contigs,
            exposure=exposure,
            mappability=mapp,
            gc_trend=gc_trend,
            autosomal_mask=auto_mask,
        )
        self.assertLess(abs(rho_hat - true_rho) / true_rho, 0.03)

        # Test forward RX modeling with 0.94 baseline
        n_cohort = 24
        # 12 females (f = 0.0), 12 males with fetal fractions 4% to 15%
        f_targets = np.array(
            [0.0] * 12 + [0.04 + 0.01 * i for i in range(12)], dtype=np.float64
        )
        valid_mask = np.ones(n_cohort, dtype=np.bool_)

        true_alpha_0 = 0.94
        true_alpha_1 = -0.5 * true_alpha_0  # -0.47
        noise = rng.normal(0.0, 0.003, n_cohort)
        rx_cohort = true_alpha_0 + true_alpha_1 * f_targets + noise

        alpha_0, alpha_1, sigma_sys = fit_forward_ffx_model(
            rx_cohort, f_targets, valid_mask
        )
        self.assertLess(abs(alpha_0 - 0.94), 0.01)
        self.assertLess(abs(alpha_1 - (-0.47)), 0.03)
        self.assertLess(sigma_sys, 0.015)

    def test_poisson_msy_ffy_clean_bins_and_analytical_se(self) -> None:
        """Test empirical MSY bin filtering and closed-form Poisson MLE FFY."""
        rng = np.random.default_rng(456)
        n_bins = 100
        n_samples = 20
        female_mask = np.array([i < 10 for i in range(n_samples)], dtype=np.bool_)
        male_mask = ~female_mask

        mapp = np.ones(n_bins, dtype=np.float32)
        # 10 low mappability bins
        mapp[:10] = 0.60

        exposure = np.full((n_bins, n_samples), 50.0, dtype=np.float64)
        counts = np.zeros((n_bins, n_samples), dtype=np.float64)

        # Background cross-mapping: 0.005 for clean bins, 0.15 for bins 10..20
        clean_bg = 0.005
        noisy_bg = 0.15
        true_bg = np.full(n_bins, clean_bg, dtype=np.float64)
        true_bg[10:20] = noisy_bg

        # Simulate female counts
        for s in range(10):
            counts[:, s] = rng.poisson(true_bg * exposure[:, s])

        # Simulate male counts with true FF = 0.10 (rate = bg + 0.5 * 0.10 = bg + 0.05)
        for s in range(10, 20):
            counts[:, s] = rng.poisson((true_bg + 0.05) * exposure[:, s])

        model = fit_poisson_msy_ffy(
            msy_bin_counts=counts,
            msy_bin_exposure=exposure,
            mappability=mapp,
            female_mask=female_mask,
            male_mask=male_mask,
        )

        # Bins 0..9 (low map) and 10..19 (high female bg) must be excluded
        clean_set = set(model.clean_bin_indices.tolist())
        for b in range(20):
            self.assertNotIn(b, clean_set)
        for b in range(20, n_bins):
            self.assertIn(b, clean_set)

        # Evaluate a test male sample with true FF = 0.08
        test_exp = np.full(n_bins, 50.0, dtype=np.float64)
        test_counts = rng.poisson((true_bg + 0.5 * 0.08) * test_exp).astype(np.float64)

        ffy_hat, ffy_se = evaluate_poisson_msy_ffy(test_counts, test_exp, model)
        self.assertAlmostEqual(ffy_hat, 0.08, delta=0.02)
        self.assertGreater(ffy_se, 0.0)
        self.assertLess(ffy_se, 0.03)

    def test_forward_rx_male_shrinkage_on_small_spread(self) -> None:
        """Test fit_forward_ffx_model shrinks slope when male FF spread < 1e-4."""
        n_samples = 10
        f_targets = np.zeros(n_samples, dtype=np.float64)
        rx_cohort = np.full(n_samples, 0.96, dtype=np.float64)
        valid_mask = np.ones(n_samples, dtype=np.bool_)

        alpha_0, alpha_1, _ = fit_forward_ffx_model(rx_cohort, f_targets, valid_mask)
        self.assertAlmostEqual(alpha_0, 0.96, delta=0.001)
        self.assertAlmostEqual(alpha_1, -0.48, delta=0.001)


if __name__ == "__main__":
    unittest.main()
