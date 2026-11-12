"""Unit tests for zero-inclusive Poisson covariate regression, basis builders, and logit ensembles."""

from __future__ import annotations

import inspect
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from ccnv.features.covariates import covariate_features
from ccnv.models import InvalidInputError
from ccnv.stats.neighbors import build_whitened_subspace_features
from ccnv.stats.regression import (
    _count_distinct_up_to,
    build_exogenous_covariate_basis,
    fit_logit_linear_ensemble,
    fit_poisson_covariate_trends,
    half_sample_mode,
    mad_scale,
    standardize_column,
)


class TestPoissonCovariates(unittest.TestCase):
    def test_zero_inclusive_poisson_trend_unbiased_at_lambda_04(self) -> None:
        rng = np.random.default_rng(42)
        n_bins = 40_000
        usable = np.ones(n_bins, dtype=np.bool_)
        usable[:200] = False

        gc = rng.uniform(0.28, 0.68, size=n_bins).astype(np.float32)
        gc[~usable] = np.nan
        exposure = np.ones(n_bins, dtype=np.float32)
        exposure[-500:] = rng.uniform(0.45, 0.95, size=500).astype(np.float32)

        g_valid = gc[usable].astype(np.float64)
        log_curve = -12.0 * np.square(g_valid - 0.45) + 0.35 * (g_valid - 0.45)
        base_rate = 0.40 / float(np.mean(np.exp(log_curve)))
        true_unit_rate = np.zeros(n_bins, dtype=np.float64)
        true_unit_rate[usable] = base_rate * np.exp(log_curve)
        true_bin_rate = exposure.astype(np.float64) * true_unit_rate

        sf = np.array([1.0, 1.35], dtype=np.float64)
        lam = true_bin_rate[:, np.newaxis] * sf[np.newaxis, :]
        counts = rng.poisson(lam).astype(np.float32)

        zero_frac = float(np.mean(counts[usable, 0] == 0.0))
        self.assertGreater(zero_frac, 0.60)

        basis = build_exogenous_covariate_basis(usable, bin_size=1_000, gc=gc)
        self.assertEqual(basis.shape[0], int(np.count_nonzero(usable)))
        self.assertGreater(basis.shape[1], 0)
        np.testing.assert_allclose(np.mean(basis, axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.std(basis, axis=0), 1.0, atol=1e-10)

        trends = fit_poisson_covariate_trends(counts, sf, exposure, usable, basis)
        self.assertEqual(trends.dtype, np.float32)
        self.assertEqual(trends.shape, (n_bins, 2))
        np.testing.assert_array_equal(trends[~usable], 0.0)
        self.assertTrue(bool(np.all(trends[usable] > 0.0)))

        low_gc_bins = usable & (gc < 0.34)
        high_gc_bins = usable & (gc > 0.62)
        for s_idx in range(2):
            rel_bias_low_gc = float(
                np.median(trends[low_gc_bins, s_idx] / true_bin_rate[low_gc_bins]) - 1.0
            )
            rel_bias_high_gc = float(
                np.median(trends[high_gc_bins, s_idx] / true_bin_rate[high_gc_bins])
                - 1.0
            )
            self.assertLess(abs(rel_bias_low_gc), 0.06)
            self.assertLess(abs(rel_bias_high_gc), 0.06)

    def test_500kb_gc_quantile_basis_resolves_bimodal_isochore_jensen_gap(self) -> None:
        rng = np.random.default_rng(101)
        n_bins = 600
        usable = np.ones(n_bins, dtype=np.bool_)

        band_gc = np.array(
            [0.20, 0.32, 0.375, 0.425, 0.475, 0.525, 0.60, 0.68], dtype=np.float64
        )
        raw_weights = rng.dirichlet(np.ones(7), size=n_bins)
        gc_q = np.zeros((n_bins, 8), dtype=np.float32)
        gc_q[:, 1:] = raw_weights.astype(np.float32)

        bin_a, bin_b = 10, 11
        gc_q[bin_a] = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0], dtype=np.float32
        )
        gc_q[bin_b] = np.array(
            [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32
        )

        gc_mean = (gc_q.astype(np.float64) @ band_gc).astype(np.float32)
        self.assertAlmostEqual(float(gc_mean[bin_a]), 0.50, places=6)
        self.assertAlmostEqual(float(gc_mean[bin_b]), 0.50, places=6)

        subwindow_rates = 300.0 * np.exp(-18.0 * np.square(band_gc - 0.50))
        true_bin_rate = gc_q.astype(np.float64) @ subwindow_rates
        exposure = np.ones(n_bins, dtype=np.float32)
        sf = np.array([1.0], dtype=np.float64)
        counts_concave_pcr = rng.poisson(true_bin_rate[:, np.newaxis]).astype(
            np.float32
        )

        basis_500kb = build_exogenous_covariate_basis(
            usable, bin_size=500_000, gc=gc_mean, gc_quantiles=gc_q
        )
        self.assertTrue(bool(np.all(np.isfinite(basis_500kb))))
        self.assertEqual(basis_500kb.shape, (n_bins, 6))

        trends = fit_poisson_covariate_trends(
            counts_concave_pcr, sf, exposure, usable, basis_500kb
        )
        self.assertGreater(float(trends[bin_a, 0]), float(trends[bin_b, 0]) * 1.15)

        basis_5kb = build_exogenous_covariate_basis(
            usable, bin_size=5_000, gc=gc_mean, gc_quantiles=gc_q
        )
        trends_5kb = fit_poisson_covariate_trends(
            counts_concave_pcr, sf, exposure, usable, basis_5kb
        )
        self.assertAlmostEqual(
            float(trends_5kb[bin_a, 0]), float(trends_5kb[bin_b, 0]), places=4
        )

    def test_mosaic_cnv_with_short_fragments_is_not_regressed_out(self) -> None:
        sig = inspect.signature(build_exogenous_covariate_basis)
        self.assertNotIn("mean_frag_len", sig.parameters)
        self.assertNotIn("endpoint_ratio", sig.parameters)
        self.assertIn("ref_frag_len", sig.parameters)
        self.assertIn("ref_ep_rate", sig.parameters)

        rng = np.random.default_rng(77)
        n_bins = 2_000
        usable = np.ones(n_bins, dtype=np.bool_)
        gc = rng.uniform(0.32, 0.62, size=n_bins).astype(np.float32)
        mapp = rng.uniform(0.75, 1.0, size=n_bins).astype(np.float32)
        ref_frag_len = rng.normal(167.0, 2.5, size=n_bins).astype(np.float32)
        ref_ep_rate = rng.uniform(0.08, 0.16, size=n_bins).astype(np.float32)
        exposure = np.ones(n_bins, dtype=np.float32)

        baseline = (
            120.0
            * np.exp(-5.0 * np.square(gc.astype(np.float64) - 0.46))
            * (0.8 + 0.2 * mapp.astype(np.float64))
            * np.exp(0.015 * (ref_frag_len.astype(np.float64) - 167.0))
        )
        cn_mult = np.ones(n_bins, dtype=np.float64)
        cn_mult[500:750] = 1.20

        counts = rng.poisson(baseline * cn_mult)[:, np.newaxis].astype(np.float32)
        sf = np.array([1.0], dtype=np.float64)

        basis = build_exogenous_covariate_basis(
            usable,
            bin_size=5_000,
            gc=gc,
            mappability=mapp,
            ref_frag_len=ref_frag_len,
            ref_ep_rate=ref_ep_rate,
        )
        trends = fit_poisson_covariate_trends(counts, sf, exposure, usable, basis)
        ratios = counts[:, 0] / trends[:, 0]
        diploid = np.ones(n_bins, dtype=np.bool_)
        diploid[500:750] = False
        mosaic_ratio = float(np.median(ratios[500:750]) / np.median(ratios[diploid]))
        self.assertAlmostEqual(mosaic_ratio, 1.20, delta=0.03)

    def test_cnv_outlier_winsorization_resists_focal_amplicon_and_deletion(
        self,
    ) -> None:
        rng = np.random.default_rng(2026)
        n_bins = 8_000
        usable = np.ones(n_bins, dtype=np.bool_)
        gc = rng.uniform(0.30, 0.65, size=n_bins).astype(np.float32)
        exposure = np.ones(n_bins, dtype=np.float32)
        sf = np.array([1.0], dtype=np.float64)

        true_rate = 35.0 * np.exp(-7.0 * np.square(gc.astype(np.float64) - 0.45))
        counts = rng.poisson(true_rate).astype(np.float32)

        amp_bins = np.flatnonzero(gc > 0.58)[:200]
        del_bins = np.flatnonzero(gc < 0.36)[:200]
        counts[amp_bins] = rng.poisson(18.0 * true_rate[amp_bins]).astype(np.float32)
        counts[del_bins] = 0.0

        basis = build_exogenous_covariate_basis(usable, bin_size=5_000, gc=gc)
        trends = fit_poisson_covariate_trends(
            counts[:, np.newaxis], sf, exposure, usable, basis
        )

        altered = np.zeros(n_bins, dtype=np.bool_)
        altered[amp_bins] = True
        altered[del_bins] = True
        diploid_hi = (gc > 0.58) & ~altered
        diploid_lo = (gc < 0.36) & ~altered

        err_hi = abs(
            float(np.median(trends[diploid_hi, 0] / true_rate[diploid_hi])) - 1.0
        )
        err_lo = abs(
            float(np.median(trends[diploid_lo, 0] / true_rate[diploid_lo])) - 1.0
        )
        self.assertLess(err_hi, 0.06)
        self.assertLess(err_lo, 0.06)

    def test_empty_basis_strided_memmap_and_distinct_value_scan(self) -> None:
        self.assertEqual(_count_distinct_up_to(np.array([1.0, 2.0, 3.0, 4.0]), 5), 4)
        self.assertEqual(
            _count_distinct_up_to(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 5), 5
        )
        with_nan = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan])
        self.assertEqual(_count_distinct_up_to(with_nan, 5), 0)

        late_distinct = np.zeros(200_000, dtype=np.float64)
        late_distinct[-4:] = [0.1, 0.2, 0.3, 0.4]
        self.assertEqual(_count_distinct_up_to(late_distinct, 5), 5)

        two_valued = np.tile([0.5, 1.0], 100_000)
        self.assertEqual(_count_distinct_up_to(two_valued, 5), 2)

        rng = np.random.default_rng(9)
        n_bins = 1_000
        usable = np.ones(n_bins, dtype=np.bool_)
        usable[:50] = False
        exposure = rng.uniform(0.5, 1.0, size=n_bins).astype(np.float32)
        sf = np.array([0.8, 1.2], dtype=np.float64)
        true_rates = np.array([15.0, 25.0], dtype=np.float64)
        counts_k0 = rng.poisson(
            exposure[:, np.newaxis] * sf[np.newaxis, :] * true_rates[np.newaxis, :]
        ).astype(np.float32)
        counts_k0[100:120, 0] = 500.0

        empty_basis = build_exogenous_covariate_basis(
            usable, bin_size=5_000, gc=np.tile([0.4, 0.5], n_bins // 2)
        )
        self.assertEqual(empty_basis.shape, (950, 0))
        trends_k0 = fit_poisson_covariate_trends(
            counts_k0, sf, exposure, usable, empty_basis
        )
        np.testing.assert_array_equal(trends_k0[~usable], 0.0)
        np.testing.assert_allclose(
            trends_k0[usable, 0] / exposure[usable], 15.0, rtol=0.05
        )
        np.testing.assert_allclose(
            trends_k0[usable, 1] / exposure[usable], 25.0, rtol=0.05
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            mmap_path = Path(tmpdir) / "counts.mmap"
            b_big, s_big = 70_000, 4
            mmap_counts = np.memmap(
                mmap_path, dtype=np.float32, mode="w+", shape=(b_big, s_big), order="F"
            )
            gc_big = rng.uniform(0.30, 0.65, size=b_big).astype(np.float32)
            usable_big = np.ones(b_big, dtype=np.bool_)
            exp_big = np.ones(b_big, dtype=np.float32)
            sf_big = np.array([0.85, 0.95, 1.05, 1.15], dtype=np.float64)
            rate_big = 4.0 * np.exp(-6.0 * np.square(gc_big.astype(np.float64) - 0.45))
            mmap_counts[:] = rng.poisson(
                rate_big[:, np.newaxis] * sf_big[np.newaxis, :]
            ).astype(np.float32)

            basis_big = build_exogenous_covariate_basis(
                usable_big, bin_size=1_000, gc=gc_big
            )
            t_seq = fit_poisson_covariate_trends(
                mmap_counts,
                sf_big,
                exp_big,
                usable_big,
                basis_big,
                max_fit_bins=16_384,
                threads=1,
            )
            t_par = fit_poisson_covariate_trends(
                mmap_counts,
                sf_big,
                exp_big,
                usable_big,
                basis_big,
                max_fit_bins=16_384,
                threads=4,
            )
            np.testing.assert_array_equal(t_seq, t_par)

        with self.assertRaises(InvalidInputError):
            build_exogenous_covariate_basis(usable, bin_size=0)
        with self.assertRaises(InvalidInputError):
            fit_poisson_covariate_trends(
                counts_k0, np.array([-1.0, 1.0]), exposure, usable, empty_basis
            )

    def test_multithreaded_fit_preserves_global_warnings_filters(self) -> None:
        rng = np.random.default_rng(314)
        n_bins, n_samples = 4_000, 16
        usable = np.ones(n_bins, dtype=np.bool_)
        gc = rng.uniform(0.30, 0.65, size=n_bins).astype(np.float32)
        exposure = np.ones(n_bins, dtype=np.float32)
        sf = np.ones(n_samples, dtype=np.float64)
        rate = 12.0 * np.exp(-6.0 * np.square(gc.astype(np.float64) - 0.45))
        counts = rng.poisson(rate[:, np.newaxis] * sf[np.newaxis, :]).astype(np.float32)
        basis = build_exogenous_covariate_basis(usable, bin_size=5_000, gc=gc)

        filters_before = list(warnings.filters)
        try:
            for _ in range(3):
                fit_poisson_covariate_trends(
                    counts, sf, exposure, usable, basis, threads=4
                )
            self.assertEqual(list(warnings.filters), filters_before)
        finally:
            warnings.filters[:] = filters_before


class TestCovariateRegression(unittest.TestCase):
    def test_poisson_covariate_trends_removes_wave_and_preserves_cnv(self) -> None:
        rng = np.random.default_rng(42)
        n_bins = 200
        gc = rng.uniform(0.30, 0.65, size=n_bins)
        mappability = rng.uniform(0.65, 1.0, size=n_bins)
        usable = np.ones(n_bins, dtype=np.bool_)

        g_centered = (gc - 0.45) / 0.10
        wave = (
            1.0 + 0.25 * g_centered - 0.15 * (g_centered**2) + 0.05 * (g_centered**3)
        ) * (0.7 + 0.3 * mappability)

        true_cn = np.ones(n_bins, dtype=np.float64)
        true_cn[80:95] = 1.5
        y = 1000.0 * true_cn * wave

        basis = build_exogenous_covariate_basis(
            usable, bin_size=100, gc=gc, mappability=mappability
        )
        trends = fit_poisson_covariate_trends(
            y,
            np.array([1.0], dtype=np.float64),
            np.ones(n_bins, dtype=np.float64),
            usable,
            basis,
        )
        corrected = y / np.maximum(trends[:, 0], 1e-12)

        neutral = np.ones(n_bins, dtype=np.bool_)
        neutral[80:95] = False
        raw_cv = float(np.std(y[neutral]) / np.mean(y[neutral]))
        corr_cv = float(np.std(corrected[neutral]) / np.mean(corrected[neutral]))
        self.assertLess(corr_cv, raw_cv * 0.15)

        gain_ratio = float(np.median(corrected[80:95]) / np.median(corrected[neutral]))
        self.assertAlmostEqual(gain_ratio, 1.5, delta=0.05)

    def test_covariate_features_standardizes_columns(self) -> None:
        usable = np.array([True, True, True, False, True])
        gc = np.array([0.3, 0.4, 0.5, 0.9, 0.6])
        mappability = np.array([0.7, 0.8, 0.9, 0.1, 1.0])
        feats = covariate_features(usable, gc=gc, mappability=mappability)
        self.assertIsNotNone(feats)
        if feats is None:
            self.fail("expected non-None covariate features")
        self.assertEqual(feats.shape, (4, 2))
        np.testing.assert_allclose(np.mean(feats, axis=0), [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(np.std(feats, axis=0), [1.0, 1.0], atol=1e-12)

        ref_frag_len = np.array([162.0, 165.0, 168.0, 199.0, 171.0])
        ref_ep_rate = np.array([0.09, 0.11, 0.13, 0.50, 0.15])
        feats_exog = covariate_features(
            usable,
            gc=gc,
            mappability=mappability,
            ref_frag_len=ref_frag_len,
            ref_ep_rate=ref_ep_rate,
        )
        self.assertIsNotNone(feats_exog)
        if feats_exog is None:
            self.fail("expected non-None exogenous covariate features")
        self.assertEqual(feats_exog.shape, (4, 4))
        np.testing.assert_allclose(np.mean(feats_exog, axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.std(feats_exog, axis=0), 1.0, atol=1e-12)

    def test_exogenous_covariate_basis_captures_gc_mappability_interaction(
        self,
    ) -> None:
        rng = np.random.default_rng(42)
        n = 200
        gc = rng.uniform(0.25, 0.75, size=n)
        mapp = rng.uniform(0.55, 1.0, size=n)
        usable = np.ones(n, dtype=np.bool_)
        basis = build_exogenous_covariate_basis(
            usable, bin_size=100, gc=gc, mappability=mapp
        )
        self.assertGreater(basis.shape[1], 0)

        z_gc = (gc - np.mean(gc)) / np.std(gc)
        z_map = (mapp - np.mean(mapp)) / np.std(mapp)
        true_bias = np.exp(0.25 * np.sin(z_gc) + 0.15 * z_map + 0.12 * (z_gc * z_map))
        raw = 100.0 * true_bias
        raw[:15] *= 2.0

        trends = fit_poisson_covariate_trends(
            raw,
            np.array([1.0], dtype=np.float64),
            np.ones(n, dtype=np.float64),
            usable,
            basis,
        )
        corrected = raw / np.maximum(trends[:, 0], 1e-12)
        diploid_cv_before = np.std(raw[15:]) / np.mean(raw[15:])
        diploid_cv_after = np.std(corrected[15:]) / np.mean(corrected[15:])
        self.assertLess(diploid_cv_after, diploid_cv_before * 0.25)
        self.assertAlmostEqual(
            float(np.median(corrected[:15]) / np.median(corrected[15:])),
            2.0,
            delta=0.18,
        )

    def test_exogenous_covariate_basis_with_fragmentomics(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        usable = np.ones(n, dtype=bool)
        gc = rng.uniform(0.3, 0.6, size=n)
        mappability = rng.uniform(0.7, 1.0, size=n)
        ref_frag_len = rng.uniform(140.0, 180.0, size=n).astype(np.float32)
        ref_ep_rate = rng.uniform(0.5, 3.0, size=n).astype(np.float32)

        basis_base = build_exogenous_covariate_basis(
            usable,
            bin_size=5_000,
            gc=gc,
            mappability=mappability,
        )
        n_cols_base = basis_base.shape[1]
        self.assertGreater(n_cols_base, 0)

        basis_aug = build_exogenous_covariate_basis(
            usable,
            bin_size=5_000,
            gc=gc,
            mappability=mappability,
            ref_frag_len=ref_frag_len,
            ref_ep_rate=ref_ep_rate,
        )
        self.assertGreater(basis_aug.shape[1], n_cols_base)
        self.assertEqual(basis_aug.shape[0], n)

        basis_const = build_exogenous_covariate_basis(
            usable,
            bin_size=5_000,
            gc=gc,
            mappability=mappability,
            ref_frag_len=np.full(n, 166.0, dtype=np.float32),
            ref_ep_rate=np.zeros(n, dtype=np.float32),
        )
        self.assertEqual(basis_const.shape[1], n_cols_base)

    def test_covariate_features_and_signal_space_features(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        usable = np.ones(n, dtype=bool)
        gc = rng.uniform(0.3, 0.6, size=n)
        mappability = rng.uniform(0.7, 1.0, size=n)
        mean_frag_len = rng.uniform(140.0, 180.0, size=n)
        endpoint_ratio = rng.uniform(0.5, 3.0, size=n)

        cov_mat_4 = covariate_features(
            usable,
            gc=gc,
            mappability=mappability,
            ref_frag_len=mean_frag_len,
            ref_ep_rate=endpoint_ratio,
        )
        self.assertIsNotNone(cov_mat_4)
        if cov_mat_4 is None:
            self.fail("expected non-None 4-column covariate matrix")
        self.assertEqual(cov_mat_4.shape, (n, 4))

        p_rank = 5
        pca_loadings = rng.standard_normal((p_rank, n))
        s_vals = np.linspace(10.0, 1.0, p_rank)

        feats_p4 = build_whitened_subspace_features(
            pca_loadings,
            s_vals,
            0.5,
            covariate_mat=cov_mat_4,
        )
        self.assertEqual(feats_p4.shape, (n, p_rank + 4))

    def test_poisson_covariates_and_robust_stats(self) -> None:
        rng = np.random.default_rng(7)
        usable = np.ones(40, dtype=bool)
        gc_q = np.tile(np.linspace(0.30, 0.65, 8, dtype=np.float32), (40, 1))
        gc_q[:, 3] = np.linspace(0.35, 0.60, 40, dtype=np.float32)
        basis = build_exogenous_covariate_basis(
            usable,
            bin_size=5000,
            gc=np.linspace(0.38, 0.55, 40),
            gc_quantiles=gc_q,
            mappability=np.linspace(0.7, 1.0, 40),
        )
        self.assertEqual(basis.shape[0], 40)
        self.assertGreater(basis.shape[1], 0)

        counts = rng.poisson(120.0, size=(40, 2)).astype(np.float32)
        mult = fit_poisson_covariate_trends(
            counts,
            np.array([1.0, 1.0], dtype=np.float64),
            np.ones(40, dtype=np.float64),
            usable,
            basis,
        )
        self.assertEqual(mult.shape, (40, 2))
        self.assertTrue(bool(np.all(mult > 0.0)))

        z_col = standardize_column(np.linspace(1.0, 5.0, 20))
        self.assertIsNotNone(z_col)
        if z_col is None:
            self.fail("expected non-None standardized column")
        self.assertAlmostEqual(float(np.mean(z_col)), 0.0, places=6)
        self.assertAlmostEqual(
            half_sample_mode(rng.normal(2.0, 0.01, 50)), 2.0, delta=0.05
        )
        self.assertGreater(mad_scale(np.array([1.0, 2.0, 3.0, 4.0])), 0.0)

    def test_fit_logit_linear_ensemble_handles_partial_nan_targets(self) -> None:
        rng = np.random.default_rng(19)
        x_mat = rng.normal(size=(12, 3))
        true_ff = np.clip(0.12 + 0.06 * x_mat[:, 0], 0.02, 0.40)
        y_partial = true_ff.copy()
        y_partial[8:] = np.nan
        fit = fit_logit_linear_ensemble(x_mat, y_partial, standardize=True, seed=0)
        self.assertGreater(float(np.linalg.norm(fit.weights)), 0.05)


if __name__ == "__main__":
    unittest.main()
