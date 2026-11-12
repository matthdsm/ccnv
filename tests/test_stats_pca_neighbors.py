"""Unit tests for variance-whitened PCA, Prototype Graph neighbors, and EB dispersion."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ccnv.models.exceptions import InvalidInputError
from ccnv.stats.neighbors import (
    _resolve_prototype_weights,
    build_whitened_subspace_features,
    calibrate_prototype_dispersion,
    evaluate_loco_prototype_baseline,
    fit_prototype_graph,
    squeeze_var,
)
from ccnv.stats.pca import (
    WhitenedPCAResult,
    fit_whitened_pca,
    pca_irlba_numpy,
    project_huber_pca_wave,
    reconstruct_whitened_pca_expected,
)
from ccnv.stats.regression import hsm_biweight_mode


class TestWhitenedPCA(unittest.TestCase):
    def test_small_sample_counts_select_rank_zero_cleanly(self) -> None:
        usable = np.array([True, True, True, False, True], dtype=np.bool_)
        for n_samples in (1, 2):
            counts = np.full((5, n_samples), 12.0, dtype=np.float32)
            expected = np.full((5, n_samples), 10.0, dtype=np.float32)
            res = fit_whitened_pca(counts, expected, usable)
            self.assertEqual(res.loadings.shape, (0, 4))
            self.assertEqual(res.singular_values.shape, (0,))
            self.assertTrue(bool(np.all(res.mean_rate[usable] > 0.0)))
            self.assertEqual(float(res.mean_rate[3]), 0.0)

    def test_max_components_zero_forces_rank_zero(self) -> None:
        rng = np.random.default_rng(19)
        usable = np.ones(120, dtype=np.bool_)
        grid = np.linspace(0.0, 2.0 * np.pi, 120)
        expected = np.full((120, 12), 25.0, dtype=np.float32)
        counts = rng.poisson(
            25.0 * np.exp(np.outer(np.sin(grid), rng.normal(0.0, 0.3, size=12)))
        ).astype(np.float32)
        res = fit_whitened_pca(counts, expected, usable, max_components=0)
        self.assertEqual(res.loadings.shape, (0, 120))
        self.assertEqual(res.singular_values.shape, (0,))

    def test_deterministic_sign_canonicalization(self) -> None:
        rng = np.random.default_rng(23)
        usable = np.ones(300, dtype=np.bool_)
        grid = np.linspace(0.0, 2.0 * np.pi, 300)
        expected = np.full((300, 20), 30.0, dtype=np.float32)
        counts = rng.poisson(
            30.0 * np.exp(np.outer(np.cos(grid), rng.normal(0.0, 0.25, size=20)))
        ).astype(np.float32)
        r1 = fit_whitened_pca(counts, expected, usable)
        r2 = fit_whitened_pca(counts.copy(), expected.copy(), usable.copy())
        self.assertEqual(r1.loadings.shape, (1, 300))
        np.testing.assert_allclose(r1.loadings, r2.loadings, atol=1e-7)

    def test_invalid_inputs_raise_invalid_input_error(self) -> None:
        with self.assertRaises(InvalidInputError):
            fit_whitened_pca(
                np.ones((4, 3), dtype=np.float32),
                np.ones((5, 3), dtype=np.float32),
                np.ones(4, dtype=np.bool_),
            )
        with self.assertRaises(InvalidInputError):
            fit_whitened_pca(
                np.ones((4, 3), dtype=np.float32),
                np.ones((4, 3), dtype=np.float32),
                np.zeros(4, dtype=np.bool_),
            )

    def test_pca_irlba_numpy_preserved_for_prenatal(self) -> None:
        rng = np.random.default_rng(5)
        x = rng.normal(size=(16, 40))
        scores, center, rotation = pca_irlba_numpy(x, n_components=3)
        self.assertEqual(scores.shape, (16, 3))
        self.assertEqual(center.shape, (40,))
        self.assertEqual(rotation.shape, (40, 3))

    def test_fit_whitened_pca_and_huber_projection(self) -> None:
        rng = np.random.default_rng(42)
        n_bins, n_samples = 60, 12
        mu_base = np.full(n_bins, 120.0, dtype=np.float64)
        size_factors = rng.uniform(0.8, 1.2, size=n_samples)
        expected_in = (mu_base[:, None] * size_factors[None, :]).astype(np.float32)
        u = rng.normal(0.0, 0.25, size=(n_bins, 2))
        v = rng.normal(0.0, 1.0, size=(2, n_samples))
        counts = rng.poisson(expected_in.astype(np.float64) * np.exp(u @ v)).astype(
            np.float32
        )
        usable = np.ones(n_bins, dtype=bool)

        res = fit_whitened_pca(
            counts,
            expected_in,
            usable,
            max_components=5,
        )
        self.assertEqual(res.mean_rate.shape, (n_bins,))
        self.assertGreaterEqual(res.loadings.shape[0], 1)
        self.assertEqual(res.loadings.shape[1], n_bins)

        recon = reconstruct_whitened_pca_expected(expected_in, usable, res)
        self.assertEqual(recon.shape, (n_bins, n_samples))
        self.assertTrue(bool(np.all(recon > 0.0)))

        log_sig = np.log(
            (counts[:, 0] + 0.5) / (expected_in[:, 0] * res.mean_rate + 0.5)
        )
        coef, wave = project_huber_pca_wave(log_sig.astype(np.float64), res.loadings)
        self.assertEqual(coef.shape, (res.loadings.shape[0],))
        self.assertEqual(wave.shape, (n_bins,))
        self.assertTrue(bool(np.all(np.isfinite(wave))))

    def test_fit_whitened_pca_invalid_shapes(self) -> None:
        with self.assertRaises(InvalidInputError):
            fit_whitened_pca(
                np.empty((0, 0), dtype=np.float32),
                np.empty((0, 0), dtype=np.float32),
                np.empty(0, dtype=bool),
            )
        with self.assertRaises(InvalidInputError):
            project_huber_pca_wave(
                np.zeros(4, dtype=np.float64),
                np.zeros((2, 5), dtype=np.float32),
            )

    def test_pca_irlba_numpy_matches_expected_shapes(self) -> None:
        rng = np.random.default_rng(123)
        mat = rng.normal(size=(20, 15))
        scores, center, rotation = pca_irlba_numpy(mat, n_components=3, seed=42)
        self.assertEqual(scores.shape, (20, 3))
        self.assertEqual(center.shape, (15,))
        self.assertEqual(rotation.shape, (15, 3))

    def test_pure_poisson_cohort_selects_rank_zero_and_cnp_does_not_leak_into_loadings(
        self,
    ) -> None:
        rng = np.random.default_rng(20261111)
        n_bins, n_samples = 2000, 40
        usable = np.ones(n_bins, dtype=np.bool_)
        usable[:40] = False
        u_idx = np.flatnonzero(usable)

        mu_true = np.ones(n_bins, dtype=np.float64)
        mu_true[usable] = rng.uniform(0.65, 1.45, size=u_idx.size)
        sf = rng.uniform(0.75, 1.35, size=n_samples)
        base_depth = 25.0
        expected = np.broadcast_to(
            (base_depth * sf)[np.newaxis, :], (n_bins, n_samples)
        ).astype(np.float32)
        lam_pure = expected.astype(np.float64) * mu_true[:, np.newaxis]
        poisson_only_counts = rng.poisson(lam_pure).astype(np.float32)

        pca_pure = fit_whitened_pca(poisson_only_counts, expected, usable)
        self.assertIsInstance(pca_pure, WhitenedPCAResult)
        self.assertEqual(pca_pure.loadings.shape, (0, u_idx.size))
        self.assertEqual(pca_pure.singular_values.shape, (0,))
        self.assertGreater(pca_pure.mp_sigma, 0.0)
        self.assertTrue(bool(np.all(pca_pure.mean_rate[~usable] == 0.0)))
        rel_err = np.median(
            np.abs(pca_pure.mean_rate[usable] - mu_true[usable]) / mu_true[usable]
        )
        self.assertLess(float(rel_err), 0.05)

        grid = np.linspace(0.0, 2.0 * np.pi, u_idx.size, dtype=np.float64)
        w1 = np.sin(2.0 * grid)
        w2 = np.cos(3.0 * grid)
        a1 = rng.normal(0.0, 0.18, size=n_samples)
        a2 = rng.normal(0.0, 0.14, size=n_samples)
        log_wave = np.zeros((n_bins, n_samples), dtype=np.float64)
        log_wave[usable] = np.outer(w1, a1) + np.outer(w2, a2)

        lam_waves = lam_pure * np.exp(log_wave)
        cnp_bins = u_idx[200:240]
        lam_waves[np.ix_(cnp_bins, np.array([0, 1, 2]))] *= 0.01
        lam_waves[np.ix_(cnp_bins, np.array([3, 4, 5]))] *= 0.50
        cnp_plus_waves_counts = rng.poisson(lam_waves).astype(np.float32)

        pca_cnp = fit_whitened_pca(cnp_plus_waves_counts, expected, usable)
        self.assertEqual(pca_cnp.loadings.shape, (2, u_idx.size))
        norms = np.linalg.norm(pca_cnp.loadings.astype(np.float64), axis=1)
        np.testing.assert_allclose(norms, np.ones(2), atol=1e-5)
        self.assertLess(
            float(np.max(np.abs(pca_cnp.loadings[:, 200:240]))),
            5.0 / float(np.sqrt(usable.sum())),
        )

    def test_memmap_chunked_pca_and_wide_sample_randomized_svd(self) -> None:
        rng = np.random.default_rng(77)
        n_bins, n_samples = 36_000, 14
        usable = np.ones(n_bins, dtype=np.bool_)
        usable[::50] = False
        u_count = int(np.count_nonzero(usable))

        grid = np.linspace(0.0, 4.0 * np.pi, n_bins, dtype=np.float64)
        amp = rng.normal(0.0, 0.16, size=n_samples)
        expected = np.full((n_bins, n_samples), 20.0, dtype=np.float32)
        counts = rng.poisson(20.0 * np.exp(np.outer(np.sin(grid), amp))).astype(
            np.float32
        )

        with tempfile.TemporaryDirectory() as tmp:
            mm_path = Path(tmp) / "counts.dat"
            mm = np.memmap(mm_path, dtype=np.float32, mode="w+", shape=counts.shape)
            mm[:] = counts
            mm.flush()
            mm_ro = np.memmap(mm_path, dtype=np.float32, mode="r", shape=counts.shape)

            res_mem = fit_whitened_pca(counts, expected, usable)
            res_mm = fit_whitened_pca(mm_ro, expected, usable)

        self.assertEqual(res_mem.loadings.shape, (1, u_count))
        np.testing.assert_allclose(res_mm.mean_rate, res_mem.mean_rate, rtol=1e-6)
        np.testing.assert_allclose(res_mm.loadings, res_mem.loadings, atol=1e-6)
        np.testing.assert_allclose(
            res_mm.singular_values, res_mem.singular_values, rtol=1e-6
        )

        b_small, s_large = 80, 120
        usable_s = np.ones(b_small, dtype=np.bool_)
        usable_s[:10] = False
        exp_s = np.full((b_small, s_large), 30.0, dtype=np.float32)
        pure_s = rng.poisson(exp_s).astype(np.float32)
        pca_pure_wide = fit_whitened_pca(pure_s, exp_s, usable_s)
        self.assertEqual(pca_pure_wide.loadings.shape[0], 0)

        g_s = np.linspace(0.0, 2.0 * np.pi, b_small, dtype=np.float64)
        wave_s = np.outer(np.sin(g_s), rng.normal(0.0, 0.25, size=s_large)) + np.outer(
            np.cos(2.0 * g_s), rng.normal(0.0, 0.20, size=s_large)
        )
        wavy_s = rng.poisson(30.0 * np.exp(wave_s)).astype(np.float32)
        pca_wavy_wide = fit_whitened_pca(wavy_s, exp_s, usable_s)
        self.assertEqual(pca_wavy_wide.loadings.shape, (2, 70))

    def test_project_huber_pca_wave_single_whitening_recovery(self) -> None:
        rng = np.random.default_rng(101)
        n_bins, n_samples = 320, 24
        e_0 = np.geomspace(0.1, 50.0, n_bins)
        w_scale = np.sqrt(e_0)
        t_grid = np.linspace(0.0, 2.0 * math.pi, n_bins, endpoint=False)
        q_orth, _ = np.linalg.qr(np.vstack([np.sin(t_grid), np.cos(2.0 * t_grid)]).T)
        amp = rng.normal(0.0, 6.0, size=(2, n_samples))
        amp -= np.mean(amp, axis=1, keepdims=True)
        eta_cohort = (q_orth @ amp) / np.sqrt(e_0[:, np.newaxis] + 0.5)
        expected = np.broadcast_to(e_0[:, np.newaxis], (n_bins, n_samples)).copy()
        counts = np.maximum(
            expected * np.exp(eta_cohort) + rng.normal(0.0, 0.02, expected.shape), 0.0
        )

        pca_fit = fit_whitened_pca(
            counts, expected, np.ones(n_bins, dtype=np.bool_), max_components=2
        )
        self.assertEqual(pca_fit.loadings.shape[0], 2)
        beta_true = np.array([2.4, -1.6], dtype=np.float64)
        true_wave = (pca_fit.loadings.astype(np.float64).T @ beta_true) / w_scale
        log_signal = true_wave.copy()
        log_signal[20:24] += 1.5

        coef_hat, wave_hat = project_huber_pca_wave(
            log_signal,
            pca_fit.loadings,
            whitening_scale=w_scale,
            eval_loadings=pca_fit.loadings,
            eval_whitening_scale=w_scale,
        )
        np.testing.assert_allclose(coef_hat, beta_true, rtol=0.04)
        np.testing.assert_allclose(wave_hat, true_wave, rtol=0.04)

    def test_pca_rank_cap_and_near_zero_singular_value_filtering(self) -> None:
        counts = np.full((30, 12), 100.0, dtype=np.float64)
        expected = np.full((30, 12), 100.0, dtype=np.float64)
        usable = np.ones(30, dtype=np.bool_)
        fit = fit_whitened_pca(counts, expected, usable, max_components=4)
        self.assertEqual(fit.singular_values.size, 0)
        self.assertEqual(fit.loadings.shape, (0, 30))

    def test_project_huber_pca_wave_empty_loadings_and_masked_whitening_scale(
        self,
    ) -> None:
        coef0, wave0 = project_huber_pca_wave(
            np.array([0.1, -0.1, 0.2]),
            np.empty((0, 0), dtype=np.float64),
        )
        self.assertEqual(coef0.shape, (0,))
        self.assertEqual(wave0.shape, (3,))

        sig = np.array([0.2, -0.2, 0.4, -0.4, 0.2, 0.0], dtype=np.float64)
        ld = np.array([[0.5, -0.5, 0.5, -0.5, 0.5, 0.0]], dtype=np.float64)
        w_scale = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 0.0], dtype=np.float64)
        coef, wave = project_huber_pca_wave(sig, ld, whitening_scale=w_scale)
        self.assertEqual(coef.shape, (1,))
        self.assertEqual(wave.shape, (6,))
        self.assertEqual(wave[5], 0.0)
        self.assertTrue(bool(np.all(np.isfinite(wave))))

        with self.assertRaises(InvalidInputError):
            project_huber_pca_wave(
                sig,
                ld,
                eval_loadings=ld,
                whitening_scale=w_scale,
                eval_whitening_scale=None,
            )


class TestPrototypeNeighbors(unittest.TestCase):
    def test_fit_prototype_graph_and_loco_dispersion(self) -> None:
        rng = np.random.default_rng(42)
        n_bins, n_samples = 40, 10
        contig_ids = np.repeat(np.arange(4, dtype=np.int16), 10)
        usable = np.ones(n_bins, dtype=bool)
        features = rng.normal(size=(n_bins, 3)).astype(np.float32)
        proto_ids, proto_weights, mean_dist = fit_prototype_graph(
            features,
            max_prototypes=12,
            top_r=4,
        )
        self.assertEqual(proto_ids.shape, (n_bins, 4))
        self.assertEqual(proto_weights.shape, (n_bins, 4))
        self.assertEqual(mean_dist.shape, (n_bins,))

        mu_hat = np.full((n_bins, n_samples), 150.0, dtype=np.float32)
        counts = rng.poisson(mu_hat).astype(np.float32)
        null_mean, bio_disp = calibrate_prototype_dispersion(
            counts,
            mu_hat,
            contig_ids,
            usable,
            proto_ids,
            proto_weights,
            pca_rank=1,
        )
        self.assertEqual(null_mean.shape, (n_bins,))
        self.assertEqual(bio_disp.shape, (n_bins,))
        self.assertTrue(bool(np.all(bio_disp >= 0.0)))

        baseline, support = evaluate_loco_prototype_baseline(
            (counts[:, 0] / 150.0).astype(np.float64),
            contig_ids,
            proto_ids,
            proto_weights,
        )
        self.assertEqual(baseline.shape, (n_bins,))
        self.assertEqual(support.shape, (n_bins,))

    def test_squeeze_var_matches_smyth_2004_closed_form(self) -> None:
        rng = np.random.default_rng(42)
        d0_true, s0_sq_true, df = 8.0, 0.04, 10.0
        true_sigma2 = (d0_true * s0_sq_true) / rng.chisquare(df=d0_true, size=2000)
        sample_var = true_sigma2 * (rng.chisquare(df=df, size=2000) / df)

        var_post, d0_hat, s0_sq_hat = squeeze_var(sample_var, df=df)
        self.assertAlmostEqual(s0_sq_hat, s0_sq_true, delta=0.008)
        self.assertAlmostEqual(d0_hat, d0_true, delta=2.5)
        expected_post = (d0_hat * s0_sq_hat + df * sample_var) / (d0_hat + df)
        np.testing.assert_allclose(var_post, expected_post, rtol=1e-12)

    def test_build_whitened_subspace_features(self) -> None:
        loadings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        s_vals = np.array([2.0, 1.0], dtype=np.float64)
        cov = np.array([[0.1], [0.2], [0.3]], dtype=np.float64)
        feats = build_whitened_subspace_features(
            loadings, s_vals, 0.5, covariate_mat=cov
        )
        self.assertEqual(feats.shape, (3, 3))
        self.assertTrue(bool(np.all(np.isfinite(feats))))

    def test_single_contig_cluster_on_aneuploid_contig_uses_secondary_off_contig_prototypes(
        self,
    ) -> None:
        pids = np.array(
            [
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
            ],
            dtype=np.uint16,
        )
        weights = np.array(
            [
                [0.8, 0.2, 0.0, 0.0],
                [0.8, 0.2, 0.0, 0.0],
                [0.8, 0.2, 0.0, 0.0],
                [0.8, 0.2, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
            ],
            dtype=np.float16,
        )
        contigs = np.array([0, 0, 1, 1, 2, 2, 2], dtype=np.int32)
        vals = np.array([1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5], dtype=np.float64)

        baselines, support = evaluate_loco_prototype_baseline(
            vals, contigs, pids, weights
        )
        np.testing.assert_allclose(baselines[4:], 1.0)
        self.assertTrue(bool(np.all(support[4:] == 4)))

    def test_singleton_float16_zero_secondary_weights_preserve_cnv_and_null_variance(
        self,
    ) -> None:
        rng = np.random.default_rng(42)
        feats = np.vstack(
            [
                rng.normal(0.0, 0.05, size=(16, 3)),
                np.array([[5000.0, -5000.0, 5000.0]]),
            ]
        ).astype(np.float32)
        pids, pweights, _ = fit_prototype_graph(feats, max_prototypes=8, top_r=4)

        self.assertEqual(pweights[16, 0], np.float16(1.0))
        np.testing.assert_array_equal(pweights[16, 1:], np.float16(0.0))

        contigs = np.array([0] * 8 + [1] * 8 + [2], dtype=np.int32)
        vals = np.array([1.0] * 16 + [2.0], dtype=np.float64)
        baselines, support = evaluate_loco_prototype_baseline(
            vals, contigs, pids, pweights
        )
        self.assertAlmostEqual(float(baselines[16]), 1.0, places=6)
        self.assertGreater(int(support[16]), 0)

        exp = np.full((17, 12), 100.0, dtype=np.float32)
        eta = rng.normal(0.0, 0.12, size=(17, 12))
        counts = rng.poisson(exp * np.exp(eta)).astype(np.float32)
        null_mean, bio_disp = calibrate_prototype_dispersion(
            counts,
            exp,
            contigs.astype(np.int16),
            np.ones(17, dtype=bool),
            pids,
            pweights,
            pca_rank=0,
        )
        self.assertTrue(bool(np.isfinite(null_mean[16])))
        self.assertGreaterEqual(float(bio_disp[16]), 0.0)

    def test_all_four_candidates_same_contig_falls_back_to_global_off_contig_median(
        self,
    ) -> None:
        pids = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 2, 3, 4],
                [2, 1, 3, 4],
                [3, 1, 2, 4],
                [4, 1, 2, 3],
            ],
            dtype=np.uint16,
        )
        weights = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float16), (6, 1))
        contigs = np.array([0, 1, 2, 2, 2, 2], dtype=np.int32)
        vals = np.array([1.0, 1.0, 1.5, 1.5, 1.5, 1.5], dtype=np.float64)

        baselines, support = evaluate_loco_prototype_baseline(
            vals, contigs, pids, weights
        )
        np.testing.assert_allclose(baselines[2:], 1.0)
        self.assertTrue(bool(np.all(support[2:] == 2)))

    def test_single_contig_branch_excludes_self_singleton(self) -> None:
        pids = np.array([[0, 1, 0, 0], [1, 0, 0, 0]], dtype=np.uint16)
        weights = np.array(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float16
        )
        contigs = np.array([0, 0], dtype=np.int32)
        vals = np.array([1.0, 1.8], dtype=np.float64)

        baselines, support = evaluate_loco_prototype_baseline(
            vals, contigs, pids, weights
        )
        self.assertAlmostEqual(float(baselines[0]), 1.8, places=6)
        self.assertAlmostEqual(float(baselines[1]), 1.0, places=6)
        np.testing.assert_array_equal(support, [1, 1])

        pids_multi = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 0],
            ],
            dtype=np.uint16,
        )
        w_multi = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float16), (5, 1))
        c_multi = np.zeros(5, dtype=np.int32)
        v_multi = np.array([1.0, 1.0, 1.0, 1.0, 2.0], dtype=np.float64)
        b_multi, s_multi = evaluate_loco_prototype_baseline(
            v_multi, c_multi, pids_multi, w_multi
        )
        self.assertAlmostEqual(float(b_multi[4]), 1.0, places=6)
        self.assertEqual(int(s_multi[4]), 4)

    def test_calibrate_prototype_dispersion_small_cohort_dof_scaling(self) -> None:
        rng = np.random.default_rng(42)
        n_bins = 200
        n_samples = 10
        pca_rank = 3
        d_eb = n_samples - 1 - pca_rank
        ddof = 1.0 + pca_rank / n_samples
        expected_scale = (n_samples - ddof) / d_eb
        self.assertAlmostEqual(expected_scale, 1.45, places=6)

        feats = rng.normal(size=(n_bins, 2)).astype(np.float32)
        pids, pw, _ = fit_prototype_graph(feats, max_prototypes=16, top_r=4)
        contigs = np.array([0] * 100 + [1] * 100, dtype=np.int16)
        usable = np.ones(n_bins, dtype=bool)

        true_phi = 0.04
        lam = 100.0
        eta = rng.normal(-0.5 * true_phi, np.sqrt(true_phi), size=(n_bins, n_samples))
        exp = np.full((n_bins, n_samples), lam, dtype=np.float32)
        counts = rng.poisson(exp * np.exp(eta)).astype(np.float32)

        null_mean, disp = calibrate_prototype_dispersion(
            counts,
            exp,
            contigs,
            usable,
            pids,
            pw,
            pca_rank=pca_rank,
        )

        self.assertGreater(float(np.mean(disp[usable] > 0.0)), 0.90)
        med_disp = float(np.median(disp[usable]))
        self.assertGreater(med_disp, 0.025)
        self.assertLess(med_disp, 0.080)
        self.assertLess(float(np.median(np.abs(null_mean[usable]))), 0.15)

    def test_hsm_biweight_mode_finite_guard_and_empty(self) -> None:
        self.assertEqual(hsm_biweight_mode(np.array([], dtype=np.float64)), 0.0)
        self.assertEqual(
            hsm_biweight_mode(np.array([np.nan, np.nan], dtype=np.float64)), 0.0
        )
        self.assertEqual(
            hsm_biweight_mode(np.array([np.inf, -np.inf], dtype=np.float64)), 0.0
        )

        vals = np.array(
            [np.nan, 1.0, 1.02, 0.98, 1.01, 0.99, 1.0, np.nan, 5.0], dtype=np.float64
        )
        weights = np.ones_like(vals)
        m = hsm_biweight_mode(vals, weights)
        self.assertGreater(m, 0.98)
        self.assertLess(m, 1.02)

    def test_full_cohort_dispersion_uses_all_samples_and_is_depth_invariant(
        self,
    ) -> None:
        rng = np.random.default_rng(101)
        n_contigs, bins_per_contig = 6, 200
        n_bins = n_contigs * bins_per_contig
        n_samples = 150
        contigs = np.repeat(np.arange(n_contigs, dtype=np.int16), bins_per_contig)
        usable = np.ones(n_bins, dtype=np.bool_)
        usable[::40] = False
        u_idx = np.flatnonzero(usable)

        feats = rng.normal(0.0, 1.0, size=(u_idx.size, 2)).astype(np.float32)
        pids_u, pw_u, _ = fit_prototype_graph(feats, max_prototypes=32, top_r=4)

        exp_150 = np.full((n_bins, n_samples), 40.0, dtype=np.float32)
        tail_noise = np.zeros((n_bins, n_samples), dtype=np.float64)
        tail_noise[:200, 100:] = rng.normal(0.0, 0.28, size=(200, 50))
        tail_counts = rng.poisson(40.0 * np.exp(tail_noise)).astype(np.float32)
        _, disp_tail = calibrate_prototype_dispersion(
            tail_counts,
            exp_150,
            contigs,
            usable,
            pids_u,
            pw_u,
            pca_rank=0,
        )
        noisy_u = usable & (np.arange(n_bins) < 200)
        clean_u = usable & (np.arange(n_bins) >= 200)
        self.assertGreater(
            float(np.median(disp_tail[noisy_u])),
            float(np.median(disp_tail[clean_u])) + 0.01,
        )

        pure_counts = rng.poisson(exp_150).astype(np.float32)
        null_m_pure, disp_pure = calibrate_prototype_dispersion(
            pure_counts,
            exp_150,
            contigs,
            usable,
            pids_u,
            pw_u,
            pca_rank=0,
        )
        self.assertLess(float(np.median(disp_pure[usable])), 1e-3)
        self.assertLess(float(np.median(np.abs(null_m_pure[usable]))), 0.03)

        true_phi = np.zeros(n_bins, dtype=np.float64)
        true_phi[:400] = 0.0144
        disp_estimates: list[float] = []
        for lam_val in (30.0, 8.0):
            eta = rng.normal(
                -0.5 * true_phi[:, np.newaxis],
                np.sqrt(true_phi)[:, np.newaxis],
                size=(n_bins, n_samples),
            )
            exp_d = np.full((n_bins, n_samples), lam_val, dtype=np.float32)
            cnt_d = rng.poisson(lam_val * np.exp(eta)).astype(np.float32)
            _, disp_d = calibrate_prototype_dispersion(
                cnt_d,
                exp_d,
                contigs,
                usable,
                pids_u,
                pw_u,
                pca_rank=0,
            )
            disp_estimates.append(float(np.median(disp_d[usable & (true_phi > 0)])))
            self.assertLess(float(np.median(disp_d[usable & (true_phi == 0)])), 0.003)

        for est in disp_estimates:
            self.assertGreater(est, 0.0065)
            self.assertLess(est, 0.022)
        self.assertLess(abs(disp_estimates[0] - disp_estimates[1]), 0.006)

    def test_prototype_graph_fitting_edge_cases_and_loco_baseline(self) -> None:
        rng = np.random.default_rng(0)
        feats = rng.standard_normal((500, 4)).astype(np.float32)
        pids, pweights, mdist = fit_prototype_graph(feats, max_prototypes=2048, top_r=4)
        self.assertLess(int(pids.max()), 8)
        self.assertEqual(pids.shape, (500, 4))
        self.assertEqual(pids.dtype, np.uint16)
        self.assertEqual(pweights.shape, (500, 4))
        self.assertEqual(pweights.dtype, np.float16)
        self.assertEqual(mdist.shape, (500,))
        self.assertEqual(mdist.dtype, np.float32)
        np.testing.assert_allclose(
            pweights.astype(np.float32).sum(axis=1), 1.0, atol=5e-3
        )

        id_small, w_small, d_small = fit_prototype_graph(
            feats[:2], max_prototypes=2048, top_r=4
        )
        self.assertEqual(id_small.shape, (2, 4))
        self.assertEqual(id_small.dtype, np.uint16)
        self.assertEqual(w_small.shape, (2, 4))
        self.assertEqual(w_small.dtype, np.float16)
        self.assertEqual(d_small.shape, (2,))
        np.testing.assert_array_equal(id_small[:, 2:], 0)
        np.testing.assert_array_equal(w_small[:, 2:], 0.0)
        np.testing.assert_allclose(
            w_small[:, :2].astype(np.float32).sum(axis=1), 1.0, atol=5e-3
        )

        id_empty, w_empty, d_empty = fit_prototype_graph(
            np.empty((0, 3), dtype=np.float32), max_prototypes=2048, top_r=4
        )
        self.assertEqual(id_empty.shape, (0, 4))
        self.assertEqual(id_empty.dtype, np.uint16)
        self.assertEqual(w_empty.shape, (0, 4))
        self.assertEqual(w_empty.dtype, np.float16)
        self.assertEqual(d_empty.shape, (0,))

        loco_pids = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0]],
            dtype=np.uint16,
        )
        loco_w = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.5, 0.25, 0.0],
            ],
            dtype=np.float16,
        )
        contigs = np.array([0, 0, 1, 1, 2], dtype=np.int32)
        vals_gain0 = np.array([2.0, 2.0, 1.0, 1.0, 1.5], dtype=np.float64)
        baselines_g0, support_g0 = evaluate_loco_prototype_baseline(
            vals_gain0, contigs, loco_pids, loco_w
        )
        np.testing.assert_allclose(baselines_g0[:2], 1.0)
        self.assertTrue(bool(np.all(support_g0[:2] == 2)))

        vals_tri2 = np.array([1.0, 1.0, 1.0, 1.0, 1.5], dtype=np.float64)
        baselines_tri2, support_tri2 = evaluate_loco_prototype_baseline(
            vals_tri2, contigs, loco_pids, loco_w
        )
        self.assertAlmostEqual(float(baselines_tri2[4]), 1.0, places=6)
        self.assertEqual(int(support_tri2[4]), 4)

        singleton_feats = np.vstack(
            [
                rng.normal(0.0, 0.05, size=(12, 2)),
                np.array([[200.0, 200.0]]),
            ]
        ).astype(np.float32)
        s_pids, s_weights, _ = fit_prototype_graph(
            singleton_feats, max_prototypes=8, top_r=4
        )
        self.assertEqual(s_weights[12, 0], np.float16(1.0))
        np.testing.assert_array_equal(s_weights[12, 1:], np.float16(0.0))
        s_contigs = np.array([0] * 6 + [1] * 6 + [2], dtype=np.int32)
        s_vals = np.array([1.0] * 12 + [1.5], dtype=np.float64)
        s_base, s_sup = evaluate_loco_prototype_baseline(
            s_vals, s_contigs, s_pids, s_weights
        )
        self.assertAlmostEqual(float(s_base[12]), 1.0, places=6)
        self.assertGreater(int(s_sup[12]), 0)

    def test_calibrate_prototype_dispersion_smyth_squeeze_var_moderation(self) -> None:
        rng = np.random.default_rng(202)
        n_bins, n_samples, phi_true, lam = 500, 20, 0.02, 2000.0
        gamma_rates = rng.gamma(
            1.0 / phi_true, phi_true * lam, size=(n_bins, n_samples)
        )
        counts = rng.poisson(gamma_rates).astype(np.float64)
        recon_expected = np.full((n_bins, n_samples), lam, dtype=np.float32)
        contigs = np.repeat(np.arange(5, dtype=np.int64), 100)
        usable = np.ones(n_bins, dtype=np.bool_)
        pids = np.zeros((n_bins, 4), dtype=np.uint16)
        pw = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float16), (n_bins, 1))

        _, d0_hat, _ = squeeze_var(
            rng.chisquare(19.0, size=n_bins) * (phi_true / 19.0), df=19.0
        )
        self.assertGreater(d0_hat, 10.0)

        _, bio_disp = calibrate_prototype_dispersion(
            counts, recon_expected, contigs, usable, pids, pw, pca_rank=0
        )
        self.assertEqual(int(np.count_nonzero(bio_disp <= 0.0)), 0)
        self.assertAlmostEqual(float(np.median(bio_disp)), phi_true, delta=0.005)
        self.assertLess(
            float(np.percentile(bio_disp, 75) - np.percentile(bio_disp, 25)), 0.006
        )

    def test_neighbors_duplicate_proto_id_pooling_and_zero_variance_squeeze(
        self,
    ) -> None:
        pids = np.array([[2, 2, 5, 2]], dtype=np.intp)
        pw = np.array([[0.25, 0.35, 0.30, 0.10]], dtype=np.float64)
        valid = np.ones_like(pids, dtype=np.bool_)
        eff_w, w_sum, used = _resolve_prototype_weights(pids, pw, valid)
        self.assertAlmostEqual(float(eff_w[0, 0]), 0.70, places=6)
        self.assertAlmostEqual(float(eff_w[0, 1]), 0.0, places=6)
        self.assertAlmostEqual(float(eff_w[0, 2]), 0.30, places=6)
        self.assertAlmostEqual(float(eff_w[0, 3]), 0.0, places=6)
        self.assertAlmostEqual(float(w_sum[0]), 1.0, places=6)
        self.assertEqual(used[0].tolist(), [True, False, True, False])

        var = np.array([0.0, 0.04, 0.05, 0.06, 0.045, 0.055], dtype=np.float64)
        var_post, d0, _ = squeeze_var(var, 8.0)
        self.assertGreater(float(var_post[0]), 0.0)
        self.assertGreater(d0, 0.0)


if __name__ == "__main__":
    unittest.main()
