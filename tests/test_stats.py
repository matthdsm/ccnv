"""Comprehensive unit tests for ccnv.stats subpackage."""

from __future__ import annotations

import unittest

import numpy as np

from ccnv import stats
from ccnv.models.exceptions import ReferenceError
from ccnv.stats.neighbors import (
    build_whitened_subspace_features,
    calibrate_prototype_dispersion,
    evaluate_loco_prototype_baseline,
    fit_prototype_graph,
    squeeze_var,
)
from ccnv.stats.pca import (
    fit_whitened_pca,
    pca_irlba_numpy,
    project_huber_pca_wave,
    reconstruct_whitened_pca_expected,
)
from ccnv.stats.prenatal_math import (
    classify_sca_ray_4d,
    fit_msy_sex_mixture,
    fuse_active_fetal_fraction_student_t4,
    ols_fit,
)
from ccnv.stats.regression import (
    build_exogenous_covariate_basis,
    fit_poisson_covariate_trends,
    half_sample_mode,
    standardize_column,
)
from ccnv.stats.segmentation import (
    ContigSegmentationInput,
    Segment,
    WeightedL2Cost,
    segment_and_score_genome,
    segment_contig,
)


class TestStatsPackageExports(unittest.TestCase):
    def test_all_symbols_exported(self) -> None:
        expected_symbols = [
            "CANONICAL_CHROMATOSOME_BP",
            "DEFAULT_MAX_BLACKLIST_GAP",
            "FRAG_MAX_LEN",
            "FRAG_MIN_LEN",
            "FRAG_NUM_BINS",
            "LN2_SQ",
            "LOG_WAVE_CLIP",
            "ContigSegmentationInput",
            "MSYSexModel",
            "PoissonFFYModel",
            "PreparedContigData",
            "Segment",
            "WeightedL2Cost",
            "WhitenedPCAResult",
            "align_histogram",
            "build_exogenous_covariate_basis",
            "build_fragmentomic_feature_vector",
            "build_whitened_subspace_features",
            "calibrate_prototype_dispersion",
            "classify_sca_ray_4d",
            "compute_autosomal_density",
            "compute_band_fisher_weights",
            "compute_helical_periodicity",
            "compute_kish_loco_inflation",
            "compute_loco_prototype_support",
            "compute_motif_diversity_score",
            "deconvolve_fetal_maternal_densities",
            "deconvolve_fetal_maternal_em",
            "encode_kmer_4",
            "estimate_dyadic_inflation",
            "estimate_sample_landmarks",
            "evaluate_band_weighted_counts",
            "evaluate_blue_scores_and_bayes_factors",
            "evaluate_loco_prototype_baseline",
            "evaluate_poisson_msy_ffy",
            "exact_poisson_log2_jensen_bias",
            "exact_poisson_log_variance",
            "fit_ff_fragmentomics_ensemble",
            "fit_ffa_logit_ensemble",
            "fit_forward_ffx_model",
            "fit_logit_linear_ensemble",
            "fit_msy_sex_mixture",
            "fit_poisson_covariate_trends",
            "fit_poisson_msy_ffy",
            "fit_prototype_graph",
            "fit_whitened_pca",
            "fuse_active_fetal_fraction_student_t4",
            "half_sample_mode",
            "hsm_biweight_mode",
            "interpolate_dyadic_inflation",
            "ols_fit",
            "pca_irlba_numpy",
            "predict_ff_fragmentomics",
            "predict_ff_kl_deconvolution",
            "project_huber_pca_wave",
            "reconstruct_whitened_pca_expected",
            "segment_and_score_genome",
            "segment_contig",
            "squeeze_var",
            "standardize_column",
        ]
        for symbol in expected_symbols:
            self.assertTrue(hasattr(stats, symbol), f"Symbol {symbol} missing in stats")
            self.assertIn(symbol, stats.__all__)


class TestWhitenedPCA(unittest.TestCase):
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
        self.assertTrue(np.all(recon > 0.0))

        log_sig = np.log(
            (counts[:, 0] + 0.5) / (expected_in[:, 0] * res.mean_rate + 0.5)
        )
        coef, wave = project_huber_pca_wave(log_sig.astype(np.float64), res.loadings)
        self.assertEqual(coef.shape, (res.loadings.shape[0],))
        self.assertEqual(wave.shape, (n_bins,))
        self.assertTrue(np.all(np.isfinite(wave)))

    def test_fit_whitened_pca_invalid_shapes(self) -> None:
        with self.assertRaises((ReferenceError, ValueError)):
            fit_whitened_pca(
                np.empty((0, 0), dtype=np.float32),
                np.empty((0, 0), dtype=np.float32),
                np.empty(0, dtype=bool),
            )

    def test_pca_irlba_numpy_matches_expected_shapes(self) -> None:
        rng = np.random.default_rng(123)
        mat = rng.normal(size=(20, 15))
        scores, center, rotation = pca_irlba_numpy(mat, n_components=3, seed=42)
        self.assertEqual(scores.shape, (20, 3))
        self.assertEqual(center.shape, (15,))
        self.assertEqual(rotation.shape, (15, 3))


class TestPrototypeNeighbors(unittest.TestCase):
    def test_fit_prototype_graph_and_loco_dispersion(self) -> None:
        rng = np.random.default_rng(42)
        n_bins, n_samples = 40, 10
        contig_ids = np.repeat(np.arange(4, dtype=np.int16), 10)
        usable = np.ones(n_bins, dtype=bool)
        features = rng.normal(size=(n_bins, 3)).astype(np.float32)
        centers, proto_ids, proto_weights, mean_dist = fit_prototype_graph(
            features,
            max_prototypes=12,
            top_r=4,
        )
        self.assertEqual(centers.shape[1], 3)
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
            n_prototypes=int(centers.shape[0]),
        )
        self.assertEqual(null_mean.shape, (n_bins,))
        self.assertEqual(bio_disp.shape, (n_bins,))
        self.assertTrue(np.all(bio_disp >= 0.0))

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

        var_post, d0_hat, s0_sq_hat = squeeze_var(
            sample_var, df=df, winsor_tail_p=(0.0, 0.0)
        )
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
        self.assertTrue(np.all(np.isfinite(feats)))


class TestSegmentationAndRegression(unittest.TestCase):
    def test_weighted_l2_cost_and_segment_genome(self) -> None:
        values = np.array([1.0, 2.0, 10.0])
        weights = np.array([1.0, 3.0, 0.0])
        cost = WeightedL2Cost().fit(values, weights)
        expected = np.sum(weights[:2] * np.square(values[:2] - 1.75))
        self.assertAlmostEqual(cost.error(0, 3), float(expected))

        starts = np.array([0, 100, 200, 300, 400], dtype=np.int64)
        ends = np.array([100, 200, 300, 400, 500], dtype=np.int64)
        ratios = np.array([0.0, 0.0, 0.8, 0.8, 0.8], dtype=np.float64)
        prec = np.full(5, 100.0, dtype=np.float64)

        segs = segment_contig(
            contig="chr1",
            starts=starts,
            ends=ends,
            ratios=ratios,
            weights=prec,
            contig_length=500,
            penalty=0.5,
            min_size=2,
        )
        self.assertGreaterEqual(len(segs), 1)
        self.assertIsInstance(segs[0], Segment)

        scored = segment_and_score_genome(
            {
                "chr1": ContigSegmentationInput(
                    starts=starts,
                    ends=ends,
                    ratios=ratios,
                    weights=prec,
                    contig_length=500,
                )
            },
            penalty=0.5,
            min_size=2,
        )
        self.assertGreaterEqual(len(scored), 1)

    def test_poisson_covariates_and_prenatal_stats(self) -> None:
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
        self.assertTrue(np.all(mult > 0.0))

        z_col = standardize_column(np.linspace(1.0, 5.0, 20))
        self.assertIsNotNone(z_col)
        assert z_col is not None
        self.assertAlmostEqual(float(np.mean(z_col)), 0.0, places=6)
        self.assertAlmostEqual(
            half_sample_mode(rng.normal(2.0, 0.01, 50)), 2.0, delta=0.05
        )

        sex_model = fit_msy_sex_mixture(
            np.array([2, 3, 1, 2, 80, 95, 110, 90], dtype=np.float64),
            np.full(8, 2_000.0),
        )
        self.assertTrue(sex_model.bimodal)
        self.assertGreater(sex_model.bg_rate, 0.0)
        b0, b1 = ols_fit(np.array([1.0, 2.0, 3.0]), np.array([3.0, 5.0, 7.0]))
        self.assertAlmostEqual(b0, 1.0, places=6)
        self.assertAlmostEqual(b1, 2.0, places=6)

        fused_ff, fused_se = fuse_active_fetal_fraction_student_t4(
            [0.10, 0.105, 0.098], [0.008, 0.02, 0.018]
        )
        self.assertAlmostEqual(fused_ff, 0.10, delta=0.01)
        self.assertGreater(fused_se, 0.0)

        call, _posteriors, act_ff, act_se, _src = classify_sca_ray_4d(
            delta_x=-0.05,
            sigma_x=0.005,
            ffy=0.10,
            sigma_ffy=0.008,
            ffa=0.10,
            sigma_ffa=0.02,
            ff_size=0.10,
            sigma_size=0.018,
        )
        self.assertEqual(call, "XY")
        self.assertAlmostEqual(act_ff, 0.10, delta=0.015)
        self.assertGreater(act_se, 0.0)


if __name__ == "__main__":
    unittest.main()
