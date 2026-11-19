"""Unit and contract tests for cfDNA fragmentomics kernels, density deconvolution, and Fisher weighting."""

from __future__ import annotations

import unittest

import numpy as np
from helpers import make_sample_table
from numpy.typing import NDArray
from scipy.stats import norm

import ccnv.features.fragmentomics as ff_mod
from ccnv._core import FRAG_BAND_EDGES
from ccnv.features.fragmentomics import (
    CohortFragmentomicsFit,
    apply_prenatal_fisher_weighting,
    fit_cohort_fragmentomics_model,
)
from ccnv.models import (
    NUM_FRAG_BANDS,
    FragmentomicsMetadata,
    InvalidInputError,
    PrenatalModelMetadata,
    SampleBuffers,
)
from ccnv.stats.fragmentomics import (
    CANONICAL_CHROMATOSOME_BP,
    FRAG_MAX_LEN,
    FRAG_MIN_LEN,
    FRAG_NUM_BINS,
    FragmentomicLandmarks,
    LogitEnsembleFit,
    align_histogram,
    build_fragmentomic_feature_vector,
    compute_band_fisher_weights,
    compute_helical_periodicity,
    compute_motif_diversity_score,
    deconvolve_fetal_maternal_densities,
    encode_kmer_4,
    estimate_sample_landmarks,
    evaluate_band_weighted_counts,
    fit_ff_fragmentomics_ensemble,
    predict_ff_fragmentomics,
    predict_ff_kl_deconvolution,
)


class TestFragmentomicsKernels(unittest.TestCase):
    def test_constants(self) -> None:
        self.assertEqual(FRAG_MIN_LEN, 30)
        self.assertEqual(FRAG_MAX_LEN, 600)
        self.assertEqual(FRAG_NUM_BINS, 571)
        self.assertEqual(CANONICAL_CHROMATOSOME_BP, 166.0)

    def test_encode_kmer_4(self) -> None:
        self.assertEqual(encode_kmer_4("AAAA"), 0)
        self.assertEqual(encode_kmer_4("AAAC"), 1)
        self.assertEqual(encode_kmer_4("AAAG"), 2)
        self.assertEqual(encode_kmer_4("AAAT"), 3)
        self.assertEqual(encode_kmer_4("TTTT"), 255)
        self.assertEqual(encode_kmer_4("CCCA"), 1 * 64 + 1 * 16 + 1 * 4 + 0)

        self.assertIsNone(encode_kmer_4("AAAN"))
        self.assertIsNone(encode_kmer_4("AAA"))
        self.assertIsNone(encode_kmer_4("AAAAA"))
        self.assertIsNone(encode_kmer_4("AAAR"))

        self.assertEqual(encode_kmer_4(b"AAAA"), 0)
        self.assertEqual(encode_kmer_4(b"TTTT"), 255)

    def test_compute_motif_diversity_score(self) -> None:
        uniform_counts = np.ones(256, dtype=np.float64) * 100.0
        self.assertAlmostEqual(
            compute_motif_diversity_score(uniform_counts), 1.0, places=5
        )

        delta_counts = np.zeros(256, dtype=np.float64)
        delta_counts[0] = 5000.0
        self.assertAlmostEqual(
            compute_motif_diversity_score(delta_counts), 0.0, places=5
        )

        scale_1 = np.arange(1, 257, dtype=np.float64)
        scale_2 = scale_1 * 17.5
        self.assertAlmostEqual(
            compute_motif_diversity_score(scale_1),
            compute_motif_diversity_score(scale_2),
            places=6,
        )
        self.assertEqual(compute_motif_diversity_score(np.zeros(256)), 0.0)

    def test_compute_helical_periodicity(self) -> None:
        lengths = np.arange(FRAG_MIN_LEN, FRAG_MAX_LEN + 1)
        hist = np.ones(FRAG_NUM_BINS, dtype=np.float64) * 1000.0
        mask_sub = (lengths >= 80) & (lengths <= 155)
        hist[mask_sub] += 200.0 * np.cos(2.0 * np.pi * lengths[mask_sub] / 10.4)

        amp, phase = compute_helical_periodicity(hist)
        self.assertGreater(amp, 1e-4)
        self.assertTrue(-np.pi <= phase <= np.pi)
        amp_scaled, _ = compute_helical_periodicity(hist * 10.0)
        self.assertAlmostEqual(amp, amp_scaled, places=6)

        flat_hist = np.ones(FRAG_NUM_BINS, dtype=np.float64) * 1000.0
        amp_flat, _ = compute_helical_periodicity(flat_hist)
        self.assertLess(amp_flat, 1e-3)

    def test_align_histogram_subpixel(self) -> None:
        hist = np.zeros(FRAG_NUM_BINS, dtype=np.float64)
        hist[136] = 1000.0

        shifted = align_histogram(hist, shift_bp=1.0)
        self.assertAlmostEqual(float(shifted[135]), 1000.0, places=3)
        self.assertAlmostEqual(float(shifted[136]), 0.0, places=3)
        self.assertAlmostEqual(float(np.sum(shifted)), 1000.0, places=3)

        shifted_half = align_histogram(hist, shift_bp=0.5)
        self.assertAlmostEqual(float(shifted_half[135]), 500.0, places=3)
        self.assertAlmostEqual(float(shifted_half[136]), 500.0, places=3)
        self.assertAlmostEqual(float(np.sum(shifted_half)), 1000.0, places=3)

    def test_estimate_sample_landmarks_paired_and_single_end(self) -> None:
        lengths = np.arange(FRAG_MIN_LEN, FRAG_MAX_LEN + 1)
        auto_hist = (
            8000.0 * np.exp(-0.5 * ((lengths - 166.0) / 10.0) ** 2)
            + 2500.0 * np.exp(-0.5 * ((lengths - 143.0) / 8.0) ** 2)
            + 2.0
        ).astype(np.int64)
        motifs = np.ones(256, dtype=np.int64) * 50

        lm = estimate_sample_landmarks(
            hist_autosomal=auto_hist,
            end_motif_counts=motifs,
            paired_end=True,
        )
        self.assertIsInstance(lm, FragmentomicLandmarks)
        self.assertTrue(lm.paired_end)
        self.assertAlmostEqual(lm.chromatosome_mode_bp, 166.0, delta=1.5)
        self.assertAlmostEqual(lm.sizing_shift_bp, 0.0, delta=1.5)
        self.assertGreater(lm.self_referenced_peak_ratio, 0.1)
        self.assertAlmostEqual(lm.motif_diversity_score, 1.0, places=4)
        self.assertLess(lm.high_mw_fraction, 0.05)

        lm_se = estimate_sample_landmarks(
            hist_autosomal=np.zeros(FRAG_NUM_BINS, dtype=np.int64),
            end_motif_counts=np.zeros(256, dtype=np.int64),
            paired_end=False,
        )
        self.assertFalse(lm_se.paired_end)
        self.assertEqual(lm_se.chromatosome_mode_bp, 166.0)
        self.assertEqual(lm_se.sizing_shift_bp, 0.0)

    def test_deconvolve_fetal_maternal_densities_and_fisher_weights(self) -> None:
        lengths = np.arange(FRAG_MIN_LEN, FRAG_MAX_LEN + 1)
        n_samples = 10
        male_mask = np.array([True] * 6 + [False] * 4)
        ffy_values = np.array([0.10, 0.12, 0.08, 0.15, 0.09, 0.11, 0.0, 0.0, 0.0, 0.0])

        p_f_true = np.exp(-0.5 * ((lengths - 143.0) / 10.0) ** 2)
        p_f_true /= np.sum(p_f_true)
        p_m_true = np.exp(-0.5 * ((lengths - 166.0) / 12.0) ** 2)
        p_m_true /= np.sum(p_m_true)

        aligned_auto = np.zeros((n_samples, FRAG_NUM_BINS), dtype=np.float64)
        aligned_chry = np.zeros((n_samples, FRAG_NUM_BINS), dtype=np.float64)

        for i in range(n_samples):
            f = ffy_values[i]
            mix = f * p_f_true + (1.0 - f) * p_m_true
            aligned_auto[i] = mix * 100_000.0
            if male_mask[i]:
                aligned_chry[i] = p_f_true * 1_000.0

        p_f_hat, p_m_hat = deconvolve_fetal_maternal_densities(
            aligned_auto, aligned_chry, ffy_values, male_mask
        )

        self.assertAlmostEqual(float(np.sum(p_f_hat)), 1.0, places=5)
        self.assertAlmostEqual(float(np.sum(p_m_hat)), 1.0, places=5)
        self.assertTrue(bool(np.all(p_f_hat >= 0.0)))
        self.assertTrue(bool(np.all(p_m_hat >= 0.0)))

        mode_f = int(lengths[np.argmax(p_f_hat)])
        mode_m = int(lengths[np.argmax(p_m_hat)])
        self.assertTrue(138 <= mode_f <= 148, f"p_f mode was {mode_f}")
        self.assertTrue(162 <= mode_m <= 170, f"p_m mode was {mode_m}")

        w_24, mu_f_w, mu_m_w = compute_band_fisher_weights(p_f_hat, p_m_hat, f0=0.10)
        self.assertEqual(len(w_24), NUM_FRAG_BANDS)
        self.assertTrue(bool(np.all(w_24 >= 0.25)))
        self.assertTrue(bool(np.all(w_24 <= 3.0)))
        self.assertAlmostEqual(mu_m_w, 1.0, places=4)
        self.assertGreater(mu_f_w, 1.0)

        band_143 = int(np.searchsorted(FRAG_BAND_EDGES, 143, side="right") - 1)
        band_166 = int(np.searchsorted(FRAG_BAND_EDGES, 166, side="right") - 1)
        self.assertGreater(float(w_24[band_143]), 1.0)
        self.assertLess(float(w_24[band_166]), 1.0)

    def test_deconvolve_low_male_fallback(self) -> None:
        aligned_auto = np.ones((5, FRAG_NUM_BINS), dtype=np.float64) * 100.0
        aligned_chry = np.zeros((5, FRAG_NUM_BINS), dtype=np.float64)
        male_mask = np.zeros(5, dtype=bool)
        ffy_values = np.zeros(5, dtype=np.float64)

        p_f_hat, p_m_hat = deconvolve_fetal_maternal_densities(
            aligned_auto, aligned_chry, ffy_values, male_mask
        )
        self.assertAlmostEqual(float(np.sum(p_f_hat)), 1.0, places=5)
        self.assertAlmostEqual(float(np.sum(p_m_hat)), 1.0, places=5)

        lengths = np.arange(FRAG_MIN_LEN, FRAG_MAX_LEN + 1, dtype=np.float64)
        pure_fetal_peak = np.zeros(FRAG_NUM_BINS, dtype=np.float64)
        pure_fetal_peak[143 - FRAG_MIN_LEN] = 500.0
        bad_low_ff_male_peak = np.zeros(FRAG_NUM_BINS, dtype=np.float64)
        bad_low_ff_male_peak[250 - FRAG_MIN_LEN] = 50_000.0
        chry_mix = np.vstack(
            [
                np.tile(pure_fetal_peak, (5, 1)),
                bad_low_ff_male_peak[np.newaxis, :],
                np.zeros((2, FRAG_NUM_BINS), dtype=np.float64),
            ]
        )
        auto_mix = np.tile(
            np.exp(-0.5 * ((lengths - 166.0) / 12.0) ** 2) * 1000.0, (8, 1)
        )
        male_mix = np.array([True] * 6 + [False] * 2, dtype=bool)
        ffy_mix = np.array([0.12, 0.11, 0.10, 0.14, 0.09, 0.03, 0.0, 0.0])
        p_f_clean, _ = deconvolve_fetal_maternal_densities(
            auto_mix, chry_mix, ffy_mix, male_mix
        )
        self.assertGreater(float(p_f_clean[143 - FRAG_MIN_LEN]), 0.05)
        self.assertAlmostEqual(float(p_f_clean[250 - FRAG_MIN_LEN]), 0.0, places=6)

    def test_feature_vector_and_ensemble_training(self) -> None:
        feat = build_fragmentomic_feature_vector(
            [100] * FRAG_NUM_BINS,
            [20] * 256,
            chromatosome_mode_bp=166.0,
            sizing_shift_bp=0.0,
            helical_10bp_amp=0.045,
            motif_mds=0.91,
        )
        self.assertEqual(len(feat), 36)
        self.assertFalse(bool(np.any(np.isnan(feat))))
        self.assertFalse(bool(np.any(np.isinf(feat))))

        rng = np.random.default_rng(42)
        n_samples = 25
        x_mat = rng.standard_normal((n_samples, 36))
        y = 0.04 + 0.18 * (x_mat[:, 0] - x_mat[:, 0].min()) / (
            x_mat[:, 0].max() - x_mat[:, 0].min() + 1e-6
        )

        ens = fit_ff_fragmentomics_ensemble(x_mat, y)
        self.assertIsInstance(ens, LogitEnsembleFit)
        self.assertEqual(len(ens.weights), 36)
        self.assertEqual(len(ens.feat_mean), 36)
        self.assertEqual(len(ens.feat_std), 36)
        self.assertGreaterEqual(ens.oof_rmse, 0.015)

        with self.assertRaises(InvalidInputError):
            fit_ff_fragmentomics_ensemble(x_mat[:4], y[:4])

        pred = predict_ff_fragmentomics(
            [100] * FRAG_NUM_BINS,
            [20] * 256,
            paired_end=True,
            chromatosome_mode_bp=166.0,
            sizing_shift_bp=0.0,
            helical_10bp_amp=0.045,
            motif_mds=0.91,
            ff_frag_weights=ens.weights,
            ff_frag_bias=ens.intercept,
            ff_frag_feature_mean=ens.feat_mean,
            ff_frag_feature_scale=ens.feat_std,
        )
        self.assertIsNotNone(pred)
        if pred is None:
            self.fail("expected non-None ff_size prediction")
        self.assertTrue(0.001 < pred < 0.999)

    def test_evaluate_band_weighted_counts(self) -> None:
        table = make_sample_table(
            [2, 0, 1],
            contigs=["chr1", "chr1", "chr1"],
            starts=[0, 100, 200],
            ends=[100, 200, 300],
            fragment_lengths=[[105, 155], [], [205]],
        )
        bufs = SampleBuffers.from_table(table)
        if bufs.frag_bands is None:
            self.fail("expected frag_bands")

        w_24 = np.ones(NUM_FRAG_BANDS, dtype=np.float64)
        band_105 = int(np.searchsorted(FRAG_BAND_EDGES, 105, side="right") - 1)
        band_155 = int(np.searchsorted(FRAG_BAND_EDGES, 155, side="right") - 1)
        band_205 = int(np.searchsorted(FRAG_BAND_EDGES, 205, side="right") - 1)
        w_24[band_105] = 2.0
        w_24[band_155] = 0.5
        w_24[band_205] = 3.0

        eff_cov, kappa = evaluate_band_weighted_counts(
            bufs.frag_bands, bufs.count, bufs.paired_count, w_24
        )
        self.assertEqual(len(eff_cov), 3)
        self.assertAlmostEqual(float(eff_cov[0]), 2.5)
        self.assertAlmostEqual(float(eff_cov[1]), 0.0)
        self.assertAlmostEqual(float(eff_cov[2]), 3.0)
        self.assertTrue(bool(np.all(kappa > 0.0)))

        table_se = make_sample_table(
            [5, 10, 15],
            contigs=["chr1", "chr1", "chr1"],
            starts=[0, 100, 200],
            ends=[100, 200, 300],
        )
        bufs_se = SampleBuffers.from_table(table_se)
        bands_se = (
            bufs_se.frag_bands
            if bufs_se.frag_bands is not None
            else np.zeros((3, NUM_FRAG_BANDS), dtype=np.uint32)
        )
        eff_se, kappa_se = evaluate_band_weighted_counts(
            bands_se, bufs_se.count, bufs_se.paired_count, w_24
        )
        np.testing.assert_allclose(eff_se, [5.0, 10.0, 15.0])
        np.testing.assert_allclose(kappa_se, [1.0, 1.0, 1.0])

    def test_fit_cohort_fragmentomics_model_fewer_than_five_samples_skips_ensemble(
        self,
    ) -> None:
        meta = FragmentomicsMetadata(
            paired_end=True,
            total_paired_fragments=10_000,
            chromatosome_mode_bp=167.0,
            core_shoulder_bp=143.0,
            sizing_shift_bp=0.0,
            self_referenced_peak_ratio=0.35,
            dinucleosome_fraction=0.08,
            high_mw_fraction=0.01,
            helical_10bp_amplitude=0.045,
            helical_10bp_phase=1.2,
            motif_diversity_score=0.91,
            hist_autosomal=[100] * FRAG_NUM_BINS,
            hist_chry=[10] * FRAG_NUM_BINS,
            hist_chrx=[50] * FRAG_NUM_BINS,
            end_motif_counts=[20] * 256,
        )
        frag_metas = [meta, meta, meta]
        is_male = np.array([True, False, True], dtype=bool)
        ff_targets = np.array([0.10, 0.08, 0.12], dtype=float)

        fit = fit_cohort_fragmentomics_model(frag_metas, is_male, ff_targets)
        self.assertIsNotNone(fit)
        if not isinstance(fit, CohortFragmentomicsFit):
            self.fail("expected CohortFragmentomicsFit")
        self.assertEqual(len(fit.p_fetal_density), FRAG_NUM_BINS)
        self.assertEqual(len(fit.p_maternal_density), FRAG_NUM_BINS)
        self.assertIsNone(fit.ff_frag_weights)
        self.assertIsNone(fit.ff_frag_bias)
        self.assertIsNone(fit.ff_frag_feature_mean)
        self.assertIsNone(fit.ff_frag_feature_scale)
        self.assertEqual(fit.ff_size_std, 0.018)

    def test_align_reference_to_weighted_sample_deleted_and_fisher_weighting(
        self,
    ) -> None:
        self.assertFalse(hasattr(ff_mod, "align_reference_to_weighted_sample"))
        self.assertFalse(hasattr(ff_mod, "extract_bin_fragmentomic_covariates"))

        sample_tbl = make_sample_table(
            [10, 20],
            contigs=["chr1", "chr1"],
            starts=[0, 100],
            ends=[100, 200],
            fragment_lengths=[[140] * 10, [166] * 20],
        )
        dummy_meta = PrenatalModelMetadata(
            prenatal_bin_size=100_000,
            yfrac_threshold=0.004,
            ffy_slope=1.0,
            olm_weights=[],
            olm_bias=0.0,
            female_chrx_ratio_mean=1.0,
            female_chrx_ratio_std=0.01,
            p_fetal_density=[1.0 / FRAG_NUM_BINS] * FRAG_NUM_BINS,
            p_maternal_density=[1.0 / FRAG_NUM_BINS] * FRAG_NUM_BINS,
            band_fisher_weights=[1.5] * 12 + [0.8] * 12,
            mu_fetal_w=1.15,
            mu_maternal_w=1.0,
        )
        weighted_bufs = apply_prenatal_fisher_weighting(
            SampleBuffers.from_table(sample_tbl), dummy_meta
        )
        self.assertIsInstance(weighted_bufs, SampleBuffers)
        self.assertIsNotNone(weighted_bufs.kappa_frag)
        if weighted_bufs.kappa_frag is None:
            self.fail("expected kappa_frag")
        self.assertTrue(bool(np.all(weighted_bufs.kappa_frag > 0.0)))

    def test_fragmentomics_non_ascii_kmer_and_unpaired_kish_kappa(self) -> None:
        self.assertIsNone(encode_kmer_4("AC\u0100T"))
        kmer_code = encode_kmer_4("ACGT")
        self.assertIsNotNone(kmer_code)
        if kmer_code is not None:
            self.assertGreaterEqual(kmer_code, 0)

        bands = np.zeros((1, NUM_FRAG_BANDS), dtype=np.uint32)
        bands[0, 0] = 25
        bands[0, 1] = 25
        counts = np.array([100.0], dtype=np.float64)
        paired = np.array([50], dtype=np.uint32)
        weights = np.ones(NUM_FRAG_BANDS, dtype=np.float64)
        weights[0] = 3.0
        weights[1] = 0.25
        w_counts, kappa = evaluate_band_weighted_counts(bands, counts, paired, weights)
        expected_w = 50.0 + 25.0 * 3.0 + 25.0 * 0.25
        expected_w2 = 50.0 * 1.0 + 25.0 * 9.0 + 25.0 * 0.0625
        expected_kappa = expected_w2 / expected_w
        self.assertAlmostEqual(float(w_counts[0]), expected_w, places=6)
        self.assertAlmostEqual(float(kappa[0]), expected_kappa, places=6)


class TestFragmentomicsDeconvolutionAndFisher(unittest.TestCase):
    lengths: NDArray[np.float64]
    p_f_true: NDArray[np.float64]
    p_m_true: NDArray[np.float64]

    def setUp(self) -> None:
        self.lengths = np.arange(FRAG_MIN_LEN, FRAG_MAX_LEN + 1, dtype=np.float64)
        p_f = norm.pdf(self.lengths, loc=143.0, scale=10.0)
        p_f[self.lengths < 155] *= 1.0 + 0.15 * np.cos(
            2.0 * np.pi * (self.lengths[self.lengths < 155] - 143.0) / 10.4
        )
        p_f = np.maximum(p_f, 0.0)
        self.p_f_true = p_f / np.sum(p_f)

        p_m = norm.pdf(self.lengths, loc=166.0, scale=12.0)
        self.p_m_true = p_m / np.sum(p_m)

    def test_deconvolve_fetal_maternal_densities_recovers_pure_maternal_despite_50pct_female_fetal_reads(
        self,
    ) -> None:
        rng = np.random.default_rng(42)
        n_samples = 20
        is_male = np.array([True] * 10 + [False] * 10, dtype=np.bool_)

        true_ff = rng.uniform(0.08, 0.20, size=n_samples)
        f_estimates = np.where(is_male, true_ff, 0.0)

        hist_auto = np.zeros((n_samples, FRAG_NUM_BINS), dtype=np.float64)
        hist_chry = np.zeros((n_samples, FRAG_NUM_BINS), dtype=np.float64)

        for i in range(n_samples):
            f = true_ff[i]
            mix = f * self.p_f_true + (1.0 - f) * self.p_m_true
            hist_auto[i] = rng.multinomial(100_000, mix)
            if is_male[i]:
                hist_chry[i] = rng.multinomial(2_000, self.p_f_true) + rng.multinomial(
                    20, self.p_m_true
                )
            else:
                hist_chry[i] = rng.multinomial(20, self.p_m_true)

        p_f_hat, p_m_hat = deconvolve_fetal_maternal_densities(
            hist_auto,
            hist_chry,
            f_estimates,
            is_male,
            smooth_sigma=1.2,
            em_steps=8,
        )

        self.assertAlmostEqual(float(np.sum(p_f_hat)), 1.0, places=5)
        self.assertAlmostEqual(float(np.sum(p_m_hat)), 1.0, places=5)
        self.assertTrue(bool(np.all(p_f_hat >= 0.0)))
        self.assertTrue(bool(np.all(p_m_hat >= 0.0)))

        tv_maternal = 0.5 * float(np.sum(np.abs(p_m_hat - self.p_m_true)))
        self.assertLess(
            tv_maternal,
            0.025,
            f"Maternal density TV error {tv_maternal:.4f} exceeded 0.025 limit",
        )

        peak_f = float(self.lengths[np.argmax(p_f_hat)])
        peak_m = float(self.lengths[np.argmax(p_m_hat)])
        separation = peak_m - peak_f
        self.assertGreaterEqual(
            separation,
            15.0,
            f"Peak separation {separation:.1f} bp was less than 15 bp (f={peak_f}, m={peak_m})",
        )
        self.assertTrue(
            138.0 <= peak_f <= 148.0, f"fetal peak {peak_f} not near 143 bp"
        )
        self.assertTrue(
            162.0 <= peak_m <= 170.0, f"maternal peak {peak_m} not near 166 bp"
        )

        short_mask = (self.lengths >= 120.0) & (self.lengths <= 145.0)
        long_mask = (self.lengths >= 185.0) & (self.lengths <= 220.0)

        short_ratio = float(
            np.mean(p_f_hat[short_mask] / np.maximum(p_m_hat[short_mask], 1e-12))
        )
        long_ratio = float(
            np.mean(p_f_hat[long_mask] / np.maximum(p_m_hat[long_mask], 1e-12))
        )

        self.assertGreater(
            short_ratio,
            2.0,
            f"Short band p_f/p_m ratio {short_ratio:.2f} was not > 2.0",
        )
        self.assertLess(
            long_ratio,
            0.5,
            f"Long band p_f/p_m ratio {long_ratio:.2f} was not < 0.5",
        )

    def test_helical_periodicity_and_feature_vector_depth_invariance(self) -> None:
        base_hist = np.ones(FRAG_NUM_BINS, dtype=np.float64) * 100.0
        mask_sub = (self.lengths >= 80.0) & (self.lengths <= 155.0)
        base_hist[mask_sub] += 30.0 * np.cos(
            2.0 * np.pi * self.lengths[mask_sub] / 10.4
        )
        base_hist += 500.0 * np.exp(-0.5 * ((self.lengths - 166.0) / 12.0) ** 2)

        hist_01x = base_hist * 10.0
        hist_15x = base_hist * 1500.0

        amp_01x, phase_01x = compute_helical_periodicity(hist_01x)
        amp_15x, phase_15x = compute_helical_periodicity(hist_15x)

        self.assertGreater(amp_01x, 0.0)
        self.assertAlmostEqual(amp_01x, amp_15x, places=6)
        self.assertAlmostEqual(phase_01x, phase_15x, places=6)

        feat_01x = build_fragmentomic_feature_vector(
            hist_01x.astype(int).tolist(),
            [100] * 256,
            chromatosome_mode_bp=166.2,
            sizing_shift_bp=0.2,
            helical_10bp_amp=amp_01x,
            motif_mds=0.92,
        )
        feat_15x = build_fragmentomic_feature_vector(
            hist_15x.astype(int).tolist(),
            [15000] * 256,
            chromatosome_mode_bp=166.2,
            sizing_shift_bp=0.2,
            helical_10bp_amp=amp_15x,
            motif_mds=0.92,
        )

        self.assertEqual(len(feat_01x), 36)
        self.assertEqual(len(feat_15x), 36)
        self.assertAlmostEqual(float(feat_01x[0]), 166.2, places=4)
        self.assertAlmostEqual(float(feat_15x[0]), 166.2, places=4)
        np.testing.assert_allclose(feat_01x, feat_15x, rtol=1e-3, atol=1e-3)

    def test_predict_ff_kl_deconvolution(self) -> None:
        rng = np.random.default_rng(123)
        for true_f in (0.04, 0.10, 0.18, 0.30):
            mix = true_f * self.p_f_true + (1.0 - true_f) * self.p_m_true
            hist_sample = rng.multinomial(150_000, mix).astype(np.float64)

            hat_f, sigma_kl = predict_ff_kl_deconvolution(
                hist_sample, self.p_f_true, self.p_m_true
            )

            self.assertAlmostEqual(
                hat_f,
                true_f,
                delta=0.012,
                msg=f"f_KL {hat_f:.4f} deviated from true {true_f:.4f}",
            )
            self.assertGreater(sigma_kl, 0.0)
            self.assertLess(sigma_kl, 0.03)

    def test_compute_band_fisher_weights_and_kish_factor(self) -> None:
        w_24, mu_f_w, mu_m_w = compute_band_fisher_weights(
            self.p_f_true, self.p_m_true, f0=0.08
        )

        self.assertEqual(len(w_24), 24)
        self.assertEqual(len(FRAG_BAND_EDGES) - 1, 24)
        self.assertTrue(bool(np.all(w_24 >= 0.25)))
        self.assertTrue(bool(np.all(w_24 <= 3.0)))
        self.assertAlmostEqual(mu_m_w, 1.0, places=3)
        self.assertGreater(mu_f_w, 1.05)

        n_bins = 100
        rng = np.random.default_rng(999)
        frag_bands = rng.integers(0, 50, size=(n_bins, 24), dtype=np.uint32)
        paired_counts = np.sum(frag_bands, axis=1, dtype=np.uint32)
        raw_counts = paired_counts + rng.integers(0, 10, size=n_bins, dtype=np.uint32)

        w_counts, kappa_frag = evaluate_band_weighted_counts(
            frag_bands, raw_counts, paired_counts, w_24
        )

        self.assertEqual(len(w_counts), n_bins)
        self.assertEqual(len(kappa_frag), n_bins)
        self.assertTrue(bool(np.all(kappa_frag > 0.0)))

        zero_bands = np.zeros((1, 24), dtype=np.uint32)
        _, zero_kappa = evaluate_band_weighted_counts(
            zero_bands,
            np.array([0], dtype=np.uint32),
            np.array([0], dtype=np.uint32),
            w_24,
        )
        self.assertEqual(float(zero_kappa[0]), 1.0)

        flat_w = np.ones(24, dtype=np.float64)
        flat_counts, flat_kappa = evaluate_band_weighted_counts(
            frag_bands, raw_counts, paired_counts, flat_w
        )
        np.testing.assert_allclose(flat_counts, raw_counts.astype(np.float64))
        np.testing.assert_allclose(flat_kappa, 1.0, atol=1e-6)

    def test_compound_poisson_kappa_scale_invariance(self) -> None:
        bands = np.zeros((3, NUM_FRAG_BANDS), dtype=np.uint32)
        bands[:, 4] = 40
        raw = np.array([40.0, 40.0, 40.0], dtype=np.float64)
        paired = np.array([40, 40, 40], dtype=np.uint32)
        for scale in (2.0, 0.5, 1.0):
            w_vec = np.full(NUM_FRAG_BANDS, scale, dtype=np.float64)
            eff_c, kappa_cp = evaluate_band_weighted_counts(bands, raw, paired, w_vec)
            np.testing.assert_allclose(eff_c, 40.0 * scale, rtol=1e-6)
            np.testing.assert_allclose(kappa_cp, scale, rtol=1e-6)
            np.testing.assert_allclose(kappa_cp / eff_c, 1.0 / 40.0, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
