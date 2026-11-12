"""Contract tests for Plan 3 Task 2:

- Contamination-Free Fragment Length Density Deconvolution (EM + male-subtraction)
- Fixed 36-Feature FF_size Ensemble Vector & Depth Invariance
- Zero-Parameter f_KL MLE with Fisher Information SE
- Reference-Side 24-Band Fisher Read Weighting & Kish Factor Propagation
- Deletion of Legacy Post-Hoc align_reference_to_weighted_sample
"""

from __future__ import annotations

import unittest

import numpy as np
from scipy.stats import norm

from ccnv._core import FRAG_BAND_EDGES
from ccnv.models import PrenatalModelMetadata
from ccnv.stats.fragmentomics import (
    FRAG_MAX_LEN,
    FRAG_MIN_LEN,
    FRAG_NUM_BINS,
    build_fragmentomic_feature_vector,
    compute_band_fisher_weights,
    compute_helical_periodicity,
    deconvolve_fetal_maternal_densities,
    evaluate_band_weighted_counts,
    predict_ff_kl_deconvolution,
)


class TestFragmentomicsAndRefFisherContract(unittest.TestCase):
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
        """EM deconvolution with male algebraic subtraction must recover pure maternal density."""
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
        self.assertTrue(np.all(p_f_hat >= 0.0))
        self.assertTrue(np.all(p_m_hat >= 0.0))

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
        """Helical 10.4 bp periodicity amplitude and 36-feature FF_size vector must be invariant to depth."""
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
        """Zero-parameter KL / Poisson mixture MLE must recover true fetal fraction."""
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
        """Reference-side 24-band Fisher weights must normalize diploid baseline to 1.0."""
        w_24, mu_f_w, mu_m_w = compute_band_fisher_weights(
            self.p_f_true, self.p_m_true, f0=0.08
        )

        self.assertEqual(len(w_24), 24)
        self.assertEqual(len(FRAG_BAND_EDGES) - 1, 24)
        self.assertTrue(np.all(w_24 >= 0.25))
        self.assertTrue(np.all(w_24 <= 3.0))
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
        self.assertTrue(np.all(kappa_frag >= 1.0))

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

    def test_legacy_align_reference_deleted_and_prenatal_metadata_contract(
        self,
    ) -> None:
        """align_reference_to_weighted_sample must be deleted and metadata must round-trip."""
        import ccnv.features.fragmentomics as ff
        import ccnv.predict as pred

        self.assertFalse(hasattr(ff, "align_reference_to_weighted_sample"))
        self.assertFalse(hasattr(pred, "align_reference_to_weighted_sample"))

        meta = PrenatalModelMetadata(
            prenatal_bin_size=100_000,
            yfrac_threshold=0.005,
            ffy_slope=1.0,
            olm_weights=[0.05] * 10,
            olm_bias=0.0,
            female_chrx_ratio_mean=1.0,
            female_chrx_ratio_std=0.01,
            band_fisher_weights=[1.1] * 24,
            mu_fetal_w=1.25,
            mu_maternal_w=1.0,
        )

        d = meta.to_dict()
        bw_raw = d["band_fisher_weights"]
        if not isinstance(bw_raw, list):
            self.fail("expected list for band_fisher_weights")
        self.assertEqual(len(bw_raw), 24)
        self.assertAlmostEqual(float(meta.mu_fetal_w or 0.0), 1.25)
        self.assertAlmostEqual(float(meta.mu_maternal_w or 0.0), 1.0)

        reloaded = PrenatalModelMetadata.from_dict(d)
        if reloaded.band_fisher_weights is None:
            self.fail("expected non-None band_fisher_weights")
        self.assertEqual(len(reloaded.band_fisher_weights), 24)
        self.assertAlmostEqual(float(reloaded.mu_fetal_w or 0.0), 1.25)
        self.assertAlmostEqual(float(reloaded.mu_maternal_w or 0.0), 1.0)


if __name__ == "__main__":
    unittest.main()
