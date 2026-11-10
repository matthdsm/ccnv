"""Unit and property tests for cfDNA fragmentomic mathematical kernels."""

import unittest

import numpy as np

from ccnv.models.schemas import FragmentomicsMetadata, PrenatalModelMetadata
from ccnv.stats.fragmentomics import (
    CANONICAL_CHROMATOSOME_BP,
    FRAG_MAX_LEN,
    FRAG_MIN_LEN,
    FRAG_NUM_BINS,
    align_histogram,
    build_fragmentomic_feature_vector,
    compute_helical_periodicity,
    compute_motif_diversity_score,
    deconvolve_fetal_maternal_densities,
    encode_kmer_4,
    estimate_sample_landmarks,
    fisher_optimal_weights,
    fit_ff_fragmentomics_ensemble,
    predict_ff_fragmentomics,
    segment_fragment_llr,
)


class TestFragmentomicsKernels(unittest.TestCase):
    def test_constants(self) -> None:
        self.assertEqual(FRAG_MIN_LEN, 30)
        self.assertEqual(FRAG_MAX_LEN, 600)
        self.assertEqual(FRAG_NUM_BINS, 571)
        self.assertEqual(CANONICAL_CHROMATOSOME_BP, 166.0)

    def test_encode_kmer_4_forward_and_reverse_complement(self) -> None:
        self.assertEqual(encode_kmer_4("AAAA"), 0)
        self.assertEqual(encode_kmer_4("AAAC"), 1)
        self.assertEqual(encode_kmer_4("AAAG"), 2)
        self.assertEqual(encode_kmer_4("AAAT"), 3)
        self.assertEqual(encode_kmer_4("TTTT"), 255)
        self.assertEqual(encode_kmer_4("CCCA"), 1 * 64 + 1 * 16 + 1 * 4 + 0)

        # Reverse complement: RC of TTTT is AAAA -> 0
        self.assertEqual(encode_kmer_4("TTTT", reverse_complement=True), 0)
        # RC of CCCA is TGGG -> 3*64 + 2*16 + 2*4 + 2 = 192 + 32 + 8 + 2 = 234
        self.assertEqual(encode_kmer_4("CCCA", reverse_complement=True), 234)

        # Invalid characters or lengths return None
        self.assertIsNone(encode_kmer_4("AAAN"))
        self.assertIsNone(encode_kmer_4("AAA"))
        self.assertIsNone(encode_kmer_4("AAAAA"))
        self.assertIsNone(encode_kmer_4("AAAR"))

        # Accepts bytes as well
        self.assertEqual(encode_kmer_4(b"AAAA"), 0)
        self.assertEqual(encode_kmer_4(b"TTTT"), 255)

    def test_compute_motif_diversity_score(self) -> None:
        # Uniform distribution over 256 motifs has MDS = 1.0
        uniform_counts = np.ones(256, dtype=np.float64) * 100.0
        self.assertAlmostEqual(
            compute_motif_diversity_score(uniform_counts), 1.0, places=5
        )

        # Delta distribution (all counts in 1 motif) has MDS = 0.0
        delta_counts = np.zeros(256, dtype=np.float64)
        delta_counts[0] = 5000.0
        self.assertAlmostEqual(
            compute_motif_diversity_score(delta_counts), 0.0, places=5
        )

        # Scale invariance
        scale_1 = np.arange(1, 257, dtype=np.float64)
        scale_2 = scale_1 * 17.5
        self.assertAlmostEqual(
            compute_motif_diversity_score(scale_1),
            compute_motif_diversity_score(scale_2),
            places=6,
        )

        # All zeros returns 0.0
        self.assertEqual(compute_motif_diversity_score(np.zeros(256)), 0.0)

    def test_compute_helical_periodicity(self) -> None:
        lengths = np.arange(FRAG_MIN_LEN, FRAG_MAX_LEN + 1)  # 30 to 600
        # Create a synthetic signal with a pure 10.4 bp oscillation in [80, 155] bp
        hist = np.ones(FRAG_NUM_BINS, dtype=np.float64) * 1000.0
        mask_sub = (lengths >= 80) & (lengths <= 155)
        hist[mask_sub] += 200.0 * np.cos(2.0 * np.pi * lengths[mask_sub] / 10.4)

        amp, phase = compute_helical_periodicity(hist)
        self.assertGreater(amp, 50.0)
        self.assertTrue(-np.pi <= phase <= np.pi)

        # Flat signal has negligible amplitude
        flat_hist = np.ones(FRAG_NUM_BINS, dtype=np.float64) * 1000.0
        amp_flat, _ = compute_helical_periodicity(flat_hist)
        self.assertLess(amp_flat, 1e-3)

    def test_align_histogram_subpixel(self) -> None:
        hist = np.zeros(FRAG_NUM_BINS, dtype=np.float64)
        # Place a delta at index 136 (corresponding to length 30 + 136 = 166 bp)
        hist[136] = 1000.0

        # Shift by +1.0 bp should shift the peak leftwards (index 135)
        shifted = align_histogram(hist, shift_bp=1.0)
        self.assertAlmostEqual(shifted[135], 1000.0, places=3)
        self.assertAlmostEqual(shifted[136], 0.0, places=3)
        self.assertAlmostEqual(np.sum(shifted), 1000.0, places=3)

        # Shift by +0.5 bp should split equally between index 135 and 136
        shifted_half = align_histogram(hist, shift_bp=0.5)
        self.assertAlmostEqual(shifted_half[135], 500.0, places=3)
        self.assertAlmostEqual(shifted_half[136], 500.0, places=3)
        self.assertAlmostEqual(np.sum(shifted_half), 1000.0, places=3)

    def test_estimate_sample_landmarks_paired_and_single_end(self) -> None:
        lengths = np.arange(FRAG_MIN_LEN, FRAG_MAX_LEN + 1)
        # Synthetic cfDNA histogram:
        # Maternal mode at 166 bp, fetal shoulder at 143 bp
        auto_hist = (
            8000.0 * np.exp(-0.5 * ((lengths - 166.0) / 10.0) ** 2)
            + 2500.0 * np.exp(-0.5 * ((lengths - 143.0) / 8.0) ** 2)
            + 2.0
        ).astype(np.int64)

        chry_hist = np.zeros(FRAG_NUM_BINS, dtype=np.int64)
        chrx_hist = (auto_hist // 2).astype(np.int64)
        motifs = np.ones(256, dtype=np.int64) * 50

        meta = estimate_sample_landmarks(
            hist_autosomal=auto_hist,
            hist_chry=chry_hist,
            hist_chrx=chrx_hist,
            end_motif_counts=motifs,
            paired_end=True,
        )

        self.assertTrue(meta.paired_end)
        self.assertAlmostEqual(meta.chromatosome_mode_bp, 166.0, delta=1.5)
        self.assertAlmostEqual(meta.sizing_shift_bp, 0.0, delta=1.5)
        self.assertGreater(meta.self_referenced_peak_ratio, 0.1)
        self.assertAlmostEqual(meta.motif_diversity_score, 1.0, places=4)
        self.assertLess(meta.high_mw_fraction, 0.05)

        # Single-end fallback
        meta_se = estimate_sample_landmarks(
            hist_autosomal=np.zeros(FRAG_NUM_BINS, dtype=np.int64),
            hist_chry=np.zeros(FRAG_NUM_BINS, dtype=np.int64),
            hist_chrx=np.zeros(FRAG_NUM_BINS, dtype=np.int64),
            end_motif_counts=np.zeros(256, dtype=np.int64),
            paired_end=False,
        )
        self.assertFalse(meta_se.paired_end)
        self.assertEqual(meta_se.chromatosome_mode_bp, 166.0)
        self.assertEqual(meta_se.sizing_shift_bp, 0.0)

    def test_deconvolve_fetal_maternal_densities_and_fisher_weights(self) -> None:
        lengths = np.arange(FRAG_MIN_LEN, FRAG_MAX_LEN + 1)
        n_samples = 10
        male_mask = np.array([True] * 6 + [False] * 4)
        ffy_values = np.array([0.10, 0.12, 0.08, 0.15, 0.09, 0.11, 0.0, 0.0, 0.0, 0.0])

        # Generate synthetic aligned autosomal and chrY histograms
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
                aligned_chry[i] = p_f_true * 1_000.0  # chrY is purely fetal in males

        p_f_hat, p_m_hat = deconvolve_fetal_maternal_densities(
            aligned_auto_hists=aligned_auto,
            aligned_chry_hists=aligned_chry,
            male_mask=male_mask,
            ffy_values=ffy_values,
        )

        # Densities must sum to 1.0 on simplex
        self.assertAlmostEqual(np.sum(p_f_hat), 1.0, places=5)
        self.assertAlmostEqual(np.sum(p_m_hat), 1.0, places=5)
        self.assertTrue(np.all(p_f_hat >= 0.0))
        self.assertTrue(np.all(p_m_hat >= 0.0))

        # Peak of p_f_hat should be near 143, p_m_hat near 166
        mode_f = lengths[np.argmax(p_f_hat)]
        mode_m = lengths[np.argmax(p_m_hat)]
        self.assertTrue(138 <= mode_f <= 148, f"p_f mode was {mode_f}")
        self.assertTrue(162 <= mode_m <= 170, f"p_m mode was {mode_m}")

        # Fisher-optimal weights
        weights = fisher_optimal_weights(p_f_hat, p_m_hat, f0=0.10)
        self.assertEqual(len(weights), FRAG_NUM_BINS)
        self.assertTrue(np.all(weights >= 0.25))
        self.assertTrue(np.all(weights <= 3.0))

        # Weight depth conservation: E_{mix}[w*] == 1.0
        p_mix = 0.10 * p_f_hat + 0.90 * p_m_hat
        expected_coverage = np.sum(p_mix * weights)
        self.assertAlmostEqual(expected_coverage, 1.0, places=4)

        # Fetal lengths (143 bp) must have weight > 1.0, maternal (166 bp) weight < 1.0
        idx_143 = 143 - FRAG_MIN_LEN
        idx_166 = 166 - FRAG_MIN_LEN
        self.assertGreater(weights[idx_143], 1.0)
        self.assertLess(weights[idx_166], 1.0)

    def test_deconvolve_low_male_fallback(self) -> None:
        # Cohort with 0 males falls back smoothly to parametric templates
        aligned_auto = np.ones((5, FRAG_NUM_BINS), dtype=np.float64) * 100.0
        aligned_chry = np.zeros((5, FRAG_NUM_BINS), dtype=np.float64)
        male_mask = np.zeros(5, dtype=bool)
        ffy_values = np.zeros(5, dtype=np.float64)

        p_f_hat, p_m_hat = deconvolve_fetal_maternal_densities(
            aligned_auto_hists=aligned_auto,
            aligned_chry_hists=aligned_chry,
            male_mask=male_mask,
            ffy_values=ffy_values,
        )
        self.assertAlmostEqual(np.sum(p_f_hat), 1.0, places=5)
        self.assertAlmostEqual(np.sum(p_m_hat), 1.0, places=5)

    def test_feature_vector_and_ensemble_training(self) -> None:
        meta = FragmentomicsMetadata(
            paired_end=True,
            total_paired_fragments=500_000,
            chromatosome_mode_bp=166.0,
            core_shoulder_bp=143.0,
            sizing_shift_bp=0.0,
            self_referenced_peak_ratio=0.35,
            dinucleosome_fraction=0.08,
            high_mw_fraction=0.01,
            helical_10bp_amplitude=0.045,
            helical_10bp_phase=1.2,
            motif_diversity_score=0.91,
            hist_autosomal=[100] * 571,
            hist_chry=[10] * 571,
            hist_chrx=[50] * 571,
            end_motif_counts=[20] * 256,
        )

        feat = build_fragmentomic_feature_vector(meta)
        self.assertEqual(len(feat), 36)
        self.assertFalse(np.any(np.isnan(feat)))
        self.assertFalse(np.any(np.isinf(feat)))

        # Train ensemble on small synthetic cohort
        n_samples = 25
        np.random.seed(42)
        X = np.random.randn(n_samples, 36)
        # FF targets between 0.04 and 0.22
        y = 0.04 + 0.18 * (X[:, 0] - X[:, 0].min()) / (
            X[:, 0].max() - X[:, 0].min() + 1e-6
        )

        weights, bias, mean, scale, loocv_std = fit_ff_fragmentomics_ensemble(X, y)
        self.assertEqual(len(weights), 36)
        self.assertEqual(len(mean), 36)
        self.assertEqual(len(scale), 36)
        self.assertGreater(loocv_std, 0.0)

        # Test predict_ff_fragmentomics
        dummy_model = PrenatalModelMetadata(
            prenatal_bin_size=100_000,
            yfrac_threshold=0.004,
            ffy_intercept=0.0,
            ffy_slope=1.0,
            mlp_w1=[[0.1] * 2] * 50,
            mlp_b1=[0.1, -0.1],
            mlp_w2=[[0.5], [-0.5]],
            mlp_b2=[0.0],
            mlp_skew_intercept=0.0,
            mlp_skew_slope=1.0,
            olm_weights=[0.01] * 50,
            olm_bias=0.0,
            olm_skew_intercept=0.0,
            olm_skew_slope=1.0,
            ffx_intercept=1.0,
            ffx_slope=-0.5,
            female_chrx_ratio_mean=1.0,
            female_chrx_ratio_std=0.01,
            ff_frag_weights=weights.tolist(),
            ff_frag_bias=bias,
            ff_frag_feature_mean=mean.tolist(),
            ff_frag_feature_scale=scale.tolist(),
            ff_size_std=loocv_std,
        )

        pred = predict_ff_fragmentomics(meta, dummy_model)
        self.assertIsNotNone(pred)
        assert pred is not None
        self.assertTrue(0.001 < pred < 0.999)

    def test_segment_fragment_llr(self) -> None:
        lengths = np.arange(FRAG_MIN_LEN, FRAG_MAX_LEN + 1)
        p_f = np.exp(-0.5 * ((lengths - 143.0) / 10.0) ** 2)
        p_f /= np.sum(p_f)
        p_m = np.exp(-0.5 * ((lengths - 166.0) / 12.0) ** 2)
        p_m /= np.sum(p_m)

        # Segment with predominantly short fragments (fetal gain)
        fetal_segment_lengths = np.random.choice(lengths, size=2000, p=p_f)
        short_ratio, llr_gain = segment_fragment_llr(
            lengths=fetal_segment_lengths,
            shift_bp=0.0,
            p_fetal=p_f,
            p_maternal=p_m,
            active_ff=0.10,
            is_gain=True,
        )
        self.assertGreater(short_ratio, 0.40)
        # Positive LLR supports fetal origin over maternal origin
        self.assertGreater(llr_gain, 1.0)

        # Segment with predominantly long maternal fragments
        maternal_segment_lengths = np.random.choice(lengths, size=2000, p=p_m)
        _, llr_mat = segment_fragment_llr(
            lengths=maternal_segment_lengths,
            shift_bp=0.0,
            p_fetal=p_f,
            p_maternal=p_m,
            active_ff=0.10,
            is_gain=True,
        )
        # Negative LLR supports maternal origin over fetal origin
        self.assertLess(llr_mat, -1.0)

        # Empty segment returns 0.0, 0.0
        empty_res = segment_fragment_llr(
            lengths=np.array([], dtype=np.uint16),
            shift_bp=0.0,
            p_fetal=p_f,
            p_maternal=p_m,
            active_ff=0.10,
            is_gain=True,
        )
        self.assertEqual(empty_res, (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
