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

        # Low-FFY (< 0.08) males are excluded from both eligible-male and female pools
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
            aligned_auto_hists=auto_mix,
            aligned_chry_hists=chry_mix,
            male_mask=male_mix,
            ffy_values=ffy_mix,
        )
        self.assertGreater(p_f_clean[143 - FRAG_MIN_LEN], 0.10)
        self.assertAlmostEqual(p_f_clean[250 - FRAG_MIN_LEN], 0.0, places=6)

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
        self.assertGreaterEqual(loocv_std, 0.015)

        with self.assertRaises(ValueError):
            fit_ff_fragmentomics_ensemble(X[:4], y[:4])

        # Test predict_ff_fragmentomics
        dummy_model = PrenatalModelMetadata(
            prenatal_bin_size=100_000,
            yfrac_threshold=0.004,
            ffy_intercept=0.0,
            ffy_slope=1.0,
            olm_weights=[0.01] * 50,
            olm_bias=0.0,
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

        # Empty segment or unknown-only (0) lengths return 0.0, 0.0 and 0s are ignored
        empty_res = segment_fragment_llr(
            lengths=np.array([], dtype=np.uint16),
            shift_bp=0.0,
            p_fetal=p_f,
            p_maternal=p_m,
            active_ff=0.10,
            is_gain=True,
        )
        self.assertEqual(empty_res, (0.0, 0.0))
        zero_only = segment_fragment_llr(
            lengths=np.array([0, 0, 0], dtype=np.uint16),
            shift_bp=0.0,
            p_fetal=p_f,
            p_maternal=p_m,
            active_ff=0.10,
            is_gain=True,
        )
        self.assertEqual(zero_only, (0.0, 0.0))
        sr_padded, llr_padded = segment_fragment_llr(
            lengths=np.concatenate(
                [maternal_segment_lengths, np.zeros(500, dtype=int)]
            ),
            shift_bp=0.0,
            p_fetal=p_f,
            p_maternal=p_m,
            active_ff=0.10,
            is_gain=True,
        )
        sr_clean, llr_clean = segment_fragment_llr(
            lengths=maternal_segment_lengths,
            shift_bp=0.0,
            p_fetal=p_f,
            p_maternal=p_m,
            active_ff=0.10,
            is_gain=True,
        )
        self.assertAlmostEqual(sr_padded, sr_clean)
        self.assertAlmostEqual(llr_padded, llr_clean)

    def test_evaluate_weighted_coverage(self) -> None:
        from helpers import make_sample_table

        from ccnv.features.fragmentomics import evaluate_weighted_coverage

        # 3 bins: bin 0 has lengths [100, 150], bin 1 is empty, bin 2 has length [200]
        frag_lens = [[100, 150], [], [200]]
        table = make_sample_table(
            [2, 0, 1],
            contigs=["chr1", "chr1", "chr1"],
            starts=[0, 100, 200],
            ends=[100, 200, 300],
            fragment_lengths=frag_lens,
        )

        weights_571 = np.ones(571, dtype=np.float64)
        # Band [100, 110) -> weight 2.0; band [150, 160) -> weight 0.5; band [200, 220) -> weight 3.0
        weights_571[100 - 30 : 110 - 30] = 2.0
        weights_571[150 - 30 : 160 - 30] = 0.5
        weights_571[200 - 30 : 220 - 30] = 3.0

        eff_cov = evaluate_weighted_coverage(table, weights_571, shift_bp=0.0)
        self.assertEqual(len(eff_cov), 3)
        self.assertAlmostEqual(eff_cov[0], 2.0 + 0.5)
        self.assertAlmostEqual(eff_cov[1], 0.0)
        self.assertAlmostEqual(eff_cov[2], 3.0)

        # Fallback to count when fragment_lengths is empty (single-end)
        table_se = make_sample_table(
            [5, 10, 15],
            contigs=["chr1", "chr1", "chr1"],
            starts=[0, 100, 200],
            ends=[100, 200, 300],
            fragment_lengths=[[], [], []],
        )
        eff_se = evaluate_weighted_coverage(table_se, weights_571)
        np.testing.assert_allclose(eff_se, [5.0, 10.0, 15.0])

    def test_extract_bin_fragmentomic_covariates(self) -> None:
        from helpers import make_sample_table

        from ccnv.features.fragmentomics import extract_bin_fragmentomic_covariates

        frag_lens = [[120, 140], [], [180]]
        table = make_sample_table(
            [2, 0, 1],
            contigs=["chr1", "chr1", "chr1"],
            starts=[0, 100, 200],
            ends=[100, 200, 300],
            fragment_lengths=frag_lens,
            endpoint_counts=[4, 0, 2],
        )

        mean_lens, ep_ratios = extract_bin_fragmentomic_covariates(table)
        self.assertEqual(len(mean_lens), 3)
        self.assertEqual(len(ep_ratios), 3)

        # Bin 0: mean(120, 140) = 130.0
        self.assertAlmostEqual(mean_lens[0], 130.0)
        # Bin 2: 180.0
        self.assertAlmostEqual(mean_lens[2], 180.0)
        # Bin 1 (empty): imputed to median of non-empty bins = median([130, 180]) = 155.0
        self.assertAlmostEqual(mean_lens[1], 155.0)

        # Endpoint ratios: 4/2 = 2.0, 0/1 = 0.0, 2/1 = 2.0
        self.assertAlmostEqual(ep_ratios[0], 2.0)
        self.assertAlmostEqual(ep_ratios[1], 0.0)
        self.assertAlmostEqual(ep_ratios[2], 2.0)

        # Out-of-range sentinel 0 in fragment_lengths is excluded from mean/count
        table_with_zeros = make_sample_table(
            [3, 1, 1],
            contigs=["chr1", "chr1", "chr1"],
            starts=[0, 100, 200],
            ends=[100, 200, 300],
            fragment_lengths=[[0, 120, 140], [0], [180]],
            endpoint_counts=[4, 0, 2],
        )
        mean_z, _ = extract_bin_fragmentomic_covariates(table_with_zeros)
        np.testing.assert_allclose(mean_z, [130.0, 155.0, 180.0])

    def test_fit_cohort_fragmentomics_model_fewer_than_five_samples_skips_ensemble(
        self,
    ) -> None:
        from ccnv.features.fragmentomics import fit_cohort_fragmentomics_model

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
            hist_autosomal=[100] * 571,
            hist_chry=[10] * 571,
            hist_chrx=[50] * 571,
            end_motif_counts=[20] * 256,
        )
        frag_metas = [meta, meta, meta]
        is_male = np.array([True, False, True], dtype=bool)
        final_ffy = np.array([0.10, 0.0, 0.12], dtype=float)
        target_ffa = np.array([0.10, 0.08, 0.12], dtype=float)

        fit = fit_cohort_fragmentomics_model(frag_metas, is_male, final_ffy, target_ffa)
        self.assertIsNotNone(fit)
        assert fit is not None
        p_fet, p_mat, w_frag, b_frag, m_frag, s_frag, loocv_std = fit
        self.assertIsNotNone(p_fet)
        self.assertEqual(len(p_fet), 571)
        self.assertIsNotNone(p_mat)
        self.assertEqual(len(p_mat), 571)
        self.assertIsNone(w_frag)
        self.assertIsNone(b_frag)
        self.assertIsNone(m_frag)
        self.assertIsNone(s_frag)
        self.assertEqual(loocv_std, 0.018)

    def test_align_reference_to_weighted_sample(self) -> None:
        from helpers import make_reference_table, make_sample_table

        from ccnv.features.fragmentomics import align_reference_to_weighted_sample

        ref_table = make_reference_table(
            branches=["autosomal", "autosomal", "autosomal"],
            contigs=["chr2", "chr1", "chr1"],
            starts=[0, 0, 100],
            ends=[100, 100, 200],
            contig_lengths={"chr1": 200, "chr2": 100},
            pca_mean=[30.0, 10.0, 20.0],
            null_mean=[0.01, -0.02, 0.03],
        )
        # Diploid baseline ratios: chr1:0 = 15/10 = 1.5, chr2:0 = 45/30 = 1.5; CNV bin chr1:100 = 40/20 = 2.0
        # Median baseline scale = median([1.5, 1.5, 2.0]) = 1.5; log2 null_mean stays invariant!
        u_sample = make_sample_table(
            [10, 20, 30],
            contigs=["chr1", "chr1", "chr2"],
            starts=[0, 100, 0],
            ends=[100, 200, 100],
        )
        w_sample = make_sample_table(
            [15, 40, 45],
            contigs=["chr1", "chr1", "chr2"],
            starts=[0, 100, 0],
            ends=[100, 200, 100],
        )

        aligned_ref = align_reference_to_weighted_sample(ref_table, u_sample, w_sample)
        self.assertEqual(
            aligned_ref.column("contig").to_pylist(), ["chr2", "chr1", "chr1"]
        )
        np.testing.assert_allclose(
            aligned_ref.column("pca_mean").to_numpy(), [45.0, 15.0, 30.0], rtol=1e-5
        )
        np.testing.assert_allclose(
            aligned_ref.column("null_mean").to_numpy(), [0.01, -0.02, 0.03], rtol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
