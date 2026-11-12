"""Regression tests for Round 2 mathematical, statistical, and I/O defect fixes."""

from __future__ import annotations

import json
import math
import re
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pyarrow as pa
from helpers import metadata_for
from numpy.typing import NDArray

import ccnv.models as cm
import ccnv.stats.fragmentomics as sf
import ccnv.stats.neighbors as sn
from ccnv.features.binning import bin_intervals
from ccnv.features.normalization import BranchModel, normalize_sample_branch
from ccnv.features.prenatal import fit_prenatal_models
from ccnv.qc.predict import build_predict_qc_payload
from ccnv.qc.sample import build_sample_qc
from ccnv.report.html import render_genome_plot
from ccnv.stats.pca import fit_whitened_pca, project_huber_pca_wave
from ccnv.stats.prenatal_math import classify_sca_ray_4d
from ccnv.stats.segmentation import ContigSegmentationInput, segment_and_score_genome


def _make_branch_model(
    contigs: tuple[str, ...],
    counts: NDArray[np.float64],
    pca_mean: float,
    pids: NDArray[np.uint16],
    pw: NDArray[np.float16],
) -> BranchModel:
    n = len(contigs)
    starts = np.array(
        [sum(5000 for prev in contigs[:i] if prev == c) for i, c in enumerate(contigs)],
        dtype=np.int64,
    )
    z_f64 = np.zeros(n, dtype=np.float64)
    return BranchModel(
        "autosomal",
        contigs,
        starts,
        starts + 5000,
        counts,
        np.zeros(n, dtype=np.bool_),
        np.full(n, pca_mean, dtype=np.float64),
        np.empty((0, n), dtype=np.float32),
        pids,
        pw,
        np.zeros(n, dtype=np.float32),
        z_f64,
        z_f64.copy(),
    )


def _make_sample_table(
    contig_lengths: dict[str, int],
    counts_list: list[int],
    qc: dict[str, int],
    frag_dict: dict[str, object] | None = None,
) -> pa.Table:
    n = len(counts_list)
    vals: dict[str, object] = {
        "bin_size": 5000,
        "contig_order": list(contig_lengths.keys()),
        "source_identity": {"id": "sample"},
        "reference_identity": {"name": "ref"},
        "qc": qc,
        "filters": asdict(cm.ReadFilters()),
    }
    if frag_dict is not None:
        vals["fragmentomics"] = frag_dict
    meta = metadata_for(
        cm.ArtifactKind.SAMPLE, contig_lengths=contig_lengths, values=vals
    )
    z32 = pa.array(np.zeros(n, dtype=np.uint32))
    c32 = pa.array(counts_list, type=pa.uint32())
    cols = {
        "count": c32,
        "paired_count": c32 if frag_dict is not None else z32,
        "frag_len_sum": pa.array(np.zeros(n, dtype=np.uint64)),
        "short_frag_count": z32,
        "endpoint_count": z32,
        "frag_gc_sum": pa.array(np.zeros(n, dtype=np.float32)),
        "frag_bands": pa.nulls(n, type=pa.list_(pa.uint32(), cm.NUM_FRAG_BANDS)),
    }
    return pa.table(cols, schema=cm.SAMPLE_SCHEMA.with_metadata(meta))


def _make_paired_prenatal_sample(
    contig_lengths: dict[str, int], is_male: bool, ff: float, seed: int
) -> pa.Table:
    rng = np.random.default_rng(seed)
    bp = np.arange(
        sf.FRAG_MIN_LEN, sf.FRAG_MIN_LEN + sf.FRAG_NUM_BINS, dtype=np.float64
    )
    pf = np.exp(-0.5 * ((bp - 143.0) / 10.0) ** 2)
    pf /= np.sum(pf)
    pm = np.exp(-0.5 * ((bp - 166.0) / 11.0) ** 2)
    pm /= np.sum(pm)
    pmix = ff * pf + (1.0 - ff) * pm
    hist_auto = rng.multinomial(60_000, pmix / np.sum(pmix)).astype(np.int64)
    hist_y = (
        rng.multinomial(max(50, int(3_000 * ff)), pf).astype(np.int64)
        if is_male
        else np.zeros(sf.FRAG_NUM_BINS, dtype=np.int64)
    )
    motifs = np.full(256, 200, dtype=np.int64)
    lm = sf.estimate_sample_landmarks(hist_auto, motifs, paired_end=True)
    frag_meta = cm.FragmentomicsMetadata(
        True,
        int(np.sum(hist_auto)),
        lm.chromatosome_mode_bp,
        lm.core_shoulder_bp,
        lm.sizing_shift_bp,
        lm.self_referenced_peak_ratio,
        lm.dinucleosome_fraction,
        lm.high_mw_fraction,
        lm.helical_10bp_amplitude,
        lm.helical_10bp_phase,
        lm.motif_diversity_score,
        hist_auto.tolist(),
        hist_y.tolist(),
        (hist_auto // 20).tolist(),
        motifs.tolist(),
    )
    counts_list: list[int] = []
    for contig, length in contig_lengths.items():
        starts, _ = bin_intervals(length, 5000)
        mean_c = (
            (100.0 * (0.5 * ff) if is_male else 0.05)
            if contig == "chrY"
            else (100.0 * (1.0 - 0.5 * ff) if (contig == "chrX" and is_male) else 100.0)
        )
        counts_list.extend(rng.poisson(lam=mean_c, size=starts.size).tolist())
    qc = {"effective_reads_examined": 60_000, "effective_reads_kept": 60_000}
    return _make_sample_table(contig_lengths, counts_list, qc, frag_meta.to_dict())


class TestRound2DefectRegressions(unittest.TestCase):
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

    def test_normalize_sample_branch_low_depth_poisson_var_and_hsm_scaling(
        self,
    ) -> None:
        pw_low = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float16), (4, 1))
        model_low = _make_branch_model(
            ("chr1", "chr1", "chr2", "chr2"),
            np.full(4, 0.25, dtype=np.float64),
            0.25,
            np.zeros((4, 4), dtype=np.uint16),
            pw_low,
        )
        res_low = normalize_sample_branch(
            model_low.counts, np.ones(4, dtype=np.float32), model_low, mask_repeats=1
        )
        exact_v_ln = float(sn.exact_poisson_log_variance(np.array([0.25]))[0])
        self.assertLess(exact_v_ln, 1.0)
        f_loco = 1.0 + math.pi / 4.0
        expected_w = sn.LN2_SQ / (exact_v_ln * f_loco)
        naive_1_over_mu_w = sn.LN2_SQ / (4.0 * f_loco)
        np.testing.assert_allclose(res_low.weights, expected_w, rtol=1e-4)
        self.assertGreater(float(res_low.weights[0]), 10.0 * naive_1_over_mu_w)

        contigs_hsm = ("chr1",) * 2 + ("chr2",) * 2 + ("chr3",) * 12
        pids_hsm = np.zeros((16, 4), dtype=np.uint16)
        pids_hsm[4:, 0] = 1
        pw_hsm = np.tile(np.array([0.8, 0.2, 0.0, 0.0], dtype=np.float16), (16, 1))
        counts_hsm = np.array([1000.0] * 4 + [2000.0] * 12, dtype=np.float64)
        model_hsm = _make_branch_model(
            contigs_hsm, counts_hsm, 1000.0, pids_hsm, pw_hsm
        )
        res_hsm = normalize_sample_branch(
            counts_hsm, np.ones(16, dtype=np.float32), model_hsm, mask_repeats=1
        )
        np.testing.assert_allclose(res_hsm.expected_counts[4:], 2000.0, rtol=0.02)
        np.testing.assert_allclose(res_hsm.ratios[4:], 0.0, atol=0.02)

    def test_segment_and_score_non_doubled_nb_bf_and_haploid_calling(self) -> None:
        starts = np.arange(10, dtype=np.int64) * 5000
        gain_inp = ContigSegmentationInput(
            starts,
            starts + 5000,
            np.full(10, math.log2(1.5)),
            np.full(10, 200.0),
            50_000,
            counts=np.full(10, 30.0),
            expected=np.full(10, 20.0),
            baseline_ploidy=2,
        )
        segs_dip = segment_and_score_genome({"chr1": gain_inp}, penalty=50.0)
        self.assertEqual((segs_dip[0].call, segs_dip[0].copy_number), ("gain", 3))
        self.assertAlmostEqual(
            segs_dip[0].log_bf or 0.0, 300.0 * math.log(1.5) - 100.0, places=5
        )

        hap_base = replace(gain_inp, weights=np.full(10, 20.0), baseline_ploidy=1)
        hap_loss = replace(hap_base, ratios=np.full(10, -3.0), counts=np.zeros(10))
        hap_dup = replace(hap_base, ratios=np.full(10, 1.0), counts=np.full(10, 40.0))
        by_c = {
            s.contig: s
            for s in segment_and_score_genome(
                {"chrX": hap_loss, "chrY": hap_dup}, penalty=50.0
            )
        }
        self.assertEqual((by_c["chrX"].call, by_c["chrX"].copy_number), ("loss", 0))
        self.assertEqual((by_c["chrY"].call, by_c["chrY"].copy_number), ("gain", 2))

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

        _, d0_hat, _ = sn.squeeze_var(
            rng.chisquare(19.0, size=n_bins) * (phi_true / 19.0), df=19.0
        )
        self.assertGreater(d0_hat, 10.0)

        _, bio_disp = sn.calibrate_prototype_dispersion(
            counts, recon_expected, contigs, usable, pids, pw, pca_rank=0
        )
        self.assertEqual(int(np.count_nonzero(bio_disp <= 0.0)), 0)
        self.assertAlmostEqual(float(np.median(bio_disp)), phi_true, delta=0.005)
        self.assertLess(
            float(np.percentile(bio_disp, 75) - np.percentile(bio_disp, 25)), 0.006
        )

    def test_unlabeled_ff_size_sizing_shift_parity_and_sca_posteriors(self) -> None:
        cl = {"chr1": 200_000, "chr2": 200_000, "chrX": 200_000, "chrY": 300_000}
        m_ffs = (0.05, 0.07, 0.09, 0.12, 0.15, 0.18)
        tables = [
            _make_paired_prenatal_sample(
                cl, i >= 4, m_ffs[i - 4] if i >= 4 else 0.06 + 0.03 * i, 300 + i
            )
            for i in range(10)
        ]
        entries = [
            cm.SamplesheetEntry(
                Path(f"u_{i}.ccnv"), sex="M" if i >= 4 else "F", ff=None
            )
            for i in range(10)
        ]
        meta, _, _ = fit_prenatal_models(tables, entries, prenatal_bin_size=100_000)
        if meta.ff_frag_weights is None:
            self.fail("Expected unlabeled paired-end cohort to train ff_frag_weights")
        self.assertEqual(len(meta.ff_frag_weights), 36)

        bp = np.arange(
            sf.FRAG_MIN_LEN, sf.FRAG_MIN_LEN + sf.FRAG_NUM_BINS, dtype=np.float64
        )
        raw_hist = np.exp(-0.5 * ((bp - 166.0) / 10.0) ** 2) * 10_000.0
        shifted_hist = sf.align_histogram(raw_hist, -3.0)
        motifs = np.full(256, 100.0, dtype=np.float64)
        feat_can = sf.build_fragmentomic_feature_vector(raw_hist, motifs, 166.0, 0.0)
        feat_shf = sf.build_fragmentomic_feature_vector(
            shifted_hist, motifs, 166.0, 3.0
        )
        np.testing.assert_allclose(feat_shf, feat_can, rtol=1e-2, atol=1e-2)

        ens = sf.fit_ff_fragmentomics_ensemble(
            np.vstack([feat_can * (1.0 + 0.02 * k) for k in range(6)]),
            np.linspace(0.05, 0.20, 6),
        )
        kw = {
            "p_fetal_density": meta.p_fetal_density,
            "p_maternal_density": meta.p_maternal_density,
            "ff_frag_weights": ens.weights,
            "ff_frag_bias": ens.intercept,
            "ff_frag_feature_mean": ens.feat_mean,
            "ff_frag_feature_scale": ens.feat_std,
        }
        ff_can = sf.predict_ff_fragmentomics(
            raw_hist, motifs, sizing_shift_bp=0.0, **kw
        )
        ff_shf = sf.predict_ff_fragmentomics(
            shifted_hist, motifs, sizing_shift_bp=3.0, **kw
        )
        if ff_can is None or ff_shf is None:
            self.fail("Expected non-None FF_size predictions")
        self.assertAlmostEqual(ff_shf, ff_can, places=3)

        disc_res = classify_sca_ray_4d(
            0.25, 0.01, 0.05, 0.01, 0.05, 0.015, ff_size=0.05
        )
        self.assertEqual(disc_res.call, "DISCORDANT_MIXED")
        self.assertAlmostEqual(sum(disc_res.posteriors.values()), 1.0, places=6)

        fetal_xo = classify_sca_ray_4d(
            -0.06, 0.008, 0.0, 0.01, 0.12, 0.015, ff_size=0.12
        )
        self.assertEqual(fetal_xo.call, "XO")
        maternal_45x = classify_sca_ray_4d(
            -0.06, 0.008, 0.0, 0.01, 0.12, 0.015, ff_size=0.12, chrx_llr_fetal=-10.0
        )
        self.assertEqual(maternal_45x.call, "DISCORDANT_MIXED")
        self.assertAlmostEqual(sum(maternal_45x.posteriors.values()), 1.0, places=6)

    def test_sample_qc_zero_preservation_and_html_effective_ff_xss_escaping(
        self,
    ) -> None:
        qc_dict = build_sample_qc(
            {
                "reads_seen": 25,
                "reads_kept": 20,
                "effective_reads_examined": 0,
                "effective_reads_kept": 0,
            }
        )
        self.assertEqual(
            (qc_dict["effective_reads_examined"], qc_dict["effective_reads_kept"]),
            (0, 0),
        )
        tbl = _make_sample_table({"chr1": 5000}, [5], qc_dict)
        rm = build_predict_qc_payload("germline", tbl, ())["read_metrics"]
        if not isinstance(rm, dict):
            self.fail("Expected read_metrics dict")
        self.assertEqual(
            (rm["effective_reads_examined"], rm["effective_reads_kept"]), (0, 0)
        )

        bins = (
            cm.BinResult(
                "autosomal",
                ("chr1",),
                np.array([0], dtype=np.int64),
                np.array([50_000], dtype=np.int64),
                np.array([0.12], dtype=np.float64),
                np.array([4.0], dtype=np.float64),
                np.array([1.0], dtype=np.float64),
                np.array([10], dtype=np.int64),
                np.array([False], dtype=np.bool_),
            ),
        )
        base_seg = cm.SegmentResult("chr1", 0, 50_000, 0.12, 4.0, 1.0, 1, "gain")
        seg = replace(
            base_seg,
            expected_ratio=0.08,
            implied_ff=0.18,
            discordant=True,
            discordance_reason="<script>alert(1)</script>",
            frag_short_ratio=0.25,
            frag_llr_size=1.8,
            origin_call="<script>evil()</script>",
        )
        sca = cm.SCAResult("XY", False, 0.99, 0.0)
        prenatal = cm.FetalFractionResult(
            "M", 0.1, "FFY", 0.1, 0.1, 0.1, 0.95, 0.01, 0.004, sca, effective_ff=0.18
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "xss_plot.html"
            render_genome_plot(
                html_path, bins, (seg,), {"chr1": 50_000}, prenatal_result=prenatal
            )
            html = html_path.read_text(encoding="utf-8")

        for forbidden in ("<script>alert(1)</script>", "<script>evil()</script>"):
            self.assertNotIn(forbidden, html)
        for required in (
            "D.prenatal.effective_ff ?? D.prenatal.active_ff",
            "escHtml(originCall)",
            'escHtml(discReason || "YES")',
        ):
            self.assertIn(required, html)

        m = re.search(
            r'<script id="ccnv-data" type="application/json">(.*?)</script>',
            html,
            re.DOTALL,
        )
        if m is None:
            self.fail("Missing ccnv-data script tag")
        payload = json.loads(m.group(1))
        self.assertEqual(payload["segment_calls"], ["gain"])
        p_dict = payload["prenatal"]
        if not isinstance(p_dict, dict):
            self.fail("Expected prenatal payload dict")
        self.assertEqual(p_dict["effective_ff"], 0.18)


if __name__ == "__main__":
    unittest.main()
