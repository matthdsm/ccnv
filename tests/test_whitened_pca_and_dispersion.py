"""Tests for whitened outlier-clipped PCA, depth-invariant EB dispersion, and normalization."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ccnv.features.cohort import Cohort, compute_bin_exposure
from ccnv.features.normalization import (
    BranchModel,
    BranchNormalizationResult,
    build_branch_model,
    normalize_sample_branch,
)
from ccnv.io.parquet import read_reference_parquet
from ccnv.models import BRANCH_AUTOSOMAL, ReferenceConfig, ReferenceMetadata
from ccnv.reference import build_reference_table
from ccnv.stats.neighbors import (
    calibrate_prototype_dispersion,
    fit_prototype_graph,
)
from ccnv.stats.pca import WhitenedPCAResult, fit_whitened_pca


class TestWhitenedPCAAndDispersion(unittest.TestCase):
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
        self.assertTrue(np.all(pca_pure.mean_rate[~usable] == 0.0))
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
        # Inject 15% cohort-frequency CNP deletion on usable bins 200..240 (3 CN=0, 3 CN=1)
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

        # U_u < S triggers randomized_svd path
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

    def test_full_cohort_dispersion_uses_all_samples_and_is_depth_invariant(
        self,
    ) -> None:
        rng = np.random.default_rng(101)
        n_contigs, bins_per_contig = 6, 200
        n_bins = n_contigs * bins_per_contig
        n_samples = 150  # Exceeds legacy 100-sample cap
        contigs = np.repeat(np.arange(n_contigs, dtype=np.int16), bins_per_contig)
        usable = np.ones(n_bins, dtype=np.bool_)
        usable[::40] = False
        u_idx = np.flatnonzero(usable)

        feats = rng.normal(0.0, 1.0, size=(u_idx.size, 2)).astype(np.float32)
        centers, pids_u, pw_u, _ = fit_prototype_graph(
            feats, max_prototypes=32, top_r=4
        )
        n_protos = int(centers.shape[0])

        # 1) Tail-only overdispersion in samples 100..149 proves all S=150 samples are used
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
            n_prototypes=n_protos,
        )
        noisy_u = usable & (np.arange(n_bins) < 200)
        clean_u = usable & (np.arange(n_bins) >= 200)
        self.assertGreater(
            float(np.median(disp_tail[noisy_u])),
            float(np.median(disp_tail[clean_u])) + 0.01,
        )

        # 2) Pure Poisson cohort has near-zero biological dispersion
        pure_counts = rng.poisson(exp_150).astype(np.float32)
        null_m_pure, disp_pure = calibrate_prototype_dispersion(
            pure_counts,
            exp_150,
            contigs,
            usable,
            pids_u,
            pw_u,
            pca_rank=0,
            n_prototypes=n_protos,
        )
        self.assertLess(float(np.median(disp_pure[usable])), 1e-3)
        self.assertLess(float(np.median(np.abs(null_m_pure[usable]))), 0.03)

        # 3) Depth invariance of bio_dispersion across 1.5x (lam=30) vs 0.4x (lam=8)
        true_phi = np.zeros(n_bins, dtype=np.float64)
        true_phi[:400] = 0.0144  # sigma_ln = 0.12
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
                n_prototypes=n_protos,
            )
            disp_estimates.append(float(np.median(disp_d[usable & (true_phi > 0)])))
            self.assertLess(float(np.median(disp_d[usable & (true_phi == 0)])), 0.003)

        for est in disp_estimates:
            self.assertGreater(est, 0.008)
            self.assertLess(est, 0.022)
        self.assertLess(abs(disp_estimates[0] - disp_estimates[1]), 0.006)

    def test_cross_depth_01x_sample_against_15x_reference_maintains_unit_z_variance(
        self,
    ) -> None:
        rng = np.random.default_rng(4242)
        bin_size = 5_000
        n_contigs, bins_per_contig = 6, 400
        n_bins = n_contigs * bins_per_contig
        n_samples = 32

        contig_names = [f"chr{i + 1}" for i in range(n_contigs)]
        contig_lengths = {c: bins_per_contig * bin_size for c in contig_names}
        contigs = tuple(c for c in contig_names for _ in range(bins_per_contig))
        starts = np.tile(
            np.arange(bins_per_contig, dtype=np.int64) * bin_size, n_contigs
        )
        ends = starts + bin_size

        gc = rng.uniform(0.32, 0.62, size=n_bins)
        mappability = rng.uniform(0.75, 1.0, size=n_bins)
        mappability[:30] = 0.20  # Masked low-mappability region

        grid = np.linspace(0.0, 3.0 * np.pi, n_bins, dtype=np.float64)
        w1 = np.sin(grid)
        w2 = np.cos(2.0 * grid)
        mu_bin = np.clip(rng.lognormal(0.0, 0.08, size=n_bins), 0.7, 1.4)
        sf_ref = rng.uniform(0.8, 1.25, size=n_samples)

        ref_counts = np.empty((n_bins, n_samples), dtype=np.float32)
        for s in range(n_samples):
            gc_slope = rng.normal(0.8, 0.25)
            wave = rng.normal(0.0, 0.12) * w1 + rng.normal(0.0, 0.09) * w2
            lam_s = 25.0 * sf_ref[s] * mu_bin * np.exp(gc_slope * (gc - 0.45) + wave)
            lam_s[:30] = 0.0
            ref_counts[:, s] = rng.poisson(lam_s).astype(np.float32)

        cohort = Cohort(
            paths=tuple(Path(f"ref_{i}.ccnv") for i in range(n_samples)),
            contigs=contigs,
            starts=starts,
            ends=ends,
            counts=ref_counts,
            identities=tuple(f"ref_{i}" for i in range(n_samples)),
            contig_lengths=contig_lengths,
            bin_size=bin_size,
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Persist synthetic GC/mappability via cohort_gc hooks through build_reference_table
            ref_cfg = ReferenceConfig(
                prefix=tmp_path / "ref",
                min_cohort_size=10,
                sex_contigs=(),
            )
            # Inject GC/mappability directly via _build_branch_table-equivalent table build
            ref_tbl = build_reference_table(cohort, ref_cfg)
            ref_meta = ReferenceMetadata.from_schema(ref_tbl.schema)
            branch_model_15x = build_branch_model(ref_tbl, BRANCH_AUTOSOMAL, ref_meta)
            branch_model_15x = BranchModel(
                branch=branch_model_15x.branch,
                contigs=branch_model_15x.contigs,
                starts=branch_model_15x.starts,
                ends=branch_model_15x.ends,
                counts=branch_model_15x.counts,
                masked=mappability < 0.50,
                pca_mean=branch_model_15x.pca_mean,
                pca_loadings=branch_model_15x.pca_loadings,
                proto_ids=branch_model_15x.proto_ids,
                proto_weights=branch_model_15x.proto_weights,
                mean_distance=branch_model_15x.mean_distance,
                null_mean=branch_model_15x.null_mean,
                null_std=branch_model_15x.null_std,
                gc=gc.astype(np.float32),
                mappability=mappability.astype(np.float32),
                bio_dispersion=branch_model_15x.bio_dispersion,
                bin_size=bin_size,
                is_flat=False,
            )

        exposure = compute_bin_exposure(starts, ends, bin_size)
        # Test sample at 0.1x (lambda ~ 1.67, ~19% zeros on mappable bins)
        lam_01x = 1.67 * mu_bin * np.exp(0.6 * (gc - 0.45) + 0.10 * w1 - 0.08 * w2)
        lam_01x[:30] = 0.0
        counts_01x = rng.poisson(lam_01x).astype(np.float64)
        mappable_zero_count_bins = (mappability >= 0.50) & (counts_01x == 0.0)
        self.assertGreater(int(np.count_nonzero(mappable_zero_count_bins)), 150)

        norm = normalize_sample_branch(counts_01x, exposure, branch_model_15x)
        self.assertIsInstance(norm, BranchNormalizationResult)
        self.assertTrue(bool(np.all(norm.usable[mappable_zero_count_bins])))
        self.assertTrue(bool(np.all(norm.weights[mappable_zero_count_bins] > 0.0)))
        self.assertTrue(bool(np.all(norm.ratios[mappable_zero_count_bins] < -0.5)))

        lambda_gc_01x = float(np.nanmedian(norm.zscores[norm.usable] ** 2) / 0.4549364)
        self.assertGreaterEqual(lambda_gc_01x, 0.85)
        self.assertLessEqual(lambda_gc_01x, 1.15)

        # Matched 1.5x test sample also maintains unit Z variance
        lam_15x = 25.0 * mu_bin * np.exp(0.9 * (gc - 0.45) - 0.11 * w1 + 0.07 * w2)
        lam_15x[:30] = 0.0
        counts_15x = rng.poisson(lam_15x).astype(np.float64)
        norm_15x = normalize_sample_branch(counts_15x, exposure, branch_model_15x)
        lambda_gc_15x = float(
            np.nanmedian(norm_15x.zscores[norm_15x.usable] ** 2) / 0.4549364
        )
        self.assertGreaterEqual(lambda_gc_15x, 0.85)
        self.assertLessEqual(lambda_gc_15x, 1.15)

    def test_whole_arm_trisomy_hsm_centering_and_coarse_donor_masking(self) -> None:
        rng = np.random.default_rng(909)
        bin_size = 5_000
        n_contigs, bins_per_contig = 5, 400
        n_bins = n_contigs * bins_per_contig
        contig_names = [f"chr{i + 1}" for i in range(n_contigs)]
        contig_lengths = {c: bins_per_contig * bin_size for c in contig_names}
        contigs = tuple(c for c in contig_names for _ in range(bins_per_contig))
        starts = np.tile(
            np.arange(bins_per_contig, dtype=np.int64) * bin_size, n_contigs
        )
        ends = starts + bin_size

        ref_counts = rng.poisson(40.0, size=(n_bins, 24)).astype(np.float32)
        cohort = Cohort(
            paths=tuple(Path(f"s_{i}.ccnv") for i in range(24)),
            contigs=contigs,
            starts=starts,
            ends=ends,
            counts=ref_counts,
            identities=tuple(f"s_{i}" for i in range(24)),
            contig_lengths=contig_lengths,
            bin_size=bin_size,
        )
        ref_cfg = ReferenceConfig(
            prefix=Path("unused"), min_cohort_size=10, sex_contigs=()
        )
        ref_tbl = build_reference_table(cohort, ref_cfg)
        model = build_branch_model(
            ref_tbl, BRANCH_AUTOSOMAL, ReferenceMetadata.from_schema(ref_tbl.schema)
        )

        # Trisomy on chr1 (bins 0..400 = 20% of genome at 1.5x), diploid chr2..chr5
        lam_test = np.full(n_bins, 40.0, dtype=np.float64)
        lam_test[:bins_per_contig] = 60.0
        test_counts = rng.poisson(lam_test).astype(np.float64)
        exposure = compute_bin_exposure(starts, ends, bin_size)

        norm = normalize_sample_branch(test_counts, exposure, model, mask_repeats=3)
        self.assertTrue(bool(np.all(norm.usable)))
        diploid_med = float(np.median(norm.ratios[bins_per_contig:]))
        trisomy_med = float(np.median(norm.ratios[:bins_per_contig]))
        self.assertAlmostEqual(diploid_med, 0.0, delta=0.035)
        self.assertAlmostEqual(trisomy_med, float(np.log2(1.5)), delta=0.05)

    def test_gc_quantiles_roundtrip_and_flat_fasta_distance_weights(self) -> None:
        rng = np.random.default_rng(31415)
        bin_size = 50_000  # >= 25 kb triggers 8-bin 200 bp GC quantile basis
        bins_per_contig = 24
        contig_names = ("chr1", "chr2", "chr3")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fasta_path = tmp_path / "ref.fa"
            lines: list[str] = []
            for c_idx, cname in enumerate(contig_names):
                lines.append(f">{cname}")
                chunks: list[str] = []
                for b in range(bins_per_contig):
                    gc_prob = 0.30 + 0.35 * (((b + 3 * c_idx) % 8) / 7.0)
                    bases = rng.choice(
                        np.array(["A", "T", "G", "C"]),
                        size=bin_size,
                        p=[
                            0.5 * (1.0 - gc_prob),
                            0.5 * (1.0 - gc_prob),
                            0.5 * gc_prob,
                            0.5 * gc_prob,
                        ],
                    )
                    chunks.append("".join(bases.tolist()))
                seq = "".join(chunks)
                for i in range(0, len(seq), 80):
                    lines.append(seq[i : i + 80])
            fasta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            flat_cfg = ReferenceConfig(
                fasta=fasta_path,
                bin_size=bin_size,
                prefix=tmp_path / "flat_ref",
                min_cohort_size=2,
                qc_report=False,
            )
            from ccnv.reference import run as run_reference

            ref_path = run_reference(flat_cfg)
            loaded_tbl, loaded_meta = read_reference_parquet(ref_path)
            flat_model = build_branch_model(loaded_tbl, BRANCH_AUTOSOMAL, loaded_meta)
            self.assertTrue(flat_model.is_flat)
            self.assertIsNotNone(flat_model.gc_quantiles)
            if flat_model.gc_quantiles is not None:
                self.assertEqual(
                    flat_model.gc_quantiles.shape,
                    (len(contig_names) * bins_per_contig, 8),
                )

            # Vary mean_distance artificially to verify flat-FASTA distance weight scaling
            varied_dist = np.linspace(
                0.05, 2.0, flat_model.counts.size, dtype=np.float32
            )
            flat_model_varied = BranchModel(
                branch=flat_model.branch,
                contigs=flat_model.contigs,
                starts=flat_model.starts,
                ends=flat_model.ends,
                counts=flat_model.counts,
                masked=flat_model.masked,
                pca_mean=flat_model.pca_mean,
                pca_loadings=flat_model.pca_loadings,
                proto_ids=flat_model.proto_ids,
                proto_weights=flat_model.proto_weights,
                mean_distance=varied_dist,
                null_mean=flat_model.null_mean,
                null_std=flat_model.null_std,
                gc=flat_model.gc,
                mappability=flat_model.mappability,
                gc_quantiles=flat_model.gc_quantiles,
                bio_dispersion=flat_model.bio_dispersion,
                bin_size=flat_model.bin_size,
                is_flat=True,
            )
            exposure = compute_bin_exposure(
                flat_model.starts, flat_model.ends, bin_size
            )
            sample_counts = rng.poisson(35.0, size=flat_model.counts.size).astype(
                np.float64
            )
            norm_flat = normalize_sample_branch(
                sample_counts, exposure, flat_model_varied
            )
            self.assertGreater(
                float(norm_flat.weights[0]), float(norm_flat.weights[-1]) * 5.0
            )


if __name__ == "__main__":
    unittest.main()
