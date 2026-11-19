"""Unit and integration tests for within-sample normalization and Fisher cohort moments."""

from __future__ import annotations

import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import ccnv.stats.neighbors as sn
from ccnv.features.binning import bin_intervals, compute_bin_exposure
from ccnv.features.cohort import (
    Cohort,
    build_fisher_weighted_cohort,
    load_and_assemble_cohort,
)
from ccnv.features.normalization import (
    BranchModel,
    BranchNormalizationResult,
    normalize_sample_branch,
)
from ccnv.features.rebinning import rebin_sample_buffers
from ccnv.io.parquet import write_sample_parquet
from ccnv.models import (
    BRANCH_AUTOSOMAL,
    COORDINATE_SYSTEM,
    NUM_FRAG_BANDS,
    NUM_GC_QUANTILES,
    NUM_PROTOTYPES_PER_BIN,
    SCHEMA_VERSION,
    PredictConfig,
    PredictError,
    ReferenceConfig,
    ReferenceMetadata,
    SampleBuffers,
    SampleMetadata,
    SamplesheetEntry,
)
from ccnv.predict import build_branch_model, normalize_branch
from ccnv.reference import build_reference_table
from ccnv.stats.neighbors import calibrate_prototype_dispersion


def _proto_arrays(n: int) -> tuple[NDArray[np.uint16], NDArray[np.float16]]:
    pids = np.zeros((n, 4), dtype=np.uint16)
    pweights = np.zeros((n, 4), dtype=np.float16)
    pweights[:, 0] = np.float16(1.0)
    return pids, pweights


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
        BRANCH_AUTOSOMAL,
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


def _make_sample_buffers(
    contig_lengths: dict[str, int],
    bin_size: int,
    counts: NDArray[np.float64],
    bands: NDArray[np.uint32] | None,
    sample_id: str = "S1",
    qc_reads: int = 10_000,
) -> SampleBuffers:
    del sample_id
    meta = SampleMetadata(
        schema_version=SCHEMA_VERSION,
        coordinate_system=COORDINATE_SYSTEM,
        bin_size=bin_size,
        contig_order=tuple(contig_lengths.keys()),
        contig_lengths=contig_lengths,
        filters={"min_mapping_quality": 1},
        qc={"effective_reads_examined": qc_reads, "effective_reads_kept": qc_reads},
    )
    n = counts.size
    z_u32 = np.zeros(n, dtype=np.uint32)
    paired = (
        np.sum(bands, axis=1, dtype=np.uint32) if bands is not None else z_u32.copy()
    )
    return SampleBuffers(
        meta=meta,
        count=counts.astype(np.float64),
        paired_count=paired,
        frag_len_sum=np.full(n, 16600, dtype=np.uint64),
        short_frag_count=z_u32,
        endpoint_count=z_u32.copy(),
        frag_gc_sum=np.zeros(n, dtype=np.float32),
        frag_bands=bands.astype(np.uint32) if bands is not None else None,
    )


class TestNormalization(unittest.TestCase):
    def test_normalization_produces_aligned_finite_evidence(self) -> None:
        pids, pweights = _proto_arrays(4)
        model = BranchModel(
            branch=BRANCH_AUTOSOMAL,
            contigs=("chr1", "chr1", "chr2", "chr2"),
            starts=np.array([0, 100, 0, 100]),
            ends=np.array([100, 200, 100, 200]),
            counts=np.array([10.0, 10.0, 10.0, 10.0]),
            masked=np.zeros(4, dtype=np.bool_),
            pca_mean=np.full(4, 0.25),
            pca_loadings=np.zeros((1, 4), dtype=np.float32),
            proto_ids=pids,
            proto_weights=pweights,
            mean_distance=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            null_mean=np.zeros(4, dtype=np.float64),
            bio_dispersion=np.full(4, 0.01, dtype=np.float64),
        )
        config = PredictConfig(
            Path("sample"),
            Path("reference"),
            Path("output"),
            min_reference_bins=2,
        )

        result = normalize_branch(model, config)

        self.assertEqual(result.ratios.shape, (4,))
        self.assertTrue(np.all(np.isfinite(result.ratios)))
        self.assertTrue(np.all(np.isfinite(result.zscores)))
        np.testing.assert_allclose(result.ratios, 0.0, atol=1e-6)

    def test_normalize_sample_branch_with_masked_and_empty_active(self) -> None:
        pids, pweights = _proto_arrays(6)
        counts = np.array([20.0, 30.0, 20.0, 30.0, 25.0, 25.0])
        model = BranchModel(
            branch=BRANCH_AUTOSOMAL,
            contigs=("chr1", "chr1", "chr2", "chr2", "chr3", "chr3"),
            starts=np.array([0, 100, 0, 100, 0, 100], dtype=np.int64),
            ends=np.array([100, 200, 100, 200, 100, 200], dtype=np.int64),
            counts=counts,
            masked=np.zeros(6, dtype=np.bool_),
            pca_mean=np.full(6, 1.0),
            pca_loadings=np.zeros((0, 6), dtype=np.float32),
            proto_ids=pids,
            proto_weights=pweights,
            mean_distance=np.ones(6, dtype=np.float32),
            null_mean=np.zeros(6, dtype=np.float64),
            bio_dispersion=np.full(6, 0.01, dtype=np.float64),
            bin_size=100,
        )

        n_mask = np.array([True, True, False, False, False, False])
        norm = normalize_sample_branch(
            counts,
            np.ones(6, dtype=np.float32),
            model,
            n_mask_override=n_mask,
            min_reference_bins=1,
        )
        np.testing.assert_array_equal(
            norm.usable, [False, False, True, True, True, True]
        )
        self.assertLess(norm.ratios[2], 0.0)
        self.assertGreater(norm.ratios[3], 0.0)

        all_masked = np.ones(6, dtype=np.bool_)
        with self.assertRaisesRegex(PredictError, "zero sample coverage"):
            normalize_sample_branch(
                counts,
                np.ones(6, dtype=np.float32),
                model,
                n_mask_override=all_masked,
                min_reference_bins=1,
            )

    def test_normalize_sample_branch_huber_resists_trisomy_outlier_chromosome(
        self,
    ) -> None:
        loadings = np.linspace(-1.0, 1.0, 20, dtype=np.float32).reshape(1, 20)
        pca_mean = np.full(20, 1.0)
        true_c = 0.25
        counts = 100.0 * pca_mean * np.exp(true_c * loadings.astype(np.float64).ravel())
        counts[16:] *= 1.8

        pids, pweights = _proto_arrays(20)
        model = BranchModel(
            branch=BRANCH_AUTOSOMAL,
            contigs=("chr1",) * 8 + ("chr2",) * 8 + ("chr21",) * 4,
            starts=np.tile(np.arange(4, dtype=np.int64) * 100, 5),
            ends=np.tile(np.arange(1, 5, dtype=np.int64) * 100, 5),
            counts=counts,
            masked=np.zeros(20, dtype=np.bool_),
            pca_mean=pca_mean,
            pca_loadings=loadings,
            proto_ids=pids,
            proto_weights=pweights,
            mean_distance=np.ones(20, dtype=np.float32),
            null_mean=np.zeros(20, dtype=np.float64),
            bio_dispersion=np.full(20, 0.005, dtype=np.float64),
            bin_size=100,
        )
        norm = normalize_sample_branch(
            counts, np.ones(20, dtype=np.float32), model, min_reference_bins=2
        )
        self.assertLess(float(np.std(norm.ratios[:16])), 0.08)
        self.assertGreater(float(np.mean(norm.ratios[16:])), 0.5)

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
        mappability[:30] = 0.20

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
            contig_lengths=contig_lengths,
            bin_size=bin_size,
        )
        with tempfile.TemporaryDirectory() as tmp:
            ref_cfg = ReferenceConfig(
                prefix=Path(tmp) / "ref",
                min_cohort_size=10,
                sex_contigs=(),
            )
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
                bio_dispersion=branch_model_15x.bio_dispersion,
                gc=gc.astype(np.float32),
                mappability=mappability.astype(np.float32),
                bin_size=bin_size,
                is_flat=False,
            )

        exposure = compute_bin_exposure(starts, ends, bin_size)
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

    def test_fisher_weighted_cohort_propagates_ref_kappa_frag_and_float64_rebin(
        self,
    ) -> None:
        contig_lengths = {"chr1": 100_000, "chr2": 100_000}
        n_bins = 40
        rng = np.random.default_rng(303)
        band_w = np.linspace(2.2, 0.2, NUM_FRAG_BANDS, dtype=np.float64)
        samples: list[SampleBuffers] = []
        with tempfile.TemporaryDirectory() as tmp:
            paths: list[Path] = []
            for s_idx in range(12):
                bands = rng.poisson(8.0, size=(n_bins, NUM_FRAG_BANDS)).astype(
                    np.uint32
                )
                counts = np.sum(bands, axis=1, dtype=np.float64)
                sb = _make_sample_buffers(
                    contig_lengths, 5000, counts, bands, f"s_{s_idx}", qc_reads=10_000
                )
                p = Path(tmp) / f"s_{s_idx}.ccnv"
                write_sample_parquet(sb.to_table(), p)
                paths.append(p)
                samples.append(sb)

            rebinned_nested = rebin_sample_buffers(
                replace(samples[0], count=samples[0].count + 0.375), 10_000
            )
            self.assertEqual(rebinned_nested.count.dtype, np.float64)
            self.assertAlmostEqual(
                float(np.sum(rebinned_nested.count)),
                float(np.sum(samples[0].count + 0.375)),
                places=6,
            )
            rebinned_interp = rebin_sample_buffers(
                replace(samples[0], count=samples[0].count + 0.375), 7_500
            )
            self.assertEqual(rebinned_interp.count.dtype, np.float64)

            starts_1, ends_1 = bin_intervals(100_000, 5000)
            base_cohort = Cohort(
                contigs=("chr1",) * 20 + ("chr2",) * 20,
                starts=np.tile(starts_1, 2),
                ends=np.tile(ends_1, 2),
                counts=np.column_stack([s.count for s in samples]).astype(np.float32),
                bin_size=5000,
                contig_lengths=contig_lengths,
                paths=tuple(paths),
            )
            weighted = build_fisher_weighted_cohort(
                base_cohort, band_w.tolist(), max_batch_memory_mb=64, threads=1
            )
            if weighted.ref_kappa_frag is None:
                self.fail("expected non-None ref_kappa_frag on Fisher-weighted cohort")
            self.assertTrue(bool(np.all(weighted.ref_kappa_frag > 1.05)))

            exp_mat = np.broadcast_to(
                np.mean(weighted.counts, axis=1, keepdims=True), weighted.counts.shape
            ).copy()
            c_codes = np.array([0] * 20 + [1] * 20, dtype=np.int16)
            pids = np.zeros((n_bins, NUM_PROTOTYPES_PER_BIN), dtype=np.uint16)
            pids[:20, 0] = 1
            pids[20:, 0] = 0
            pw = np.zeros((n_bins, NUM_PROTOTYPES_PER_BIN), dtype=np.float16)
            pw[:, 0] = np.float16(1.0)
            _, disp_unscaled = calibrate_prototype_dispersion(
                weighted.counts,
                exp_mat,
                c_codes,
                np.ones(n_bins, dtype=np.bool_),
                pids,
                pw,
                pca_rank=0,
            )
            _, disp_scaled = calibrate_prototype_dispersion(
                weighted.counts,
                exp_mat,
                c_codes,
                np.ones(n_bins, dtype=np.bool_),
                pids,
                pw,
                pca_rank=0,
                kappa_frag=weighted.ref_kappa_frag,
            )
            self.assertLessEqual(
                float(np.mean(disp_scaled)), float(np.mean(disp_unscaled))
            )

    def test_build_fisher_weighted_cohort_pools_kish_moments_across_samples(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            lengths = {"chr1": 10000, "chrY": 10000}
            weights = np.ones(NUM_FRAG_BANDS, dtype=np.float64)
            weights[0] = 3.0
            weights[1] = 0.25

            entries: list[SamplesheetEntry] = []
            for idx in range(4):
                is_male = idx >= 2
                bands = np.zeros((4, NUM_FRAG_BANDS), dtype=np.uint32)
                bands[:2, 0] = 20
                bands[:2, 1] = 20
                if is_male:
                    bands[2:, 0] = 20
                    bands[2:, 1] = 20
                cnt = np.sum(bands, axis=1).astype(np.float64)
                bufs = _make_sample_buffers(
                    lengths, 5000, cnt, bands, sample_id=f"S{idx}", qc_reads=1000
                )
                p = tmp_dir / f"s{idx}.ccnv"
                write_sample_parquet(bufs.to_table(), p)
                entries.append(
                    SamplesheetEntry(filepath=p, sex="M" if is_male else "F")
                )

            raw_cohort = load_and_assemble_cohort(entries, target_bin_size=5000)
            w_cohort = build_fisher_weighted_cohort(raw_cohort, weights)
            kappa_frag = w_cohort.ref_kappa_frag
            self.assertIsNotNone(kappa_frag)
            if kappa_frag is None:
                self.fail("expected non-None kappa_frag")
            self.assertAlmostEqual(float(kappa_frag[2]), float(kappa_frag[0]), places=5)
            self.assertGreater(float(kappa_frag[2]), 1.35)

    def test_normalize_sample_branch_defaults_to_model_kappa_and_shrinks_low_counts(
        self,
    ) -> None:
        n_bins = 8
        contigs = ("chr1",) * 4 + ("chr2",) * 4
        starts = np.array([0, 5000, 10000, 15000] * 2, dtype=np.int64)
        ends = starts + 5000
        model_hi_kappa = BranchModel(
            branch=BRANCH_AUTOSOMAL,
            contigs=contigs,
            starts=starts,
            ends=ends,
            counts=np.full(n_bins, 50.0, dtype=np.float64),
            masked=np.zeros(n_bins, dtype=np.bool_),
            pca_mean=np.zeros(n_bins, dtype=np.float64),
            pca_loadings=np.zeros((0, n_bins), dtype=np.float32),
            proto_ids=np.array(
                [[1, 1, 1, 1]] * 4 + [[0, 0, 0, 0]] * 4, dtype=np.uint16
            ),
            proto_weights=np.full((n_bins, 4), 0.25, dtype=np.float16),
            mean_distance=np.zeros(n_bins, dtype=np.float32),
            null_mean=np.full(n_bins, 50.0, dtype=np.float64),
            bio_dispersion=np.full(n_bins, 1e-4, dtype=np.float64),
            gc=np.full(n_bins, 0.42, dtype=np.float32),
            mappability=np.ones(n_bins, dtype=np.float32),
            gc_quantiles=np.zeros((n_bins, NUM_GC_QUANTILES), dtype=np.float32),
            ref_frag_len=np.zeros(n_bins, dtype=np.float32),
            ref_ep_rate=np.zeros(n_bins, dtype=np.float32),
            bin_size=5000,
            kappa_frag=np.full(n_bins, 2.0, dtype=np.float32),
        )
        model_unit_kappa = replace(
            model_hi_kappa, kappa_frag=np.ones(n_bins, dtype=np.float32)
        )
        counts_arr = np.full(n_bins, 50.0, dtype=np.float64)
        exp_arr = np.ones(n_bins, dtype=np.float32)
        res_hi = normalize_sample_branch(counts_arr, exp_arr, model_hi_kappa)
        res_unit = normalize_sample_branch(counts_arr, exp_arr, model_unit_kappa)
        self.assertLess(float(res_hi.weights[0]), float(res_unit.weights[0]))


if __name__ == "__main__":
    unittest.main()
