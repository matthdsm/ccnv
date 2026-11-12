"""Regression tests for Round 3 mathematical, statistical, schema, and I/O fixes."""

from __future__ import annotations

import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa

from ccnv._core import FRAG_BAND_EDGES
from ccnv.features.binning import bin_intervals
from ccnv.features.cohort import Cohort, build_fisher_weighted_cohort
from ccnv.features.normalization import BranchModel, normalize_sample_branch
from ccnv.features.prenatal import fit_prenatal_models, predict_prenatal
from ccnv.features.prenatal_origin import (
    BandPrefixSums,
    annotate_prenatal_origin_and_lod,
)
from ccnv.features.rebinning import rebin_sample_buffers
from ccnv.io.parquet import write_sample_parquet
from ccnv.models import (
    BRANCH_MALE_SEX,
    COORDINATE_SYSTEM,
    NUM_FRAG_BANDS,
    NUM_GC_QUANTILES,
    NUM_PROTOTYPES_PER_BIN,
    SCHEMA_VERSION,
    BranchFitArrays,
    ReferenceMetadata,
    SampleBuffers,
    SampleMetadata,
    SamplesheetEntry,
    SchemaError,
    SchemaVersionError,
    SegmentResult,
    assemble_reference_table,
    canonical_dictionary_array,
)
from ccnv.qc.reference import compute_branch_qc
from ccnv.report._common import write_bgzf_atomic
from ccnv.stats.neighbors import calibrate_prototype_dispersion
from ccnv.stats.prenatal_math import (
    classify_sca_ray_4d,
    fit_msy_sex_mixture,
    fit_poisson_msy_ffy,
    fuse_active_fetal_fraction_student_t4,
)
from ccnv.stats.segmentation import _evaluate_integer_cn_bf, _find_breakpoints


def _paired_sample_buffers(
    contig_lengths: dict[str, int],
    bin_size: int,
    counts: np.ndarray,
    bands: np.ndarray,
    sample_id: str,
) -> SampleBuffers:
    meta = SampleMetadata(
        schema_version=SCHEMA_VERSION,
        coordinate_system=COORDINATE_SYSTEM,
        bin_size=bin_size,
        contig_order=tuple(contig_lengths.keys()),
        contig_lengths=contig_lengths,
        source_identity={"id": sample_id},
        reference_identity={"name": "ref"},
        filters={"min_mapping_quality": 1},
        qc={"effective_reads_examined": 10_000, "effective_reads_kept": 10_000},
    )
    n = counts.size
    z_u32 = np.zeros(n, dtype=np.uint32)
    return SampleBuffers(
        meta=meta,
        count=counts.astype(np.float64),
        paired_count=np.sum(bands, axis=1, dtype=np.uint32),
        frag_len_sum=np.full(n, 16600, dtype=np.uint64),
        short_frag_count=z_u32,
        endpoint_count=z_u32.copy(),
        frag_gc_sum=np.zeros(n, dtype=np.float32),
        frag_bands=bands.astype(np.uint32),
    )


class TestRound3DefectRegressions(unittest.TestCase):
    def test_classify_sca_ray_4d_bounded_ray_d2_and_tail_log_ndtr(self) -> None:
        gain_res = classify_sca_ray_4d(
            delta_x=0.12,
            sigma_x=0.01,
            ffy=0.0,
            sigma_ffy=0.01,
            ffa=None,
            sigma_ffa=0.02,
            ff_size=None,
            clinical_priors={
                "XX": 0.90,
                "XO": 0.05,
                "XXX": 0.05,
                "XY": 1e-6,
                "XXY": 1e-6,
                "XYY": 1e-6,
            },
        )
        self.assertEqual(gain_res.call, "XXX")
        self.assertNotEqual(gain_res.call, "XO")

        extreme_gain = classify_sca_ray_4d(
            delta_x=0.85,
            sigma_x=0.01,
            ffy=0.0,
            sigma_ffy=0.01,
            ffa=None,
            sigma_ffa=0.02,
            ff_size=None,
        )
        self.assertEqual(extreme_gain.call, "DISCORDANT_MIXED")
        self.assertGreater(extreme_gain.mahalanobis_d2, 20.0)

    def test_integer_cn_bf_calls_cn0_with_proportional_background_and_low_depth(
        self,
    ) -> None:
        cn_hap, bf_hap = _evaluate_integer_cn_bf(
            seg_ratio=-5.5,
            sigma_s_sq=0.02,
            baseline_ploidy=1,
            has_counts=True,
            c_s=100.0,
            lam_s=5000.0,
        )
        self.assertEqual(cn_hap, 0)
        self.assertGreater(bf_hap, 100.0)

        cn_dip, bf_dip = _evaluate_integer_cn_bf(
            seg_ratio=-5.5,
            sigma_s_sq=0.01,
            baseline_ploidy=2,
            has_counts=True,
            c_s=100.0,
            lam_s=5000.0,
        )
        self.assertEqual(cn_dip, 0)
        self.assertGreater(bf_dip, 100.0)

        cn_low, bf_low = _evaluate_integer_cn_bf(
            seg_ratio=-4.0,
            sigma_s_sq=0.25,
            baseline_ploidy=2,
            has_counts=True,
            c_s=0.0,
            lam_s=10.0,
        )
        self.assertEqual(cn_low, 0)
        self.assertGreater(bf_low, 4.5)

    def test_fit_poisson_msy_ffy_origin_constrained_calibration_slope(self) -> None:
        e_mat = np.full((4, 6), 1000.0, dtype=np.float64)
        c_mat = np.zeros((4, 6), dtype=np.float64)
        fem_mask = np.array([True, True, True, False, False, False])
        male_mask = ~fem_mask
        labeled_ff = np.array([np.nan, np.nan, np.nan, 0.098, 0.100, 0.102])
        raw_male_ff = np.array([0.092, 0.085, 0.079])
        for j, idx in enumerate([3, 4, 5]):
            c_mat[:, idx] = 0.5 * raw_male_ff[j] * e_mat[:, idx]

        model = fit_poisson_msy_ffy(
            c_mat,
            e_mat,
            np.ones(4, dtype=np.float64),
            fem_mask,
            male_mask,
            labeled_ff,
        )
        c_sub = c_mat[:, 3:]
        e_sub = e_mat[:, 3:]
        raw_post_bg = (
            2.0
            * (np.sum(c_sub, axis=0) - model.lambda_bg @ e_sub)
            / np.sum(e_sub, axis=0)
        )
        expected_slope = float(
            np.dot(raw_post_bg, labeled_ff[3:]) / np.dot(raw_post_bg, raw_post_bg)
        )
        self.assertAlmostEqual(model.kappa_cal, expected_slope, places=6)
        self.assertGreater(model.kappa_cal, 1.15)

        unimodal = fit_msy_sex_mixture(
            np.array([10.0, 11.0, 9.0, 10.0, 12.0, 10.0]),
            np.full(6, 1000.0),
        )
        self.assertFalse(unimodal.bimodal)

    def test_find_breakpoints_scale_invariance_small_weights_and_zero_weights(
        self,
    ) -> None:
        vals = np.concatenate([np.zeros(12), np.full(12, 2.0)])
        base_w = np.tile(np.array([1.0, 10.0], dtype=np.float64), 12)
        bkps_unit = _find_breakpoints(
            signal=np.column_stack((vals, base_w)),
            penalty=15.0,
            min_size=2,
            jump=1,
        )
        bkps_tiny = _find_breakpoints(
            signal=np.column_stack((vals, base_w * 1e-9)),
            penalty=15.0 * 1e-9,
            min_size=2,
            jump=1,
        )
        self.assertEqual(bkps_tiny, bkps_unit)
        self.assertEqual(bkps_tiny, [12, 24])

        non_contig_vals = np.column_stack((vals, vals))[::2, 0]
        self.assertFalse(non_contig_vals.flags["C_CONTIGUOUS"])
        bkps_uniform = _find_breakpoints(
            signal=np.column_stack((non_contig_vals, np.ones_like(non_contig_vals))),
            penalty=4.0,
            min_size=2,
            jump=1,
        )
        self.assertEqual(bkps_uniform, [6, 12])

        bkps_zero = _find_breakpoints(
            signal=np.column_stack((vals, np.zeros_like(vals))),
            penalty=4.0,
            min_size=2,
            jump=1,
        )
        self.assertEqual(bkps_zero, [24])

    def test_student_t4_fusion_respects_cramer_rao_se_and_converged_weights(
        self,
    ) -> None:
        f_eq, se_eq = fuse_active_fetal_fraction_student_t4([0.10, 0.10], [0.01, 0.01])
        self.assertAlmostEqual(f_eq, 0.10, places=6)
        self.assertAlmostEqual(se_eq, 0.01 / math.sqrt(2.0), places=6)

        f_out, se_out = fuse_active_fetal_fraction_student_t4(
            [0.10, 0.10, 0.45], [0.01, 0.01, 0.01]
        )
        self.assertLess(abs(f_out - 0.10), 0.005)
        self.assertGreater(se_out, 0.01 / math.sqrt(3.0))

    def test_prenatal_origin_short_frag_ratio_spans_under_150bp_and_empty_usable(
        self,
    ) -> None:
        band_138_150 = int(np.searchsorted(FRAG_BAND_EDGES, 143, side="right") - 1)
        band_150_162 = int(np.searchsorted(FRAG_BAND_EDGES, 155, side="right") - 1)
        self.assertEqual(band_138_150, 9)
        self.assertEqual(band_150_162, 10)

        bands = np.zeros((2, NUM_FRAG_BANDS), dtype=np.uint32)
        bands[0, band_138_150] = 60
        bands[0, band_150_162] = 40
        bands[1, 0] = 500
        ps = BandPrefixSums(
            bands,
            np.array([0, 0], dtype=np.int16),
            ["chr1"],
            np.array([0, 5000], dtype=np.int64),
            np.array([5000, 10000], dtype=np.int64),
            bin_usable=np.array([True, False]),
        )
        seg = SegmentResult("chr1", 0, 5000, 0.08, 4.0, 20.0, 1, "gain")
        annotated, _ = annotate_prenatal_origin_and_lod(
            [seg],
            prefix_sums=ps,
            sample_bands=bands,
            bin_usable=np.array([False, False]),
            active_ff=0.10,
        )
        self.assertAlmostEqual(float(annotated[0].frag_short_ratio or 0.0), 0.60)

    def test_compute_branch_qc_male_chry_usable_pct_uses_total_branch_chry_bins(
        self,
    ) -> None:
        n_x, n_y = 20, 100
        contigs = ("chrX",) * n_x + ("chrY",) * n_y
        starts = np.concatenate(
            [
                np.arange(n_x, dtype=np.int64) * 5000,
                np.arange(n_y, dtype=np.int64) * 5000,
            ]
        )
        masked = np.zeros(n_x + n_y, dtype=np.bool_)
        masked[n_x + 1 :] = True
        pids = np.zeros((n_x + n_y, NUM_PROTOTYPES_PER_BIN), dtype=np.uint16)
        pweights = np.zeros((n_x + n_y, NUM_PROTOTYPES_PER_BIN), dtype=np.float16)
        pweights[~masked, 0] = np.float16(1.0)
        pids[n_x, 0] = np.uint16(1)
        pids[:n_x, 0] = np.uint16(1)

        fit = BranchFitArrays(
            branch=BRANCH_MALE_SEX,
            contigs=contigs,
            starts=starts,
            ends=starts + 5000,
            masked=masked,
            pca_mean=np.ones(n_x + n_y, dtype=np.float32),
            proto_ids=pids,
            proto_weights=pweights,
            mean_distance=np.full(n_x + n_y, 0.5, dtype=np.float32),
        )
        ref_meta = ReferenceMetadata(
            bin_size=5000,
            contig_lengths={"chrX": n_x * 5000, "chrY": n_y * 5000},
            branches=(BRANCH_MALE_SEX,),
            autosomal_contigs=(),
            sex_contigs=("chrX", "chrY"),
            sex_signal_contig="chrY",
            pca_components=1,
        )
        table = assemble_reference_table(
            [fit],
            contig_order=("chrX", "chrY"),
            metadata=ref_meta.build_arrow_metadata(),
        )
        qc = compute_branch_qc(table, BRANCH_MALE_SEX, "chrY", min_ref_bins=1)
        if qc is None:
            self.fail("expected non-None male_sex branch QC dictionary")
        self.assertEqual(qc["verdict"], "FAIL")
        sig = qc["signal_contig_metrics"]
        if not isinstance(sig, dict):
            self.fail("expected signal_contig_metrics dictionary")
        self.assertEqual(sig["n_bins"], 100)
        self.assertAlmostEqual(float(sig["usable_pct"]), 1.0)

    def test_gc_override_without_quantiles_clears_stored_fasta_quantiles(self) -> None:
        n = 40
        contigs = ("chr1",) * 20 + ("chr2",) * 20
        starts = np.tile(np.arange(20, dtype=np.int64) * 50_000, 2)
        gc_new = np.linspace(0.35, 0.65, n, dtype=np.float64)
        counts = 500.0 * np.exp(1.8 * (gc_new - 0.50))
        stale_gc_q = np.zeros((n, NUM_GC_QUANTILES), dtype=np.float32)
        stale_gc_q[:, 0] = 1.0
        pids = np.zeros((n, NUM_PROTOTYPES_PER_BIN), dtype=np.uint16)
        pids[:20, 0] = 1
        pids[20:, 0] = 0
        pw = np.zeros((n, NUM_PROTOTYPES_PER_BIN), dtype=np.float16)
        pw[:, 0] = np.float16(1.0)

        model = BranchModel(
            branch="autosomal",
            contigs=contigs,
            starts=starts,
            ends=starts + 50_000,
            counts=counts,
            masked=np.zeros(n, dtype=np.bool_),
            pca_mean=np.ones(n, dtype=np.float64),
            pca_loadings=np.empty((0, n), dtype=np.float32),
            proto_ids=pids,
            proto_weights=pw,
            mean_distance=np.zeros(n, dtype=np.float32),
            null_mean=np.zeros(n, dtype=np.float64),
            bio_dispersion=np.zeros(n, dtype=np.float64),
            gc=np.full(n, 0.50, dtype=np.float64),
            gc_quantiles=stale_gc_q,
            bin_size=50_000,
        )
        res = normalize_sample_branch(
            counts,
            np.ones(n, dtype=np.float32),
            model,
            mask_repeats=1,
            min_reference_bins=1,
            gc_override=gc_new,
        )
        self.assertLess(float(np.std(res.ratios)), 0.05)

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
                sb = _paired_sample_buffers(
                    contig_lengths, 5000, counts, bands, f"s_{s_idx}"
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
                identities=tuple({"path": str(p)} for p in paths),
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

    def test_canonical_dictionary_and_metadata_post_init_contracts(self) -> None:
        ordered_dict = pa.DictionaryArray.from_arrays(
            pa.array([0, 1], type=pa.int8()),
            pa.array(["autosomal", "female_sex"], type=pa.string()),
            ordered=True,
        )
        canon_ord = canonical_dictionary_array(
            ordered_dict, "branch", index_type=pa.int8()
        )
        self.assertEqual(canon_ord.type, pa.dictionary(pa.int8(), pa.string()))
        self.assertFalse(canon_ord.type.ordered)

        large_str_arr = pa.array(["chr1", "chr2"], type=pa.large_string())
        canon_large = canonical_dictionary_array(
            large_str_arr, "contig", index_type=pa.int16()
        )
        self.assertEqual(canon_large.type, pa.dictionary(pa.int16(), pa.string()))

        cats_128 = [f"b_{i}" for i in range(128)]
        canon_128 = canonical_dictionary_array(
            pa.array(cats_128, type=pa.string()), "branch", index_type=pa.int8()
        )
        self.assertEqual(len(canon_128.dictionary), 128)
        with self.assertRaises(SchemaError):
            canonical_dictionary_array(
                pa.array([f"b_{i}" for i in range(129)], type=pa.string()),
                "branch",
                index_type=pa.int8(),
            )

        with self.assertRaises(SchemaVersionError):
            SampleMetadata(
                schema_version="1.0.0",
                coordinate_system=COORDINATE_SYSTEM,
                bin_size=5000,
                contig_order=("chr1",),
                contig_lengths={"chr1": 5000},
                source_identity={"id": "s"},
                reference_identity={"name": "ref"},
                filters={"min_mapping_quality": 1},
                qc={"effective_reads_examined": 1, "effective_reads_kept": 1},
            )
        with self.assertRaises(SchemaError):
            ReferenceMetadata(
                schema_version=SCHEMA_VERSION,
                coordinate_system="1-based-closed",
                bin_size=5000,
                contig_lengths={"chr1": 5000},
                branches=("autosomal",),
                autosomal_contigs=("chr1",),
                sex_contigs=(),
                pca_components=0,
            )

    def test_write_bgzf_atomic_unlinks_stale_opposite_index_and_linear_ffa_mode(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bed_gz = Path(tmp) / "out.bed.gz"
            write_bgzf_atomic(
                bed_gz,
                "bed",
                ["chr1"],
                100_000,
                [b"#contig\tstart\tend\nchr1\t0\t5000\n"],
                threads=1,
            )
            self.assertTrue(Path(f"{bed_gz}.tbi").exists())
            write_bgzf_atomic(
                bed_gz,
                "bed",
                ["chr1"],
                600_000_000,
                [b"#contig\tstart\tend\nchr1\t0\t5000\n"],
                threads=1,
            )
            self.assertTrue(Path(f"{bed_gz}.csi").exists())
            self.assertFalse(Path(f"{bed_gz}.tbi").exists())

        contig_lengths = {
            "chr1": 200_000,
            "chr2": 200_000,
            "chrX": 200_000,
            "chrY": 100_000,
        }
        rng = np.random.default_rng(404)
        samples: list[SampleBuffers] = []
        entries: list[SamplesheetEntry] = []
        for idx in range(6):
            is_m = idx >= 3
            ff_val = 0.12 if is_m else 0.08
            bands = rng.poisson(6.0, size=(140, NUM_FRAG_BANDS)).astype(np.uint32)
            counts = np.sum(bands, axis=1, dtype=np.float64)
            if not is_m:
                counts[120:] = 0.0
            samples.append(
                _paired_sample_buffers(contig_lengths, 5000, counts, bands, f"p_{idx}")
            )
            entries.append(
                SamplesheetEntry(
                    filepath=Path(f"p_{idx}.ccnv"), sex="M" if is_m else "F", ff=ff_val
                )
            )
        p_meta, p_fits, _ = fit_prenatal_models(
            samples, entries, prenatal_bin_size=100_000, seed=42
        )
        linear_meta = replace(
            p_meta,
            ffa_logit_mode=False,
            olm_weights=[0.0] * len(p_meta.olm_weights),
            olm_bias=0.1234,
        )
        ref_meta = ReferenceMetadata(
            bin_size=5000,
            contig_lengths=contig_lengths,
            branches=tuple(f.branch for f in p_fits),
            autosomal_contigs=("chr1", "chr2"),
            sex_contigs=("chrX", "chrY"),
            sex_signal_contig="chrY",
            pca_components=1,
            mode="prenatal",
            prenatal_bin_size=100_000,
            prenatal=linear_meta,
        )
        ref_tbl = assemble_reference_table(
            p_fits,
            contig_order=tuple(contig_lengths.keys()),
            metadata=ref_meta.build_arrow_metadata(),
        )
        pred = predict_prenatal(samples[0], ref_tbl, linear_meta)
        self.assertAlmostEqual(pred.ffa, 0.1234, places=6)


if __name__ == "__main__":
    unittest.main()
