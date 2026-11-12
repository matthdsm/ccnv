"""Regression tests for Round 4 Rust, schema, stats, CLI, prenatal, and report fixes."""

from __future__ import annotations

import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from typer.testing import CliRunner

from ccnv._core import (
    scan_alignment_bins,
    scan_bigwig_means,
    scan_fasta_gc_quantiles,
    write_bgzf_indexed,
)
from ccnv.cli import app
from ccnv.features import (
    BandPrefixSums,
    BranchModel,
    annotate_prenatal_origin_and_lod,
    annotate_sample_prenatal_segments,
    build_fisher_weighted_cohort,
    load_and_assemble_cohort,
    normalize_sample_branch,
    rebin_counts_matrix,
    resolve_fisher_moment_weights,
)
from ccnv.features.prenatal import _compute_effective_ff
from ccnv.features.prenatal_origin import _integrate_mosaic_state
from ccnv.io.parquet import write_sample_parquet
from ccnv.models import (
    BRANCH_AUTOSOMAL,
    COORDINATE_SYSTEM,
    NUM_FRAG_BANDS,
    NUM_GC_QUANTILES,
    NUM_PROTOTYPES_PER_BIN,
    SCHEMA_VERSION,
    AlignmentError,
    BranchFitArrays,
    CovariateError,
    FetalFractionResult,
    InvalidInputError,
    PrenatalModelMetadata,
    SampleBuffers,
    SampleMetadata,
    SamplesheetEntry,
    SCAResult,
    SchemaError,
    SegmentResult,
    reconstruct_sample_grid,
)
from ccnv.qc.reference import _evaluate_base_support
from ccnv.report._common import (
    resolve_segments_dataframe,
    segments_to_dataframe,
)
from ccnv.stats.fragmentomics import encode_kmer_4, evaluate_band_weighted_counts
from ccnv.stats.neighbors import _resolve_prototype_weights, squeeze_var
from ccnv.stats.pca import fit_whitened_pca
from ccnv.stats.prenatal_math import compute_autosomal_density
from ccnv.stats.segmentation import (
    ContigSegmentationInput,
    segment_and_score_genome,
    segment_contig,
)


def _make_sample_buffers(
    contig_lengths: dict[str, int],
    bin_size: int,
    counts: NDArray[np.float64],
    bands: NDArray[np.uint32] | None,
    paired_counts: NDArray[np.uint32] | None = None,
    sample_id: str = "S1",
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
        qc={"effective_reads_examined": 1000, "effective_reads_kept": 1000},
    )
    n = counts.size
    z_u32 = np.zeros(n, dtype=np.uint32)
    p_cnt = (
        paired_counts
        if paired_counts is not None
        else (
            np.sum(bands, axis=1, dtype=np.uint32)
            if bands is not None
            else z_u32.copy()
        )
    )
    return SampleBuffers(
        meta=meta,
        count=counts.astype(np.float64),
        paired_count=p_cnt,
        frag_len_sum=np.full(n, 16600, dtype=np.uint64),
        short_frag_count=z_u32,
        endpoint_count=z_u32.copy(),
        frag_gc_sum=np.zeros(n, dtype=np.float32),
        frag_bands=bands,
    )


class TestRound4DefectRegressions(unittest.TestCase):
    def test_rust_signed_parameter_validation_and_bgzf_overflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            with self.assertRaises(AlignmentError):
                scan_alignment_bins(str(tmp_dir / "missing.bam"), [("chr1", 1000)], 0)
            with self.assertRaises(AlignmentError):
                scan_alignment_bins(str(tmp_dir / "missing.bam"), [("chr1", -10)], 5000)
            with self.assertRaises(AlignmentError):
                scan_alignment_bins(
                    str(tmp_dir / "missing.bam"), [("chr1", 1000)], 5000, threads=0
                )
            with self.assertRaises(AlignmentError):
                scan_alignment_bins(
                    str(tmp_dir / "missing.bam"),
                    [("chr1", 1000)],
                    5000,
                    se_fragment_length=-1,
                )

            with self.assertRaises(CovariateError):
                scan_fasta_gc_quantiles(
                    str(tmp_dir / "missing.fa"), [("chr1", 1000)], 0
                )
            with self.assertRaises(CovariateError):
                scan_fasta_gc_quantiles(
                    str(tmp_dir / "missing.fa"), [("chr1", -1)], 5000
                )
            with self.assertRaises(CovariateError):
                scan_bigwig_means(str(tmp_dir / "missing.bw"), [("chr1", 1000)], 0)
            with self.assertRaises(CovariateError):
                scan_bigwig_means(str(tmp_dir / "missing.bw"), [("chr1", -5)], 5000)

            gz_path = str(tmp_dir / "out.vcf.gz")
            with self.assertRaises(ValueError):
                write_bgzf_indexed(gz_path, "vcf", ["chr1"], 1000, [b""], threads=0)
            with self.assertRaises(ValueError):
                write_bgzf_indexed(gz_path, "vcf", ["chr1"], -1, [b""], threads=1)

            overflow_vcf = (
                b"##fileformat=VCFv4.2\n"
                b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                b"chr1\t10\t.\tN\t<DUP>\t.\tPASS\tEND=99999999999999999999999999\n"
            )
            with self.assertRaises(ValueError):
                write_bgzf_indexed(
                    gz_path, "vcf", ["chr1"], 100_000, [overflow_vcf], threads=1
                )

            overflow_bed = b"chr1\t18446744073709551615\t18446744073709551615\n"
            with self.assertRaises(ValueError):
                write_bgzf_indexed(
                    str(tmp_dir / "out.bed.gz"),
                    "bed",
                    ["chr1"],
                    100_000,
                    [overflow_bed],
                    threads=1,
                )

    def test_int16_contig_dictionary_bound_32768(self) -> None:
        names_ok = tuple(f"c{i}" for i in range(32_768))
        lengths_ok = {c: 5000 for c in names_ok}
        meta_ok = SampleMetadata(
            schema_version=SCHEMA_VERSION,
            coordinate_system=COORDINATE_SYSTEM,
            bin_size=5000,
            contig_order=names_ok,
            contig_lengths=lengths_ok,
            source_identity={"id": "ok"},
            reference_identity={"name": "ref"},
            filters={},
            qc={},
        )
        codes, decoded_names, starts, ends = reconstruct_sample_grid(meta_ok)
        self.assertEqual(len(decoded_names), 32_768)
        self.assertEqual(int(codes[-1]), 32_767)
        self.assertEqual(starts.size, 32_768)
        self.assertEqual(ends.size, 32_768)

        names_bad = tuple(f"c{i}" for i in range(32_769))
        lengths_bad = {c: 5000 for c in names_bad}
        meta_bad = SampleMetadata(
            schema_version=SCHEMA_VERSION,
            coordinate_system=COORDINATE_SYSTEM,
            bin_size=5000,
            contig_order=names_bad,
            contig_lengths=lengths_bad,
            source_identity={"id": "bad"},
            reference_identity={"name": "ref"},
            filters={},
            qc={},
        )
        with self.assertRaises(SchemaError):
            reconstruct_sample_grid(meta_bad)

        n_bad = len(names_bad)
        fit_bad = BranchFitArrays(
            branch=BRANCH_AUTOSOMAL,
            contigs=names_bad,
            starts=np.zeros(n_bad, dtype=np.int64),
            ends=np.full(n_bad, 5000, dtype=np.int64),
            masked=np.zeros(n_bad, dtype=np.bool_),
            gc=np.full(n_bad, 0.42, dtype=np.float32),
            mappability=np.ones(n_bad, dtype=np.float32),
            gc_quantiles=np.zeros((n_bad, NUM_GC_QUANTILES), dtype=np.float32),
            ref_frag_len=np.zeros(n_bad, dtype=np.float32),
            ref_ep_rate=np.zeros(n_bad, dtype=np.float32),
            pca_mean=np.zeros(n_bad, dtype=np.float32),
            pca_loadings=np.zeros((n_bad, 0), dtype=np.float32),
            proto_ids=np.zeros((n_bad, NUM_PROTOTYPES_PER_BIN), dtype=np.uint16),
            proto_weights=np.zeros((n_bad, NUM_PROTOTYPES_PER_BIN), dtype=np.float16),
            mean_distance=np.zeros(n_bad, dtype=np.float32),
            null_mean=np.ones(n_bad, dtype=np.float32),
            bio_dispersion=np.full(n_bad, 1e-3, dtype=np.float32),
        )
        with self.assertRaises(SchemaError):
            fit_bad.to_table()

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

    def test_pca_rank_cap_and_near_zero_singular_value_filtering(self) -> None:
        counts = np.full((30, 12), 100.0, dtype=np.float64)
        expected = np.full((30, 12), 100.0, dtype=np.float64)
        usable = np.ones(30, dtype=np.bool_)
        fit = fit_whitened_pca(counts, expected, usable, max_components=4)
        self.assertEqual(fit.singular_values.size, 0)
        self.assertEqual(fit.loadings.shape, (0, 30))

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
        expected_kappa = (100.0 * expected_w2) / (expected_w * expected_w)
        self.assertAlmostEqual(float(w_counts[0]), expected_w, places=6)
        self.assertAlmostEqual(float(kappa[0]), expected_kappa, places=6)

    def test_segmentation_validation_and_integer_cn_sign_concordance(self) -> None:
        starts = np.array([0, 5000, 10000, 15000], dtype=np.int64)
        ends = np.array([5000, 10000, 15000, 20000], dtype=np.int64)
        ratios = np.full(4, -0.5, dtype=np.float64)
        weights = np.full(4, 25.0, dtype=np.float64)

        with self.assertRaises(InvalidInputError):
            segment_contig(
                contig="chr1",
                starts=starts,
                ends=ends,
                ratios=ratios,
                weights=weights,
                contig_length=20000,
                sensitivity=0.0,
            )
        with self.assertRaises(InvalidInputError):
            segment_contig(
                contig="chr1",
                starts=starts,
                ends=ends,
                ratios=ratios,
                weights=weights,
                contig_length=20000,
                kappa_local=math.nan,
            )

        inp = ContigSegmentationInput(
            starts=starts,
            ends=ends,
            ratios=ratios,
            weights=weights,
            contig_length=20000,
            zscores=np.full(4, -2.5, dtype=np.float64),
            counts=np.full(4, 180.0, dtype=np.float64),
            expected=np.full(4, 100.0, dtype=np.float64),
        )
        for bad_kwargs in (
            {"sensitivity": 0.0},
            {"z_threshold": -1.0},
            {"ratio_threshold": -0.1},
            {"threads": 0},
        ):
            with self.assertRaises(InvalidInputError):
                segment_and_score_genome({"chr1": inp}, **bad_kwargs)

        segs = segment_and_score_genome({"chr1": inp}, z_threshold=3.0)
        self.assertEqual(len(segs), 1)
        self.assertIsNone(segs[0].call)
        self.assertEqual(segs[0].copy_number, 2)

    def test_compute_autosomal_density_zero_coverage_floor(self) -> None:
        counts = np.zeros(8, dtype=np.float64)
        cids = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int16)
        exp = np.ones(8, dtype=np.float32)
        mask = np.ones(8, dtype=np.bool_)
        density = compute_autosomal_density(counts, cids, exp, exp, exp, mask)
        self.assertAlmostEqual(density, 0.001, places=6)

    def test_cli_rejects_zero_thresholds_and_nan_unit_floats(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            sample_p = tmp_dir / "s.ccnv"
            ref_p = tmp_dir / "r.ccnv"
            sample_p.write_bytes(b"placeholder")
            ref_p.write_bytes(b"placeholder")

            for bad_args in (
                ["predict", str(sample_p), str(ref_p), "--penalty", "0"],
                ["predict", str(sample_p), str(ref_p), "--z-threshold", "0"],
                ["predict", str(sample_p), str(ref_p), "--ratio-threshold", "0"],
                ["predict", str(sample_p), str(ref_p), "--min-mappability", "nan"],
                [
                    "reference",
                    "--prefix",
                    str(tmp_dir / "out"),
                    "--sex-cutoff",
                    "nan",
                ],
            ):
                res = runner.invoke(app, bad_args)
                self.assertNotEqual(res.exit_code, 0)

    def test_rebin_counts_matrix_rejects_non_contiguous_contig_runs(self) -> None:
        counts = np.ones((3, 2), dtype=np.float64)
        src_starts = np.array([0, 0, 5000], dtype=np.int64)
        src_ends = np.array([5000, 5000, 10000], dtype=np.int64)
        tgt_starts = np.array([0, 0], dtype=np.int64)
        tgt_ends = np.array([10000, 5000], dtype=np.int64)
        with self.assertRaises(ValueError):
            rebin_counts_matrix(
                counts,
                ["chr1", "chr2", "chr1"],
                src_starts,
                src_ends,
                ["chr1", "chr2"],
                tgt_starts,
                tgt_ends,
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
                    lengths, 5000, cnt, bands, sample_id=f"S{idx}"
                )
                p = tmp_dir / f"s{idx}.ccnv"
                write_sample_parquet(bufs.to_table(), p)
                entries.append(
                    SamplesheetEntry(
                        filepath=p,
                        sex="M" if is_male else "F",
                    )
                )

            raw_cohort = load_and_assemble_cohort(entries, target_bin_size=5000)
            w_cohort = build_fisher_weighted_cohort(raw_cohort, weights)
            kappa_frag = w_cohort.ref_kappa_frag
            self.assertIsNotNone(kappa_frag)
            if kappa_frag is None:
                self.fail("expected non-None kappa_frag")
            self.assertAlmostEqual(
                float(kappa_frag[2]),
                float(kappa_frag[0]),
                places=5,
            )
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

    def test_mosaic_quadrature_captures_narrow_interior_and_boundary_peaks(
        self,
    ) -> None:
        p_f = np.full(NUM_FRAG_BANDS, 1.0 / NUM_FRAG_BANDS, dtype=np.float64)
        p_m = p_f.copy()
        k_s = np.full(NUM_FRAG_BANDS, 10.0, dtype=np.float64)
        m_true = 0.317
        f_bio = 0.12
        r_obs = 1.0 + 0.5 * m_true * f_bio
        log_ev, m_hat, var_m = _integrate_mosaic_state(
            0.05,
            0.95,
            0.5,
            f_bio,
            f_bio,
            0.0002,
            is_fetal=True,
            r_obs=r_obs,
            sigma_r=0.0003,
            k_band_counts=k_s,
            p_fetal_24=p_f,
            p_maternal_24=p_m,
            alpha_dm=10000.0,
        )
        self.assertTrue(math.isfinite(log_ev))
        self.assertAlmostEqual(m_hat, m_true, places=3)
        self.assertGreater(var_m, 0.0)

        log_ev_b, m_hat_b, _ = _integrate_mosaic_state(
            0.05,
            0.95,
            0.5,
            f_bio,
            f_bio,
            0.0002,
            is_fetal=True,
            r_obs=1.15,
            sigma_r=0.0005,
            k_band_counts=k_s,
            p_fetal_24=p_f,
            p_maternal_24=p_m,
            alpha_dm=10000.0,
        )
        self.assertTrue(math.isfinite(log_ev_b))
        self.assertGreater(m_hat_b, 0.90)

    def test_resolve_fisher_moment_weights_and_clamped_active_ff_plasma(self) -> None:
        lengths = {"chr1": 10000}
        bands = np.ones((2, NUM_FRAG_BANDS), dtype=np.uint32)
        pe_bufs = _make_sample_buffers(
            lengths, 5000, np.array([24.0, 24.0], dtype=np.float64), bands
        )
        se_bufs = _make_sample_buffers(
            lengths, 5000, np.array([24.0, 24.0], dtype=np.float64), None
        )
        meta = PrenatalModelMetadata(
            prenatal_bin_size=100_000,
            yfrac_threshold=1e-4,
            ffy_slope=2.0,
            olm_weights=[0.0],
            olm_bias=-2.2,
            female_chrx_ratio_mean=1.0,
            female_chrx_ratio_std=0.01,
            band_fisher_weights=[1.0] * NUM_FRAG_BANDS,
            mu_fetal_w=1.5,
            mu_maternal_w=0.9,
            p_fetal_density=None,
            p_maternal_density=None,
        )
        self.assertEqual(resolve_fisher_moment_weights(pe_bufs, meta), (1.5, 0.9))
        self.assertEqual(resolve_fisher_moment_weights(se_bufs, meta), (1.0, 1.0))

        eff_pe = _compute_effective_ff(0.10, pe_bufs, meta)
        expected_eff = (0.10 * 1.5) / (0.10 * 1.5 + 0.90 * 0.9)
        self.assertAlmostEqual(eff_pe, expected_eff, places=6)
        self.assertAlmostEqual(_compute_effective_ff(0.10, se_bufs, meta), 0.10)

        seg = SegmentResult(
            contig="chr1",
            start=0,
            end=10000,
            ratio=math.log2(1.0 + 0.5 * expected_eff),
            zscore=5.0,
            weight=100.0,
            bin_count=2,
            call="gain",
        )
        ff_res = FetalFractionResult(
            fetal_sex="F",
            ffy=None,
            ffx=0.10,
            ffa=0.10,
            active_ff=0.10,
            active_ff_source="FFA",
            chry_score=0.0,
            yfrac_threshold=1e-4,
            chrx_ratio=1.0,
            sca=SCAResult(
                call="XX",
                is_aneuploid=False,
                confidence=0.99,
                chrx_zscore=0.0,
                ff_delta=None,
            ),
        )
        ann_segs, _ = annotate_sample_prenatal_segments(
            (seg,), pe_bufs, prenatal_meta=meta, prenatal_result=ff_res
        )
        self.assertAlmostEqual(
            float(ann_segs[0].expected_ratio or 0.0),
            math.log2(1.0 + 0.5 * expected_eff),
            places=6,
        )

        ps = BandPrefixSums.from_sample_table(pe_bufs)
        ann_clamped, _ = annotate_prenatal_origin_and_lod(
            (seg,),
            prefix_sums=ps,
            sample_bands=bands,
            active_ff=1.5,
            active_ff_se=0.01,
            sca_call="XX",
            sample_depth_per_mb=1000.0,
        )
        self.assertIsNotNone(ann_clamped[0].origin_call)

    def test_resolve_segments_dataframe_sorts_prebuilt_dataframe_when_requested(
        self,
    ) -> None:
        segs = (
            SegmentResult("chr10", 0, 5000, 0.4, 4.0, 20.0, 1, "gain"),
            SegmentResult("chr2", 0, 5000, -0.4, -4.0, 20.0, 1, "loss"),
        )
        unsorted_df = segments_to_dataframe(segs, is_prenatal=False)
        sorted_df = resolve_segments_dataframe(
            (),
            unsorted_df,
            calls_only=False,
            is_prenatal=False,
            sort_natural=True,
        )
        self.assertEqual(sorted_df.get_column("contig").to_list(), ["chr2", "chr10"])

    def test_reference_qc_low_refs_uses_total_unmasked_bins(self) -> None:
        n_val, (verdict, msg) = _evaluate_base_support(
            {"n_bins": 1000, "n_masked": 0, "n_valid": 200, "n_low_refs": 4},
            min_ref_bins=3,
        )
        self.assertEqual(n_val, 200)
        self.assertEqual(verdict, "PASS")
        self.assertEqual(msg, "")


if __name__ == "__main__":
    unittest.main()
