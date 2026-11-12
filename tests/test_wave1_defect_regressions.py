"""Regression tests covering all 7 Wave 1 correctness, statistical, and I/O fixes."""

from __future__ import annotations

import gzip
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
import yaml

import ccnv.predict
from ccnv.features.cohort import (
    _alloc_cohort_counts,
    build_fisher_weighted_cohort,
    load_and_assemble_cohort,
)
from ccnv.features.fragmentomics import apply_prenatal_fisher_weighting
from ccnv.features.prenatal import fit_prenatal_models, predict_prenatal
from ccnv.features.prenatal_origin import (
    BandPrefixSums,
    annotate_prenatal_origin_and_lod,
    classify_segment_origin_5state,
)
from ccnv.io.parquet import write_sample_parquet
from ccnv.io.tracks import load_gc_track
from ccnv.models import (
    BRANCH_AUTOSOMAL,
    NUM_FRAG_BANDS,
    BranchFitArrays,
    FetalFractionResult,
    FragmentomicsMetadata,
    PredictConfig,
    PrenatalModelMetadata,
    ReferenceConfig,
    SampleBuffers,
    SampleMetadata,
    SamplesheetEntry,
    SCAResult,
    SegmentResult,
    assemble_reference_table,
    parquet_path,
)
from ccnv.predict import run as run_predict
from ccnv.reference import run as run_reference
from ccnv.stats.prenatal_math import classify_sca_ray_4d, fit_poisson_msy_ffy


def _make_paired_sample_buffers(
    contig_lengths: dict[str, int],
    bin_size: int,
    counts: np.ndarray,
    frag_bands: np.ndarray,
    hist_auto: np.ndarray,
    hist_chry: np.ndarray,
    sample_id: str,
) -> SampleBuffers:
    counts_u32 = np.asarray(counts, dtype=np.uint32)
    bands_u32 = np.asarray(frag_bands, dtype=np.uint32)
    frag_meta = FragmentomicsMetadata(
        paired_end=True,
        total_paired_fragments=int(np.sum(hist_auto)),
        chromatosome_mode_bp=166.0,
        core_shoulder_bp=143.0,
        sizing_shift_bp=0.0,
        self_referenced_peak_ratio=0.35,
        dinucleosome_fraction=0.08,
        high_mw_fraction=0.01,
        helical_10bp_amplitude=0.04,
        helical_10bp_phase=1.2,
        motif_diversity_score=0.92,
        hist_autosomal=[int(x) for x in hist_auto],
        hist_chry=[int(x) for x in hist_chry],
        hist_chrx=[int(x) for x in (hist_auto // 20)],
        end_motif_counts=[100] * 256,
    )
    meta = SampleMetadata(
        bin_size=bin_size,
        contig_lengths=contig_lengths,
        contig_order=tuple(contig_lengths.keys()),
        source_identity={"id": sample_id},
        reference_identity={"name": "synthetic"},
        qc={
            "effective_reads_examined": int(np.sum(counts_u32)),
            "effective_reads_kept": int(np.sum(counts_u32)),
            "autosomal_reads_kept": int(np.sum(counts_u32)),
            "sex_reads_kept": 0,
        },
        filters={"min_mapping_quality": 1},
        fragmentomics=frag_meta,
    )
    short_counts = np.sum(bands_u32[:, :8], axis=1, dtype=np.uint32)
    return SampleBuffers(
        meta=meta,
        count=counts_u32,
        paired_count=counts_u32.copy(),
        frag_len_sum=(counts_u32.astype(np.uint64) * 165),
        short_frag_count=short_counts,
        endpoint_count=(counts_u32 * 2).astype(np.uint32),
        frag_gc_sum=(counts_u32.astype(np.float32) * 0.45),
        frag_bands=bands_u32,
    )


class TestWave1DefectRegressions(unittest.TestCase):
    def test_defect1_prenatal_reference_and_sample_fisher_weighting_parity(
        self,
    ) -> None:
        contig_lengths = {
            "chr1": 200_000,
            "chr2": 200_000,
            "chrX": 200_000,
            "chrY": 200_000,
        }
        bin_size = 5000
        n_bins = sum(length // bin_size for length in contig_lengths.values())

        lengths_bp = np.arange(30, 601, dtype=np.float64)
        fetal_pdf = np.exp(-0.5 * ((lengths_bp - 143.0) / 12.0) ** 2)
        fetal_pdf /= np.sum(fetal_pdf)
        maternal_pdf = np.exp(-0.5 * ((lengths_bp - 168.0) / 14.0) ** 2)
        maternal_pdf /= np.sum(maternal_pdf)

        rng = np.random.default_rng(101)
        # Alternating bins have markedly different short-vs-long fragment composition
        bin_short_bias = np.tile([0.45, 0.08], n_bins // 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_paths: list[Path] = []
            entries: list[SamplesheetEntry] = []

            for i in range(6):
                is_male = i >= 3
                ff = 0.06 + 0.04 * (i - 3) if is_male else 0.10
                counts = np.full(n_bins, 200, dtype=np.uint32)
                counts[120:] = round(200 * 0.5 * ff) if is_male else 1
                counts[80:120] = round(200 * (1.0 - 0.5 * ff)) if is_male else 200

                bands = np.zeros((n_bins, NUM_FRAG_BANDS), dtype=np.uint32)
                for b in range(n_bins):
                    p_short = bin_short_bias[b]
                    c_tot = int(counts[b])
                    c_short = round(c_tot * p_short)
                    bands[b, 4] = c_short
                    bands[b, 10] = max(0, c_tot - c_short)

                mix_pdf = ff * fetal_pdf + (1.0 - ff) * maternal_pdf
                hist_auto = rng.multinomial(50_000, mix_pdf).astype(np.uint64)
                hist_chry = (
                    rng.multinomial(2_000, fetal_pdf).astype(np.uint64)
                    if is_male
                    else rng.multinomial(20, maternal_pdf).astype(np.uint64)
                )
                bufs = _make_paired_sample_buffers(
                    contig_lengths,
                    bin_size,
                    counts,
                    bands,
                    hist_auto,
                    hist_chry,
                    sample_id=f"s_{i}",
                )
                sp = root / f"s_{i}.ccnv"
                write_sample_parquet(bufs.to_table(), sp)
                sample_paths.append(sp)
                entries.append(
                    SamplesheetEntry(
                        filepath=sp,
                        sex="M" if is_male else "F",
                        ff=ff if is_male else None,
                    )
                )

            sheet = root / "sheet.yaml"
            sheet.write_text(
                yaml.safe_dump(
                    [
                        {"filepath": str(e.filepath), "sex": e.sex, "ff": e.ff}
                        for e in entries
                    ]
                ),
                encoding="utf-8",
            )
            ref_prefix = root / "prenatal_ref"
            run_reference(
                ReferenceConfig(
                    samplesheet_path=sheet,
                    prefix=ref_prefix,
                    mode="prenatal",
                    bin_size=bin_size,
                    prenatal_bin_size=100_000,
                    min_cohort_size=4,
                    min_sex_group_size=2,
                    pca_components=2,
                    qc_report=False,
                )
            )
            ref_path = parquet_path(ref_prefix)
            ref_tbl = pq.read_table(ref_path)
            p_meta = PrenatalModelMetadata.from_table(ref_tbl)
            self.assertIsNotNone(p_meta)
            assert p_meta is not None
            self.assertIsNotNone(p_meta.band_fisher_weights)

            # Verify build_fisher_weighted_cohort matches apply_prenatal_fisher_weighting
            test_bufs = SampleBuffers.from_table(pq.read_table(sample_paths[4]))
            weighted_bufs = apply_prenatal_fisher_weighting(test_bufs, p_meta)
            assert p_meta.band_fisher_weights is not None
            base_cohort = load_and_assemble_cohort(entries, bin_size)
            weighted_cohort = build_fisher_weighted_cohort(
                base_cohort, p_meta.band_fisher_weights
            )
            np.testing.assert_allclose(
                weighted_cohort.counts[:, 4],
                weighted_bufs.count,
                rtol=1e-5,
                atol=1e-5,
            )

            out_prefix = root / "pred"
            run_predict(
                PredictConfig(
                    sample=sample_paths[4],
                    reference=ref_path,
                    prefix=out_prefix,
                    mode="prenatal",
                    plot=False,
                    min_reference_bins=1,
                )
            )
            with gzip.open(Path(f"{out_prefix}_bins.bed.gz"), "rt") as fh:
                auto_ratios = [
                    float(line.split("\t")[4])
                    for line in fh
                    if line.startswith(("chr1\t", "chr2\t"))
                    and line.split("\t")[4] != "nan"
                ]
            self.assertGreater(len(auto_ratios), 0)
            self.assertLess(abs(float(np.median(auto_ratios))), 0.05)

    def test_defect2_msy_ffy_train_predict_exposure_and_kappa_calibration(self) -> None:
        rng = np.random.default_rng(202)
        n_bins_y = 30
        n_samples = 8
        female_mask = np.array([True, True, True, True, False, False, False, False])
        male_mask = ~female_mask
        true_ff = np.array([0.0, 0.0, 0.0, 0.0, 0.06, 0.10, 0.14, 0.18])
        labeled_ff = np.where(male_mask, true_ff, np.nan)

        exp = np.full((n_bins_y, n_samples), 400.0, dtype=np.float64)
        mapp = np.full(n_bins_y, 0.95, dtype=np.float64)
        # True counts generated with slope 0.40 instead of canonical 0.50 -> kappa_cal ~ 1.25
        lam = 0.002 + 0.40 * true_ff[None, :]
        counts = rng.poisson(exp * lam).astype(np.float64)
        model = fit_poisson_msy_ffy(
            msy_bin_counts=counts,
            msy_bin_exposure=exp,
            mappability=mapp,
            female_mask=female_mask,
            male_mask=male_mask,
            labeled_ff=labeled_ff,
        )
        self.assertAlmostEqual(model.kappa_cal, 1.25, delta=0.25)

        # Verify BRANCH_PRENATAL_FFY persists non-unit gc and mappability and uses
        # chromosome-median autosomal density consistently across train and predict.
        contig_lengths = {
            "chr1": 200_000,
            "chr2": 200_000,
            "chr3": 200_000,
            "chrX": 200_000,
            "chrY": 200_000,
        }
        bin_size = 5000
        n_tot = 200
        cohort_gc = np.full(n_tot, 0.42, dtype=np.float32)
        cohort_gc[160:] = 0.55
        cohort_map = np.full(n_tot, 0.95, dtype=np.float32)
        cohort_map[160:] = 0.80

        tables = []
        entries = []
        for i in range(6):
            is_male = i >= 3
            ff = 0.08 + 0.04 * (i - 3) if is_male else 0.0
            c_arr = np.full(n_tot, 100, dtype=np.uint32)
            c_arr[120:160] = round(100 * (1.0 - 0.5 * ff))
            c_arr[160:] = (
                round(100 * 0.5 * ff * 0.80 * 0.55 / (0.95 * 0.42)) if is_male else 0
            )
            meta = SampleMetadata(
                bin_size=bin_size,
                contig_lengths=contig_lengths,
                contig_order=tuple(contig_lengths.keys()),
                source_identity={},
                reference_identity={},
                qc={"effective_reads_kept": 1000},
                filters={"min_mapping_quality": 1},
            )
            z_u32 = np.zeros(n_tot, dtype=np.uint32)
            bufs = SampleBuffers(
                meta=meta,
                count=c_arr,
                paired_count=z_u32,
                frag_len_sum=np.zeros(n_tot, dtype=np.uint64),
                short_frag_count=z_u32,
                endpoint_count=z_u32,
                frag_gc_sum=np.zeros(n_tot, dtype=np.float32),
                frag_bands=None,
            )
            tables.append(bufs)
            entries.append(
                SamplesheetEntry(
                    filepath=Path(f"m_{i}.ccnv"),
                    sex="M" if is_male else "F",
                    ff=ff if is_male else None,
                )
            )

        auto_fit, ffy_fit, p_meta, _ = fit_prenatal_models(
            tables,
            entries,
            bin_size=bin_size,
            prenatal_bin_size=100_000,
            cohort_gc=cohort_gc,
            cohort_mappability=cohort_map,
        )
        self.assertIsNotNone(ffy_fit.gc)
        self.assertIsNotNone(ffy_fit.mappability)
        assert ffy_fit.gc is not None and ffy_fit.mappability is not None
        np.testing.assert_allclose(ffy_fit.gc, 0.55, atol=1e-6)
        np.testing.assert_allclose(ffy_fit.mappability, 0.80, atol=1e-6)

        autosomal_5k_fit = BranchFitArrays(
            branch=BRANCH_AUTOSOMAL,
            contigs=("chr1",) * 40 + ("chr2",) * 40 + ("chr3",) * 40,
            starts=np.tile(np.arange(40, dtype=np.int64) * bin_size, 3),
            ends=np.tile((np.arange(40, dtype=np.int64) + 1) * bin_size, 3),
            masked=np.zeros(120, dtype=np.bool_),
            pca_mean=np.ones(120, dtype=np.float64),
            gc=cohort_gc[:120],
            mappability=cohort_map[:120],
        )
        ref_table = assemble_reference_table([autosomal_5k_fit, auto_fit, ffy_fit])
        res = predict_prenatal(tables[4], ref_table, p_meta)
        self.assertIsNotNone(res.ffy)
        assert res.ffy is not None
        self.assertAlmostEqual(res.ffy, 0.12, delta=0.03)

    def test_defect3_fetal_fraction_result_accepts_ffx_and_ffa_logit_default(
        self,
    ) -> None:
        call, _, act_ff, act_se, act_source = classify_sca_ray_4d(
            delta_x=-0.06,
            sigma_x=0.01,
            ffy=0.0,
            sigma_ffy=0.01,
            ffa=None,
            sigma_ffa=0.02,
            ff_size=None,
            sigma_size=0.02,
            clinical_priors={
                "XX": 0.90,
                "XO": 0.05,
                "XXX": 0.05,
                "XY": 1e-6,
                "XXY": 1e-6,
                "XYY": 1e-6,
            },
        )
        self.assertEqual(call, "XO")
        self.assertEqual(act_source, "FFX")
        ff_res = FetalFractionResult(
            fetal_sex="F",
            active_ff=act_ff,
            active_ff_source=act_source,
            ffy=None,
            ffx=act_ff,
            ffa=0.0,
            chrx_ratio=0.94,
            chry_score=0.0,
            yfrac_threshold=0.005,
            sca=SCAResult("XO", True, 0.95, -6.0, None),
            active_ff_se=act_se,
        )
        self.assertEqual(ff_res.active_ff_source, "FFX")

        raw_meta = {
            "prenatal_bin_size": 100_000,
            "yfrac_threshold": 0.005,
            "ffy_intercept": 0.0,
            "ffy_slope": 1.0,
            "olm_weights": [0.2],
            "olm_bias": -2.0,
            "ffx_intercept": 1.0,
            "ffx_slope": -0.5,
            "female_chrx_ratio_mean": 1.0,
            "female_chrx_ratio_std": 0.01,
        }
        parsed = PrenatalModelMetadata.from_dict(raw_meta)
        self.assertTrue(parsed.ffa_logit_mode)

    def test_defect4_single_pass_gc_track_across_multi_branch_predict(self) -> None:
        contig_lengths = {
            "chr1": 100_000,
            "chr2": 100_000,
            "chrX": 100_000,
            "chrY": 100_000,
        }
        bin_size = 5000
        n_tot = 80
        rng = np.random.default_rng(404)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fasta_path = root / "ref.fa"
            seq_chunk = ("ACGTGCGC" * 12_500)[:100_000]
            fasta_path.write_text(
                f">chr1\n{seq_chunk}\n>chr2\n{seq_chunk}\n"
                f">chrX\n{seq_chunk}\n>chrY\n{seq_chunk}\n",
                encoding="utf-8",
            )

            sample_paths: list[Path] = []
            for i in range(6):
                is_male = i >= 3
                c_arr = rng.poisson(120, size=n_tot).astype(np.uint32)
                if is_male:
                    c_arr[40:60] = rng.poisson(60, size=20).astype(np.uint32)
                    c_arr[60:] = rng.poisson(60, size=20).astype(np.uint32)
                else:
                    c_arr[60:] = 0
                meta = SampleMetadata(
                    bin_size=bin_size,
                    contig_lengths=contig_lengths,
                    contig_order=tuple(contig_lengths.keys()),
                    source_identity={},
                    reference_identity={},
                    qc={"effective_reads_kept": 5000},
                    filters={"min_mapping_quality": 1},
                )
                z_u32 = np.zeros(n_tot, dtype=np.uint32)
                bufs = SampleBuffers(
                    meta=meta,
                    count=c_arr,
                    paired_count=z_u32,
                    frag_len_sum=np.zeros(n_tot, dtype=np.uint64),
                    short_frag_count=z_u32,
                    endpoint_count=z_u32,
                    frag_gc_sum=np.zeros(n_tot, dtype=np.float32),
                    frag_bands=None,
                )
                sp = root / f"g_{i}.ccnv"
                write_sample_parquet(bufs.to_table(), sp)
                sample_paths.append(sp)

            ref_prefix = root / "germ_ref"
            run_reference(
                ReferenceConfig(
                    entries=tuple(
                        SamplesheetEntry(filepath=p, sex="M" if idx >= 3 else "F")
                        for idx, p in enumerate(sample_paths)
                    ),
                    prefix=ref_prefix,
                    bin_size=bin_size,
                    min_cohort_size=4,
                    min_sex_group_size=2,
                    qc_report=False,
                )
            )
            with patch.object(
                ccnv.predict, "load_gc_track", wraps=load_gc_track
            ) as spy_gc:
                run_predict(
                    PredictConfig(
                        sample=sample_paths[0],
                        reference=parquet_path(ref_prefix),
                        prefix=root / "pred_f",
                        sex="female",
                        gc=fasta_path,
                        plot=False,
                        min_reference_bins=1,
                    )
                )
                self.assertEqual(spy_gc.call_count, 1)

    def test_defect5_alloc_cohort_counts_memmap_threshold_exact_bytes(self) -> None:
        # 262,144 float32 elements == 1,048,576 bytes == 1 MiB exact -> in-RAM ndarray
        in_ram = _alloc_cohort_counts(262_144, 1, max_batch_memory_mb=1)
        self.assertFalse(isinstance(in_ram, np.memmap))
        # 262,145 float32 elements == 1,048,580 bytes > 1 MiB -> disk-backed memmap
        spilled = _alloc_cohort_counts(262_145, 1, max_batch_memory_mb=1)
        self.assertIsInstance(spilled, np.memmap)

    def test_defect6_pca_components_none_dynamic_svht_vs_explicit_five_cap(
        self,
    ) -> None:
        contig_lengths = {"chr1": 500_000, "chr2": 500_000}
        bin_size = 5000
        n_bins = 200
        n_samples = 30
        rng = np.random.default_rng(606)

        # Synthesize 7 orthogonal latent wave factors well above Marchenko-Pastur bulk
        u_raw, _ = np.linalg.qr(rng.standard_normal((n_bins, 7)))
        v_raw, _ = np.linalg.qr(rng.standard_normal((n_samples, 7)))
        log_waves = u_raw @ np.diag([14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.5]) @ v_raw.T
        rates = 600.0 * np.exp(0.15 * log_waves)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            entries: list[SamplesheetEntry] = []
            for j in range(n_samples):
                c_arr = rng.poisson(rates[:, j]).astype(np.uint32)
                meta = SampleMetadata(
                    bin_size=bin_size,
                    contig_lengths=contig_lengths,
                    contig_order=("chr1", "chr2"),
                    source_identity={},
                    reference_identity={},
                    qc={"effective_reads_kept": 80_000},
                    filters={"min_mapping_quality": 1},
                )
                z_u32 = np.zeros(n_bins, dtype=np.uint32)
                bufs = SampleBuffers(
                    meta=meta,
                    count=c_arr,
                    paired_count=z_u32,
                    frag_len_sum=np.zeros(n_bins, dtype=np.uint64),
                    short_frag_count=z_u32,
                    endpoint_count=z_u32,
                    frag_gc_sum=np.zeros(n_bins, dtype=np.float32),
                    frag_bands=None,
                )
                sp = root / f"rk_{j}.ccnv"
                write_sample_parquet(bufs.to_table(), sp)
                entries.append(SamplesheetEntry(filepath=sp))

            dyn_prefix = root / "ref_dyn"
            run_reference(
                ReferenceConfig(
                    entries=tuple(entries),
                    prefix=dyn_prefix,
                    bin_size=bin_size,
                    pca_components=None,
                    qc_report=False,
                )
            )
            dyn_schema = pq.read_schema(parquet_path(dyn_prefix))
            self.assertGreaterEqual(dyn_schema.field("pca_loadings").type.list_size, 6)

            cap_prefix = root / "ref_cap5"
            run_reference(
                ReferenceConfig(
                    entries=tuple(entries),
                    prefix=cap_prefix,
                    bin_size=bin_size,
                    pca_components=5,
                    qc_report=False,
                )
            )
            cap_schema = pq.read_schema(parquet_path(cap_prefix))
            self.assertEqual(cap_schema.field("pca_loadings").type.list_size, 5)

    def test_defect7_band_prefix_sums_masking_z_lod_and_depth_ff_decoupling(
        self,
    ) -> None:
        contig_lengths = {"chr1": 20_000}
        bands = np.zeros((4, NUM_FRAG_BANDS), dtype=np.uint32)
        bands[0, 3] = 50
        bands[1, 20] = 9_999  # Blacklisted centromere artifact bin
        bands[2, 3] = 50
        bands[3, 3] = 50
        counts = np.sum(bands, axis=1, dtype=np.uint32)
        meta = SampleMetadata(
            bin_size=5000,
            contig_lengths=contig_lengths,
            contig_order=("chr1",),
            source_identity={},
            reference_identity={},
            qc={"effective_reads_kept": 10_149},
            filters={"min_mapping_quality": 1},
        )
        z_u32 = np.zeros(4, dtype=np.uint32)
        bufs = SampleBuffers(
            meta=meta,
            count=counts,
            paired_count=counts,
            frag_len_sum=np.zeros(4, dtype=np.uint64),
            short_frag_count=z_u32,
            endpoint_count=z_u32,
            frag_gc_sum=np.zeros(4, dtype=np.float32),
            frag_bands=bands,
        )
        usable = np.array([True, False, True, True], dtype=np.bool_)
        ps = BandPrefixSums.from_sample_table(bufs, bin_usable=usable)
        seg_counts = ps.query("chr1", 0, 20_000)
        self.assertEqual(int(seg_counts[20]), 0)
        self.assertEqual(int(seg_counts[3]), 150)

        _, lod_low = annotate_prenatal_origin_and_lod([], z_threshold=2.0)
        _, lod_high = annotate_prenatal_origin_and_lod([], z_threshold=5.0)
        self.assertGreater(lod_high.lod_ff_35mb, lod_low.lod_ff_35mb)
        self.assertGreater(lod_high.lod_ff_3mb, lod_low.lod_ff_3mb)

        # Decoupled depth_ff (Fisher-enriched 0.22) vs unweighted active_ff (0.10)
        p_f24 = np.zeros(NUM_FRAG_BANDS, dtype=np.float64)
        p_f24[3:7] = [0.20, 0.35, 0.30, 0.15]
        p_m24 = np.zeros(NUM_FRAG_BANDS, dtype=np.float64)
        p_m24[8:12] = [0.20, 0.35, 0.30, 0.15]
        active_ff = 0.10
        depth_ff = 0.22
        p_plasma_0 = active_ff * p_f24 + (1.0 - active_ff) * p_m24
        # Genuine fetal constitutional trisomy band mixture uses unweighted active_ff:
        r_unweighted = 1.0 + 0.5 * active_ff
        p_trisomy = (1.5 * active_ff * p_f24 + (1.0 - active_ff) * p_m24) / r_unweighted
        k_seg = np.round(20_000.0 * p_trisomy).astype(np.float64)
        r_obs_weighted = 1.0 + 0.5 * depth_ff
        call, posteriors, _, _ = classify_segment_origin_5state(
            r_obs=r_obs_weighted,
            k_band_counts=k_seg,
            p_fetal_24=p_f24,
            p_maternal_24=p_m24,
            p_plasma_0=p_plasma_0,
            active_ff=active_ff,
            active_ff_se=0.01,
            is_gain=True,
            depth_ff=depth_ff,
        )
        self.assertEqual(call, "FETAL_CONSTITUTIONAL")
        self.assertGreater(posteriors["FETAL_CONSTITUTIONAL"], 0.85)
        # Verify SegmentResult annotation also threads mu_fetal_w / mu_maternal_w -> depth_ff
        seg = SegmentResult(
            contig="chr1",
            start=0,
            end=20_000,
            ratio=math.log2(r_obs_weighted),
            zscore=8.0,
            weight=500.0,
            bin_count=4,
            call="gain",
        )
        ps_trisomy = BandPrefixSums(
            sample_bands=np.tile((k_seg / 4.0).astype(np.uint32), (4, 1)),
            bin_contigs=np.zeros(4, dtype=np.int16),
            contig_names=("chr1",),
            bin_starts=np.array([0, 5000, 10000, 15000], dtype=np.int64),
            bin_ends=np.array([5000, 10000, 15000, 20000], dtype=np.int64),
        )
        # Solve mu_fetal_w so f_eff(0.10, mu_f, 1.0) == 0.22 -> mu_f = 0.22 * 0.90 / (0.78 * 0.10)
        mu_f_w = (depth_ff * (1.0 - active_ff)) / ((1.0 - depth_ff) * active_ff)
        annotated, _ = annotate_prenatal_origin_and_lod(
            [seg],
            prefix_sums=ps_trisomy,
            p_fetal_24=p_f24,
            p_maternal_24=p_m24,
            mu_fetal_w=mu_f_w,
            mu_maternal_w=1.0,
            active_ff=active_ff,
            active_ff_se=0.01,
        )
        self.assertEqual(annotated[0].origin_call, "FETAL_CONSTITUTIONAL")


if __name__ == "__main__":
    unittest.main()
