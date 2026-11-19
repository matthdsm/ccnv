"""Unit and end-to-end tests for prenatal male-fetus 5 kb chrY MSY normalization."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pysam
from test_prenatal import (
    _build_synthetic_sample,
    _make_sample_buffers,
    _read_bed_rows,
    _setup_synthetic_cohort_and_references,
)

from ccnv.cli import AnalysisMode
from ccnv.features.fragmentomics import apply_prenatal_fisher_weighting
from ccnv.features.prenatal import (
    fit_prenatal_models,
    normalize_prenatal_chry,
    predict_prenatal,
)
from ccnv.features.prenatal_origin import annotate_sample_prenatal_segments
from ccnv.io.parquet import (
    filter_reference_branches,
    write_reference_parquet,
    write_sample_parquet,
)
from ccnv.models import (
    BRANCH_MALE_SEX,
    BRANCH_PRENATAL_FFY,
    NUM_FRAG_BANDS,
    FetalFractionResult,
    PredictConfig,
    PredictError,
    PredictionOutputs,
    PrenatalModelMetadata,
    ReferenceMetadata,
    SampleBuffers,
    SamplesheetEntry,
    SCAResult,
    assemble_reference_table,
    parquet_path,
)
from ccnv.predict import load_inputs, segment_and_call
from ccnv.predict import run as run_predict


def _bed_contigs(bed_path: Path) -> set[str]:
    return {r[0] for r in _read_bed_rows(bed_path)[1:]}


def _plot_contigs(plot_path: Path | None) -> list[str]:
    if plot_path is None:
        raise ValueError("expected plot_html path")
    html = plot_path.read_text(encoding="utf-8")
    marker = '<script id="ccnv-data" type="application/json">'
    start = html.index(marker) + len(marker)
    raw = json.loads(html[start : html.index("</script>", start)])
    contigs = raw.get("contigs") if isinstance(raw, dict) else None
    if not isinstance(contigs, list):
        raise TypeError("expected contigs list in ccnv-data")
    return [str(c) for c in contigs]


class TestNormalizePrenatalChrYUnit(unittest.TestCase):
    contig_lengths: dict[str, int]
    sample_tables: list[pa.Table]
    entries: list[SamplesheetEntry]

    @classmethod
    def setUpClass(cls) -> None:
        cls.contig_lengths = dict.fromkeys(("chr1", "chr2", "chrX"), 200_000) | {
            "chrY": 300_000
        }
        cls.sample_tables = []
        cls.entries = []
        for i in range(8):
            is_male = i >= 4
            ff = 0.06 + 0.03 * (i - 4) if is_male else None
            counts = {
                "chr1": 200.0,
                "chr2": 200.0,
                "chrX": 200.0 * (1.0 - 0.5 * (ff or 0.0)),
                "chrY": 200.0 * (0.005 + 0.5 * (ff or 0.0)) if is_male else 1.0,
            }
            cls.sample_tables.append(
                _build_synthetic_sample(cls.contig_lengths, counts, 5000, 0.2, 100 + i)
            )
            cls.entries.append(
                SamplesheetEntry(
                    filepath=Path(f"s{i}.ccnv"),
                    sex="M" if is_male else "F",
                    ff=ff,
                )
            )

    def test_normalize_prenatal_chry_female_xy_xyy_and_fisher(self) -> None:
        raw_meta, (auto_fit, ffy_fit), _ = fit_prenatal_models(
            self.sample_tables, self.entries, prenatal_bin_size=100_000, seed=42
        )
        meta = replace(raw_meta, ffa_std=0.20)
        ref_tbl = assemble_reference_table([auto_fit, ffy_fit])
        fem_bufs = SampleBuffers.from_table(self.sample_tables[1])
        fem_pred = predict_prenatal(fem_bufs, ref_tbl, meta)
        self.assertEqual(fem_pred.fetal_sex, "F")
        self.assertIsNone(normalize_prenatal_chry(fem_bufs, ref_tbl, meta, fem_pred))

        male_bufs = SampleBuffers.from_table(self.sample_tables[4])
        male_pred = predict_prenatal(male_bufs, ref_tbl, meta)
        auto_only = auto_fit.to_table(target_pca_rank=len(meta.olm_weights))
        self.assertIsNone(
            normalize_prenatal_chry(male_bufs, auto_only, meta, male_pred)
        )

        def _mk(cx: float, cy: float, s: int) -> SampleBuffers:
            c = {"chr1": 400.0, "chr2": 400.0, "chrX": cx, "chrY": cy}
            return SampleBuffers.from_table(
                _build_synthetic_sample(self.contig_lengths, c, 5000, 0.2, s)
            )

        ffy_ref = filter_reference_branches(ref_tbl, (BRANCH_PRENATAL_FFY,))
        lam_bg = np.maximum(
            ffy_ref.column("pca_mean").to_numpy(zero_copy_only=False), 0.0
        )
        bg = float(np.mean(lam_bg))
        for idx, ff in enumerate((0.05, 0.15)):
            bufs = _mk(
                400.0 * (1.0 - 0.5 * ff),
                400.0 * (bg + 0.5 * ff / meta.ffy_slope),
                520 + idx,
            )
            pred = predict_prenatal(bufs, ref_tbl, meta)
            self.assertEqual((pred.fetal_sex, pred.sca.call), ("M", "XY"))
            res = normalize_prenatal_chry(bufs, ref_tbl, meta, pred)
            if res is None or res.expected_counts is None:
                self.fail("expected BinResult with expected_counts")
            u = ~res.masked
            self.assertEqual(res.branch, BRANCH_PRENATAL_FFY)
            self.assertGreater(int(np.count_nonzero(u)), 0)
            self.assertLess(abs(float(np.mean(res.ratios[u]))), 0.08)
            self.assertLess(abs(float(np.median(res.ratios[u]))), 0.12)
            self.assertTrue(bool(np.all(np.isfinite(res.zscores[u]))))
            self.assertTrue(bool(np.all(res.weights[u] > 0.0)))
            exp_1copy = 400.0 * (lam_bg[u] + 0.5 * pred.active_ff / meta.ffy_slope)
            np.testing.assert_allclose(res.expected_counts[u], exp_1copy, rtol=0.05)

        xyy_bufs = _mk(376.0, 400.0 * (bg + 0.12 / meta.ffy_slope), 530)
        xyy_pred = predict_prenatal(xyy_bufs, ref_tbl, meta)
        self.assertEqual((xyy_pred.fetal_sex, xyy_pred.sca.call), ("M", "XYY"))
        xyy_res = normalize_prenatal_chry(xyy_bufs, ref_tbl, meta, xyy_pred)
        if xyy_res is None:
            self.fail("expected BinResult for 47,XYY")
        self.assertAlmostEqual(
            float(np.median(xyy_res.ratios[~xyy_res.masked])), 1.0, delta=0.12
        )

    def test_normalize_prenatal_chry_fisher_weighted_background_scaling(self) -> None:
        _, (auto_fit, ffy_fit), _ = fit_prenatal_models(
            self.sample_tables, self.entries, prenatal_bin_size=100_000, seed=42
        )
        high_bg_ffy = replace(
            ffy_fit,
            pca_mean=np.full(ffy_fit.num_rows, 0.10, dtype=np.float32),
            null_mean=np.zeros(ffy_fit.num_rows, dtype=np.float32),
        )
        ref_tbl = assemble_reference_table([auto_fit, high_bg_ffy])

        bands = np.zeros((180, NUM_FRAG_BANDS), dtype=np.uint32)
        bands[:80, 5], bands[:80, 12] = 200, 800
        bands[80:120, 5], bands[80:120, 12] = 100, 800
        bands[120:, 5], bands[120:, 12] = 100, 100
        raw_pe = _make_sample_buffers(
            self.contig_lengths, 5000, np.sum(bands, axis=1, dtype=np.uint32), bands
        )
        bw = [2.0] * 10 + [0.8] * 14
        meta_w = PrenatalModelMetadata(
            prenatal_bin_size=100_000,
            yfrac_threshold=0.005,
            ffy_slope=1.0,
            olm_weights=[0.0],
            olm_bias=0.0,
            female_chrx_ratio_mean=1.0,
            female_chrx_ratio_std=0.02,
            band_fisher_weights=bw,
            mu_fetal_w=2.0,
            mu_maternal_w=0.8,
        )
        f_act = 0.20
        w_mix = (1.0 - f_act) * 0.8 + f_act * 2.0
        f_eff = (f_act * 2.0) / w_mix
        pred = FetalFractionResult(
            fetal_sex="M",
            ffy=f_act,
            ffx=f_act,
            ffa=f_act,
            active_ff=f_act,
            active_ff_source="FUSED",
            chrx_ratio=0.90,
            chry_score=0.20,
            yfrac_threshold=0.005,
            sca=SCAResult(
                call="XY", is_aneuploid=False, confidence=0.99, chrx_zscore=0.0
            ),
            low_ff_warning=False,
            effective_ff=f_eff,
        )

        unweighted = normalize_prenatal_chry(raw_pe, ref_tbl, meta_w, pred)
        weighted_sample = apply_prenatal_fisher_weighting(raw_pe, meta_w)
        weighted = normalize_prenatal_chry(weighted_sample, ref_tbl, meta_w, pred)
        if (
            unweighted is None
            or weighted is None
            or weighted.expected_counts is None
            or unweighted.expected_counts is None
        ):
            self.fail("expected BinResult for unweighted and Fisher-weighted male")

        u = ~weighted.masked
        np.testing.assert_allclose(unweighted.expected_counts[u], 200.0, rtol=1e-6)
        np.testing.assert_allclose(weighted.expected_counts[u], 280.0, rtol=1e-6)
        self.assertLess(abs(float(np.median(unweighted.ratios[u]))), 0.01)
        self.assertLess(abs(float(np.median(weighted.ratios[u]))), 0.01)

    def test_euploid_xy_ff_jitter_and_haploid_bf_suppress_whole_chry_fp(self) -> None:
        rng = np.random.default_rng(77)
        lens = {
            "chr1": 1_000_000,
            "chr2": 1_000_000,
            "chrX": 1_000_000,
            "chrY": 5_000_000,
        }
        cohort: list[SampleBuffers] = []
        entries: list[SamplesheetEntry] = []
        for i in range(8):
            is_m = i >= 4
            ff = 0.08 + 0.02 * (i - 4) if is_m else 0.0
            c = np.concatenate(
                [
                    rng.poisson(30.0, 200),
                    rng.poisson(30.0, 200),
                    rng.poisson(30.0 * (1.0 - 0.5 * ff), 200),
                    rng.poisson(30.0 * (0.001 + 0.5 * ff) if is_m else 0.03, 1000),
                ]
            ).astype(np.uint32)
            cohort.append(_make_sample_buffers(lens, 5000, c))
            entries.append(
                SamplesheetEntry(
                    filepath=Path(f"c{i}.ccnv"),
                    sex="M" if is_m else "F",
                    ff=ff if is_m else None,
                )
            )
        pm, fits, _ = fit_prenatal_models(
            cohort, entries, prenatal_bin_size=100_000, seed=42
        )
        pm = replace(pm, ffy_std=0.01, ffa_std=0.025)
        ref_tbl = assemble_reference_table(fits)

        # 2x WGS (66 reads/5kb) where FFY=10.2% vs fused active_ff=9.0% (+0.18 log2 shift)
        c_test = np.concatenate(
            [
                rng.poisson(66.0, 200),
                rng.poisson(66.0, 200),
                rng.poisson(66.0 * 0.95, 200),
                rng.poisson(66.0 * (0.001 + 0.051), 1000),
            ]
        ).astype(np.uint32)
        sb = _make_sample_buffers(lens, 5000, c_test)
        pr = FetalFractionResult(
            fetal_sex="M",
            ffy=0.102,
            ffx=0.088,
            ffa=0.085,
            ff_size=0.087,
            active_ff=0.090,
            active_ff_source="FUSED",
            active_ff_se=0.009,
            chrx_ratio=0.956,
            chry_score=0.02,
            yfrac_threshold=pm.yfrac_threshold,
            sca=SCAResult(
                call="XY", is_aneuploid=False, confidence=0.98, chrx_zscore=0.0
            ),
            low_ff_warning=False,
        )
        br = normalize_prenatal_chry(sb, ref_tbl, pm, pr)
        if br is None:
            self.fail("expected BinResult")
        segs = segment_and_call(
            (br,), lens, penalty=1.0, z_threshold=3.0, mode="prenatal"
        )
        self.assertEqual([s for s in segs if s.call is not None], [])
        for seg in segs:
            self.assertEqual(seg.copy_number, 1)
            self.assertLess(abs(seg.zscore), 3.0)

    def test_lowpass_xyy_loco_rescue_yq_macrodeletion_and_azfa_ratio(self) -> None:
        rng = np.random.default_rng(91)
        lens = {
            "chr1": 1_000_000,
            "chr2": 1_000_000,
            "chrX": 1_000_000,
            "chrY": 4_000_000,
        }
        cohort: list[SampleBuffers] = []
        entries: list[SamplesheetEntry] = []
        for i in range(8):
            is_m = i >= 4
            ff = 0.08 + 0.02 * (i - 4) if is_m else 0.0
            c = np.concatenate(
                [
                    rng.poisson(16.0, 200),
                    rng.poisson(16.0, 200),
                    rng.poisson(16.0 * (1.0 - 0.5 * ff), 200),
                    rng.poisson(16.0 * (0.001 + 0.5 * ff) if is_m else 0.02, 800),
                ]
            ).astype(np.uint32)
            cohort.append(_make_sample_buffers(lens, 5000, c))
            entries.append(
                SamplesheetEntry(
                    filepath=Path(f"lp{i}.ccnv"),
                    sex="M" if is_m else "F",
                    ff=ff if is_m else None,
                )
            )
        pm, fits, _ = fit_prenatal_models(
            cohort, entries, prenatal_bin_size=100_000, seed=42
        )
        pm = replace(pm, ffy_std=0.008, ffa_std=0.020, ff_size_std=0.015)
        ref_tbl = assemble_reference_table(fits)

        # 1) Low-pass 0.3x NIPT 47,XYY at FF=5% where prior kept sca.call == "XY"
        d_lp = 10.0
        bg_ref = float(np.mean(fits[1].pca_mean))
        c_xyy = np.concatenate(
            [
                rng.poisson(d_lp, 200),
                rng.poisson(d_lp, 200),
                rng.poisson(d_lp * 0.975, 200),
                rng.poisson(d_lp * (bg_ref + 0.05), 800),
            ]
        ).astype(np.uint32)
        sb_xyy = _make_sample_buffers(lens, 5000, c_xyy)
        pr_xyy = FetalFractionResult(
            fetal_sex="M",
            ffy=0.10,
            ffx=0.05,
            ffa=0.05,
            ff_size=0.05,
            active_ff=0.092,
            active_ff_source="FUSED",
            active_ff_se=0.008,
            chrx_ratio=0.975,
            chry_score=0.015,
            yfrac_threshold=pm.yfrac_threshold,
            sca=SCAResult(
                call="XY", is_aneuploid=False, confidence=0.60, chrx_zscore=0.0
            ),
            low_ff_warning=False,
        )
        br_xyy = normalize_prenatal_chry(sb_xyy, ref_tbl, pm, pr_xyy)
        if br_xyy is None:
            self.fail("expected BinResult for low-FF 47,XYY")
        segs_xyy, _ = annotate_sample_prenatal_segments(
            segment_and_call(
                (br_xyy,), lens, penalty=1.0, z_threshold=3.0, mode="prenatal"
            ),
            sb_xyy,
            prenatal_meta=pm,
            prenatal_result=pr_xyy,
            z_threshold=3.0,
        )
        gains = [s for s in segs_xyy if s.call == "gain"]
        self.assertGreaterEqual(len(gains), 1)
        self.assertAlmostEqual(gains[0].ratio, 1.0, delta=0.20)
        self.assertEqual(gains[0].copy_number, 2)
        self.assertTrue(gains[0].discordant)
        self.assertEqual(gains[0].discordance_reason, "SCA_DISCORDANT")

        # 2) 50% (2 Mb) Yq macrodeletion at FF=5% (verifies yfrac_threshold + intact Yp + step detrend)
        d_yq = 33.0
        cy_yq = np.concatenate(
            [rng.poisson(d_yq * (0.001 + 0.025), 400), rng.poisson(d_yq * 0.001, 400)]
        )
        sb_yq = _make_sample_buffers(
            lens,
            5000,
            np.concatenate(
                [
                    rng.poisson(d_yq, 200),
                    rng.poisson(d_yq, 200),
                    rng.poisson(d_yq * 0.975, 200),
                    cy_yq,
                ]
            ).astype(np.uint32),
        )
        pr_yq = predict_prenatal(sb_yq, ref_tbl, pm)
        self.assertEqual(pr_yq.fetal_sex, "M")
        br_yq = normalize_prenatal_chry(sb_yq, ref_tbl, pm, pr_yq)
        if br_yq is None:
            self.fail("expected BinResult for 50% Yq deletion")
        segs_yq = segment_and_call(
            (br_yq,), lens, penalty=1.0, z_threshold=3.0, mode="prenatal"
        )
        self.assertTrue(any(s.call == "loss" and s.copy_number == 0 for s in segs_yq))

        # 3) 0.25x low-pass NIPT (8.25 reads/5kb, FF=8%) 800 kb AZFa deletion with ratio_threshold=0.5
        d_az = 8.25
        cy_az = rng.poisson(d_az * (0.001 + 0.04), 800)
        cy_az[200:360] = 0
        sb_az = _make_sample_buffers(
            lens,
            5000,
            np.concatenate(
                [
                    rng.poisson(d_az, 200),
                    rng.poisson(d_az, 200),
                    rng.poisson(d_az * 0.96, 200),
                    cy_az,
                ]
            ).astype(np.uint32),
        )
        pr_az = predict_prenatal(sb_az, ref_tbl, pm)
        br_az = normalize_prenatal_chry(sb_az, ref_tbl, pm, pr_az)
        if br_az is None:
            self.fail("expected BinResult for low-pass AZFa deletion")
        segs_az = segment_and_call(
            (br_az,),
            lens,
            penalty=1.0,
            z_threshold=3.0,
            ratio_threshold=0.5,
            mode="prenatal",
        )
        losses_az = [s for s in segs_az if s.call == "loss"]
        self.assertGreaterEqual(len(losses_az), 1)
        self.assertLess(losses_az[0].ratio, -1.5)
        self.assertEqual(losses_az[0].copy_number, 0)

    def test_msy_palindrome_yield_and_pcr_free_gc_flatness(self) -> None:
        rng = np.random.default_rng(104)
        pal_tables: list[SampleBuffers] = []
        for i, entry in enumerate(self.entries):
            is_m = entry.sex == "M"
            ff = entry.ff or 0.0
            cy_r = np.full(60, 200.0 * (0.005 + 0.5 * ff) if is_m else 1.0)
            if is_m:
                cy_r[20:35] = 200.0 * (0.005 + 1.0 * ff)  # 2-copy MSY palindrome
            c = np.concatenate(
                [
                    rng.poisson(200.0, 40),
                    rng.poisson(200.0, 40),
                    rng.poisson(200.0 * (1.0 - 0.5 * ff), 40),
                    rng.poisson(cy_r),
                ]
            ).astype(np.uint32)
            pal_tables.append(_make_sample_buffers(self.contig_lengths, 5000, c))

        pm_pal, fits_pal, _ = fit_prenatal_models(
            pal_tables, self.entries, prenatal_bin_size=100_000, seed=42
        )
        ref_pal = assemble_reference_table(fits_pal)
        pr_pal = predict_prenatal(pal_tables[6], ref_pal, pm_pal)
        br_pal = normalize_prenatal_chry(pal_tables[6], ref_pal, pm_pal, pr_pal)
        if br_pal is None:
            self.fail("expected BinResult for palindrome cohort")
        segs_pal = segment_and_call(
            (br_pal,),
            self.contig_lengths,
            penalty=1.0,
            z_threshold=3.0,
            mode="prenatal",
        )
        self.assertEqual([s for s in segs_pal if s.call is not None], [])
        self.assertLess(abs(float(np.nanmedian(br_pal.ratios[20:35]))), 0.12)

        # PCR-free flat GC coverage across AT-rich (0.34) vs GC-rich (0.49) MSY
        gc_track = np.full(180, 0.41, dtype=np.float64)
        gc_track[:20], gc_track[20:40] = 0.34, 0.49
        gc_track[120:150], gc_track[150:] = 0.34, 0.49
        flat_c = np.concatenate(
            [
                np.full(80, 200, dtype=np.uint32),
                np.full(40, 190, dtype=np.uint32),
                np.full(60, 11, dtype=np.uint32),
            ]
        )
        sb_gc = _make_sample_buffers(self.contig_lengths, 5000, flat_c)
        _, (auto_fit, ffy_fit), _ = fit_prenatal_models(
            self.sample_tables, self.entries, prenatal_bin_size=100_000, seed=42
        )
        ref_gc = assemble_reference_table(
            [
                auto_fit,
                replace(
                    ffy_fit, null_mean=np.zeros(ffy_fit.num_rows, dtype=np.float32)
                ),
            ]
        )
        pr_gc = predict_prenatal(sb_gc, ref_gc, pm_pal, sample_gc_override=gc_track)
        br_gc = normalize_prenatal_chry(
            sb_gc, ref_gc, pm_pal, pr_gc, sample_gc_override=gc_track
        )
        if br_gc is None:
            self.fail("expected BinResult for PCR-free GC check")
        diff_gc = abs(
            float(np.nanmedian(br_gc.ratios[:30]) - np.nanmedian(br_gc.ratios[30:]))
        )
        self.assertLess(diff_gc, 0.05)


class TestPrenatalChrYDeliverables(unittest.TestCase):
    _tmpdir: tempfile.TemporaryDirectory[str]
    root: Path
    contig_lengths: dict[str, int]
    prenatal_ref: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._tmpdir.cleanup)
        cls.root = Path(cls._tmpdir.name)
        cls.contig_lengths = dict.fromkeys(("chr1", "chr2", "chrX"), 200_000) | {
            "chrY": 300_000
        }
        cls.prenatal_ref, _ = _setup_synthetic_cohort_and_references(
            cls.root, cls.contig_lengths
        )

    def _run_case(
        self,
        name: str,
        means: tuple[float, float, float],
        seed: int,
        cnv: tuple[str, int, int, float] | None = None,
        ref: Path | None = None,
    ) -> PredictionOutputs:
        c_auto, c_x, c_y = means
        s_path = self.root / f"{name}.ccnv"
        counts = {"chr1": c_auto, "chr2": c_auto, "chrX": c_x, "chrY": c_y}
        tbl = _build_synthetic_sample(
            self.contig_lengths, counts, 5000, 0.05, seed, cnv
        )
        write_sample_parquet(tbl, s_path)
        return run_predict(
            PredictConfig(
                sample=s_path,
                reference=ref or self.prenatal_ref,
                prefix=self.root / f"{name}_out",
                mode=AnalysisMode.prenatal,
                plot=True,
                vcf=True,
                penalty=0.5,
                min_reference_bins=1,
                z_threshold=3.0,
            )
        )

    def test_prenatal_male_xy_chry_deliverables_and_no_male_sex_branch(self) -> None:
        ref_table = pq.read_table(self.prenatal_ref)
        ref_meta = ReferenceMetadata.from_table(ref_table)
        keep = tuple(b for b in ref_meta.branches if b != BRANCH_MALE_SEX)
        ranks = ref_meta.branch_pca_ranks or {}
        no_male_meta = replace(
            ref_meta,
            branches=keep,
            branch_pca_ranks={k: ranks[k] for k in keep if k in ranks} or None,
        )
        no_male_ref = parquet_path(self.root / "prenatal_ref_no_male_sex")
        pruned_tbl = filter_reference_branches(ref_table, keep).replace_schema_metadata(
            no_male_meta.build_arrow_metadata()
        )
        write_reference_parquet(pruned_tbl, no_male_ref)
        self.assertNotIn(
            BRANCH_MALE_SEX,
            set(pq.read_table(no_male_ref).column("branch").unique().to_pylist()),
        )

        for tag, ref_path in (("full", self.prenatal_ref), ("nomale", no_male_ref)):
            out = self._run_case(f"m_{tag}", (100.0, 94.0, 6.0), 610, ref=ref_path)
            chry_bins = [b for b in out.bins if "chrY" in b.contigs]
            self.assertEqual([b.branch for b in chry_bins], [BRANCH_PRENATAL_FFY])
            self.assertTrue(any(s.contig == "chrY" for s in out.segments))
            bin_branches = {
                r[9] for r in _read_bed_rows(out.bins_bed)[1:] if r[0] == "chrY"
            }
            self.assertEqual(bin_branches, {BRANCH_PRENATAL_FFY})
            self.assertIn("chrY", _bed_contigs(out.segments_bed))
            if out.segments_vcf is None:
                self.fail("expected segments_vcf")
            with pysam.VariantFile(str(out.segments_vcf)) as vf:
                self.assertGreaterEqual(len(list(vf.fetch("chrY"))), 1)
            self.assertIn("chrY", _plot_contigs(out.plot_html))

    def test_prenatal_female_xx_omits_chry(self) -> None:
        out = self._run_case("female_xx", (100.0, 100.0, 0.5), 620)
        if out.prenatal_result is None:
            self.fail("expected prenatal_result")
        self.assertEqual(out.prenatal_result.fetal_sex, "F")
        self.assertFalse(any("chrY" in b.contigs for b in out.bins))
        self.assertFalse(any(s.contig == "chrY" for s in out.segments))
        self.assertNotIn("chrY", _bed_contigs(out.bins_bed))
        self.assertNotIn("chrY", _bed_contigs(out.segments_bed))
        self.assertNotIn("chrY", _plot_contigs(out.plot_html))

    def test_prenatal_male_47_xyy_gain_and_sca_discordant(self) -> None:
        out = self._run_case("male_47_xyy", (200.0, 188.0, 24.0), 630)
        if out.prenatal_result is None:
            self.fail("expected prenatal_result")
        self.assertEqual(out.prenatal_result.sca.call, "XYY")
        gains = [s for s in out.segments if s.contig == "chrY" and s.call == "gain"]
        self.assertGreaterEqual(len(gains), 1)
        for seg in gains:
            self.assertAlmostEqual(seg.ratio, 1.0, delta=0.20)
            self.assertTrue(seg.discordant)
            self.assertEqual(seg.discordance_reason, "SCA_DISCORDANT")

    def test_prenatal_male_xy_focal_chry_microdeletion(self) -> None:
        del_reg = ("chrY", 130_000, 170_000, 0.0)
        out = self._run_case("male_chry_del", (200.0, 185.0, 15.0), 640, cnv=del_reg)
        if out.prenatal_result is None or out.calls_vcf is None:
            self.fail("expected prenatal_result and calls_vcf")
        self.assertEqual(
            (out.prenatal_result.fetal_sex, out.prenatal_result.sca.call), ("M", "XY")
        )
        self.assertEqual(
            {s.call for s in out.segments if s.contig == "chrY"}, {None, "loss"}
        )
        call_types = {r[5] for r in _read_bed_rows(out.calls_bed)[1:] if r[0] == "chrY"}
        self.assertIn("loss", call_types)
        with pysam.VariantFile(str(out.calls_vcf)) as vf:
            svtypes = {rec.info.get("SVTYPE") for rec in vf.fetch("chrY")}
            self.assertIn("DEL", svtypes)

    def test_prenatal_load_inputs_validates_chry_presence_and_length(self) -> None:
        no_y_lens = {"chr1": 200_000, "chr2": 200_000, "chrX": 200_000}
        no_y_path = self.root / "missing_chry.ccnv"
        write_sample_parquet(
            _build_synthetic_sample(
                no_y_lens,
                {"chr1": 100.0, "chr2": 100.0, "chrX": 95.0},
                5000,
                0.1,
                701,
            ),
            no_y_path,
        )
        with self.assertRaisesRegex(PredictError, "missing reference bins.*chrY"):
            load_inputs(no_y_path, self.prenatal_ref, mode=AnalysisMode.prenatal)

        bad_y_lens = dict(self.contig_lengths) | {"chrY": 500_000}
        bad_y_path = self.root / "mismatch_chry.ccnv"
        write_sample_parquet(
            _build_synthetic_sample(
                bad_y_lens,
                {"chr1": 100.0, "chr2": 100.0, "chrX": 95.0, "chrY": 6.0},
                5000,
                0.1,
                702,
            ),
            bad_y_path,
        )
        with self.assertRaisesRegex(PredictError, "contig length mismatch for 'chrY'"):
            load_inputs(bad_y_path, self.prenatal_ref, mode=AnalysisMode.prenatal)


class TestPrenatalCalibrationAndGridRegressions(unittest.TestCase):
    def test_fit_prenatal_models_ffa_cv_no_pca_subspace_leakage(self) -> None:
        from unittest.mock import patch

        from ccnv.stats.pca import pca_irlba_numpy

        contig_lengths = {
            "chr1": 1_000_000,
            "chr2": 1_000_000,
            "chrX": 200_000,
            "chrY": 100_000,
        }
        n_5k = sum(contig_lengths.values()) // 5000
        ff_vals = [0.05, 0.08, 0.11, 0.14, 0.17, 0.20, 0.23, 0.26]
        samples: list[SampleBuffers] = []
        entries: list[SamplesheetEntry] = []
        for idx, ff_v in enumerate(ff_vals):
            is_m = idx >= 4
            counts = np.full(n_5k, 100.0, dtype=np.float64)
            bands = np.zeros((n_5k, NUM_FRAG_BANDS), dtype=np.uint32)
            s_bin, e_bin = idx * 20, (idx + 1) * 20
            counts[s_bin:e_bin] = 100.0 * (1.5 + 0.3 * idx)
            if not is_m:
                counts[-20:] = 0.0
            else:
                counts[-20:] = 100.0 * 0.5 * ff_v
            bands[:, 0] = np.round(counts).astype(np.uint32)
            samples.append(_make_sample_buffers(contig_lengths, 5000, counts, bands))
            entries.append(
                SamplesheetEntry(
                    Path(f"leak_{idx}.ccnv"),
                    sex="M" if is_m else "F",
                    ff=ff_v,
                )
            )
        with patch(
            "ccnv.features.prenatal.pca_irlba_numpy", wraps=pca_irlba_numpy
        ) as mock_pca:
            meta, _, _ = fit_prenatal_models(
                samples, entries, prenatal_bin_size=100_000, seed=42
            )
        self.assertGreaterEqual(mock_pca.call_count, 6)
        fold_row_counts = [
            call.args[0].shape[0] for call in mock_pca.call_args_list[1:]
        ]
        self.assertTrue(all(n_rows < len(samples) for n_rows in fold_row_counts))
        self.assertGreaterEqual(meta.ffa_std, float(np.std(ff_vals)) * 0.90)

    def test_zero_inclusive_gc_exponent_all_at_bin_and_low_depth_100k_chrx_ratio(
        self,
    ) -> None:
        from ccnv.features.prenatal_grid import (
            compute_100k_profiles,
            fit_sample_gc_exponent,
            prepare_covariate_multipliers,
        )

        rng = np.random.default_rng(101)
        n_bins = 2000
        gc_mult = np.linspace(0.5, 1.5, n_bins)
        exp_b = np.full(n_bins, 0.5, dtype=np.float64)
        lam_true = exp_b * np.power(gc_mult, 2.0)
        counts_low = rng.poisson(lam_true).astype(np.float64)
        beta_hat = fit_sample_gc_exponent(
            counts_low, exp_b, np.ones(n_bins), gc_mult, np.ones(n_bins, dtype=np.bool_)
        )
        self.assertLess(abs(beta_hat - 2.0), 0.15)

        c_x = rng.poisson(1.5, size=400).astype(np.float64)
        self.assertGreater(int(np.count_nonzero(c_x == 0.0)), 40)
        c_mat = np.concatenate([np.full(400, 1.5), c_x]).reshape(-1, 1)
        contigs = ["chr1"] * 400 + ["chrX"] * 400
        starts = np.tile(np.arange(400, dtype=np.int64) * 100_000, 2)
        ends = starts + 100_000
        auto_m = np.array([c == "chr1" for c in contigs], dtype=np.bool_)
        chrx_m = np.array([c == "chrX" for c in contigs], dtype=np.bool_)
        _, chrx_rat = compute_100k_profiles(
            c_mat, contigs, starts, ends, 100_000, auto_m, chrx_m
        )
        self.assertAlmostEqual(float(chrx_rat[0]), 1.0, delta=0.06)

        gc_mult_at, _ = prepare_covariate_multipliers(
            np.array([0.0, 0.4, 0.5, 0.6]),
            np.ones(4),
            None,
            0.5,
            4,
        )
        self.assertLess(float(gc_mult_at[0]), 0.01)

    def test_classify_sca_ray_4d_female_ghost_ffy_df_and_xyy_arclength_invariance(
        self,
    ) -> None:
        from ccnv.stats.prenatal_math import classify_sca_ray_4d

        p0 = dict.fromkeys(("XX", "XY", "XO", "XXX", "XXY", "XYY"), 1e-6)
        fem_priors = p0 | {"XX": 0.90, "XO": 0.05, "XXX": 0.05}
        res_ghost = classify_sca_ray_4d(
            delta_x=0.0,
            sigma_x=0.01,
            ffy=0.0,
            sigma_ffy=0.005,
            ffa=0.10,
            sigma_ffa=0.01,
            ff_size=0.1535,
            sigma_size=0.01,
            clinical_priors=fem_priors,
        )
        res_none = classify_sca_ray_4d(
            delta_x=0.0,
            sigma_x=0.01,
            ffy=None,
            sigma_ffy=0.005,
            ffa=0.10,
            sigma_ffa=0.01,
            ff_size=0.1535,
            sigma_size=0.01,
            clinical_priors=fem_priors,
        )
        self.assertEqual(res_ghost.call, "DISCORDANT_MIXED")
        self.assertEqual(res_none.call, "DISCORDANT_MIXED")
        self.assertAlmostEqual(
            res_ghost.mahalanobis_d2, res_none.mahalanobis_d2, places=6
        )
        self.assertAlmostEqual(res_ghost.posterior, res_none.posterior, places=6)

        unif = dict.fromkeys(("XX", "XY", "XO", "XXX", "XXY", "XYY"), 1.0 / 6.0)
        res_xy = classify_sca_ray_4d(
            delta_x=-0.05,
            sigma_x=0.01,
            ffy=0.10,
            sigma_ffy=0.005,
            ffa=0.10,
            sigma_ffa=0.02,
            ff_size=0.10,
            sigma_size=0.02,
            clinical_priors=unif,
        )
        res_xyy = classify_sca_ray_4d(
            delta_x=-0.05,
            sigma_x=0.01,
            ffy=0.20,
            sigma_ffy=0.005,
            ffa=0.10,
            sigma_ffa=0.02,
            ff_size=0.10,
            sigma_size=0.02,
            clinical_priors=unif,
        )
        self.assertEqual(res_xy.call, "XY")
        self.assertEqual(res_xyy.call, "XYY")
        self.assertGreaterEqual(
            res_xyy.posteriors["XYY"], res_xy.posteriors["XY"] - 0.05
        )

    def test_female_singleton_twin_posteriors_and_symmetric_fm_interval(self) -> None:
        from ccnv.stats.prenatal_math import classify_twin_pregnancy

        res_fem = classify_twin_pregnancy(
            fetal_sex="F",
            chry_score=0.85 * 0.0001,
            yfrac_threshold=0.0001,
            ffy=None,
            sigma_ffy=0.005,
            delta_x=0.0,
            sigma_x=0.008,
            ffa=0.08,
            sigma_ffa=0.008,
            ff_size=0.08,
            sigma_size=0.008,
            active_ff=0.08,
            active_ff_se=0.006,
            active_ff_source="FUSED",
            twins=False,
        )
        self.assertEqual(res_fem.gestation, "singleton")
        self.assertFalse(res_fem.twin.is_twin)
        self.assertGreater(res_fem.twin.posteriors["FF"], 0.90)
        self.assertGreater(res_fem.twin.posteriors["FM"], res_fem.twin.posteriors["MM"])
        self.assertLess(res_fem.twin.posteriors["MM"], 0.03)
        self.assertLess(res_fem.twin.twin_probability, 0.05)

        fm_lo = classify_twin_pregnancy(
            fetal_sex="M",
            chry_score=0.00010,
            yfrac_threshold=0.000043,
            ffy=0.036,
            sigma_ffy=0.004,
            delta_x=-0.018,
            sigma_x=0.006,
            ffa=0.20,
            sigma_ffa=0.004,
            ff_size=0.20,
            sigma_size=0.004,
            active_ff=0.20,
            active_ff_se=0.003,
            active_ff_source="FUSED",
            twins=True,
        )
        self.assertEqual(fm_lo.twin.call, "FM")
        self.assertGreater(fm_lo.twin.posteriors["FM"], 0.98)


if __name__ == "__main__":
    unittest.main()
