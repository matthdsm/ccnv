"""Unit and end-to-end tests for prenatal modeling, inference, and SCA calling."""

from __future__ import annotations

import gzip
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pysam
import yaml
from helpers import make_sample_table
from numpy.typing import NDArray

from ccnv.cli import AnalysisMode
from ccnv.features.binning import bin_intervals
from ccnv.features.prenatal import (
    _evaluate_sca_and_active_ff,
    fit_prenatal_models,
    predict_prenatal,
)
from ccnv.features.prenatal_origin import annotate_prenatal_origin_and_lod
from ccnv.io.parquet import write_sample_parquet
from ccnv.models import (
    BRANCH_AUTOSOMAL,
    BRANCH_FEMALE_SEX,
    BRANCH_MALE_SEX,
    BRANCH_PRENATAL_AUTOSOMAL,
    BRANCH_PRENATAL_FFY,
    COORDINATE_SYSTEM,
    NUM_FRAG_BANDS,
    SCHEMA_VERSION,
    BranchFitArrays,
    FetalFractionResult,
    FetalSex,
    InvalidInputError,
    PredictConfig,
    PredictError,
    PredictionOutputs,
    PrenatalModelMetadata,
    ReferenceConfig,
    ReferenceMetadata,
    SampleBuffers,
    SampleMetadata,
    SamplesheetEntry,
    SCAResult,
    SegmentResult,
    assemble_reference_table,
    build_reference_schema,
    parquet_path,
)
from ccnv.predict import run as run_predict
from ccnv.reference import run as run_reference
from ccnv.report import PRENATAL_SEGMENTS_BED_HEADER
from ccnv.stats.prenatal_math import classify_sca_ray_4d, fit_poisson_msy_ffy


def _build_synthetic_sample(
    contig_lengths: dict[str, int],
    contig_mean_counts: dict[str, float],
    bin_size: int = 5000,
    sample_id: str = "sample",
    noise_scale: float = 0.5,
    seed: int = 42,
    cnv_region: tuple[str, int, int, float] | None = None,
) -> pa.Table:
    rng = np.random.default_rng(seed)
    c_list: list[str] = []
    s_list: list[int] = []
    e_list: list[int] = []
    counts_list: list[int] = []
    for contig, length in contig_lengths.items():
        starts, ends = bin_intervals(length, bin_size)
        base = np.full(
            starts.size, contig_mean_counts.get(contig, 100.0), dtype=np.float64
        )
        if cnv_region is not None and cnv_region[0] == contig:
            _, c_start, c_end, mult = cnv_region
            base[(starts >= c_start) & (ends <= c_end)] *= mult
        raw_c = np.clip(
            np.round(base + rng.normal(0.0, noise_scale, size=starts.size)), 0, None
        ).astype(np.uint32)
        c_list.extend([contig] * starts.size)
        s_list.extend(starts.tolist())
        e_list.extend(ends.tolist())
        counts_list.extend(raw_c.tolist())
    return make_sample_table(
        counts_list,
        contigs=c_list,
        starts=s_list,
        ends=e_list,
        contig_lengths=contig_lengths,
        bin_size=bin_size,
        identity={"id": sample_id},
    )


def _make_sample_buffers(
    contig_lengths: dict[str, int],
    bin_size: int,
    counts: NDArray[np.uint32] | NDArray[np.float64],
    bands: NDArray[np.uint32] | None = None,
    sample_id: str = "s",
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
    paired = (
        np.sum(bands, axis=1, dtype=np.uint32) if bands is not None else z_u32.copy()
    )
    return SampleBuffers(
        meta=meta,
        count=counts,
        paired_count=paired,
        frag_len_sum=(
            np.full(n, 16600, dtype=np.uint64)
            if bands is not None
            else np.zeros(n, dtype=np.uint64)
        ),
        short_frag_count=z_u32,
        endpoint_count=z_u32.copy(),
        frag_gc_sum=np.zeros(n, dtype=np.float32),
        frag_bands=bands.astype(np.uint32) if bands is not None else None,
    )


def classify_sca(
    fetal_sex: FetalSex,
    *,
    ffy: float | None,
    ffx: float,
    ffa: float,
    chrx_ratio: float,
    prenatal_meta: PrenatalModelMetadata,
) -> SCAResult:
    ffy_se = 0.01 if ffy is not None else None
    sca, _, _, _ = _evaluate_sca_and_active_ff(
        fetal_sex, ffy, ffy_se, ffx, ffa, chrx_ratio, prenatal_meta
    )
    return sca


def annotate_prenatal_segments(
    segments: list[SegmentResult],
    prenatal_result: FetalFractionResult,
    *,
    z_threshold: float = 3.0,
) -> list[SegmentResult]:
    eff_ff = (
        prenatal_result.effective_ff
        if prenatal_result.effective_ff is not None
        else prenatal_result.active_ff
    )
    annotated, _ = annotate_prenatal_origin_and_lod(
        segments,
        active_ff=eff_ff,
        sca_call=prenatal_result.sca.call,
        z_threshold=z_threshold,
    )
    return list(annotated)


class TestClassifySCA(unittest.TestCase):
    def setUp(self) -> None:
        self.meta = PrenatalModelMetadata(
            100000, 0.005, 1.0, [0.1], 0.0, 1.0, 0.01, alpha_1=-0.5
        )

    def test_female_xx(self) -> None:
        result = classify_sca(
            "F", ffy=None, ffx=0.0, ffa=0.08, chrx_ratio=1.0, prenatal_meta=self.meta
        )
        self.assertEqual(result.call, "XX")
        self.assertFalse(result.is_aneuploid)
        self.assertGreater(result.confidence, 0.5)

    def test_female_xo(self) -> None:
        result = classify_sca(
            "F", ffy=None, ffx=0.10, ffa=0.10, chrx_ratio=0.95, prenatal_meta=self.meta
        )
        self.assertEqual(result.call, "XO")
        self.assertTrue(result.is_aneuploid)
        self.assertGreater(result.confidence, 0.7)

    def test_female_xxx(self) -> None:
        result = classify_sca(
            "F", ffy=None, ffx=-0.10, ffa=0.10, chrx_ratio=1.05, prenatal_meta=self.meta
        )
        self.assertEqual(result.call, "XXX")
        self.assertTrue(result.is_aneuploid)
        self.assertGreater(result.confidence, 0.7)

    def test_male_xy(self) -> None:
        result = classify_sca(
            "M", ffy=0.10, ffx=0.10, ffa=0.10, chrx_ratio=0.95, prenatal_meta=self.meta
        )
        self.assertEqual(result.call, "XY")
        self.assertFalse(result.is_aneuploid)
        self.assertGreater(result.confidence, 0.5)

    def test_male_xxy(self) -> None:
        result = classify_sca(
            "M", ffy=0.10, ffx=0.0, ffa=0.10, chrx_ratio=1.0, prenatal_meta=self.meta
        )
        self.assertEqual(result.call, "XXY")
        self.assertTrue(result.is_aneuploid)
        self.assertGreater(result.confidence, 0.7)

    def test_male_xyy(self) -> None:
        result = classify_sca(
            "M", ffy=0.20, ffx=0.10, ffa=0.10, chrx_ratio=0.95, prenatal_meta=self.meta
        )
        self.assertEqual(result.call, "XYY")
        self.assertTrue(result.is_aneuploid)
        self.assertGreater(result.confidence, 0.7)

    def test_male_discordant_mixed(self) -> None:
        result = classify_sca(
            "M", ffy=0.05, ffx=0.30, ffa=0.05, chrx_ratio=0.85, prenatal_meta=self.meta
        )
        self.assertEqual(result.call, "DISCORDANT_MIXED")
        self.assertTrue(result.is_aneuploid)

    def test_classify_sca_bayesian_posterior_and_mahalanobis_discordant_gate(
        self,
    ) -> None:
        cases: list[tuple[FetalSex, float | None, float, float, float, str, bool]] = [
            ("F", None, 0.0, 0.10, 1.0, "XX", False),
            ("M", 0.10, 0.10, 0.10, 0.95, "XY", False),
            ("F", None, 0.10, 0.10, 0.95, "XO", True),
            ("F", None, -0.10, 0.10, 1.05, "XXX", True),
            ("M", 0.10, 0.0, 0.10, 1.0, "XXY", True),
            ("M", 0.20, 0.10, 0.10, 0.95, "XYY", True),
            ("M", 0.05, 0.30, 0.05, 0.85, "DISCORDANT_MIXED", True),
        ]
        for fsex, ffy, ffx, pref, rx, exp_call, exp_aneu in cases:
            res = classify_sca(
                fsex, ffy=ffy, ffx=ffx, ffa=pref, chrx_ratio=rx, prenatal_meta=self.meta
            )
            self.assertEqual(res.call, exp_call)
            self.assertEqual(res.is_aneuploid, exp_aneu)
            self.assertGreater(res.confidence, 0.5)
            self.assertLessEqual(res.confidence, 1.0)

        res_xy = classify_sca(
            "M", ffy=0.10, ffx=0.10, ffa=0.10, chrx_ratio=0.95, prenatal_meta=self.meta
        )
        self.assertEqual(res_xy.call, "XY")
        self.assertGreater(res_xy.confidence, 0.85)

    def test_classify_sca_ray_4d_female_and_male(self) -> None:
        p0 = {k: 1e-6 for k in ("XX", "XO", "XXX", "XY", "XXY", "XYY")}
        fem_p = p0 | {"XX": 0.90, "XO": 0.05, "XXX": 0.05}
        male_p = p0 | {"XY": 0.90, "XXY": 0.05, "XYY": 0.05}
        self.assertEqual(
            classify_sca_ray_4d(
                0.0, 0.02, 0.0, 0.01, 0.10, 0.02, clinical_priors=fem_p
            ).call,
            "XX",
        )
        self.assertEqual(
            classify_sca_ray_4d(
                -0.05, 0.01, 0.0, 0.01, 0.10, 0.02, clinical_priors=fem_p
            ).call,
            "XO",
        )
        self.assertEqual(
            classify_sca_ray_4d(
                0.05, 0.01, 0.0, 0.01, 0.10, 0.02, clinical_priors=fem_p
            ).call,
            "XXX",
        )
        self.assertEqual(
            classify_sca_ray_4d(
                -0.05, 0.02, 0.10, 0.01, 0.10, 0.02, clinical_priors=male_p
            ).call,
            "XY",
        )
        self.assertEqual(
            classify_sca_ray_4d(
                0.0, 0.01, 0.10, 0.01, 0.10, 0.02, clinical_priors=male_p
            ).call,
            "XXY",
        )


class TestPrenatalModelTrainingAndInference(unittest.TestCase):
    def setUp(self) -> None:
        c200 = {"chr1": 200_000, "chr2": 200_000, "chrX": 200_000}
        self.contig_lengths = c200 | {"chrY": 300_000}
        self.sample_tables: list[pa.Table] = []
        self.entries: list[SamplesheetEntry] = []

        for i in range(6):
            is_male = i >= 3
            ff = 0.08 + 0.02 * (i - 3) if is_male else None
            counts: dict[str, float] = {
                "chr1": 100.0,
                "chr2": 100.0,
                "chrX": 100.0 * (1.0 - 0.5 * (ff or 0.0)),
                "chrY": 100.0 * (0.5 * (ff or 0.0)) if is_male else 0.5,
            }
            self.sample_tables.append(
                _build_synthetic_sample(
                    self.contig_lengths,
                    counts,
                    bin_size=5000,
                    sample_id=f"sample_{i}",
                    noise_scale=0.1,
                    seed=100 + i,
                )
            )
            self.entries.append(
                SamplesheetEntry(
                    Path(f"sample_{i}.ccnv"), sex="M" if is_male else "F", ff=ff
                )
            )

    def test_fit_prenatal_models(self) -> None:
        meta, (auto_fit, ffy_fit), resolved_sex = fit_prenatal_models(
            self.sample_tables, self.entries, prenatal_bin_size=100000, seed=42
        )
        auto_tbl = auto_fit.to_table(target_pca_rank=len(meta.olm_weights))
        ffy_tbl = ffy_fit.to_table(target_pca_rank=len(meta.olm_weights))

        expected_schema = build_reference_schema(len(meta.olm_weights))
        self.assertEqual(auto_tbl.schema, expected_schema)
        self.assertEqual(ffy_tbl.schema, expected_schema)
        self.assertTrue(
            all(
                b == BRANCH_PRENATAL_AUTOSOMAL
                for b in auto_tbl.column("branch").to_pylist()
            )
        )
        self.assertTrue(
            all(b == BRANCH_PRENATAL_FFY for b in ffy_tbl.column("branch").to_pylist())
        )
        self.assertEqual(
            resolved_sex, ("female", "female", "female", "male", "male", "male")
        )
        self.assertEqual(meta.prenatal_bin_size, 100000)
        self.assertGreater(meta.yfrac_threshold, 0.0)
        self.assertGreater(len(meta.olm_weights), 0)
        self.assertTrue(meta.ffa_logit_mode)

    def test_predict_prenatal(self) -> None:
        meta, fits, _ = fit_prenatal_models(
            self.sample_tables, self.entries, prenatal_bin_size=100000, seed=42
        )
        ref_table = assemble_reference_table(list(fits))

        res_male = predict_prenatal(
            SampleBuffers.from_table(self.sample_tables[4]),
            ref_table,
            meta,
            min_ff_warning=0.04,
        )
        self.assertEqual(res_male.fetal_sex, "M")
        self.assertIsNotNone(res_male.ffy)
        self.assertGreater(res_male.active_ff, 0.0)
        self.assertIsNotNone(res_male.effective_ff)
        self.assertIn(res_male.active_ff_source, ("FFY", "FUSED"))

        res_female = predict_prenatal(
            SampleBuffers.from_table(self.sample_tables[1]),
            ref_table,
            meta,
            min_ff_warning=0.04,
        )
        self.assertEqual(res_female.fetal_sex, "F")
        self.assertIsNone(res_female.ffy)
        self.assertIsNotNone(res_female.effective_ff)
        self.assertIn(res_female.active_ff_source, ("FFA", "FUSED"))

    def test_predict_prenatal_olm_fallback(self) -> None:
        meta, fits, _ = fit_prenatal_models(
            self.sample_tables, self.entries, prenatal_bin_size=100000, seed=42
        )
        ref_table = assemble_reference_table(list(fits))
        fallback_meta = replace(
            meta,
            olm_weights=[0.0] * len(meta.olm_weights),
            olm_bias=math.log(0.12 / 0.88),
        )
        res = predict_prenatal(
            SampleBuffers.from_table(self.sample_tables[1]), ref_table, fallback_meta
        )
        self.assertIn(res.active_ff_source, ("FFA", "FUSED"))
        self.assertAlmostEqual(res.ffa, 0.12, places=3)

    def test_fit_prenatal_models_validation_errors(self) -> None:
        with self.assertRaises(InvalidInputError):
            fit_prenatal_models([], [], prenatal_bin_size=100000)
        with self.assertRaises(InvalidInputError):
            fit_prenatal_models(
                self.sample_tables, self.entries[:2], prenatal_bin_size=100000
            )

    def test_fit_prenatal_models_all_female_cohort(self) -> None:
        meta, (auto_fit, ffy_fit), resolved_sex = fit_prenatal_models(
            self.sample_tables[:3], self.entries[:3], prenatal_bin_size=100000
        )
        self.assertEqual(resolved_sex, ("female", "female", "female"))
        self.assertEqual(auto_fit.num_rows, 4)
        self.assertEqual(ffy_fit.num_rows, 60)
        self.assertIsNotNone(meta)

    def test_predict_prenatal_missing_branch_errors(self) -> None:
        meta, (auto_fit, ffy_fit), _ = fit_prenatal_models(
            self.sample_tables, self.entries, prenatal_bin_size=100000
        )
        sample_bufs = SampleBuffers.from_table(self.sample_tables[0])
        with self.assertRaises(PredictError):
            predict_prenatal(sample_bufs, auto_fit.to_table(), meta)
        with self.assertRaises(PredictError):
            predict_prenatal(sample_bufs, ffy_fit.to_table(), meta)

    def test_predict_prenatal_low_ff_warning(self) -> None:
        meta, fits, _ = fit_prenatal_models(
            self.sample_tables, self.entries, prenatal_bin_size=100000
        )
        ref_table = assemble_reference_table(list(fits))
        res = predict_prenatal(
            self.sample_tables[0], ref_table, meta, min_ff_warning=0.50
        )
        self.assertTrue(res.low_ff_warning)
        self.assertTrue(any("below warning threshold" in w for w in res.warnings))

    def test_annotate_prenatal_segments_edge_cases(self) -> None:
        sca = SCAResult("XX", False, 0.9, 0.0)
        ff_zero = FetalFractionResult(
            "F", 0.0, "FFA", None, None, 0.0, 1.0, 0.001, 0.005, sca
        )
        self.assertEqual(annotate_prenatal_segments([], ff_zero), [])

        seg = SegmentResult("chr1", 0, 1000, 0.01, 0.5, 1.0, 5, None)
        res = annotate_prenatal_segments([seg], ff_zero)
        self.assertEqual(len(res), 1)
        self.assertFalse(res[0].discordant)
        self.assertIsNone(res[0].discordance_reason)
        self.assertEqual(res[0].expected_ratio, 0.0)

    def test_predict_prenatal_autosomal_coordinate_fallback_with_extra_contigs(
        self,
    ) -> None:
        meta, fits, _ = fit_prenatal_models(
            self.sample_tables, self.entries, prenatal_bin_size=100000
        )
        ref_table = assemble_reference_table(list(fits))

        sample_lengths = {
            k: 200_000 for k in ("chr1", "chr2", "chr13", "chr18", "chr21", "chrX")
        } | {"chrY": 300_000}
        counts = {k: 100.0 for k in ("chr1", "chr2", "chr13", "chr18", "chr21")} | {
            "chrX": 95.0,
            "chrY": 5.0,
        }
        sample_with_extra = _build_synthetic_sample(
            sample_lengths, counts, bin_size=5000, sample_id="sample_extra"
        )
        res = predict_prenatal(sample_with_extra, ref_table, meta)
        self.assertEqual(res.fetal_sex, "M")
        self.assertIsNotNone(res.ffy)
        self.assertGreater(res.active_ff, 0.0)

    def test_predict_prenatal_chry_coordinate_fallback_with_mismatched_bins(
        self,
    ) -> None:
        meta, fits, _ = fit_prenatal_models(
            self.sample_tables, self.entries, prenatal_bin_size=100000
        )
        ref_table = assemble_reference_table(list(fits))

        sample_lengths = {k: 200_000 for k in ("chr1", "chr2", "chrX", "chrY")}
        counts = {"chr1": 100.0, "chr2": 100.0, "chrX": 95.0, "chrY": 5.0}
        sample_short_chry = _build_synthetic_sample(
            sample_lengths, counts, bin_size=5000, sample_id="sample_short_chry"
        )
        res = predict_prenatal(sample_short_chry, ref_table, meta)
        self.assertEqual(res.fetal_sex, "M")
        self.assertIsNotNone(res.ffy)

    def test_annotate_prenatal_segments_z_threshold_without_call(self) -> None:
        sca = SCAResult("XX", False, 0.9, 0.0)
        ff_active = FetalFractionResult(
            "F", 0.10, "FFA", None, None, 0.10, 1.0, 0.001, 0.005, sca
        )
        seg = SegmentResult("chr1", 0, 1000, 0.01, 3.5, 1.0, 5, None)
        res = annotate_prenatal_segments([seg], ff_active, z_threshold=3.0)
        self.assertTrue(res[0].discordant)
        self.assertEqual(res[0].discordance_reason, "SUB_FETAL_RATIO")

    def test_defect2_msy_ffy_train_predict_exposure_and_kappa_calibration(self) -> None:
        rng = np.random.default_rng(202)
        n_bins_y, n_samples = 30, 8
        female_mask = np.array([True, True, True, True, False, False, False, False])
        male_mask = ~female_mask
        true_ff = np.array([0.0, 0.0, 0.0, 0.0, 0.06, 0.10, 0.14, 0.18])
        exp = np.full((n_bins_y, n_samples), 400.0, dtype=np.float64)
        counts = rng.poisson(exp * (0.002 + 0.40 * true_ff[None, :])).astype(np.float64)
        model = fit_poisson_msy_ffy(
            counts,
            exp,
            np.full(n_bins_y, 0.95, dtype=np.float64),
            female_mask,
            male_mask,
            np.where(male_mask, true_ff, np.nan),
        )
        self.assertAlmostEqual(model.kappa_cal, 1.25, delta=0.25)

        contig_lengths = {k: 200_000 for k in ("chr1", "chr2", "chr3", "chrX", "chrY")}
        bin_size, n_tot = 5000, 200
        cohort_gc = np.full(n_tot, 0.42, dtype=np.float32)
        cohort_gc[160:] = 0.55
        cohort_map = np.full(n_tot, 0.95, dtype=np.float32)
        cohort_map[160:] = 0.80

        tables: list[SampleBuffers] = []
        entries: list[SamplesheetEntry] = []
        for i in range(6):
            is_male = i >= 3
            ff = 0.08 + 0.04 * (i - 3) if is_male else 0.0
            c_arr = np.full(n_tot, 100, dtype=np.uint32)
            c_arr[120:160] = round(100 * (1.0 - 0.5 * ff))
            c_arr[160:] = (
                round(100 * 0.5 * ff * 0.80 * 0.55 / (0.95 * 0.42)) if is_male else 0
            )
            tables.append(_make_sample_buffers(contig_lengths, bin_size, c_arr))
            entries.append(
                SamplesheetEntry(
                    Path(f"m_{i}.ccnv"),
                    sex="M" if is_male else "F",
                    ff=ff if is_male else None,
                )
            )

        p_meta, (auto_fit, ffy_fit), _ = fit_prenatal_models(
            tables,
            entries,
            prenatal_bin_size=100_000,
            cohort_gc=cohort_gc,
            cohort_mappability=cohort_map,
        )
        self.assertIsNotNone(ffy_fit.gc)
        self.assertIsNotNone(ffy_fit.mappability)
        if ffy_fit.gc is None or ffy_fit.mappability is None:
            self.fail("expected ffy_fit gc and mappability")
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
        if res.ffy is None:
            self.fail("expected res.ffy")
        self.assertAlmostEqual(res.ffy, 0.12, delta=0.03)

    def test_predict_prenatal_linear_ffa_mode(self) -> None:
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
            bands = rng.poisson(6.0, size=(140, NUM_FRAG_BANDS)).astype(np.uint32)
            counts = np.sum(bands, axis=1, dtype=np.float64)
            if not is_m:
                counts[120:] = 0.0
            samples.append(
                _make_sample_buffers(contig_lengths, 5000, counts, bands, f"p_{idx}")
            )
            entries.append(
                SamplesheetEntry(
                    Path(f"p_{idx}.ccnv"),
                    sex="M" if is_m else "F",
                    ff=0.12 if is_m else 0.08,
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


def _setup_synthetic_cohort_and_references(
    root: Path, contig_lengths: dict[str, int]
) -> tuple[Path, Path]:
    sample_dir = root / "cohort"
    sample_dir.mkdir()
    entries_data: list[dict[str, object]] = []

    for i in range(6):
        is_male = i >= 3
        ff = 0.08 + 0.02 * (i - 3) if is_male else None
        counts = {
            "chr1": 100.0,
            "chr2": 100.0,
            "chrX": 100.0 * (1.0 - 0.5 * (ff or 0.0)),
            "chrY": 100.0 * (0.5 * (ff or 0.0)) if is_male else 0.5,
        }
        s_path = sample_dir / f"cohort_{i}.ccnv"
        write_sample_parquet(
            _build_synthetic_sample(
                contig_lengths, counts, 5000, f"cohort_{i}", 0.1, 200 + i
            ),
            s_path,
        )
        entries_data.append(
            {"filepath": str(s_path), "sex": "M" if is_male else "F", "ff": ff}
        )

    sheet_path = root / "samplesheet.yaml"
    sheet_path.write_text(yaml.dump(entries_data), encoding="utf-8")

    prenatal_prefix = root / "prenatal_ref"
    run_reference(
        ReferenceConfig(
            samplesheet_path=sheet_path,
            prefix=prenatal_prefix,
            mode="prenatal",
            bin_size=5000,
            prenatal_bin_size=100000,
            min_cohort_size=4,
            min_sex_group_size=2,
            pca_components=2,
            qc_report=True,
        )
    )

    test_sample_tbl = _build_synthetic_sample(
        contig_lengths,
        {"chr1": 100.0, "chr2": 100.0, "chrX": 130.625, "chrY": 6.875},
        5000,
        "test_male_sample",
        0.1,
        999,
        ("chr1", 100_000, 200_000, 2.5),
    )
    test_sample_path = root / "test_male_sample.ccnv"
    write_sample_parquet(test_sample_tbl, test_sample_path)
    return parquet_path(prenatal_prefix), test_sample_path


def _read_bed_rows(bed_path: Path) -> list[list[str]]:
    with gzip.open(bed_path, "rt", encoding="utf-8") as fh:
        return [line.rstrip("\r\n").split("\t") for line in fh if line.strip()]


class TestPrenatalEndToEndIntegration(unittest.TestCase):
    _tmpdir: tempfile.TemporaryDirectory[str]
    root: Path
    contig_lengths: dict[str, int]
    prenatal_ref: Path
    test_sample: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._tmpdir.cleanup)
        cls.root = Path(cls._tmpdir.name)
        cls.contig_lengths = dict.fromkeys(("chr1", "chr2", "chrX"), 200_000) | {
            "chrY": 300_000
        }
        cls.prenatal_ref, cls.test_sample = _setup_synthetic_cohort_and_references(
            cls.root, cls.contig_lengths
        )

    def _predict_path(
        self, s_path: Path, prefix: Path, ref: Path | None = None, z: float = 3.0
    ) -> PredictionOutputs:
        return run_predict(
            PredictConfig(
                sample=s_path,
                reference=ref or self.prenatal_ref,
                prefix=prefix,
                mode=AnalysisMode.prenatal,
                plot=True,
                vcf=True,
                penalty=0.5,
                min_reference_bins=1,
                z_threshold=z,
            )
        )

    def test_prenatal_pipeline_deliverables(self) -> None:
        ref_table = pq.read_table(self.prenatal_ref)
        branches = set(ref_table.column("branch").unique().to_pylist())
        for b in (BRANCH_AUTOSOMAL, BRANCH_FEMALE_SEX, BRANCH_MALE_SEX):
            self.assertIn(b, branches)
        for b in (BRANCH_PRENATAL_AUTOSOMAL, BRANCH_PRENATAL_FFY):
            self.assertIn(b, branches)

        pred_prefix = self.root / "predict_out"
        outputs = self._predict_path(self.test_sample, pred_prefix, z=1.5)
        if outputs.qc_json_path is None or outputs.segments_vcf is None:
            self.fail("Expected qc_json_path and segments_vcf")
        qc = json.loads(outputs.qc_json_path.read_text(encoding="utf-8"))
        self.assertEqual(qc["mode"], "prenatal")
        p_qc = qc["prenatal"]
        for k in ("fetal_sex", "active_ff", "effective_ff", "active_ff_source"):
            self.assertIn(k, p_qc)
        for k in ("sca", "ffy", "ffa", "ffx"):
            self.assertIn(k, p_qc)
        self.assertEqual(p_qc["fetal_sex"], "M")
        self.assertGreater(p_qc["active_ff"], 0.0)
        self.assertGreater(p_qc["effective_ff"], 0.0)
        self.assertIn(p_qc["active_ff_source"], ("FFY", "FUSED"))
        self.assertTrue(any(s.discordant for s in outputs.segments))

        bed_lines = _read_bed_rows(outputs.segments_bed)
        self.assertEqual(tuple(bed_lines[0]), PRENATAL_SEGMENTS_BED_HEADER)
        self.assertTrue(all(len(row) == 15 for row in bed_lines))
        disc_reasons = {r[11] for r in bed_lines[1:] if r[10] == "true"}
        self.assertIn("SUPER_FETAL_RATIO", disc_reasons)

        with pysam.VariantFile(str(outputs.segments_vcf)) as vf:
            hdr = str(vf.header)
            for t in ("EXP_LOG2R", "IMPLIED_FF", "DISCORDANT", "DISC_REASON"):
                self.assertIn(f"ID={t}", hdr)
            for t in ("FRAG_SHORT_R", "FRAG_LLR", "ORIGIN"):
                self.assertIn(f"ID={t}", hdr)
            self.assertTrue(any(r.info.get("DISCORDANT") for r in vf.fetch("chr1")))

        html_text = Path(f"{pred_prefix}_plot.html").read_text(encoding="utf-8")
        for token in ('id="prenatal-badge"', "Fetal Sex:", "Active FF:", "#9333ea"):
            self.assertIn(token, html_text)


if __name__ == "__main__":
    unittest.main()
