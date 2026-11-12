"""Unit tests for prenatal models, inference, SCA calling, and segment annotation."""

from __future__ import annotations

import gzip
import json
import math
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pysam
import yaml

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
    NUM_FRAG_BANDS,
    FetalSex,
    PredictConfig,
    ReadFilters,
    ReferenceConfig,
    SamplesheetEntry,
    assemble_reference_table,
    parquet_path,
)
from ccnv.models.results import (
    FetalFractionResult,
    SCAResult,
    SegmentResult,
)
from ccnv.models.schemas import (
    BRANCH_AUTOSOMAL,
    BRANCH_FEMALE_SEX,
    BRANCH_MALE_SEX,
    BRANCH_PRENATAL_AUTOSOMAL,
    BRANCH_PRENATAL_FFY,
    SAMPLE_SCHEMA,
    ArtifactKind,
    PrenatalModelMetadata,
    build_reference_schema,
    metadata_for,
)
from ccnv.predict import run as run_predict
from ccnv.reference import run as run_reference
from ccnv.report import (
    PRENATAL_SEGMENTS_BED_HEADER,
)


def _build_synthetic_sample(
    contig_lengths: dict[str, int],
    contig_mean_counts: dict[str, float],
    bin_size: int = 5000,
    sample_id: str = "sample",
    noise_scale: float = 0.5,
    seed: int = 42,
    cnv_region: tuple[str, int, int, float] | None = None,
) -> pa.Table:
    """Create a synthetic 5 kb sample table conforming to SAMPLE_SCHEMA."""
    rng = np.random.default_rng(seed)
    contig_list: list[str] = []
    starts_list: list[int] = []
    ends_list: list[int] = []
    counts_list: list[int] = []

    for contig, length in contig_lengths.items():
        starts, ends = bin_intervals(length, bin_size)
        mean_c = contig_mean_counts.get(contig, 100.0)
        base = np.full(starts.size, mean_c, dtype=np.float64)
        if cnv_region is not None and cnv_region[0] == contig:
            _, c_start, c_end, mult = cnv_region
            mask = (starts >= c_start) & (ends <= c_end)
            base[mask] *= mult
        noise = rng.normal(loc=0.0, scale=noise_scale, size=starts.size)
        raw_c = np.clip(np.round(base + noise), 0, None).astype(np.uint64)

        contig_list.extend([contig] * starts.size)
        starts_list.extend(starts.tolist())
        ends_list.extend(ends.tolist())
        counts_list.extend(raw_c.tolist())

    metadata = metadata_for(
        ArtifactKind.SAMPLE,
        contig_lengths=contig_lengths,
        values={
            "bin_size": bin_size,
            "contig_order": list(contig_lengths.keys()),
            "source_identity": {"id": sample_id},
            "reference_identity": {"name": "test_ref"},
            "qc": {
                "effective_reads_examined": 1000,
                "effective_reads_kept": 800,
                "autosomal_reads_kept": 760,
                "sex_reads_kept": 40,
            },
            "filters": asdict(ReadFilters()),
        },
    )

    n_rows = len(contig_list)
    return pa.table(
        {
            "count": pa.array(counts_list, type=pa.uint32()),
            "paired_count": pa.array(np.zeros(n_rows, dtype=np.uint32)),
            "frag_len_sum": pa.array(np.zeros(n_rows, dtype=np.uint64)),
            "short_frag_count": pa.array(np.zeros(n_rows, dtype=np.uint32)),
            "endpoint_count": pa.array(np.zeros(n_rows, dtype=np.uint32)),
            "frag_gc_sum": pa.array(np.zeros(n_rows, dtype=np.float32)),
            "frag_bands": pa.nulls(n_rows, type=pa.list_(pa.uint32(), NUM_FRAG_BANDS)),
        },
        schema=SAMPLE_SCHEMA.with_metadata(metadata),
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
    """Thin test wrapper around 4D ray SCA classification."""
    sca, _, _, _ = _evaluate_sca_and_active_ff(
        fetal_sex=fetal_sex,
        ffy=ffy,
        ffx=ffx,
        ffa=ffa,
        chrx_ratio=chrx_ratio,
        prenatal_meta=prenatal_meta,
    )
    return sca


def annotate_prenatal_segments(
    segments: list[SegmentResult],
    prenatal_result: FetalFractionResult,
    *,
    z_threshold: float = 3.0,
) -> list[SegmentResult]:
    """Thin test wrapper around annotate_prenatal_origin_and_lod."""
    annotated, _ = annotate_prenatal_origin_and_lod(
        segments,
        active_ff=prenatal_result.active_ff,
        sca_call=prenatal_result.sca.call,
        z_threshold=z_threshold,
    )
    return list(annotated)


class TestClassifySCA(unittest.TestCase):
    def setUp(self) -> None:
        self.meta = PrenatalModelMetadata(
            prenatal_bin_size=100000,
            yfrac_threshold=0.005,
            ffy_intercept=0.0,
            ffy_slope=1.0,
            olm_weights=[0.1],
            olm_bias=0.0,
            ffx_intercept=2.0,
            ffx_slope=-2.0,
            female_chrx_ratio_mean=1.0,
            female_chrx_ratio_std=0.01,
        )

    def test_female_xx(self) -> None:
        # Concordant female: chrx_ratio ~ 1.0 -> ffx ~ 0.0, |Z_FFX| < 3.0
        result = classify_sca(
            fetal_sex="F",
            ffy=None,
            ffx=0.0,
            ffa=0.08,
            chrx_ratio=1.0,
            prenatal_meta=self.meta,
        )
        self.assertEqual(result.call, "XX")
        self.assertFalse(result.is_aneuploid)
        self.assertGreater(result.confidence, 0.5)

    def test_female_xo(self) -> None:
        # Turner syndrome (XO): depleted X ratio -> high FFX, Z_FFX >= 3.0
        # female_chrx_ratio_std is 0.01, std_ffx_f is |-2.0| * 0.01 = 0.02
        # ffx = 0.10 -> Z_FFX = 0.10 / 0.02 = 5.0 >= 3.0
        result = classify_sca(
            fetal_sex="F",
            ffy=None,
            ffx=0.10,
            ffa=0.10,
            chrx_ratio=0.95,
            prenatal_meta=self.meta,
        )
        self.assertEqual(result.call, "XO")
        self.assertTrue(result.is_aneuploid)
        self.assertGreater(result.confidence, 0.7)

    def test_female_xxx(self) -> None:
        # Triple X syndrome (XXX): elevated X ratio -> negative FFX, Z_FFX <= -3.0
        # ffx = -0.10 -> Z_FFX = -5.0 <= -3.0
        result = classify_sca(
            fetal_sex="F",
            ffy=None,
            ffx=-0.10,
            ffa=0.10,
            chrx_ratio=1.05,
            prenatal_meta=self.meta,
        )
        self.assertEqual(result.call, "XXX")
        self.assertTrue(result.is_aneuploid)
        self.assertGreater(result.confidence, 0.7)

    def test_male_xy(self) -> None:
        # Concordant male: ffy ~ 0.10, ffx ~ 0.10, ffa ~ 0.10
        result = classify_sca(
            fetal_sex="M",
            ffy=0.10,
            ffx=0.10,
            ffa=0.10,
            chrx_ratio=0.95,
            prenatal_meta=self.meta,
        )
        self.assertEqual(result.call, "XY")
        self.assertFalse(result.is_aneuploid)
        self.assertGreater(result.confidence, 0.5)

    def test_male_xxy(self) -> None:
        # Klinefelter syndrome (XXY): male Y present (ffy=0.10), female 2X dosage
        # Z_XY = (0.0 - 0.10) / 0.02 = -5.0 <= -3.0
        # |Z_FFX| = 0.0 < 3.0
        result = classify_sca(
            fetal_sex="M",
            ffy=0.10,
            ffx=0.0,
            ffa=0.10,
            chrx_ratio=1.0,
            prenatal_meta=self.meta,
        )
        self.assertEqual(result.call, "XXY")
        self.assertTrue(result.is_aneuploid)
        self.assertGreater(result.confidence, 0.7)

    def test_male_xyy(self) -> None:
        # Jacob's syndrome (XYY): doubled Y dosage (ffy=0.20, ffa=0.10, ffx=0.10)
        # Z_XY = (0.10 - 0.20) / 0.02 = -5.0 <= -3.0
        # Z_Y_PREF = (0.20 - 0.10) / 0.015 = 6.67 >= 3.0
        # ffy / ffx = 0.20 / 0.10 = 2.0 > 1.5
        result = classify_sca(
            fetal_sex="M",
            ffy=0.20,
            ffx=0.10,
            ffa=0.10,
            chrx_ratio=0.95,
            prenatal_meta=self.meta,
        )
        self.assertEqual(result.call, "XYY")
        self.assertTrue(result.is_aneuploid)
        self.assertGreater(result.confidence, 0.7)

    def test_male_discordant_mixed(self) -> None:
        # Discordant mixed: e.g. ffx much higher than ffy and ffa
        # Z_XY = (0.30 - 0.05) / 0.02 = 12.5 >= 3.0
        result = classify_sca(
            fetal_sex="M",
            ffy=0.05,
            ffx=0.30,
            ffa=0.05,
            chrx_ratio=0.85,
            prenatal_meta=self.meta,
        )
        self.assertEqual(result.call, "DISCORDANT_MIXED")
        self.assertTrue(result.is_aneuploid)

    def test_classify_sca_bayesian_posterior_and_mahalanobis_discordant_gate(
        self,
    ) -> None:
        # Verify all 7 phenotypes (XX, XY, XO, XXX, XXY, XYY, DISCORDANT_MIXED)
        # return calibrated Bayesian posterior probabilities in (0.5, 1.0]
        cases = [
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
                fetal_sex=fsex,
                ffy=ffy,
                ffx=ffx,
                ffa=pref,
                chrx_ratio=rx,
                prenatal_meta=self.meta,
            )
            self.assertEqual(res.call, exp_call, f"Failed call for {exp_call}")
            self.assertEqual(
                res.is_aneuploid, exp_aneu, f"Failed aneuploidy for {exp_call}"
            )
            self.assertGreater(
                res.confidence, 0.5, f"Confidence too low for {exp_call}"
            )
            self.assertLessEqual(res.confidence, 1.0, f"Confidence > 1 for {exp_call}")

        res_xy = classify_sca(
            "M",
            ffy=0.10,
            ffx=0.10,
            ffa=0.10,
            chrx_ratio=0.95,
            prenatal_meta=self.meta,
        )
        self.assertEqual(res_xy.call, "XY")
        self.assertGreater(res_xy.confidence, 0.85)


class TestAnnotatePrenatalSegments(unittest.TestCase):
    def setUp(self) -> None:
        sca_concordant = SCAResult(
            call="XY",
            is_aneuploid=False,
            confidence=0.9,
            chrx_zscore=0.1,
            ff_delta=0.0,
        )
        self.prenatal_result = FetalFractionResult(
            fetal_sex="M",
            active_ff=0.10,
            active_ff_source="FFY",
            ffy=0.10,
            ffx=0.10,
            ffa=0.10,
            chrx_ratio=0.95,
            chry_score=0.02,
            yfrac_threshold=0.005,
            sca=sca_concordant,
            low_ff_warning=False,
        )

    def test_concordant_gain_and_loss(self) -> None:
        # active_ff = 0.10 -> expected gain log2(1 + 0.05) ~ 0.070389
        expected_gain = math.log2(1.05)
        expected_loss = math.log2(0.95)

        segs = [
            SegmentResult(
                contig="chr1",
                start=0,
                end=100000,
                ratio=expected_gain,
                zscore=4.5,
                weight=10.0,
                bin_count=20,
                call="gain",
            ),
            SegmentResult(
                contig="chr2",
                start=0,
                end=100000,
                ratio=expected_loss,
                zscore=-4.5,
                weight=10.0,
                bin_count=20,
                call="loss",
            ),
        ]
        annotated = annotate_prenatal_segments(segs, self.prenatal_result)
        self.assertEqual(len(annotated), 2)

        self.assertAlmostEqual(
            annotated[0].expected_ratio or 0.0, expected_gain, places=5
        )
        self.assertAlmostEqual(annotated[0].implied_ff or 0.0, 0.10, places=5)
        self.assertFalse(annotated[0].discordant)
        self.assertIsNone(annotated[0].discordance_reason)

        self.assertAlmostEqual(
            annotated[1].expected_ratio or 0.0, expected_loss, places=5
        )
        self.assertAlmostEqual(annotated[1].implied_ff or 0.0, 0.10, places=5)
        self.assertFalse(annotated[1].discordant)
        self.assertIsNone(annotated[1].discordance_reason)

    def test_sub_fetal_ratio(self) -> None:
        # Gain call with ratio < 0.5 * expected_ratio
        expected_gain = math.log2(1.05)
        small_ratio = 0.3 * expected_gain

        seg = SegmentResult(
            contig="chr1",
            start=0,
            end=100000,
            ratio=small_ratio,
            zscore=3.5,
            weight=10.0,
            bin_count=20,
            call="gain",
        )
        annotated = annotate_prenatal_segments([seg], self.prenatal_result)
        self.assertTrue(annotated[0].discordant)
        self.assertEqual(annotated[0].discordance_reason, "SUB_FETAL_RATIO")

    def test_super_fetal_ratio(self) -> None:
        # Full maternal CNV or mosaicism: implied_ff >> active_ff
        # active_ff = 0.10 -> max(2.5 * 0.10, 0.10 + 0.15) = 0.25
        # Maternal dup ratio log2(1.5) ~ 0.585 -> implied_ff = 1.0 > 0.25
        large_ratio = math.log2(1.5)

        seg = SegmentResult(
            contig="chr1",
            start=0,
            end=100000,
            ratio=large_ratio,
            zscore=8.0,
            weight=10.0,
            bin_count=20,
            call="gain",
        )
        annotated = annotate_prenatal_segments([seg], self.prenatal_result)
        self.assertTrue(annotated[0].discordant)
        self.assertEqual(annotated[0].discordance_reason, "SUPER_FETAL_RATIO")

    def test_sca_discordant(self) -> None:
        # When sample has SCA aneuploidy, any X/Y segment is marked SCA_DISCORDANT
        sca_aneuploid = SCAResult(
            call="XO",
            is_aneuploid=True,
            confidence=0.95,
            chrx_zscore=-4.0,
            ff_delta=None,
        )
        prenatal_xo = FetalFractionResult(
            fetal_sex="F",
            active_ff=0.10,
            active_ff_source="FFA",
            ffy=None,
            ffx=0.10,
            ffa=0.10,
            chrx_ratio=0.95,
            chry_score=0.001,
            yfrac_threshold=0.005,
            sca=sca_aneuploid,
            low_ff_warning=False,
        )
        seg_x = SegmentResult(
            contig="chrX",
            start=0,
            end=100000,
            ratio=-0.05,
            zscore=-2.0,
            weight=10.0,
            bin_count=20,
            call=None,
        )
        annotated = annotate_prenatal_segments([seg_x], prenatal_xo)
        self.assertTrue(annotated[0].discordant)
        self.assertEqual(annotated[0].discordance_reason, "SCA_DISCORDANT")


class TestPrenatalModelTrainingAndInference(unittest.TestCase):
    def setUp(self) -> None:
        # Chromosome lengths: 2 autosomes, X, Y
        # Contig lengths multiple of 100 kb
        self.contig_lengths = {
            "chr1": 200_000,
            "chr2": 200_000,
            "chrX": 200_000,
            "chrY": 300_000,
        }

        # Build 6 samples: 3 males, 3 females
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
            sample = _build_synthetic_sample(
                self.contig_lengths,
                counts,
                bin_size=5000,
                sample_id=f"sample_{i}",
                noise_scale=0.1,
                seed=100 + i,
            )
            self.sample_tables.append(sample)
            self.entries.append(
                SamplesheetEntry(
                    filepath=Path(f"sample_{i}.ccnv"),
                    sex="M" if is_male else "F",
                    ff=ff,
                )
            )

    def test_fit_prenatal_models(self) -> None:
        auto_fit, ffy_fit, meta, resolved_sex = fit_prenatal_models(
            self.sample_tables,
            self.entries,
            bin_size=5000,
            prenatal_bin_size=100000,
            seed=42,
        )
        auto_tbl = auto_fit.to_table(target_pca_rank=len(meta.olm_weights))
        ffy_tbl = ffy_fit.to_table(target_pca_rank=len(meta.olm_weights))

        # Validate returned table schemas
        expected_schema = build_reference_schema(len(meta.olm_weights))
        self.assertEqual(auto_tbl.schema, expected_schema)
        self.assertEqual(ffy_tbl.schema, expected_schema)

        # Validate branches
        auto_branches = auto_tbl.column("branch").to_pylist()
        self.assertTrue(all(b == BRANCH_PRENATAL_AUTOSOMAL for b in auto_branches))
        self.assertTrue(
            all(b == BRANCH_PRENATAL_FFY for b in ffy_tbl.column("branch").to_pylist())
        )

        # Validate sex classification
        self.assertEqual(resolved_sex, (False, False, False, True, True, True))

        # Validate metadata fields
        self.assertEqual(meta.prenatal_bin_size, 100000)
        self.assertGreater(meta.yfrac_threshold, 0.0)
        self.assertGreater(len(meta.olm_weights), 0)
        self.assertTrue(meta.ffa_logit_mode)

    def test_predict_prenatal(self) -> None:
        auto_fit, ffy_fit, meta, _ = fit_prenatal_models(
            self.sample_tables,
            self.entries,
            bin_size=5000,
            prenatal_bin_size=100000,
            seed=42,
        )

        # Combine into reference table
        ref_table = assemble_reference_table([auto_fit, ffy_fit])

        # Predict on a male sample
        male_sample = self.sample_tables[4]
        res_male = predict_prenatal(male_sample, ref_table, meta, min_ff_warning=0.04)
        self.assertEqual(res_male.fetal_sex, "M")
        self.assertIsNotNone(res_male.ffy)
        self.assertGreater(res_male.active_ff, 0.0)
        self.assertIn(res_male.active_ff_source, ("FFY", "FUSED"))

        # Predict on a female sample
        female_sample = self.sample_tables[1]
        res_female = predict_prenatal(
            female_sample, ref_table, meta, min_ff_warning=0.04
        )
        self.assertEqual(res_female.fetal_sex, "F")
        self.assertIsNone(res_female.ffy)
        self.assertEqual(res_female.active_ff_source, "FFA")

    def test_predict_prenatal_olm_fallback(self) -> None:
        auto_fit, ffy_fit, meta, _ = fit_prenatal_models(
            self.sample_tables,
            self.entries,
            bin_size=5000,
            prenatal_bin_size=100000,
            seed=42,
        )
        ref_table = assemble_reference_table([auto_fit, ffy_fit])
        fallback_meta = replace(
            meta,
            olm_weights=[0.0] * len(meta.olm_weights),
            olm_bias=0.12,
            ffa_logit_mode=False,
        )
        female_sample = self.sample_tables[1]
        res = predict_prenatal(female_sample, ref_table, fallback_meta)
        self.assertEqual(res.active_ff_source, "FFA")
        self.assertAlmostEqual(res.ffa, 0.12, places=1)

    def test_fit_prenatal_models_validation_errors(self) -> None:
        with self.assertRaises(ValueError):
            fit_prenatal_models([], [], 5000, 100000)

        with self.assertRaises(ValueError):
            fit_prenatal_models(self.sample_tables, self.entries[:2], 5000, 100000)

    def test_fit_prenatal_models_all_female_cohort(self) -> None:
        females = self.sample_tables[:3]
        f_entries = self.entries[:3]
        auto_fit, ffy_fit, meta, resolved_sex = fit_prenatal_models(
            females, f_entries, bin_size=5000, prenatal_bin_size=100000
        )
        self.assertEqual(resolved_sex, (False, False, False))
        self.assertEqual(auto_fit.num_rows, 4)
        self.assertEqual(ffy_fit.num_rows, 60)
        self.assertIsNotNone(meta)

    def test_predict_prenatal_missing_branch_errors(self) -> None:
        auto_fit, ffy_fit, meta, _ = fit_prenatal_models(
            self.sample_tables, self.entries, 5000, 100000
        )
        with self.assertRaises(ValueError):
            predict_prenatal(self.sample_tables[0], auto_fit.to_table(), meta)
        with self.assertRaises(ValueError):
            predict_prenatal(self.sample_tables[0], ffy_fit.to_table(), meta)

    def test_predict_prenatal_low_ff_warning(self) -> None:
        auto_fit, ffy_fit, meta, _ = fit_prenatal_models(
            self.sample_tables, self.entries, 5000, 100000
        )
        ref_table = assemble_reference_table([auto_fit, ffy_fit])

        # High min_ff_warning threshold (0.50) triggers low_ff_warning
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
        # Empty segments
        self.assertEqual(annotate_prenatal_segments([], ff_zero), [])

        # Uncalled autosomal segment
        seg = SegmentResult("chr1", 0, 1000, 0.01, 0.5, 1.0, 5, None)
        res = annotate_prenatal_segments([seg], ff_zero)
        self.assertEqual(len(res), 1)
        self.assertFalse(res[0].discordant)
        self.assertIsNone(res[0].discordance_reason)
        self.assertEqual(res[0].expected_ratio, 0.0)

    def test_predict_prenatal_autosomal_coordinate_fallback_with_extra_contigs(
        self,
    ) -> None:
        auto_fit, ffy_fit, meta, _ = fit_prenatal_models(
            self.sample_tables, self.entries, 5000, 100000
        )
        ref_table = assemble_reference_table([auto_fit, ffy_fit])

        # Extra contigs (chr13, chr18, chr21) present in sample
        # but not in prenatal_autosomal reference
        sample_lengths = {
            "chr1": 200_000,
            "chr2": 200_000,
            "chr13": 200_000,
            "chr18": 200_000,
            "chr21": 200_000,
            "chrX": 200_000,
            "chrY": 300_000,
        }
        counts = {
            "chr1": 100.0,
            "chr2": 100.0,
            "chr13": 100.0,
            "chr18": 100.0,
            "chr21": 100.0,
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
        auto_fit, ffy_fit, meta, _ = fit_prenatal_models(
            self.sample_tables, self.entries, 5000, 100000
        )
        ref_table = assemble_reference_table([auto_fit, ffy_fit])

        # Fewer chrY bins (200 kb vs reference 300 kb)
        sample_lengths = {
            "chr1": 200_000,
            "chr2": 200_000,
            "chrX": 200_000,
            "chrY": 200_000,
        }
        counts = {
            "chr1": 100.0,
            "chr2": 100.0,
            "chrX": 95.0,
            "chrY": 5.0,
        }
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


def _setup_synthetic_cohort_and_references(
    root: Path,
    contig_lengths: dict[str, int],
    *,
    build_germline: bool = False,
    qc_report: bool = True,
) -> tuple[Path, Path | None, Path]:
    sample_dir = root / "cohort"
    sample_dir.mkdir()
    entries_data = []

    for i in range(6):
        is_male = i >= 3
        ff = 0.08 + 0.02 * (i - 3) if is_male else None
        counts = {
            "chr1": 100.0,
            "chr2": 100.0,
            "chrX": 100.0 * (1.0 - 0.5 * (ff or 0.0)),
            "chrY": 100.0 * (0.5 * (ff or 0.0)) if is_male else 0.5,
        }
        s_tbl = _build_synthetic_sample(
            contig_lengths,
            counts,
            bin_size=5000,
            sample_id=f"cohort_{i}",
            noise_scale=0.1,
            seed=200 + i,
        )
        s_path = sample_dir / f"cohort_{i}.ccnv"
        write_sample_parquet(s_tbl, s_path)
        entries_data.append(
            {
                "filepath": str(s_path),
                "sex": "M" if is_male else "F",
                "ff": ff,
            }
        )

    sheet_path = root / "samplesheet.yaml"
    with open(sheet_path, "w", encoding="utf-8") as f:
        yaml.dump(entries_data, f)

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
            qc_report=qc_report,
        )
    )

    germline_ref: Path | None = None
    if build_germline:
        germline_prefix = root / "germline_ref"
        run_reference(
            ReferenceConfig(
                samplesheet_path=sheet_path,
                prefix=germline_prefix,
                mode="germline",
                bin_size=5000,
                min_cohort_size=4,
                min_sex_group_size=2,
                pca_components=2,
                qc_report=qc_report,
            )
        )
        germline_ref = parquet_path(germline_prefix)

    test_counts = {
        "chr1": 100.0,
        "chr2": 100.0,
        "chrX": 130.625,
        "chrY": 6.875,
    }
    test_sample_tbl = _build_synthetic_sample(
        contig_lengths,
        test_counts,
        bin_size=5000,
        sample_id="test_male_sample",
        noise_scale=0.1,
        seed=999,
        cnv_region=("chr1", 100_000, 200_000, 2.5),
    )
    test_sample_path = root / "test_male_sample.ccnv"
    write_sample_parquet(test_sample_tbl, test_sample_path)

    return (
        parquet_path(prenatal_prefix),
        germline_ref,
        test_sample_path,
    )


class TestPrenatalEndToEndIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.contig_lengths = {
            "chr1": 200_000,
            "chr2": 200_000,
            "chrX": 200_000,
            "chrY": 300_000,
        }

    def test_prenatal_pipeline_deliverables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prenatal_ref, _, test_sample = _setup_synthetic_cohort_and_references(
                root, self.contig_lengths
            )

            # Validate reference branches
            ref_table = pq.read_table(prenatal_ref)
            branches = set(ref_table.column("branch").unique().to_pylist())
            for branch in (
                BRANCH_AUTOSOMAL,
                BRANCH_FEMALE_SEX,
                BRANCH_MALE_SEX,
                BRANCH_PRENATAL_AUTOSOMAL,
                BRANCH_PRENATAL_FFY,
            ):
                self.assertIn(branch, branches)

            pred_prefix = root / "predict_out"
            outputs = run_predict(
                PredictConfig(
                    sample=test_sample,
                    reference=prenatal_ref,
                    prefix=pred_prefix,
                    mode=AnalysisMode.prenatal,
                    plot=True,
                    vcf=True,
                    penalty=0.5,
                    min_reference_bins=1,
                    z_threshold=1.5,
                )
            )

            # 1. Assert <prefix>_qc.json
            self.assertIsNotNone(outputs.qc_json_path)
            self.assertTrue(outputs.qc_json_path.is_file())
            with open(outputs.qc_json_path, encoding="utf-8") as f:
                qc = json.load(f)
            self.assertEqual(qc["mode"], "prenatal")
            self.assertIn("prenatal", qc)
            p_qc = qc["prenatal"]
            self.assertIsNotNone(p_qc)
            for key in (
                "fetal_sex",
                "active_ff",
                "active_ff_source",
                "sca",
                "ffy",
                "ffa",
                "ffx",
            ):
                self.assertIn(key, p_qc)
            self.assertEqual(p_qc["fetal_sex"], "M")
            self.assertGreater(p_qc["active_ff"], 0.0)
            self.assertIn(p_qc["active_ff_source"], ("FFY", "FUSED"))
            self.assertTrue(any(s.discordant for s in outputs.segments))

            # 2. Assert <prefix>_segments.bed.gz (15 columns with #contig header)
            self.assertTrue(outputs.segments_bed.is_file())
            with gzip.open(outputs.segments_bed, "rt") as f:
                bed_lines = [
                    line.rstrip("\r\n").split("\t") for line in f if line.strip()
                ]
            self.assertGreater(len(bed_lines), 1)
            self.assertEqual(tuple(bed_lines[0]), PRENATAL_SEGMENTS_BED_HEADER)
            self.assertEqual(len(bed_lines[0]), 15)
            for expected_col in (
                "expected_ratio",
                "implied_ff",
                "discordant",
                "discordance_reason",
                "frag_short_ratio",
                "frag_llr_size",
                "origin_call",
            ):
                self.assertIn(expected_col, bed_lines[0])
            for row in bed_lines[1:]:
                self.assertEqual(len(row), 15)
            disc_rows = [r for r in bed_lines[1:] if r[10] == "true"]
            self.assertGreaterEqual(len(disc_rows), 1)
            disc_reasons = {r[11] for r in disc_rows}
            self.assertIn("SUPER_FETAL_RATIO", disc_reasons)

            # 3. Assert <prefix>_segments.vcf.gz (VCF headers for prenatal tags)
            self.assertIsNotNone(outputs.segments_vcf)
            self.assertTrue(outputs.segments_vcf.is_file())
            with pysam.VariantFile(str(outputs.segments_vcf)) as vf:
                vcf_header_text = str(vf.header)
                for tag in (
                    "EXP_LOG2R",
                    "IMPLIED_FF",
                    "DISCORDANT",
                    "DISC_REASON",
                    "FRAG_SHORT_R",
                    "FRAG_LLR",
                    "ORIGIN",
                ):
                    self.assertIn(f"ID={tag}", vcf_header_text)
                records = list(vf.fetch("chr1"))
                self.assertTrue(any(rec.info.get("DISCORDANT") for rec in records))

            # 4. Assert <prefix>_plot.html (status badge and discordant traces)
            plot_path = Path(f"{pred_prefix}_plot.html")
            self.assertTrue(plot_path.is_file())
            html_text = plot_path.read_text(encoding="utf-8")
            self.assertIn('id="prenatal-badge"', html_text)
            self.assertIn("Fetal Sex:", html_text)
            self.assertIn("Active FF:", html_text)
            self.assertIn("#9333ea", html_text)
            self.assertIn("Discordant", html_text)


if __name__ == "__main__":
    unittest.main()
