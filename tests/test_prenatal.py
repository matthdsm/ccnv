"""Unit tests for prenatal models, inference, SCA calling, and segment annotation."""

from __future__ import annotations

import math
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa

from ccnv.convert import bin_intervals
from ccnv.models import ReadFilters, SamplesheetEntry
from ccnv.prenatal import (
    annotate_prenatal_segments,
    classify_sca,
    fit_prenatal_models,
    predict_prenatal,
)
from ccnv.results import (
    FetalFractionResult,
    SCAResult,
    SegmentResult,
)
from ccnv.schemas import (
    BRANCH_PRENATAL_AUTOSOMAL,
    BRANCH_PRENATAL_FFY,
    REFERENCE_SCHEMA,
    SAMPLE_SCHEMA,
    ArtifactKind,
    PrenatalModelMetadata,
    metadata_for,
)


def _build_synthetic_sample(
    contig_lengths: dict[str, int],
    contig_mean_counts: dict[str, float],
    bin_size: int = 5000,
    sample_id: str = "sample",
    noise_scale: float = 0.5,
    seed: int = 42,
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
        noise = rng.normal(loc=0.0, scale=noise_scale, size=starts.size)
        raw_c = np.clip(np.round(mean_c + noise), 0, None).astype(np.uint64)

        contig_list.extend([contig] * starts.size)
        starts_list.extend(starts.tolist())
        ends_list.extend(ends.tolist())
        counts_list.extend(raw_c.tolist())

    metadata = metadata_for(
        ArtifactKind.SAMPLE,
        contig_lengths=contig_lengths,
        values={
            "bin_size": bin_size,
            "source_identity": {"id": sample_id},
            "reference_identity": {"name": "test_ref"},
            "qc": {},
            "filters": asdict(ReadFilters()),
        },
    )

    return pa.table(
        {
            "contig": pa.array(contig_list, type=pa.string()),
            "start": pa.array(starts_list, type=pa.int64()),
            "end": pa.array(ends_list, type=pa.int64()),
            "count": pa.array(counts_list, type=pa.uint64()),
        },
        schema=SAMPLE_SCHEMA.with_metadata(metadata),
    )


class TestClassifySCA(unittest.TestCase):
    def setUp(self) -> None:
        self.meta = PrenatalModelMetadata(
            prenatal_bin_size=100000,
            yfrac_threshold=0.005,
            ffy_intercept=0.0,
            ffy_slope=1.0,
            mlp_w1=[[0.1, -0.1]],
            mlp_b1=[0.0, 0.0],
            mlp_w2=[[0.5], [-0.5]],
            mlp_b2=[0.0],
            mlp_skew_intercept=0.0,
            mlp_skew_slope=1.0,
            olm_weights=[0.1],
            olm_bias=0.0,
            olm_skew_intercept=0.0,
            olm_skew_slope=1.0,
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
            preface=0.08,
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
            preface=0.10,
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
            preface=0.10,
            chrx_ratio=1.05,
            prenatal_meta=self.meta,
        )
        self.assertEqual(result.call, "XXX")
        self.assertTrue(result.is_aneuploid)
        self.assertGreater(result.confidence, 0.7)

    def test_male_xy(self) -> None:
        # Concordant male: ffy ~ 0.10, ffx ~ 0.10, preface ~ 0.10
        result = classify_sca(
            fetal_sex="M",
            ffy=0.10,
            ffx=0.10,
            preface=0.10,
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
            preface=0.10,
            chrx_ratio=1.0,
            prenatal_meta=self.meta,
        )
        self.assertEqual(result.call, "XXY")
        self.assertTrue(result.is_aneuploid)
        self.assertGreater(result.confidence, 0.7)

    def test_male_xyy(self) -> None:
        # Jacob's syndrome (XYY): doubled Y dosage (ffy=0.20, preface=0.10, ffx=0.10)
        # Z_XY = (0.10 - 0.20) / 0.02 = -5.0 <= -3.0
        # Z_Y_PREF = (0.20 - 0.10) / 0.015 = 6.67 >= 3.0
        # ffy / ffx = 0.20 / 0.10 = 2.0 > 1.5
        result = classify_sca(
            fetal_sex="M",
            ffy=0.20,
            ffx=0.10,
            preface=0.10,
            chrx_ratio=0.95,
            prenatal_meta=self.meta,
        )
        self.assertEqual(result.call, "XYY")
        self.assertTrue(result.is_aneuploid)
        self.assertGreater(result.confidence, 0.7)

    def test_male_discordant_mixed(self) -> None:
        # Discordant mixed: e.g. ffx much higher than ffy and preface
        # Z_XY = (0.30 - 0.05) / 0.02 = 12.5 >= 3.0
        result = classify_sca(
            fetal_sex="M",
            ffy=0.05,
            ffx=0.30,
            preface=0.05,
            chrx_ratio=0.85,
            prenatal_meta=self.meta,
        )
        self.assertEqual(result.call, "DISCORDANT_MIXED")
        self.assertTrue(result.is_aneuploid)


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
            preface=0.10,
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
            active_ff_source="PREFACE",
            ffy=None,
            ffx=0.10,
            preface=0.10,
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
        auto_tbl, ffy_tbl, meta, resolved_sex = fit_prenatal_models(
            self.sample_tables,
            self.entries,
            bin_size=5000,
            prenatal_bin_size=100000,
            seed=42,
        )

        # Validate returned table schemas
        self.assertEqual(auto_tbl.schema, REFERENCE_SCHEMA)
        self.assertEqual(ffy_tbl.schema, REFERENCE_SCHEMA)

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
        self.assertEqual(len(meta.mlp_b1), 2)
        self.assertEqual(len(meta.mlp_b2), 1)
        self.assertEqual(len(meta.mlp_w2), 2)

    def test_predict_prenatal(self) -> None:
        auto_tbl, ffy_tbl, meta, _ = fit_prenatal_models(
            self.sample_tables,
            self.entries,
            bin_size=5000,
            prenatal_bin_size=100000,
            seed=42,
        )

        # Combine into reference table
        ref_table = pa.concat_tables([auto_tbl, ffy_tbl])

        # Predict on a male sample
        male_sample = self.sample_tables[4]
        res_male = predict_prenatal(male_sample, ref_table, meta, min_ff_warning=0.04)
        self.assertEqual(res_male.fetal_sex, "M")
        self.assertIsNotNone(res_male.ffy)
        self.assertGreater(res_male.active_ff, 0.0)
        self.assertEqual(res_male.active_ff_source, "FFY")

        # Predict on a female sample
        female_sample = self.sample_tables[1]
        res_female = predict_prenatal(
            female_sample, ref_table, meta, min_ff_warning=0.04
        )
        self.assertEqual(res_female.fetal_sex, "F")
        self.assertIsNone(res_female.ffy)
        self.assertEqual(res_female.active_ff_source, "PREFACE")

    def test_fit_prenatal_models_validation_errors(self) -> None:
        with self.assertRaises(ValueError):
            fit_prenatal_models([], [], 5000, 100000)

        with self.assertRaises(ValueError):
            fit_prenatal_models(self.sample_tables, self.entries[:2], 5000, 100000)

    def test_fit_prenatal_models_all_female_cohort(self) -> None:
        females = self.sample_tables[:3]
        f_entries = self.entries[:3]
        auto_tbl, ffy_tbl, meta, resolved_sex = fit_prenatal_models(
            females, f_entries, bin_size=5000, prenatal_bin_size=100000
        )
        self.assertEqual(resolved_sex, (False, False, False))
        self.assertEqual(auto_tbl.num_rows, 4)
        self.assertEqual(ffy_tbl.num_rows, 60)
        self.assertIsNotNone(meta)

    def test_predict_prenatal_missing_branch_errors(self) -> None:
        auto_tbl, ffy_tbl, meta, _ = fit_prenatal_models(
            self.sample_tables, self.entries, 5000, 100000
        )
        with self.assertRaises(ValueError):
            predict_prenatal(self.sample_tables[0], auto_tbl, meta)
        with self.assertRaises(ValueError):
            predict_prenatal(self.sample_tables[0], ffy_tbl, meta)

    def test_predict_prenatal_low_ff_warning(self) -> None:
        auto_tbl, ffy_tbl, meta, _ = fit_prenatal_models(
            self.sample_tables, self.entries, 5000, 100000
        )
        ref_table = pa.concat_tables([auto_tbl, ffy_tbl])

        # High min_ff_warning threshold (0.50) triggers low_ff_warning
        res = predict_prenatal(
            self.sample_tables[0], ref_table, meta, min_ff_warning=0.50
        )
        self.assertTrue(res.low_ff_warning)
        self.assertTrue(any("below warning threshold" in w for w in res.warnings))

    def test_annotate_prenatal_segments_edge_cases(self) -> None:
        sca = SCAResult("XX", False, 0.9, 0.0)
        ff_zero = FetalFractionResult(
            "F", 0.0, "PREFACE", None, None, 0.0, 1.0, 0.001, 0.005, sca
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
        auto_tbl, ffy_tbl, meta, _ = fit_prenatal_models(
            self.sample_tables, self.entries, 5000, 100000
        )
        ref_table = pa.concat_tables([auto_tbl, ffy_tbl])

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
        auto_tbl, ffy_tbl, meta, _ = fit_prenatal_models(
            self.sample_tables, self.entries, 5000, 100000
        )
        ref_table = pa.concat_tables([auto_tbl, ffy_tbl])

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
            "F", 0.10, "PREFACE", None, None, 0.10, 1.0, 0.001, 0.005, sca
        )
        seg = SegmentResult("chr1", 0, 1000, 0.01, 3.5, 1.0, 5, None)
        res = annotate_prenatal_segments([seg], ff_active, z_threshold=3.0)
        self.assertTrue(res[0].discordant)
        self.assertEqual(res[0].discordance_reason, "SUB_FETAL_RATIO")


if __name__ == "__main__":
    unittest.main()
