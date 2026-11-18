"""Unit tests for domain models, configurations, and result types."""

import unittest
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from pathlib import Path

from ccnv.models import (
    PredictConfig,
    PrenatalModelMetadata,
    ReferenceConfig,
    SamplesheetEntry,
)
from ccnv.models.results import (
    BranchQCStatus,
    FetalFractionResult,
    GestationType,
    PredictionOutputs,
    ReferenceQCOutputs,
    SCACall,
    SCAResult,
    SegmentResult,
    TwinResult,
    TwinSexPair,
)
from ccnv.stats.prenatal_math import classify_sca_ray_4d


def _call[T](fn: Callable[..., T], *args: object, **kwargs: object) -> T:
    return fn(*args, **kwargs)


def _set_attr(obj: object, name: str, value: object) -> None:
    setattr(obj, name, value)


class TestSamplesheetEntry(unittest.TestCase):
    def test_samplesheet_entry_defaults_and_types(self) -> None:
        entry = SamplesheetEntry(filepath=Path("sample1.ccnv"))
        self.assertEqual(entry.filepath, Path("sample1.ccnv"))
        self.assertIsNone(entry.sex)
        self.assertIsNone(entry.ff)

    def test_samplesheet_entry_string_path_conversion(self) -> None:
        entry = _call(SamplesheetEntry, filepath="sample1.ccnv")
        self.assertEqual(entry.filepath, Path("sample1.ccnv"))
        self.assertIsInstance(entry.filepath, Path)

    def test_samplesheet_entry_explicit_values(self) -> None:
        entry = SamplesheetEntry(
            filepath=Path("sample_male.ccnv"),
            sex="M",
            ff=0.12,
        )
        self.assertEqual(entry.filepath, Path("sample_male.ccnv"))
        self.assertEqual(entry.sex, "M")
        self.assertAlmostEqual(entry.ff or 0.0, 0.12)

    def test_samplesheet_entry_invalid_sex(self) -> None:
        with self.assertRaisesRegex(ValueError, "sex must be 'M' or 'F'"):
            _call(SamplesheetEntry, filepath=Path("sample.ccnv"), sex="X")

    def test_samplesheet_entry_invalid_ff_out_of_bounds(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "fetal fraction ff must be between 0.0 and 1.0"
        ):
            SamplesheetEntry(filepath=Path("sample.ccnv"), ff=-0.05)
        with self.assertRaisesRegex(
            ValueError, "fetal fraction ff must be between 0.0 and 1.0"
        ):
            SamplesheetEntry(filepath=Path("sample.ccnv"), ff=1.05)

    def test_samplesheet_entry_immutability(self) -> None:
        entry = SamplesheetEntry(filepath=Path("sample1.ccnv"), sex="F", ff=0.08)
        with self.assertRaises(FrozenInstanceError):
            _set_attr(entry, "sex", "M")


class TestReferenceConfig(unittest.TestCase):
    def test_reference_config_defaults(self) -> None:
        config = ReferenceConfig(prefix=Path("ref_out"))
        self.assertEqual(config.prefix, Path("ref_out"))
        self.assertEqual(config.mode, "germline")
        self.assertEqual(config.prenatal_bin_size, 100000)
        self.assertIsNone(config.samplesheet_path)
        self.assertIsNone(config.pca_components)
        self.assertEqual(config.entries, ())
        self.assertEqual(config.threads, 1)

    def test_reference_config_entries_preserved(self) -> None:
        e1 = SamplesheetEntry(filepath=Path("s1.ccnv"), sex="M", ff=0.10)
        e2 = SamplesheetEntry(filepath=Path("s2.ccnv"), sex="F", ff=None)
        config = ReferenceConfig(prefix=Path("ref"), entries=(e1, e2))
        self.assertEqual(config.entries, (e1, e2))
        self.assertEqual(config.prefix, Path("ref"))

    def test_reference_config_prenatal_mode(self) -> None:
        config = ReferenceConfig(
            prefix=Path("ref_prenatal"),
            mode="prenatal",
            prenatal_bin_size=50000,
            samplesheet_path=Path("samplesheet.yaml"),
        )
        self.assertEqual(config.mode, "prenatal")
        self.assertEqual(config.prenatal_bin_size, 50000)
        self.assertEqual(config.samplesheet_path, Path("samplesheet.yaml"))

    def test_reference_config_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "threads must be a positive integer"):
            ReferenceConfig(prefix=Path("ref"), threads=0)
        with self.assertRaisesRegex(ValueError, "bin_size must be a positive integer"):
            ReferenceConfig(prefix=Path("ref"), bin_size=0)
        with self.assertRaisesRegex(
            ValueError, "min_mappability must be between 0.0 and 1.0"
        ):
            ReferenceConfig(prefix=Path("ref"), min_mappability=1.1)
        with self.assertRaisesRegex(
            ValueError, "prenatal_bin_size must be a positive integer"
        ):
            ReferenceConfig(prefix=Path("ref"), prenatal_bin_size=0)
        with self.assertRaisesRegex(
            ValueError, "mode must be 'germline' or 'prenatal'"
        ):
            _call(ReferenceConfig, prefix=Path("ref"), mode="invalid")

    def test_reference_config_immutability(self) -> None:
        config = ReferenceConfig(prefix=Path("ref"))
        with self.assertRaises(FrozenInstanceError):
            _set_attr(config, "mode", "prenatal")

    def test_reference_config_threads_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "threads must be a positive integer"):
            ReferenceConfig(threads=0)
        with self.assertRaisesRegex(ValueError, "threads must be a positive integer"):
            ReferenceConfig(threads=-1)


class TestPredictConfig(unittest.TestCase):
    def test_predict_config_defaults(self) -> None:
        config = PredictConfig(
            sample=Path("sample.ccnv"),
            reference=Path("ref.ccnv"),
            prefix=Path("out"),
        )
        self.assertEqual(config.mode, "germline")
        self.assertAlmostEqual(config.min_fetal_fraction_warning, 0.04)
        self.assertEqual(config.sex, "autosomal")
        self.assertEqual(config.threads, 1)
        self.assertFalse(config.bins_vcf)
        self.assertFalse(config.twins)
        self.assertFalse(hasattr(config, "bin_size"))

    def test_predict_config_prenatal_mode(self) -> None:
        config = PredictConfig(
            sample=Path("sample.ccnv"),
            reference=Path("ref.ccnv"),
            prefix=Path("out"),
            mode="prenatal",
            min_fetal_fraction_warning=0.05,
            twins=True,
        )
        self.assertEqual(config.mode, "prenatal")
        self.assertAlmostEqual(config.min_fetal_fraction_warning, 0.05)
        self.assertTrue(config.twins)

    def test_predict_config_validation(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "mode must be 'germline' or 'prenatal'"
        ):
            _call(
                PredictConfig,
                sample=Path("s"),
                reference=Path("r"),
                prefix=Path("o"),
                mode="other",
            )
        with self.assertRaisesRegex(
            ValueError, "min_fetal_fraction_warning must be between 0.0 and 1.0"
        ):
            PredictConfig(
                sample=Path("s"),
                reference=Path("r"),
                prefix=Path("o"),
                min_fetal_fraction_warning=-0.01,
            )
        with self.assertRaisesRegex(
            ValueError, "min_mappability must be between 0.0 and 1.0"
        ):
            PredictConfig(
                sample=Path("s"),
                reference=Path("r"),
                prefix=Path("o"),
                min_mappability=-0.1,
            )
        with self.assertRaisesRegex(ValueError, "threads must be a positive integer"):
            PredictConfig(
                sample=Path("s"),
                reference=Path("r"),
                prefix=Path("o"),
                threads=0,
            )
        with self.assertRaisesRegex(TypeError, "twins must be a boolean"):
            _call(
                PredictConfig,
                sample=Path("s"),
                reference=Path("r"),
                prefix=Path("o"),
                twins="yes",
            )

    def test_predict_config_immutability(self) -> None:
        config = PredictConfig(
            sample=Path("s"),
            reference=Path("r"),
            prefix=Path("o"),
        )
        with self.assertRaises(FrozenInstanceError):
            _set_attr(config, "mode", "prenatal")


class TestSegmentResult(unittest.TestCase):
    def test_segment_result_defaults(self) -> None:
        seg = SegmentResult(
            contig="chr1",
            start=0,
            end=100000,
            ratio=0.5,
            zscore=3.5,
            weight=1.0,
            bin_count=20,
            call="gain",
        )
        self.assertIsNone(seg.expected_ratio)
        self.assertIsNone(seg.implied_ff)
        self.assertFalse(seg.discordant)
        self.assertIsNone(seg.discordance_reason)

    def test_segment_result_prenatal_fields(self) -> None:
        seg = SegmentResult(
            contig="chr21",
            start=0,
            end=100000,
            ratio=0.08,
            zscore=4.2,
            weight=1.0,
            bin_count=20,
            call="gain",
            expected_ratio=0.075,
            implied_ff=0.15,
            discordant=True,
            discordance_reason="SUB_FETAL_RATIO",
        )
        self.assertEqual(seg.expected_ratio, 0.075)
        self.assertEqual(seg.implied_ff, 0.15)
        self.assertTrue(seg.discordant)
        self.assertEqual(seg.discordance_reason, "SUB_FETAL_RATIO")

    def test_segment_result_immutability(self) -> None:
        seg = SegmentResult("chr1", 0, 100, 0.0, 0.0, 1.0, 1, None)
        with self.assertRaises(FrozenInstanceError):
            _set_attr(seg, "discordant", True)


class TestSCAResult(unittest.TestCase):
    def test_sca_result_calls(self) -> None:
        valid_calls: tuple[SCACall, ...] = (
            "XX",
            "XY",
            "XO",
            "XXX",
            "XXY",
            "XYY",
            "DISCORDANT_MIXED",
        )
        for call in valid_calls:
            sca = SCAResult(
                call=call,
                is_aneuploid=call not in ("XX", "XY"),
                confidence=0.95,
                chrx_zscore=1.2,
            )
            self.assertEqual(sca.call, call)
            self.assertIsNone(sca.ff_delta)

    def test_sca_result_explicit_ff_delta(self) -> None:
        sca = SCAResult(
            call="XO",
            is_aneuploid=True,
            confidence=0.99,
            chrx_zscore=-4.5,
            ff_delta=0.08,
        )
        self.assertEqual(sca.ff_delta, 0.08)

    def test_sca_result_invalid_call(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid SCA call"):
            _call(
                SCAResult,
                call="YY",
                is_aneuploid=True,
                confidence=0.5,
                chrx_zscore=0.0,
            )

    def test_sca_result_immutability(self) -> None:
        sca = SCAResult("XX", False, 0.99, 0.1)
        with self.assertRaises(FrozenInstanceError):
            _set_attr(sca, "confidence", 1.0)


class TestFetalFractionResult(unittest.TestCase):
    def test_fetal_fraction_result_instantiation(self) -> None:
        sca = SCAResult(
            call="XY",
            is_aneuploid=False,
            confidence=0.98,
            chrx_zscore=-0.5,
        )
        ff_res = FetalFractionResult(
            fetal_sex="M",
            active_ff=0.105,
            active_ff_source="FFY",
            ffy=0.105,
            ffx=0.098,
            ffa=0.112,
            chrx_ratio=0.95,
            chry_score=0.045,
            yfrac_threshold=0.015,
            sca=sca,
        )
        self.assertEqual(ff_res.fetal_sex, "M")
        self.assertEqual(ff_res.active_ff, 0.105)
        self.assertEqual(ff_res.active_ff_source, "FFY")
        self.assertEqual(ff_res.ffy, 0.105)
        self.assertEqual(ff_res.ffx, 0.098)
        self.assertEqual(ff_res.ffa, 0.112)
        self.assertEqual(ff_res.sca, sca)
        self.assertFalse(ff_res.low_ff_warning)
        self.assertEqual(ff_res.warnings, ())

    def test_fetal_fraction_result_warnings_tuple_conversion(self) -> None:
        sca = SCAResult("XX", False, 0.99, 0.0)
        ff_res = _call(
            FetalFractionResult,
            fetal_sex="F",
            active_ff=0.03,
            active_ff_source="FFA",
            ffy=None,
            ffx=0.03,
            ffa=0.03,
            chrx_ratio=1.0,
            chry_score=0.001,
            yfrac_threshold=0.015,
            sca=sca,
            low_ff_warning=True,
            warnings=["fetal fraction below threshold", "female fetus"],
        )
        self.assertTrue(ff_res.low_ff_warning)
        self.assertIsInstance(ff_res.warnings, tuple)
        self.assertEqual(
            ff_res.warnings,
            ("fetal fraction below threshold", "female fetus"),
        )

    def test_fetal_fraction_result_validation(self) -> None:
        sca = SCAResult("XX", False, 0.99, 0.0)
        with self.assertRaisesRegex(ValueError, "fetal_sex must be 'M' or 'F'"):
            _call(
                FetalFractionResult,
                fetal_sex="U",
                active_ff=0.1,
                active_ff_source="FFA",
                ffy=None,
                ffx=0.1,
                ffa=0.1,
                chrx_ratio=1.0,
                chry_score=0.0,
                yfrac_threshold=0.01,
                sca=sca,
            )
        ffx_res = FetalFractionResult(
            fetal_sex="F",
            active_ff=0.09,
            active_ff_source="FFX",
            ffy=None,
            ffx=0.09,
            ffa=0.09,
            chrx_ratio=1.0,
            chry_score=0.0,
            yfrac_threshold=0.01,
            sca=sca,
        )
        self.assertEqual(ffx_res.active_ff_source, "FFX")
        with self.assertRaisesRegex(
            ValueError,
            "active_ff_source must be 'FFY', 'FFA', 'FFX', 'FF_SIZE', or 'FUSED'",
        ):
            _call(
                FetalFractionResult,
                fetal_sex="F",
                active_ff=0.1,
                active_ff_source="OTHER",
                ffy=None,
                ffx=0.1,
                ffa=0.1,
                chrx_ratio=1.0,
                chry_score=0.0,
                yfrac_threshold=0.01,
                sca=sca,
            )

    def test_fetal_fraction_result_immutability(self) -> None:
        sca = SCAResult("XX", False, 0.99, 0.0)
        ff_res = FetalFractionResult(
            fetal_sex="F",
            active_ff=0.08,
            active_ff_source="FFA",
            ffy=None,
            ffx=0.08,
            ffa=0.08,
            chrx_ratio=1.0,
            chry_score=0.0,
            yfrac_threshold=0.01,
            sca=sca,
        )
        with self.assertRaises(FrozenInstanceError):
            _set_attr(ff_res, "active_ff", 0.15)

    def test_defect3_fetal_fraction_result_accepts_ffx_and_ffa_logit_default(
        self,
    ) -> None:
        sca_cls = classify_sca_ray_4d(
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
        self.assertEqual(sca_cls.call, "XO")
        self.assertEqual(sca_cls.active_ff_source, "FFX")
        ff_res = FetalFractionResult(
            fetal_sex="F",
            active_ff=sca_cls.active_ff,
            active_ff_source=sca_cls.active_ff_source,
            ffy=None,
            ffx=sca_cls.active_ff,
            ffa=0.0,
            chrx_ratio=0.94,
            chry_score=0.0,
            yfrac_threshold=0.005,
            sca=SCAResult("XO", True, 0.95, -6.0, None),
            active_ff_se=sca_cls.active_ff_se,
        )
        self.assertEqual(ff_res.active_ff_source, "FFX")

        raw_meta: dict[str, object] = {
            "prenatal_bin_size": 100_000,
            "yfrac_threshold": 0.005,
            "ffy_slope": 1.0,
            "olm_weights": [0.2],
            "olm_bias": -2.0,
            "female_chrx_ratio_mean": 1.0,
            "female_chrx_ratio_std": 0.01,
        }
        parsed = PrenatalModelMetadata.from_dict(raw_meta)
        self.assertTrue(parsed.ffa_logit_mode)

    def test_twin_result_and_fetal_fraction_result_validation(self) -> None:
        twin = TwinResult(
            call="FM",
            fetal_sexes=("F", "M"),
            is_twin=True,
            twin_probability=0.95,
            confidence=0.97,
            posteriors={"FF": 0.001, "FM": 0.969, "MM": 0.030},
            male_ff=0.14,
            female_ff=0.08,
        )
        ff_res = FetalFractionResult(
            fetal_sex="M",
            active_ff=0.22,
            active_ff_source="FF_SIZE",
            ffy=0.14,
            ffx=0.15,
            ffa=0.10,
            chrx_ratio=0.81,
            chry_score=0.00023,
            yfrac_threshold=0.000043,
            sca=SCAResult(
                call="XY", is_aneuploid=False, confidence=0.99, chrx_zscore=-6.5
            ),
            twin=twin,
        )
        self.assertEqual(ff_res.gestation, "twin")
        self.assertEqual(ff_res.fetal_sexes, ("F", "M"))
        self.assertEqual(ff_res.twin, twin)
        self.assertAlmostEqual(ff_res.male_ff or 0.0, 0.14)
        self.assertAlmostEqual(ff_res.female_ff or 0.0, 0.08)
        self.assertAlmostEqual(ff_res.twin_confidence or 0.0, 0.97)
        with self.assertRaisesRegex(ValueError, "conflicts with twin.is_twin"):
            _call(
                FetalFractionResult,
                fetal_sex="M",
                active_ff=0.12,
                active_ff_source="FFY",
                ffy=0.12,
                ffx=0.12,
                ffa=0.12,
                chrx_ratio=0.94,
                chry_score=0.01,
                yfrac_threshold=0.001,
                sca=SCAResult("XY", False, 0.99, -2.0),
                twin=TwinResult(
                    call="MM",
                    fetal_sexes=("M", "M"),
                    is_twin=False,
                    twin_probability=0.05,
                    confidence=0.95,
                    posteriors={"FF": 0.0, "FM": 0.05, "MM": 0.95},
                    male_ff=0.12,
                    female_ff=0.0,
                ),
                gestation="twin",
            )
        with self.assertRaisesRegex(ValueError, "does not match fetal_sexes"):
            _call(
                TwinResult,
                call="FF",
                fetal_sexes=("M", "M"),
                is_twin=True,
                twin_probability=0.9,
                confidence=0.9,
                posteriors={},
                male_ff=0.1,
                female_ff=0.0,
            )

        for pair, sexes in (("FF", ("F", "F")), ("FM", ("F", "M")), ("MM", ("M", "M"))):
            t_pair: TwinSexPair = pair
            g_type: GestationType = "twin"
            tr = TwinResult(
                call=t_pair,
                fetal_sexes=sexes,
                is_twin=True,
                twin_probability=1.0,
                confidence=0.9,
                posteriors={"FF": 0.1, "FM": 0.8, "MM": 0.1},
                male_ff=0.1,
                female_ff=0.1,
            )
            self.assertEqual(tr.call, t_pair)
            self.assertEqual(tr.fetal_sexes, sexes)
            self.assertEqual(g_type, "twin")

        with self.assertRaisesRegex(ValueError, "invalid twin call"):
            _call(
                TwinResult,
                call="MF",
                fetal_sexes=("F", "M"),
                is_twin=True,
                twin_probability=0.9,
                confidence=0.9,
                posteriors={},
                male_ff=0.1,
                female_ff=0.1,
            )
        with self.assertRaisesRegex(ValueError, "invalid twin fetal_sexes"):
            _call(
                TwinResult,
                call="FM",
                fetal_sexes=("M", "F"),
                is_twin=True,
                twin_probability=0.9,
                confidence=0.9,
                posteriors={},
                male_ff=0.1,
                female_ff=0.1,
            )
        with self.assertRaisesRegex(
            ValueError, "gestation must be 'singleton' or 'twin'"
        ):
            _call(
                FetalFractionResult,
                fetal_sex="F",
                active_ff=0.1,
                active_ff_source="FFA",
                ffy=None,
                ffx=0.1,
                ffa=0.1,
                chrx_ratio=1.0,
                chry_score=0.0,
                yfrac_threshold=0.01,
                sca=SCAResult("XX", False, 0.99, 0.0),
                gestation="triplet",
            )
        with self.assertRaisesRegex(ValueError, "fetal_sexes must be"):
            _call(
                FetalFractionResult,
                fetal_sex="M",
                active_ff=0.1,
                active_ff_source="FFA",
                ffy=0.1,
                ffx=0.1,
                ffa=0.1,
                chrx_ratio=0.95,
                chry_score=0.02,
                yfrac_threshold=0.01,
                sca=SCAResult("XY", False, 0.99, -1.0),
                fetal_sexes=("M", "F"),
            )


class TestPredictionOutputsAliasesAndFields(unittest.TestCase):
    def test_prediction_outputs_optional_qc_and_prenatal(self) -> None:
        outputs = PredictionOutputs(
            bins_bed=Path("bins.bed.gz"),
            segments_bed=Path("segments.bed.gz"),
            calls_bed=Path("calls.bed.gz"),
            plot_html=None,
            bins_vcf=None,
            segments_vcf=None,
            calls_vcf=None,
            bins=(),
            segments=(),
        )
        self.assertIsNone(outputs.qc_json_path)
        self.assertIsNone(outputs.prenatal_result)

    def test_prediction_outputs_explicit_qc_and_prenatal(self) -> None:
        sca = SCAResult("XY", False, 0.95, -0.2)
        prenatal = FetalFractionResult(
            fetal_sex="M",
            active_ff=0.10,
            active_ff_source="FFY",
            ffy=0.10,
            ffx=0.09,
            ffa=0.11,
            chrx_ratio=0.96,
            chry_score=0.05,
            yfrac_threshold=0.015,
            sca=sca,
        )
        outputs = PredictionOutputs(
            bins_bed=Path("bins.bed.gz"),
            segments_bed=Path("segments.bed.gz"),
            calls_bed=Path("calls.bed.gz"),
            plot_html=Path("plot.html"),
            bins_vcf=Path("bins.vcf.gz"),
            segments_vcf=Path("segments.vcf.gz"),
            calls_vcf=Path("calls.vcf.gz"),
            bins=(),
            segments=(),
            qc_json_path=Path("sample_qc.json"),
            prenatal_result=prenatal,
        )
        self.assertEqual(outputs.qc_json_path, Path("sample_qc.json"))
        self.assertEqual(outputs.prenatal_result, prenatal)

    def test_branch_qc_status_uses_status_property(self) -> None:
        branch_qc = BranchQCStatus({"verdict": "WARN", "pca_rank": 3})
        self.assertEqual(branch_qc.status, "WARN")
        self.assertFalse(hasattr(branch_qc, "verdict"))
        self.assertEqual(branch_qc["pca_rank"], 3)
        self.assertEqual(branch_qc.get("missing", "fallback"), "fallback")

        ref_qc = ReferenceQCOutputs(
            json_path=Path("ref_qc.json"),
            overall_verdict="PASS",
            worst_severity=0,
            data={"branches": {"autosomal": {"verdict": "PASS"}}},
        )
        self.assertEqual(ref_qc.branches["autosomal"].status, "PASS")


class TestExceptions(unittest.TestCase):
    def test_exception_hierarchy(self) -> None:
        from ccnv.models.exceptions import (
            AlignmentError,
            BinCompatibilityError,
            CcnvError,
            CovariateError,
            ExtractError,
            PredictError,
            ReferenceError,
            ReferenceQCError,
            SchemaError,
            SchemaVersionError,
        )

        for err_cls in (
            AlignmentError,
            BinCompatibilityError,
            CovariateError,
            ExtractError,
            PredictError,
            ReferenceError,
            ReferenceQCError,
            SchemaError,
            SchemaVersionError,
        ):
            self.assertTrue(issubclass(err_cls, CcnvError))
            self.assertTrue(issubclass(err_cls, Exception))
        self.assertTrue(issubclass(SchemaVersionError, SchemaError))
        self.assertTrue(issubclass(BinCompatibilityError, ReferenceError))
        self.assertTrue(issubclass(BinCompatibilityError, PredictError))

    def test_exception_instantiation(self) -> None:
        from ccnv.models.exceptions import CcnvError, ReferenceError

        err = ReferenceError("reference build failed")
        self.assertIsInstance(err, CcnvError)
        self.assertEqual(str(err), "reference build failed")


class TestDomainModels(unittest.TestCase):
    def test_contig_lookup(self) -> None:
        from ccnv.models.domain import ContigLookup

        lookup = ContigLookup.from_names(c for c in ["chr1", "chrX"])
        self.assertEqual(lookup.resolve("1"), "chr1")
        self.assertEqual(lookup.resolve("CHRX"), "chrX")
        self.assertIsNone(lookup.resolve("chrY"))

        plain_lookup = ContigLookup.from_names(["1", "2"])
        self.assertEqual(plain_lookup.resolve("chr1"), "1")
        self.assertIsNone(plain_lookup.resolve("chrZ"))

    def test_cytoband(self) -> None:
        from ccnv.models.domain import Cytoband

        band = Cytoband("chr1", 0, 10000, "p36.33", "gneg")
        self.assertEqual(band.contig, "chr1")
        self.assertEqual(band.start, 0)
        self.assertEqual(band.end, 10000)
        self.assertEqual(band.name, "p36.33")
        self.assertEqual(band.stain, "gneg")

    def test_chromosome_helpers(self) -> None:
        from ccnv.models import domain
        from ccnv.models.domain import (
            is_autosome_name,
            is_canonical_contig,
            normalize_chr,
            select_layout,
        )

        self.assertFalse(hasattr(domain, "resolve_contig_name"))
        self.assertEqual(normalize_chr("chr1"), "1")
        self.assertEqual(normalize_chr("chrX"), "X")
        self.assertEqual(normalize_chr("chrx"), "X")
        self.assertEqual(normalize_chr("chrY"), "Y")
        self.assertEqual(normalize_chr("1"), "1")

        self.assertTrue(is_autosome_name("chr1"))
        self.assertTrue(is_autosome_name("22"))
        self.assertFalse(is_autosome_name("chrX"))
        self.assertFalse(is_autosome_name("chrM"))

        self.assertTrue(is_canonical_contig("chr1"))
        self.assertTrue(is_canonical_contig("chrX"))
        self.assertTrue(is_canonical_contig("chrY"))
        self.assertFalse(is_canonical_contig("chrM"))

        layout = select_layout([("chr1", 1000), ("chrM", 500)], all_contigs=False)
        self.assertEqual(layout, [("chr1", 1000)])

    def test_contig_lookup_handles_single_pass_generator_and_config_preserves_prefix(
        self,
    ) -> None:
        from ccnv.models.domain import ContigLookup

        lookup = ContigLookup.from_names(c for c in ["chr1", "chr2"])
        self.assertEqual(lookup.resolve("CHR1"), "chr1")
        cfg = ReferenceConfig(
            samplesheet_path=Path("sheet.yaml"), prefix=Path("out/ref")
        )
        self.assertEqual(cfg.prefix, Path("out/ref"))

        p_cfg = PredictConfig(
            sample=Path("s.bam"),
            reference=Path("ref.ccnv"),
            prefix=Path("out/pred"),
        )
        self.assertEqual(p_cfg.threads, 1)
        self.assertFalse(p_cfg.bins_vcf)
        self.assertFalse(hasattr(p_cfg, "bin_size"))

        with self.assertRaises(ValueError):
            ReferenceConfig(prefix=Path("out/ref"), bin_size=0)
        with self.assertRaises(ValueError):
            ReferenceConfig(prefix=Path("out/ref"), min_mappability=1.2)
        with self.assertRaises(ValueError):
            PredictConfig(
                sample=Path("s.bam"),
                reference=Path("r.ccnv"),
                prefix=Path("o"),
                min_mappability=-0.1,
            )
        with self.assertRaises(ValueError):
            PredictConfig(
                sample=Path("s.bam"),
                reference=Path("r.ccnv"),
                prefix=Path("o"),
                threads=0,
            )


class TestModelReExports(unittest.TestCase):
    def test_reexports_from_root_models(self) -> None:
        from ccnv import models

        expected_symbols = [
            "ExtractConfig",
            "ReferenceConfig",
            "PredictConfig",
            "ReferenceQCConfig",
            "ReadFilters",
            "DEFAULT_BIN_SIZE",
            "DEFAULT_MIN_MAPPABILITY",
            "AnalysisMode",
            "parquet_path",
            "ContigLookup",
            "SamplesheetEntry",
            "Cytoband",
            "select_layout",
            "is_canonical_contig",
            "is_autosome_name",
            "normalize_chr",
            "BinResult",
            "BranchQCStatus",
            "SegmentResult",
            "SCAResult",
            "ScanAlignmentResult",
            "ScanQcCounts",
            "FetalFractionResult",
            "PredictionOutputs",
            "ReferenceQCOutputs",
            "CallType",
            "SCACall",
            "FetalSex",
            "ActiveFFSource",
            "GestationType",
            "TwinSexPair",
            "TwinResult",
            "SAMPLE_SCHEMA",
            "REFERENCE_SCHEMA",
            "SCHEMA_VERSION",
            "NUM_FRAG_BANDS",
            "NUM_GC_QUANTILES",
            "NUM_PROTOTYPES_PER_BIN",
            "BRANCH_AUTOSOMAL",
            "BRANCH_FEMALE_SEX",
            "BRANCH_MALE_SEX",
            "BRANCH_PRENATAL_AUTOSOMAL",
            "BRANCH_PRENATAL_FFY",
            "KNOWN_REFERENCE_BRANCHES",
            "ArtifactKind",
            "SampleBuffers",
            "BranchFitArrays",
            "SampleMetadata",
            "ReferenceMetadata",
            "PrenatalModelMetadata",
            "PRENATAL_METADATA_KEY",
            "COORDINATE_SYSTEM",
            "build_reference_schema",
            "reconstruct_sample_grid",
            "decode_metadata",
            "encode_metadata",
            "validate_metadata",
            "validate_schema",
            "validate_sample_table",
            "validate_reference_table",
            "CcnvError",
            "AlignmentError",
            "BinCompatibilityError",
            "ReferenceError",
            "CovariateError",
            "PredictError",
            "SchemaError",
            "SchemaVersionError",
            "ExtractError",
            "ReferenceQCError",
        ]
        for sym in expected_symbols:
            self.assertTrue(
                hasattr(models, sym), f"ccnv.models missing re-export {sym}"
            )

        removed_symbols = (
            "CNV_REFERENCE_BRANCHES",
            "metadata_for",
            "resolve_contig_name",
            "validate_table",
        )
        for sym in removed_symbols:
            self.assertFalse(
                hasattr(models, sym), f"ccnv.models still exports removed symbol {sym}"
            )


if __name__ == "__main__":
    unittest.main()
