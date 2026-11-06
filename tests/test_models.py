"""Unit tests for domain models, configurations, and result types."""

import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from ccnv.models import (
    PredictConfig,
    ReferenceConfig,
    SamplesheetEntry,
)
from ccnv.models.results import (
    FetalFractionResult,
    PredictionOutputs,
    SCACall,
    SCAResult,
    SegmentResult,
)


class TestSamplesheetEntry(unittest.TestCase):
    def test_samplesheet_entry_defaults_and_types(self) -> None:
        entry = SamplesheetEntry(filepath=Path("sample1.ccnv"))
        self.assertEqual(entry.filepath, Path("sample1.ccnv"))
        self.assertIsNone(entry.sex)
        self.assertIsNone(entry.ff)

    def test_samplesheet_entry_string_path_conversion(self) -> None:
        entry = SamplesheetEntry(filepath="sample1.ccnv")  # type: ignore[arg-type]
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
            SamplesheetEntry(filepath=Path("sample.ccnv"), sex="X")  # type: ignore[arg-type]

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
            entry.sex = "M"  # type: ignore[misc]


class TestReferenceConfig(unittest.TestCase):
    def test_reference_config_defaults(self) -> None:
        config = ReferenceConfig(prefix=Path("ref_out"))
        self.assertEqual(config.prefix, Path("ref_out"))
        self.assertEqual(config.mode, "germline")
        self.assertEqual(config.prenatal_bin_size, 100000)
        self.assertIsNone(config.samplesheet_path)
        self.assertEqual(config.entries, ())
        self.assertEqual(config.samples, ())
        self.assertEqual(config.input_paths, ())
        self.assertEqual(config.threads, 1)

    def test_reference_config_positional_prefix(self) -> None:
        config = ReferenceConfig(Path("ref_out"))
        self.assertEqual(config.prefix, Path("ref_out"))
        self.assertEqual(config.samples, ())

    def test_reference_config_positional_samples_and_prefix(self) -> None:
        p1 = Path("s1.ccnv")
        p2 = Path("s2.ccnv")
        config = ReferenceConfig((p1, p2), Path("ref_out"))
        self.assertEqual(config.samples, (p1, p2))
        self.assertEqual(config.prefix, Path("ref_out"))
        self.assertEqual(config.input_paths, (p1, p2))

    def test_reference_config_samples_synchronization(self) -> None:
        p1 = Path("s1.ccnv")
        p2 = Path("s2.ccnv")
        config = ReferenceConfig(prefix=Path("ref"), samples=(p1, p2))
        self.assertEqual(config.samples, (p1, p2))
        self.assertEqual(config.input_paths, (p1, p2))
        self.assertEqual(
            config.entries,
            (SamplesheetEntry(filepath=p1), SamplesheetEntry(filepath=p2)),
        )

    def test_reference_config_entries_synchronization(self) -> None:
        e1 = SamplesheetEntry(filepath=Path("s1.ccnv"), sex="M", ff=0.10)
        e2 = SamplesheetEntry(filepath=Path("s2.ccnv"), sex="F", ff=None)
        config = ReferenceConfig(prefix=Path("ref"), entries=(e1, e2))
        self.assertEqual(config.entries, (e1, e2))
        self.assertEqual(config.samples, (Path("s1.ccnv"), Path("s2.ccnv")))
        self.assertEqual(config.input_paths, (Path("s1.ccnv"), Path("s2.ccnv")))

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
        with self.assertRaisesRegex(
            ValueError, "prenatal_bin_size must be a positive integer"
        ):
            ReferenceConfig(prefix=Path("ref"), prenatal_bin_size=0)
        with self.assertRaisesRegex(
            ValueError, "mode must be 'germline' or 'prenatal'"
        ):
            ReferenceConfig(prefix=Path("ref"), mode="invalid")  # type: ignore[arg-type]

    def test_reference_config_immutability(self) -> None:
        config = ReferenceConfig(prefix=Path("ref"))
        with self.assertRaises(FrozenInstanceError):
            config.mode = "prenatal"  # type: ignore[misc]


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

    def test_predict_config_prenatal_mode(self) -> None:
        config = PredictConfig(
            sample=Path("sample.ccnv"),
            reference=Path("ref.ccnv"),
            prefix=Path("out"),
            mode="prenatal",
            min_fetal_fraction_warning=0.05,
        )
        self.assertEqual(config.mode, "prenatal")
        self.assertAlmostEqual(config.min_fetal_fraction_warning, 0.05)

    def test_predict_config_validation(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "mode must be 'germline' or 'prenatal'"
        ):
            PredictConfig(
                sample=Path("s"),
                reference=Path("r"),
                prefix=Path("o"),
                mode="other",  # type: ignore[arg-type]
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

    def test_predict_config_immutability(self) -> None:
        config = PredictConfig(
            sample=Path("s"),
            reference=Path("r"),
            prefix=Path("o"),
        )
        with self.assertRaises(FrozenInstanceError):
            config.mode = "prenatal"  # type: ignore[misc]


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
            seg.discordant = True  # type: ignore[misc]


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
            SCAResult(
                call="YY",  # type: ignore[arg-type]
                is_aneuploid=True,
                confidence=0.5,
                chrx_zscore=0.0,
            )

    def test_sca_result_immutability(self) -> None:
        sca = SCAResult("XX", False, 0.99, 0.1)
        with self.assertRaises(FrozenInstanceError):
            sca.confidence = 1.0  # type: ignore[misc]


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
            preface=0.112,
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
        self.assertEqual(ff_res.preface, 0.112)
        self.assertEqual(ff_res.sca, sca)
        self.assertFalse(ff_res.low_ff_warning)
        self.assertEqual(ff_res.warnings, ())

    def test_fetal_fraction_result_warnings_tuple_conversion(self) -> None:
        sca = SCAResult("XX", False, 0.99, 0.0)
        ff_res = FetalFractionResult(
            fetal_sex="F",
            active_ff=0.03,
            active_ff_source="PREFACE",
            ffy=None,
            ffx=None,
            preface=0.03,
            chrx_ratio=1.0,
            chry_score=0.001,
            yfrac_threshold=0.015,
            sca=sca,
            low_ff_warning=True,
            warnings=["fetal fraction below threshold", "female fetus"],  # type: ignore[arg-type]
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
            FetalFractionResult(
                fetal_sex="U",  # type: ignore[arg-type]
                active_ff=0.1,
                active_ff_source="PREFACE",
                ffy=None,
                ffx=None,
                preface=0.1,
                chrx_ratio=1.0,
                chry_score=0.0,
                yfrac_threshold=0.01,
                sca=sca,
            )
        with self.assertRaisesRegex(
            ValueError, "active_ff_source must be 'FFY' or 'PREFACE'"
        ):
            FetalFractionResult(
                fetal_sex="F",
                active_ff=0.1,
                active_ff_source="OTHER",  # type: ignore[arg-type]
                ffy=None,
                ffx=None,
                preface=0.1,
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
            active_ff_source="PREFACE",
            ffy=None,
            ffx=0.08,
            preface=0.08,
            chrx_ratio=1.0,
            chry_score=0.0,
            yfrac_threshold=0.01,
            sca=sca,
        )
        with self.assertRaises(FrozenInstanceError):
            ff_res.active_ff = 0.15  # type: ignore[misc]


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
            preface=0.11,
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


class TestExceptions(unittest.TestCase):
    def test_exception_hierarchy(self) -> None:
        from ccnv.models.exceptions import (
            CcnvError,
            ConvertError,
            CovariateError,
            PredictError,
            ReferenceError,
            ReferenceQCError,
            SchemaError,
        )

        for err_cls in (
            ConvertError,
            CovariateError,
            PredictError,
            ReferenceError,
            ReferenceQCError,
            SchemaError,
        ):
            self.assertTrue(issubclass(err_cls, CcnvError))
            self.assertTrue(issubclass(err_cls, Exception))

    def test_exception_instantiation(self) -> None:
        from ccnv.models.exceptions import CcnvError, ReferenceError

        err = ReferenceError("reference build failed")
        self.assertIsInstance(err, CcnvError)
        self.assertEqual(str(err), "reference build failed")


class TestDomainModels(unittest.TestCase):
    def test_contig_validation(self) -> None:
        from ccnv.models.domain import Contig

        c = Contig("chr1", 1000)
        self.assertEqual(c.name, "chr1")
        self.assertEqual(c.length, 1000)
        with self.assertRaisesRegex(ValueError, "contig name must not be empty"):
            Contig("", 1000)
        with self.assertRaisesRegex(ValueError, "contig length must be positive"):
            Contig("chr1", 0)

    def test_genome_profile(self) -> None:
        from ccnv.models.domain import Contig, GenomeProfile

        c1 = Contig("chr1", 1000)
        c2 = Contig("chr2", 2000)
        gp = GenomeProfile((c1, c2))
        self.assertEqual(gp.names, ("chr1", "chr2"))
        self.assertEqual(gp.lengths, (1000, 2000))
        self.assertEqual(gp.length_map(), {"chr1": 1000, "chr2": 2000})

        with self.assertRaisesRegex(
            ValueError, "genome profile must contain at least one contig"
        ):
            GenomeProfile(())
        with self.assertRaisesRegex(ValueError, "contig names must be unique"):
            GenomeProfile((c1, Contig("chr1", 500)))

    def test_cytoband(self) -> None:
        from ccnv.models.domain import Cytoband

        band = Cytoband("chr1", 0, 10000, "p36.33", "gneg")
        self.assertEqual(band.contig, "chr1")
        self.assertEqual(band.start, 0)
        self.assertEqual(band.end, 10000)
        self.assertEqual(band.name, "p36.33")
        self.assertEqual(band.stain, "gneg")

    def test_chromosome_helpers(self) -> None:
        from ccnv.models.domain import (
            is_autosome_name,
            is_canonical_contig,
            normalize_chr,
            resolve_contig_name,
            select_layout,
        )

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

        self.assertEqual(resolve_contig_name("1", ["chr1", "chr2"]), "chr1")
        self.assertEqual(resolve_contig_name("chr1", ["1", "2"]), "1")
        self.assertIsNone(resolve_contig_name("chrZ", ["chr1", "chr2"]))

        layout = select_layout(["chr1", "chrM"], [1000, 500], all_contigs=False)
        self.assertEqual(len(layout), 1)
        self.assertEqual(layout[0].name, "chr1")


class TestModelReExports(unittest.TestCase):
    def test_reexports_from_root_models(self) -> None:
        import ccnv.models as models

        expected_symbols = [
            "ConvertConfig",
            "ReferenceConfig",
            "PredictConfig",
            "ReferenceQCConfig",
            "ReadFilters",
            "DEFAULT_BIN_SIZE",
            "DEFAULT_MIN_MAPPABILITY",
            "AnalysisMode",
            "parquet_path",
            "Contig",
            "GenomeProfile",
            "SamplesheetEntry",
            "Cytoband",
            "select_layout",
            "is_canonical_contig",
            "is_autosome_name",
            "resolve_contig_name",
            "normalize_chr",
            "BinResult",
            "SegmentResult",
            "SCAResult",
            "FetalFractionResult",
            "PredictionOutputs",
            "ReferenceQCOutputs",
            "CallType",
            "SCACall",
            "FetalSex",
            "ActiveFFSource",
            "SAMPLE_SCHEMA",
            "REFERENCE_SCHEMA",
            "SCHEMA_VERSION",
            "BRANCH_AUTOSOMAL",
            "BRANCH_FEMALE_SEX",
            "BRANCH_MALE_SEX",
            "BRANCH_PRENATAL_AUTOSOMAL",
            "BRANCH_PRENATAL_FFY",
            "KNOWN_REFERENCE_BRANCHES",
            "CNV_REFERENCE_BRANCHES",
            "ArtifactKind",
            "ReferenceMetadata",
            "PrenatalModelMetadata",
            "PRENATAL_METADATA_KEY",
            "COORDINATE_SYSTEM",
            "decode_metadata",
            "metadata_for",
            "encode_metadata",
            "validate_metadata",
            "validate_table",
            "validate_schema",
            "validate_sample_table",
            "validate_reference_table",
            "bin_coordinates",
            "CcnvError",
            "ReferenceError",
            "CovariateError",
            "PredictError",
            "SchemaError",
            "ConvertError",
            "ReferenceQCError",
        ]
        for sym in expected_symbols:
            self.assertTrue(
                hasattr(models, sym), f"ccnv.models missing re-export {sym}"
            )


if __name__ == "__main__":
    unittest.main()
