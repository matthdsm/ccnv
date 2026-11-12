"""Unit tests for SampleMetadata, ReferenceMetadata, PrenatalModelMetadata, and FragmentomicsMetadata."""

from __future__ import annotations

import json
import unittest
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from helpers import make_prenatal_metadata, make_reference_table, metadata_for

from ccnv.models import (
    COORDINATE_SYSTEM,
    FRAGMENTOMICS_METADATA_KEY,
    PRENATAL_METADATA_KEY,
    SAMPLE_SCHEMA,
    SCHEMA_VERSION,
    ArtifactKind,
    ExtractConfig,
    FetalFractionResult,
    FragmentomicsMetadata,
    PrenatalModelMetadata,
    ReadFilters,
    ReferenceMetadata,
    SampleMetadata,
    SCAResult,
    SchemaError,
    SchemaVersionError,
    SegmentResult,
    decode_metadata,
    encode_metadata,
    validate_metadata,
    validate_reference_table,
    validate_schema,
)


def _call[T](fn: Callable[..., T], *args: object, **kwargs: object) -> T:
    return fn(*args, **kwargs)


class TestMetadataContracts(unittest.TestCase):
    def test_fragmentomics_metadata_dataclass_and_serialization(self) -> None:
        meta = FragmentomicsMetadata(
            paired_end=True,
            total_paired_fragments=123_456,
            chromatosome_mode_bp=166.2,
            core_shoulder_bp=143.1,
            sizing_shift_bp=0.2,
            self_referenced_peak_ratio=0.35,
            dinucleosome_fraction=0.08,
            high_mw_fraction=0.012,
            helical_10bp_amplitude=0.045,
            helical_10bp_phase=1.23,
            motif_diversity_score=0.92,
            hist_autosomal=[1] * 571,
            hist_chry=[0] * 571,
            hist_chrx=[2] * 571,
            end_motif_counts=[10] * 256,
        )
        self.assertTrue(meta.paired_end)
        self.assertEqual(len(meta.hist_autosomal), 571)
        self.assertEqual(len(meta.end_motif_counts), 256)
        encoded = json.dumps(asdict(meta))
        decoded = json.loads(encoded)
        restored = FragmentomicsMetadata.from_dict(decoded)
        self.assertEqual(meta, restored)
        self.assertEqual(FRAGMENTOMICS_METADATA_KEY, b"ccnv.fragmentomics")

    def test_prenatal_model_metadata_with_fragmentomics_fields(self) -> None:
        p_fetal = [0.001] * 571
        p_maternal = [0.002] * 571
        weights = [0.1] * 36
        meta = PrenatalModelMetadata(
            prenatal_bin_size=100_000,
            yfrac_threshold=0.0041,
            ffy_slope=0.985,
            olm_weights=[0.02 for _ in range(50)],
            olm_bias=0.005,
            female_chrx_ratio_mean=1.0003,
            female_chrx_ratio_std=0.0112,
            alpha_1=-0.498,
            p_fetal_density=p_fetal,
            p_maternal_density=p_maternal,
            ff_frag_weights=weights,
            ff_frag_bias=-1.25,
            ff_frag_feature_mean=[0.0] * 36,
            ff_frag_feature_scale=[1.0] * 36,
            ff_size_std=0.018,
        )
        self.assertEqual(meta.p_fetal_density, p_fetal)
        self.assertEqual(meta.ff_size_std, 0.018)
        self.assertEqual(meta.ff_frag_bias, -1.25)
        self.assertEqual(meta.alpha_1, -0.498)
        for removed in (
            "ffy_intercept",
            "ffx_intercept",
            "ffx_slope",
            "ffx_std",
            "alpha_0",
            "sigma_x_sys",
        ):
            self.assertFalse(hasattr(meta, removed))

    def test_results_and_config_fragmentomics_fields(self) -> None:
        sca_res = SCAResult(
            call="XY",
            is_aneuploid=False,
            confidence=0.99,
            chrx_zscore=-0.1,
        )
        ff_res = FetalFractionResult(
            fetal_sex="M",
            active_ff=0.12,
            active_ff_source="FFY",
            ffy=0.12,
            ffx=0.11,
            ffa=0.10,
            chrx_ratio=0.95,
            chry_score=0.05,
            yfrac_threshold=0.004,
            sca=sca_res,
            ff_size=0.115,
        )
        self.assertEqual(ff_res.ff_size, 0.115)

        seg_res = SegmentResult(
            contig="chr1",
            start=0,
            end=100_000,
            ratio=1.1,
            zscore=3.2,
            weight=1.0,
            bin_count=20,
            call="gain",
            frag_short_ratio=0.42,
            frag_llr_size=2.85,
            origin_call="FETAL_CONFIRMED",
        )
        self.assertEqual(seg_res.frag_short_ratio, 0.42)
        self.assertEqual(seg_res.frag_llr_size, 2.85)
        self.assertEqual(seg_res.origin_call, "FETAL_CONFIRMED")

        cfg = ExtractConfig(
            alignment=Path("sample.bam"),
            prefix=Path("out"),
        )
        self.assertFalse(hasattr(cfg, "min_frag_len"))
        self.assertFalse(hasattr(cfg, "max_frag_len"))
        with self.assertRaisesRegex(ValueError, "bin_size must be a positive integer"):
            ExtractConfig(
                alignment=Path("sample.bam"),
                prefix=Path("out"),
                bin_size=0,
            )

    def test_unsupported_schema_version_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaVersionError, "unsupported schema version"):
            metadata_for(
                ArtifactKind.REFERENCE,
                contig_lengths={"chr1": 200},
                values={
                    "schema_version": "9.9.9",
                    "sample_identities": [],
                    "model": {"branches": ["autosomal"]},
                },
            )

        with self.assertRaisesRegex(
            SchemaVersionError, "unsupported schema version '1.0.0'"
        ):
            metadata_for(
                ArtifactKind.REFERENCE,
                contig_lengths={"chr1": 200},
                values={
                    "schema_version": "1.0.0",
                    "sample_identities": [],
                    "model": {"branches": ["autosomal"]},
                },
            )

    def test_prenatal_model_metadata_serialization_round_trip(self) -> None:
        meta = make_prenatal_metadata()
        as_dict = meta.to_dict()
        self.assertEqual(as_dict["prenatal_bin_size"], 100_000)
        self.assertEqual(as_dict["alpha_1"], -0.5)
        self.assertNotIn("mlp_w1", as_dict)
        self.assertNotIn("preface_std", as_dict)

        encoded = meta.encode()
        decoded = PrenatalModelMetadata.decode(encoded)
        self.assertEqual(decoded, meta)

        from_dict_meta = PrenatalModelMetadata.from_dict(as_dict)
        self.assertEqual(from_dict_meta, meta)

    def test_prenatal_model_metadata_roundtrip_all_fragmentomic_fields(self) -> None:
        meta = make_prenatal_metadata(
            alpha_1=-0.498,
            p_fetal_density=[0.01] * 571,
            p_maternal_density=[0.02] * 571,
            ff_frag_weights=[0.5] * 36,
            ff_frag_bias=-1.25,
            ff_frag_feature_mean=[0.1] * 36,
            ff_frag_feature_scale=[1.2] * 36,
            ff_size_std=0.035,
        )
        as_dict = meta.to_dict()
        for key in (
            "alpha_1",
            "p_fetal_density",
            "p_maternal_density",
            "ff_frag_weights",
            "ff_frag_bias",
            "ff_frag_feature_mean",
            "ff_frag_feature_scale",
            "ff_size_std",
        ):
            self.assertIn(key, as_dict)

        restored = PrenatalModelMetadata.from_dict(as_dict)
        self.assertEqual(restored, meta)
        self.assertEqual(restored.ff_size_std, 0.035)

        encoded = meta.encode()
        decoded = PrenatalModelMetadata.decode(encoded)
        self.assertEqual(decoded, meta)
        self.assertEqual(decoded.ff_size_std, 0.035)

    def test_prenatal_model_metadata_defaults_for_optional_keys(self) -> None:
        meta = make_prenatal_metadata()
        as_dict = meta.to_dict()
        for k in ("alpha_1", "ffy_std", "ffa_std", "ffa_logit_mode"):
            as_dict.pop(k, None)

        loaded = PrenatalModelMetadata.from_dict(as_dict)
        self.assertEqual(loaded.alpha_1, -0.5)
        self.assertEqual(loaded.ffy_std, 0.01)
        self.assertEqual(loaded.ffa_std, 0.02)
        self.assertTrue(loaded.ffa_logit_mode)

    def test_prenatal_model_metadata_rejects_missing_keys_or_empty_dict(self) -> None:
        with self.assertRaises(KeyError):
            PrenatalModelMetadata.from_dict({})

        with self.assertRaises(SchemaError):
            metadata_for(
                ArtifactKind.REFERENCE,
                contig_lengths={"chr1": 200},
                values={
                    "schema_version": "2.0.0",
                    "sample_identities": [],
                    "model": {"branches": ["autosomal"]},
                    "prenatal": {},
                },
            )

    def test_prenatal_model_metadata_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            PrenatalModelMetadata(
                prenatal_bin_size=0,
                yfrac_threshold=0.0,
                ffy_slope=1.0,
                olm_weights=[],
                olm_bias=0.0,
                female_chrx_ratio_mean=1.0,
                female_chrx_ratio_std=0.0,
            )
        with self.assertRaisesRegex(ValueError, "alpha_1 must be negative"):
            PrenatalModelMetadata(
                prenatal_bin_size=100_000,
                yfrac_threshold=0.0,
                ffy_slope=1.0,
                olm_weights=[],
                olm_bias=0.0,
                female_chrx_ratio_mean=1.0,
                female_chrx_ratio_std=0.01,
                alpha_1=0.0,
            )

    def test_reference_metadata_round_trip(self) -> None:
        prenatal_meta = make_prenatal_metadata()
        ref_meta = ReferenceMetadata(
            bin_size=5000,
            contig_lengths={"chr1": 1_000_000, "chrY": 500_000},
            branches=("autosomal", "prenatal_autosomal", "prenatal_ffy"),
            mode="prenatal",
            prenatal_bin_size=100_000,
            prenatal=prenatal_meta,
        )
        self.assertEqual(ref_meta.mode, "prenatal")
        self.assertEqual(ref_meta.prenatal_bin_size, 100_000)

        metadata_dict = ref_meta.to_metadata()
        model_dict = metadata_dict["model"]
        if not isinstance(model_dict, dict):
            self.fail("expected dict for metadata_dict['model']")
        self.assertEqual(model_dict["mode"], "prenatal")
        self.assertEqual(model_dict["prenatal_bin_size"], 100_000)
        self.assertIn("prenatal", metadata_dict)

        arrow_meta = ref_meta.build_arrow_metadata()
        self.assertIn(PRENATAL_METADATA_KEY, arrow_meta)

        reconstructed = ReferenceMetadata.from_dict(
            decode_metadata(arrow_meta)
            | {"contig_lengths": {"chr1": 1_000_000, "chrY": 500_000}}
        )
        self.assertEqual(reconstructed.mode, "prenatal")
        self.assertEqual(reconstructed.bin_size, 5000)
        self.assertEqual(reconstructed.prenatal_bin_size, 100_000)
        if reconstructed.prenatal is None:
            self.fail("expected non-None reconstructed.prenatal")
        self.assertEqual(
            reconstructed.prenatal.yfrac_threshold, prenatal_meta.yfrac_threshold
        )

    def test_reference_metadata_validation(self) -> None:
        with self.assertRaisesRegex(SchemaError, "bin_size must be a positive integer"):
            ReferenceMetadata(
                bin_size=-1,
                contig_lengths={"chr1": 100},
                branches=("autosomal",),
            )
        with self.assertRaisesRegex(
            SchemaError, "mode must be 'germline' or 'prenatal'"
        ):
            _call(
                ReferenceMetadata,
                bin_size=5000,
                contig_lengths={"chr1": 100},
                branches=("autosomal",),
                mode="invalid_mode",
            )
        with self.assertRaisesRegex(
            SchemaError, "contig_lengths must be a non-empty object"
        ):
            ReferenceMetadata(
                bin_size=5000,
                contig_lengths={},
                branches=("autosomal",),
            )

    def test_sample_metadata_validation_and_error_boundary(self) -> None:
        with self.assertRaises(SchemaError):
            SampleMetadata(
                bin_size=0,
                contig_lengths={"chr1": 100},
                contig_order=("chr1",),
                source_identity={},
                reference_identity={},
                qc={},
                filters={},
            )
        with self.assertRaises(SchemaError):
            SampleMetadata.from_dict({"bin_size": 5000, "contig_lengths": "invalid"})
        with self.assertRaises(SchemaError):
            ReferenceMetadata.from_dict(
                {"contig_lengths": {"chr1": 100}, "model": {"pca_components": "bad"}}
            )

    def test_metadata_schema_error_boundary_and_sample_metadata_post_init(
        self,
    ) -> None:
        with self.assertRaises(SchemaError):
            SampleMetadata(
                bin_size=0,
                contig_lengths={"chr1": 10_000},
                contig_order=("chr1",),
                source_identity={},
                reference_identity={},
                qc={},
                filters={},
            )
        with self.assertRaises(SchemaError):
            SampleMetadata(
                bin_size=True,
                contig_lengths={"chr1": 10_000},
                contig_order=("chr1",),
                source_identity={},
                reference_identity={},
                qc={},
                filters={},
            )
        with self.assertRaises(SchemaError):
            SampleMetadata(
                bin_size=5_000,
                contig_lengths={"chr1": 0},
                contig_order=("chr1",),
                source_identity={},
                reference_identity={},
                qc={},
                filters={},
            )
        with self.assertRaises(SchemaError):
            SampleMetadata(
                bin_size=5_000,
                contig_lengths={"chr1": 10_000},
                contig_order=("chr2",),
                source_identity={},
                reference_identity={},
                qc={},
                filters={},
            )

        base_s_meta = decode_metadata(
            metadata_for(
                ArtifactKind.SAMPLE,
                contig_lengths={"chr1": 5_000},
                values={
                    "bin_size": 5_000,
                    "source_identity": {},
                    "reference_identity": {},
                    "qc": {},
                    "filters": asdict(ReadFilters()),
                },
            )
        )
        for bad_frag in (123, ["not", "mapping"], {"paired_end": True}):
            bad_dict = {**base_s_meta, "fragmentomics": bad_frag}
            with self.assertRaises(SchemaError):
                validate_metadata(bad_dict, ArtifactKind.SAMPLE)
            with self.assertRaises(SchemaError):
                validate_schema(
                    SAMPLE_SCHEMA.with_metadata(encode_metadata(bad_dict)),
                    ArtifactKind.SAMPLE,
                )
            with self.assertRaises(SchemaError):
                SampleMetadata.from_dict(bad_dict)

        with self.assertRaises(SchemaError):
            SampleMetadata.from_dict({})
        with self.assertRaises(SchemaError):
            SampleMetadata.from_dict({**base_s_meta, "bin_size": "five_kb"})

        base_ref = make_reference_table(
            branches=["autosomal", "autosomal"],
            contigs=["chr1", "chr1"],
            starts=[0, 5_000],
            ends=[5_000, 10_000],
            contig_lengths={"chr1": 10_000},
            pca_loadings=[[0.0], [0.0]],
            model={
                "bin_size": 5_000,
                "branches": ["autosomal"],
                "pca_components": 1,
            },
            sample_identities=["s1", "s2"],
        )
        ref_dict = decode_metadata(base_ref.schema.metadata)
        model_obj = ref_dict["model"]
        if not isinstance(model_obj, dict):
            self.fail("expected dict for ref_dict['model']")
        for bad_field, bad_val in (
            ("pca_components", "five"),
            ("min_mappability", "high"),
            ("max_batch_memory_mb", "many"),
        ):
            corrupt_ref_dict = {
                **ref_dict,
                "model": {**model_obj, bad_field: bad_val},
            }
            corrupt_table = base_ref.replace_schema_metadata(
                encode_metadata(corrupt_ref_dict)
            )
            with self.assertRaises(SchemaError):
                ReferenceMetadata.from_dict(corrupt_ref_dict)
            with self.assertRaises(SchemaError):
                ReferenceMetadata.from_table(corrupt_table)
            with self.assertRaises(SchemaError):
                validate_reference_table(corrupt_table)

    def test_metadata_post_init_contracts(self) -> None:
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

    def test_rfc8259_metadata_rejects_non_finite_floats(self) -> None:
        with self.assertRaises(SchemaError):
            encode_metadata({"bad": {"x": float("nan")}})
        with self.assertRaises(SchemaError):
            decode_metadata({b"ccnv.bad": b'{"x":NaN}'})
        with self.assertRaises(SchemaError):
            PrenatalModelMetadata.decode(b'{"prenatal_bin_size":100000,"x":Infinity}')


if __name__ == "__main__":
    unittest.main()
