"""Unit tests for Schema v2.0.0 Arrow contracts, metadata containers, and validation."""

from __future__ import annotations

import json
import unittest
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np
import pyarrow as pa

from ccnv.models import AnalysisMode, ReadFilters
from ccnv.models.config import ExtractConfig
from ccnv.models.results import FetalFractionResult, SCAResult, SegmentResult
from ccnv.models.schemas import (
    BRANCH_AUTOSOMAL,
    BRANCH_FEMALE_SEX,
    BRANCH_MALE_SEX,
    BRANCH_PRENATAL_AUTOSOMAL,
    BRANCH_PRENATAL_FFY,
    COORDINATE_SYSTEM,
    FRAGMENTOMICS_METADATA_KEY,
    KNOWN_REFERENCE_BRANCHES,
    NUM_FRAG_BANDS,
    NUM_GC_QUANTILES,
    NUM_PROTOTYPES_PER_BIN,
    PRENATAL_METADATA_KEY,
    SAMPLE_SCHEMA,
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    ArtifactKind,
    FragmentomicsMetadata,
    PrenatalModelMetadata,
    ReferenceMetadata,
    SchemaError,
    SchemaVersionError,
    bin_coordinates,
    build_reference_schema,
    decode_metadata,
    metadata_for,
    reconstruct_sample_grid,
    validate_reference_table,
    validate_sample_table,
    validate_schema,
    validate_table,
)


def _make_prenatal_metadata() -> PrenatalModelMetadata:
    return PrenatalModelMetadata(
        prenatal_bin_size=100_000,
        yfrac_threshold=0.0041,
        ffy_intercept=0.0012,
        ffy_slope=0.985,
        olm_weights=[0.02 for _ in range(50)],
        olm_bias=0.005,
        ffx_intercept=1.0004,
        ffx_slope=-0.498,
        female_chrx_ratio_mean=1.0003,
        female_chrx_ratio_std=0.0112,
    )


def _make_ref_table(
    branches: list[str],
    contigs: list[str],
    starts: list[int],
    ends: list[int],
    metadata: dict[bytes, bytes],
    *,
    pca_rank: int = 1,
) -> pa.Table:
    n_rows = len(branches)
    branch_vocab = list(dict.fromkeys(branches))
    branch_idx = np.array([branch_vocab.index(b) for b in branches], dtype=np.int8)
    contig_vocab = list(dict.fromkeys(contigs))
    contig_idx = np.array([contig_vocab.index(c) for c in contigs], dtype=np.int16)
    loadings_col: pa.Array = (
        pa.nulls(n_rows, type=pa.list_(pa.float32(), 0))
        if pca_rank == 0
        else pa.FixedSizeListArray.from_arrays(
            np.full(n_rows * pca_rank, 0.1, dtype=np.float32), pca_rank
        )
    )
    return pa.table(
        {
            "branch": pa.DictionaryArray.from_arrays(branch_idx, branch_vocab),
            "contig": pa.DictionaryArray.from_arrays(contig_idx, contig_vocab),
            "start": pa.array(starts, type=pa.uint32()),
            "end": pa.array(ends, type=pa.uint32()),
            "masked": pa.array([False] * n_rows, type=pa.bool_()),
            "gc": pa.array([0.42] * n_rows, type=pa.float32()),
            "mappability": pa.array([0.99] * n_rows, type=pa.float32()),
            "gc_quantiles": pa.nulls(
                n_rows, type=pa.list_(pa.float32(), NUM_GC_QUANTILES)
            ),
            "ref_frag_len": pa.array([166.0] * n_rows, type=pa.float32()),
            "ref_ep_rate": pa.array([2.0] * n_rows, type=pa.float32()),
            "pca_mean": pa.array([0.0] * n_rows, type=pa.float32()),
            "pca_loadings": loadings_col,
            "proto_ids": pa.FixedSizeListArray.from_arrays(
                np.zeros(n_rows * NUM_PROTOTYPES_PER_BIN, dtype=np.uint16),
                NUM_PROTOTYPES_PER_BIN,
            ),
            "proto_weights": pa.FixedSizeListArray.from_arrays(
                np.full(n_rows * NUM_PROTOTYPES_PER_BIN, 0.25, dtype=np.float16),
                NUM_PROTOTYPES_PER_BIN,
            ),
            "mean_distance": pa.array([0.2] * n_rows, type=pa.float32()),
            "null_mean": pa.array([0.0] * n_rows, type=pa.float32()),
            "bio_dispersion": pa.array([0.01] * n_rows, type=pa.float32()),
        },
        schema=build_reference_schema(pca_rank).with_metadata(metadata),
    )


class TestSchemaContracts(unittest.TestCase):
    def test_schema_version_is_2_0_0(self) -> None:
        self.assertEqual(SCHEMA_VERSION, "2.0.0")
        self.assertEqual(SUPPORTED_SCHEMA_VERSIONS, frozenset({"2.0.0"}))
        self.assertEqual(PRENATAL_METADATA_KEY, b"ccnv.prenatal")
        self.assertEqual(FRAGMENTOMICS_METADATA_KEY, b"ccnv.fragmentomics")

    def test_sample_schema_has_compact_v2_columns(self) -> None:
        expected_fields = [
            ("count", pa.uint32(), False),
            ("paired_count", pa.uint32(), False),
            ("frag_len_sum", pa.uint64(), False),
            ("short_frag_count", pa.uint32(), False),
            ("endpoint_count", pa.uint32(), False),
            ("frag_gc_sum", pa.float32(), False),
            ("frag_bands", pa.list_(pa.uint32(), NUM_FRAG_BANDS), True),
        ]
        self.assertEqual(SAMPLE_SCHEMA.names, [f[0] for f in expected_fields])
        for name, expected_type, expected_nullable in expected_fields:
            field = SAMPLE_SCHEMA.field(name)
            self.assertEqual(field.type, expected_type, f"Field {name} type mismatch")
            self.assertEqual(
                field.nullable, expected_nullable, f"Field {name} nullability mismatch"
            )

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

    def test_prenatal_model_metadata_with_fragmentomics_fields(self) -> None:
        p_fetal = [0.001] * 571
        p_maternal = [0.002] * 571
        weights = [0.1] * 36
        meta = PrenatalModelMetadata(
            prenatal_bin_size=100_000,
            yfrac_threshold=0.0041,
            ffy_intercept=0.0012,
            ffy_slope=0.985,
            olm_weights=[0.02 for _ in range(50)],
            olm_bias=0.005,
            ffx_intercept=1.0004,
            ffx_slope=-0.498,
            female_chrx_ratio_mean=1.0003,
            female_chrx_ratio_std=0.0112,
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
            min_frag_len=30,
            max_frag_len=600,
        )
        self.assertEqual(cfg.min_frag_len, 30)
        self.assertEqual(cfg.max_frag_len, 600)
        with self.assertRaisesRegex(ValueError, "invalid min_frag_len"):
            ExtractConfig(
                alignment=Path("sample.bam"),
                prefix=Path("out"),
                min_frag_len=600,
                max_frag_len=30,
            )

    def test_sample_contract_round_trip(self) -> None:
        metadata = metadata_for(
            ArtifactKind.SAMPLE,
            contig_lengths={"chr1": 250_001, "scaffold-A": 99},
            values={
                "bin_size": 100_000,
                "source_identity": {"path": "sample.bam"},
                "reference_identity": {"name": "GRCh-test"},
                "qc": {"mapped_reads": 42},
                "filters": asdict(ReadFilters()),
            },
        )
        schema = SAMPLE_SCHEMA.with_metadata(metadata)

        validate_schema(schema, ArtifactKind.SAMPLE)
        decoded = decode_metadata(schema.metadata)
        self.assertEqual(decoded["schema_version"], SCHEMA_VERSION)
        self.assertEqual(decoded["coordinate_system"], COORDINATE_SYSTEM)
        self.assertEqual(decoded["contig_lengths"], {"chr1": 250_001, "scaffold-A": 99})
        self.assertEqual(decoded["contig_order"], ["chr1", "scaffold-A"])

    def test_schema_rejects_invalid_coordinate_system(self) -> None:
        metadata = metadata_for(
            ArtifactKind.SAMPLE,
            contig_lengths={"chr1": 100},
            values={
                "bin_size": 100,
                "source_identity": {},
                "reference_identity": {},
                "qc": {},
                "filters": asdict(ReadFilters()),
            },
        )
        values = decode_metadata(metadata)
        values["coordinate_system"] = "1-based-closed"
        invalid = {
            f"ccnv.{key}".encode(): json.dumps(value).encode()
            for key, value in values.items()
        }

        with self.assertRaisesRegex(SchemaError, "0-based-half-open"):
            validate_schema(SAMPLE_SCHEMA.with_metadata(invalid), ArtifactKind.SAMPLE)

    def test_schema_rejects_wrong_arrow_fields(self) -> None:
        schema = pa.schema(
            [
                pa.field("contig", pa.string(), nullable=False),
                pa.field("start", pa.int64(), nullable=False),
            ]
        )

        with self.assertRaisesRegex(SchemaError, "fields do not match"):
            validate_schema(schema, ArtifactKind.SAMPLE)

    def test_table_accepts_partial_final_bin_and_arbitrary_contigs(self) -> None:
        metadata = metadata_for(
            ArtifactKind.SAMPLE,
            contig_lengths={"scaffold-A": 250_001},
            values={
                "bin_size": 100_000,
                "source_identity": {},
                "reference_identity": {},
                "qc": {},
                "filters": asdict(ReadFilters()),
            },
        )
        table = pa.table(
            {
                "count": pa.array([10, 11, 12], type=pa.uint32()),
                "paired_count": pa.array([10, 11, 12], type=pa.uint32()),
                "frag_len_sum": pa.array([1500, 1826, 1716], type=pa.uint64()),
                "short_frag_count": pa.array([10, 0, 12], type=pa.uint32()),
                "endpoint_count": pa.array([20, 22, 24], type=pa.uint32()),
                "frag_gc_sum": pa.array([4.0, 4.4, 4.8], type=pa.float32()),
                "frag_bands": pa.nulls(3, type=pa.list_(pa.uint32(), NUM_FRAG_BANDS)),
            },
            schema=SAMPLE_SCHEMA.with_metadata(metadata),
        )

        validate_table(table, ArtifactKind.SAMPLE)
        smeta = validate_sample_table(table)
        codes, names, starts, ends = reconstruct_sample_grid(smeta)
        self.assertEqual(names, ("scaffold-A",))
        self.assertEqual(codes.tolist(), [0, 0, 0])
        self.assertEqual(starts.tolist(), [0, 100_000, 200_000])
        self.assertEqual(ends.tolist(), [100_000, 200_000, 250_001])

    def test_table_rejects_mismatched_sample_row_count(self) -> None:
        metadata = metadata_for(
            ArtifactKind.SAMPLE,
            contig_lengths={"chr1": 10},
            values={
                "bin_size": 10,
                "source_identity": {},
                "reference_identity": {},
                "qc": {},
                "filters": asdict(ReadFilters()),
            },
        )
        table = pa.table(
            {
                "count": pa.array([1, 2], type=pa.uint32()),
                "paired_count": pa.array([0, 0], type=pa.uint32()),
                "frag_len_sum": pa.array([0, 0], type=pa.uint64()),
                "short_frag_count": pa.array([0, 0], type=pa.uint32()),
                "endpoint_count": pa.array([2, 2], type=pa.uint32()),
                "frag_gc_sum": pa.array([0.4, 0.4], type=pa.float32()),
                "frag_bands": pa.nulls(2, type=pa.list_(pa.uint32(), NUM_FRAG_BANDS)),
            },
            schema=SAMPLE_SCHEMA.with_metadata(metadata),
        )

        with self.assertRaisesRegex(SchemaError, "outside"):
            validate_table(table, ArtifactKind.SAMPLE)
        with self.assertRaisesRegex(SchemaError, "outside"):
            validate_sample_table(table)

    def test_bin_coordinates_are_contig_start_end_tuples(self) -> None:
        metadata = metadata_for(
            ArtifactKind.SAMPLE,
            contig_lengths={"chr1": 100, "chr2": 80},
            values={
                "bin_size": 100,
                "source_identity": {},
                "reference_identity": {},
                "qc": {},
                "filters": asdict(ReadFilters()),
            },
        )
        table = pa.table(
            {
                "count": pa.array([1, 2], type=pa.uint32()),
                "paired_count": pa.array([1, 2], type=pa.uint32()),
                "frag_len_sum": pa.array([150, 330], type=pa.uint64()),
                "short_frag_count": pa.array([1, 0], type=pa.uint32()),
                "endpoint_count": pa.array([2, 4], type=pa.uint32()),
                "frag_gc_sum": pa.array([0.4, 0.9], type=pa.float32()),
                "frag_bands": pa.nulls(2, type=pa.list_(pa.uint32(), NUM_FRAG_BANDS)),
            },
            schema=SAMPLE_SCHEMA.with_metadata(metadata),
        )
        self.assertEqual(
            bin_coordinates(table),
            (("chr1", 0, 100), ("chr2", 0, 80)),
        )

    def test_reference_schema_accepted_and_old_rejected(self) -> None:
        ref_meta = metadata_for(
            ArtifactKind.REFERENCE,
            contig_lengths={"chr1": 200},
            values={
                "sample_identities": [],
                "model": {"bin_size": 100, "branches": ["autosomal"]},
            },
        )
        ref_table = _make_ref_table(
            ["autosomal", "autosomal"],
            ["chr1", "chr1"],
            [0, 100],
            [100, 200],
            ref_meta,
            pca_rank=2,
        )
        validate_schema(ref_table.schema, ArtifactKind.REFERENCE)
        validate_table(ref_table, ArtifactKind.REFERENCE)
        validate_reference_table(ref_table)

        old_schema = pa.schema(
            [
                pa.field("branch", pa.string(), nullable=False),
                pa.field("contig", pa.string(), nullable=False),
                pa.field("start", pa.int64(), nullable=False),
                pa.field("end", pa.int64(), nullable=False),
                pa.field("masked", pa.bool_(), nullable=False),
                pa.field("pca_mean", pa.float64(), nullable=False),
                pa.field("pca_loadings", pa.list_(pa.float64()), nullable=False),
                pa.field("nearest_ref_indices", pa.list_(pa.int32()), nullable=False),
            ]
        )
        with self.assertRaisesRegex(SchemaError, "fields do not match"):
            validate_schema(old_schema.with_metadata(ref_meta), ArtifactKind.REFERENCE)

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
        meta = _make_prenatal_metadata()
        as_dict = meta.to_dict()
        self.assertEqual(as_dict["prenatal_bin_size"], 100_000)
        self.assertNotIn("mlp_w1", as_dict)
        self.assertNotIn("preface_std", as_dict)

        encoded = meta.encode()
        decoded = PrenatalModelMetadata.decode(encoded)
        self.assertEqual(decoded, meta)

        from_dict_meta = PrenatalModelMetadata.from_dict(as_dict)
        self.assertEqual(from_dict_meta, meta)

    def test_prenatal_model_metadata_roundtrip_all_fragmentomic_fields(self) -> None:
        meta = PrenatalModelMetadata(
            prenatal_bin_size=100_000,
            yfrac_threshold=0.0041,
            ffy_intercept=0.0012,
            ffy_slope=0.985,
            olm_weights=[0.02 for _ in range(50)],
            olm_bias=0.005,
            ffx_intercept=1.0004,
            ffx_slope=-0.498,
            female_chrx_ratio_mean=1.0003,
            female_chrx_ratio_std=0.0112,
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
        meta = _make_prenatal_metadata()
        as_dict = meta.to_dict()
        for k in ("ffy_std", "ffa_std", "ffx_std", "ffa_logit_mode"):
            as_dict.pop(k, None)

        loaded = PrenatalModelMetadata.from_dict(as_dict)
        self.assertEqual(loaded.ffy_std, 0.01)
        self.assertEqual(loaded.ffa_std, 0.02)
        self.assertEqual(loaded.ffx_std, 0.025)
        self.assertFalse(loaded.ffa_logit_mode)

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
                ffy_intercept=0.0,
                ffy_slope=1.0,
                olm_weights=[],
                olm_bias=0.0,
                ffx_intercept=0.0,
                ffx_slope=1.0,
                female_chrx_ratio_mean=1.0,
                female_chrx_ratio_std=0.0,
            )

    def test_reference_metadata_round_trip(self) -> None:
        prenatal_meta = _make_prenatal_metadata()
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
        model_dict = cast(dict[str, object], metadata_dict["model"])
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
        assert reconstructed.prenatal is not None
        self.assertEqual(
            reconstructed.prenatal.yfrac_threshold, prenatal_meta.yfrac_threshold
        )

    def test_reference_metadata_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "bin_size must be a positive integer"):
            ReferenceMetadata(
                bin_size=-1,
                contig_lengths={"chr1": 100},
                branches=("autosomal",),
            )
        bad_mode = cast(AnalysisMode, "invalid_mode")
        with self.assertRaisesRegex(
            ValueError, "mode must be 'germline' or 'prenatal'"
        ):
            ReferenceMetadata(
                bin_size=5000,
                contig_lengths={"chr1": 100},
                branches=("autosomal",),
                mode=bad_mode,
            )
        with self.assertRaisesRegex(ValueError, "non-empty mapping"):
            ReferenceMetadata(
                bin_size=5000,
                contig_lengths={},
                branches=("autosomal",),
            )

    def test_validate_reference_table_all_five_branches(self) -> None:
        prenatal_meta = _make_prenatal_metadata()
        contig_lengths = {"chr1": 200_000, "chrX": 100_000, "chrY": 100_000}
        branches_list = [
            BRANCH_AUTOSOMAL,
            BRANCH_AUTOSOMAL,
            BRANCH_FEMALE_SEX,
            BRANCH_MALE_SEX,
            BRANCH_PRENATAL_AUTOSOMAL,
            BRANCH_PRENATAL_AUTOSOMAL,
            BRANCH_PRENATAL_FFY,
        ]
        contigs_list = ["chr1", "chr1", "chrX", "chrY", "chr1", "chr1", "chrY"]
        starts_list = [0, 5000, 0, 0, 0, 100_000, 0]
        ends_list = [5000, 10_000, 5000, 5000, 100_000, 200_000, 5000]

        ref_meta = metadata_for(
            ArtifactKind.REFERENCE,
            contig_lengths=contig_lengths,
            values={
                "sample_identities": ["s1", "s2"],
                "model": {
                    "bin_size": 5000,
                    "prenatal_bin_size": 100_000,
                    "mode": "prenatal",
                    "branches": sorted(KNOWN_REFERENCE_BRANCHES),
                },
                "prenatal": prenatal_meta.to_dict(),
            },
        )
        table = _make_ref_table(
            branches_list, contigs_list, starts_list, ends_list, ref_meta
        )

        validate_reference_table(table)

        extracted_prenatal = PrenatalModelMetadata.from_table(table)
        assert extracted_prenatal is not None
        self.assertEqual(
            extracted_prenatal.yfrac_threshold, prenatal_meta.yfrac_threshold
        )

        ref_metadata_obj = ReferenceMetadata.from_table(table)
        self.assertEqual(ref_metadata_obj.mode, "prenatal")
        self.assertEqual(ref_metadata_obj.bin_size, 5000)
        self.assertEqual(ref_metadata_obj.prenatal_bin_size, 100_000)

    def test_validate_reference_table_rejects_unknown_branch(self) -> None:
        ref_meta = metadata_for(
            ArtifactKind.REFERENCE,
            contig_lengths={"chr1": 10_000},
            values={
                "sample_identities": [],
                "model": {"bin_size": 5000, "branches": ["unknown_branch"]},
            },
        )
        table = _make_ref_table(["unknown_branch"], ["chr1"], [0], [5000], ref_meta)

        with self.assertRaisesRegex(SchemaError, "unknown branch"):
            validate_reference_table(table)

    def test_validate_reference_table_bin_spacing_checks(self) -> None:
        contig_lengths = {"chr1": 200_000}
        ref_meta = metadata_for(
            ArtifactKind.REFERENCE,
            contig_lengths=contig_lengths,
            values={
                "sample_identities": [],
                "model": {
                    "bin_size": 5000,
                    "prenatal_bin_size": 100_000,
                    "branches": [
                        BRANCH_AUTOSOMAL,
                        BRANCH_PRENATAL_AUTOSOMAL,
                    ],
                },
            },
        )

        table_bad_auto = _make_ref_table(
            [BRANCH_AUTOSOMAL], ["chr1"], [0], [10_000], ref_meta
        )
        with self.assertRaisesRegex(SchemaError, "bin width 10000 does not match"):
            validate_reference_table(table_bad_auto)

        table_bad_prenatal = _make_ref_table(
            [BRANCH_PRENATAL_AUTOSOMAL], ["chr1"], [0], [5000], ref_meta
        )
        with self.assertRaisesRegex(SchemaError, "bin width 5000 does not match"):
            validate_reference_table(table_bad_prenatal)

        ref_meta_no_p_size = metadata_for(
            ArtifactKind.REFERENCE,
            contig_lengths=contig_lengths,
            values={
                "sample_identities": [],
                "model": {
                    "bin_size": 5000,
                    "branches": [BRANCH_PRENATAL_AUTOSOMAL],
                },
            },
        )
        table_no_p_size = _make_ref_table(
            [BRANCH_PRENATAL_AUTOSOMAL], ["chr1"], [0], [100_000], ref_meta_no_p_size
        )
        with self.assertRaisesRegex(SchemaError, "prenatal_bin_size is not specified"):
            validate_reference_table(table_no_p_size)

        table_unaligned_start = _make_ref_table(
            [BRANCH_AUTOSOMAL], ["chr1"], [100], [5100], ref_meta
        )
        with self.assertRaisesRegex(
            SchemaError, "start coordinate 100 is not a multiple"
        ):
            validate_reference_table(table_unaligned_start)

    def test_validate_reference_table_large_table_fast_path(self) -> None:
        n_bins = 1005
        bin_size = 5000
        contig_len = n_bins * bin_size
        starts = [i * bin_size for i in range(n_bins)]
        ends = [(i + 1) * bin_size for i in range(n_bins)]

        ref_meta = metadata_for(
            ArtifactKind.REFERENCE,
            contig_lengths={"chr1": contig_len},
            values={
                "sample_identities": ["s1"],
                "model": {
                    "bin_size": bin_size,
                    "branches": [BRANCH_AUTOSOMAL],
                },
            },
        )
        table = _make_ref_table(
            [BRANCH_AUTOSOMAL] * n_bins,
            ["chr1"] * n_bins,
            starts,
            ends,
            ref_meta,
        )
        validate_reference_table(table)

        contigs_with_undeclared = ["chr1"] * (n_bins - 1) + ["chrUndeclared"]
        bad_table = _make_ref_table(
            [BRANCH_AUTOSOMAL] * n_bins,
            contigs_with_undeclared,
            starts,
            ends,
            ref_meta,
        )
        with self.assertRaisesRegex(SchemaError, "undeclared contig 'chrUndeclared'"):
            validate_reference_table(bad_table)


if __name__ == "__main__":
    unittest.main()
