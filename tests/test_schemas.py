import json
import unittest
from dataclasses import asdict

import pyarrow as pa

from ccnv.models import ReadFilters
from ccnv.schemas import (
    BRANCH_AUTOSOMAL,
    BRANCH_FEMALE_SEX,
    BRANCH_MALE_SEX,
    BRANCH_PRENATAL_AUTOSOMAL,
    BRANCH_PRENATAL_FFY,
    COORDINATE_SYSTEM,
    KNOWN_REFERENCE_BRANCHES,
    PRENATAL_METADATA_KEY,
    REFERENCE_SCHEMA,
    SAMPLE_SCHEMA,
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    ArtifactKind,
    PrenatalModelMetadata,
    ReferenceMetadata,
    SchemaError,
    bin_coordinates,
    decode_metadata,
    metadata_for,
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
        mlp_w1=[[0.1, 0.2] for _ in range(50)],
        mlp_b1=[0.12, -0.34],
        mlp_w2=[[0.85], [-0.61]],
        mlp_b2=[0.04],
        mlp_skew_intercept=-0.002,
        mlp_skew_slope=1.018,
        olm_weights=[0.02 for _ in range(50)],
        olm_bias=0.005,
        olm_skew_intercept=-0.001,
        olm_skew_slope=1.01,
        ffx_intercept=1.0004,
        ffx_slope=-0.498,
        female_chrx_ratio_mean=1.0003,
        female_chrx_ratio_std=0.0112,
    )


class TestSchemaContracts(unittest.TestCase):
    def test_schema_version_is_2_0_0(self) -> None:
        self.assertEqual(SCHEMA_VERSION, "2.0.0")
        self.assertEqual(SUPPORTED_SCHEMA_VERSIONS, frozenset({"2.0.0"}))
        self.assertEqual(PRENATAL_METADATA_KEY, b"ccnv.prenatal")

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
                "contig": ["scaffold-A", "scaffold-A", "scaffold-A"],
                "start": pa.array([0, 100_000, 200_000], type=pa.int64()),
                "end": pa.array([100_000, 200_000, 250_001], type=pa.int64()),
                "count": pa.array([10, 11, 12], type=pa.uint64()),
            },
            schema=SAMPLE_SCHEMA.with_metadata(metadata),
        )

        validate_table(table, ArtifactKind.SAMPLE)
        validate_sample_table(table)

    def test_table_rejects_interval_past_true_contig_end(self) -> None:
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
                "contig": ["chr1"],
                "start": pa.array([0], type=pa.int64()),
                "end": pa.array([11], type=pa.int64()),
                "count": pa.array([1], type=pa.uint64()),
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
            contig_lengths={"chr1": 200, "chr2": 80},
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
                "contig": ["chr1", "chr2"],
                "start": pa.array([0, 0], type=pa.int64()),
                "end": pa.array([100, 80], type=pa.int64()),
                "count": pa.array([1, 2], type=pa.uint64()),
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
        ref_table = pa.table(
            {
                "branch": ["autosomal", "autosomal"],
                "contig": ["chr1", "chr1"],
                "start": pa.array([0, 100], type=pa.int64()),
                "end": pa.array([100, 200], type=pa.int64()),
                "masked": [False, False],
                "gc": pa.array([0.42, 0.45], type=pa.float32()),
                "mappability": pa.array([0.98, 0.99], type=pa.float32()),
                "pca_mean": [0.0, 0.0],
                "pca_loadings": [[0.1], [0.1]],
                "nearest_ref_indices": [[1], [0]],
                "mean_distance": pa.array([0.5, 0.5], type=pa.float32()),
                "null_mean": [0.0, 0.0],
                "null_std": [0.1, 0.1],
            },
            schema=REFERENCE_SCHEMA.with_metadata(ref_meta),
        )
        validate_schema(ref_table.schema, ArtifactKind.REFERENCE)
        validate_table(ref_table, ArtifactKind.REFERENCE)
        validate_reference_table(ref_table)

        # Old schema with nearest_ref_distances and null_ratios is rejected
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
                pa.field(
                    "nearest_ref_distances", pa.list_(pa.float64()), nullable=False
                ),
                pa.field("null_ratios", pa.list_(pa.float64()), nullable=False),
            ]
        )
        with self.assertRaisesRegex(SchemaError, "fields do not match"):
            validate_schema(old_schema.with_metadata(ref_meta), ArtifactKind.REFERENCE)

    def test_unsupported_schema_version_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            metadata_for(
                ArtifactKind.REFERENCE,
                contig_lengths={"chr1": 200},
                values={
                    "schema_version": "9.9.9",
                    "sample_identities": [],
                    "model": {"branches": ["autosomal"]},
                },
            )

        # Version 1.0.0 is also rejected now that schema version is 2.0.0
        with self.assertRaisesRegex(SchemaError, "unsupported schema version '1.0.0'"):
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
        self.assertEqual(len(as_dict["mlp_w1"]), 50)
        self.assertEqual(as_dict["mlp_b2"], [0.04])

        encoded = meta.encode()
        decoded = PrenatalModelMetadata.decode(encoded)
        self.assertEqual(decoded, meta)

        from_dict_meta = PrenatalModelMetadata.from_dict(as_dict)
        self.assertEqual(from_dict_meta, meta)

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
                mlp_w1=[],
                mlp_b1=[],
                mlp_w2=[],
                mlp_b2=[],
                mlp_skew_intercept=0.0,
                mlp_skew_slope=1.0,
                olm_weights=[],
                olm_bias=0.0,
                olm_skew_intercept=0.0,
                olm_skew_slope=1.0,
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
        self.assertEqual(metadata_dict["model"]["mode"], "prenatal")
        self.assertEqual(metadata_dict["model"]["prenatal_bin_size"], 100_000)
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
        self.assertIsNotNone(reconstructed.prenatal)
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
        with self.assertRaisesRegex(
            ValueError, "mode must be 'germline' or 'prenatal'"
        ):
            ReferenceMetadata(
                bin_size=5000,
                contig_lengths={"chr1": 100},
                branches=("autosomal",),
                mode="invalid_mode",  # type: ignore[arg-type]
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
        branches_list: list[str] = []
        contigs_list: list[str] = []
        starts_list: list[int] = []
        ends_list: list[int] = []

        # 1. autosomal (5 kb bins)
        branches_list.extend([BRANCH_AUTOSOMAL, BRANCH_AUTOSOMAL])
        contigs_list.extend(["chr1", "chr1"])
        starts_list.extend([0, 5000])
        ends_list.extend([5000, 10000])

        # 2. female_sex (5 kb bins)
        branches_list.append(BRANCH_FEMALE_SEX)
        contigs_list.append("chrX")
        starts_list.append(0)
        ends_list.append(5000)

        # 3. male_sex (5 kb bins)
        branches_list.append(BRANCH_MALE_SEX)
        contigs_list.append("chrY")
        starts_list.append(0)
        ends_list.append(5000)

        # 4. prenatal_autosomal (100 kb bins)
        branches_list.extend([BRANCH_PRENATAL_AUTOSOMAL, BRANCH_PRENATAL_AUTOSOMAL])
        contigs_list.extend(["chr1", "chr1"])
        starts_list.extend([0, 100_000])
        ends_list.extend([100_000, 200_000])

        # 5. prenatal_ffy (5 kb bins on chrY)
        branches_list.append(BRANCH_PRENATAL_FFY)
        contigs_list.append("chrY")
        starts_list.append(0)
        ends_list.append(5000)

        n_rows = len(branches_list)
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
        table = pa.table(
            {
                "branch": branches_list,
                "contig": contigs_list,
                "start": pa.array(starts_list, type=pa.int64()),
                "end": pa.array(ends_list, type=pa.int64()),
                "masked": [False] * n_rows,
                "gc": pa.array([0.4] * n_rows, type=pa.float32()),
                "mappability": pa.array([1.0] * n_rows, type=pa.float32()),
                "pca_mean": [0.0] * n_rows,
                "pca_loadings": [[0.1]] * n_rows,
                "nearest_ref_indices": [[0]] * n_rows,
                "mean_distance": pa.array([0.1] * n_rows, type=pa.float32()),
                "null_mean": [0.0] * n_rows,
                "null_std": [0.1] * n_rows,
            },
            schema=REFERENCE_SCHEMA.with_metadata(ref_meta),
        )

        validate_reference_table(table)

        # Also verify PrenatalModelMetadata.from_table
        extracted_prenatal = PrenatalModelMetadata.from_table(table)
        self.assertIsNotNone(extracted_prenatal)
        self.assertEqual(
            extracted_prenatal.yfrac_threshold, prenatal_meta.yfrac_threshold
        )

        # Verify ReferenceMetadata.from_table
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
        table = pa.table(
            {
                "branch": ["unknown_branch"],
                "contig": ["chr1"],
                "start": pa.array([0], type=pa.int64()),
                "end": pa.array([5000], type=pa.int64()),
                "masked": [False],
                "gc": pa.array([0.4], type=pa.float32()),
                "mappability": pa.array([1.0], type=pa.float32()),
                "pca_mean": [0.0],
                "pca_loadings": [[0.1]],
                "nearest_ref_indices": [[0]],
                "mean_distance": pa.array([0.1], type=pa.float32()),
                "null_mean": [0.0],
                "null_std": [0.1],
            },
            schema=REFERENCE_SCHEMA.with_metadata(ref_meta),
        )

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

        # 1. Invalid bin width on autosomal (expected 5000, got 10000)
        table_bad_auto = pa.table(
            {
                "branch": [BRANCH_AUTOSOMAL],
                "contig": ["chr1"],
                "start": pa.array([0], type=pa.int64()),
                "end": pa.array([10_000], type=pa.int64()),
                "masked": [False],
                "gc": pa.array([0.4], type=pa.float32()),
                "mappability": pa.array([1.0], type=pa.float32()),
                "pca_mean": [0.0],
                "pca_loadings": [[0.1]],
                "nearest_ref_indices": [[0]],
                "mean_distance": pa.array([0.1], type=pa.float32()),
                "null_mean": [0.0],
                "null_std": [0.1],
            },
            schema=REFERENCE_SCHEMA.with_metadata(ref_meta),
        )
        with self.assertRaisesRegex(SchemaError, "bin width 10000 does not match"):
            validate_reference_table(table_bad_auto)

        # 2. Invalid bin width on prenatal_autosomal (expected 100000, got 5000)
        table_bad_prenatal = pa.table(
            {
                "branch": [BRANCH_PRENATAL_AUTOSOMAL],
                "contig": ["chr1"],
                "start": pa.array([0], type=pa.int64()),
                "end": pa.array([5000], type=pa.int64()),
                "masked": [False],
                "gc": pa.array([0.4], type=pa.float32()),
                "mappability": pa.array([1.0], type=pa.float32()),
                "pca_mean": [0.0],
                "pca_loadings": [[0.1]],
                "nearest_ref_indices": [[0]],
                "mean_distance": pa.array([0.1], type=pa.float32()),
                "null_mean": [0.0],
                "null_std": [0.1],
            },
            schema=REFERENCE_SCHEMA.with_metadata(ref_meta),
        )
        with self.assertRaisesRegex(SchemaError, "bin width 5000 does not match"):
            validate_reference_table(table_bad_prenatal)

        # 3. Missing prenatal_bin_size when prenatal_autosomal branch is present
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
        table_no_p_size = pa.table(
            {
                "branch": [BRANCH_PRENATAL_AUTOSOMAL],
                "contig": ["chr1"],
                "start": pa.array([0], type=pa.int64()),
                "end": pa.array([100_000], type=pa.int64()),
                "masked": [False],
                "gc": pa.array([0.4], type=pa.float32()),
                "mappability": pa.array([1.0], type=pa.float32()),
                "pca_mean": [0.0],
                "pca_loadings": [[0.1]],
                "nearest_ref_indices": [[0]],
                "mean_distance": pa.array([0.1], type=pa.float32()),
                "null_mean": [0.0],
                "null_std": [0.1],
            },
            schema=REFERENCE_SCHEMA.with_metadata(ref_meta_no_p_size),
        )
        with self.assertRaisesRegex(SchemaError, "prenatal_bin_size is not specified"):
            validate_reference_table(table_no_p_size)

        # 4. Unaligned start coordinate (start=100 instead of multiple of 5000)
        table_unaligned_start = pa.table(
            {
                "branch": [BRANCH_AUTOSOMAL],
                "contig": ["chr1"],
                "start": pa.array([100], type=pa.int64()),
                "end": pa.array([5100], type=pa.int64()),
                "masked": [False],
                "gc": pa.array([0.4], type=pa.float32()),
                "mappability": pa.array([1.0], type=pa.float32()),
                "pca_mean": [0.0],
                "pca_loadings": [[0.1]],
                "nearest_ref_indices": [[0]],
                "mean_distance": pa.array([0.1], type=pa.float32()),
                "null_mean": [0.0],
                "null_std": [0.1],
            },
            schema=REFERENCE_SCHEMA.with_metadata(ref_meta),
        )
        with self.assertRaisesRegex(
            SchemaError, "start coordinate 100 is not a multiple"
        ):
            validate_reference_table(table_unaligned_start)

    def test_validate_reference_table_large_table_fast_path(self) -> None:
        # > 1000 rows to exercise the vectorized fast path
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
        table = pa.table(
            {
                "branch": [BRANCH_AUTOSOMAL] * n_bins,
                "contig": ["chr1"] * n_bins,
                "start": pa.array(starts, type=pa.int64()),
                "end": pa.array(ends, type=pa.int64()),
                "masked": [False] * n_bins,
                "gc": pa.array([0.4] * n_bins, type=pa.float32()),
                "mappability": pa.array([1.0] * n_bins, type=pa.float32()),
                "pca_mean": [0.0] * n_bins,
                "pca_loadings": [[0.1]] * n_bins,
                "nearest_ref_indices": [[0]] * n_bins,
                "mean_distance": pa.array([0.1] * n_bins, type=pa.float32()),
                "null_mean": [0.0] * n_bins,
                "null_std": [0.1] * n_bins,
            },
            schema=REFERENCE_SCHEMA.with_metadata(ref_meta),
        )

        # Valid > 1000 row table passes
        validate_reference_table(table)

        # > 1000 row table with undeclared contig must raise SchemaError, not KeyError
        contigs_with_undeclared = ["chr1"] * (n_bins - 1) + ["chrUndeclared"]
        bad_table = pa.table(
            {
                "branch": [BRANCH_AUTOSOMAL] * n_bins,
                "contig": contigs_with_undeclared,
                "start": pa.array(starts, type=pa.int64()),
                "end": pa.array(ends, type=pa.int64()),
                "masked": [False] * n_bins,
                "gc": pa.array([0.4] * n_bins, type=pa.float32()),
                "mappability": pa.array([1.0] * n_bins, type=pa.float32()),
                "pca_mean": [0.0] * n_bins,
                "pca_loadings": [[0.1]] * n_bins,
                "nearest_ref_indices": [[0]] * n_bins,
                "mean_distance": pa.array([0.1] * n_bins, type=pa.float32()),
                "null_mean": [0.0] * n_bins,
                "null_std": [0.1] * n_bins,
            },
            schema=REFERENCE_SCHEMA.with_metadata(ref_meta),
        )

        with self.assertRaisesRegex(
            SchemaError, "uses undeclared contig 'chrUndeclared'"
        ):
            validate_reference_table(bad_table)


if __name__ == "__main__":
    unittest.main()
