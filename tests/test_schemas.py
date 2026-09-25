import json
import unittest
from dataclasses import asdict

import pyarrow as pa

from ccnv.models import ReadFilters
from ccnv.schemas import (
    COORDINATE_SYSTEM,
    REFERENCE_SCHEMA,
    SAMPLE_SCHEMA,
    SCHEMA_VERSION,
    ArtifactKind,
    SchemaError,
    bin_coordinates,
    decode_metadata,
    metadata_for,
    validate_schema,
    validate_table,
)


class TestSchemaContracts(unittest.TestCase):
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

    def test_schema_rejects_legacy_one_based_coordinates(self) -> None:
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
                "model": {"branches": ["autosomal"]},
            },
        )
        ref_table = pa.table(
            {
                "branch": ["autosomal", "autosomal"],
                "contig": ["chr1", "chr1"],
                "start": pa.array([0, 100], type=pa.int64()),
                "end": pa.array([100, 200], type=pa.int64()),
                "masked": [False, False],
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


if __name__ == "__main__":
    unittest.main()
