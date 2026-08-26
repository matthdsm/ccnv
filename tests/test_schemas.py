import json
import unittest
from dataclasses import asdict

import pyarrow as pa

from ccnv.models import ReadFilters
from ccnv.schemas import (
    COORDINATE_SYSTEM,
    SAMPLE_SCHEMA,
    SCHEMA_VERSION,
    ArtifactKind,
    SchemaError,
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


if __name__ == "__main__":
    unittest.main()
