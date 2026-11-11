"""Unit tests for ccnv.io subpackage."""

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa

from ccnv.io.alignments import (
    alignment_format,
    read_alignment_header,
    rejection_reason,
    validate_alignment_inputs,
)
from ccnv.io.parquet import (
    is_sample_parquet,
    read_reference_parquet,
    read_sample_parquet,
    write_reference_parquet,
    write_sample_parquet,
)
from ccnv.io.samplesheet import (
    _get_samplesheet_schema,
    load_samplesheet,
)
from ccnv.io.tracks import (
    load_cytobands_track,
)
from ccnv.models import (
    BRANCH_AUTOSOMAL,
    BRANCH_FEMALE_SEX,
    BRANCH_MALE_SEX,
    AlignmentError,
    ArtifactKind,
    Cytoband,
    ExtractError,
    ReadFilters,
    ReferenceError,
    metadata_for,
)
from ccnv.models.schemas import (
    REFERENCE_SCHEMA,
    SAMPLE_SCHEMA,
)


class FakeRead:
    """Minimal fake aligned read for filter tests."""

    def __init__(
        self,
        reference_start: int | None = 0,
        mapping_quality: int = 60,
        *,
        is_unmapped: bool = False,
        is_secondary: bool = False,
        is_supplementary: bool = False,
        is_duplicate: bool = False,
        is_paired: bool = False,
        is_proper_pair: bool = True,
    ) -> None:
        self.reference_start = reference_start
        self.mapping_quality = mapping_quality
        self.is_unmapped = is_unmapped
        self.is_secondary = is_secondary
        self.is_supplementary = is_supplementary
        self.is_duplicate = is_duplicate
        self.is_paired = is_paired
        self.is_proper_pair = is_proper_pair


class TestAlignmentsIO(unittest.TestCase):
    def test_alignment_format_valid(self) -> None:
        self.assertEqual(alignment_format(Path("sample.bam")), "bam")
        self.assertEqual(alignment_format(Path("sample.BAM")), "bam")
        self.assertEqual(alignment_format(Path("sample.cram")), "cram")
        self.assertEqual(alignment_format(Path("sample.CRAM")), "cram")

    def test_alignment_format_invalid(self) -> None:
        with self.assertRaises(AlignmentError):
            alignment_format(Path("sample.sam"))
        with self.assertRaises(ExtractError):
            alignment_format(Path("sample.sam"))

    def test_validate_alignment_cram_missing_fasta(self) -> None:
        with self.assertRaises(AlignmentError):
            validate_alignment_inputs(Path("sample.cram"), fasta=None)

    def test_validate_alignment_cram_nonexistent_fasta(self) -> None:
        with self.assertRaises(AlignmentError):
            validate_alignment_inputs(
                Path("sample.cram"),
                fasta=Path("/nonexistent/reference.fa"),
            )

    def test_validate_alignment_missing_bam(self) -> None:
        with self.assertRaises(AlignmentError):
            read_alignment_header(Path("/nonexistent/sample.bam"))

    def test_rejection_reason(self) -> None:
        filters = ReadFilters(
            min_mapping_quality=20,
            exclude_duplicates=True,
            exclude_secondary=True,
            exclude_supplementary=True,
            require_proper_pair=True,
        )
        self.assertIsNone(rejection_reason(FakeRead(mapping_quality=60), filters))
        self.assertEqual(
            rejection_reason(FakeRead(is_unmapped=True), filters), "filter_unmapped"
        )
        self.assertEqual(
            rejection_reason(FakeRead(reference_start=None), filters),
            "filter_no_start",
        )
        self.assertEqual(
            rejection_reason(FakeRead(mapping_quality=10), filters), "filter_mapq"
        )
        self.assertEqual(
            rejection_reason(FakeRead(is_duplicate=True), filters), "filter_duplicate"
        )
        self.assertEqual(
            rejection_reason(FakeRead(is_secondary=True), filters), "filter_secondary"
        )
        self.assertEqual(
            rejection_reason(FakeRead(is_supplementary=True), filters),
            "filter_supplementary",
        )
        self.assertEqual(
            rejection_reason(FakeRead(is_paired=True, is_proper_pair=False), filters),
            "filter_improper_pair",
        )


class TestParquetIO(unittest.TestCase):
    def test_sample_parquet_roundtrip_both_signatures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.parquet"
            metadata = metadata_for(
                ArtifactKind.SAMPLE,
                contig_lengths={"chr1": 200},
                values={
                    "bin_size": 100,
                    "source_identity": {"path": "test.bam"},
                    "reference_identity": {"name": "test"},
                    "qc": {},
                    "filters": asdict(ReadFilters()),
                },
            )
            table = pa.table(
                {
                    "contig": pa.array(["chr1", "chr1"], type=pa.string()),
                    "start": pa.array([0, 100], type=pa.int64()),
                    "end": pa.array([100, 200], type=pa.int64()),
                    "count": pa.array([10, 20], type=pa.uint64()),
                    "fragment_lengths": pa.array([[], []], type=pa.list_(pa.uint16())),
                    "endpoint_count": pa.array([0, 0], type=pa.uint32()),
                    "frag_gc_sum": pa.array([0.0, 0.0], type=pa.float32()),
                },
                schema=SAMPLE_SCHEMA.with_metadata(metadata),
            )
            # Write (table, path)
            out = write_sample_parquet(table, path)
            self.assertEqual(out, path)
            self.assertTrue(path.is_file())

            # Read back
            loaded = read_sample_parquet(path)
            self.assertEqual(loaded.num_rows, 2)
            self.assertEqual(loaded.column("start").to_pylist(), [0, 100])
            self.assertTrue(is_sample_parquet(path))

    def test_reference_parquet_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ref.parquet"
            metadata = metadata_for(
                ArtifactKind.REFERENCE,
                contig_lengths={"chr1": 100},
                values={
                    "bin_size": 100,
                    "cohort_samples": ["s1", "s2"],
                    "sample_identities": [{"id": "s1"}, {"id": "s2"}],
                    "n_samples": 2,
                    "model": {
                        "mode": "germline",
                        "branches": [
                            BRANCH_AUTOSOMAL,
                            BRANCH_FEMALE_SEX,
                            BRANCH_MALE_SEX,
                        ],
                    },
                },
            )
            table = pa.table(
                {
                    "branch": pa.array([BRANCH_AUTOSOMAL], type=pa.string()),
                    "contig": pa.array(["chr1"], type=pa.string()),
                    "start": pa.array([0], type=pa.int64()),
                    "end": pa.array([100], type=pa.int64()),
                    "masked": pa.array([False], type=pa.bool_()),
                    "gc": pa.array([0.45], type=pa.float32()),
                    "mappability": pa.array([1.0], type=pa.float32()),
                    "pca_mean": pa.array([1.0], type=pa.float32()),
                    "pca_loadings": pa.array([[0.1]], type=pa.list_(pa.float32())),
                    "nearest_ref_indices": pa.array([[0]], type=pa.list_(pa.int32())),
                    "mean_distance": pa.array([0.05], type=pa.float32()),
                    "null_mean": pa.array([0.0], type=pa.float32()),
                    "null_std": pa.array([1.0], type=pa.float32()),
                },
                schema=REFERENCE_SCHEMA.with_metadata(metadata),
            )
            write_reference_parquet(table, path)
            loaded = read_reference_parquet(path)
            self.assertEqual(loaded.num_rows, 1)
            self.assertEqual(loaded.column("branch").to_pylist(), [BRANCH_AUTOSOMAL])


class TestTracksIO(unittest.TestCase):
    def test_load_cytobands_track(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".bed", delete=False) as f:
            f.write("#chrom\tchromStart\tchromEnd\tname\tgieStain\n")
            f.write("chr1\t0\t2300000\tp36.33\tgneg\n")
            f.write("chr1\t2300000\t5400000\tp36.32\tgpos25\n")
            f.flush()
            path = Path(f.name)

        try:
            bands = load_cytobands_track(path, {"chr1": 10000000})
            self.assertEqual(len(bands), 2)
            self.assertIsInstance(bands[0], Cytoband)
            self.assertEqual(bands[0].contig, "chr1")
            self.assertEqual(bands[0].start, 0)
            self.assertEqual(bands[0].end, 2300000)
            self.assertEqual(bands[0].name, "p36.33")
            self.assertEqual(bands[0].stain, "gneg")
        finally:
            path.unlink(missing_ok=True)


class TestSamplesheetIO(unittest.TestCase):
    def test_schema_loads_successfully(self) -> None:
        schema = _get_samplesheet_schema()
        self.assertIn("properties", schema["items"])
        self.assertIn("filepath", schema["items"]["required"])

    def test_load_samplesheet_missing_file(self) -> None:
        with self.assertRaises(ReferenceError):
            load_samplesheet(Path("/nonexistent/samplesheet.yaml"))
