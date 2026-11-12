"""Unit tests for ccnv.io subpackage."""

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa

from ccnv.io.alignments import (
    alignment_format,
    read_alignment_header,
    validate_alignment_inputs,
)
from ccnv.io.parquet import (
    is_sample_parquet,
    read_reference_parquet,
    read_sample_buffers,
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
    NUM_FRAG_BANDS,
    NUM_GC_QUANTILES,
    NUM_PROTOTYPES_PER_BIN,
    SAMPLE_SCHEMA,
    SampleMetadata,
    build_reference_schema,
    reconstruct_sample_grid,
)


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


class TestParquetIO(unittest.TestCase):
    def test_sample_parquet_roundtrip_and_buffers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.parquet"
            metadata = metadata_for(
                ArtifactKind.SAMPLE,
                contig_lengths={"chr1": 200},
                values={
                    "bin_size": 100,
                    "contig_order": ["chr1"],
                    "source_identity": {"path": "test.bam"},
                    "reference_identity": {"name": "test"},
                    "qc": {},
                    "filters": asdict(ReadFilters()),
                },
            )
            table = pa.table(
                {
                    "count": pa.array([10, 20], type=pa.uint32()),
                    "paired_count": pa.array([0, 0], type=pa.uint32()),
                    "frag_len_sum": pa.array([0, 0], type=pa.uint64()),
                    "short_frag_count": pa.array([0, 0], type=pa.uint32()),
                    "endpoint_count": pa.array([0, 0], type=pa.uint32()),
                    "frag_gc_sum": pa.array([0.0, 0.0], type=pa.float32()),
                    "frag_bands": pa.nulls(
                        2, type=pa.list_(pa.uint32(), NUM_FRAG_BANDS)
                    ),
                },
                schema=SAMPLE_SCHEMA.with_metadata(metadata),
            )
            out = write_sample_parquet(table, path)
            self.assertEqual(out, path)
            self.assertTrue(path.is_file())

            loaded = read_sample_parquet(path)
            self.assertEqual(loaded.num_rows, 2)
            _, _, starts, _ = reconstruct_sample_grid(SampleMetadata.from_table(loaded))
            self.assertEqual(starts.tolist(), [0, 100])
            self.assertTrue(is_sample_parquet(path))

            bufs = read_sample_buffers(path)
            self.assertEqual(bufs.num_rows, 2)
            self.assertEqual(bufs.count.tolist(), [10, 20])

    def test_reference_parquet_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ref.parquet"
            metadata = metadata_for(
                ArtifactKind.REFERENCE,
                contig_lengths={"chr1": 100},
                values={
                    "cohort_samples": ["s1", "s2"],
                    "sample_identities": [{"id": "s1"}, {"id": "s2"}],
                    "n_samples": 2,
                    "model": {
                        "bin_size": 100,
                        "mode": "germline",
                        "branches": [
                            BRANCH_AUTOSOMAL,
                            BRANCH_FEMALE_SEX,
                            BRANCH_MALE_SEX,
                        ],
                    },
                },
            )
            ref_schema = build_reference_schema(1).with_metadata(metadata)
            table = pa.table(
                {
                    "branch": pa.DictionaryArray.from_arrays(
                        pa.array([0], type=pa.int8()), [BRANCH_AUTOSOMAL]
                    ),
                    "contig": pa.DictionaryArray.from_arrays(
                        pa.array([0], type=pa.int16()), ["chr1"]
                    ),
                    "start": pa.array([0], type=pa.uint32()),
                    "end": pa.array([100], type=pa.uint32()),
                    "masked": pa.array([False], type=pa.bool_()),
                    "gc": pa.array([0.45], type=pa.float32()),
                    "mappability": pa.array([1.0], type=pa.float32()),
                    "gc_quantiles": pa.nulls(
                        1, type=pa.list_(pa.float32(), NUM_GC_QUANTILES)
                    ),
                    "ref_frag_len": pa.array([166.0], type=pa.float32()),
                    "ref_ep_rate": pa.array([2.0], type=pa.float32()),
                    "pca_mean": pa.array([1.0], type=pa.float32()),
                    "pca_loadings": pa.FixedSizeListArray.from_arrays(
                        pa.array([0.1], type=pa.float32()), 1
                    ),
                    "proto_ids": pa.FixedSizeListArray.from_arrays(
                        pa.array([0] * NUM_PROTOTYPES_PER_BIN, type=pa.uint16()),
                        NUM_PROTOTYPES_PER_BIN,
                    ),
                    "proto_weights": pa.FixedSizeListArray.from_arrays(
                        pa.array([0.25] * NUM_PROTOTYPES_PER_BIN, type=pa.float16()),
                        NUM_PROTOTYPES_PER_BIN,
                    ),
                    "mean_distance": pa.array([0.05], type=pa.float32()),
                    "null_mean": pa.array([0.0], type=pa.float32()),
                    "bio_dispersion": pa.array([0.01], type=pa.float32()),
                },
                schema=ref_schema,
            )
            write_reference_parquet(table, path)
            loaded, ref_meta = read_reference_parquet(path)
            self.assertEqual(loaded.num_rows, 1)
            self.assertEqual(loaded.column("branch").to_pylist(), [BRANCH_AUTOSOMAL])
            self.assertEqual(ref_meta.bin_size, 100)


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
