"""Tests for sample/reference Parquet v2 I/O, atomic writers, and samplesheet ingestion."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from helpers import (
    make_reference_table,
    make_sample_buffers,
    metadata_for,
    write_synthetic_reference,
    write_synthetic_sample,
)

from ccnv.features.cohort import load_and_assemble_cohort
from ccnv.io import parquet as parquet_io
from ccnv.io.parquet import (
    read_reference_parquet,
    read_sample_buffers,
    write_reference_parquet,
    write_sample_parquet,
)
from ccnv.io.samplesheet import _get_samplesheet_schema, load_samplesheet
from ccnv.models import (
    BRANCH_AUTOSOMAL,
    BRANCH_FEMALE_SEX,
    BRANCH_MALE_SEX,
    NUM_FRAG_BANDS,
    NUM_GC_QUANTILES,
    NUM_PROTOTYPES_PER_BIN,
    SAMPLE_SCHEMA,
    ArtifactKind,
    BranchFitArrays,
    ReadFilters,
    ReferenceError,
    ReferenceMetadata,
    SampleBuffers,
    SamplesheetEntry,
    SchemaError,
    assemble_reference_table,
    build_reference_schema,
    reconstruct_sample_grid,
    validate_sample_table,
)
from ccnv.predict import build_branch_model


def _write_sample(path: Path) -> Path:
    write_synthetic_sample(
        path,
        [1],
        contigs=["chr1"],
        starts=[0],
        ends=[100],
        contig_lengths={"chr1": 100},
        bin_size=100,
    )
    return path.resolve()


def _write_reference(path: Path) -> Path:
    table = make_reference_table(
        branches=["autosomal"],
        contigs=["chr1"],
        starts=[0],
        ends=[100],
        contig_lengths={"chr1": 100},
        pca_loadings=[[0.1]],
        model={
            "bin_size": 100,
            "branches": ["autosomal"],
            "pca_components": 1,
            "nearest_bins": 1,
            "min_cohort_size": 1,
            "min_sex_group_size": 1,
            "mode": "germline",
        },
        sample_count=1,
    )
    write_synthetic_reference(path, table)
    return path.resolve()


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
            self.assertFalse(hasattr(parquet_io, "is_sample_parquet"))
            self.assertFalse(hasattr(parquet_io, "read_sample_parquet"))

            loaded = pq.read_table(path)
            self.assertEqual(loaded.num_rows, 2)
            _, _, starts, _ = reconstruct_sample_grid(validate_sample_table(loaded))
            self.assertEqual(starts.tolist(), [0, 100])

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
                    "model": {
                        "bin_size": 100,
                        "mode": "germline",
                        "sample_count": 2,
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

    def test_branch_partitioned_reference_io_and_zero_copy_branch_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            schema = build_reference_schema(2)
            starts = np.array([0, 100, 0, 100, 0, 100, 0, 0], dtype=np.uint32)
            ends = starts + np.uint32(100)
            n_rows = 8

            meta = ReferenceMetadata(
                bin_size=100,
                contig_lengths={"chr1": 200, "chr2": 200, "chrX": 200, "chrY": 100},
                branches=(BRANCH_AUTOSOMAL, BRANCH_FEMALE_SEX, BRANCH_MALE_SEX),
                autosomal_contigs=("chr1", "chr2"),
                sex_contigs=("chrX", "chrY"),
                pca_components=2,
                branch_pca_ranks={
                    BRANCH_AUTOSOMAL: 2,
                    BRANCH_FEMALE_SEX: 1,
                    BRANCH_MALE_SEX: 0,
                },
            )
            loadings = np.array(
                [
                    [0.5, -0.5],
                    [0.5, 0.5],
                    [-0.5, -0.5],
                    [-0.5, 0.5],
                    [0.7, 0.0],
                    [-0.7, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
                dtype=np.float32,
            )
            pids = np.zeros((n_rows, 4), dtype=np.uint16)
            pweights = np.tile(
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float16), (n_rows, 1)
            )
            branch_dict = pa.DictionaryArray.from_arrays(
                pa.array([0, 0, 0, 0, 1, 1, 2, 2], type=pa.int8()),
                pa.array(
                    [BRANCH_AUTOSOMAL, BRANCH_FEMALE_SEX, BRANCH_MALE_SEX],
                    type=pa.string(),
                ),
            )
            contig_dict = pa.DictionaryArray.from_arrays(
                pa.array([0, 0, 1, 1, 2, 2, 2, 3], type=pa.int16()),
                pa.array(["chr1", "chr2", "chrX", "chrY"], type=pa.string()),
            )
            table = pa.table(
                {
                    "branch": branch_dict,
                    "contig": contig_dict,
                    "start": starts,
                    "end": ends,
                    "masked": np.zeros(n_rows, dtype=np.bool_),
                    "gc": np.full(n_rows, 0.42, dtype=np.float32),
                    "mappability": np.ones(n_rows, dtype=np.float32),
                    "gc_quantiles": pa.nulls(n_rows, type=pa.list_(pa.float32(), 8)),
                    "ref_frag_len": np.full(n_rows, 167.0, dtype=np.float32),
                    "ref_ep_rate": np.full(n_rows, 0.1, dtype=np.float32),
                    "pca_mean": np.full(n_rows, 0.25, dtype=np.float32),
                    "pca_loadings": pa.FixedSizeListArray.from_arrays(
                        pa.array(loadings.ravel(), type=pa.float32()), 2
                    ),
                    "proto_ids": pa.FixedSizeListArray.from_arrays(
                        pa.array(pids.ravel(), type=pa.uint16()), 4
                    ),
                    "proto_weights": pa.FixedSizeListArray.from_arrays(
                        pa.array(pweights.ravel(), type=pa.float16()), 4
                    ),
                    "mean_distance": np.full(n_rows, 0.1, dtype=np.float32),
                    "null_mean": np.zeros(n_rows, dtype=np.float32),
                    "bio_dispersion": np.full(n_rows, 0.01, dtype=np.float32),
                },
                schema=schema.with_metadata(meta.build_arrow_metadata()),
            )

            ref_path = write_reference_parquet(table, tmp_path / "ref.ccnv")
            pf = pq.ParquetFile(ref_path)
            self.assertEqual(pf.num_row_groups, 3)

            auto_only, loaded_meta = read_reference_parquet(
                ref_path, branches=[BRANCH_AUTOSOMAL], check_rows=False
            )
            self.assertEqual(auto_only.num_rows, 4)
            self.assertEqual(
                set(auto_only.column("branch").unique().to_pylist()),
                {BRANCH_AUTOSOMAL},
            )
            self.assertEqual(
                loaded_meta.branch_pca_ranks,
                {
                    BRANCH_AUTOSOMAL: 2,
                    BRANCH_FEMALE_SEX: 1,
                    BRANCH_MALE_SEX: 0,
                },
            )

            model = build_branch_model(auto_only, BRANCH_AUTOSOMAL, loaded_meta)
            self.assertEqual(model.proto_ids.shape, (4, 4))
            self.assertEqual(model.proto_ids.dtype, np.uint16)
            self.assertEqual(model.proto_weights.shape, (4, 4))
            self.assertEqual(model.proto_weights.dtype, np.float16)
            self.assertEqual(model.pca_loadings.dtype, np.float32)
            self.assertEqual(model.pca_loadings.shape, (2, 4))

    def test_write_reference_parquet_handles_string_and_duplicate_vocab_branch_columns(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            schema_dict = build_reference_schema(1)
            n_rows = 4
            meta = ReferenceMetadata(
                bin_size=100,
                contig_lengths={"chr1": 200, "chrX": 200},
                branches=(BRANCH_AUTOSOMAL, BRANCH_FEMALE_SEX),
                pca_components=1,
            )
            base_cols: dict[str, object] = {
                "branch": pa.DictionaryArray.from_arrays(
                    pa.array([0, 1, 2, 2], type=pa.int8()),
                    pa.array(
                        [BRANCH_AUTOSOMAL, BRANCH_AUTOSOMAL, BRANCH_FEMALE_SEX],
                        type=pa.string(),
                    ),
                ),
                "contig": pa.DictionaryArray.from_arrays(
                    pa.array([0, 0, 1, 1], type=pa.int16()),
                    pa.array(["chr1", "chrX"], type=pa.string()),
                ),
                "start": np.array([0, 100, 0, 100], dtype=np.uint32),
                "end": np.array([100, 200, 100, 200], dtype=np.uint32),
                "masked": np.zeros(n_rows, dtype=np.bool_),
                "gc": np.full(n_rows, 0.42, dtype=np.float32),
                "mappability": np.ones(n_rows, dtype=np.float32),
                "gc_quantiles": pa.nulls(n_rows, type=pa.list_(pa.float32(), 8)),
                "ref_frag_len": np.full(n_rows, 166.0, dtype=np.float32),
                "ref_ep_rate": np.full(n_rows, 0.1, dtype=np.float32),
                "pca_mean": np.full(n_rows, 0.25, dtype=np.float32),
                "pca_loadings": pa.FixedSizeListArray.from_arrays(
                    pa.array([0.5, -0.5, 0.7, -0.7], type=pa.float32()), 1
                ),
                "proto_ids": pa.FixedSizeListArray.from_arrays(
                    pa.array(np.zeros(n_rows * 4, dtype=np.uint16)), 4
                ),
                "proto_weights": pa.FixedSizeListArray.from_arrays(
                    pa.array(
                        np.tile(
                            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float16),
                            (n_rows, 1),
                        ).ravel()
                    ),
                    4,
                ),
                "mean_distance": np.full(n_rows, 0.1, dtype=np.float32),
                "null_mean": np.zeros(n_rows, dtype=np.float32),
                "bio_dispersion": np.full(n_rows, 0.01, dtype=np.float32),
            }

            dup_vocab_table = pa.table(
                base_cols, schema=schema_dict.with_metadata(meta.build_arrow_metadata())
            )
            dup_out = write_reference_parquet(
                dup_vocab_table, tmp_path / "dup_vocab.ccnv"
            )
            self.assertEqual(pq.ParquetFile(dup_out).num_row_groups, 2)
            reloaded_dup, _ = read_reference_parquet(
                dup_out, branches=[BRANCH_AUTOSOMAL], check_rows=True
            )
            self.assertEqual(reloaded_dup.num_rows, 2)

            str_schema = (
                schema_dict.set(0, pa.field("branch", pa.string(), nullable=False))
                .set(1, pa.field("contig", pa.string(), nullable=False))
                .with_metadata(meta.build_arrow_metadata())
            )
            str_table = pa.table(
                {
                    **base_cols,
                    "branch": pa.array(
                        [
                            BRANCH_AUTOSOMAL,
                            BRANCH_AUTOSOMAL,
                            BRANCH_FEMALE_SEX,
                            BRANCH_FEMALE_SEX,
                        ],
                        type=pa.string(),
                    ),
                    "contig": pa.array(
                        ["chr1", "chr1", "chrX", "chrX"], type=pa.string()
                    ),
                },
                schema=str_schema,
            )
            str_out = write_reference_parquet(str_table, tmp_path / "str_branch.ccnv")
            self.assertEqual(pq.ParquetFile(str_out).num_row_groups, 2)
            reloaded_str, _ = read_reference_parquet(str_out, check_rows=True)
            self.assertEqual(reloaded_str.num_rows, 4)

            auto_only_meta = ReferenceMetadata(
                bin_size=100,
                contig_lengths={"chr1": 400},
                branches=(BRANCH_AUTOSOMAL,),
                pca_components=1,
            )
            b_c0 = pa.DictionaryArray.from_arrays(
                pa.array([0, 0], type=pa.int8()),
                pa.array([BRANCH_AUTOSOMAL, BRANCH_FEMALE_SEX], type=pa.string()),
            )
            b_c1 = pa.DictionaryArray.from_arrays(
                pa.array([1, 1], type=pa.int8()),
                pa.array([BRANCH_FEMALE_SEX, BRANCH_AUTOSOMAL], type=pa.string()),
            )
            multichunk_table = pa.table(
                {
                    **base_cols,
                    "branch": pa.chunked_array([b_c0, b_c1]),
                    "contig": pa.DictionaryArray.from_arrays(
                        pa.array([0, 0, 0, 0], type=pa.int16()),
                        pa.array(["chr1"], type=pa.string()),
                    ),
                    "start": np.array([0, 100, 200, 300], dtype=np.uint32),
                    "end": np.array([100, 200, 300, 400], dtype=np.uint32),
                },
                schema=schema_dict.with_metadata(auto_only_meta.build_arrow_metadata()),
            )
            mc_out = write_reference_parquet(
                multichunk_table, tmp_path / "mc_branch.ccnv"
            )
            self.assertEqual(pq.ParquetFile(mc_out).num_row_groups, 1)

            with self.assertRaises(SchemaError):
                SampleBuffers.from_table(pa.table({"count": [1]}))

    def test_parquet_writers_are_atomic_and_preserve_existing_file_on_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            lengths = {"chr1": 10_000}
            bufs_v1 = make_sample_buffers(
                lengths, 5000, np.array([100, 120], dtype=np.uint32)
            )
            bufs_v2 = make_sample_buffers(
                lengths, 5000, np.array([999, 888], dtype=np.uint32)
            )
            s_path = tmp / "sample.ccnv"
            write_sample_parquet(bufs_v1.to_table(), s_path)

            with (
                patch("pyarrow.parquet.write_table", side_effect=OSError("disk full")),
                self.assertRaises(OSError),
            ):
                write_sample_parquet(bufs_v2.to_table(), s_path)

            reloaded_s = read_sample_buffers(s_path)
            np.testing.assert_array_equal(
                reloaded_s.count, np.array([100, 120], dtype=np.uint32)
            )
            self.assertEqual(list(tmp.glob("*.tmp")), [])

            fit = BranchFitArrays(
                branch=BRANCH_AUTOSOMAL,
                contigs=("chr1", "chr1"),
                starts=np.array([0, 5000], dtype=np.int64),
                ends=np.array([5000, 10_000], dtype=np.int64),
                masked=np.array([False, False], dtype=np.bool_),
                gc=None,
                mappability=None,
                gc_quantiles=None,
                ref_frag_len=None,
                ref_ep_rate=None,
                pca_mean=np.array([100.0, 100.0], dtype=np.float32),
                pca_loadings=np.zeros((2, 0), dtype=np.float32),
                proto_ids=np.zeros((2, NUM_PROTOTYPES_PER_BIN), dtype=np.uint16),
                proto_weights=np.tile(
                    np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float16), (2, 1)
                ),
                mean_distance=np.zeros(2, dtype=np.float32),
                null_mean=np.zeros(2, dtype=np.float32),
                bio_dispersion=np.full(2, 0.01, dtype=np.float32),
            )
            ref_meta = ReferenceMetadata(
                bin_size=5000,
                contig_lengths=lengths,
                branches=(BRANCH_AUTOSOMAL,),
                autosomal_contigs=("chr1",),
                sex_contigs=(),
                pca_components=0,
            )
            ref_tbl = assemble_reference_table(
                [fit], contig_order=("chr1",), metadata=ref_meta.build_arrow_metadata()
            )
            r_path = tmp / "ref.ccnv"
            write_reference_parquet(ref_tbl, r_path)

            with (
                patch(
                    "pyarrow.parquet.ParquetWriter.write_table",
                    side_effect=OSError("disk full"),
                ),
                self.assertRaises(OSError),
            ):
                write_reference_parquet(ref_tbl, r_path)

            reloaded_r, _ = read_reference_parquet(r_path)
            self.assertEqual(reloaded_r.num_rows, 2)
            self.assertEqual(list(tmp.glob("*.tmp")), [])


class TestSamplesheetIO(unittest.TestCase):
    def test_schema_loads_successfully(self) -> None:
        schema = _get_samplesheet_schema()
        self.assertIn("properties", schema["items"])
        self.assertIn("filepath", schema["items"]["required"])

    def test_load_samplesheet_missing_file(self) -> None:
        with self.assertRaises(ReferenceError):
            load_samplesheet(Path("/nonexistent/samplesheet.yaml"))


class TestSamplesheet(unittest.TestCase):
    def test_load_valid_yaml_full(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            s1 = _write_sample(root / "sample1.ccnv")
            s2 = _write_sample(root / "sample2.ccnv")
            s3 = _write_sample(root / "sample3.ccnv")
            s4 = _write_sample(root / "sample4.ccnv")

            sheet = root / "samplesheet.yaml"
            sheet.write_text(
                f"- filepath: {s1.name}\n  sex: M\n  ff: 0.12\n"
                f"- filepath: {s2.name}\n  sex: F\n  ff: 0.0\n"
                f"- filepath: {s3.name}\n  sex: null\n  ff: 1.0\n"
                f"- filepath: {s4.name}\n",
                encoding="utf-8",
            )

            entries = load_samplesheet(sheet)
            self.assertEqual(
                entries,
                (
                    SamplesheetEntry(filepath=s1, sex="M", ff=0.12),
                    SamplesheetEntry(filepath=s2, sex="F", ff=0.0),
                    SamplesheetEntry(filepath=s3, sex=None, ff=1.0),
                    SamplesheetEntry(filepath=s4, sex=None, ff=None),
                ),
            )

    def test_load_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            s1 = _write_sample(root / "s1.ccnv")
            s2 = _write_sample(root / "s2.ccnv")

            sheet = root / "samplesheet.json"
            sheet.write_text(
                json.dumps(
                    [
                        {"filepath": s1.name, "sex": "M", "ff": 0.08},
                        {"filepath": s2.name},
                    ]
                ),
                encoding="utf-8",
            )

            entries = load_samplesheet(sheet)
            self.assertEqual(
                entries,
                (
                    SamplesheetEntry(filepath=s1, sex="M", ff=0.08),
                    SamplesheetEntry(filepath=s2, sex=None, ff=None),
                ),
            )

    def test_load_preserves_deterministic_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            s_a = _write_sample(root / "zeta.ccnv")
            s_b = _write_sample(root / "alpha.ccnv")
            s_c = _write_sample(root / "mid.ccnv")

            sheet = root / "samplesheet.yaml"
            sheet.write_text(
                f"- filepath: {s_a.name}\n"
                f"- filepath: {s_b.name}\n"
                f"- filepath: {s_c.name}\n",
                encoding="utf-8",
            )

            entries = load_samplesheet(sheet)
            self.assertEqual(tuple(e.filepath for e in entries), (s_a, s_b, s_c))

    def test_relative_path_resolution_against_samplesheet_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sub = root / "cohort_dir"
            sub.mkdir()
            nested = sub / "samples"
            nested.mkdir()

            s1 = _write_sample(nested / "s1.ccnv")
            s2 = _write_sample(sub / "s2.ccnv")

            sheet = sub / "cohort.yaml"
            sheet.write_text(
                "- filepath: samples/s1.ccnv\n- filepath: ./s2.ccnv\n",
                encoding="utf-8",
            )

            entries = load_samplesheet(sheet)
            self.assertEqual(entries[0].filepath, s1)
            self.assertEqual(entries[1].filepath, s2)

    def test_absolute_path_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            s1 = _write_sample(root / "abs_sample.ccnv")
            sheet = root / "sheet.yaml"
            sheet.write_text(f"- filepath: {s1.resolve()}\n", encoding="utf-8")
            entries = load_samplesheet(sheet)
            self.assertEqual(entries[0].filepath, s1)

    def test_schema_error_missing_filepath(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sheet = Path(tmpdir) / "sheet.yaml"
            sheet.write_text("- sex: M\n  ff: 0.05\n", encoding="utf-8")
            with self.assertRaises(ReferenceError) as ctx:
                load_samplesheet(sheet)
            self.assertIn("entry 0", str(ctx.exception))
            self.assertIn("filepath", str(ctx.exception))

    def test_schema_error_empty_filepath(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sheet = Path(tmpdir) / "sheet.yaml"
            sheet.write_text("- filepath: ''\n", encoding="utf-8")
            with self.assertRaises(ReferenceError) as ctx:
                load_samplesheet(sheet)
            self.assertIn("entry 0", str(ctx.exception))
            self.assertIn("filepath", str(ctx.exception))

    def test_schema_error_invalid_sex(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sheet = Path(tmpdir) / "sheet.yaml"
            sheet.write_text("- filepath: s.ccnv\n  sex: Unknown\n", encoding="utf-8")
            with self.assertRaises(ReferenceError) as ctx:
                load_samplesheet(sheet)
            self.assertIn("entry 0", str(ctx.exception))
            self.assertIn("sex", str(ctx.exception))

    def test_schema_error_out_of_range_ff_low(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sheet = Path(tmpdir) / "sheet.yaml"
            sheet.write_text("- filepath: s.ccnv\n  ff: -0.01\n", encoding="utf-8")
            with self.assertRaises(ReferenceError) as ctx:
                load_samplesheet(sheet)
            self.assertIn("entry 0", str(ctx.exception))
            self.assertIn("ff", str(ctx.exception))

    def test_schema_error_out_of_range_ff_high(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sheet = Path(tmpdir) / "sheet.yaml"
            sheet.write_text("- filepath: s.ccnv\n  ff: 1.05\n", encoding="utf-8")
            with self.assertRaises(ReferenceError) as ctx:
                load_samplesheet(sheet)
            self.assertIn("entry 0", str(ctx.exception))
            self.assertIn("ff", str(ctx.exception))

    def test_schema_error_non_numeric_ff(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sheet = Path(tmpdir) / "sheet.yaml"
            sheet.write_text("- filepath: s.ccnv\n  ff: invalid\n", encoding="utf-8")
            with self.assertRaises(ReferenceError) as ctx:
                load_samplesheet(sheet)
            self.assertIn("entry 0", str(ctx.exception))
            self.assertIn("ff", str(ctx.exception))

    def test_schema_error_extra_properties(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sheet = Path(tmpdir) / "sheet.yaml"
            sheet.write_text("- filepath: s.ccnv\n  extra_key: 123\n", encoding="utf-8")
            with self.assertRaises(ReferenceError) as ctx:
                load_samplesheet(sheet)
            self.assertIn("entry 0", str(ctx.exception))
            self.assertIn("extra_key", str(ctx.exception))

    def test_schema_error_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sheet = Path(tmpdir) / "sheet.yaml"
            sheet.write_text("[]\n", encoding="utf-8")
            with self.assertRaises(ReferenceError) as ctx:
                load_samplesheet(sheet)
            self.assertIn("schema violation", str(ctx.exception))

    def test_schema_error_non_list_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sheet = Path(tmpdir) / "sheet.yaml"
            sheet.write_text("filepath: s.ccnv\n", encoding="utf-8")
            with self.assertRaises(ReferenceError) as ctx:
                load_samplesheet(sheet)
            self.assertIn("schema violation", str(ctx.exception))

    def test_missing_samplesheet_file(self) -> None:
        with self.assertRaises(ReferenceError) as ctx:
            load_samplesheet(Path("/does/not/exist/samplesheet.yaml"))
        self.assertIn("samplesheet not found", str(ctx.exception))

    def test_malformed_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sheet = Path(tmpdir) / "bad.yaml"
            sheet.write_text("[unclosed bracket\n", encoding="utf-8")
            with self.assertRaises(ReferenceError) as ctx:
                load_samplesheet(sheet)
            self.assertIn("failed to parse YAML/JSON", str(ctx.exception))

    def test_missing_sample_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sheet = Path(tmpdir) / "sheet.yaml"
            sheet.write_text("- filepath: missing.ccnv\n", encoding="utf-8")
            with self.assertRaises(ReferenceError) as ctx:
                load_samplesheet(sheet)
            self.assertIn("sample file not found", str(ctx.exception))

    def test_non_parquet_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            corrupt = root / "corrupt.ccnv"
            corrupt.write_text("not a parquet file", encoding="utf-8")
            sheet = root / "sheet.yaml"
            sheet.write_text(f"- filepath: {corrupt.name}\n", encoding="utf-8")
            entries = load_samplesheet(sheet)
            with self.assertRaises(ReferenceError) as ctx:
                load_and_assemble_cohort(entries)
            self.assertIn("cannot read sample Parquet", str(ctx.exception))

    def test_non_sample_parquet_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref_parquet = _write_reference(root / "ref.ccnv")
            sheet = root / "sheet.yaml"
            sheet.write_text(f"- filepath: {ref_parquet.name}\n", encoding="utf-8")
            entries = load_samplesheet(sheet)
            with self.assertRaises(ReferenceError) as ctx:
                load_and_assemble_cohort(entries)
            self.assertIn("artifact_kind", str(ctx.exception))

    def test_duplicate_sample_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            s1 = _write_sample(root / "s1.ccnv")
            sheet = root / "sheet.yaml"
            sheet.write_text(
                f"- filepath: {s1.name}\n- filepath: ./{s1.name}\n",
                encoding="utf-8",
            )
            with self.assertRaises(ReferenceError) as ctx:
                load_samplesheet(sheet)
            self.assertIn("duplicate sample path", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
