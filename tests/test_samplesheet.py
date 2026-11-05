import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ccnv.cohort import ReferenceError
from ccnv.models import ReadFilters, SamplesheetEntry
from ccnv.samplesheet import load_samplesheet
from ccnv.schemas import REFERENCE_SCHEMA, SAMPLE_SCHEMA, ArtifactKind, metadata_for


def _write_sample(path: Path) -> Path:
    metadata = metadata_for(
        ArtifactKind.SAMPLE,
        contig_lengths={"chr1": 100},
        values={
            "bin_size": 100,
            "source_identity": {"id": path.stem},
            "reference_identity": {"name": "test"},
            "qc": {},
            "filters": asdict(ReadFilters()),
        },
    )
    table = pa.table(
        {
            "contig": ["chr1"],
            "start": pa.array([0], type=pa.int64()),
            "end": pa.array([100], type=pa.int64()),
            "count": pa.array([1], type=pa.uint64()),
        },
        schema=SAMPLE_SCHEMA.with_metadata(metadata),
    )
    pq.write_table(table, path)
    return path.resolve()


def _write_reference(path: Path) -> Path:
    metadata = metadata_for(
        ArtifactKind.REFERENCE,
        contig_lengths={"chr1": 100},
        values={
            "sample_identities": [{"path": "s1.ccnv"}],
            "model": {
                "bin_size": 100,
                "pca_components": 1,
                "nearest_bins": 1,
                "min_cohort_size": 1,
                "min_sex_group_size": 1,
                "mode": "germline",
            },
            "sex_contigs": ["chrX", "chrY"],
            "sex_signal_contigs": ["chrX", "chrY"],
            "sex_cutoff": 0.5,
        },
    )
    table = pa.table(
        {
            "contig": ["chr1"],
            "start": pa.array([0], type=pa.int64()),
            "end": pa.array([100], type=pa.int64()),
            "branch": ["autosomal"],
            "masked": pa.array([False], type=pa.bool_()),
            "gc": pa.array([0.5], type=pa.float32()),
            "mappability": pa.array([1.0], type=pa.float32()),
            "pca_mean": pa.array([1.0], type=pa.float32()),
            "pca_loadings": pa.array([[0.1]], type=pa.list_(pa.float32())),
            "nearest_ref_indices": pa.array([[0]], type=pa.list_(pa.int32())),
            "mean_distance": pa.array([0.0], type=pa.float32()),
            "null_mean": pa.array([1.0], type=pa.float32()),
            "null_std": pa.array([0.1], type=pa.float32()),
        },
        schema=REFERENCE_SCHEMA.with_metadata(metadata),
    )
    pq.write_table(table, path)
    return path.resolve()


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
                f"""
- filepath: {s1.name}
  sex: M
  ff: 0.12
- filepath: {s2.name}
  sex: F
  ff: 0.0
- filepath: {s3.name}
  sex: null
  ff: 1.0
- filepath: {s4.name}
""",
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
                f"""
- filepath: {s_a.name}
- filepath: {s_b.name}
- filepath: {s_c.name}
""",
                encoding="utf-8",
            )

            entries = load_samplesheet(sheet)
            self.assertEqual(
                tuple(e.filepath for e in entries),
                (s_a, s_b, s_c),
            )

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
                """
- filepath: samples/s1.ccnv
- filepath: ./s2.ccnv
""",
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
            with self.assertRaises(ReferenceError) as ctx:
                load_samplesheet(sheet)
            self.assertIn("cannot read sample Parquet", str(ctx.exception))

    def test_non_sample_parquet_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref_parquet = _write_reference(root / "ref.ccnv")

            sheet = root / "sheet.yaml"
            sheet.write_text(f"- filepath: {ref_parquet.name}\n", encoding="utf-8")
            with self.assertRaises(ReferenceError) as ctx:
                load_samplesheet(sheet)
            self.assertIn("invalid sample Parquet", str(ctx.exception))

    def test_duplicate_sample_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            s1 = _write_sample(root / "s1.ccnv")

            sheet = root / "sheet.yaml"
            sheet.write_text(
                f"""
- filepath: {s1.name}
- filepath: ./{s1.name}
""",
                encoding="utf-8",
            )
            with self.assertRaises(ReferenceError) as ctx:
                load_samplesheet(sheet)
            self.assertIn("duplicate sample path", str(ctx.exception))
