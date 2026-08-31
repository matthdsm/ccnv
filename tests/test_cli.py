import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
from typer.testing import CliRunner

from ccnv.cli import SampleInputError, app, expand_sample_inputs
from ccnv.models import ReadFilters
from ccnv.reference import DEFAULT_SEX_CONTIGS, DEFAULT_SEX_SIGNAL_CONTIGS
from ccnv.schemas import SAMPLE_SCHEMA, ArtifactKind, metadata_for


class TestCliPathChecks(unittest.TestCase):
    def test_convert_rejects_missing_alignment(self) -> None:
        result = CliRunner().invoke(
            app, ["convert", "/does/not/exist.bam", "--prefix", "out"]
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("does not exist", result.output)

    def test_convert_rejects_missing_fasta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bam_path = Path(tmpdir) / "sample.bam"
            bam_path.write_bytes(b"BAM")
            result = CliRunner().invoke(
                app,
                [
                    "convert",
                    str(bam_path),
                    "--fasta",
                    str(Path(tmpdir) / "missing.fa"),
                ],
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("does not exist", result.output)

    def test_predict_rejects_missing_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            reference = Path(tmpdir) / "reference.parquet"
            reference.write_bytes(b"PARQUET")
            result = CliRunner().invoke(
                app,
                [
                    "predict",
                    str(Path(tmpdir) / "missing.parquet"),
                    str(reference),
                ],
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("does not exist", result.output)


class TestReferenceSampleExpansion(unittest.TestCase):
    def test_directory_is_searched_recursively_for_sample_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nested = root / "batch" / "lane"
            nested.mkdir(parents=True)
            first = _write_sample(nested / "a.parquet")
            second = _write_sample(root / "batch" / "c.parquet")
            (nested / "notes.txt").write_text("ignore")
            pq.write_table(pa.table({"x": [1]}), nested / "other.parquet")
            extra = _write_sample(root / "explicit.parquet")

            expanded = expand_sample_inputs([root / "batch", extra])
            self.assertEqual(expanded, (second, first, extra.resolve()))

    def test_empty_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            empty = Path(tmpdir) / "empty"
            empty.mkdir()
            with self.assertRaisesRegex(SampleInputError, "no sample Parquet"):
                expand_sample_inputs([empty])

    def test_cli_reference_expands_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nested = root / "controls" / "set-a"
            nested.mkdir(parents=True)
            first = _write_sample(nested / "s1.parquet")
            second = _write_sample(root / "controls" / "s2.parquet")
            with patch("ccnv.cli.run_reference") as run:
                result = CliRunner().invoke(
                    app,
                    [
                        "reference",
                        str(root / "controls"),
                        "--prefix",
                        str(root / "ref"),
                    ],
                )
            self.assertEqual(result.exit_code, 0, result.output)
            config = run.call_args.args[0]
            self.assertEqual(config.samples, (second, first))
            self.assertEqual(config.sex_contigs, DEFAULT_SEX_CONTIGS)
            self.assertEqual(config.sex_signal_contig, DEFAULT_SEX_SIGNAL_CONTIGS[0])

    def test_reference_help_shows_typer_defaults(self) -> None:
        result = CliRunner().invoke(app, ["reference", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("[default: X, Y, chrX,", result.output)
        self.assertIn("chrY]", result.output)
        self.assertIn("[default: chrY]", result.output)
        self.assertIn("[default: 5]", result.output)
        self.assertIn("[default: 300]", result.output)

    def test_short_help_flag_works_on_app_and_subcommands(self) -> None:
        for args in (["-h"], ["convert", "-h"], ["reference", "-h"], ["predict", "-h"]):
            with self.subTest(args=args):
                result = CliRunner().invoke(app, args)
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("Usage:", result.output)

    def test_help_uses_ccnv_executable_name(self) -> None:
        result = CliRunner().invoke(app, ["-h"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Usage: ccnv", result.output)

    def test_predict_help_shows_sex_choices_and_defaults(self) -> None:
        result = CliRunner().invoke(app, ["predict", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("[default: auto]", result.output)
        self.assertIn("<auto|female|male", result.output)

    def test_loglevel_flag_is_on_every_command(self) -> None:
        for command in ("convert", "reference", "predict"):
            with self.subTest(command=command):
                result = CliRunner().invoke(app, [command, "--help"])
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("--loglevel", result.output)
                self.assertIn("[default: WARNING]", result.output)

    def test_invalid_loglevel_is_rejected(self) -> None:
        result = CliRunner().invoke(app, ["convert", "--loglevel", "verbose"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("verbose", result.output)

    def test_cli_reference_forwards_sex_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample = _write_sample(root / "s1.parquet")
            with patch("ccnv.cli.run_reference") as run:
                result = CliRunner().invoke(
                    app,
                    [
                        "reference",
                        str(sample),
                        "--prefix",
                        str(root / "ref"),
                        "--sex-contig",
                        "chrX",
                        "--sex-contig",
                        "chrY",
                        "--sex-signal-contig",
                        "chrY",
                        "--sex-cutoff",
                        "0.004",
                    ],
                )
            self.assertEqual(result.exit_code, 0, result.output)
            config = run.call_args.args[0]
            self.assertEqual(config.sex_contigs, ("chrX", "chrY"))
            self.assertEqual(config.sex_signal_contig, "chrY")
            self.assertEqual(config.sex_cutoff, 0.004)

    def test_reference_rejects_sample_sex_option(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample = _write_sample(root / "s1.parquet")
            result = CliRunner().invoke(
                app,
                [
                    "reference",
                    str(sample),
                    "--prefix",
                    str(root / "ref"),
                    "--sample-sex",
                    "female",
                ],
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("No such option", result.output)


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


if __name__ == "__main__":
    unittest.main()
