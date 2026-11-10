import re
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from typer.testing import CliRunner

from ccnv.cli import SampleInputError, app, expand_sample_inputs
from ccnv.models import ReadFilters
from ccnv.models.schemas import (
    REFERENCE_SCHEMA,
    SAMPLE_SCHEMA,
    ArtifactKind,
    metadata_for,
)
from ccnv.reference import DEFAULT_SEX_CONTIGS, DEFAULT_SEX_SIGNAL_CONTIGS

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _cli_text(output: str) -> str:
    """Plain CLI text, ignoring Rich color, boxes, and wrapping."""

    plain = _ANSI.sub("", output)
    return " ".join(re.sub(r"[^\x20-\x7e]+", " ", plain).split())


class TestCliPathChecks(unittest.TestCase):
    def test_extract_rejects_missing_alignment(self) -> None:
        result = CliRunner().invoke(
            app, ["extract", "/does/not/exist.bam", "--prefix", "out"]
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("does not exist", _cli_text(result.output))

    def test_extract_rejects_missing_fasta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bam_path = Path(tmpdir) / "sample.bam"
            bam_path.write_bytes(b"BAM")
            result = CliRunner().invoke(
                app,
                [
                    "extract",
                    str(bam_path),
                    "--fasta",
                    str(Path(tmpdir) / "missing.fa"),
                ],
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("does not exist", _cli_text(result.output))

    def test_predict_rejects_missing_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            reference = Path(tmpdir) / "reference.ccnv"
            reference.write_bytes(b"PARQUET")
            result = CliRunner().invoke(
                app,
                [
                    "predict",
                    str(Path(tmpdir) / "missing.ccnv"),
                    str(reference),
                ],
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("does not exist", _cli_text(result.output))


class TestReferenceSampleExpansion(unittest.TestCase):
    def test_directory_is_searched_recursively_for_sample_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nested = root / "batch" / "lane"
            nested.mkdir(parents=True)
            first = _write_sample(nested / "a.ccnv")
            second = _write_sample(root / "batch" / "c.ccnv")
            (nested / "notes.txt").write_text("ignore")
            pq.write_table(pa.table({"x": [1]}), nested / "other.ccnv")
            extra = _write_sample(root / "explicit.ccnv")

            expanded = expand_sample_inputs([root / "batch", extra])
            self.assertEqual(expanded, (second, first, extra.resolve()))

    def test_empty_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            empty = Path(tmpdir) / "empty"
            empty.mkdir()
            with self.assertRaisesRegex(SampleInputError, "no sample CCNV files"):
                expand_sample_inputs([empty])

    def test_cli_reference_with_samplesheet(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample = _write_sample(root / "s1.ccnv")
            sheet = _write_samplesheet(root / "samples.yaml", [sample])
            with patch("ccnv.cli.run_reference") as run:
                result = CliRunner().invoke(
                    app,
                    [
                        "reference",
                        "--samplesheet",
                        str(sheet),
                        "--prefix",
                        str(root / "ref"),
                    ],
                )
            self.assertEqual(result.exit_code, 0, result.output)
            config = run.call_args.args[0]
            self.assertEqual(config.samplesheet_path, sheet)
            self.assertEqual(config.mode, "germline")
            self.assertEqual(config.prenatal_bin_size, 100_000)
            self.assertEqual(config.sex_contigs, DEFAULT_SEX_CONTIGS)
            self.assertEqual(config.sex_signal_contig, DEFAULT_SEX_SIGNAL_CONTIGS[0])

    def test_cli_reference_short_samplesheet_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample = _write_sample(root / "s1.ccnv")
            sheet = _write_samplesheet(root / "samples.yaml", [sample])
            with patch("ccnv.cli.run_reference") as run:
                result = CliRunner().invoke(
                    app,
                    [
                        "reference",
                        "-s",
                        str(sheet),
                        "--prefix",
                        str(root / "ref"),
                    ],
                )
            self.assertEqual(result.exit_code, 0, result.output)
            config = run.call_args.args[0]
            self.assertEqual(config.samplesheet_path, sheet)

    def test_reference_help_shows_typer_defaults(self) -> None:
        result = CliRunner().invoke(
            app, ["reference", "--help"], env={"COLUMNS": "200"}
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("chrY]", _cli_text(result.output))
        self.assertIn("[default: chrY]", _cli_text(result.output))
        self.assertIn("[default: 5]", _cli_text(result.output))
        self.assertIn("[default: 300]", _cli_text(result.output))
        self.assertIn("--samplesheet", _cli_text(result.output))
        self.assertIn("-s", _cli_text(result.output))
        self.assertIn("--fasta", _cli_text(result.output))
        self.assertIn("--mode", _cli_text(result.output))
        self.assertIn("[default: germline]", _cli_text(result.output))
        self.assertIn("--prenatal-binsize", _cli_text(result.output))
        self.assertIn("[default: 100000]", _cli_text(result.output))

    def test_short_help_flag_works_on_app_and_subcommands(self) -> None:
        for args in (
            ["-h"],
            ["extract", "-h"],
            ["reference", "-h"],
            ["ref-qc", "-h"],
            ["predict", "-h"],
        ):
            with self.subTest(args=args):
                result = CliRunner().invoke(app, args)
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("Usage:", _cli_text(result.output))

    def test_help_uses_ccnv_executable_name(self) -> None:
        result = CliRunner().invoke(app, ["-h"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Usage: ccnv", _cli_text(result.output))

    def test_predict_help_shows_sex_choices_and_defaults(self) -> None:
        result = CliRunner().invoke(app, ["predict", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("BAM, CRAM, or sample Parquet input", _cli_text(result.output))
        self.assertIn("[default: autosomal]", _cli_text(result.output))
        self.assertRegex(_cli_text(result.output), r"<autosomal\|female.*\|male>")
        self.assertIn("--mode", _cli_text(result.output))
        self.assertIn("[default: germline]", _cli_text(result.output))

    def test_loglevel_flag_is_on_every_command(self) -> None:
        for command in ("extract", "reference", "ref-qc", "predict"):
            with self.subTest(command=command):
                result = CliRunner().invoke(app, [command, "--help"])
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("--loglevel", _cli_text(result.output))
                self.assertIn("[default: WARNING]", _cli_text(result.output))

    def test_invalid_loglevel_is_rejected(self) -> None:
        result = CliRunner().invoke(app, ["extract", "--loglevel", "verbose"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("verbose", _cli_text(result.output))

    def test_cli_reference_forwards_sex_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample = _write_sample(root / "s1.ccnv")
            sheet = _write_samplesheet(root / "samples.yaml", [sample])
            with patch("ccnv.cli.run_reference") as run:
                result = CliRunner().invoke(
                    app,
                    [
                        "reference",
                        "-s",
                        str(sheet),
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

    def test_cli_reference_forwards_threads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample = _write_sample(root / "s1.ccnv")
            sheet = _write_samplesheet(root / "samples.yaml", [sample])
            with patch("ccnv.cli.run_reference") as run:
                result = CliRunner().invoke(
                    app,
                    [
                        "reference",
                        "-s",
                        str(sheet),
                        "--prefix",
                        str(root / "ref"),
                        "-t",
                        "4",
                    ],
                )
            self.assertEqual(result.exit_code, 0, result.output)
            config = run.call_args.args[0]
            self.assertEqual(config.threads, 4)

    def test_reference_rejects_sample_sex_option(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample = _write_sample(root / "s1.ccnv")
            sheet = _write_samplesheet(root / "samples.yaml", [sample])
            result = CliRunner().invoke(
                app,
                [
                    "reference",
                    "-s",
                    str(sheet),
                    "--prefix",
                    str(root / "ref"),
                    "--sample-sex",
                    "female",
                ],
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("No such option", _cli_text(result.output))

    def test_cli_reference_qc_report_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample = _write_sample(root / "s1.ccnv")
            sheet = _write_samplesheet(root / "samples.yaml", [sample])
            with patch("ccnv.cli.run_reference") as run:
                result = CliRunner().invoke(
                    app,
                    [
                        "reference",
                        "-s",
                        str(sheet),
                        "--prefix",
                        str(root / "ref"),
                        "--no-qc-report",
                    ],
                )
            self.assertEqual(result.exit_code, 0, result.output)
            config = run.call_args.args[0]
            self.assertFalse(config.qc_report)

    def test_reference_requires_samplesheet_or_fasta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = CliRunner().invoke(
                app,
                ["reference", "--prefix", str(Path(tmpdir) / "ref")],
            )
            self.assertEqual(result.exit_code, 2)
            self.assertIn(
                "provide exactly one of --samplesheet or --fasta",
                _cli_text(result.output),
            )

    def test_reference_rejects_both_samplesheet_and_fasta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample = _write_sample(root / "s1.ccnv")
            sheet = _write_samplesheet(root / "samples.yaml", [sample])
            fasta = root / "genome.fa"
            fasta.write_text(">chr1\nACGT\n", encoding="utf-8")
            result = CliRunner().invoke(
                app,
                [
                    "reference",
                    "--samplesheet",
                    str(sheet),
                    "--fasta",
                    str(fasta),
                    "--prefix",
                    str(root / "ref"),
                ],
            )
            self.assertEqual(result.exit_code, 2)
            self.assertIn(
                "provide either --samplesheet or --fasta, not both",
                _cli_text(result.output),
            )

    def test_reference_rejects_samples_with_samplesheet_or_fasta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample = _write_sample(root / "s1.ccnv")
            sheet = _write_samplesheet(root / "samples.yaml", [sample])
            fasta = root / "genome.fa"
            fasta.write_text(">chr1\nACGT\n", encoding="utf-8")
            expected = (
                "cannot combine positional sample arguments with "
                "--samplesheet or --fasta"
            )

            res1 = CliRunner().invoke(
                app,
                [
                    "reference",
                    str(sample),
                    "--samplesheet",
                    str(sheet),
                    "--prefix",
                    str(root / "ref"),
                ],
            )
            self.assertEqual(res1.exit_code, 2)
            self.assertIn(expected, _cli_text(res1.output))

            res2 = CliRunner().invoke(
                app,
                [
                    "reference",
                    str(sample),
                    "--fasta",
                    str(fasta),
                    "--prefix",
                    str(root / "ref"),
                ],
            )
            self.assertEqual(res2.exit_code, 2)
            self.assertIn(expected, _cli_text(res2.output))

    def test_reference_forwards_fasta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fasta = root / "genome.fa"
            fasta.write_text(">chr1\nACGT\n", encoding="utf-8")
            with patch("ccnv.cli.run_reference") as run:
                result = CliRunner().invoke(
                    app,
                    [
                        "reference",
                        "--fasta",
                        str(fasta),
                        "--prefix",
                        str(root / "ref"),
                    ],
                )
            self.assertEqual(result.exit_code, 0, result.output)
            config = run.call_args.args[0]
            self.assertEqual(config.fasta, fasta)
            self.assertIsNone(config.samplesheet_path)

    def test_reference_mode_and_prenatal_binsize(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample = _write_sample(root / "s1.ccnv")
            sheet = _write_samplesheet(root / "samples.yaml", [sample])
            with patch("ccnv.cli.run_reference") as run:
                result = CliRunner().invoke(
                    app,
                    [
                        "reference",
                        "-s",
                        str(sheet),
                        "--mode",
                        "prenatal",
                        "--prenatal-binsize",
                        "50000",
                        "--prefix",
                        str(root / "ref"),
                    ],
                )
            self.assertEqual(result.exit_code, 0, result.output)
            config = run.call_args.args[0]
            self.assertEqual(config.mode, "prenatal")
            self.assertEqual(config.prenatal_bin_size, 50_000)

    def test_reference_rejects_invalid_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample = _write_sample(root / "s1.ccnv")
            sheet = _write_samplesheet(root / "samples.yaml", [sample])
            result = CliRunner().invoke(
                app,
                [
                    "reference",
                    "-s",
                    str(sheet),
                    "--mode",
                    "invalid_mode",
                    "--prefix",
                    str(root / "ref"),
                ],
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("invalid_mode", _cli_text(result.output))

    def test_cli_ref_qc_generates_json_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref_path = _write_reference(root / "my_ref.ccnv")
            result = CliRunner().invoke(app, ["ref-qc", str(ref_path)])
            self.assertEqual(result.exit_code, 0, result.output)
            expected_json = root / "my_ref_qc.json"
            self.assertTrue(expected_json.is_file())
            self.assertIn("Reference QC: PASS", result.output)

    def test_cli_ref_qc_custom_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref_path = _write_reference(root / "my_ref.ccnv")
            custom_prefix = root / "custom_report"
            result = CliRunner().invoke(
                app, ["ref-qc", str(ref_path), "--prefix", str(custom_prefix)]
            )
            self.assertEqual(result.exit_code, 0, result.output)
            expected_json = root / "custom_report_qc.json"
            self.assertTrue(expected_json.is_file())

    def test_cli_predict_vcf_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample = _write_sample(root / "s1.ccnv")
            reference = _write_reference(root / "ref.ccnv")

            # Default: vcf=True
            with patch("ccnv.cli.run_predict") as run:
                result = CliRunner().invoke(
                    app, ["predict", str(sample), str(reference)]
                )
                self.assertEqual(result.exit_code, 0, result.output)
                config = run.call_args.args[0]
                self.assertTrue(config.vcf)

            # Explicit --no-vcf
            with patch("ccnv.cli.run_predict") as run:
                result = CliRunner().invoke(
                    app, ["predict", str(sample), str(reference), "--no-vcf"]
                )
                self.assertEqual(result.exit_code, 0, result.output)
                config = run.call_args.args[0]
                self.assertFalse(config.vcf)

            # Explicit --vcf
            with patch("ccnv.cli.run_predict") as run:
                result = CliRunner().invoke(
                    app, ["predict", str(sample), str(reference), "--vcf"]
                )
                self.assertEqual(result.exit_code, 0, result.output)
                config = run.call_args.args[0]
                self.assertTrue(config.vcf)

    def test_cli_predict_cytobands_and_no_plot_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample = _write_sample(root / "s1.ccnv")
            reference = _write_reference(root / "ref.ccnv")
            cyto = root / "cytoBand.txt"
            cyto.write_text("chr1\t0\t50\tp11.1\tgneg\n", encoding="utf-8")

            with patch("ccnv.cli.run_predict") as run:
                result = CliRunner().invoke(
                    app,
                    [
                        "predict",
                        str(sample),
                        str(reference),
                        "--cytobands",
                        str(cyto),
                    ],
                )
                self.assertEqual(result.exit_code, 0, result.output)
                config = run.call_args.args[0]
                self.assertEqual(config.cytobands, cyto)

            result = CliRunner().invoke(
                app,
                [
                    "predict",
                    str(sample),
                    str(reference),
                    "--prefix",
                    str(root / "out_no_plot"),
                    "--no-plot",
                    "--cytobands",
                    str(cyto),
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("--cytobands is ignored when --no-plot is set", result.output)
            self.assertFalse((root / "out_no_plot_plot.html").exists())

    def test_predict_forwards_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample = _write_sample(root / "s1.ccnv")
            reference = _write_reference(root / "ref.ccnv")

            # Default mode: germline
            with patch("ccnv.cli.run_predict") as run:
                result = CliRunner().invoke(
                    app, ["predict", str(sample), str(reference)]
                )
                self.assertEqual(result.exit_code, 0, result.output)
                config = run.call_args.args[0]
                self.assertEqual(config.mode, "germline")

            # Explicit prenatal mode
            with patch("ccnv.cli.run_predict") as run:
                result = CliRunner().invoke(
                    app,
                    [
                        "predict",
                        str(sample),
                        str(reference),
                        "--mode",
                        "prenatal",
                    ],
                )
                self.assertEqual(result.exit_code, 0, result.output)
                config = run.call_args.args[0]
                self.assertEqual(config.mode, "prenatal")

    def test_predict_rejects_invalid_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample = _write_sample(root / "s1.ccnv")
            reference = _write_reference(root / "ref.ccnv")
            result = CliRunner().invoke(
                app,
                [
                    "predict",
                    str(sample),
                    str(reference),
                    "--mode",
                    "not_a_mode",
                ],
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("not_a_mode", _cli_text(result.output))


def _write_samplesheet(path: Path, samples: list[Path]) -> Path:
    data = [{"id": f"s{i}", "filepath": str(s)} for i, s in enumerate(samples)]
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


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
            "fragment_lengths": pa.array([[]], type=pa.list_(pa.uint16())),
            "endpoint_count": pa.array([0], type=pa.uint32()),
            "frag_gc_sum": pa.array([0.0], type=pa.float32()),
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
                "branches": ["autosomal"],
                "autosomal_contigs": ["chr1"],
                "sex_contigs": [],
                "sex_signal_contig": None,
                "sex_cutoff": None,
                "sex_source": "none",
                "sample_sexes": None,
                "sex_inference": None,
                "pca_components": 1,
                "nearest_bins": 1,
                "max_batch_memory_mb": 64,
                "flat": False,
            },
        },
    )
    table = pa.table(
        {
            "branch": ["autosomal"],
            "contig": ["chr1"],
            "start": pa.array([0], type=pa.int64()),
            "end": pa.array([100], type=pa.int64()),
            "masked": pa.array([False], type=pa.bool_()),
            "gc": pa.array([float("nan")], type=pa.float32()),
            "mappability": pa.array([float("nan")], type=pa.float32()),
            "pca_mean": pa.array([1.0], type=pa.float64()),
            "pca_loadings": pa.array([[0.1]], type=pa.list_(pa.float64())),
            "nearest_ref_indices": pa.array([[0]], type=pa.list_(pa.int32())),
            "mean_distance": pa.array([0.5], type=pa.float32()),
            "null_mean": pa.array([0.0], type=pa.float64()),
            "null_std": pa.array([0.1], type=pa.float64()),
        },
        schema=REFERENCE_SCHEMA.with_metadata(metadata),
    )
    pq.write_table(table, path)
    return path.resolve()


if __name__ == "__main__":
    unittest.main()
