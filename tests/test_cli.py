"""Unit tests for ccnv CLI options, validation, and logging verbosity."""

from __future__ import annotations

import logging
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pysam
import yaml
from helpers import (
    make_reference_table,
    write_synthetic_reference,
    write_synthetic_sample,
)
from typer.testing import CliRunner

from ccnv.cli import app
from ccnv.models import parquet_path

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_LOG_CONTIGS = (("chr1", 1000), ("chr2", 1000))
_LOG_BIN_SIZE = 100


def _cli_text(output: str) -> str:
    """Plain CLI text, ignoring Rich color, boxes, and wrapping."""
    plain = _ANSI.sub("", output)
    return " ".join(re.sub(r"[^\x20-\x7e]+", " ", plain).split())


def _write_samplesheet(path: Path, samples: list[Path]) -> Path:
    data = [{"id": f"s{i}", "filepath": str(s)} for i, s in enumerate(samples)]
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _write_sample(path: Path) -> Path:
    return write_synthetic_sample(
        path,
        [1],
        contig_lengths={"chr1": 100},
        bin_size=100,
        identity={"id": path.stem},
    ).resolve()


def _write_reference(path: Path) -> Path:
    table = make_reference_table(
        branches=["autosomal"],
        contigs=["chr1"],
        starts=[0],
        ends=[100],
        contig_lengths={"chr1": 100},
        pca_mean=[1.0],
        pca_loadings=[[0.1]],
        mean_distance=[0.5],
        null_mean=[0.0],
        bio_dispersion=[0.01],
        sample_identities=[{"path": "s1.ccnv"}],
        model={
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
            "max_batch_memory_mb": 64,
            "flat": False,
        },
    )
    return write_synthetic_reference(path, table).resolve()


def _write_log_bam(path: Path, read_count: int = 5) -> None:
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": name, "LN": length} for name, length in _LOG_CONTIGS],
    }
    with pysam.AlignmentFile(path, "wb", header=header) as handle:
        for i in range(read_count):
            read = pysam.AlignedSegment()
            read.query_name = f"r_{i}"
            read.query_sequence = "A" * 50
            read.flag = 0
            read.reference_id = 0
            read.reference_start = i * 100
            read.mapping_quality = 60
            read.cigar = ((0, 50),)
            read.query_qualities = pysam.qualitystring_to_array("I" * 50)
            handle.write(read)
    pysam.index(str(path))


def _build_log_cohort(tmpdir: Path, n_samples: int = 4) -> tuple[list[Path], Path]:
    sample_paths: list[Path] = []
    for s in range(n_samples):
        bam = tmpdir / f"sample_{s}.bam"
        _write_log_bam(bam, read_count=5 + s)
        prefix = tmpdir / f"sample_{s}"
        result = CliRunner().invoke(
            app,
            [
                "extract",
                str(bam),
                "--prefix",
                str(prefix),
                "--bin-size",
                str(_LOG_BIN_SIZE),
            ],
        )
        if result.exit_code != 0:
            raise AssertionError(result.output)
        sample_paths.append(parquet_path(prefix))
    sheet_path = tmpdir / "samplesheet.yaml"
    sheet_path.write_text(
        "".join(f"- filepath: {p}\n" for p in sample_paths),
        encoding="utf-8",
    )
    ref_prefix = tmpdir / "reference"
    cmd = [
        "reference",
        "-s",
        str(sheet_path),
        "--prefix",
        str(ref_prefix),
        "--bin-size",
        str(_LOG_BIN_SIZE),
        "--min-cohort-size",
        str(n_samples),
        "--pca-components",
        "2",
    ]
    res = CliRunner().invoke(app, cmd)
    if res.exit_code != 0:
        raise AssertionError(res.output)
    return sample_paths, parquet_path(ref_prefix)


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

    def test_cli_rejects_zero_thresholds_and_nan_unit_floats(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            sample_p = tmp_dir / "s.ccnv"
            ref_p = tmp_dir / "r.ccnv"
            sample_p.write_bytes(b"placeholder")
            ref_p.write_bytes(b"placeholder")

            for bad_args in (
                ["predict", str(sample_p), str(ref_p), "--penalty", "0"],
                ["predict", str(sample_p), str(ref_p), "--z-threshold", "0"],
                ["predict", str(sample_p), str(ref_p), "--ratio-threshold", "0"],
                ["predict", str(sample_p), str(ref_p), "--min-mappability", "nan"],
                [
                    "reference",
                    "--prefix",
                    str(tmp_dir / "out"),
                    "--sex-cutoff",
                    "nan",
                ],
            ):
                res = runner.invoke(app, bad_args)
                self.assertNotEqual(res.exit_code, 0)


class TestReferenceSampleExpansion(unittest.TestCase):
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
            self.assertIsNone(config.sex_contigs)
            self.assertIsNone(config.sex_signal_contig)

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
        self.assertIn("--sex-contig", _cli_text(result.output))
        self.assertIn("--sex-signal-contig", _cli_text(result.output))
        self.assertNotIn("[default: chrY]", _cli_text(result.output))
        self.assertIn("[default: 5]", _cli_text(result.output))
        self.assertNotIn("--nearest-bins", _cli_text(result.output))
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

    def test_reference_rejects_positional_samples_and_nearest_bins(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample = _write_sample(root / "s1.ccnv")
            sheet = _write_samplesheet(root / "samples.yaml", [sample])

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
            self.assertNotEqual(res1.exit_code, 0)
            self.assertIn("unexpected extra argument", _cli_text(res1.output).lower())

            res2 = CliRunner().invoke(
                app,
                [
                    "reference",
                    "--samplesheet",
                    str(sheet),
                    "--nearest-bins",
                    "10",
                    "--prefix",
                    str(root / "ref"),
                ],
            )
            self.assertNotEqual(res2.exit_code, 0)
            self.assertIn("No such option", _cli_text(res2.output))

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

            with patch("ccnv.cli.run_predict") as run:
                result = CliRunner().invoke(
                    app, ["predict", str(sample), str(reference)]
                )
                self.assertEqual(result.exit_code, 0, result.output)
                config = run.call_args.args[0]
                self.assertTrue(config.vcf)
                self.assertFalse(config.bins_vcf)

            with patch("ccnv.cli.run_predict") as run:
                result = CliRunner().invoke(
                    app,
                    ["predict", str(sample), str(reference), "--bins-vcf", "-t", "3"],
                )
                self.assertEqual(result.exit_code, 0, result.output)
                config = run.call_args.args[0]
                self.assertTrue(config.vcf)
                self.assertTrue(config.bins_vcf)
                self.assertEqual(config.threads, 3)

            with patch("ccnv.cli.run_predict") as run:
                result = CliRunner().invoke(
                    app, ["predict", str(sample), str(reference), "--no-bins-vcf"]
                )
                self.assertEqual(result.exit_code, 0, result.output)
                config = run.call_args.args[0]
                self.assertFalse(config.bins_vcf)

            with patch("ccnv.cli.run_predict") as run:
                result = CliRunner().invoke(
                    app, ["predict", str(sample), str(reference), "--no-vcf"]
                )
                self.assertEqual(result.exit_code, 0, result.output)
                config = run.call_args.args[0]
                self.assertFalse(config.vcf)

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

            with patch("ccnv.cli.run_predict") as run:
                result = CliRunner().invoke(
                    app, ["predict", str(sample), str(reference)]
                )
                self.assertEqual(result.exit_code, 0, result.output)
                config = run.call_args.args[0]
                self.assertEqual(config.mode, "germline")

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
                self.assertFalse(config.twins)

            with patch("ccnv.cli.run_predict") as run:
                result = CliRunner().invoke(
                    app,
                    [
                        "predict",
                        str(sample),
                        str(reference),
                        "--mode",
                        "prenatal",
                        "--twins",
                    ],
                )
                self.assertEqual(result.exit_code, 0, result.output)
                config = run.call_args.args[0]
                self.assertEqual(config.mode, "prenatal")
                self.assertTrue(config.twins)

            with patch("ccnv.cli.run_predict") as run:
                result = CliRunner().invoke(
                    app,
                    [
                        "predict",
                        str(sample),
                        str(reference),
                        "--mode",
                        "prenatal",
                        "--no-twins",
                    ],
                )
                self.assertEqual(result.exit_code, 0, result.output)
                config = run.call_args.args[0]
                self.assertFalse(config.twins)

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


class TestLoggingVerbosity(unittest.TestCase):
    def tearDown(self) -> None:
        ccnv_logger = logging.getLogger("ccnv")
        ccnv_logger.handlers.clear()
        ccnv_logger.setLevel(logging.WARNING)

    def test_extract_debug_vs_default_logging(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bam = root / "sample.bam"
            _write_log_bam(bam)
            prefix = root / "sample"

            res_default = CliRunner().invoke(
                app,
                [
                    "extract",
                    str(bam),
                    "--prefix",
                    str(prefix),
                    "--bin-size",
                    str(_LOG_BIN_SIZE),
                ],
            )
            self.assertEqual(res_default.exit_code, 0, res_default.output)
            self.assertNotIn("starting extract:", res_default.output)
            self.assertNotIn("filter drops:", res_default.output)
            self.assertNotIn("extract QC summary:", res_default.output)

            prefix_debug = root / "sample_debug"
            res_debug = CliRunner().invoke(
                app,
                [
                    "extract",
                    str(bam),
                    "--prefix",
                    str(prefix_debug),
                    "--bin-size",
                    str(_LOG_BIN_SIZE),
                    "--loglevel",
                    "DEBUG",
                ],
            )
            self.assertEqual(res_debug.exit_code, 0, res_debug.output)
            self.assertIn("starting extract:", res_debug.output)
            self.assertIn("opening alignment", res_debug.output)
            self.assertIn("extract QC summary:", res_debug.output)
            self.assertIn("writing sample parquet", res_debug.output)
            self.assertIn("wrote sample parquet", res_debug.output)
            self.assertIn("extract completed in", res_debug.output)

    def test_reference_and_qc_debug_logging(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _samples, _ = _build_log_cohort(root, n_samples=4)
            ref_prefix = root / "ref_debug"

            sheet_path = root / "samplesheet.yaml"
            res = CliRunner().invoke(
                app,
                [
                    "reference",
                    "-s",
                    str(sheet_path),
                    "--prefix",
                    str(ref_prefix),
                    "--bin-size",
                    str(_LOG_BIN_SIZE),
                    "--min-cohort-size",
                    "4",
                    "--pca-components",
                    "2",
                    "--loglevel",
                    "DEBUG",
                ],
            )
            self.assertEqual(res.exit_code, 0, res.output)
            self.assertIn("starting reference:", res.output)
            self.assertIn("loaded cohort sample", res.output)
            self.assertIn("PCA fitted", res.output)
            self.assertIn("components: explained variance", res.output)
            self.assertIn("mask kept", res.output)
            self.assertIn("null models computed for", res.output)
            self.assertIn("wrote reference parquet", res.output)
            self.assertIn("evaluating reference QC", res.output)
            self.assertIn("branch autosomal QC:", res.output)
            self.assertIn("reference completed in", res.output)

            ref_path = parquet_path(ref_prefix)
            qc_res = CliRunner().invoke(
                app,
                ["ref-qc", str(ref_path), "--loglevel", "DEBUG"],
            )
            self.assertEqual(qc_res.exit_code, 0, qc_res.output)
            self.assertIn("starting ref-qc:", qc_res.output)
            self.assertIn("evaluating reference QC", qc_res.output)
            self.assertIn("reference QC overall:", qc_res.output)
            self.assertIn("ref-qc completed in", qc_res.output)

    def test_predict_and_report_debug_logging(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            samples, ref_path = _build_log_cohort(root, n_samples=4)
            pred_prefix = root / "pred_out"

            res = CliRunner().invoke(
                app,
                [
                    "predict",
                    str(samples[0]),
                    str(ref_path),
                    "--prefix",
                    str(pred_prefix),
                    "--penalty",
                    "1.0",
                    "--plot",
                    "--loglevel",
                    "DEBUG",
                ],
            )
            self.assertEqual(res.exit_code, 0, res.output)
            self.assertIn("starting predict:", res.output)
            self.assertIn("normalizing branch autosomal", res.output)
            self.assertIn("within-sample iteration", res.output)
            self.assertIn("median shift=", res.output)
            self.assertIn("segmenting and scoring 2 contigs with PELT", res.output)
            self.assertIn("segment and score finished:", res.output)
            self.assertIn("called", res.output)
            self.assertIn("writing prediction outputs", res.output)
            self.assertIn("wrote interactive genome HTML plot", res.output)
            self.assertIn("predict completed in", res.output)


if __name__ == "__main__":
    unittest.main()
