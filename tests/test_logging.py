import tempfile
import unittest
from pathlib import Path

import pysam
from typer.testing import CliRunner

from ccnv.cli import app
from ccnv.models import parquet_path

CONTIGS = (("chr1", 1000), ("chr2", 1000))
BIN_SIZE = 100


def _write_bam(path: Path, read_count: int = 5) -> None:
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": name, "LN": length} for name, length in CONTIGS],
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


def _build_test_cohort(tmpdir: Path, n_samples: int = 4) -> tuple[list[Path], Path]:
    sample_paths: list[Path] = []
    for s in range(n_samples):
        bam = tmpdir / f"sample_{s}.bam"
        _write_bam(bam, read_count=5 + s)
        prefix = tmpdir / f"sample_{s}"
        result = CliRunner().invoke(
            app,
            ["convert", str(bam), "--prefix", str(prefix), "--bin-size", str(BIN_SIZE)],
        )
        assert result.exit_code == 0, result.output
        sample_paths.append(parquet_path(prefix))
    ref_prefix = tmpdir / "reference"
    cmd = [
        "reference",
        "--prefix",
        str(ref_prefix),
        "--bin-size",
        str(BIN_SIZE),
        "--min-cohort-size",
        str(n_samples),
        "--pca-components",
        "2",
        "--nearest-bins",
        "5",
    ] + [str(p) for p in sample_paths]
    res = CliRunner().invoke(app, cmd)
    assert res.exit_code == 0, res.output
    return sample_paths, parquet_path(ref_prefix)


class TestLoggingVerbosity(unittest.TestCase):
    def test_convert_debug_vs_default_logging(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bam = root / "sample.bam"
            _write_bam(bam)
            prefix = root / "sample"

            # Run with default loglevel (warning)
            res_default = CliRunner().invoke(
                app,
                [
                    "convert",
                    str(bam),
                    "--prefix",
                    str(prefix),
                    "--bin-size",
                    str(BIN_SIZE),
                ],
            )
            self.assertEqual(res_default.exit_code, 0, res_default.output)
            self.assertNotIn("starting convert:", res_default.output)
            self.assertNotIn("filter drops:", res_default.output)
            self.assertNotIn("convert QC summary:", res_default.output)

            # Run with DEBUG loglevel
            prefix_debug = root / "sample_debug"
            res_debug = CliRunner().invoke(
                app,
                [
                    "convert",
                    str(bam),
                    "--prefix",
                    str(prefix_debug),
                    "--bin-size",
                    str(BIN_SIZE),
                    "--loglevel",
                    "DEBUG",
                ],
            )
            self.assertEqual(res_debug.exit_code, 0, res_debug.output)
            self.assertIn("starting convert:", res_debug.output)
            self.assertIn("read filters:", res_debug.output)
            self.assertIn("opening alignment", res_debug.output)
            self.assertIn("binned contig chr1", res_debug.output)
            self.assertIn("filter drops:", res_debug.output)
            self.assertIn("convert QC summary:", res_debug.output)
            self.assertIn("writing sample parquet", res_debug.output)
            self.assertIn("wrote sample parquet", res_debug.output)
            self.assertIn("convert completed in", res_debug.output)

    def test_reference_and_qc_debug_logging(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            samples, _ = _build_test_cohort(root, n_samples=4)
            ref_prefix = root / "ref_debug"

            res = CliRunner().invoke(
                app,
                [
                    "reference",
                    "--prefix",
                    str(ref_prefix),
                    "--bin-size",
                    str(BIN_SIZE),
                    "--min-cohort-size",
                    "4",
                    "--pca-components",
                    "2",
                    "--nearest-bins",
                    "5",
                    "--loglevel",
                    "DEBUG",
                ]
                + [str(p) for p in samples],
            )
            self.assertEqual(res.exit_code, 0, res.output)
            self.assertIn("starting reference:", res.output)
            self.assertIn("loaded cohort sample", res.output)
            self.assertIn("PCA fitted 2 components: explained variance", res.output)
            self.assertIn("mask kept", res.output)
            self.assertIn("selecting 5 nearest off-contig bins", res.output)
            self.assertIn("null models computed for", res.output)
            self.assertIn("wrote reference parquet", res.output)
            self.assertIn("evaluating reference QC", res.output)
            self.assertIn("branch autosomal QC:", res.output)
            self.assertIn("reference completed in", res.output)

            # Test standalone ref-qc debug logging
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
            samples, ref_path = _build_test_cohort(root, n_samples=4)
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
            self.assertIn("segmenting 2 contigs with binseg", res.output)
            self.assertIn("segmentation finished:", res.output)
            self.assertIn("called", res.output)
            self.assertIn("writing prediction outputs", res.output)
            self.assertIn("wrote interactive genome HTML plot", res.output)
            self.assertIn("predict completed in", res.output)
