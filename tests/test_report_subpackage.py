"""Unit tests for ccnv.report subpackage modules."""

import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pysam

from ccnv.models import (
    BinResult,
    SegmentResult,
)
from ccnv.report import write_outputs
from ccnv.report.bed import (
    format_bed_float,
    natural_sort_key,
    sort_bin_result,
    sort_segments,
    write_compressed_bed,
)
from ccnv.report.html import (
    PLOTLY_CDN_URL,
    encode_b64,
    render_genome_plot,
)
from ccnv.report.vcf import (
    build_vcf_header,
    vcf_float,
    write_bins_vcf,
    write_segments_vcf,
)


class TestBedReport(unittest.TestCase):
    def test_natural_sort_key(self) -> None:
        contigs = ["chr10", "chr2", "chr1", "chrX", "chrY", "chr21"]
        sorted_contigs = sorted(contigs, key=natural_sort_key)
        self.assertEqual(
            sorted_contigs, ["chr1", "chr2", "chr10", "chr21", "chrX", "chrY"]
        )

    def test_sort_bin_result_order(self) -> None:
        bin_res = BinResult(
            branch="autosomal",
            contigs=("chr2", "chr1"),
            starts=np.array([0, 0], dtype=np.int64),
            ends=np.array([100, 100], dtype=np.int64),
            ratios=np.array([0.1, -0.1], dtype=np.float64),
            zscores=np.array([1.0, -1.0], dtype=np.float64),
            weights=np.array([1.0, 1.0], dtype=np.float64),
            reference_sizes=np.array([5, 5], dtype=np.int64),
            masked=np.array([False, False], dtype=np.bool_),
        )
        sorted_res = sort_bin_result(bin_res)
        self.assertEqual(sorted_res.contigs, ("chr1", "chr2"))
        self.assertEqual(sorted_res.ratios[0], -0.1)
        self.assertEqual(sorted_res.ratios[1], 0.1)

    def test_sort_segments_order(self) -> None:
        segs = [
            SegmentResult(
                contig="chr2",
                start=0,
                end=100,
                ratio=0.0,
                zscore=0.0,
                weight=1.0,
                bin_count=1,
                call=None,
            ),
            SegmentResult(
                contig="chr1",
                start=50,
                end=100,
                ratio=0.0,
                zscore=0.0,
                weight=1.0,
                bin_count=1,
                call=None,
            ),
            SegmentResult(
                contig="chr1",
                start=0,
                end=50,
                ratio=0.0,
                zscore=0.0,
                weight=1.0,
                bin_count=1,
                call=None,
            ),
        ]
        sorted_segs = sort_segments(segs)
        self.assertEqual(
            [(s.contig, s.start) for s in sorted_segs],
            [("chr1", 0), ("chr1", 50), ("chr2", 0)],
        )

    def test_format_bed_float(self) -> None:
        self.assertEqual(format_bed_float(float("nan")), "NaN")
        self.assertEqual(format_bed_float(1.23456789), "1.2345679")

    def test_write_compressed_bed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bed_path = Path(tmpdir) / "test.bed.gz"
            header = ("#contig", "start", "end", "val")
            rows = [("chr1", 0, 100, 1.5), ("chr1", 100, 200, 2.5)]
            out = write_compressed_bed(bed_path, header, rows)
            self.assertEqual(out, bed_path)
            self.assertTrue(bed_path.is_file())
            self.assertTrue(Path(f"{bed_path}.tbi").is_file())

            # Read back using pysam.TabixFile
            with pysam.TabixFile(str(bed_path)) as tbx:
                self.assertEqual(tbx.header, ["#contig\tstart\tend\tval"])
                records = list(tbx.fetch("chr1", 0, 200))
                self.assertEqual(len(records), 2)


class TestVcfReport(unittest.TestCase):
    def test_vcf_float(self) -> None:
        self.assertIsNone(vcf_float(float("nan")))
        self.assertIsNone(vcf_float(float("inf")))
        self.assertEqual(vcf_float(2.5), 2.5)

    def test_build_vcf_header(self) -> None:
        header = build_vcf_header(
            contig_lengths={"chr1": 1000, "chr2": 2000},
            sample_name="TEST_SAMPLE",
            record_type="segments",
        )
        self.assertIn("TEST_SAMPLE", list(header.samples))
        self.assertIn("chr1", list(header.contigs))
        self.assertIn("LOG2R", list(header.formats))
        self.assertIn("SVTYPE", list(header.info))

    def test_write_segments_vcf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            vcf_path = Path(tmpdir) / "test_segments.vcf.gz"
            header = build_vcf_header(
                contig_lengths={"chr1": 1000},
                sample_name="SAMPLE1",
                record_type="segments",
            )
            segs = (
                SegmentResult(
                    contig="chr1",
                    start=100,
                    end=500,
                    ratio=0.5,
                    zscore=3.5,
                    weight=1.2,
                    bin_count=4,
                    call="gain",
                ),
            )
            out = write_segments_vcf(vcf_path, header, segs, "SAMPLE1")
            self.assertEqual(out, vcf_path)
            self.assertTrue(vcf_path.is_file())
            self.assertTrue(Path(f"{vcf_path}.tbi").is_file())

    def test_write_bins_vcf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            vcf_path = Path(tmpdir) / "test_bins.vcf.gz"
            header = build_vcf_header(
                contig_lengths={"chr1": 1000},
                sample_name="SAMPLE1",
                record_type="bins",
            )
            bins = (
                BinResult(
                    branch="autosomal",
                    contigs=("chr1",),
                    starts=np.array([0], dtype=np.int64),
                    ends=np.array([100], dtype=np.int64),
                    ratios=np.array([0.1], dtype=np.float64),
                    zscores=np.array([1.0], dtype=np.float64),
                    weights=np.array([1.0], dtype=np.float64),
                    reference_sizes=np.array([5], dtype=np.int64),
                    masked=np.array([False], dtype=np.bool_),
                ),
            )
            out = write_bins_vcf(vcf_path, header, bins, "SAMPLE1")
            self.assertEqual(out, vcf_path)
            self.assertTrue(vcf_path.is_file())
            self.assertTrue(Path(f"{vcf_path}.tbi").is_file())


class TestWriteOutputs(unittest.TestCase):
    def test_write_outputs_full(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = Path(tmpdir) / "deliverable"
            bins = (
                BinResult(
                    branch="autosomal",
                    contigs=("chr1",),
                    starts=np.array([0], dtype=np.int64),
                    ends=np.array([100], dtype=np.int64),
                    ratios=np.array([0.1], dtype=np.float64),
                    zscores=np.array([1.0], dtype=np.float64),
                    weights=np.array([1.0], dtype=np.float64),
                    reference_sizes=np.array([5], dtype=np.int64),
                    masked=np.array([False], dtype=np.bool_),
                ),
            )
            segs = (
                SegmentResult(
                    contig="chr1",
                    start=0,
                    end=100,
                    ratio=0.1,
                    zscore=1.0,
                    weight=1.0,
                    bin_count=1,
                    call="gain",
                ),
            )
            res = write_outputs(
                prefix,
                bins,
                segs,
                contig_lengths={"chr1": 1000},
                plot=True,
                vcf=True,
            )
            self.assertTrue(res.bins_bed.is_file())
            self.assertTrue(res.segments_bed.is_file())
            self.assertTrue(res.calls_bed.is_file())
            self.assertIsNotNone(res.bins_vcf)
            self.assertIsNotNone(res.plot_html)
            self.assertTrue(res.bins_vcf.is_file())
            self.assertTrue(res.plot_html.is_file())


class TestHtmlReport(unittest.TestCase):
    def test_encode_b64(self) -> None:
        arr = np.array([1, 2, 3], dtype="<i4")
        encoded = encode_b64(arr)
        decoded = np.frombuffer(__import__("base64").b64decode(encoded), dtype="<i4")
        np.testing.assert_array_equal(arr, decoded)

    def test_render_genome_plot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "test_plot.html"
            bins = (
                BinResult(
                    branch="autosomal",
                    contigs=("chr1", "chr1"),
                    starts=np.array([0, 100], dtype=np.int64),
                    ends=np.array([100, 200], dtype=np.int64),
                    ratios=np.array([0.0, 0.5], dtype=np.float64),
                    zscores=np.array([0.0, 3.2], dtype=np.float64),
                    weights=np.array([1.0, 1.0], dtype=np.float64),
                    reference_sizes=np.array([10, 10], dtype=np.int64),
                    masked=np.array([False, False], dtype=np.bool_),
                ),
            )
            segs = (
                SegmentResult(
                    contig="chr1",
                    start=0,
                    end=200,
                    ratio=0.25,
                    zscore=1.6,
                    weight=1.0,
                    bin_count=2,
                    call="gain",
                ),
            )
            contig_lengths = {"chr1": 1000}
            render_genome_plot(
                html_path,
                bins,
                segs,
                contig_lengths,
                sample_name="TEST_SAMPLE",
            )
            self.assertTrue(html_path.is_file())
            content = html_path.read_text(encoding="utf-8")
            self.assertIn("TEST_SAMPLE", content)
            self.assertIn(PLOTLY_CDN_URL, content)
            match = re.search(
                r'<script id="ccnv-data" type="application/json">(.*?)</script>',
                content,
                re.DOTALL,
            )
            self.assertIsNotNone(match)
            if match:
                payload = json.loads(match.group(1))
                self.assertEqual(payload["sample"], "TEST_SAMPLE")
                self.assertEqual(payload["contigs"], ["chr1"])
