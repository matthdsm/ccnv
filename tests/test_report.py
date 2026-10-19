import base64
import gzip
import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pysam

from ccnv.report import (
    BINS_BED_HEADER,
    CALLS_BED_HEADER,
    PLOTLY_CDN_URL,
    SEGMENTS_BED_HEADER,
    natural_sort_key,
    read_cytobands,
    write_genome_plot,
    write_outputs,
)
from ccnv.results import BinResult, SegmentResult


def _extract_payload(html: str) -> dict[str, object]:
    match = re.search(
        r'<script id="ccnv-data" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    if match is None:
        raise AssertionError("ccnv-data JSON payload not found in HTML")
    return json.loads(match.group(1))


class TestReport(unittest.TestCase):
    def test_write_genome_plot_html_and_binary_payload(self) -> None:
        contig_lengths = {"chr1": 100_000, "chr2": 80_000, "chr3": 60_000}
        bins = (
            BinResult(
                branch="autosomal",
                contigs=("chr1", "chr1", "chr2", "chr3"),
                starts=np.array([0, 50_000, 0, 0], dtype=np.int64),
                ends=np.array([50_000, 100_000, 80_000, 60_000], dtype=np.int64),
                ratios=np.array([0.5, 0.0, -0.6, 0.1], dtype=np.float64),
                zscores=np.array([3.0, 0.0, -3.5, 0.5], dtype=np.float64),
                weights=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
                reference_sizes=np.array([10, 10, 10, 10], dtype=np.int64),
                masked=np.array([False, True, False, False], dtype=bool),
            ),
        )
        segments = (
            SegmentResult("chr1", 0, 50_000, 0.5, 3.0, 1.0, 1, "gain"),
            SegmentResult("chr2", 0, 80_000, -0.6, -3.5, 1.0, 1, "loss"),
            SegmentResult("chr3", 0, 60_000, 0.1, 0.5, 1.0, 1, None),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "test_plot.html"
            write_genome_plot(
                html_path,
                bins,
                segments,
                contig_lengths,
                z_threshold=3.0,
                ratio_threshold=0.25,
                sample_name="sample_alpha",
            )
            self.assertTrue(html_path.is_file())
            content = html_path.read_text(encoding="utf-8")

        self.assertIn(PLOTLY_CDN_URL, content)
        self.assertIn('id="coord-input"', content)
        self.assertIn('id="z-input"', content)
        self.assertIn('id="r-input"', content)
        self.assertIn("Plotly.newPlot", content)
        self.assertIn("scattergl", content)

        payload = _extract_payload(content)
        self.assertEqual(payload["sample"], "sample_alpha")
        self.assertEqual(payload["contigs"], ["chr1", "chr2", "chr3"])
        self.assertEqual(payload["lengths"], [100_000, 80_000, 60_000])
        self.assertEqual(payload["z_thresh"], 3.0)
        self.assertEqual(payload["r_thresh"], 0.25)
        self.assertEqual(payload["n_bins"], 3)

        bins_dict = payload["bins"]
        assert isinstance(bins_dict, dict)
        c_arr = np.frombuffer(base64.b64decode(bins_dict["c"]), dtype="<u2")
        s_arr = np.frombuffer(base64.b64decode(bins_dict["s"]), dtype="<i4")
        e_arr = np.frombuffer(base64.b64decode(bins_dict["e"]), dtype="<i4")
        r_arr = np.frombuffer(base64.b64decode(bins_dict["r"]), dtype="<f4")
        z_arr = np.frombuffer(base64.b64decode(bins_dict["z"]), dtype="<f4")
        np.testing.assert_array_equal(c_arr, [0, 1, 2])
        np.testing.assert_array_equal(s_arr, [0, 0, 0])
        np.testing.assert_array_equal(e_arr, [50_000, 80_000, 60_000])
        np.testing.assert_allclose(r_arr, [0.5, -0.6, 0.1], rtol=1e-5)
        np.testing.assert_allclose(z_arr, [3.0, -3.5, 0.5], rtol=1e-5)

        self.assertEqual(
            payload["segments"],
            [
                [0, 0, 50_000, 0.5, 3.0, 1],
                [1, 0, 80_000, -0.6, -3.5, 1],
                [2, 0, 60_000, 0.1, 0.5, 1],
            ],
        )
        self.assertEqual(payload["cytobands"], [])

    def test_cytobands_parsing_and_plot_embedding(self) -> None:
        contig_lengths = {"1": 100_000, "chr2": 80_000}
        cyto_lines = (
            "#chrom\tchromStart\tchromEnd\tname\tgieStain\n"
            "chr1\t0\t40000\tp36.33\tgneg\n"
            "chr1\t40000\t120000\tp36.32\tacen\n"
            "2\t0\t80000\tq11.1\tgpos75\n"
            "chrUn_1\t0\t1000\tp1\tgneg\n"
        )
        bins = (
            BinResult(
                branch="autosomal",
                contigs=("1", "chr2"),
                starts=np.array([0, 0], dtype=np.int64),
                ends=np.array([100_000, 80_000], dtype=np.int64),
                ratios=np.array([0.0, 0.0], dtype=np.float64),
                zscores=np.array([0.0, 0.0], dtype=np.float64),
                weights=np.array([1.0, 1.0], dtype=np.float64),
                reference_sizes=np.array([10, 10], dtype=np.int64),
                masked=np.array([False, False], dtype=bool),
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cyto_gz = root / "cytoBand.txt.gz"
            with gzip.open(cyto_gz, "wt", encoding="utf-8") as handle:
                handle.write(cyto_lines)

            parsed = read_cytobands(cyto_gz, contig_lengths)
            self.assertEqual(len(parsed), 3)
            self.assertEqual(parsed[0].contig, "1")
            self.assertEqual((parsed[0].start, parsed[0].end), (0, 40_000))
            self.assertEqual(parsed[1].contig, "1")
            self.assertEqual((parsed[1].start, parsed[1].end), (40_000, 100_000))
            self.assertEqual(parsed[1].stain, "acen")
            self.assertEqual(parsed[2].contig, "chr2")
            self.assertEqual(parsed[2].name, "q11.1")

            html_path = root / "cyto_plot.html"
            write_genome_plot(
                html_path,
                bins,
                (),
                contig_lengths,
                cytobands=cyto_gz,
            )
            payload = _extract_payload(html_path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["cytobands"],
            [
                [0, 0, 40_000, "p36.33", "gneg"],
                [0, 40_000, 100_000, "p36.32", "acen"],
                [1, 0, 80_000, "q11.1", "gpos75"],
            ],
        )

    def test_write_outputs_creates_all_artifacts(self) -> None:
        contig_lengths = {"chr1": 100_000}
        bins = (
            BinResult(
                branch="autosomal",
                contigs=("chr1",),
                starts=np.array([0], dtype=np.int64),
                ends=np.array([100_000], dtype=np.int64),
                ratios=np.array([0.0], dtype=np.float64),
                zscores=np.array([0.0], dtype=np.float64),
                weights=np.array([1.0], dtype=np.float64),
                reference_sizes=np.array([10], dtype=np.int64),
                masked=np.array([False], dtype=bool),
            ),
        )
        segments = (SegmentResult("chr1", 0, 100_000, 0.0, 0.0, 1.0, 1, None),)

        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = Path(tmpdir) / "sample_out"
            outputs = write_outputs(prefix, bins, segments, contig_lengths, plot=True)
            self.assertTrue(outputs.bins_bed.is_file())
            self.assertTrue(outputs.segments_bed.is_file())
            self.assertTrue(outputs.calls_bed.is_file())
            self.assertTrue(Path(str(outputs.bins_bed) + ".tbi").is_file())
            self.assertTrue(Path(str(outputs.segments_bed) + ".tbi").is_file())
            self.assertTrue(Path(str(outputs.calls_bed) + ".tbi").is_file())
            self.assertIsNotNone(outputs.bins_vcf)
            self.assertTrue(outputs.bins_vcf.is_file())
            self.assertTrue(Path(str(outputs.bins_vcf) + ".tbi").is_file())
            self.assertIsNotNone(outputs.segments_vcf)
            self.assertTrue(outputs.segments_vcf.is_file())
            self.assertTrue(Path(str(outputs.segments_vcf) + ".tbi").is_file())
            self.assertIsNotNone(outputs.calls_vcf)
            self.assertTrue(outputs.calls_vcf.is_file())
            self.assertTrue(Path(str(outputs.calls_vcf) + ".tbi").is_file())
            self.assertFalse(Path(f"{prefix}_bins.ccnv").exists())
            self.assertFalse(Path(f"{prefix}_segments.ccnv").exists())
            self.assertIsNotNone(outputs.plot_html)
            self.assertTrue(outputs.plot_html.is_file())
            self.assertEqual(outputs.plot_html.suffix, ".html")

            with gzip.open(outputs.bins_bed, "rt") as h:
                bins_lines = h.read().splitlines()
            self.assertEqual(tuple(bins_lines[0].split("\t")), BINS_BED_HEADER)
            with gzip.open(outputs.segments_bed, "rt") as h:
                seg_lines = h.read().splitlines()
            self.assertEqual(tuple(seg_lines[0].split("\t")), SEGMENTS_BED_HEADER)
            with gzip.open(outputs.calls_bed, "rt") as h:
                call_lines = h.read().splitlines()
            self.assertEqual(tuple(call_lines[0].split("\t")), CALLS_BED_HEADER)

            with gzip.open(outputs.bins_vcf, "rt") as h:
                bins_vcf_lines = h.read().splitlines()
            self.assertTrue(bins_vcf_lines[0].startswith("##fileformat=VCFv4.2"))
            chrom_lines = [line for line in bins_vcf_lines if line.startswith("#CHROM")]
            self.assertEqual(len(chrom_lines), 1)
            self.assertTrue(
                chrom_lines[0].startswith(
                    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample_out"
                )
            )

            with pysam.TabixFile(str(outputs.bins_bed)) as tb:
                self.assertEqual(list(tb.header), ["\t".join(BINS_BED_HEADER)])
                records = list(tb.fetch("chr1", 0, 100_000))
                self.assertEqual(len(records), 1)

            with pysam.TabixFile(str(outputs.bins_vcf)) as tb:
                records = list(tb.fetch("chr1", 0, 100_000))
                self.assertEqual(len(records), 1)
                fields = records[0].split("\t")
                self.assertEqual(fields[0], "chr1")
                self.assertEqual(fields[1], "1")
                self.assertEqual(fields[4], "<CNV>")
                self.assertEqual(fields[6], "PASS")
                self.assertIn("SVTYPE=CNV", fields[7])
                self.assertEqual(fields[8], "LOG2R:ZSCORE:WEIGHT:REF_SIZE")

            with pysam.VariantFile(str(outputs.bins_vcf)) as vf:
                self.assertEqual(list(vf.header.samples), ["sample_out"])
                self.assertIn("chr1", vf.header.contigs)
                self.assertEqual(vf.header.contigs["chr1"].length, 100_000)
                recs = list(vf.fetch("chr1", 0, 100_000))
                self.assertEqual(len(recs), 1)
                self.assertEqual(recs[0].chrom, "chr1")
                self.assertEqual(recs[0].pos, 1)
                self.assertEqual(recs[0].stop, 100_000)
                self.assertEqual(recs[0].id, "chr1:0-100000")
                self.assertEqual(recs[0].alts, ("<CNV>",))
                self.assertEqual(list(recs[0].filter), ["PASS"])
                self.assertEqual(recs[0].info["SVTYPE"], "CNV")
                self.assertEqual(recs[0].info["BRANCH"], "autosomal")
                self.assertAlmostEqual(recs[0].samples["sample_out"]["LOG2R"], 0.0)
                self.assertEqual(recs[0].samples["sample_out"]["REF_SIZE"], 10)

            with pysam.VariantFile(str(outputs.segments_vcf)) as vf:
                self.assertEqual(list(vf.header.samples), ["sample_out"])
                recs = list(vf.fetch("chr1", 0, 100_000))
                self.assertEqual(len(recs), 1)
                self.assertEqual(recs[0].stop, 100_000)
                self.assertEqual(recs[0].info["SVTYPE"], "CNV")
                self.assertTrue(recs[0].info["IMPRECISE"])
                self.assertEqual(recs[0].samples["sample_out"]["BINS"], 1)

            with pysam.VariantFile(str(outputs.calls_vcf)) as vf:
                self.assertEqual(list(vf.header.samples), ["sample_out"])
                recs = list(vf.fetch("chr1", 0, 100_000))
                self.assertEqual(len(recs), 0)

    def test_write_outputs_vcf_disabled(self) -> None:
        contig_lengths = {"chr1": 100_000}
        bins = (
            BinResult(
                branch="autosomal",
                contigs=("chr1",),
                starts=np.array([0], dtype=np.int64),
                ends=np.array([100_000], dtype=np.int64),
                ratios=np.array([0.0], dtype=np.float64),
                zscores=np.array([0.0], dtype=np.float64),
                weights=np.array([1.0], dtype=np.float64),
                reference_sizes=np.array([10], dtype=np.int64),
                masked=np.array([False], dtype=bool),
            ),
        )
        segments = (SegmentResult("chr1", 0, 100_000, 0.0, 0.0, 1.0, 1, None),)
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = Path(tmpdir) / "no_vcf"
            outputs = write_outputs(
                prefix, bins, segments, contig_lengths, plot=False, vcf=False
            )
            self.assertIsNone(outputs.bins_vcf)
            self.assertIsNone(outputs.segments_vcf)
            self.assertIsNone(outputs.calls_vcf)
            self.assertFalse(Path(f"{prefix}_bins.vcf.gz").exists())
            self.assertFalse(Path(f"{prefix}_bins.vcf.gz.tbi").exists())
            self.assertTrue(outputs.bins_bed.is_file())
            self.assertTrue(Path(f"{prefix}_bins.bed.gz.tbi").is_file())

    def test_natural_sort_key(self) -> None:
        raw = [
            "chr1",
            "chr10",
            "chr11",
            "chr2",
            "chr20",
            "chr22",
            "chr3",
            "chrX",
            "chrY",
        ]
        sorted_contigs = sorted(raw, key=natural_sort_key)
        self.assertEqual(
            sorted_contigs,
            [
                "chr1",
                "chr2",
                "chr3",
                "chr10",
                "chr11",
                "chr20",
                "chr22",
                "chrX",
                "chrY",
            ],
        )

    def test_write_genome_plot_natural_chrom_sort_order(self) -> None:
        contig_lengths = {
            "chr10": 50_000,
            "chr2": 50_000,
            "chr1": 50_000,
            "chrY": 50_000,
            "chrX": 50_000,
        }
        bins = (
            BinResult(
                branch="autosomal",
                contigs=("chr10", "chr2", "chr1", "chrY", "chrX"),
                starts=np.array([0, 0, 0, 0, 0], dtype=np.int64),
                ends=np.array([50_000, 50_000, 50_000, 50_000, 50_000], dtype=np.int64),
                ratios=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
                zscores=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
                weights=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
                reference_sizes=np.array([10, 10, 10, 10, 10], dtype=np.int64),
                masked=np.array([False, False, False, False, False], dtype=bool),
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "natural_plot.html"
            write_genome_plot(html_path, bins, (), contig_lengths)
            payload = _extract_payload(html_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["contigs"], ["chr1", "chr2", "chr10", "chrX", "chrY"])

    def test_write_outputs_naturally_sorts_all_artifacts(self) -> None:
        contig_lengths = {
            "chr10": 50_000,
            "chr2": 50_000,
            "chr1": 50_000,
            "chrY": 50_000,
            "chrX": 50_000,
        }
        bins = (
            BinResult(
                branch="autosomal",
                contigs=("chr10", "chr2", "chr1", "chrY", "chrX"),
                starts=np.array([0, 0, 0, 0, 0], dtype=np.int64),
                ends=np.array([50_000, 50_000, 50_000, 50_000, 50_000], dtype=np.int64),
                ratios=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
                zscores=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
                weights=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
                reference_sizes=np.array([10, 10, 10, 10, 10], dtype=np.int64),
                masked=np.array([False, False, False, False, False], dtype=bool),
            ),
        )
        segments = (
            SegmentResult("chr10", 0, 50_000, 0.4, 3.2, 1.0, 1, "gain"),
            SegmentResult("chr2", 0, 50_000, -0.5, -3.1, 1.0, 1, "loss"),
            SegmentResult("chr1", 0, 50_000, 0.0, 0.0, 1.0, 1, None),
            SegmentResult("chrY", 0, 50_000, 0.0, 0.0, 1.0, 1, None),
            SegmentResult("chrX", 0, 50_000, 0.0, 0.0, 1.0, 1, None),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = Path(tmpdir) / "test_out"
            outputs = write_outputs(prefix, bins, segments, contig_lengths, plot=True)

            # Check bins BED order
            with gzip.open(outputs.bins_bed, "rt") as h:
                bins_lines = h.read().splitlines()[1:]
            bins_contigs = [line.split("\t")[0] for line in bins_lines]
            self.assertEqual(bins_contigs, ["chr1", "chr2", "chr10", "chrX", "chrY"])

            # Check segments BED order
            with gzip.open(outputs.segments_bed, "rt") as h:
                seg_lines = h.read().splitlines()[1:]
            seg_contigs = [line.split("\t")[0] for line in seg_lines]
            self.assertEqual(seg_contigs, ["chr1", "chr2", "chr10", "chrX", "chrY"])

            # Check calls BED order
            with gzip.open(outputs.calls_bed, "rt") as h:
                call_lines = h.read().splitlines()[1:]
            call_contigs = [line.split("\t")[0] for line in call_lines]
            self.assertEqual(call_contigs, ["chr2", "chr10"])

            # Check VCF records and sorting
            self.assertIsNotNone(outputs.bins_vcf)
            with gzip.open(outputs.bins_vcf, "rt") as h:
                vcf_bin_records = [
                    line for line in h.read().splitlines() if not line.startswith("#")
                ]
            self.assertEqual(
                [line.split("\t")[0] for line in vcf_bin_records],
                ["chr1", "chr2", "chr10", "chrX", "chrY"],
            )

            self.assertIsNotNone(outputs.segments_vcf)
            with gzip.open(outputs.segments_vcf, "rt") as h:
                vcf_seg_records = [
                    line for line in h.read().splitlines() if not line.startswith("#")
                ]
            self.assertEqual(
                [line.split("\t")[0] for line in vcf_seg_records],
                ["chr1", "chr2", "chr10", "chrX", "chrY"],
            )
            # chr1: neutral -> <CNV>, chr2: loss -> <DEL>, chr10: gain -> <DUP>
            self.assertEqual(vcf_seg_records[0].split("\t")[4], "<CNV>")
            self.assertEqual(vcf_seg_records[1].split("\t")[4], "<DEL>")
            self.assertEqual(vcf_seg_records[2].split("\t")[4], "<DUP>")

            self.assertIsNotNone(outputs.calls_vcf)
            with gzip.open(outputs.calls_vcf, "rt") as h:
                vcf_call_records = [
                    line for line in h.read().splitlines() if not line.startswith("#")
                ]
            self.assertEqual(
                [line.split("\t")[0] for line in vcf_call_records], ["chr2", "chr10"]
            )
            self.assertEqual(vcf_call_records[0].split("\t")[4], "<DEL>")
            self.assertEqual(vcf_call_records[1].split("\t")[4], "<DUP>")

            # Check returned outputs
            self.assertEqual(
                outputs.bins[0].contigs,
                ("chr1", "chr2", "chr10", "chrX", "chrY"),
            )
            self.assertEqual(
                [s.contig for s in outputs.segments],
                ["chr1", "chr2", "chr10", "chrX", "chrY"],
            )


if __name__ == "__main__":
    unittest.main()
