import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ccnv.report import (
    BINS_BED_HEADER,
    CALLS_BED_HEADER,
    SEGMENTS_BED_HEADER,
    natural_sort_key,
    write_genome_plot,
    write_outputs,
)
from ccnv.results import BinResult, SegmentResult


class TestReport(unittest.TestCase):
    def test_write_genome_plot_xml_and_elements(self) -> None:
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
                masked=np.array([False, False, False, False], dtype=bool),
            ),
        )
        segments = (
            SegmentResult("chr1", 0, 50_000, 0.5, 3.0, 1.0, 1, "gain"),
            SegmentResult("chr2", 0, 80_000, -0.6, -3.5, 1.0, 1, "loss"),
            SegmentResult("chr3", 0, 60_000, 0.1, 0.5, 1.0, 1, None),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = Path(tmpdir) / "test_plot.svg"
            write_genome_plot(svg_path, bins, segments, contig_lengths)
            self.assertTrue(svg_path.is_file())
            content = svg_path.read_text(encoding="utf-8")

        root = ET.fromstring(content)
        self.assertTrue(root.tag.endswith("svg"))
        self.assertEqual(root.attrib["width"], "1200")
        self.assertEqual(root.attrib["height"], "380")

        # Check alternating background rect for odd contig chr2
        rects = root.findall("{http://www.w3.org/2000/svg}rect")
        self.assertGreaterEqual(len(rects), 2)  # base + chr2
        bg_fills = [r.attrib.get("fill") for r in rects]
        self.assertIn("white", bg_fills)
        self.assertIn("#f5f5f5", bg_fills)

        # Check Y-axis title and X-axis title
        texts = root.findall("{http://www.w3.org/2000/svg}text")
        text_values = [t.text for t in texts]
        self.assertIn("Log2 Ratio", text_values)
        self.assertIn("Chromosome", text_values)
        self.assertIn("chr1", text_values)
        self.assertIn("chr2", text_values)
        self.assertIn("chr3", text_values)

        # Check 45 degree rotation transform on contig labels
        rotated_labels = [
            t for t in texts if "rotate(45)" in t.attrib.get("transform", "")
        ]
        self.assertEqual(len(rotated_labels), 3)

        # Check Y-axis tick values
        for tick in ("-2", "-1", "0", "1", "2"):
            self.assertIn(tick, text_values)

        # Check bin points rendered as single path, not separate circles
        circles = root.findall("{http://www.w3.org/2000/svg}circle")
        self.assertEqual(len(circles), 0)
        paths = root.findall("{http://www.w3.org/2000/svg}path")
        bin_paths = [p for p in paths if p.attrib.get("stroke") == "#4776b4"]
        self.assertEqual(len(bin_paths), 1)
        self.assertIn("h.1", bin_paths[0].attrib["d"])

        # Check segment paths
        gain_paths = [p for p in paths if p.attrib.get("stroke") == "#c43c39"]
        loss_paths = [p for p in paths if p.attrib.get("stroke") == "#3c8c5a"]
        neutral_paths = [p for p in paths if p.attrib.get("stroke") == "#222"]
        self.assertEqual(len(gain_paths), 1)
        self.assertEqual(len(loss_paths), 1)
        self.assertEqual(len(neutral_paths), 1)

    def test_write_genome_plot_deduplicates_overlapping_points(self) -> None:
        contig_lengths = {"chr1": 100_000}
        n_duplicate_bins = 50
        bins = (
            BinResult(
                branch="autosomal",
                contigs=("chr1",) * n_duplicate_bins,
                starts=np.array([10_000] * n_duplicate_bins, dtype=np.int64),
                ends=np.array([20_000] * n_duplicate_bins, dtype=np.int64),
                ratios=np.array([0.0] * n_duplicate_bins, dtype=np.float64),
                zscores=np.array([0.0] * n_duplicate_bins, dtype=np.float64),
                weights=np.array([1.0] * n_duplicate_bins, dtype=np.float64),
                reference_sizes=np.array([10] * n_duplicate_bins, dtype=np.int64),
                masked=np.array([False] * n_duplicate_bins, dtype=bool),
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = Path(tmpdir) / "dedup_plot.svg"
            write_genome_plot(svg_path, bins, (), contig_lengths)
            content = svg_path.read_text(encoding="utf-8")

        root = ET.fromstring(content)
        paths = root.findall("{http://www.w3.org/2000/svg}path")
        bin_path = next(iter([p for p in paths if p.attrib.get("stroke") == "#4776b4"]))
        # Should only contain 1 move command despite 50 identical input bins
        self.assertEqual(bin_path.attrib["d"].count("M"), 1)

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
            self.assertTrue(outputs.bins_parquet.is_file())
            self.assertTrue(outputs.segments_parquet.is_file())
            self.assertIsNotNone(outputs.plot_svg)
            self.assertTrue(outputs.plot_svg.is_file())

            bins_lines = outputs.bins_bed.read_text().splitlines()
            self.assertEqual(tuple(bins_lines[0].split("\t")), BINS_BED_HEADER)
            seg_lines = outputs.segments_bed.read_text().splitlines()
            self.assertEqual(tuple(seg_lines[0].split("\t")), SEGMENTS_BED_HEADER)
            call_lines = outputs.calls_bed.read_text().splitlines()
            self.assertEqual(tuple(call_lines[0].split("\t")), CALLS_BED_HEADER)

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
            svg_path = Path(tmpdir) / "natural_plot.svg"
            write_genome_plot(svg_path, bins, (), contig_lengths)
            content = svg_path.read_text(encoding="utf-8")

        root = ET.fromstring(content)
        texts = [
            t.text
            for t in root.findall("{http://www.w3.org/2000/svg}text")
            if "rotate(45)" in t.attrib.get("transform", "")
        ]
        self.assertEqual(texts, ["chr1", "chr2", "chr10", "chrX", "chrY"])

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
            bins_lines = outputs.bins_bed.read_text().splitlines()[1:]
            bins_contigs = [line.split("\t")[0] for line in bins_lines]
            self.assertEqual(bins_contigs, ["chr1", "chr2", "chr10", "chrX", "chrY"])

            # Check segments BED order
            seg_lines = outputs.segments_bed.read_text().splitlines()[1:]
            seg_contigs = [line.split("\t")[0] for line in seg_lines]
            self.assertEqual(seg_contigs, ["chr1", "chr2", "chr10", "chrX", "chrY"])

            # Check calls BED order
            call_lines = outputs.calls_bed.read_text().splitlines()[1:]
            call_contigs = [line.split("\t")[0] for line in call_lines]
            self.assertEqual(call_contigs, ["chr2", "chr10"])

            # Check bins Parquet order
            bins_tbl = pq.read_table(outputs.bins_parquet)
            self.assertEqual(
                bins_tbl.column("contig").to_pylist(),
                ["chr1", "chr2", "chr10", "chrX", "chrY"],
            )

            # Check segments Parquet order
            seg_tbl = pq.read_table(outputs.segments_parquet)
            self.assertEqual(
                seg_tbl.column("contig").to_pylist(),
                ["chr1", "chr2", "chr10", "chrX", "chrY"],
            )

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
