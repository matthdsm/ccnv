import base64
import gzip
import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pysam

from ccnv.io.tracks import load_cytobands_track
from ccnv.models.results import (
    BinResult,
    FetalFractionResult,
    SCAResult,
    SegmentResult,
)
from ccnv.report import (
    BINS_BED_HEADER,
    CALLS_BED_HEADER,
    PLOTLY_CDN_URL,
    PRENATAL_CALLS_BED_HEADER,
    PRENATAL_SEGMENTS_BED_HEADER,
    SEGMENTS_BED_HEADER,
    decimate_bins_m4,
    natural_sort_key,
    write_bins_bed_streaming,
    write_deliverables,
    write_genome_plot,
    write_outputs,
    write_vcf_streaming,
)


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
                ratios=np.array([3.5, 0.0, -0.6, 0.1], dtype=np.float64),
                zscores=np.array([3.0, 0.0, -3.5, 0.5], dtype=np.float64),
                weights=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
                reference_sizes=np.array([10, 10, 10, 10], dtype=np.int64),
                masked=np.array([False, True, False, False], dtype=bool),
            ),
        )
        segments = (
            SegmentResult("chr1", 0, 50_000, 3.5, 3.0, 1.0, 1, "gain"),
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
        self.assertIn("const Y_LIM = 2.5;", content)
        self.assertIn("fixedrange: true", content)

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
        np.testing.assert_allclose(r_arr, [3.5, -0.6, 0.1], rtol=1e-5)
        np.testing.assert_allclose(z_arr, [3.0, -3.5, 0.5], rtol=1e-5)

        self.assertEqual(
            payload["segments"],
            [
                [0, 0, 50_000, 3.5, 3.0, 1],
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

            parsed = load_cytobands_track(cyto_gz, contig_lengths)
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
                cytobands=parsed,
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
            outputs = write_outputs(
                prefix, bins, segments, contig_lengths, plot=True, bins_vcf=True
            )
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

    def test_write_outputs_prenatal_mode_beds_and_vcfs(self) -> None:
        contig_lengths = {"chr1": 200_000}
        bins = (
            BinResult(
                branch="autosomal",
                contigs=("chr1", "chr1"),
                starts=np.array([0, 100_000], dtype=np.int64),
                ends=np.array([100_000, 200_000], dtype=np.int64),
                ratios=np.array([0.08, 0.01], dtype=np.float64),
                zscores=np.array([3.1, 3.1], dtype=np.float64),
                weights=np.array([1.0, 1.0], dtype=np.float64),
                reference_sizes=np.array([10, 10], dtype=np.int64),
                masked=np.array([False, False], dtype=bool),
            ),
        )
        segments = (
            SegmentResult(
                "chr1",
                0,
                100_000,
                0.08,
                3.1,
                1.0,
                1,
                "gain",
                expected_ratio=0.084,
                implied_ff=0.115,
                discordant=False,
                frag_short_ratio=0.35,
                frag_llr_size=2.45,
                origin_call="FETAL_CONFIRMED",
            ),
            SegmentResult(
                "chr1",
                100_000,
                200_000,
                0.01,
                3.1,
                1.0,
                1,
                "gain",
                expected_ratio=0.084,
                implied_ff=0.014,
                discordant=True,
                discordance_reason="SUB_FETAL_RATIO",
                frag_short_ratio=0.15,
                frag_llr_size=-3.2,
                origin_call="MATERNAL_CONFIRMED",
            ),
        )
        sca = SCAResult("XY", False, 0.98, -0.2)
        prenatal = FetalFractionResult(
            fetal_sex="M",
            active_ff=0.12,
            active_ff_source="FFY",
            ffy=0.12,
            ffx=0.115,
            ffa=0.118,
            ff_size=0.125,
            chrx_ratio=0.94,
            chry_score=0.006,
            yfrac_threshold=0.004,
            sca=sca,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = Path(tmpdir) / "prenatal_sample"
            outputs = write_outputs(
                prefix,
                bins,
                segments,
                contig_lengths,
                plot=False,
                vcf=True,
                mode="prenatal",
                prenatal_result=prenatal,
            )
            with gzip.open(outputs.segments_bed, "rt") as h:
                seg_lines = h.read().splitlines()
            self.assertEqual(
                tuple(seg_lines[0].split("\t")), PRENATAL_SEGMENTS_BED_HEADER
            )
            row_non_disc = seg_lines[1].split("\t")
            self.assertEqual(row_non_disc[10], "false")
            self.assertEqual(row_non_disc[11], "")
            self.assertEqual(row_non_disc[12], "0.350000")
            self.assertEqual(row_non_disc[13], "2.450000")
            self.assertEqual(row_non_disc[14], "FETAL_CONFIRMED")
            row_disc = seg_lines[2].split("\t")
            self.assertEqual(row_disc[10], "true")
            self.assertEqual(row_disc[11], "SUB_FETAL_RATIO")
            self.assertEqual(row_disc[12], "0.150000")
            self.assertEqual(row_disc[13], "-3.200000")
            self.assertEqual(row_disc[14], "MATERNAL_CONFIRMED")

            with gzip.open(outputs.calls_bed, "rt") as h:
                calls_lines = h.read().splitlines()
            self.assertEqual(
                tuple(calls_lines[0].split("\t")), PRENATAL_CALLS_BED_HEADER
            )
            self.assertEqual(len(calls_lines), 3)

            assert outputs.segments_vcf is not None
            with pysam.VariantFile(str(outputs.segments_vcf)) as vf:
                vcf_header_text = str(vf.header)
                self.assertIn("##fetalSex=M", vcf_header_text)
                self.assertIn("##fetalFraction=0.1200", vcf_header_text)
                self.assertIn("##scaCall=XY", vcf_header_text)
                self.assertIn("##INFO=<ID=DISCORDANT", vcf_header_text)
                self.assertIn("##INFO=<ID=DISC_REASON", vcf_header_text)
                self.assertIn("##INFO=<ID=FRAG_SHORT_R", vcf_header_text)
                self.assertIn("##INFO=<ID=FRAG_LLR", vcf_header_text)
                self.assertIn("##INFO=<ID=ORIGIN", vcf_header_text)
                self.assertIn(
                    'Description="Fragmentomic origin call: FETAL_CONFIRMED, MATERNAL_CONFIRMED, or INDETERMINATE"',
                    vcf_header_text,
                )
                self.assertIn("##FORMAT=<ID=EXP_LOG2R", vcf_header_text)
                self.assertIn("##FORMAT=<ID=IMPLIED_FF", vcf_header_text)

                recs = list(vf.fetch("chr1"))
                self.assertEqual(len(recs), 2)
                self.assertNotIn("DISCORDANT", recs[0].info)
                self.assertAlmostEqual(recs[0].info["FRAG_SHORT_R"], 0.35, places=2)
                self.assertAlmostEqual(recs[0].info["FRAG_LLR"], 2.45, places=2)
                self.assertEqual(recs[0].info["ORIGIN"], "FETAL_CONFIRMED")
                self.assertAlmostEqual(
                    recs[0].samples["prenatal_sample"]["EXP_LOG2R"],
                    0.084,
                    places=3,
                )
                self.assertAlmostEqual(
                    recs[0].samples["prenatal_sample"]["IMPLIED_FF"],
                    0.115,
                    places=3,
                )

                self.assertTrue(recs[1].info["DISCORDANT"])
                self.assertEqual(recs[1].info["DISC_REASON"], "SUB_FETAL_RATIO")
                self.assertAlmostEqual(recs[1].info["FRAG_SHORT_R"], 0.15, places=2)
                self.assertAlmostEqual(recs[1].info["FRAG_LLR"], -3.2, places=1)
                self.assertEqual(recs[1].info["ORIGIN"], "MATERNAL_CONFIRMED")
                self.assertAlmostEqual(
                    recs[0].samples["prenatal_sample"]["EXP_LOG2R"],
                    0.084,
                    places=3,
                )
                self.assertAlmostEqual(
                    recs[0].samples["prenatal_sample"]["IMPLIED_FF"],
                    0.115,
                    places=3,
                )

                self.assertTrue(recs[1].info["DISCORDANT"])

                self.assertEqual(recs[1].info["DISC_REASON"], "SUB_FETAL_RATIO")
                self.assertAlmostEqual(
                    recs[1].samples["prenatal_sample"]["EXP_LOG2R"],
                    0.084,
                    places=3,
                )
                self.assertAlmostEqual(
                    recs[1].samples["prenatal_sample"]["IMPLIED_FF"],
                    0.014,
                    places=3,
                )

    def test_write_genome_plot_with_prenatal_badge_and_discordant_traces(
        self,
    ) -> None:
        contig_lengths = {"chr1": 200_000}
        bins = (
            BinResult(
                branch="autosomal",
                contigs=("chr1",),
                starts=np.array([0], dtype=np.int64),
                ends=np.array([200_000], dtype=np.int64),
                ratios=np.array([0.0], dtype=np.float64),
                zscores=np.array([0.0], dtype=np.float64),
                weights=np.array([1.0], dtype=np.float64),
                reference_sizes=np.array([10], dtype=np.int64),
                masked=np.array([False], dtype=bool),
            ),
        )
        segments = (
            SegmentResult(
                "chr1",
                0,
                200_000,
                0.01,
                3.5,
                1.0,
                1,
                "gain",
                expected_ratio=0.084,
                implied_ff=0.014,
                discordant=True,
                discordance_reason="SUB_FETAL_RATIO",
            ),
        )
        sca = SCAResult("XY", False, 0.98, -0.2)
        prenatal = FetalFractionResult(
            fetal_sex="M",
            active_ff=0.10,
            active_ff_source="FFY",
            ffy=0.10,
            ffx=0.095,
            ffa=0.102,
            chrx_ratio=0.95,
            chry_score=0.005,
            yfrac_threshold=0.004,
            sca=sca,
            low_ff_warning=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "prenatal_plot.html"
            write_genome_plot(
                html_path,
                bins,
                segments,
                contig_lengths,
                prenatal_result=prenatal,
            )
            content = html_path.read_text(encoding="utf-8")

        self.assertIn('id="prenatal-badge"', content)
        self.assertIn("Segments (Discordant)", content)
        self.assertIn("#9333ea", content)
        self.assertIn("diamond", content)
        payload = _extract_payload(content)
        self.assertIn("prenatal", payload)
        prenatal_dict = payload["prenatal"]
        assert isinstance(prenatal_dict, dict)
        self.assertEqual(prenatal_dict["fetal_sex"], "M")
        self.assertEqual(prenatal_dict["active_ff"], 0.10)
        seg_rows = payload["segments"]
        assert isinstance(seg_rows, list)
        self.assertEqual(len(seg_rows[0]), 13)
        self.assertTrue(seg_rows[0][6])
        self.assertEqual(seg_rows[0][9], "SUB_FETAL_RATIO")

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
            outputs = write_outputs(
                prefix, bins, segments, contig_lengths, plot=True, bins_vcf=True
            )

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


class TestScalableReports(unittest.TestCase):
    def test_default_skips_bins_vcf_and_segments_calls_vcf_parse_in_pysam(
        self,
    ) -> None:
        contig_lengths = {"chr1": 200_000, "chr2": 150_000}
        bins = (
            BinResult(
                branch="autosomal",
                contigs=("chr1", "chr1", "chr2"),
                starts=np.array([0, 100_000, 0], dtype=np.int64),
                ends=np.array([100_000, 200_000, 150_000], dtype=np.int64),
                ratios=np.array([0.45, np.nan, -0.55], dtype=np.float64),
                zscores=np.array([4.2, np.inf, -3.8], dtype=np.float64),
                weights=np.array([1.2, 0.0, 1.5], dtype=np.float64),
                reference_sizes=np.array([12, 0, 14], dtype=np.int64),
                masked=np.array([False, True, False], dtype=np.bool_),
            ),
        )
        segments = (
            SegmentResult("chr1", 0, 100_000, 0.45, 4.2, 1.2, 1, "gain"),
            SegmentResult(
                "chr1", 100_000, 200_000, float("nan"), float("inf"), 0.0, 1, None
            ),
            SegmentResult("chr2", 0, 150_000, -0.55, -3.8, 1.5, 1, "loss"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            prefix_default = Path(tmpdir) / "default_out"
            outputs = write_deliverables(
                prefix_default,
                bins,
                segments,
                contig_lengths,
                plot=False,
                vcf=True,
                threads=2,
            )
            self.assertIsNone(outputs.bins_vcf)
            self.assertFalse(Path(f"{prefix_default}_bins.vcf.gz").exists())
            self.assertIsNotNone(outputs.segments_vcf)
            self.assertIsNotNone(outputs.calls_vcf)
            assert outputs.segments_vcf is not None
            assert outputs.calls_vcf is not None

            with pysam.VariantFile(str(outputs.segments_vcf)) as vf:
                recs = list(vf)
                self.assertEqual(len(recs), len(segments))
                self.assertAlmostEqual(
                    recs[0].samples["default_out"]["LOG2R"], 0.45, places=5
                )
                self.assertIsNone(recs[1].samples["default_out"]["LOG2R"])
                self.assertIsNone(recs[1].samples["default_out"]["ZSCORE"])

            with pysam.VariantFile(str(outputs.calls_vcf)) as vf:
                call_recs = list(vf)
                self.assertEqual(len(call_recs), 2)

            prefix_opt_in = Path(tmpdir) / "opt_in_out"
            outputs_bins = write_outputs(
                prefix_opt_in,
                bins,
                segments,
                contig_lengths,
                plot=False,
                vcf=True,
                bins_vcf=True,
                threads=2,
            )
            self.assertIsNotNone(outputs_bins.bins_vcf)
            assert outputs_bins.bins_vcf is not None
            with pysam.VariantFile(str(outputs_bins.bins_vcf)) as vf:
                bin_recs = list(vf)
                self.assertEqual(len(bin_recs), 3)
                self.assertIsNone(bin_recs[1].samples["opt_in_out"]["LOG2R"])
                self.assertEqual(list(bin_recs[1].filter), ["MASKED"])

    def test_streaming_bed_and_vcf_multi_chunk_and_csi_above_512mb(self) -> None:
        n = 70_000
        starts = np.arange(n, dtype=np.int64) * 100
        ends = starts + 100
        ratios = np.linspace(-0.5, 0.5, n, dtype=np.float64)
        ratios[65_536] = np.nan
        zscores = ratios * 4.0
        weights = np.ones(n, dtype=np.float64)
        ref_sizes = np.full(n, 20, dtype=np.int64)
        masked = np.zeros(n, dtype=np.bool_)
        masked[65_536] = True
        bin_res = BinResult(
            branch="autosomal",
            contigs=("chr1",) * n,
            starts=starts,
            ends=ends,
            ratios=ratios,
            zscores=zscores,
            weights=weights,
            reference_sizes=ref_sizes,
            masked=masked,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bed_gz = root / "chunked_bins.bed.gz"
            out_bed = write_bins_bed_streaming(bed_gz, (bin_res,), threads=2)
            self.assertEqual(out_bed, bed_gz)
            self.assertTrue(Path(f"{bed_gz}.tbi").is_file())

            with pysam.TabixFile(str(bed_gz)) as tbx:
                hit = list(tbx.fetch("chr1", 65_536 * 100, 65_537 * 100))
                self.assertEqual(len(hit), 1)
                fields = hit[0].split("\t")
                self.assertEqual(fields[4], "NaN")
                self.assertEqual(fields[8], "true")
                tail = list(tbx.fetch("chr1", 69_999 * 100, 70_000 * 100))
                self.assertEqual(len(tail), 1)

            csi_vcf = root / "giant.vcf.gz"
            hdr = [
                "##fileformat=VCFv4.2",
                '##FILTER=<ID=PASS,Description="All filters passed">',
                "##contig=<ID=chr1,length=700000000>",
                '##ALT=<ID=DUP,Description="Duplication">',
                '##INFO=<ID=END,Number=1,Type=Integer,Description="End">',
                '##INFO=<ID=SVTYPE,Number=1,Type=String,Description="SVType">',
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1",
            ]
            body = [
                b"chr1\t600000001\t.\tN\t<DUP>\t.\tPASS\tEND=600050000;SVTYPE=DUP\t.\t.\n"
            ]
            write_vcf_streaming(csi_vcf, hdr, body, ["chr1"], 700_000_000, threads=2)
            self.assertTrue(Path(f"{csi_vcf}.csi").is_file())
            with pysam.VariantFile(str(csi_vcf)) as vf:
                recs = list(vf.fetch("chr1", 600_010_000, 600_020_000))
                self.assertEqual(len(recs), 1)
                self.assertEqual(recs[0].stop, 600_050_000)

    def test_m4_decimation_caps_500k_bins_while_preserving_outlier_zscores(
        self,
    ) -> None:
        rng = np.random.default_rng(7)
        x = np.arange(500_000, dtype=np.float32) * 1_000.0
        y = rng.normal(0.0, 0.2, size=500_000).astype(np.float32)
        y[10_000:15_000] = np.nan
        z = (y / 0.2).astype(np.float32)
        y[123_456] = 1.3
        z[123_456] = 6.5
        keep = decimate_bins_m4(x, y, z, max_points=48_000, z_preserve=3.0)
        self.assertEqual(keep.dtype, np.int32)
        self.assertLessEqual(keep.size, 55_000)
        self.assertIn(123_456, set(keep.tolist()))
        self.assertTrue(bool(np.all(keep[1:] > keep[:-1])))

    def test_500k_bin_genome_plot_decimates_uses_typed_buffers_lazy_hover_and_debounce(
        self,
    ) -> None:
        n = 500_000
        rng = np.random.default_rng(11)
        starts = np.arange(n, dtype=np.int64) * 100
        ends = starts + 100
        ratios = rng.normal(0.0, 0.15, size=n).astype(np.float64)
        zscores = rng.normal(0.0, 0.8, size=n).astype(np.float64)
        ratios[250_000] = 0.95
        zscores[250_000] = 7.2
        bins = (
            BinResult(
                branch="autosomal",
                contigs=("chr1",) * n,
                starts=starts,
                ends=ends,
                ratios=ratios,
                zscores=zscores,
                weights=np.ones(n, dtype=np.float64),
                reference_sizes=np.full(n, 16, dtype=np.int64),
                masked=np.zeros(n, dtype=np.bool_),
            ),
        )
        contig_lengths = {"chr1": n * 100}

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "big_plot.html"
            write_genome_plot(html_path, bins, (), contig_lengths)
            content = html_path.read_text(encoding="utf-8")

        payload = _extract_payload(content)
        n_serialized = payload["n_bins"]
        assert isinstance(n_serialized, int)
        self.assertLessEqual(n_serialized, 55_000)
        bins_dict = payload["bins"]
        assert isinstance(bins_dict, dict)
        s_arr = np.frombuffer(base64.b64decode(bins_dict["s"]), dtype="<u4")
        self.assertIn(250_000 * 100, set(s_arr.tolist()))
        self.assertIn("Uint32Array", content)
        self.assertIn("Float32Array", content)
        self.assertIn("plotly_hover", content)
        self.assertIn("100", content)
        self.assertNotIn("const binHover = new Array", content)


if __name__ == "__main__":
    unittest.main()
