"""Unit tests for ccnv.report BED, VCF, and output orchestration."""

from __future__ import annotations

import gzip
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pysam
from numpy.typing import ArrayLike

from ccnv.models.results import (
    BinResult,
    FetalFractionResult,
    SCAResult,
    SegmentResult,
)
from ccnv.report import (
    BINS_BED_HEADER,
    CALLS_BED_HEADER,
    PRENATAL_CALLS_BED_HEADER,
    PRENATAL_SEGMENTS_BED_HEADER,
    SEGMENTS_BED_HEADER,
    build_vcf_header,
    natural_sort_key,
    write_bins_bed_streaming,
    write_bins_vcf,
    write_outputs,
    write_segments_vcf,
    write_vcf_streaming,
)
from ccnv.report._common import resolve_segments_dataframe, segments_to_dataframe
from ccnv.report.bed import sort_bin_result, sort_segments


def _make_bin_result(
    branch: str,
    contigs: tuple[str, ...],
    starts: ArrayLike,
    ends: ArrayLike,
    ratios: ArrayLike = 0.2,
    zscores: ArrayLike = 2.0,
    *,
    weights: ArrayLike = 1.0,
    ref_size: ArrayLike = 12,
    masked: ArrayLike = False,
) -> BinResult:
    n = len(contigs)
    return BinResult(
        branch=branch,
        contigs=contigs,
        starts=np.asarray(starts, dtype=np.int64),
        ends=np.asarray(ends, dtype=np.int64),
        ratios=np.broadcast_to(np.asarray(ratios, dtype=np.float64), (n,)).copy(),
        zscores=np.broadcast_to(np.asarray(zscores, dtype=np.float64), (n,)).copy(),
        weights=np.broadcast_to(np.asarray(weights, dtype=np.float64), (n,)).copy(),
        reference_sizes=np.broadcast_to(
            np.asarray(ref_size, dtype=np.int64), (n,)
        ).copy(),
        masked=np.broadcast_to(np.asarray(masked, dtype=np.bool_), (n,)).copy(),
    )


class TestBedReport(unittest.TestCase):
    def test_natural_sort_key(self) -> None:
        contigs = ["chr10", "chr2", "chr1", "chrX", "chrY", "chr21"]
        self.assertEqual(
            sorted(contigs, key=natural_sort_key),
            ["chr1", "chr2", "chr10", "chr21", "chrX", "chrY"],
        )

    def test_sort_bin_result_order(self) -> None:
        bin_res = _make_bin_result(
            "autosomal",
            ("chr2", "chr1"),
            [0, 0],
            [100, 100],
            [0.1, -0.1],
            [1.0, -1.0],
            ref_size=5,
        )
        sorted_res = sort_bin_result(bin_res)
        self.assertEqual(sorted_res.contigs, ("chr1", "chr2"))
        self.assertEqual(sorted_res.ratios[0], -0.1)
        self.assertEqual(sorted_res.ratios[1], 0.1)

    def test_sort_segments_order(self) -> None:
        segs = [
            SegmentResult("chr2", 0, 100, 0.0, 0.0, 1.0, 1, None),
            SegmentResult("chr1", 50, 100, 0.0, 0.0, 1.0, 1, None),
            SegmentResult("chr1", 0, 50, 0.0, 0.0, 1.0, 1, None),
        ]
        sorted_segs = sort_segments(segs)
        self.assertEqual(
            [(s.contig, s.start) for s in sorted_segs],
            [("chr1", 0), ("chr1", 50), ("chr2", 0)],
        )

    def test_write_bins_bed_indexed_and_nan_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bed_path = Path(tmpdir) / "test_bins.bed.gz"
            bins = (
                _make_bin_result(
                    "autosomal",
                    ("chr1", "chr1"),
                    [0, 100],
                    [100, 200],
                    [1.23456789, float("nan")],
                    [2.5, float("nan")],
                    weights=[1.0, 0.0],
                    ref_size=[5, 0],
                    masked=[False, True],
                ),
            )
            out = write_bins_bed_streaming(bed_path, bins, contig_lengths={"chr1": 200})
            self.assertEqual(out, bed_path)
            self.assertTrue(bed_path.is_file())
            self.assertTrue(Path(f"{bed_path}.tbi").is_file())

            with pysam.TabixFile(str(bed_path)) as tbx:
                records = list(tbx.fetch("chr1", 0, 200))
                self.assertEqual(len(records), 2)
                self.assertIn("1.234568", records[0])
                self.assertIn("\t.\t.\t0.000000\t0\ttrue\tautosomal", records[1])

    def test_resolve_segments_dataframe_sorts_prebuilt_dataframe_when_requested(
        self,
    ) -> None:
        segs = (
            SegmentResult("chr10", 0, 5000, 0.4, 4.0, 20.0, 1, "gain"),
            SegmentResult("chr2", 0, 5000, -0.4, -4.0, 20.0, 1, "loss"),
        )
        unsorted_df = segments_to_dataframe(segs, is_prenatal=False)
        sorted_df = resolve_segments_dataframe(
            (), unsorted_df, calls_only=False, is_prenatal=False, sort_natural=True
        )
        self.assertEqual(sorted_df.get_column("contig").to_list(), ["chr2", "chr10"])


class TestVcfReport(unittest.TestCase):
    def test_build_vcf_header(self) -> None:
        header = build_vcf_header(
            contig_lengths={"chr1": 1000, "chr2": 2000},
            sample_name="TEST_SAMPLE",
            record_type="segments",
        )
        self.assertEqual(header[0], "##fileformat=VCFv4.2")
        self.assertTrue(
            any("##contig=<ID=chr1,length=1000>" in line for line in header)
        )
        self.assertTrue(any("##FORMAT=<ID=LOG2R," in line for line in header))
        self.assertTrue(any("##INFO=<ID=SVTYPE," in line for line in header))
        self.assertTrue(header[-1].endswith("\tTEST_SAMPLE"))

    def test_write_segments_vcf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            vcf_path = Path(tmpdir) / "test_segments.vcf.gz"
            header = build_vcf_header({"chr1": 1000}, "SAMPLE1", "segments")
            segs = (SegmentResult("chr1", 100, 500, 0.5, 3.5, 1.2, 4, "gain"),)
            out = write_segments_vcf(vcf_path, header, segs, "SAMPLE1")
            self.assertEqual(out, vcf_path)
            self.assertTrue(vcf_path.is_file())
            self.assertTrue(Path(f"{vcf_path}.tbi").is_file())

    def test_write_bins_vcf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            vcf_path = Path(tmpdir) / "test_bins.vcf.gz"
            header = build_vcf_header({"chr1": 1000}, "SAMPLE1", "bins")
            bins = (
                _make_bin_result(
                    "autosomal", ("chr1",), [0], [100], 0.1, 1.0, ref_size=5
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
                _make_bin_result(
                    "autosomal", ("chr1",), [0], [100], 0.1, 1.0, ref_size=5
                ),
            )
            segs = (SegmentResult("chr1", 0, 100, 0.1, 1.0, 1.0, 1, "gain"),)
            res = write_outputs(
                prefix, bins, segs, contig_lengths={"chr1": 1000}, plot=True, vcf=True
            )
            self.assertTrue(res.bins_bed.is_file())
            self.assertTrue(res.segments_bed.is_file())
            self.assertTrue(res.calls_bed.is_file())
            self.assertIsNone(res.bins_vcf)
            self.assertFalse(Path(f"{prefix}_bins.vcf.gz").exists())
            self.assertIsNotNone(res.segments_vcf)
            self.assertIsNotNone(res.calls_vcf)
            self.assertIsNotNone(res.plot_html)
            if res.segments_vcf is None or res.plot_html is None:
                self.fail("expected segments_vcf and plot_html to be generated")
            self.assertTrue(res.segments_vcf.is_file())
            self.assertTrue(res.plot_html.is_file())

            prefix_bins = Path(tmpdir) / "deliverable_with_bins"
            res_bins = write_outputs(
                prefix_bins,
                bins,
                segs,
                contig_lengths={"chr1": 1000},
                plot=False,
                vcf=True,
                bins_vcf=True,
            )
            self.assertIsNotNone(res_bins.bins_vcf)
            if res_bins.bins_vcf is None:
                self.fail("expected bins_vcf to be generated")
            self.assertTrue(res_bins.bins_vcf.is_file())


class TestReport(unittest.TestCase):
    def test_write_outputs_creates_all_artifacts(self) -> None:
        contig_lengths = {"chr1": 100_000}
        bins = (
            _make_bin_result(
                "autosomal", ("chr1",), [0], [100_000], 0.0, 0.0, ref_size=10
            ),
        )
        segments = (SegmentResult("chr1", 0, 100_000, 0.0, 0.0, 1.0, 1, None),)

        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = Path(tmpdir) / "sample_out"
            outputs = write_outputs(
                prefix, bins, segments, contig_lengths, plot=True, bins_vcf=True
            )
            for bed_p in (outputs.bins_bed, outputs.segments_bed, outputs.calls_bed):
                self.assertTrue(bed_p.is_file())
                self.assertTrue(Path(f"{bed_p}.tbi").is_file())
            if (
                outputs.bins_vcf is None
                or outputs.segments_vcf is None
                or outputs.calls_vcf is None
                or outputs.plot_html is None
            ):
                self.fail("Expected VCF and HTML artifacts to be generated")
            for vcf_p in (outputs.bins_vcf, outputs.segments_vcf, outputs.calls_vcf):
                self.assertTrue(vcf_p.is_file())
                self.assertTrue(Path(f"{vcf_p}.tbi").is_file())
            self.assertFalse(Path(f"{prefix}_bins.ccnv").exists())
            self.assertFalse(Path(f"{prefix}_segments.ccnv").exists())
            self.assertTrue(outputs.plot_html.is_file())
            self.assertEqual(outputs.plot_html.suffix, ".html")

            for bed_p, expected_hdr in (
                (outputs.bins_bed, BINS_BED_HEADER),
                (outputs.segments_bed, SEGMENTS_BED_HEADER),
                (outputs.calls_bed, CALLS_BED_HEADER),
            ):
                with gzip.open(bed_p, "rt") as h:
                    first_line = h.read().splitlines()[0]
                self.assertEqual(tuple(first_line.split("\t")), expected_hdr)

            with gzip.open(outputs.bins_vcf, "rt") as h:
                bins_vcf_lines = h.read().splitlines()
            self.assertTrue(bins_vcf_lines[0].startswith("##fileformat=VCFv4.2"))
            chrom_lines = [ln for ln in bins_vcf_lines if ln.startswith("#CHROM")]
            self.assertEqual(len(chrom_lines), 1)
            self.assertTrue(
                chrom_lines[0].startswith(
                    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample_out"
                )
            )

            with pysam.TabixFile(str(outputs.bins_bed)) as tb:
                self.assertEqual(list(tb.header), ["\t".join(BINS_BED_HEADER)])
                self.assertEqual(len(list(tb.fetch("chr1", 0, 100_000))), 1)

            with pysam.TabixFile(str(outputs.bins_vcf)) as tb:
                records = list(tb.fetch("chr1", 0, 100_000))
                self.assertEqual(len(records), 1)
                fields = records[0].split("\t")
                self.assertEqual(
                    (fields[0], fields[1], fields[4], fields[6]),
                    ("chr1", "1", "<CNV>", "PASS"),
                )
                self.assertIn("SVTYPE=CNV", fields[7])
                self.assertEqual(fields[8], "LOG2R:ZSCORE:WEIGHT:REF_SIZE")

            with pysam.VariantFile(str(outputs.bins_vcf)) as vf:
                self.assertEqual(list(vf.header.samples), ["sample_out"])
                self.assertEqual(vf.header.contigs["chr1"].length, 100_000)
                recs = list(vf.fetch("chr1", 0, 100_000))
                self.assertEqual(len(recs), 1)
                self.assertEqual(
                    (recs[0].chrom, recs[0].pos, recs[0].stop), ("chr1", 1, 100_000)
                )
                self.assertEqual(recs[0].id, "chr1:0-100000")
                self.assertEqual(recs[0].alts, ("<CNV>",))
                self.assertEqual(list(recs[0].filter), ["PASS"])
                self.assertEqual(recs[0].info["SVTYPE"], "CNV")
                self.assertEqual(recs[0].info["BRANCH"], "autosomal")
                self.assertAlmostEqual(recs[0].samples["sample_out"]["LOG2R"], 0.0)
                self.assertEqual(recs[0].samples["sample_out"]["REF_SIZE"], 10)

            with pysam.VariantFile(str(outputs.segments_vcf)) as vf:
                recs = list(vf.fetch("chr1", 0, 100_000))
                self.assertEqual(len(recs), 1)
                self.assertEqual(recs[0].stop, 100_000)
                self.assertEqual(recs[0].info["SVTYPE"], "CNV")
                self.assertTrue(recs[0].info["IMPRECISE"])
                self.assertEqual(recs[0].samples["sample_out"]["BINS"], 1)

            with pysam.VariantFile(str(outputs.calls_vcf)) as vf:
                self.assertEqual(len(list(vf.fetch("chr1", 0, 100_000))), 0)

    def test_write_outputs_vcf_disabled(self) -> None:
        contig_lengths = {"chr1": 100_000}
        bins = (_make_bin_result("autosomal", ("chr1",), [0], [100_000], 0.0, 0.0),)
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
            _make_bin_result(
                "autosomal",
                ("chr1", "chr1"),
                [0, 100_000],
                [100_000, 200_000],
                [0.08, 0.01],
                [3.1, 3.1],
                ref_size=10,
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
                origin_call="FETAL_CONSTITUTIONAL",
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
                origin_call="MATERNAL_UNINHERITED",
            ),
        )
        prenatal = FetalFractionResult(
            "M",
            0.12,
            "FFY",
            0.12,
            0.115,
            0.118,
            0.94,
            0.006,
            0.004,
            SCAResult("XY", False, 0.98, -0.2),
            ff_size=0.125,
            effective_ff=0.12,
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
            self.assertEqual(
                seg_lines[1].split("\t")[10:15],
                ["false", "", "0.350000", "2.450000", "FETAL_CONSTITUTIONAL"],
            )
            self.assertEqual(
                seg_lines[2].split("\t")[10:15],
                [
                    "true",
                    "SUB_FETAL_RATIO",
                    "0.150000",
                    "-3.200000",
                    "MATERNAL_UNINHERITED",
                ],
            )

            with gzip.open(outputs.calls_bed, "rt") as h:
                calls_lines = h.read().splitlines()
            self.assertEqual(
                tuple(calls_lines[0].split("\t")), PRENATAL_CALLS_BED_HEADER
            )
            self.assertEqual(len(calls_lines), 3)

            if outputs.segments_vcf is None:
                self.fail("Expected segments_vcf to be populated")
            with pysam.VariantFile(str(outputs.segments_vcf)) as vf:
                vcf_header_text = str(vf.header)
                for tok in (
                    "##fetalSex=M",
                    "##fetalFraction=0.1200",
                    "##scaCall=XY",
                    "ID=DISCORDANT",
                    "ID=DISC_REASON",
                    "ID=FRAG_SHORT_R",
                    "ID=FRAG_LLR",
                    "ID=ORIGIN",
                    "ID=EXP_LOG2R",
                    "ID=IMPLIED_FF",
                ):
                    self.assertIn(tok, vcf_header_text)
                recs = list(vf.fetch("chr1"))
                self.assertEqual(len(recs), 2)
                self.assertNotIn("DISCORDANT", recs[0].info)
                self.assertEqual(recs[0].info["ORIGIN"], "FETAL_CONSTITUTIONAL")
                self.assertAlmostEqual(recs[0].info["FRAG_SHORT_R"], 0.35, places=2)
                self.assertAlmostEqual(
                    recs[0].samples["prenatal_sample"]["IMPLIED_FF"], 0.115, places=3
                )
                self.assertTrue(recs[1].info["DISCORDANT"])
                self.assertEqual(recs[1].info["DISC_REASON"], "SUB_FETAL_RATIO")
                self.assertEqual(recs[1].info["ORIGIN"], "MATERNAL_UNINHERITED")
                self.assertAlmostEqual(recs[1].info["FRAG_LLR"], -3.2, places=1)
                self.assertAlmostEqual(
                    recs[1].samples["prenatal_sample"]["IMPLIED_FF"], 0.014, places=3
                )

    def test_natural_sort_key(self) -> None:
        raw = ["chr1", "chr10", "chr11", "chr2", "chr20", "chrX", "chrY"]
        exp = ["chr1", "chr2", "chr10", "chr11", "chr20", "chrX", "chrY"]
        self.assertEqual(sorted(raw, key=natural_sort_key), exp)

    def test_write_outputs_naturally_sorts_all_artifacts(self) -> None:
        contig_lengths = {c: 50_000 for c in ("chr10", "chr2", "chr1", "chrY", "chrX")}
        bins = (
            _make_bin_result(
                "autosomal",
                ("chr10", "chr2", "chr1", "chrY", "chrX"),
                [0] * 5,
                [50_000] * 5,
                0.0,
                0.0,
            ),
        )
        segments = (
            SegmentResult("chr10", 0, 50_000, 0.4, 3.2, 1.0, 1, "gain"),
            SegmentResult("chr2", 0, 50_000, -0.5, -3.1, 1.0, 1, "loss"),
            SegmentResult("chr1", 0, 50_000, 0.0, 0.0, 1.0, 1, None),
            SegmentResult("chrY", 0, 50_000, 0.0, 0.0, 1.0, 1, None),
            SegmentResult("chrX", 0, 50_000, 0.0, 0.0, 1.0, 1, None),
        )
        expected_all = ["chr1", "chr2", "chr10", "chrX", "chrY"]

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = write_outputs(
                Path(tmpdir) / "test_out",
                bins,
                segments,
                contig_lengths,
                plot=True,
                bins_vcf=True,
            )
            for bed_p, expected_order in (
                (outputs.bins_bed, expected_all),
                (outputs.segments_bed, expected_all),
                (outputs.calls_bed, ["chr2", "chr10"]),
            ):
                with gzip.open(bed_p, "rt") as h:
                    lines = h.read().splitlines()[1:]
                self.assertEqual([ln.split("\t")[0] for ln in lines], expected_order)

            if (
                outputs.bins_vcf is None
                or outputs.segments_vcf is None
                or outputs.calls_vcf is None
            ):
                self.fail("Expected VCF outputs to be generated")
            with gzip.open(outputs.bins_vcf, "rt") as h:
                vcf_bin_records = [
                    ln for ln in h.read().splitlines() if not ln.startswith("#")
                ]
            self.assertEqual(
                [ln.split("\t")[0] for ln in vcf_bin_records], expected_all
            )

            with gzip.open(outputs.segments_vcf, "rt") as h:
                vcf_seg_records = [
                    ln for ln in h.read().splitlines() if not ln.startswith("#")
                ]
            self.assertEqual(
                [ln.split("\t")[0] for ln in vcf_seg_records], expected_all
            )
            self.assertEqual(
                [vcf_seg_records[i].split("\t")[4] for i in range(3)],
                ["<CNV>", "<DEL>", "<DUP>"],
            )

            with gzip.open(outputs.calls_vcf, "rt") as h:
                vcf_call_records = [
                    ln for ln in h.read().splitlines() if not ln.startswith("#")
                ]
            self.assertEqual(
                [ln.split("\t")[0] for ln in vcf_call_records], ["chr2", "chr10"]
            )
            self.assertEqual(
                [r.split("\t")[4] for r in vcf_call_records], ["<DEL>", "<DUP>"]
            )
            self.assertEqual(outputs.bins[0].contigs, tuple(expected_all))
            self.assertEqual([s.contig for s in outputs.segments], expected_all)


class TestScalableReports(unittest.TestCase):
    def test_default_skips_bins_vcf_and_segments_calls_vcf_parse_in_pysam(self) -> None:
        contig_lengths = {"chr1": 200_000, "chr2": 150_000}
        bins = (
            _make_bin_result(
                "autosomal",
                ("chr1", "chr1", "chr2"),
                [0, 100_000, 0],
                [100_000, 200_000, 150_000],
                [0.45, np.nan, -0.55],
                [4.2, np.inf, -3.8],
                weights=[1.2, 0.0, 1.5],
                ref_size=[12, 0, 14],
                masked=[False, True, False],
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
            outputs = write_outputs(
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
            if outputs.segments_vcf is None or outputs.calls_vcf is None:
                self.fail("Expected segments_vcf and calls_vcf to be populated")

            with pysam.VariantFile(str(outputs.segments_vcf)) as vf:
                recs = list(vf)
                self.assertEqual(len(recs), len(segments))
                self.assertAlmostEqual(
                    recs[0].samples["default_out"]["LOG2R"], 0.45, places=5
                )
                self.assertIsNone(recs[1].samples["default_out"]["LOG2R"])
                self.assertIsNone(recs[1].samples["default_out"]["ZSCORE"])

            with pysam.VariantFile(str(outputs.calls_vcf)) as vf:
                self.assertEqual(len(list(vf)), 2)

            outputs_bins = write_outputs(
                Path(tmpdir) / "opt_in_out",
                bins,
                segments,
                contig_lengths,
                plot=False,
                vcf=True,
                bins_vcf=True,
                threads=2,
            )
            if outputs_bins.bins_vcf is None:
                self.fail("Expected bins_vcf when bins_vcf=True")
            with pysam.VariantFile(str(outputs_bins.bins_vcf)) as vf:
                bin_recs = list(vf)
                self.assertEqual(len(bin_recs), 3)
                self.assertIsNone(bin_recs[1].samples["opt_in_out"]["LOG2R"])
                self.assertEqual(list(bin_recs[1].filter), ["MASKED"])

    def test_streaming_bed_and_vcf_multi_chunk_and_csi_above_512mb(self) -> None:
        n = 70_000
        starts = np.arange(n, dtype=np.int64) * 100
        ratios = np.linspace(-0.5, 0.5, n, dtype=np.float64)
        ratios[65_536] = np.nan
        masked = np.zeros(n, dtype=np.bool_)
        masked[65_536] = True
        bin_res = _make_bin_result(
            "autosomal",
            ("chr1",) * n,
            starts,
            starts + 100,
            ratios,
            ratios * 4.0,
            masked=masked,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bed_gz = root / "chunked_bins.bed.gz"
            self.assertEqual(
                write_bins_bed_streaming(bed_gz, (bin_res,), threads=2), bed_gz
            )
            self.assertTrue(Path(f"{bed_gz}.tbi").is_file())

            with pysam.TabixFile(str(bed_gz)) as tbx:
                hit = list(tbx.fetch("chr1", 65_536 * 100, 65_537 * 100))
                self.assertEqual(len(hit), 1)
                fields = hit[0].split("\t")
                self.assertEqual((fields[4], fields[8]), (".", "true"))
                self.assertEqual(
                    len(list(tbx.fetch("chr1", 69_999 * 100, 70_000 * 100))), 1
                )

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

    def test_interleaved_multibranch_csi_and_custom_vcf_header_order(self) -> None:
        cl = {"chr1": 700_000_000, "chrX": 156_000_000, "scaffold_1": 50_000}
        auto_c = ("chr1", "chr1", "scaffold_1")
        bins = (
            _make_bin_result("autosomal", auto_c, [0, 5000, 0], [2500, 10000, 5000]),
            _make_bin_result("female_sex", ("chrX",), [0], [5000], 0.4, 3.5),
            _make_bin_result("par_shared", ("chr1",), [2500], [5000], 0.15, 1.5),
        )
        segments = (
            SegmentResult("scaffold_1", 0, 5_000, -0.1, -1.0, 1.0, 1, None),
            SegmentResult("chr1", 0, 10_000, 0.5, 4.2, 1.0, 3, "gain"),
            SegmentResult("chrX", 0, 5_000, 0.4, 3.5, 1.0, 1, "gain"),
        )
        custom_contigs = (
            "##contig=<ID=scaffold_1,length=50000>",
            "##contig=<ID=chrX,length=156000000>",
            "##contig=<ID=chr1,length=700000000>",
        )
        nat_order = ["chr1", "chr1", "chr1", "chrX", "scaffold_1"]
        branches_chr1 = ["autosomal", "par_shared", "autosomal"]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bed_path = root / "multi_bins.bed.gz"
            write_bins_bed_streaming(bed_path, bins, contig_lengths=cl)
            self.assertTrue(Path(f"{bed_path}.csi").is_file())
            with gzip.open(bed_path, "rt") as h:
                bed_rows = [ln.split("\t") for ln in h.read().splitlines()[1:]]
            self.assertEqual([r[0] for r in bed_rows], nat_order)
            self.assertEqual([int(r[1]) for r in bed_rows[:3]], [0, 2_500, 5_000])
            self.assertEqual([r[9] for r in bed_rows[:3]], branches_chr1)

            bins_vcf = root / "multi_bins.vcf.gz"
            hdr_b = build_vcf_header(cl, "S1", "bins")
            write_bins_vcf(bins_vcf, hdr_b, bins, "S1", contig_lengths=cl)
            self.assertTrue(Path(f"{bins_vcf}.csi").is_file())
            with pysam.VariantFile(str(bins_vcf)) as vf:
                recs = list(vf)
                self.assertEqual([r.chrom for r in recs], nat_order)
                self.assertEqual([r.info["BRANCH"] for r in recs[:3]], branches_chr1)
                self.assertEqual(len(list(vf.fetch("chrX", 0, 5_000))), 1)

            base_seg = build_vcf_header({}, "S1", "segments")
            c_seg_hdr = (*base_seg[:3], *custom_contigs, *base_seg[3:])
            seg_vcf = root / "custom_order_segments.vcf.gz"
            write_segments_vcf(seg_vcf, c_seg_hdr, segments, "S1", contig_lengths=cl)
            self.assertTrue(Path(f"{seg_vcf}.csi").is_file())
            with pysam.VariantFile(str(seg_vcf)) as vf:
                self.assertEqual([r.chrom for r in vf], ["scaffold_1", "chrX", "chr1"])
                self.assertEqual(len(list(vf.fetch("chr1", 0, 10_000))), 1)

            base_bin = build_vcf_header({}, "S1", "bins")
            c_bin_hdr = (*base_bin[:3], *custom_contigs, *base_bin[3:])
            c_bins_vcf = root / "custom_order_bins.vcf.gz"
            write_bins_vcf(c_bins_vcf, c_bin_hdr, bins, "S1", contig_lengths=cl)
            self.assertTrue(Path(f"{c_bins_vcf}.csi").is_file())
            with pysam.VariantFile(str(c_bins_vcf)) as vf:
                expected_c = ["scaffold_1", "chrX", "chr1", "chr1", "chr1"]
                self.assertEqual([r.chrom for r in vf], expected_c)


if __name__ == "__main__":
    unittest.main()
