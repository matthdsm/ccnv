"""Unit tests for ccnv.report.html interactive Plotly genome viewer."""

from __future__ import annotations

import base64
import gzip
import json
import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from ccnv.io.tracks import load_cytobands_track
from ccnv.models import BRANCH_AUTOSOMAL
from ccnv.models.results import (
    BinResult,
    FetalFractionResult,
    SCAResult,
    SegmentResult,
)
from ccnv.report import (
    PLOTLY_CDN_URL,
    decimate_bins_m4,
)
from ccnv.report.html import (
    encode_b64,
    render_genome_plot,
)


def _extract_payload(html: str) -> dict[str, object]:
    match = re.search(
        r'<script id="ccnv-data" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    if match is None:
        raise AssertionError("ccnv-data JSON payload not found in HTML")
    loaded = json.loads(match.group(1))
    if not isinstance(loaded, dict):
        raise TypeError("ccnv-data JSON payload is not an object")
    return loaded


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


class TestHtmlReport(unittest.TestCase):
    def test_encode_b64(self) -> None:
        arr = np.array([1, 2, 3], dtype="<i4")
        encoded = encode_b64(arr)
        decoded = np.frombuffer(base64.b64decode(encoded), dtype="<i4")
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
            payload = _extract_payload(content)
            self.assertEqual(payload["sample"], "TEST_SAMPLE")
            self.assertEqual(payload["contigs"], ["chr1"])
            self.assertEqual(payload["r_thresh"], 0.0)
            self.assertEqual(payload["segment_calls"], ["gain"])

    def test_render_genome_plot_html_and_binary_payload(self) -> None:
        contig_lengths = {"chr1": 100_000, "chr2": 80_000, "chr3": 60_000}
        bins = (
            _make_bin_result(
                "autosomal",
                ("chr1", "chr1", "chr2", "chr3"),
                [0, 50_000, 0, 0],
                [50_000, 100_000, 80_000, 60_000],
                [3.5, 0.0, -0.6, 0.1],
                [3.0, 0.0, -3.5, 0.5],
                ref_size=10,
                masked=[False, True, False, False],
            ),
        )
        segments = (
            SegmentResult("chr1", 0, 50_000, 3.5, 3.0, 1.0, 1, "gain"),
            SegmentResult("chr2", 0, 80_000, -0.6, -3.5, 1.0, 1, "loss"),
            SegmentResult("chr3", 0, 60_000, 0.1, 0.5, 1.0, 1, None),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "test_plot.html"
            render_genome_plot(
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

        for token in (
            PLOTLY_CDN_URL,
            'id="coord-input"',
            'id="z-input"',
            'id="r-input"',
            "Plotly.newPlot",
            "scattergl",
            "const Y_LIM = 2.5;",
            "fixedrange: true",
        ):
            self.assertIn(token, content)

        payload = _extract_payload(content)
        self.assertEqual(payload["sample"], "sample_alpha")
        self.assertEqual(payload["contigs"], ["chr1", "chr2", "chr3"])
        self.assertEqual(payload["lengths"], [100_000, 80_000, 60_000])
        self.assertEqual(payload["z_thresh"], 3.0)
        self.assertEqual(payload["r_thresh"], 0.25)
        self.assertEqual(payload["n_bins"], 3)

        bins_dict = payload["bins"]
        if not isinstance(bins_dict, dict):
            self.fail("Expected payload['bins'] to be a dict")
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
        self.assertEqual(payload["segment_calls"], ["gain", "loss", None])
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
            _make_bin_result(
                "autosomal", ("1", "chr2"), [0, 0], [100_000, 80_000], 0.0, 0.0
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cyto_gz = root / "cytoBand.txt.gz"
            with gzip.open(cyto_gz, "wt", encoding="utf-8") as handle:
                handle.write(cyto_lines)

            parsed = load_cytobands_track(cyto_gz, contig_lengths)
            self.assertEqual(len(parsed), 3)
            self.assertEqual(
                (parsed[0].contig, parsed[0].start, parsed[0].end), ("1", 0, 40_000)
            )
            self.assertEqual(
                (parsed[1].contig, parsed[1].start, parsed[1].end, parsed[1].stain),
                ("1", 40_000, 100_000, "acen"),
            )
            self.assertEqual((parsed[2].contig, parsed[2].name), ("chr2", "q11.1"))

            html_path = root / "cyto_plot.html"
            render_genome_plot(html_path, bins, (), contig_lengths, cytobands=parsed)
            payload = _extract_payload(html_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["r_thresh"], 0.0)
        self.assertEqual(
            payload["cytobands"],
            [
                [0, 0, 40_000, "p36.33", "gneg"],
                [0, 40_000, 100_000, "p36.32", "acen"],
                [1, 0, 80_000, "q11.1", "gpos75"],
            ],
        )

    def test_render_genome_plot_with_prenatal_badge_and_discordant_traces(self) -> None:
        contig_lengths = {"chr1": 200_000}
        bins = (_make_bin_result("autosomal", ("chr1",), [0], [200_000], 0.0, 0.0),)
        seg = SegmentResult("chr1", 0, 200_000, 0.01, 3.5, 1.0, 1, "gain")
        segments = (
            SegmentResult(
                seg.contig,
                seg.start,
                seg.end,
                seg.ratio,
                seg.zscore,
                seg.weight,
                seg.bin_count,
                seg.call,
                expected_ratio=0.084,
                implied_ff=0.014,
                discordant=True,
                discordance_reason="SUB_FETAL_RATIO",
            ),
        )
        sca = SCAResult("XY", False, 0.98, -0.2)
        prenatal = FetalFractionResult(
            "M",
            0.10,
            "FFY",
            0.10,
            0.095,
            0.102,
            0.95,
            0.005,
            0.004,
            sca,
            effective_ff=0.14,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "prenatal_plot.html"
            render_genome_plot(
                html_path, bins, segments, contig_lengths, prenatal_result=prenatal
            )
            content = html_path.read_text(encoding="utf-8")

        for token in ('id="prenatal-badge"', "Segments (Discordant)", "#9333ea"):
            self.assertIn(token, content)
        payload = _extract_payload(content)
        self.assertEqual(
            (payload["r_thresh"], payload["segment_calls"]), (0.0, ["gain"])
        )
        prenatal_dict = payload["prenatal"]
        if not isinstance(prenatal_dict, dict):
            self.fail("Expected payload['prenatal'] to be a dict")
        self.assertEqual(
            (
                prenatal_dict["fetal_sex"],
                prenatal_dict["active_ff"],
                prenatal_dict["effective_ff"],
            ),
            ("M", 0.10, 0.14),
        )
        seg_rows = payload["segments"]
        if not isinstance(seg_rows, list):
            self.fail("Expected payload['segments'] to be a list")
        self.assertEqual(
            (len(seg_rows[0]), seg_rows[0][6], seg_rows[0][9]),
            (13, True, "SUB_FETAL_RATIO"),
        )

    def test_render_genome_plot_natural_chrom_sort_order(self) -> None:
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
        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "natural_plot.html"
            render_genome_plot(html_path, bins, (), contig_lengths)
            payload = _extract_payload(html_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["contigs"], ["chr1", "chr2", "chr10", "chrX", "chrY"])

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
        ratios = rng.normal(0.0, 0.15, size=n).astype(np.float64)
        zscores = rng.normal(0.0, 0.8, size=n).astype(np.float64)
        ratios[250_000] = 0.95
        zscores[250_000] = 7.2
        bins = (
            _make_bin_result(
                "autosomal", ("chr1",) * n, starts, starts + 100, ratios, zscores
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "big_plot.html"
            render_genome_plot(html_path, bins, (), {"chr1": n * 100})
            content = html_path.read_text(encoding="utf-8")

        payload = _extract_payload(content)
        n_serialized = payload["n_bins"]
        if not isinstance(n_serialized, int):
            self.fail("Expected payload['n_bins'] to be an int")
        self.assertLessEqual(n_serialized, 55_000)
        bins_dict = payload["bins"]
        if not isinstance(bins_dict, dict):
            self.fail("Expected payload['bins'] to be a dict")
        s_arr = np.frombuffer(base64.b64decode(bins_dict["s"]), dtype="<u4")
        self.assertIn(250_000 * 100, set(s_arr.tolist()))
        for token in ("Uint32Array", "Float32Array", "plotly_hover", "100"):
            self.assertIn(token, content)
        self.assertNotIn("const binHover = new Array", content)

    def test_html_effective_ff_and_xss_escaping(self) -> None:
        bins = (
            BinResult(
                "autosomal",
                ("chr1",),
                np.array([0], dtype=np.int64),
                np.array([50_000], dtype=np.int64),
                np.array([0.12], dtype=np.float64),
                np.array([4.0], dtype=np.float64),
                np.array([1.0], dtype=np.float64),
                np.array([10], dtype=np.int64),
                np.array([False], dtype=np.bool_),
            ),
        )
        base_seg = SegmentResult("chr1", 0, 50_000, 0.12, 4.0, 1.0, 1, "gain")
        seg = replace(
            base_seg,
            expected_ratio=0.08,
            implied_ff=0.18,
            discordant=True,
            discordance_reason="<script>alert(1)</script>",
            frag_short_ratio=0.25,
            frag_llr_size=1.8,
            origin_call="<script>evil()</script>",
        )
        sca = SCAResult("XY", False, 0.99, 0.0)
        prenatal = FetalFractionResult(
            "M", 0.1, "FFY", 0.1, 0.1, 0.1, 0.95, 0.01, 0.004, sca, effective_ff=0.18
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "xss_plot.html"
            render_genome_plot(
                html_path, bins, (seg,), {"chr1": 50_000}, prenatal_result=prenatal
            )
            html = html_path.read_text(encoding="utf-8")

        for forbidden in ("<script>alert(1)</script>", "<script>evil()</script>"):
            self.assertNotIn(forbidden, html)
        for required in (
            "D.prenatal.effective_ff ?? D.prenatal.active_ff",
            "escHtml(originCall)",
            'escHtml(discReason || "YES")',
        ):
            self.assertIn(required, html)

        payload = _extract_payload(html)
        self.assertEqual(payload["segment_calls"], ["gain"])
        p_dict = payload["prenatal"]
        if not isinstance(p_dict, dict):
            self.fail("Expected prenatal payload dict")
        self.assertEqual(p_dict["effective_ff"], 0.18)

    def test_html_sanitizes_non_finite_floats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "test_plot.html"
            bins = (
                BinResult(
                    branch=BRANCH_AUTOSOMAL,
                    contigs=("chr1", "chr1"),
                    starts=np.array([0, 5000], dtype=np.int64),
                    ends=np.array([5000, 10_000], dtype=np.int64),
                    counts=np.array([50.0, 52.0]),
                    masked=np.array([False, False]),
                    ratios=np.array([0.0, 0.02]),
                    zscores=np.array([0.0, 0.2]),
                    weights=np.array([25.0, 25.0]),
                    reference_sizes=np.array([10, 10], dtype=np.int64),
                    expected_counts=np.array([50.0, 50.0]),
                ),
            )
            ff_res = FetalFractionResult(
                fetal_sex="F",
                active_ff=float("nan"),
                active_ff_source="FFA",
                ffy=None,
                ffx=float("inf"),
                ffa=0.08,
                chrx_ratio=1.0,
                chry_score=0.0,
                yfrac_threshold=0.001,
                sca=SCAResult(
                    call="XX", is_aneuploid=False, confidence=0.99, chrx_zscore=0.0
                ),
                low_ff_warning=True,
                warnings=(),
                ff_size=float("-inf"),
            )
            render_genome_plot(
                html_path,
                bins,
                (),
                {"chr1": 10_000},
                prenatal_result=ff_res,
            )
            rendered = html_path.read_text(encoding="utf-8")
            match = re.search(
                r'<script id="ccnv-data" type="application/json">(.*?)</script>',
                rendered,
                re.DOTALL,
            )
            if match is None:
                self.fail("missing ccnv-data script tag in rendered HTML")
            parsed = json.loads(
                match.group(1),
                parse_constant=lambda tok: (_ for _ in ()).throw(ValueError(tok)),
            )
            self.assertIsNone(parsed["prenatal"]["active_ff"])
            self.assertIsNone(parsed["prenatal"]["ffx"])
            self.assertIsNone(parsed["prenatal"]["ff_size"])


if __name__ == "__main__":
    unittest.main()
