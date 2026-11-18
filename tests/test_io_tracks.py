"""Tests for genomic track I/O, FASTA/BigWig Rust scanners, and indexed BGZF writers."""

from __future__ import annotations

import gzip
import os
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pysam
from helpers import write_test_bigwig, write_tiny_fasta

from ccnv._core import (
    read_fasta_contigs_core,
    scan_bigwig_means,
    scan_fasta_gc_quantiles,
    write_bgzf_indexed,
)
from ccnv.features import covariates as covariates_mod
from ccnv.features.covariates import mask_covariates, validate_min_mappability
from ccnv.io.tracks import (
    _bin_weighted_mean,
    _load_bundled_hg38_cytobands,
    load_cytobands_track,
    load_default_cytobands_track,
    load_gc_track,
    load_mappability_track,
)
from ccnv.models import CovariateError, Cytoband
from ccnv.report._common import write_bgzf_atomic


def _write_indexed_fasta(path: Path, records: dict[str, str]) -> Path:
    write_tiny_fasta(path, records)
    pysam.faidx(str(path))
    return path


class TestTracksIO(unittest.TestCase):
    def test_load_cytobands_track(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cytobands.bed"
            path.write_text(
                "#chrom\tchromStart\tchromEnd\tname\tgieStain\n"
                "chr1\t0\t2300000\tp36.33\tgneg\n"
                "chr1\t2300000\t5400000\tp36.32\tgpos25\n",
                encoding="utf-8",
            )
            bands = load_cytobands_track(path, {"chr1": 10_000_000})
            self.assertEqual(len(bands), 2)
            self.assertIsInstance(bands[0], Cytoband)
            self.assertEqual(bands[0].contig, "chr1")
            self.assertEqual(bands[0].start, 0)
            self.assertEqual(bands[0].end, 2_300_000)
            self.assertEqual(bands[0].name, "p36.33")
            self.assertEqual(bands[0].stain, "gneg")

    def test_load_default_cytobands_track_hg38_detection_and_scoping(self) -> None:
        _, hg38_lengths, _ = _load_bundled_hg38_cytobands()
        canon_ucsc = {
            f"chr{k}": hg38_lengths[f"chr{k}"] for k in (*range(1, 23), "X", "Y")
        }
        ucsc_bands = load_default_cytobands_track(canon_ucsc)
        if ucsc_bands is None:
            self.fail("expected bundled hg38 cytobands for canonical UCSC lengths")
        self.assertEqual(len(ucsc_bands), 862)
        self.assertEqual(ucsc_bands[0].contig, "chr1")
        self.assertEqual(ucsc_bands[-1].contig, "chrY")

        canon_ensembl = {k.removeprefix("chr"): v for k, v in canon_ucsc.items()}
        ens_bands = load_default_cytobands_track(canon_ensembl)
        if ens_bands is None:
            self.fail("expected bundled hg38 cytobands for Ensembl contig aliases")
        self.assertEqual(len(ens_bands), 862)
        self.assertEqual(ens_bands[0].contig, "1")
        self.assertEqual(ens_bands[-1].contig, "Y")

        subset_bands = load_default_cytobands_track({"chr21": 46_709_983})
        if subset_bands is None:
            self.fail("expected bundled hg38 cytobands for chr21 subset")
        self.assertEqual(len(subset_bands), 14)
        self.assertTrue(all(b.contig == "chr21" for b in subset_bands))

        scoped_bands = load_default_cytobands_track(
            canon_ucsc, target_contigs=("chr21", "chrX")
        )
        if scoped_bands is None:
            self.fail("expected scoped hg38 cytobands")
        self.assertEqual({b.contig for b in scoped_bands}, {"chr21", "chrX"})

        all_contigs_bands = load_default_cytobands_track(
            {
                "chr21": 46_709_983,
                "chrM": 16_569,
                "chr1_KI270706v1_random": hg38_lengths["chr1_KI270706v1_random"],
                "chrEBV": 171_823,
            }
        )
        if all_contigs_bands is None:
            self.fail("expected hg38 cytobands when unannotated decoy contigs exist")
        self.assertEqual(
            {b.contig for b in all_contigs_bands},
            {"chr21", "chrM", "chr1_KI270706v1_random"},
        )

        for non_hg38 in (
            {},
            {"chr1": 249_250_621},
            {"chr1": 248_387_328},
            {"chr21": 46_709_983, "chr23": 50_000_000},
            {"chr21": 46_709_983, "chr22": 999_999},
            {"chrM": 16_569, "chrEBV": 171_823},
            {"chr1": 100_000},
        ):
            with self.subTest(non_hg38=non_hg38):
                self.assertIsNone(load_default_cytobands_track(non_hg38))

        with self.assertRaises(TypeError):
            load_default_cytobands_track(canon_ensembl, target_contigs="21")

        by_contig, frozen_lengths, lookup = _load_bundled_hg38_cytobands()
        self.assertIsInstance(by_contig, MappingProxyType)
        self.assertIsInstance(frozen_lengths, MappingProxyType)
        self.assertIsInstance(lookup.lower_normalized, MappingProxyType)

        with tempfile.TemporaryDirectory() as tmpdir:
            bad_deflate = Path(tmpdir) / "corrupt_deflate.txt.gz"
            valid_gz = bytearray(gzip.compress(b"chr1\t0\t1000\tp11\tgneg\n"))
            valid_gz[10] = 0xFF
            bad_deflate.write_bytes(bytes(valid_gz))
            with self.assertRaises(CovariateError):
                load_cytobands_track(bad_deflate, {"chr1": 1000})


class TestTrackLoadingAndMasking(unittest.TestCase):
    def test_bin_weighted_mean_full_bin_vs_covered_len(self) -> None:
        starts = np.array([0, 100, 200], dtype=np.int64)
        ends = np.array([100, 200, 300], dtype=np.int64)
        iv_s = np.array([0], dtype=np.int64)
        iv_e = np.array([40], dtype=np.int64)
        iv_v = np.array([0.5], dtype=np.float64)

        map_mean = _bin_weighted_mean(
            starts, ends, iv_s, iv_e, iv_v, full_bin_denominator=True
        )
        np.testing.assert_allclose(map_mean, [0.2, 0.0, 0.0], atol=1e-6)

        gc_mean = _bin_weighted_mean(
            starts, ends, iv_s, iv_e, iv_v, full_bin_denominator=False
        )
        self.assertAlmostEqual(float(gc_mean[0]), 0.5)
        self.assertTrue(bool(np.isnan(gc_mean[1])))
        self.assertTrue(bool(np.isnan(gc_mean[2])))

    def test_load_mappability_bigwig_and_chr_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bw_path = write_test_bigwig(
                Path(tmpdir) / "map.bw",
                {"1": 300, "2": 400},
                [
                    ("1", 0, 50, 1.0),
                    ("1", 50, 100, 0.5),
                    ("1", 100, 200, 0.2),
                    ("1", 200, 300, 0.9),
                    ("2", 0, 200, 0.8),
                    ("2", 200, 250, 0.6),
                ],
            )
            contigs = ["chr1", "chr1", "chr1", "chr2", "chr2", "chr2", "chr2"]
            starts = np.array([0, 100, 200, 0, 100, 200, 300], dtype=np.int64)
            ends = np.array([100, 200, 300, 100, 200, 300, 400], dtype=np.int64)

            values = load_mappability_track(bw_path, contigs, starts, ends)
            np.testing.assert_allclose(
                values, [0.75, 0.2, 0.9, 0.8, 0.8, 0.3, 0.0], atol=1e-6
            )

            usable = np.ones(7, dtype=np.bool_)
            masked = mask_covariates(usable, mappability=values, min_mappability=0.5)
            np.testing.assert_array_equal(
                masked, [True, False, True, True, True, False, False]
            )

    def test_load_gc_from_fasta_bed_and_bigwig(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seq_chr1 = (
                ("G" * 40 + "A" * 60) + ("C" * 70 + "T" * 30) + ("N" * 80 + "G" * 20)
            )
            fasta = _write_indexed_fasta(root / "genome.fa", {"chr1": seq_chr1})
            contigs = ["chr1", "chr1", "chr1"]
            starts = np.array([0, 100, 200], dtype=np.int64)
            ends = np.array([100, 200, 300], dtype=np.int64)

            gc_fa, n_mask_fa, gc_q_fa = load_gc_track(fasta, contigs, starts, ends)
            np.testing.assert_allclose(gc_fa[:2], [0.4, 0.7], atol=1e-6)
            self.assertTrue(bool(np.isnan(gc_fa[2])))
            np.testing.assert_array_equal(n_mask_fa, [False, False, True])
            if gc_q_fa is None:
                self.fail("expected non-null gc_q_fa from FASTA track")
            self.assertEqual(gc_q_fa.shape, (3, 8))

            bed_gz = root / "gc.bed.gz"
            with gzip.open(bed_gz, "wt", encoding="utf-8") as handle:
                handle.write("#chrom\tstart\tend\tgc\n")
                handle.write("chr1\t0\t100\t40.0\n")
                handle.write("chr1\t100\t200\t70.0\n")
            gc_bed, n_mask_bed, gc_q_bed = load_gc_track(bed_gz, contigs, starts, ends)
            np.testing.assert_allclose(gc_bed[:2], [0.4, 0.7], atol=1e-6)
            self.assertTrue(bool(np.isnan(gc_bed[2])))
            np.testing.assert_array_equal(n_mask_bed, [False, False, True])
            self.assertIsNone(gc_q_bed)

            bw_gc = write_test_bigwig(
                root / "gc.bw",
                {"chr1": 300},
                [("chr1", 0, 100, 0.4), ("chr1", 100, 200, 0.7)],
            )
            gc_bw, n_mask_bw, gc_q_bw = load_gc_track(bw_gc, contigs, starts, ends)
            np.testing.assert_allclose(gc_bw[:2], [0.4, 0.7], atol=1e-6)
            np.testing.assert_array_equal(n_mask_bw, [False, False, True])
            self.assertIsNone(gc_q_bw)

    def test_validate_min_mappability_rejects_invalid_values(self) -> None:
        check_fn: Callable[..., None] = validate_min_mappability
        for bad in (-0.1, 1.1, float("nan"), float("inf"), True):
            with self.subTest(bad=bad), self.assertRaises(CovariateError):
                check_fn(bad)
        self.assertFalse(hasattr(covariates_mod, "build_exogenous_covariate_basis"))
        self.assertFalse(hasattr(covariates_mod, "fit_poisson_covariate_trends"))

    def test_load_mappability_track_rejects_percent_scaled_bigwig(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bad_bw_path = write_test_bigwig(
                tmp / "percent_map.bw",
                {"chr1": 10_000},
                [("chr1", 0, 10_000, 95.0)],
            )
            contigs = ("chr1", "chr1")
            starts = np.array([0, 5000], dtype=np.int64)
            ends = np.array([5000, 10_000], dtype=np.int64)
            with self.assertRaisesRegex(CovariateError, r"outside \[0\.0, 1\.0\]"):
                load_mappability_track(bad_bw_path, contigs, starts, ends)

            good_bw_path = write_test_bigwig(
                tmp / "unit_map.bw",
                {"chr1": 10_000},
                [("chr1", 0, 5000, 0.85)],
            )
            loaded = load_mappability_track(good_bw_path, contigs, starts, ends)
            np.testing.assert_allclose(loaded, np.array([0.85, 0.0]), atol=1e-6)


class TestRustCoreTracksAndBgzf(unittest.TestCase):
    def test_fasta_gc_quantiles_bigwig_and_corrupt_single_bin_gc_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seq_bins = [
                ("G" * 2 + "A" * 8) * 500,
                ("G" * 21 + "A" * 29) * 100,
                ("C" * 6 + "T" * 4) * 500,
                ("G" * 8 + "A" * 2) * 500,
            ]
            fa_path = _write_indexed_fasta(root / "gc.fa", {"chr1": "".join(seq_bins)})
            self.assertEqual(read_fasta_contigs_core(str(fa_path)), [("chr1", 20_000)])

            gc_mean, n_frac, gc_q = scan_fasta_gc_quantiles(
                str(fa_path), [("chr1", 20_000)], 5_000
            )
            self.assertEqual(gc_q.shape, (4, 8))
            np.testing.assert_allclose(gc_mean, [0.20, 0.42, 0.60, 0.80], atol=1e-6)
            np.testing.assert_allclose(n_frac, [0.0, 0.0, 0.0, 0.0], atol=1e-6)
            np.testing.assert_allclose(gc_q.sum(axis=1), 1.0, atol=1e-6)
            self.assertAlmostEqual(float(gc_q[0, 0]), 1.0, places=5)
            self.assertAlmostEqual(float(gc_q[1, 3]), 1.0, places=5)
            self.assertAlmostEqual(float(gc_q[2, 6]), 1.0, places=5)
            self.assertAlmostEqual(float(gc_q[3, 7]), 1.0, places=5)

            os.unlink(f"{fa_path}.fai")
            self.assertEqual(read_fasta_contigs_core(str(fa_path)), [("chr1", 20_000)])
            gc_u, nf_u, gq_u = scan_fasta_gc_quantiles(
                str(fa_path), [("chr1", 20_000)], 5_000
            )
            np.testing.assert_allclose(gc_u, gc_mean, atol=1e-6)
            np.testing.assert_allclose(nf_u, n_frac, atol=1e-6)
            np.testing.assert_allclose(gq_u, gc_q, atol=1e-6)

            bw_path = write_test_bigwig(
                root / "track.bw",
                {"chr1": 20_000},
                [
                    ("chr1", 0, 2_500, 1.0),
                    ("chr1", 2_500, 5_000, 0.5),
                    ("chr1", 5_000, 10_000, 0.8),
                ],
            )
            bw_means = scan_bigwig_means(str(bw_path), [("chr1", 20_000)], 5_000)
            self.assertAlmostEqual(float(bw_means[0]), 0.75, places=5)
            self.assertAlmostEqual(float(bw_means[1]), 0.80, places=5)
            self.assertTrue(bool(np.isnan(bw_means[2])))
            self.assertTrue(bool(np.isnan(bw_means[3])))

            corrupt_bed_path = root / "corrupt_gc.bed"
            corrupt_bed_path.write_text(
                "#chrom\tstart\tend\tgc\n"
                "chr1\t0\t5000\t0.45\n"
                "chr1\t5000\t10000\t1.05\n"
                "chr1\t10000\t15000\t0.50\n"
                "chr1\t15000\t20000\t0.42\n",
                encoding="utf-8",
            )
            contigs = ["chr1", "chr1", "chr1", "chr1"]
            starts = np.array([0, 5_000, 10_000, 15_000], dtype=np.int64)
            ends = np.array([5_000, 10_000, 15_000, 20_000], dtype=np.int64)
            gc_track, n_mask, gc_q_bed = load_gc_track(
                corrupt_bed_path, contigs, starts, ends
            )
            self.assertIsNone(gc_q_bed)
            self.assertTrue(bool(np.isnan(gc_track[1])))
            self.assertTrue(bool(n_mask[1]))
            self.assertGreater(float(gc_track[0]), 0.35)
            self.assertLess(float(gc_track[0]), 0.65)

    def test_bgzf_indexed_emits_tbi_readable_by_pysam_and_csi_above_512mb(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bed_gz = root / "test.bed.gz"
            rows = [b"#contig\tstart\tend\tval\n"]
            for idx in range(5_000):
                s = idx * 100
                e = s + 100
                rows.append(f"chr1\t{s}\t{e}\t{idx * 0.125:.6f}\n".encode())
            payload = b"".join(rows)
            chunks = [payload[i : i + 17_777] for i in range(0, len(payload), 17_777)]
            idx_tbi = write_bgzf_indexed(
                str(bed_gz), "bed", ["chr1"], 600_000, chunks, 2
            )
            self.assertTrue(idx_tbi.endswith(".tbi"))
            with pysam.TabixFile(str(bed_gz)) as tbx:
                self.assertEqual(
                    list(tbx.fetch("chr1", 0, 100)),
                    ["chr1\t0\t100\t0.000000"],
                )
                self.assertEqual(
                    list(tbx.fetch("chr1", 499_900, 500_000)),
                    [f"chr1\t499900\t500000\t{4999 * 0.125:.6f}"],
                )

            vcf_gz = root / "test.vcf.gz"
            vcf_bytes = (
                b"##fileformat=VCFv4.2\n"
                b"##contig=<ID=chr1,length=20000>\n"
                b'##ALT=<ID=DUP,Description="Duplication">\n'
                b'##FILTER=<ID=PASS,Description="All filters passed">\n'
                b'##INFO=<ID=SVTYPE,Number=1,Type=String,Description="SV type">\n'
                b'##INFO=<ID=END,Number=1,Type=Integer,Description="End pos">\n'
                b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                b"chr1\t1001\t.\tN\t<DUP>\t.\tPASS\tSVTYPE=DUP;END=15000\n"
            )
            idx_vcf = write_bgzf_indexed(
                str(vcf_gz), "vcf", ["chr1"], 20_000, [vcf_bytes], 1
            )
            self.assertTrue(idx_vcf.endswith(".tbi"))
            with pysam.VariantFile(str(vcf_gz)) as vf:
                fetched = list(vf.fetch("chr1", 7_000, 8_000))
                self.assertEqual(len(fetched), 1)
                self.assertEqual(fetched[0].stop, 15_000)

            giant_gz = root / "giant.bed.gz"
            idx_csi = write_bgzf_indexed(
                str(giant_gz),
                "bed",
                ["chr1"],
                700_000_000,
                [b"#contig\tstart\tend\nchr1\t600000000\t600005000\n"],
                1,
            )
            self.assertTrue(idx_csi.endswith(".csi"))
            with pysam.TabixFile(str(giant_gz), index=idx_csi) as tbx_csi:
                self.assertEqual(
                    list(tbx_csi.fetch("chr1", 600_000_000, 600_005_000)),
                    ["chr1\t600000000\t600005000"],
                )

    def test_bgzf_parallel_batch_boundary_crossing_and_canonical_eof_frames(
        self,
    ) -> None:
        bgzf_eof = (
            b"\x1f\x8b\x08\x04\x00\x00\x00\x00\x00\xff"
            b"\x06\x00\x42\x43\x02\x00\x1b\x00\x03\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bed_gz = root / "batch_boundary.bed.gz"
            rows = [b"#contig\tstart\tend\tlog2_ratio\n"]
            boundary_row_idx = 0
            cum_len = len(rows[0])
            for idx in range(80_000):
                s = idx * 100
                e = s + 100
                line = f"chr1\t{s}\t{e}\t{idx * 0.001:.6f}\n".encode()
                if cum_len < 2_088_960 <= cum_len + len(line):
                    boundary_row_idx = idx
                cum_len += len(line)
                rows.append(line)
            self.assertGreater(cum_len, 2_400_000)
            self.assertGreater(boundary_row_idx, 0)

            payload = b"".join(rows)
            chunks = [payload[i : i + 31_337] for i in range(0, len(payload), 31_337)]
            idx_tbi = write_bgzf_indexed(
                str(bed_gz), "bed", ["chr1"], 10_000_000, chunks, 2
            )
            self.assertTrue(idx_tbi.endswith(".tbi"))

            b_start = boundary_row_idx * 100
            b_end = b_start + 100
            with pysam.TabixFile(str(bed_gz)) as tbx:
                self.assertEqual(
                    list(tbx.fetch("chr1", b_start, b_end)),
                    [f"chr1\t{b_start}\t{b_end}\t{boundary_row_idx * 0.001:.6f}"],
                )
                self.assertEqual(
                    list(tbx.fetch("chr1", 7_999_900, 8_000_000)),
                    [f"chr1\t7999900\t8000000\t{79999 * 0.001:.6f}"],
                )

            raw = bed_gz.read_bytes()
            self.assertTrue(raw.endswith(bgzf_eof))
            self.assertFalse(raw.endswith(bgzf_eof * 2))

            pos = 0
            frame_isizes: list[int] = []
            while pos < len(raw):
                self.assertEqual(raw[pos : pos + 4], b"\x1f\x8b\x08\x04")
                bsize = int.from_bytes(raw[pos + 16 : pos + 18], "little") + 1
                isize = int.from_bytes(raw[pos + bsize - 4 : pos + bsize], "little")
                frame_isizes.append(isize)
                pos += bsize
            self.assertEqual(pos, len(raw))
            self.assertGreater(len(frame_isizes), 33)
            self.assertTrue(all(sz > 0 for sz in frame_isizes[:-1]))
            self.assertEqual(frame_isizes[-1], 0)

    def test_load_gc_track_bedgraph_with_ucsc_track_and_browser_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bedgraph_path = root / "ucsc_gc.bedgraph"
            bedgraph_path.write_text(
                "# UCSC GC track\n"
                "browser position chr1:1-20000\n"
                'track type=bedGraph name="GC" description="GC fraction"\n'
                "chr1\t0\t5000\t0.44\n"
                "chr1\t5000\t10000\t0.51\n"
                "chr1\t10000\t15000\t0.39\n"
                "chr1\t15000\t20000\t0.48\tignored_extra\n",
                encoding="utf-8",
            )
            contigs = ["chr1", "chr1", "chr1", "chr1"]
            starts = np.array([0, 5_000, 10_000, 15_000], dtype=np.int64)
            ends = np.array([5_000, 10_000, 15_000, 20_000], dtype=np.int64)
            gc_track, n_mask, gc_q = load_gc_track(bedgraph_path, contigs, starts, ends)
            self.assertIsNone(gc_q)
            np.testing.assert_allclose(gc_track, [0.44, 0.51, 0.39, 0.48], atol=1e-6)
            self.assertFalse(bool(np.any(n_mask)))

    def test_extract_contig_grid_rejects_mismatched_or_nonuniform_ends(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fa_path = _write_indexed_fasta(
                root / "grid.fa",
                {"chr1": "ACGT" * 2_500, "chr2": "GGCC" * 2_500},
            )
            bw_path = write_test_bigwig(
                root / "grid.bw",
                {"chr1": 10_000, "chr2": 10_000},
                [("chr1", 0, 10_000, 0.9), ("chr2", 0, 10_000, 0.8)],
            )

            with self.assertRaises(CovariateError):
                load_gc_track(
                    fa_path,
                    ["chr1", "chr1"],
                    np.array([0, 2_500], dtype=np.int64),
                    np.array([2_500, 7_500], dtype=np.int64),
                )
            with self.assertRaises(CovariateError):
                load_mappability_track(
                    bw_path,
                    ["chr1", "chr1"],
                    np.array([0, 2_500], dtype=np.int64),
                    np.array([2_500, 7_500], dtype=np.int64),
                )

            with self.assertRaises(CovariateError):
                load_mappability_track(
                    bw_path,
                    ["chr1", "chr1", "chr2"],
                    np.array([0, 5_000, 0], dtype=np.int64),
                    np.array([5_000, 10_000, 10_000], dtype=np.int64),
                )

    def test_write_bgzf_atomic_unlinks_stale_opposite_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bed_gz = Path(tmp) / "out.bed.gz"
            write_bgzf_atomic(
                bed_gz,
                "bed",
                ["chr1"],
                100_000,
                [b"#contig\tstart\tend\nchr1\t0\t5000\n"],
                threads=1,
            )
            self.assertTrue(Path(f"{bed_gz}.tbi").exists())
            write_bgzf_atomic(
                bed_gz,
                "bed",
                ["chr1"],
                600_000_000,
                [b"#contig\tstart\tend\nchr1\t0\t5000\n"],
                threads=1,
            )
            self.assertTrue(Path(f"{bed_gz}.csi").exists())
            self.assertFalse(Path(f"{bed_gz}.tbi").exists())

    def test_fasta_bigwig_and_bgzf_signed_parameter_and_overflow_validation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            with self.assertRaises(CovariateError):
                scan_fasta_gc_quantiles(
                    str(tmp_dir / "missing.fa"), [("chr1", 1000)], 0
                )
            with self.assertRaises(CovariateError):
                scan_fasta_gc_quantiles(
                    str(tmp_dir / "missing.fa"), [("chr1", -1)], 5000
                )
            with self.assertRaises(CovariateError):
                scan_bigwig_means(str(tmp_dir / "missing.bw"), [("chr1", 1000)], 0)
            with self.assertRaises(CovariateError):
                scan_bigwig_means(str(tmp_dir / "missing.bw"), [("chr1", -5)], 5000)

            gz_path = str(tmp_dir / "out.vcf.gz")
            with self.assertRaises(ValueError):
                write_bgzf_indexed(gz_path, "vcf", ["chr1"], 1000, [b""], threads=0)
            with self.assertRaises(ValueError):
                write_bgzf_indexed(gz_path, "vcf", ["chr1"], -1, [b""], threads=1)

            overflow_vcf = (
                b"##fileformat=VCFv4.2\n"
                b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                b"chr1\t10\t.\tN\t<DUP>\t.\tPASS\tEND=99999999999999999999999999\n"
            )
            with self.assertRaises(ValueError):
                write_bgzf_indexed(
                    gz_path, "vcf", ["chr1"], 100_000, [overflow_vcf], threads=1
                )

            overflow_bed = b"chr1\t18446744073709551615\t18446744073709551615\n"
            with self.assertRaises(ValueError):
                write_bgzf_indexed(
                    str(tmp_dir / "out.bed.gz"),
                    "bed",
                    ["chr1"],
                    100_000,
                    [overflow_bed],
                    threads=1,
                )


if __name__ == "__main__":
    unittest.main()
