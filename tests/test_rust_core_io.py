"""Parity and edge-case tests for the ccnv._core Rust format I/O bindings."""

from __future__ import annotations

import ast
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyBigWig
import pysam

import ccnv._core as core_mod
from ccnv._core import (
    FRAG_BAND_EDGES,
    GC_QUANTILE_EDGES,
    read_alignment_header,
    read_fasta_contigs_core,
    scan_alignment_bins,
    scan_bigwig_means,
    scan_fasta_gc_quantiles,
    write_bgzf_indexed,
)
from ccnv.io.tracks import load_gc_track, load_mappability_track
from ccnv.models import AlignmentError, CovariateError, ScanAlignmentResult
from ccnv.stats.fragmentomics import encode_kmer_4


def _assert_scan_parity(lhs: ScanAlignmentResult, rhs: ScanAlignmentResult) -> None:
    np.testing.assert_array_equal(lhs["count"], rhs["count"], err_msg="count")
    np.testing.assert_array_equal(
        lhs["paired_count"], rhs["paired_count"], err_msg="paired_count"
    )
    np.testing.assert_array_equal(
        lhs["frag_len_sum"], rhs["frag_len_sum"], err_msg="frag_len_sum"
    )
    np.testing.assert_array_equal(
        lhs["short_frag_count"], rhs["short_frag_count"], err_msg="short_frag_count"
    )
    np.testing.assert_array_equal(
        lhs["endpoint_count"], rhs["endpoint_count"], err_msg="endpoint_count"
    )
    if lhs["frag_bands"] is None or rhs["frag_bands"] is None:
        assert lhs["frag_bands"] is rhs["frag_bands"]
    else:
        np.testing.assert_array_equal(
            lhs["frag_bands"], rhs["frag_bands"], err_msg="frag_bands"
        )
    np.testing.assert_array_equal(
        lhs["hist_autosomal"], rhs["hist_autosomal"], err_msg="hist_autosomal"
    )
    np.testing.assert_array_equal(
        lhs["hist_chrx"], rhs["hist_chrx"], err_msg="hist_chrx"
    )
    np.testing.assert_array_equal(
        lhs["hist_chry"], rhs["hist_chry"], err_msg="hist_chry"
    )
    np.testing.assert_array_equal(
        lhs["motif_counts"], rhs["motif_counts"], err_msg="motif_counts"
    )


def _make_fasta(path: Path, contigs: dict[str, str]) -> Path:
    with path.open("w", encoding="ascii") as handle:
        for name, seq in contigs.items():
            handle.write(f">{name}\n")
            for idx in range(0, len(seq), 80):
                handle.write(seq[idx : idx + 80] + "\n")
    pysam.faidx(str(path))
    return path


def _make_proper_pair(
    header: pysam.AlignmentHeader,
    qname: str,
    contig: str,
    start: int,
    tlen: int,
    ref_seq: str,
    read_len: int = 50,
    mapq: int = 60,
) -> tuple[pysam.AlignedSegment, pysam.AlignedSegment]:
    ref_id = header.get_tid(contig)
    end_excl = min(len(ref_seq), start + tlen)
    mate_start = max(start, end_excl - read_len)

    r1 = pysam.AlignedSegment(header)
    r1.query_name = qname
    r1.flag = 0x1 | 0x2 | 0x20 | 0x40
    r1.reference_id = ref_id
    r1.reference_start = start
    r1.mapping_quality = mapq
    r1_seq = ref_seq[start : start + read_len]
    r1.cigarstring = f"{len(r1_seq)}M"
    r1.next_reference_id = ref_id
    r1.next_reference_start = mate_start
    r1.template_length = tlen
    r1.query_sequence = r1_seq
    r1.query_qualities = pysam.qualitystring_to_array("I" * len(r1_seq))

    r2 = pysam.AlignedSegment(header)
    r2.query_name = qname
    r2.flag = 0x1 | 0x2 | 0x10 | 0x80
    r2.reference_id = ref_id
    r2.reference_start = mate_start
    r2.mapping_quality = mapq
    r2_seq = ref_seq[mate_start : mate_start + read_len]
    r2.cigarstring = f"{len(r2_seq)}M"
    r2.next_reference_id = ref_id
    r2.next_reference_start = start
    r2.template_length = -tlen
    r2.query_sequence = r2_seq
    r2.query_qualities = pysam.qualitystring_to_array("I" * len(r2_seq))
    return r1, r2


class TestRustCoreIO(unittest.TestCase):
    def test_exported_constants_match_v2_spec(self) -> None:
        expected_frag_edges = (
            30,
            50,
            70,
            80,
            90,
            100,
            110,
            120,
            130,
            140,
            150,
            160,
            170,
            180,
            190,
            200,
            210,
            220,
            230,
            240,
            250,
            300,
            350,
            400,
            601,
        )
        expected_gc_edges = (0.0, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.65, 1.0)
        self.assertEqual(FRAG_BAND_EDGES, expected_frag_edges)
        self.assertEqual(len(FRAG_BAND_EDGES), 25)
        self.assertEqual(GC_QUANTILE_EDGES, expected_gc_edges)
        self.assertEqual(len(GC_QUANTILE_EDGES), 9)
        self.assertFalse(hasattr(core_mod, "ScanAlignmentResult"))
        self.assertFalse(hasattr(core_mod, "ScanQcCounts"))
        stub_path = Path(__file__).resolve().parents[1] / "src" / "ccnv" / "_core.pyi"
        stub_tree = ast.parse(stub_path.read_text(encoding="utf-8"))
        stub_all: list[str] = []
        for node in stub_tree.body:
            if (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
                )
                and isinstance(node.value, (ast.List, ast.Tuple))
            ):
                stub_all = [
                    elt.value
                    for elt in node.value.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                ]
        self.assertNotIn("ScanAlignmentResult", stub_all)
        self.assertNotIn("ScanQcCounts", stub_all)

    def test_bam_and_cram_parity_with_pysam_and_mate_dedup_accounting(self) -> None:
        rng = np.random.default_rng(12345)
        alphabet = np.array(list("ACGT"))
        seq_chr1 = "".join(rng.choice(alphabet, size=20_000))
        seq_chrx = "".join(rng.choice(alphabet, size=10_000))
        seq_chry = "".join(rng.choice(alphabet, size=10_000))

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fa_path = _make_fasta(
                root / "ref.fa",
                {"chr1": seq_chr1, "chrX": seq_chrx, "chrY": seq_chry},
            )
            header = pysam.AlignmentHeader.from_dict(
                {
                    "HD": {"VN": "1.6", "SO": "coordinate"},
                    "SQ": [
                        {"SN": "chr1", "LN": 20_000},
                        {"SN": "chrX", "LN": 10_000},
                        {"SN": "chrY", "LN": 10_000},
                    ],
                }
            )

            records: list[pysam.AlignedSegment] = []
            # 1. Inclusive downstream endpoint test: start=4900, tlen=100 -> end_pos=4999 (bin 0, not bin 1)
            r1_a, r2_a = _make_proper_pair(
                header, "p_bin0", "chr1", 4_900, 100, seq_chr1
            )
            records.extend([r1_a, r2_a])
            # 2. Standard nucleosomal pair: start=6000, tlen=166 (bin 1)
            records.extend(
                _make_proper_pair(header, "p_166", "chr1", 6_000, 166, seq_chr1)
            )
            # 3. Sub-30 insert: start=11000, tlen=15 (bin 2; excluded from paired_count/bands/hist)
            records.extend(
                _make_proper_pair(
                    header, "p_short15", "chr1", 11_000, 15, seq_chr1, read_len=15
                )
            )
            # 4. Supra-600 insert: start=16000, tlen=900 (bin 3; excluded from paired_count/bands/hist)
            records.extend(
                _make_proper_pair(header, "p_long900", "chr1", 16_000, 900, seq_chr1)
            )
            # 5. Sex chromosome proper pairs for hist_chrx / hist_chry
            records.extend(
                _make_proper_pair(header, "p_chrx", "chrX", 1_000, 140, seq_chrx)
            )
            records.extend(
                _make_proper_pair(header, "p_chry", "chrY", 2_000, 150, seq_chry)
            )

            # Filtered records on chr1: unmapped, low-MAPQ, duplicate, secondary, supplementary, improper pair
            unmapped = pysam.AlignedSegment(header)
            unmapped.query_name = "flt_unmapped"
            unmapped.flag = 0x4
            unmapped.reference_id = 0
            unmapped.reference_start = 500
            unmapped.mapping_quality = 0
            unmapped.query_sequence = seq_chr1[500:550]
            unmapped.query_qualities = pysam.qualitystring_to_array("I" * 50)
            records.append(unmapped)

            low_mq1, low_mq2 = _make_proper_pair(
                header, "flt_mapq", "chr1", 1_000, 160, seq_chr1, mapq=5
            )
            records.extend([low_mq1, low_mq2])

            dup1, _ = _make_proper_pair(header, "flt_dup", "chr1", 1_500, 160, seq_chr1)
            dup1.flag |= 0x400
            records.append(dup1)

            sec1, _ = _make_proper_pair(header, "flt_sec", "chr1", 2_000, 160, seq_chr1)
            sec1.flag |= 0x100
            records.append(sec1)

            sup1, _ = _make_proper_pair(header, "flt_sup", "chr1", 2_500, 160, seq_chr1)
            sup1.flag |= 0x800
            records.append(sup1)

            imp1, _ = _make_proper_pair(header, "flt_imp", "chr1", 3_000, 160, seq_chr1)
            imp1.flag &= ~0x2
            records.append(imp1)

            records.sort(key=lambda r: (r.reference_id, r.reference_start))

            unsorted_bam = root / "unsorted.bam"
            bam_path = root / "sample.bam"
            cram_path = root / "sample.cram"
            with pysam.AlignmentFile(str(unsorted_bam), "wb", header=header) as out_bam:
                for rec in records:
                    out_bam.write(rec)
            pysam.sort("-o", str(bam_path), str(unsorted_bam))
            pysam.index(str(bam_path))

            with (
                pysam.AlignmentFile(str(bam_path), "rb") as in_bam,
                pysam.AlignmentFile(
                    str(cram_path), "wc", header=header, reference_filename=str(fa_path)
                ) as out_cram,
            ):
                for rec in in_bam:
                    out_cram.write(rec)
            pysam.index(str(cram_path))

            self.assertEqual(
                read_alignment_header(str(bam_path)),
                [("chr1", 20_000), ("chrX", 10_000), ("chrY", 10_000)],
            )
            self.assertEqual(
                read_alignment_header(str(cram_path), str(fa_path)),
                [("chr1", 20_000), ("chrX", 10_000), ("chrY", 10_000)],
            )
            with self.assertRaises(AlignmentError):
                read_alignment_header(str(cram_path), None)

            target_contigs = [("chr1", 20_000), ("chrX", 10_000), ("chrY", 10_000)]
            res_bam = scan_alignment_bins(
                str(bam_path), target_contigs, 5_000, 20, True, 2
            )
            res_bam_t1 = scan_alignment_bins(
                str(bam_path), target_contigs, 5_000, 20, True, 1
            )
            res_cram = scan_alignment_bins(
                str(cram_path), target_contigs, 5_000, 20, True, 2, str(fa_path)
            )
            res_no_bands = scan_alignment_bins(
                str(bam_path), target_contigs, 5_000, 20, True, 1, None, False
            )
            self.assertIsNone(res_no_bands["frag_bands"])

            _assert_scan_parity(res_bam, res_bam_t1)
            _assert_scan_parity(res_bam, res_cram)
            np.testing.assert_allclose(
                res_bam["frag_gc_sum"], res_cram["frag_gc_sum"], atol=1e-6
            )

            self.assertIs(res_bam["paired_end"], True)
            counts = res_bam["count"]
            paired_counts = res_bam["paired_count"]
            frag_len_sum = res_bam["frag_len_sum"]
            short_counts = res_bam["short_frag_count"]
            endpoints = res_bam["endpoint_count"]
            bands = res_bam["frag_bands"]
            assert bands is not None
            hist_auto = res_bam["hist_autosomal"]
            hist_x = res_bam["hist_chrx"]
            hist_y = res_bam["hist_chry"]
            motifs = res_bam["motif_counts"]

            # 8 bins total: 4 on chr1, 2 on chrX, 2 on chrY
            np.testing.assert_array_equal(counts, [1, 1, 1, 1, 1, 0, 1, 0])
            # Only [30, 600] inserts enter paired_count / frag_len_sum / bands
            np.testing.assert_array_equal(paired_counts, [1, 1, 0, 0, 1, 0, 1, 0])
            np.testing.assert_array_equal(
                frag_len_sum, [100, 166, 0, 0, 140, 0, 150, 0]
            )
            np.testing.assert_array_equal(short_counts, [1, 0, 0, 0, 1, 0, 1, 0])
            # Inclusive downstream endpoint: start=4900, tlen=100 -> end_pos=4999 (both in bin 0)
            self.assertEqual(int(endpoints[0]), 2)
            self.assertEqual(int(endpoints[1]), 2)
            self.assertEqual(bands.shape, (8, 24))
            self.assertEqual(int(bands[0, 5]), 1)  # 100 bp in [100, 110)
            self.assertEqual(int(bands[1, 11]), 1)  # 166 bp in [160, 170)
            self.assertEqual(int(np.sum(bands[2])), 0)  # tlen=15 excluded
            self.assertEqual(int(np.sum(bands[3])), 0)  # tlen=900 excluded
            self.assertEqual(int(hist_auto[100 - 30]), 1)
            self.assertEqual(int(hist_auto[166 - 30]), 1)
            self.assertEqual(int(np.sum(hist_auto)), 2)
            self.assertEqual(int(hist_x[140 - 30]), 1)
            self.assertEqual(int(hist_y[150 - 30]), 1)
            # 6 proper pairs -> 12 valid 5' 4-mer motifs across both upstream and downstream mates
            self.assertEqual(int(np.sum(motifs)), 12)
            m_up = encode_kmer_4(r1_a.query_sequence[:4])
            dn_rc = r2_a.query_sequence[-4:][::-1].translate(
                str.maketrans("ACGTacgt", "TGCAtgca")
            )
            m_dn = encode_kmer_4(dn_rc)
            assert m_up is not None and m_dn is not None
            self.assertGreaterEqual(int(motifs[m_up]), 1)
            self.assertGreaterEqual(int(motifs[m_dn]), 1)

            qc = res_bam["qc_counts"]
            self.assertEqual(
                qc["reads_seen"],
                qc["reads_kept"] + qc["mate_dedup_skipped"] + qc["reads_filtered"],
            )
            self.assertEqual(qc["reads_kept"], 6)
            self.assertEqual(qc["mate_dedup_skipped"], 6)
            self.assertEqual(qc["reads_filtered"], 7)

    def test_single_end_fragment_shift_and_unindexed_readonly_behavior(self) -> None:
        tmpdir = self.enterContext(tempfile.TemporaryDirectory())
        root = Path(tmpdir)
        self.addCleanup(os.chmod, root, 0o755)
        header = pysam.AlignmentHeader.from_dict(
            {
                "HD": {"VN": "1.6", "SO": "coordinate"},
                "SQ": [{"SN": "chr1", "LN": 20_000}],
            }
        )
        bam_path = root / "se.bam"
        with pysam.AlignmentFile(str(bam_path), "wb", header=header) as out_bam:
            # Forward read at 4950: with se_fragment_length=160 -> p = 4950 + 80 = 5030 (bin 1)
            fwd = pysam.AlignedSegment(header)
            fwd.query_name = "se_fwd"
            fwd.flag = 0
            fwd.reference_id = 0
            fwd.reference_start = 4_950
            fwd.mapping_quality = 60
            fwd.cigarstring = "50M"
            fwd.query_sequence = "ACGT" * 12 + "AC"
            fwd.query_qualities = pysam.qualitystring_to_array("I" * 50)
            out_bam.write(fwd)

            # Reverse read at 5010..5060: end=5060, p = 5060 - 1 - 80 = 4979 (bin 0)
            rev = pysam.AlignedSegment(header)
            rev.query_name = "se_rev"
            rev.flag = 0x10
            rev.reference_id = 0
            rev.reference_start = 5_010
            rev.mapping_quality = 60
            rev.cigarstring = "50M"
            rev.query_sequence = "GGCC" * 12 + "GG"
            rev.query_qualities = pysam.qualitystring_to_array("I" * 50)
            out_bam.write(rev)

        pysam.index(str(bam_path))
        os.unlink(f"{bam_path}.bai")
        os.chmod(root, 0o555)

        res_unshifted = scan_alignment_bins(
            str(bam_path), [("chr1", 20_000)], 5_000, 20, True, 1, None, False, 0
        )
        self.assertIs(res_unshifted["paired_end"], False)
        np.testing.assert_array_equal(res_unshifted["count"], [1, 1, 0, 0])

        res_shifted = scan_alignment_bins(
            str(bam_path), [("chr1", 20_000)], 5_000, 20, True, 1, None, False, 160
        )
        # Both shifted midpoints (5030 in bin 1 and 4979 in bin 0) swap bins cleanly
        np.testing.assert_array_equal(res_shifted["count"], [1, 1, 0, 0])
        # Forward 5' start=4950 (bin 0), reverse 5' end-1=5059 (bin 1)
        np.testing.assert_array_equal(res_shifted["endpoint_count"], [1, 1, 0, 0])

        with self.assertRaisesRegex(AlignmentError, "index|--threads 1"):
            scan_alignment_bins(str(bam_path), [("chr1", 20_000)], 5_000, 20, True, 4)

    def test_fasta_gc_quantiles_bigwig_and_corrupt_single_bin_gc_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # 4 bins of 5000 bp on chr1:
            # bin 0: 20% GC, bin 1: 42% GC, bin 2: 60% GC, bin 3: 80% GC
            seq_bins = [
                ("G" * 2 + "A" * 8) * 500,
                ("G" * 21 + "A" * 29) * 100,
                ("C" * 6 + "T" * 4) * 500,
                ("G" * 8 + "A" * 2) * 500,
            ]
            fa_path = _make_fasta(root / "gc.fa", {"chr1": "".join(seq_bins)})
            self.assertEqual(read_fasta_contigs_core(str(fa_path)), [("chr1", 20_000)])

            gc_mean, n_frac, gc_q = scan_fasta_gc_quantiles(
                str(fa_path), [("chr1", 20_000)], 5_000
            )
            self.assertEqual(gc_q.shape, (4, 8))
            np.testing.assert_allclose(gc_mean, [0.20, 0.42, 0.60, 0.80], atol=1e-6)
            np.testing.assert_allclose(n_frac, [0.0, 0.0, 0.0, 0.0], atol=1e-6)
            np.testing.assert_allclose(gc_q.sum(axis=1), 1.0, atol=1e-6)
            self.assertAlmostEqual(
                float(gc_q[0, 0]), 1.0, places=5
            )  # 0.20 in [0.0, 0.30)
            self.assertAlmostEqual(
                float(gc_q[1, 3]), 1.0, places=5
            )  # 0.42 in [0.40, 0.45)
            self.assertAlmostEqual(
                float(gc_q[2, 6]), 1.0, places=5
            )  # 0.60 in [0.55, 0.65)
            self.assertAlmostEqual(
                float(gc_q[3, 7]), 1.0, places=5
            )  # 0.80 in [0.65, 1.0]

            os.unlink(f"{fa_path}.fai")
            self.assertEqual(read_fasta_contigs_core(str(fa_path)), [("chr1", 20_000)])
            gc_u, nf_u, gq_u = scan_fasta_gc_quantiles(
                str(fa_path), [("chr1", 20_000)], 5_000
            )
            np.testing.assert_allclose(gc_u, gc_mean, atol=1e-6)
            np.testing.assert_allclose(nf_u, n_frac, atol=1e-6)
            np.testing.assert_allclose(gq_u, gc_q, atol=1e-6)

            bw_path = root / "track.bw"
            bw = pyBigWig.open(str(bw_path), "w")
            assert bw is not None
            try:
                bw.addHeader([("chr1", 20_000)])
                bw.addEntries(
                    ["chr1", "chr1", "chr1"],
                    [0, 2_500, 5_000],
                    ends=[2_500, 5_000, 10_000],
                    values=[1.0, 0.5, 0.8],
                )
            finally:
                bw.close()

            bw_means = scan_bigwig_means(str(bw_path), [("chr1", 20_000)], 5_000)
            self.assertAlmostEqual(float(bw_means[0]), 0.75, places=5)
            self.assertAlmostEqual(float(bw_means[1]), 0.80, places=5)
            self.assertTrue(np.isnan(bw_means[2]))
            self.assertTrue(np.isnan(bw_means[3]))

            # Single corrupt bin 1.05 in [0, 1] BED track masks bin 1 without dividing valid bins by 100
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
            gc_track, n_mask, gc_q = load_gc_track(
                corrupt_bed_path, contigs, starts, ends
            )
            self.assertIsNone(gc_q)
            self.assertTrue(np.isnan(gc_track[1]))
            self.assertTrue(bool(n_mask[1]))
            self.assertGreater(float(gc_track[0]), 0.35)
            self.assertLess(float(gc_track[0]), 0.65)

    def test_bgzf_indexed_emits_tbi_readable_by_pysam_and_csi_above_512mb(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bed_gz = root / "test.bed.gz"
            # Generate >150 KB across multiple chunks to cross 64 KB BGZF block boundaries
            rows = [b"#contig\tstart\tend\tval\n"]
            for idx in range(5_000):
                s = idx * 100
                e = s + 100
                rows.append(f"chr1\t{s}\t{e}\t{idx * 0.125:.6f}\n".encode())
            payload = b"".join(rows)
            # Split at arbitrary mid-line byte offsets to verify chunk reassembly
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

            # Structural VCF with INFO/END queried mid-interval via pysam.VariantFile
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
            # 80,000 rows (~2.65 MB uncompressed > 32 * 65,280 = 2,088,960 bytes)
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
            fa_path = _make_fasta(
                root / "grid.fa",
                {"chr1": "ACGT" * 2_500, "chr2": "GGCC" * 2_500},
            )
            bw_path = root / "grid.bw"
            bw = pyBigWig.open(str(bw_path), "w")
            assert bw is not None
            try:
                bw.addHeader([("chr1", 10_000), ("chr2", 10_000)])
                bw.addEntries(
                    ["chr1", "chr2"], [0, 0], ends=[10_000, 10_000], values=[0.9, 0.8]
                )
            finally:
                bw.close()

            # Case 1: starts=[0, 2500] aligned to bin_size=2500, but ends=[2500, 7500] spans 5000 bp
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

            # Case 2: multi-contig mixing 5 kb bins on chr1 with a 10 kb bin on chr2
            with self.assertRaises(CovariateError):
                load_mappability_track(
                    bw_path,
                    ["chr1", "chr1", "chr2"],
                    np.array([0, 5_000, 0], dtype=np.int64),
                    np.array([5_000, 10_000, 10_000], dtype=np.int64),
                )

    def test_cram_header_skips_fasta_body_and_55mb_tile_boundary_crai_dedup(
        self,
    ) -> None:
        import gzip

        chr1_len = 55_000_000
        chr2_len = 10_000
        seq_chr1 = "ACGTGCAT" * (chr1_len // 8)
        seq_chr2 = "GCTATAGC" * (chr2_len // 8)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fa_path = _make_fasta(
                root / "large_ref.fa",
                {"chr1": seq_chr1, "chr2": seq_chr2},
            )
            header = pysam.AlignmentHeader.from_dict(
                {
                    "HD": {"VN": "1.6", "SO": "coordinate"},
                    "SQ": [
                        {"SN": "chr1", "LN": chr1_len},
                        {"SN": "chr2", "LN": chr2_len},
                    ],
                }
            )

            records: list[pysam.AlignedSegment] = []
            # 1. Tile 0 interior pair: bin 2
            records.extend(
                _make_proper_pair(header, "p_t0", "chr1", 10_100, 166, seq_chr1)
            )
            # 2. Spans 50 Mb boundary with midpoint in tile 0 (49_999_990 -> bin 9999)
            #    and downstream mate/endpoint in tile 1 (50_000_039 -> bin 10000)
            records.extend(
                _make_proper_pair(
                    header,
                    "p_span_mid_t0",
                    "chr1",
                    49_999_940,
                    100,
                    seq_chr1,
                    read_len=40,
                )
            )
            # 3. Spans 50 Mb boundary with upstream read in tile 0 (49_999_900)
            #    and midpoint (50_000_025 -> bin 10000) + mate (50_000_100) in tile 1
            records.extend(
                _make_proper_pair(
                    header, "p_span_mid_t1", "chr1", 49_999_900, 250, seq_chr1
                )
            )
            # 4. Tile 1 interior pair: bin 10400
            records.extend(
                _make_proper_pair(header, "p_t1", "chr1", 52_000_100, 150, seq_chr1)
            )
            # 5. Second contig pair
            records.extend(
                _make_proper_pair(header, "p_chr2", "chr2", 1_000, 140, seq_chr2)
            )
            records.sort(key=lambda r: (r.reference_id, r.reference_start))

            bam_path = root / "boundary.bam"
            cram_path = root / "boundary.cram"
            with pysam.AlignmentFile(str(bam_path), "wb", header=header) as out_bam:
                for rec in records:
                    out_bam.write(rec)
            pysam.index(str(bam_path))

            with (
                pysam.AlignmentFile(str(bam_path), "rb") as in_bam,
                pysam.AlignmentFile(
                    str(cram_path), "wc", header=header, reference_filename=str(fa_path)
                ) as out_cram,
            ):
                for rec in in_bam:
                    out_cram.write(rec)
            pysam.index(str(cram_path))

            # Finding #2: read_alignment_header validates FASTA readability without parsing FASTA records
            header_only_placeholder = root / "unindexed_placeholder.fa"
            header_only_placeholder.write_bytes(b"")
            self.assertEqual(
                read_alignment_header(str(cram_path), str(header_only_placeholder)),
                [("chr1", chr1_len), ("chr2", chr2_len)],
            )
            with self.assertRaises(AlignmentError):
                read_alignment_header(str(cram_path), str(root / "missing.fa"))

            # Finding #3: inject non-adjacent duplicate container offset entries into .crai
            crai_path = Path(f"{cram_path}.crai")
            with gzip.open(crai_path, "rt", encoding="ascii") as gz_in:
                crai_lines = [ln for ln in gz_in.read().splitlines() if ln.strip()]
            self.assertGreaterEqual(len(crai_lines), 1)
            interleaved = [*crai_lines, crai_lines[0]]
            if len(crai_lines) == 1:
                fields = crai_lines[0].split("\t")
                # Add a synthetic chr2 entry between duplicate chr1 container entries
                synth_other = "\t".join(
                    ["1", "1", "100", fields[3], fields[4], fields[5]]
                )
                interleaved = [crai_lines[0], synth_other, crai_lines[0]]
            with gzip.open(crai_path, "wt", encoding="ascii") as gz_out:
                gz_out.write("\n".join(interleaved) + "\n")

            targets = [("chr1", chr1_len), ("chr2", chr2_len)]
            bam_t1 = scan_alignment_bins(str(bam_path), targets, 5_000, 20, True, 1)
            bam_t2 = scan_alignment_bins(str(bam_path), targets, 5_000, 20, True, 2)
            cram_t1 = scan_alignment_bins(
                str(cram_path), targets, 5_000, 20, True, 1, str(fa_path)
            )
            cram_t2 = scan_alignment_bins(
                str(cram_path), targets, 5_000, 20, True, 2, str(fa_path)
            )

            _assert_scan_parity(bam_t1, bam_t2)
            _assert_scan_parity(bam_t1, cram_t1)
            _assert_scan_parity(bam_t1, cram_t2)

            self.assertEqual(bam_t1["qc_counts"], bam_t2["qc_counts"])
            self.assertEqual(bam_t1["qc_counts"], cram_t1["qc_counts"])
            self.assertEqual(bam_t1["qc_counts"], cram_t2["qc_counts"])

            counts = cram_t2["count"]
            endpoints = cram_t2["endpoint_count"]
            self.assertEqual(int(counts.sum()), 5)
            self.assertEqual(int(counts[9_999]), 1)
            self.assertEqual(int(counts[10_000]), 1)
            # Bin 9999 has upstream endpoints of p_span_mid_t0 and p_span_mid_t1 (2);
            # Bin 10000 has downstream endpoints of both boundary-spanning pairs (2).
            self.assertEqual(int(endpoints[9_999]), 2)
            self.assertEqual(int(endpoints[10_000]), 2)


if __name__ == "__main__":
    unittest.main()
