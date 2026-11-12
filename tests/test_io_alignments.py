"""Unit and Rust-core parity tests for BAM/CRAM alignment headers, scanning, and indexing."""

from __future__ import annotations

import gzip
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pysam
from helpers import write_test_bam, write_tiny_fasta

from ccnv._core import (
    read_alignment_header as read_alignment_header_core,
)
from ccnv._core import (
    scan_alignment_bins as scan_alignment_bins_core,
)
from ccnv.io.alignments import (
    alignment_format,
    read_alignment_header,
    scan_alignment_bins,
    validate_alignment_inputs,
)
from ccnv.models import (
    AlignmentError,
    ExtractError,
    ReadFilters,
    ScanAlignmentResult,
    select_layout,
)
from ccnv.stats.fragmentomics import encode_kmer_4


def _make_indexed_fasta(path: Path, contigs: dict[str, str]) -> Path:
    write_tiny_fasta(path, contigs)
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


class TestAlignmentsIO(unittest.TestCase):
    def _assert_scan_parity(
        self, lhs: ScanAlignmentResult, rhs: ScanAlignmentResult
    ) -> None:
        np.testing.assert_array_equal(lhs["count"], rhs["count"], err_msg="count")
        np.testing.assert_array_equal(
            lhs["paired_count"], rhs["paired_count"], err_msg="paired_count"
        )
        np.testing.assert_array_equal(
            lhs["frag_len_sum"], rhs["frag_len_sum"], err_msg="frag_len_sum"
        )
        np.testing.assert_array_equal(
            lhs["short_frag_count"],
            rhs["short_frag_count"],
            err_msg="short_frag_count",
        )
        np.testing.assert_array_equal(
            lhs["endpoint_count"], rhs["endpoint_count"], err_msg="endpoint_count"
        )
        if lhs["frag_bands"] is None or rhs["frag_bands"] is None:
            self.assertIs(lhs["frag_bands"], rhs["frag_bands"])
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

    def test_alignment_format_valid(self) -> None:
        self.assertEqual(alignment_format(Path("sample.bam")), "bam")
        self.assertEqual(alignment_format(Path("sample.BAM")), "bam")
        self.assertEqual(alignment_format(Path("sample.cram")), "cram")
        self.assertEqual(alignment_format(Path("sample.CRAM")), "cram")

    def test_alignment_format_invalid(self) -> None:
        with self.assertRaises(AlignmentError):
            alignment_format(Path("sample.sam"))
        with self.assertRaises(ExtractError):
            alignment_format(Path("sample.sam"))

    def test_validate_alignment_cram_missing_fasta(self) -> None:
        with self.assertRaises(AlignmentError):
            validate_alignment_inputs(Path("sample.cram"), fasta=None)

    def test_validate_alignment_cram_nonexistent_fasta(self) -> None:
        with self.assertRaises(AlignmentError):
            validate_alignment_inputs(
                Path("sample.cram"),
                fasta=Path("/nonexistent/reference.fa"),
            )

    def test_validate_alignment_missing_bam(self) -> None:
        with self.assertRaises(AlignmentError):
            read_alignment_header(Path("/nonexistent/sample.bam"))

    def test_canine_38_autosomes_preserved_when_all_contigs_false(self) -> None:
        header = [(f"chr{i}", 100_000) for i in range(1, 39)] + [
            ("chrX", 100_000),
            ("chrM", 16_000),
            ("chr0", 5_000),
        ]
        selected = select_layout(header, all_contigs=False)
        self.assertEqual(len(selected), 39)
        self.assertIn(("chr38", 100_000), selected)
        self.assertNotIn(("chrM", 16_000), selected)
        self.assertNotIn(("chr0", 5_000), selected)

    def test_scan_alignment_bins_signed_parameter_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            with self.assertRaises(AlignmentError):
                scan_alignment_bins_core(
                    str(tmp_dir / "missing.bam"), [("chr1", 1000)], 0
                )
            with self.assertRaises(AlignmentError):
                scan_alignment_bins_core(
                    str(tmp_dir / "missing.bam"), [("chr1", -10)], 5000
                )
            with self.assertRaises(AlignmentError):
                scan_alignment_bins_core(
                    str(tmp_dir / "missing.bam"), [("chr1", 1000)], 5000, threads=0
                )
            with self.assertRaises(AlignmentError):
                scan_alignment_bins_core(
                    str(tmp_dir / "missing.bam"),
                    [("chr1", 1000)],
                    5000,
                    se_fragment_length=-1,
                )

    def test_picard_multi_dot_bam_index_resolution_ignores_sibling_stem_bai(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bam_path = tmp / "sample.sorted.bam"
            header = pysam.AlignmentHeader.from_dict(
                {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 20_000}]}
            )
            reads: list[pysam.AlignedSegment] = []
            for idx in range(25):
                rec = pysam.AlignedSegment(header)
                rec.query_name = f"r{idx}"
                rec.query_sequence = "ACGT" * 25
                rec.flag = 0
                rec.reference_id = 0
                rec.reference_start = idx * 500 + 50
                rec.mapping_quality = 60
                rec.cigar = ((0, 100),)
                rec.query_qualities = pysam.qualitystring_to_array("I" * 100)
                reads.append(rec)
            write_test_bam(bam_path, reads, header=header, index=True)

            samtools_bai = tmp / "sample.sorted.bam.bai"
            picard_bai = tmp / "sample.sorted.bai"
            decoy_bai = tmp / "sample.bai"
            shutil.move(str(samtools_bai), str(picard_bai))
            decoy_bai.write_bytes(b"NOT_A_VALID_BAI_INDEX")

            filters = ReadFilters(min_mapping_quality=1, require_proper_pair=False)
            res_1 = scan_alignment_bins(
                bam_path,
                (("chr1", 20_000),),
                bin_size=5000,
                filters=filters,
                threads=1,
            )
            res_2 = scan_alignment_bins(
                bam_path,
                (("chr1", 20_000),),
                bin_size=5000,
                filters=filters,
                threads=2,
            )
            np.testing.assert_array_equal(res_1["count"], res_2["count"])

            picard_bai.unlink()
            with self.assertRaises(AlignmentError):
                scan_alignment_bins(
                    bam_path,
                    (("chr1", 20_000),),
                    bin_size=5000,
                    filters=filters,
                    threads=2,
                )

    def test_bam_and_cram_parity_with_pysam_and_mate_dedup_accounting(self) -> None:
        rng = np.random.default_rng(12345)
        alphabet = np.array(list("ACGT"))
        seq_chr1 = "".join(rng.choice(alphabet, size=20_000))
        seq_chrx = "".join(rng.choice(alphabet, size=10_000))
        seq_chry = "".join(rng.choice(alphabet, size=10_000))

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fa_path = _make_indexed_fasta(
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
            r1_a, r2_a = _make_proper_pair(
                header, "p_bin0", "chr1", 4_900, 100, seq_chr1
            )
            records.extend([r1_a, r2_a])
            records.extend(
                _make_proper_pair(header, "p_166", "chr1", 6_000, 166, seq_chr1)
            )
            records.extend(
                _make_proper_pair(
                    header, "p_short15", "chr1", 11_000, 15, seq_chr1, read_len=15
                )
            )
            records.extend(
                _make_proper_pair(header, "p_long900", "chr1", 16_000, 900, seq_chr1)
            )
            records.extend(
                _make_proper_pair(header, "p_chrx", "chrX", 1_000, 140, seq_chrx)
            )
            records.extend(
                _make_proper_pair(header, "p_chry", "chrY", 2_000, 150, seq_chry)
            )

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

            unsorted_bam = root / "unsorted.bam"
            bam_path = root / "sample.bam"
            cram_path = root / "sample.cram"
            write_test_bam(unsorted_bam, records, header=header, index=False)
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
                read_alignment_header_core(str(bam_path)),
                [("chr1", 20_000), ("chrX", 10_000), ("chrY", 10_000)],
            )
            self.assertEqual(
                read_alignment_header_core(str(cram_path), str(fa_path)),
                [("chr1", 20_000), ("chrX", 10_000), ("chrY", 10_000)],
            )
            with self.assertRaises(AlignmentError):
                read_alignment_header_core(str(cram_path), None)

            target_contigs = [("chr1", 20_000), ("chrX", 10_000), ("chrY", 10_000)]
            res_bam = scan_alignment_bins_core(
                str(bam_path), target_contigs, 5_000, 20, True, 2
            )
            res_bam_t1 = scan_alignment_bins_core(
                str(bam_path), target_contigs, 5_000, 20, True, 1
            )
            res_cram = scan_alignment_bins_core(
                str(cram_path), target_contigs, 5_000, 20, True, 2, str(fa_path)
            )
            res_no_bands = scan_alignment_bins_core(
                str(bam_path), target_contigs, 5_000, 20, True, 1, None, False
            )
            self.assertIsNone(res_no_bands["frag_bands"])

            self._assert_scan_parity(res_bam, res_bam_t1)
            self._assert_scan_parity(res_bam, res_cram)
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
            if bands is None:
                self.fail("expected non-None frag_bands")
            hist_auto = res_bam["hist_autosomal"]
            hist_x = res_bam["hist_chrx"]
            hist_y = res_bam["hist_chry"]
            motifs = res_bam["motif_counts"]

            np.testing.assert_array_equal(counts, [1, 1, 1, 1, 1, 0, 1, 0])
            np.testing.assert_array_equal(paired_counts, [1, 1, 0, 0, 1, 0, 1, 0])
            np.testing.assert_array_equal(
                frag_len_sum, [100, 166, 0, 0, 140, 0, 150, 0]
            )
            np.testing.assert_array_equal(short_counts, [1, 0, 0, 0, 1, 0, 1, 0])
            self.assertEqual(int(endpoints[0]), 2)
            self.assertEqual(int(endpoints[1]), 2)
            self.assertEqual(bands.shape, (8, 24))
            self.assertEqual(int(bands[0, 5]), 1)
            self.assertEqual(int(bands[1, 11]), 1)
            self.assertEqual(int(np.sum(bands[2])), 0)
            self.assertEqual(int(np.sum(bands[3])), 0)
            self.assertEqual(int(hist_auto[100 - 30]), 1)
            self.assertEqual(int(hist_auto[166 - 30]), 1)
            self.assertEqual(int(np.sum(hist_auto)), 2)
            self.assertEqual(int(hist_x[140 - 30]), 1)
            self.assertEqual(int(hist_y[150 - 30]), 1)
            self.assertEqual(int(np.sum(motifs)), 12)
            r1_seq_str = r1_a.query_sequence or ""
            r2_seq_str = r2_a.query_sequence or ""
            m_up = encode_kmer_4(r1_seq_str[:4])
            dn_rc = r2_seq_str[-4:][::-1].translate(
                str.maketrans("ACGTacgt", "TGCAtgca")
            )
            m_dn = encode_kmer_4(dn_rc)
            if m_up is None or m_dn is None:
                self.fail("expected valid encoded 4-mer motifs")
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
        fwd = pysam.AlignedSegment(header)
        fwd.query_name = "se_fwd"
        fwd.flag = 0
        fwd.reference_id = 0
        fwd.reference_start = 4_950
        fwd.mapping_quality = 60
        fwd.cigarstring = "50M"
        fwd.query_sequence = "ACGT" * 12 + "AC"
        fwd.query_qualities = pysam.qualitystring_to_array("I" * 50)

        rev = pysam.AlignedSegment(header)
        rev.query_name = "se_rev"
        rev.flag = 0x10
        rev.reference_id = 0
        rev.reference_start = 5_010
        rev.mapping_quality = 60
        rev.cigarstring = "50M"
        rev.query_sequence = "GGCC" * 12 + "GG"
        rev.query_qualities = pysam.qualitystring_to_array("I" * 50)

        write_test_bam(bam_path, [fwd, rev], header=header, index=False)
        os.chmod(root, 0o555)

        res_unshifted = scan_alignment_bins_core(
            str(bam_path), [("chr1", 20_000)], 5_000, 20, True, 1, None, False, 0
        )
        self.assertIs(res_unshifted["paired_end"], False)
        np.testing.assert_array_equal(res_unshifted["count"], [1, 1, 0, 0])

        res_shifted = scan_alignment_bins_core(
            str(bam_path), [("chr1", 20_000)], 5_000, 20, True, 1, None, False, 160
        )
        np.testing.assert_array_equal(res_shifted["count"], [1, 1, 0, 0])
        np.testing.assert_array_equal(res_shifted["endpoint_count"], [1, 1, 0, 0])

        with self.assertRaisesRegex(AlignmentError, "index|--threads 1"):
            scan_alignment_bins_core(
                str(bam_path), [("chr1", 20_000)], 5_000, 20, True, 4
            )

    def test_cram_header_skips_fasta_body_and_55mb_tile_boundary_crai_dedup(
        self,
    ) -> None:
        chr1_len = 55_000_000
        chr2_len = 10_000
        seq_chr1 = "ACGTGCAT" * (chr1_len // 8)
        seq_chr2 = "GCTATAGC" * (chr2_len // 8)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fa_path = _make_indexed_fasta(
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
            records.extend(
                _make_proper_pair(header, "p_t0", "chr1", 10_100, 166, seq_chr1)
            )
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
            records.extend(
                _make_proper_pair(
                    header, "p_span_mid_t1", "chr1", 49_999_900, 250, seq_chr1
                )
            )
            records.extend(
                _make_proper_pair(header, "p_t1", "chr1", 52_000_100, 150, seq_chr1)
            )
            records.extend(
                _make_proper_pair(header, "p_chr2", "chr2", 1_000, 140, seq_chr2)
            )

            bam_path = root / "boundary.bam"
            cram_path = root / "boundary.cram"
            write_test_bam(bam_path, records, header=header, index=True)

            with (
                pysam.AlignmentFile(str(bam_path), "rb") as in_bam,
                pysam.AlignmentFile(
                    str(cram_path), "wc", header=header, reference_filename=str(fa_path)
                ) as out_cram,
            ):
                for rec in in_bam:
                    out_cram.write(rec)
            pysam.index(str(cram_path))

            header_only_placeholder = root / "unindexed_placeholder.fa"
            header_only_placeholder.write_bytes(b"")
            self.assertEqual(
                read_alignment_header_core(
                    str(cram_path), str(header_only_placeholder)
                ),
                [("chr1", chr1_len), ("chr2", chr2_len)],
            )
            with self.assertRaises(AlignmentError):
                read_alignment_header_core(str(cram_path), str(root / "missing.fa"))

            crai_path = Path(f"{cram_path}.crai")
            with gzip.open(crai_path, "rt", encoding="ascii") as gz_in:
                crai_lines = [ln for ln in gz_in.read().splitlines() if ln.strip()]
            self.assertGreaterEqual(len(crai_lines), 1)
            interleaved = [*crai_lines, crai_lines[0]]
            if len(crai_lines) == 1:
                fields = crai_lines[0].split("\t")
                synth_other = "\t".join(
                    ["1", "1", "100", fields[3], fields[4], fields[5]]
                )
                interleaved = [crai_lines[0], synth_other, crai_lines[0]]
            with gzip.open(crai_path, "wt", encoding="ascii") as gz_out:
                gz_out.write("\n".join(interleaved) + "\n")

            targets = [("chr1", chr1_len), ("chr2", chr2_len)]
            bam_t1 = scan_alignment_bins_core(
                str(bam_path), targets, 5_000, 20, True, 1
            )
            bam_t2 = scan_alignment_bins_core(
                str(bam_path), targets, 5_000, 20, True, 2
            )
            cram_t1 = scan_alignment_bins_core(
                str(cram_path), targets, 5_000, 20, True, 1, str(fa_path)
            )
            cram_t2 = scan_alignment_bins_core(
                str(cram_path), targets, 5_000, 20, True, 2, str(fa_path)
            )

            self._assert_scan_parity(bam_t1, bam_t2)
            self._assert_scan_parity(bam_t1, cram_t1)
            self._assert_scan_parity(bam_t1, cram_t2)

            self.assertEqual(bam_t1["qc_counts"], bam_t2["qc_counts"])
            self.assertEqual(bam_t1["qc_counts"], cram_t1["qc_counts"])
            self.assertEqual(bam_t1["qc_counts"], cram_t2["qc_counts"])

            counts = cram_t2["count"]
            endpoints = cram_t2["endpoint_count"]
            self.assertEqual(int(counts.sum()), 5)
            self.assertEqual(int(counts[9_999]), 1)
            self.assertEqual(int(counts[10_000]), 1)
            self.assertEqual(int(endpoints[9_999]), 2)
            self.assertEqual(int(endpoints[10_000]), 2)


if __name__ == "__main__":
    unittest.main()
