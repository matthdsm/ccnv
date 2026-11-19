"""Unit and integration tests for BAM/CRAM sample sketch extraction."""

from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pysam
from published_fixtures import (
    CHR21_LENGTH,
    PUBLISHED_BIN_SIZE,
    PUBLISHED_CRAM,
    PUBLISHED_CRAM_URL,
    ensure_published_fasta,
    ensure_published_file,
)
from typer.testing import CliRunner

from ccnv.cli import app
from ccnv.extract import extract_sample_table, run
from ccnv.models import (
    FRAGMENTOMICS_METADATA_KEY,
    AlignmentError,
    ExtractConfig,
    ExtractError,
    FragmentomicsMetadata,
    ReadFilters,
    SampleMetadata,
    is_canonical_contig,
    normalize_chr,
    parquet_path,
    reconstruct_sample_grid,
    select_layout,
    validate_sample_table,
)
from ccnv.models.schemas import (
    COORDINATE_SYSTEM,
    SCHEMA_VERSION,
    decode_metadata,
)
from ccnv.qc.sample import QC_COUNTERS, build_sample_qc, empty_qc


def _read_sample_table(path: Path) -> pa.Table:
    table = pq.read_table(path)
    validate_sample_table(table)
    return table


def _read(
    name: str,
    reference_id: int,
    start: int,
    mapq: int,
    *,
    is_duplicate: bool = False,
) -> pysam.AlignedSegment:
    read = pysam.AlignedSegment()
    read.query_name = name
    read.query_sequence = "ACGTACGTAC"
    read.flag = 1024 if is_duplicate else 0
    read.reference_id = reference_id
    read.reference_start = start
    read.mapping_quality = mapq
    read.cigar = ((0, 10),)
    read.next_reference_id = -1
    read.next_reference_start = -1
    read.template_length = 0
    read.query_qualities = pysam.qualitystring_to_array("I" * 10)
    return read


def _has_alignment_index(path: Path) -> bool:
    return any(
        candidate.exists()
        for candidate in (
            Path(f"{path}.bai"),
            Path(f"{path}.csi"),
            Path(f"{path}.crai"),
            path.with_suffix(".bai"),
            path.with_suffix(".crai"),
        )
    )


def _write_bam(
    path: Path,
    reads: list[pysam.AlignedSegment],
    *,
    header: dict[str, object] | None = None,
    index: bool,
) -> None:
    bam_header = header or {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "chr1", "LN": 250}],
    }
    with pysam.AlignmentFile(str(path), "wb", header=bam_header) as bam:
        for read in sorted(
            reads, key=lambda item: (item.reference_id, item.reference_start)
        ):
            bam.write(read)
    if index:
        pysam.index(str(path))


class TestExtractAlignment(unittest.TestCase):
    def tearDown(self) -> None:
        logging.getLogger("ccnv").handlers.clear()

    def test_invalid_header_raises_extract_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bam_path = Path(tmpdir) / "sample.bam"
            bam_path.touch()
            with (
                patch("ccnv.extract.read_alignment_header", return_value=[]),
                self.assertRaises(ExtractError),
            ):
                run(ExtractConfig(bam_path, Path(tmpdir) / "out", bin_size=100))

    def test_unindexed_bam_streams_sequentially_and_parallel_requires_index(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bam_path = Path(tmpdir) / "sample.bam"
            prefix = Path(tmpdir) / "sample"
            _write_bam(bam_path, [_read("r1", 0, 0, 10)], index=False)
            self.assertFalse(_has_alignment_index(bam_path))
            run(ExtractConfig(bam_path, prefix, bin_size=100, threads=1))
            self.assertFalse(_has_alignment_index(bam_path))
            table = _read_sample_table(parquet_path(prefix))
            self.assertEqual(table.column("count").to_pylist(), [1, 0, 0])

            with self.assertRaises(AlignmentError):
                run(ExtractConfig(bam_path, prefix, bin_size=100, threads=2))

    def test_cram_without_fasta_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cram_path = Path(tmpdir) / "sample.cram"
            cram_path.touch()
            with self.assertRaisesRegex(ExtractError, r"CRAM.*--fasta"):
                run(ExtractConfig(cram_path, Path(tmpdir) / "out", bin_size=100))

    def test_unindexed_cram_streams_sequentially(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta = Path(tmpdir) / "ref.fa"
            fasta.write_text(">chr1\n" + "A" * 250 + "\n", encoding="utf-8")
            pysam.faidx(str(fasta))
            bam_path = Path(tmpdir) / "sample.bam"
            cram_path = Path(tmpdir) / "sample.cram"
            _write_bam(bam_path, [_read("r1", 0, 0, 60)], index=True)
            with (
                pysam.AlignmentFile(str(bam_path), "rb") as bam,
                pysam.AlignmentFile(
                    str(cram_path),
                    "wc",
                    header=bam.header,
                    reference_filename=str(fasta),
                ) as cram,
            ):
                for read in bam:
                    cram.write(read)
            self.assertFalse(_has_alignment_index(cram_path))
            prefix = Path(tmpdir) / "sample"
            run(
                ExtractConfig(
                    cram_path,
                    prefix,
                    bin_size=100,
                    fasta=fasta,
                    threads=1,
                )
            )
            self.assertFalse(_has_alignment_index(cram_path))
            table = _read_sample_table(parquet_path(prefix))
            self.assertEqual(sum(table.column("count").to_pylist()), 1)

    def test_partial_last_bin_and_non_human_contigs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bam_path = Path(tmpdir) / "sample.bam"
            prefix = Path(tmpdir) / "sample"
            out_path = parquet_path(prefix)
            reads = [
                _read("keep1", 0, 0, 60),
                _read("edge", 0, 99, 60),
                _read("bin1", 0, 100, 60),
                _read("partial", 0, 200, 60),
                _read("lowq", 0, 10, 0),
                _read("dup", 0, 20, 60, is_duplicate=True),
                _read("scaf", 1, 0, 60),
                _read("mito", 2, 0, 60),
            ]
            header = {
                "HD": {"VN": "1.6", "SO": "coordinate"},
                "SQ": [
                    {"SN": "chr1", "LN": 250},
                    {"SN": "scaffold-A", "LN": 99},
                    {"SN": "chrM", "LN": 50},
                ],
            }
            _write_bam(bam_path, reads, header=header, index=True)
            run(ExtractConfig(bam_path, prefix, bin_size=100, all_contigs=True))

            table = _read_sample_table(out_path)
            metadata = decode_metadata(table.schema.metadata)
            self.assertEqual(metadata["schema_version"], SCHEMA_VERSION)
            self.assertEqual(metadata["coordinate_system"], COORDINATE_SYSTEM)
            self.assertEqual(metadata["bin_size"], 100)
            self.assertEqual(
                metadata["contig_lengths"],
                {"chr1": 250, "scaffold-A": 99, "chrM": 50},
            )
            self.assertEqual(metadata["contig_order"], ["chr1", "scaffold-A", "chrM"])
            self.assertNotIn("source_identity", metadata)
            self.assertNotIn("reference_identity", metadata)
            qc = metadata["qc"]
            if not isinstance(qc, dict):
                self.fail("expected qc dict")
            self.assertEqual(qc["reads_seen"], 8)
            self.assertEqual(qc["reads_kept"], 6)
            self.assertEqual(qc["effective_reads_examined"], 8)
            self.assertEqual(qc["effective_reads_kept"], 6)
            self.assertEqual(qc["autosomal_reads_kept"], 4)
            self.assertEqual(qc["sex_reads_kept"], 0)
            self.assertEqual(qc["filter_mapq"], 1)
            self.assertEqual(qc["filter_duplicate"], 1)
            self.assertEqual(set(qc), set(QC_COUNTERS))
            filters = metadata["filters"]
            if not isinstance(filters, dict):
                self.fail("expected filters dict")
            self.assertEqual(filters["min_mapping_quality"], 1)

            codes, names, starts, ends = reconstruct_sample_grid(
                SampleMetadata.from_table(table)
            )
            self.assertEqual(
                [names[int(c)] for c in codes],
                ["chr1", "chr1", "chr1", "scaffold-A", "chrM"],
            )
            self.assertEqual(starts.tolist(), [0, 100, 200, 0, 0])
            self.assertEqual(ends.tolist(), [100, 200, 250, 99, 50])
            self.assertEqual(table.column("count").to_pylist(), [2, 1, 1, 1, 1])

    def test_extract_applies_configured_min_mapping_quality(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bam_path = Path(tmpdir) / "sample.bam"
            prefix = Path(tmpdir) / "sample"
            _write_bam(
                bam_path,
                [
                    _read("keep", 0, 0, 20),
                    _read("drop", 0, 10, 19),
                ],
                index=True,
            )
            run(
                ExtractConfig(
                    bam_path,
                    prefix,
                    bin_size=100,
                    filters=ReadFilters(min_mapping_quality=20),
                )
            )
            table = _read_sample_table(parquet_path(prefix))
            metadata = decode_metadata(table.schema.metadata)
            filters = metadata["filters"]
            qc = metadata["qc"]
            if not isinstance(filters, dict) or not isinstance(qc, dict):
                self.fail("expected filters and qc dicts")
            self.assertEqual(filters["min_mapping_quality"], 20)
            self.assertEqual(qc["filter_mapq"], 1)
            self.assertEqual(qc["reads_kept"], 1)
            self.assertEqual(table.column("count").to_pylist(), [1, 0, 0])

    def test_cli_min_mapq_filters_reads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bam_path = Path(tmpdir) / "sample.bam"
            prefix = Path(tmpdir) / "sample"
            _write_bam(
                bam_path,
                [
                    _read("keep", 0, 0, 40),
                    _read("drop", 0, 50, 10),
                ],
                index=True,
            )
            result = CliRunner().invoke(
                app,
                [
                    "extract",
                    str(bam_path),
                    "--prefix",
                    str(prefix),
                    "--bin-size",
                    "100",
                    "--min-mapq",
                    "30",
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            table = _read_sample_table(parquet_path(prefix))
            metadata = decode_metadata(table.schema.metadata)
            filters = metadata["filters"]
            if not isinstance(filters, dict):
                self.fail("expected filters dict")
            self.assertEqual(filters["min_mapping_quality"], 30)
            self.assertEqual(table.column("count").to_pylist(), [1, 0, 0])

    def test_cli_extract_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bam_path = Path(tmpdir) / "sample.bam"
            prefix = Path(tmpdir) / "sample"
            out_path = parquet_path(prefix)
            _write_bam(bam_path, [_read("r1", 0, 0, 60)], index=True)
            result = CliRunner().invoke(
                app,
                [
                    "extract",
                    str(bam_path),
                    "--prefix",
                    str(prefix),
                    "--bin-size",
                    "100",
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            table = _read_sample_table(out_path)
            self.assertEqual(table.column("count").to_pylist(), [1, 0, 0])

    def test_cli_debug_loglevel_prints_extract_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bam_path = Path(tmpdir) / "sample.bam"
            prefix = Path(tmpdir) / "sample"
            _write_bam(bam_path, [_read("r1", 0, 0, 60)], index=True)
            result = CliRunner().invoke(
                app,
                [
                    "extract",
                    str(bam_path),
                    "--prefix",
                    str(prefix),
                    "--bin-size",
                    "100",
                    "--loglevel",
                    "DEBUG",
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("opening alignment", result.output)
            self.assertIn("extract QC summary", result.output)
            self.assertIn("writing sample parquet", result.output)

    def test_extract_debug_logs_each_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bam_path = Path(tmpdir) / "sample.bam"
            prefix = Path(tmpdir) / "sample"
            _write_bam(bam_path, [_read("r1", 0, 0, 60)], index=True)
            with self.assertLogs("ccnv.extract", level="DEBUG") as captured:
                run(
                    ExtractConfig(
                        bam_path,
                        prefix,
                        bin_size=100,
                        filters=ReadFilters(),
                    )
                )
            messages = "\n".join(captured.output)
            self.assertIn("opening alignment", messages)
            self.assertIn("extract QC summary", messages)
            self.assertIn("writing sample parquet", messages)


class TestParallelBinning(unittest.TestCase):
    def _multi_contig_bam(self, path: Path) -> None:
        header = {
            "HD": {"VN": "1.6", "SO": "coordinate"},
            "SQ": [
                {"SN": "chr1", "LN": 250},
                {"SN": "chr2", "LN": 150},
                {"SN": "chr3", "LN": 100},
            ],
        }
        reads = [
            _read("a", 0, 0, 60),
            _read("b", 0, 120, 60),
            _read("c", 1, 10, 60),
            _read("d", 2, 5, 60),
            _read("e", 2, 30, 60),
        ]
        _write_bam(path, reads, header=header, index=True)

    def test_parallel_matches_serial_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bam_path = root / "sample.bam"
            self._multi_contig_bam(bam_path)

            serial_prefix = root / "serial"
            parallel_prefix = root / "parallel"
            run(ExtractConfig(bam_path, serial_prefix, bin_size=100, threads=1))
            run(ExtractConfig(bam_path, parallel_prefix, bin_size=100, threads=3))

            serial = _read_sample_table(parquet_path(serial_prefix))
            parallel = _read_sample_table(parquet_path(parallel_prefix))
            self.assertEqual(serial.to_pydict(), parallel.to_pydict())
            self.assertEqual(
                decode_metadata(serial.schema.metadata)["qc"],
                decode_metadata(parallel.schema.metadata)["qc"],
            )

    def test_parallel_preserves_header_contig_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bam_path = root / "sample.bam"
            self._multi_contig_bam(bam_path)
            prefix = root / "sample"
            run(ExtractConfig(bam_path, prefix, bin_size=100, threads=3))

            table = _read_sample_table(parquet_path(prefix))
            smeta = SampleMetadata.from_table(table)
            self.assertEqual(smeta.contig_order, ("chr1", "chr2", "chr3"))
            codes, names, _, _ = reconstruct_sample_grid(smeta)
            self.assertEqual(
                list(dict.fromkeys(names[int(c)] for c in codes)),
                ["chr1", "chr2", "chr3"],
            )
            self.assertEqual(smeta.qc["reads_kept"], 5)

    def test_threads_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "threads must be a positive integer"):
            ExtractConfig(Path("a.bam"), Path("out"), bin_size=100, threads=0)


class TestCanonicalContigs(unittest.TestCase):
    def test_select_layout_drops_non_canonical_names(self) -> None:
        layout = select_layout(
            [("chr1", 100), ("chrM", 50), ("chrX", 80), ("scaffold", 40)],
            all_contigs=False,
        )
        self.assertEqual(layout, [("chr1", 100), ("chrX", 80)])

    def test_select_layout_keeps_every_contig_when_requested(self) -> None:
        layout = select_layout(
            [("chr1", 100), ("chrM", 50)],
            all_contigs=True,
        )
        self.assertEqual(tuple(name for name, _ in layout), ("chr1", "chrM"))

    def test_canonical_spellings_including_non_human_autosomes_are_kept(self) -> None:
        for name in (
            "chr1",
            "chr22",
            "chr23",
            "chr38",
            "chrX",
            "chrY",
            "1",
            "22",
            "38",
            "X",
            "Y",
            "CHR7",
            "y",
        ):
            with self.subTest(name=name):
                self.assertTrue(is_canonical_contig(name))

    def test_non_canonical_contigs_are_dropped(self) -> None:
        for name in (
            "chr1_KI270762v1_alt",
            "chrUn_KN707606v1_decoy",
            "chr1_KI270706v1_random",
            "chrUn_GL000195v1",
            "chrEBV",
            "chrM",
            "chrMT",
            "M",
            "MT",
            "chr0",
            "0",
            "scaffold-A",
        ):
            with self.subTest(name=name):
                self.assertFalse(is_canonical_contig(name))

    def test_run_drops_non_canonical_contigs_from_bins_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bam_path = Path(tmpdir) / "sample.bam"
            prefix = Path(tmpdir) / "sample"
            header = {
                "HD": {"VN": "1.6", "SO": "coordinate"},
                "SQ": [
                    {"SN": "chr1", "LN": 200},
                    {"SN": "chr1_KI270706v1_random", "LN": 100},
                    {"SN": "chrUn_KN707606v1_decoy", "LN": 100},
                    {"SN": "chr1_KI270762v1_alt", "LN": 100},
                    {"SN": "chrEBV", "LN": 100},
                    {"SN": "chrM", "LN": 100},
                ],
            }
            reads = [
                _read("keep", 0, 0, 60),
                _read("rand", 1, 0, 60),
                _read("decoy", 2, 0, 60),
                _read("alt", 3, 0, 60),
                _read("ebv", 4, 0, 60),
                _read("mito", 5, 0, 60),
            ]
            _write_bam(bam_path, reads, header=header, index=True)
            run(ExtractConfig(bam_path, prefix, bin_size=100))

            table = _read_sample_table(parquet_path(prefix))
            smeta = SampleMetadata.from_table(table)
            self.assertEqual(smeta.contig_order, ("chr1",))
            self.assertEqual(smeta.contig_lengths, {"chr1": 200})
            self.assertEqual(smeta.qc["reads_seen"], 1)
            self.assertEqual(smeta.qc["reads_kept"], 1)
            self.assertEqual(smeta.qc["effective_reads_examined"], 1)
            self.assertEqual(smeta.qc["effective_reads_kept"], 1)
            self.assertEqual(smeta.qc["autosomal_reads_kept"], 1)
            self.assertEqual(smeta.qc["sex_reads_kept"], 0)

    def test_all_contigs_keeps_every_header_contig(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bam_path = Path(tmpdir) / "sample.bam"
            prefix = Path(tmpdir) / "sample"
            header = {
                "HD": {"VN": "1.6", "SO": "coordinate"},
                "SQ": [
                    {"SN": "chr1", "LN": 200},
                    {"SN": "chrM", "LN": 100},
                    {"SN": "scaffold-A", "LN": 100},
                ],
            }
            reads = [
                _read("keep", 0, 0, 60),
                _read("mito", 1, 0, 60),
                _read("scaf", 2, 0, 60),
            ]
            _write_bam(bam_path, reads, header=header, index=True)
            run(ExtractConfig(bam_path, prefix, bin_size=100, all_contigs=True))

            table = _read_sample_table(parquet_path(prefix))
            smeta = SampleMetadata.from_table(table)
            self.assertEqual(smeta.contig_order, ("chr1", "chrM", "scaffold-A"))
            self.assertEqual(smeta.qc["reads_kept"], 3)

    def test_header_without_canonical_contigs_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bam_path = Path(tmpdir) / "sample.bam"
            header = {
                "HD": {"VN": "1.6", "SO": "coordinate"},
                "SQ": [
                    {"SN": "chrUn_KN707606v1_decoy", "LN": 100},
                    {"SN": "chrM", "LN": 100},
                ],
            }
            _write_bam(bam_path, [], header=header, index=True)
            with self.assertRaisesRegex(ExtractError, "no canonical chromosomes"):
                run(ExtractConfig(bam_path, Path(tmpdir) / "sample", bin_size=100))


class TestEffectiveReadMetrics(unittest.TestCase):
    def test_empty_qc_contains_all_counters_initialized_to_zero(self) -> None:
        qc = empty_qc()
        self.assertEqual(set(qc.keys()), set(QC_COUNTERS))
        self.assertTrue(all(val == 0 for val in qc.values()))
        self.assertEqual(qc["effective_reads_examined"], 0)
        self.assertEqual(qc["effective_reads_kept"], 0)
        self.assertEqual(qc["autosomal_reads_kept"], 0)
        self.assertEqual(qc["sex_reads_kept"], 0)
        self.assertEqual(qc["mate_dedup_skipped"], 0)
        self.assertEqual(qc["reads_filtered"], 0)

    def test_normalize_chr(self) -> None:
        cases = {
            "chr1": "1",
            "chr22": "22",
            "1": "1",
            "22": "22",
            "chrX": "X",
            "chrx": "X",
            "X": "X",
            "x": "X",
            "chrY": "Y",
            "chry": "Y",
            "Y": "Y",
            "y": "Y",
            "chrM": "M",
            "chrEBV": "EBV",
            "scaffold-A": "scaffold-A",
        }
        for contig, expected in cases.items():
            with self.subTest(contig=contig):
                self.assertEqual(normalize_chr(contig), expected)

    def test_effective_and_autosomal_sex_qc_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bam_path = Path(tmpdir) / "sample.bam"
            prefix = Path(tmpdir) / "sample"
            header = {
                "HD": {"VN": "1.6", "SO": "coordinate"},
                "SQ": [
                    {"SN": "chr1", "LN": 200},
                    {"SN": "chr2", "LN": 200},
                    {"SN": "chrX", "LN": 200},
                    {"SN": "chrY", "LN": 200},
                    {"SN": "chrM", "LN": 100},
                ],
            }
            reads = [
                _read("c1_k1", 0, 0, 60),
                _read("c1_k2", 0, 10, 60),
                _read("c1_drop", 0, 20, 0),
                _read("c2_k1", 1, 0, 60),
                _read("c2_dup", 1, 10, 60, is_duplicate=True),
                _read("cx_k1", 2, 0, 60),
                _read("cx_k2", 2, 10, 60),
                _read("cx_k3", 2, 20, 60),
                _read("cy_k1", 3, 0, 60),
                _read("cy_k2", 3, 10, 60),
                _read("cm_k1", 4, 0, 60),
            ]
            _write_bam(bam_path, reads, header=header, index=True)
            run(ExtractConfig(bam_path, prefix, bin_size=100))

            table = _read_sample_table(parquet_path(prefix))
            qc = SampleMetadata.from_table(table).qc

            self.assertEqual(qc["reads_seen"], 10)
            self.assertEqual(qc["reads_kept"], 8)
            self.assertEqual(qc["effective_reads_examined"], 10)
            self.assertEqual(qc["effective_reads_kept"], 8)
            self.assertEqual(qc["autosomal_reads_kept"], 3)
            self.assertEqual(qc["sex_reads_kept"], 5)
            self.assertEqual(qc["filter_mapq"], 1)
            self.assertEqual(qc["filter_duplicate"], 1)
            self.assertEqual(set(qc), set(QC_COUNTERS))

    def test_effective_and_autosomal_sex_qc_metrics_all_contigs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bam_path = Path(tmpdir) / "sample.bam"
            prefix = Path(tmpdir) / "sample"
            header = {
                "HD": {"VN": "1.6", "SO": "coordinate"},
                "SQ": [
                    {"SN": "chr1", "LN": 200},
                    {"SN": "chrX", "LN": 200},
                    {"SN": "chrM", "LN": 100},
                ],
            }
            reads = [
                _read("c1_k1", 0, 0, 60),
                _read("cx_k1", 1, 0, 60),
                _read("cm_k1", 2, 0, 60),
            ]
            _write_bam(bam_path, reads, header=header, index=True)
            run(ExtractConfig(bam_path, prefix, bin_size=100, all_contigs=True))

            table = _read_sample_table(parquet_path(prefix))
            qc = SampleMetadata.from_table(table).qc

            self.assertEqual(qc["reads_seen"], 3)
            self.assertEqual(qc["reads_kept"], 3)
            self.assertEqual(qc["effective_reads_examined"], 3)
            self.assertEqual(qc["effective_reads_kept"], 3)
            self.assertEqual(qc["autosomal_reads_kept"], 1)
            self.assertEqual(qc["sex_reads_kept"], 1)
            self.assertEqual(set(qc), set(QC_COUNTERS))

    def test_skipped_contigs_only_preserves_zero_effective_reads_examined(self) -> None:
        direct_qc = build_sample_qc(
            {"reads_seen": 5, "effective_reads_examined": 0, "reads_kept": 0}
        )
        self.assertEqual(direct_qc["effective_reads_examined"], 0)
        self.assertEqual(direct_qc["effective_reads_kept"], 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            bam_path = Path(tmpdir) / "mito_only.bam"
            prefix = Path(tmpdir) / "sample"
            header = {
                "HD": {"VN": "1.6", "SO": "coordinate"},
                "SQ": [{"SN": "chr1", "LN": 200}, {"SN": "chrM", "LN": 100}],
            }
            reads = [_read("m1", 1, 0, 60), _read("m2", 1, 20, 60)]
            _write_bam(bam_path, reads, header=header, index=True)
            run(ExtractConfig(bam_path, prefix, bin_size=100))

            qc = SampleMetadata.from_table(_read_sample_table(parquet_path(prefix))).qc
            self.assertEqual(qc["reads_seen"], 0)
            self.assertEqual(qc["effective_reads_examined"], 0)
            self.assertEqual(qc["effective_reads_kept"], 0)

    def test_paired_fragment_length_bounds_and_dyad_midpoint_binning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bam_path = Path(tmpdir) / "test.bam"
            header = {
                "HD": {"VN": "1.6", "SO": "coordinate"},
                "SQ": [{"SN": "chr1", "LN": 100_000}],
            }
            r1 = _read("r1", 0, 100, 60)
            r1.flag = 0x1 | 0x2 | 0x20 | 0x40
            r1.next_reference_id = 0
            r1.next_reference_start = 110
            r1.template_length = 20

            r2 = _read("r2", 0, 200, 60)
            r2.flag = 0x1 | 0x2 | 0x20 | 0x40
            r2.next_reference_id = 0
            r2.next_reference_start = 357
            r2.template_length = 167

            r3 = _read("r3", 0, 300, 60)
            r3.flag = 0x1 | 0x2 | 0x20 | 0x40
            r3.next_reference_id = 0
            r3.next_reference_start = 990
            r3.template_length = 700

            r4 = _read("r4", 0, 400, 60)
            r4.flag = 0x1 | 0x2 | 0x20 | 0x40
            r4.next_reference_id = 0
            r4.next_reference_start = 70_390
            r4.template_length = 70_000

            _write_bam(bam_path, [r1, r2, r3, r4], header=header, index=True)
            table = extract_sample_table(
                ExtractConfig(bam_path, Path(tmpdir) / "out", bin_size=1000)
            )
            counts = table.column("count").to_numpy()
            paired_counts = table.column("paired_count").to_numpy()
            frag_len_sums = table.column("frag_len_sum").to_numpy()
            self.assertEqual(int(counts.sum()), 4)
            self.assertEqual(int(counts[0]), 3)
            self.assertEqual(int(counts[35]), 1)
            self.assertEqual(int(paired_counts.sum()), 1)
            self.assertEqual(int(frag_len_sums[0]), 167)


class TestExtractPublishedCram(unittest.TestCase):
    def test_cli_extracts_published_chr21_cram(self) -> None:
        cram = ensure_published_file(PUBLISHED_CRAM, PUBLISHED_CRAM_URL)
        fasta = ensure_published_fasta()

        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = Path(tmpdir) / "sample"
            result = CliRunner().invoke(
                app,
                [
                    "extract",
                    str(cram),
                    "--fasta",
                    str(fasta),
                    "--prefix",
                    str(prefix),
                    "--bin-size",
                    str(PUBLISHED_BIN_SIZE),
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            table = _read_sample_table(parquet_path(prefix))
            smeta = SampleMetadata.from_table(table)
            codes, names, starts, ends = reconstruct_sample_grid(smeta)
            self.assertEqual(table.num_rows, 468)
            self.assertEqual(names, ("chr21",))
            self.assertEqual(int(codes.max()), 0)
            self.assertEqual(int(starts[0]), 0)
            self.assertEqual(int(ends[-1]), CHR21_LENGTH)
            self.assertGreater(sum(table.column("count").to_pylist()), 0)
            self.assertEqual(smeta.contig_lengths, {"chr21": CHR21_LENGTH})
            self.assertEqual(smeta.bin_size, PUBLISHED_BIN_SIZE)
            self.assertEqual(smeta.qc["reads_kept"], 524_785)
            self.assertEqual(smeta.qc["effective_reads_kept"], 524_785)
            self.assertEqual(smeta.qc["autosomal_reads_kept"], 524_785)
            self.assertEqual(smeta.qc["sex_reads_kept"], 0)

            self.assertIn("frag_bands", table.column_names)
            self.assertIn("paired_count", table.column_names)
            self.assertIn("frag_len_sum", table.column_names)
            self.assertIn("short_frag_count", table.column_names)
            self.assertIn("endpoint_count", table.column_names)
            self.assertIn("frag_gc_sum", table.column_names)
            if table.schema.metadata is None:
                self.fail("expected non-None table schema metadata")
            self.assertIn(FRAGMENTOMICS_METADATA_KEY, table.schema.metadata)
            frag_meta = FragmentomicsMetadata.from_dict(
                json.loads(
                    table.schema.metadata[FRAGMENTOMICS_METADATA_KEY].decode("utf-8")
                )
            )
            self.assertTrue(frag_meta.paired_end)
            self.assertGreater(frag_meta.total_paired_fragments, 0)
            self.assertGreater(frag_meta.chromatosome_mode_bp, 50)
            self.assertGreater(sum(table.column("endpoint_count").to_pylist()), 0)
            self.assertGreater(sum(table.column("frag_gc_sum").to_pylist()), 0.0)


if __name__ == "__main__":
    unittest.main()
