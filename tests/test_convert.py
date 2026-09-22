import json
import tempfile
import unittest
from dataclasses import asdict, dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import polars_bio as pb
import pyarrow as pa
from published_fixtures import (
    CHR21_LENGTH,
    PUBLISHED_BIN_SIZE,
    PUBLISHED_CRAM,
    PUBLISHED_CRAM_URL,
    ensure_published_fasta,
    ensure_published_file,
    write_fai,
)
from typer.testing import CliRunner

from ccnv.cli import app
from ccnv.convert import (
    QC_COUNTERS,
    ConvertError,
    bin_counts,
    bin_intervals,
    is_canonical_contig,
    read_sample_parquet,
    rebin_counts,
    rebin_sample_table,
    rejection_reason,
    run,
    select_layout,
    split_thread_budget,
)
from ccnv.models import Contig, ConvertConfig, GenomeProfile, ReadFilters, parquet_path
from ccnv.schemas import (
    COORDINATE_SYSTEM,
    SAMPLE_SCHEMA,
    SCHEMA_VERSION,
    ArtifactKind,
    decode_metadata,
    metadata_for,
)


class FakeRead:
    def __init__(
        self,
        reference_start: int | None = 0,
        mapping_quality: int = 60,
        *,
        is_unmapped: bool = False,
        is_secondary: bool = False,
        is_supplementary: bool = False,
        is_duplicate: bool = False,
        is_paired: bool = False,
        is_proper_pair: bool = True,
    ) -> None:
        self.reference_start = reference_start
        self.mapping_quality = mapping_quality
        self.is_unmapped = is_unmapped
        self.is_secondary = is_secondary
        self.is_supplementary = is_supplementary
        self.is_duplicate = is_duplicate
        self.is_paired = is_paired
        self.is_proper_pair = is_proper_pair


class TestBinEdges(unittest.TestCase):
    def test_full_bins_when_length_divides_evenly(self) -> None:
        starts, ends = bin_intervals(300, 100)
        np.testing.assert_array_equal(starts, [0, 100, 200])
        np.testing.assert_array_equal(ends, [100, 200, 300])

    def test_partial_last_bin_uses_true_contig_end(self) -> None:
        starts, ends = bin_intervals(250, 100)
        np.testing.assert_array_equal(starts, [0, 100, 200])
        np.testing.assert_array_equal(ends, [100, 200, 250])

    def test_single_partial_bin_shorter_than_bin_size(self) -> None:
        starts, ends = bin_intervals(99, 100)
        np.testing.assert_array_equal(starts, [0])
        np.testing.assert_array_equal(ends, [99])

    def test_boundary_starts_land_in_expected_bins(self) -> None:
        counts = bin_counts(
            np.array([0, 99, 100, 199, 200, 249], dtype=np.int64),
            bin_size=100,
            n_bins=3,
        )
        np.testing.assert_array_equal(counts, [2, 2, 2])


class TestRebin(unittest.TestCase):
    def test_rebin_coarsens_by_summing_nested_bins(self) -> None:
        source_starts = np.array([0, 50, 100, 150], dtype=np.int64)
        source_ends = np.array([50, 100, 150, 200], dtype=np.int64)
        source_counts = np.array([10, 20, 30, 40], dtype=np.float64)

        target_starts = np.array([0, 100], dtype=np.int64)
        target_ends = np.array([100, 200], dtype=np.int64)

        rebinned = rebin_counts(
            source_starts, source_ends, source_counts, target_starts, target_ends
        )
        np.testing.assert_allclose(rebinned, [30.0, 70.0])

    def test_rebin_refines_by_splitting_uniformly(self) -> None:
        source_starts = np.array([0, 100], dtype=np.int64)
        source_ends = np.array([100, 200], dtype=np.int64)
        source_counts = np.array([30, 70], dtype=np.float64)

        target_starts = np.array([0, 50, 100, 150], dtype=np.int64)
        target_ends = np.array([50, 100, 150, 200], dtype=np.int64)

        rebinned = rebin_counts(
            source_starts, source_ends, source_counts, target_starts, target_ends
        )
        np.testing.assert_allclose(rebinned, [15.0, 15.0, 35.0, 35.0])

    def test_rebin_handles_partial_last_bin(self) -> None:
        source_starts = np.array([0, 100, 200], dtype=np.int64)
        source_ends = np.array([100, 200, 250], dtype=np.int64)
        source_counts = np.array([100, 100, 50], dtype=np.float64)

        target_starts, target_ends = bin_intervals(250, 50)
        rebinned = rebin_counts(
            source_starts, source_ends, source_counts, target_starts, target_ends
        )
        np.testing.assert_allclose(rebinned, [50.0, 50.0, 50.0, 50.0, 50.0])

    def test_rebin_rejects_contig_length_mismatch(self) -> None:
        source_starts = np.array([0, 100], dtype=np.int64)
        source_ends = np.array([100, 200], dtype=np.int64)
        source_counts = np.array([10, 20], dtype=np.float64)

        target_starts = np.array([0, 100], dtype=np.int64)
        target_ends = np.array([100, 250], dtype=np.int64)

        with self.assertRaisesRegex(ValueError, "contig length mismatch"):
            rebin_counts(
                source_starts, source_ends, source_counts, target_starts, target_ends
            )

    def test_rebin_sample_table_updates_metadata_and_counts(self) -> None:
        metadata = metadata_for(
            ArtifactKind.SAMPLE,
            contig_lengths={"chr1": 200},
            values={
                "bin_size": 100,
                "source_identity": {"id": "s1"},
                "reference_identity": {"name": "test"},
                "qc": {},
                "filters": asdict(ReadFilters()),
            },
        )
        table = pa.table(
            {
                "contig": ["chr1", "chr1"],
                "start": pa.array([0, 100], type=pa.int64()),
                "end": pa.array([100, 200], type=pa.int64()),
                "count": pa.array([10, 20], type=pa.uint64()),
            },
            schema=SAMPLE_SCHEMA.with_metadata(metadata),
        )

        rebinned = rebin_sample_table(table, target_bin_size=50)
        rebinned_meta = decode_metadata(rebinned.schema.metadata)
        self.assertEqual(rebinned_meta["bin_size"], 50)
        self.assertEqual(rebinned_meta["contig_lengths"], {"chr1": 200})
        self.assertEqual(rebinned.num_rows, 4)
        np.testing.assert_allclose(
            rebinned.column("count").to_pylist(), [5.0, 5.0, 10.0, 10.0]
        )


class TestReadFilters(unittest.TestCase):
    def test_default_filters_keep_primary_mapped_read(self) -> None:
        self.assertIsNone(rejection_reason(FakeRead(), ReadFilters()))

    def test_mapq_duplicate_secondary_and_improper_pair(self) -> None:
        filters = ReadFilters()
        self.assertEqual(
            rejection_reason(FakeRead(mapping_quality=0), filters), "filter_mapq"
        )
        self.assertEqual(
            rejection_reason(FakeRead(is_duplicate=True), filters),
            "filter_duplicate",
        )
        self.assertEqual(
            rejection_reason(FakeRead(is_secondary=True), filters),
            "filter_secondary",
        )
        self.assertEqual(
            rejection_reason(FakeRead(is_supplementary=True), filters),
            "filter_supplementary",
        )
        self.assertEqual(
            rejection_reason(FakeRead(is_paired=True, is_proper_pair=False), filters),
            "filter_improper_pair",
        )

    def test_unpaired_reads_are_not_subject_to_proper_pair_rule(self) -> None:
        self.assertIsNone(
            rejection_reason(
                FakeRead(is_paired=False, is_proper_pair=False), ReadFilters()
            )
        )

    def test_custom_min_mapping_quality_is_inclusive(self) -> None:
        filters = ReadFilters(min_mapping_quality=20)
        self.assertEqual(
            rejection_reason(FakeRead(mapping_quality=19), filters), "filter_mapq"
        )
        self.assertIsNone(rejection_reason(FakeRead(mapping_quality=20), filters))


class TestConvertAlignment(unittest.TestCase):
    def test_alignment_validation_fails_on_empty_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bam_path = Path(tmpdir) / "sample.bam"
            bam_path.touch()
            with (
                patch("ccnv.convert._scan_alignment", return_value=(None, [], [])),
                self.assertRaises(ConvertError),
            ):
                run(ConvertConfig(bam_path, Path(tmpdir) / "out", bin_size=100))

    def test_unindexed_bam_converts_without_creating_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bam_path = Path(tmpdir) / "sample.bam"
            prefix = Path(tmpdir) / "sample"
            _write_bam(bam_path, [_read("r1", 0, 0, 10)], index=False)
            self.assertFalse(_has_alignment_index(bam_path))
            run(ConvertConfig(bam_path, prefix, bin_size=100))
            self.assertFalse(_has_alignment_index(bam_path))
            table = read_sample_parquet(parquet_path(prefix))
            self.assertEqual(table.column("count").to_pylist(), [1, 0, 0])

    def test_cram_without_fasta_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cram_path = Path(tmpdir) / "sample.cram"
            cram_path.touch()
            with self.assertRaisesRegex(ConvertError, r"CRAM.*--fasta"):
                run(ConvertConfig(cram_path, Path(tmpdir) / "out", bin_size=100))

    def test_cram_missing_fasta_index_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta = Path(tmpdir) / "ref.fa"
            fasta.write_text(">chr1\n" + "A" * 250 + "\n")
            cram_path = Path(tmpdir) / "sample.cram"
            cram_path.touch()
            with self.assertRaisesRegex(ConvertError, r"FASTA index not found"):
                run(
                    ConvertConfig(
                        cram_path, Path(tmpdir) / "out", bin_size=100, fasta=fasta
                    )
                )

    def test_cram_converts_with_fasta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta = Path(tmpdir) / "ref.fa"
            fasta.write_text(">chr1\n" + "A" * 250 + "\n")
            write_fai(fasta)
            cram_path = Path(tmpdir) / "sample.cram"
            _write_cram(cram_path, [_read("r1", 0, 0, 60)], fasta)
            prefix = Path(tmpdir) / "sample"
            run(
                ConvertConfig(
                    cram_path,
                    prefix,
                    bin_size=100,
                    fasta=fasta,
                )
            )
            table = read_sample_parquet(parquet_path(prefix))
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
            run(ConvertConfig(bam_path, prefix, bin_size=100, all_contigs=True))

            table = read_sample_parquet(out_path)
            metadata = decode_metadata(table.schema.metadata)
            self.assertEqual(metadata["schema_version"], SCHEMA_VERSION)
            self.assertEqual(metadata["coordinate_system"], COORDINATE_SYSTEM)
            self.assertEqual(metadata["bin_size"], 100)
            self.assertEqual(
                metadata["contig_lengths"],
                {"chr1": 250, "scaffold-A": 99, "chrM": 50},
            )
            self.assertEqual(
                metadata["reference_identity"]["contig_order"],
                ["chr1", "scaffold-A", "chrM"],
            )
            qc = metadata["qc"]
            self.assertEqual(qc["reads_seen"], 8)
            self.assertEqual(qc["reads_kept"], 6)
            self.assertEqual(qc["filter_mapq"], 1)
            self.assertEqual(qc["filter_duplicate"], 1)
            self.assertEqual(set(qc), set(QC_COUNTERS))
            self.assertEqual(metadata["filters"]["min_mapping_quality"], 1)

            rows = table.to_pydict()
            self.assertEqual(
                rows["contig"],
                ["chr1", "chr1", "chr1", "scaffold-A", "chrM"],
            )
            self.assertEqual(rows["start"], [0, 100, 200, 0, 0])
            self.assertEqual(rows["end"], [100, 200, 250, 99, 50])
            self.assertEqual(rows["count"], [2, 1, 1, 1, 1])

    def test_convert_applies_configured_min_mapping_quality(self) -> None:
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
                ConvertConfig(
                    bam_path,
                    prefix,
                    bin_size=100,
                    filters=ReadFilters(min_mapping_quality=20),
                )
            )
            table = read_sample_parquet(parquet_path(prefix))
            metadata = decode_metadata(table.schema.metadata)
            self.assertEqual(metadata["filters"]["min_mapping_quality"], 20)
            self.assertEqual(metadata["qc"]["filter_mapq"], 1)
            self.assertEqual(metadata["qc"]["reads_kept"], 1)
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
                    "convert",
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
            table = read_sample_parquet(parquet_path(prefix))
            metadata = decode_metadata(table.schema.metadata)
            self.assertEqual(metadata["filters"]["min_mapping_quality"], 30)
            self.assertEqual(table.column("count").to_pylist(), [1, 0, 0])

    def test_cli_convert_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bam_path = Path(tmpdir) / "sample.bam"
            prefix = Path(tmpdir) / "sample"
            out_path = parquet_path(prefix)
            _write_bam(bam_path, [_read("r1", 0, 0, 60)], index=True)
            result = CliRunner().invoke(
                app,
                [
                    "convert",
                    str(bam_path),
                    "--prefix",
                    str(prefix),
                    "--bin-size",
                    "100",
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            table = read_sample_parquet(out_path)
            self.assertEqual(table.column("count").to_pylist(), [1, 0, 0])

    def test_cli_debug_loglevel_prints_convert_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bam_path = Path(tmpdir) / "sample.bam"
            prefix = Path(tmpdir) / "sample"
            _write_bam(bam_path, [_read("r1", 0, 0, 60)], index=True)
            result = CliRunner().invoke(
                app,
                [
                    "convert",
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
            self.assertIn("binned contig chr1", result.output)
            self.assertIn("writing sample parquet", result.output)

    def test_convert_debug_logs_each_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bam_path = Path(tmpdir) / "sample.bam"
            prefix = Path(tmpdir) / "sample"
            _write_bam(bam_path, [_read("r1", 0, 0, 60)], index=True)
            with self.assertLogs("ccnv.convert", level="DEBUG") as captured:
                run(
                    ConvertConfig(
                        bam_path,
                        prefix,
                        bin_size=100,
                        filters=ReadFilters(),
                    )
                )
            messages = "\n".join(captured.output)
            self.assertIn("opening alignment", messages)
            self.assertIn("binned contig chr1", messages)
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
            run(ConvertConfig(bam_path, serial_prefix, bin_size=100, threads=1))
            run(ConvertConfig(bam_path, parallel_prefix, bin_size=100, threads=3))

            serial = read_sample_parquet(parquet_path(serial_prefix))
            parallel = read_sample_parquet(parquet_path(parallel_prefix))
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
            run(ConvertConfig(bam_path, prefix, bin_size=100, threads=3))

            table = read_sample_parquet(parquet_path(prefix))
            metadata = decode_metadata(table.schema.metadata)
            self.assertEqual(
                metadata["reference_identity"]["contig_order"],
                ["chr1", "chr2", "chr3"],
            )
            self.assertEqual(
                dict.fromkeys(table.column("contig").to_pylist()),
                dict.fromkeys(["chr1", "chr2", "chr3"]),
            )
            self.assertEqual(metadata["qc"]["reads_kept"], 5)

    def test_threads_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "threads must be a positive integer"):
            ConvertConfig(Path("a.bam"), Path("out"), bin_size=100, threads=0)

    def test_thread_budget_splits_workers_and_decode_threads(self) -> None:
        cases = (
            (1, 24, 1, 1),
            (8, 1, 1, 8),
            (8, 24, 4, 2),
            (16, 24, 8, 2),
            (4, 24, 2, 2),
            (3, 24, 3, 1),
            (8, 3, 3, 2),
        )
        for budget, n_tasks, workers, decode in cases:
            with self.subTest(budget=budget, n_tasks=n_tasks):
                self.assertEqual(
                    split_thread_budget(budget, n_tasks), (workers, decode)
                )
                self.assertLessEqual(workers * decode, budget)

    def test_scan_alignment_forwards_threads(self) -> None:
        with (
            patch("ccnv.convert.pb.set_option") as set_option,
            patch("ccnv.convert.pb.scan_bam"),
            patch(
                "ccnv.convert.pb.get_metadata",
                return_value={
                    "header": {
                        "reference_sequences": '[{"name": "chr1", "length": 250}]'
                    }
                },
            ),
        ):
            from ccnv.convert import _scan_alignment

            _scan_alignment(Path("sample.bam"), "bam", None, threads=4)
            set_option.assert_called_with(pb.POLARS_BIO_MAX_THREADS, "4")


class TestCanonicalContigs(unittest.TestCase):
    def test_select_layout_drops_non_canonical_names(self) -> None:
        layout = select_layout(
            ["chr1", "chrM", "chrX", "scaffold"],
            [100, 50, 80, 40],
            all_contigs=False,
        )
        self.assertEqual(layout, (Contig("chr1", 100), Contig("chrX", 80)))
        profile = GenomeProfile(layout)
        self.assertEqual(profile.names, ("chr1", "chrX"))
        self.assertEqual(profile.length_map(), {"chr1": 100, "chrX": 80})

    def test_select_layout_keeps_every_contig_when_requested(self) -> None:
        layout = select_layout(
            ["chr1", "chrM"],
            [100, 50],
            all_contigs=True,
        )
        self.assertEqual(tuple(contig.name for contig in layout), ("chr1", "chrM"))

    def test_canonical_spellings_are_kept(self) -> None:
        for name in ("chr1", "chr22", "chrX", "chrY", "1", "22", "X", "Y", "CHR7", "y"):
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
            "chr23",
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
            run(ConvertConfig(bam_path, prefix, bin_size=100))

            table = read_sample_parquet(parquet_path(prefix))
            metadata = decode_metadata(table.schema.metadata)
            self.assertEqual(set(table.column("contig").to_pylist()), {"chr1"})
            self.assertEqual(metadata["contig_lengths"], {"chr1": 200})
            self.assertEqual(metadata["reference_identity"]["contig_order"], ["chr1"])
            self.assertEqual(metadata["qc"]["reads_seen"], 1)
            self.assertEqual(metadata["qc"]["reads_kept"], 1)

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
            run(ConvertConfig(bam_path, prefix, bin_size=100, all_contigs=True))

            table = read_sample_parquet(parquet_path(prefix))
            metadata = decode_metadata(table.schema.metadata)
            self.assertEqual(
                metadata["reference_identity"]["contig_order"],
                ["chr1", "chrM", "scaffold-A"],
            )
            self.assertEqual(metadata["qc"]["reads_kept"], 3)

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
            with self.assertRaisesRegex(ConvertError, "no canonical chromosomes"):
                run(ConvertConfig(bam_path, Path(tmpdir) / "sample", bin_size=100))


class TestConvertPublishedCram(unittest.TestCase):
    def test_cli_converts_published_chr21_cram(self) -> None:
        cram = ensure_published_file(PUBLISHED_CRAM, PUBLISHED_CRAM_URL)
        fasta = ensure_published_fasta()

        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = Path(tmpdir) / "sample"
            result = CliRunner().invoke(
                app,
                [
                    "convert",
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
            table = read_sample_parquet(parquet_path(prefix))
            metadata = decode_metadata(table.schema.metadata)
            self.assertEqual(table.num_rows, 468)
            self.assertEqual(set(table.column("contig").to_pylist()), {"chr21"})
            self.assertEqual(table.column("start").to_pylist()[0], 0)
            self.assertEqual(table.column("end").to_pylist()[-1], CHR21_LENGTH)
            self.assertGreater(sum(table.column("count").to_pylist()), 0)
            self.assertEqual(metadata["contig_lengths"], {"chr21": CHR21_LENGTH})
            self.assertEqual(metadata["bin_size"], PUBLISHED_BIN_SIZE)
            self.assertEqual(metadata["source_identity"]["format"], "cram")
            self.assertEqual(metadata["qc"]["reads_kept"], 1_049_534)


@dataclass
class _Read:
    name: str
    reference_id: int
    start: int
    mapq: int
    is_duplicate: bool = False


def _read(
    name: str,
    reference_id: int,
    start: int,
    mapq: int,
    *,
    is_duplicate: bool = False,
) -> _Read:
    return _Read(
        name=name,
        reference_id=reference_id,
        start=start,
        mapq=mapq,
        is_duplicate=is_duplicate,
    )


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
    reads: list[_Read],
    *,
    header: dict[str, object] | None = None,
    index: bool = False,
) -> None:
    bam_header = header or {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "chr1", "LN": 250}],
    }
    sq_list = bam_header.get("SQ", [{"SN": "chr1", "LN": 250}])
    ref_seqs = [{"name": sq["SN"], "length": sq["LN"]} for sq in sq_list]
    ref_id_to_name = {idx: sq["SN"] for idx, sq in enumerate(sq_list)}
    sorted_reads = sorted(reads, key=lambda r: (r.reference_id, r.start))
    if sorted_reads:
        names = [r.name for r in sorted_reads]
        chroms = [
            ref_id_to_name.get(r.reference_id, f"chr{r.reference_id + 1}")
            for r in sorted_reads
        ]
        starts = [r.start for r in sorted_reads]
        ends = [r.start + 10 for r in sorted_reads]
        flags = [1024 if r.is_duplicate else 0 for r in sorted_reads]
        mapqs = [r.mapq for r in sorted_reads]
        cigars = ["10M" for _ in sorted_reads]
        sequences = ["ACGTACGTAC" for _ in sorted_reads]
        qualities = ["I" * 10 for _ in sorted_reads]
        mate_chroms = [None for _ in sorted_reads]
        mate_starts = [None for _ in sorted_reads]
        tlen = [0 for _ in sorted_reads]
    else:
        names = []
        chroms = []
        starts = []
        ends = []
        flags = []
        mapqs = []
        cigars = []
        sequences = []
        qualities = []
        mate_chroms = []
        mate_starts = []
        tlen = []

    df = pl.DataFrame(
        {
            "name": pl.Series(names, dtype=pl.String),
            "chrom": pl.Series(chroms, dtype=pl.String),
            "start": pl.Series(starts, dtype=pl.UInt32),
            "end": pl.Series(ends, dtype=pl.UInt32),
            "flags": pl.Series(flags, dtype=pl.UInt32),
            "cigar": pl.Series(cigars, dtype=pl.String),
            "mapping_quality": pl.Series(mapqs, dtype=pl.UInt32),
            "mate_chrom": pl.Series(mate_chroms, dtype=pl.String),
            "mate_start": pl.Series(mate_starts, dtype=pl.UInt32),
            "sequence": pl.Series(sequences, dtype=pl.String),
            "quality_scores": pl.Series(qualities, dtype=pl.String),
            "template_length": pl.Series(tlen, dtype=pl.Int32),
        }
    )
    meta_header = {
        "file_format_version": "1.6",
        "reference_sequences": json.dumps(ref_seqs),
    }
    pb.set_option(pb.POLARS_BIO_COORDINATE_SYSTEM_ZERO_BASED, True)
    pb.set_source_metadata(df, format="bam", path=str(path), header=meta_header)
    pb.write_bam(df, str(path), sort_on_write=True)


def _write_cram(
    path: Path,
    reads: list[_Read],
    fasta_path: Path,
    *,
    header: dict[str, object] | None = None,
) -> None:
    bam_header = header or {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "chr1", "LN": 250}],
    }
    sq_list = bam_header.get("SQ", [{"SN": "chr1", "LN": 250}])
    ref_seqs = [{"name": sq["SN"], "length": sq["LN"]} for sq in sq_list]
    ref_id_to_name = {idx: sq["SN"] for idx, sq in enumerate(sq_list)}
    sorted_reads = sorted(reads, key=lambda r: (r.reference_id, r.start))
    if sorted_reads:
        names = [r.name for r in sorted_reads]
        chroms = [
            ref_id_to_name.get(r.reference_id, f"chr{r.reference_id + 1}")
            for r in sorted_reads
        ]
        starts = [r.start for r in sorted_reads]
        ends = [r.start + 10 for r in sorted_reads]
        flags = [1024 if r.is_duplicate else 0 for r in sorted_reads]
        mapqs = [r.mapq for r in sorted_reads]
        cigars = ["10M" for _ in sorted_reads]
        sequences = ["ACGTACGTAC" for _ in sorted_reads]
        qualities = ["I" * 10 for _ in sorted_reads]
        mate_chroms = [None for _ in sorted_reads]
        mate_starts = [None for _ in sorted_reads]
        tlen = [0 for _ in sorted_reads]
    else:
        names = []
        chroms = []
        starts = []
        ends = []
        flags = []
        mapqs = []
        cigars = []
        sequences = []
        qualities = []
        mate_chroms = []
        mate_starts = []
        tlen = []

    df = pl.DataFrame(
        {
            "name": pl.Series(names, dtype=pl.String),
            "chrom": pl.Series(chroms, dtype=pl.String),
            "start": pl.Series(starts, dtype=pl.UInt32),
            "end": pl.Series(ends, dtype=pl.UInt32),
            "flags": pl.Series(flags, dtype=pl.UInt32),
            "cigar": pl.Series(cigars, dtype=pl.String),
            "mapping_quality": pl.Series(mapqs, dtype=pl.UInt32),
            "mate_chrom": pl.Series(mate_chroms, dtype=pl.String),
            "mate_start": pl.Series(mate_starts, dtype=pl.UInt32),
            "sequence": pl.Series(sequences, dtype=pl.String),
            "quality_scores": pl.Series(qualities, dtype=pl.String),
            "template_length": pl.Series(tlen, dtype=pl.Int32),
        }
    )
    meta_header = {
        "file_format_version": "1.6",
        "reference_sequences": json.dumps(ref_seqs),
    }
    pb.set_option(pb.POLARS_BIO_COORDINATE_SYSTEM_ZERO_BASED, True)
    pb.set_source_metadata(df, format="cram", path=str(path), header=meta_header)
    pb.write_cram(df, str(path), reference_path=str(fasta_path), sort_on_write=True)


if __name__ == "__main__":
    unittest.main()
