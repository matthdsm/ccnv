"""Unit tests for genomic binning, sample/cohort rebinning, and streaming cohort assembly."""

from __future__ import annotations

import json
import logging
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pysam
from helpers import make_sample_table, write_synthetic_sample
from published_fixtures import (
    PUBLISHED_BIN_SIZE,
    PUBLISHED_CANONICAL_BINS,
    ensure_published_cram,
    ensure_published_fasta,
)

from ccnv.extract import extract_sample_table
from ccnv.features import rebin_counts_matrix
from ccnv.features.binning import bin_intervals, compute_bin_exposure, rebin_counts
from ccnv.features.cohort import (
    Cohort,
    alloc_cohort_counts,
    estimate_size_factors,
    load_and_assemble_cohort,
    mask_low_coverage_per_contig,
    rebin_cohort,
    validate_cohort,
)
from ccnv.features.rebinning import rebin_sample_buffers
from ccnv.io.parquet import read_sample_buffers, write_sample_parquet
from ccnv.models import (
    FRAGMENTOMICS_METADATA_KEY,
    NUM_FRAG_BANDS,
    SAMPLE_SCHEMA,
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    BinCompatibilityError,
    CcnvError,
    ExtractConfig,
    FragmentomicsMetadata,
    InvalidInputError,
    ReferenceError,
    SampleBuffers,
    SampleMetadata,
    SamplesheetEntry,
    decode_metadata,
    fixed_size_list_to_numpy,
    parquet_path,
    reconstruct_sample_grid,
    validate_sample_table,
)


def _rebin_table(table: pa.Table, target_bin_size: int) -> pa.Table:
    return rebin_sample_buffers(
        SampleBuffers.from_table(table), target_bin_size
    ).to_table(base_metadata=table.schema.metadata)


def _write_bam(
    path: Path, contigs: list[tuple[str, int]], *, paired: bool = True
) -> None:
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": name, "LN": length} for name, length in contigs],
    }
    quals = pysam.qualitystring_to_array("I" * 50)
    seq = "ACGTGCATGC" * 5

    def _seg(
        qname: str, flag: int, rid: int, pos: int, mrid: int, mpos: int, tlen: int
    ) -> pysam.AlignedSegment:
        rec = pysam.AlignedSegment()
        rec.query_name, rec.query_sequence, rec.flag = qname, seq, flag
        rec.reference_id, rec.reference_start = rid, pos
        rec.mapping_quality, rec.cigar = 60, ((0, 50),)
        rec.next_reference_id, rec.next_reference_start = mrid, mpos
        rec.template_length, rec.query_qualities = tlen, quals
        return rec

    records: list[pysam.AlignedSegment] = []
    if paired:
        rng = np.random.default_rng(17)
        read_idx = 0
        for ref_id, (_, length) in enumerate(contigs):
            positions = rng.integers(100, max(200, length - 400), size=24)
            inserts = rng.integers(90, 260, size=24)
            for pos_raw, tlen_raw in zip(positions, inserts, strict=True):
                pos, tlen = int(pos_raw), int(tlen_raw)
                mpos = pos + tlen - 50
                records.append(
                    _seg(f"frag_{read_idx}", 0x63, ref_id, pos, ref_id, mpos, tlen)
                )
                records.append(
                    _seg(f"frag_{read_idx}", 0x93, ref_id, mpos, ref_id, pos, -tlen)
                )
                read_idx += 1
        records.sort(key=lambda r: (r.reference_id, r.reference_start))
    else:
        records = [
            _seg(f"se_{i}", 0, 0, p, -1, -1, 0)
            for i, p in enumerate((50, 5_100, 12_300))
        ]
    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        for rec in records:
            bam.write(rec)
    pysam.index(str(path))


def _make_paired_cohort_files(
    tmp_path: Path, *, n_samples: int = 8, bin_size: int = 100
) -> list[Path]:
    rng = np.random.default_rng(42)
    contigs = ["chr1"] * 6 + ["chr2"] * 6 + ["chrX"] * 4 + ["chrY"] * 4
    starts = [i * bin_size for i in range(6)] * 2 + [i * bin_size for i in range(4)] * 2
    ends = [s + bin_size for s in starts]
    cl = {
        "chr1": 6 * bin_size,
        "chr2": 6 * bin_size,
        "chrX": 4 * bin_size,
        "chrY": 4 * bin_size,
    }
    paths: list[Path] = []
    for idx in range(n_samples):
        is_male = idx % 2 == 1
        sex_counts = ([20] * 4 + [18] * 4) if is_male else ([40] * 4 + [0] * 4)
        counts = np.array([40 + idx] * 12 + sex_counts, dtype=np.uint32)
        frag_lens = [
            rng.integers(145, 190, size=int(c)).tolist() if c > 0 else []
            for c in counts
        ]
        p = tmp_path / f"sample_{idx:02d}.ccnv"
        ep = np.minimum(counts, 6).astype(np.uint32)
        write_synthetic_sample(
            p,
            counts,
            contigs=contigs,
            starts=starts,
            ends=ends,
            contig_lengths=cl,
            bin_size=bin_size,
            fragment_lengths=frag_lens,
            endpoint_counts=ep,
        )
        paths.append(p)
    return paths


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

    def test_bin_exposure_reflects_partial_last_bin_width(self) -> None:
        starts, ends = bin_intervals(250, 100)
        exp = compute_bin_exposure(starts, ends, 100)
        np.testing.assert_allclose(exp, [1.0, 1.0, 0.5])


class TestRebin(unittest.TestCase):
    def test_rebin_coarsens_by_summing_nested_bins(self) -> None:
        s0, e0 = np.array([0, 50, 100, 150]), np.array([50, 100, 150, 200])
        c0 = np.array([10.0, 20.0, 30.0, 40.0])
        rebinned = rebin_counts(s0, e0, c0, np.array([0, 100]), np.array([100, 200]))
        np.testing.assert_allclose(rebinned, [30.0, 70.0])

    def test_rebin_refines_by_splitting_uniformly(self) -> None:
        s1, e1 = np.array([0, 50, 100, 150]), np.array([50, 100, 150, 200])
        rebinned = rebin_counts(
            np.array([0, 100]), np.array([100, 200]), np.array([30.0, 70.0]), s1, e1
        )
        np.testing.assert_allclose(rebinned, [15.0, 15.0, 35.0, 35.0])

    def test_rebin_handles_partial_last_bin(self) -> None:
        s0, e0 = np.array([0, 100, 200]), np.array([100, 200, 250])
        s1, e1 = bin_intervals(250, 50)
        rebinned = rebin_counts(s0, e0, np.array([100.0, 100.0, 50.0]), s1, e1)
        np.testing.assert_allclose(rebinned, [50.0, 50.0, 50.0, 50.0, 50.0])

    def test_rebin_rejects_contig_length_mismatch(self) -> None:
        s0, e0 = np.array([0, 100]), np.array([100, 200])
        with self.assertRaisesRegex(ValueError, "contig length mismatch"):
            rebin_counts(
                s0, e0, np.array([10.0, 20.0]), s0, np.array([100, 250], dtype=np.int64)
            )

    def test_rebin_sample_buffers_coarsens_and_rejects_upsampling(self) -> None:
        table = make_sample_table(
            [5, 5, 10, 10],
            contig_lengths={"chr1": 200},
            bin_size=50,
        )
        rebinned_bufs = rebin_sample_buffers(
            SampleBuffers.from_table(table), target_bin_size=100
        )
        rebinned = rebinned_bufs.to_table(base_metadata=table.schema.metadata)
        rebinned_meta = decode_metadata(rebinned.schema.metadata)
        self.assertEqual(rebinned_meta["bin_size"], 100)
        self.assertEqual(rebinned_meta["contig_lengths"], {"chr1": 200})
        self.assertEqual(rebinned.num_rows, 2)
        self.assertEqual(rebinned.column("count").to_pylist(), [10, 20])
        with self.assertRaises(BinCompatibilityError):
            rebin_sample_buffers(rebinned_bufs, target_bin_size=50)


class TestFeaturesBinning(unittest.TestCase):
    def test_bin_intervals_exact_and_partial(self) -> None:
        starts, ends = bin_intervals(250, 100)
        np.testing.assert_array_equal(starts, [0, 100, 200])
        np.testing.assert_array_equal(ends, [100, 200, 250])

    def test_bin_intervals_invalid(self) -> None:
        with self.assertRaises(ValueError):
            bin_intervals(0, 100)
        with self.assertRaises(ValueError):
            bin_intervals(100, 0)

    def test_compute_bin_exposure(self) -> None:
        exp = compute_bin_exposure(
            np.array([0, 100, 200]), np.array([100, 200, 250]), 100
        )
        np.testing.assert_allclose(exp, [1.0, 1.0, 0.5])

    def test_compute_bin_exposure_handles_partial_terminal_bins_and_validates(
        self,
    ) -> None:
        starts = np.array([0, 1000, 2000], dtype=np.uint32)
        ends = np.array([1000, 2000, 2350], dtype=np.uint32)
        exp = compute_bin_exposure(starts, ends, 1000)
        self.assertEqual(exp.dtype, np.float32)
        np.testing.assert_allclose(exp, [1.0, 1.0, 0.35], rtol=1e-6)
        with self.assertRaises(InvalidInputError):
            compute_bin_exposure(starts, ends, 0)
        with self.assertRaises(InvalidInputError):
            compute_bin_exposure(
                np.array([500], np.uint32), np.array([400], np.uint32), 1000
            )

    def test_rebin_counts_coarsening(self) -> None:
        s0, e0 = np.array([0, 50, 100, 150]), np.array([50, 100, 150, 200])
        res = rebin_counts(
            s0,
            e0,
            np.array([10.0, 20.0, 30.0, 40.0]),
            np.array([0, 100]),
            np.array([100, 200]),
        )
        np.testing.assert_allclose(res, [30.0, 70.0])

    def test_rebin_counts_refining(self) -> None:
        s1, e1 = np.array([0, 50, 100, 150]), np.array([50, 100, 150, 200])
        res = rebin_counts(
            np.array([0, 100]), np.array([100, 200]), np.array([40.0, 60.0]), s1, e1
        )
        np.testing.assert_allclose(res, [20.0, 20.0, 30.0, 30.0])

    def test_rebin_counts_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            rebin_counts(
                np.array([0]),
                np.array([100]),
                np.array([1.0]),
                np.array([0]),
                np.array([150]),
            )


class TestCohortCompatibility(unittest.TestCase):
    def test_rejects_bin_size_mismatch(self) -> None:
        left = make_sample_table([10, 10], bin_size=100)
        right = make_sample_table([5, 5, 5, 5], bin_size=50)
        with self.assertRaisesRegex(ReferenceError, "bin_size"):
            validate_cohort((left, right))

    def test_rejects_coordinate_mismatch(self) -> None:
        lens = {"chr1": 100, "chr2": 100}
        left = make_sample_table(
            [10, 10],
            contigs=["chr1", "chr2"],
            starts=[0, 0],
            ends=[100, 100],
            contig_lengths=lens,
        )
        right = make_sample_table(
            [10, 10],
            contigs=["chr2", "chr1"],
            starts=[0, 0],
            ends=[100, 100],
            contig_lengths=lens,
        )
        with self.assertRaisesRegex(ReferenceError, "bin coordinates"):
            validate_cohort((left, right))

    def test_mixed_native_bin_sizes_succeed_when_target_bin_size_set(self) -> None:
        left = make_sample_table([10, 10], bin_size=100)
        right = make_sample_table([5, 5, 5, 5], bin_size=50)
        cohort = validate_cohort((left, right), target_bin_size=100)
        self.assertEqual(cohort.bin_size, 100)
        self.assertEqual(cohort.counts.shape, (2, 2))
        np.testing.assert_allclose(cohort.counts[:, 0], [10.0, 10.0])
        np.testing.assert_allclose(cohort.counts[:, 1], [10.0, 10.0])

    def test_mismatched_contig_lengths_fail_even_with_target_bin_size(self) -> None:
        left = make_sample_table([10, 10], contig_lengths={"chr1": 200}, bin_size=100)
        right = make_sample_table([10], contig_lengths={"chr1": 100}, bin_size=100)
        with self.assertRaisesRegex(ReferenceError, "contig_lengths do not match"):
            validate_cohort((left, right), target_bin_size=100)

    def test_accepts_matching_nonhuman_layout(self) -> None:
        contigs = ["scaffold-A", "scaffold-A", "lgX"]
        starts = [0, 100, 0]
        ends = [100, 200, 100]
        lengths = {"scaffold-A": 200, "lgX": 100}
        tables = [
            make_sample_table(
                [8, 9, 2],
                contigs=contigs,
                starts=starts,
                ends=ends,
                contig_lengths=lengths,
            )
            for _ in range(3)
        ]
        cohort = validate_cohort(tuple(tables))
        self.assertEqual(cohort.counts.shape, (3, 3))
        self.assertEqual(cohort.contigs, ("scaffold-A", "scaffold-A", "lgX"))

    def test_mask_low_coverage_per_contig(self) -> None:
        counts = np.array([[0.0], [0.01], [100.0], [110.0], [100.0]], dtype=np.float32)
        exp = np.ones(5, dtype=np.float32)
        mask = mask_low_coverage_per_contig(counts, ("chr1",) * 5, exp)
        self.assertFalse(mask[0])
        self.assertFalse(mask[1])
        self.assertTrue(mask[2])
        self.assertTrue(mask[3])
        self.assertTrue(mask[4])

    def test_validate_cohort_and_rebin(self) -> None:
        t1 = make_sample_table([10, 20], contig_lengths={"chr1": 200}, bin_size=100)
        t2 = make_sample_table(
            [5, 5, 10, 10], contig_lengths={"chr1": 200}, bin_size=50
        )
        cohort = validate_cohort((t1, t2), target_bin_size=100)
        self.assertIsInstance(cohort, Cohort)
        self.assertEqual(cohort.counts.shape, (2, 2))
        np.testing.assert_allclose(cohort.counts[:, 0], [10.0, 20.0])
        np.testing.assert_allclose(cohort.counts[:, 1], [10.0, 20.0])

    def test_rebin_sample_buffers(self) -> None:
        t = make_sample_table([5, 5, 10, 10], contig_lengths={"chr1": 200}, bin_size=50)
        src = SampleBuffers.from_table(t)
        bufs = rebin_sample_buffers(src, 100)
        self.assertEqual(bufs.num_rows, 2)
        np.testing.assert_allclose(bufs.count, [10, 20])
        with self.assertRaises(BinCompatibilityError):
            rebin_sample_buffers(src, 25)


class TestCohortLoading(unittest.TestCase):
    def test_load_cohort_parallel_equivalence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths: list[Path] = []
            for i in range(4):
                p = Path(tmpdir) / f"sample_{i}.ccnv"
                t = make_sample_table(
                    [10, 20, 30],
                    contigs=["chr1", "chr1", "chr2"],
                    starts=[0, 1000, 0],
                    ends=[1000, 2000, 1000],
                    bin_size=1000,
                )
                pq.write_table(t, p)
                paths.append(p)

            entries = tuple(SamplesheetEntry(filepath=p) for p in paths)
            cohort_seq = load_and_assemble_cohort(entries, threads=1)
            cohort_par = load_and_assemble_cohort(entries, threads=2)

            np.testing.assert_array_equal(cohort_seq.counts, cohort_par.counts)
            self.assertEqual(cohort_seq.contigs, cohort_par.contigs)
            np.testing.assert_array_equal(cohort_seq.starts, cohort_par.starts)
            np.testing.assert_array_equal(cohort_seq.ends, cohort_par.ends)
            self.assertEqual(cohort_seq.contig_lengths, cohort_par.contig_lengths)
            self.assertEqual(cohort_seq.bin_size, cohort_par.bin_size)

            rebinned = rebin_cohort(cohort_seq, 2000)
            self.assertEqual(rebinned.bin_size, 2000)
            self.assertEqual(rebinned.counts.shape, (2, 4))

    def test_streaming_cohort_ram_and_memmap_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            paths = _make_paired_cohort_files(tmp_path, n_samples=8, bin_size=100)
            entries = [
                SamplesheetEntry(filepath=p, sex="M" if i % 2 == 1 else "F")
                for i, p in enumerate(paths)
            ]

            cohort_ram = load_and_assemble_cohort(
                entries, target_bin_size=200, threads=2, max_batch_memory_mb=1
            )
            self.assertEqual(cohort_ram.counts.shape, (10, 8))
            self.assertEqual(cohort_ram.counts.dtype, np.float32)
            self.assertTrue(bool(cohort_ram.counts.flags["F_CONTIGUOUS"]))
            self.assertNotIsInstance(cohort_ram.counts, np.memmap)
            self.assertIsNotNone(cohort_ram.ref_frag_len)
            self.assertIsNotNone(cohort_ram.ref_ep_rate)
            if cohort_ram.ref_frag_len is None or cohort_ram.ref_ep_rate is None:
                self.fail("expected non-None fragmentomic cohort summaries")
            self.assertEqual(cohort_ram.ref_frag_len.dtype, np.float32)
            self.assertEqual(cohort_ram.ref_ep_rate.dtype, np.float32)

            cohort_mmap = load_and_assemble_cohort(
                entries, target_bin_size=200, threads=4, max_batch_memory_mb=0
            )
            self.assertIsInstance(cohort_mmap.counts, np.memmap)
            self.assertEqual(cohort_mmap.counts.shape, (10, 8))
            self.assertEqual(cohort_mmap.counts.dtype, np.float32)
            self.assertTrue(bool(cohort_mmap.counts.flags["F_CONTIGUOUS"]))
            np.testing.assert_allclose(cohort_mmap.counts, cohort_ram.counts)
            self.assertIsNotNone(cohort_mmap.ref_frag_len)
            self.assertIsNotNone(cohort_mmap.ref_ep_rate)
            if cohort_mmap.ref_frag_len is None or cohort_mmap.ref_ep_rate is None:
                self.fail("expected non-None memmap fragmentomic cohort summaries")
            np.testing.assert_allclose(
                cohort_mmap.ref_frag_len, cohort_ram.ref_frag_len
            )
            np.testing.assert_allclose(cohort_mmap.ref_ep_rate, cohort_ram.ref_ep_rate)

    def test_streaming_cohort_rejects_upsampling_mixed_layout_and_corrupt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            paths = _make_paired_cohort_files(tmp_path, n_samples=2, bin_size=200)
            entries = [SamplesheetEntry(filepath=p) for p in paths]

            with self.assertRaises(BinCompatibilityError):
                load_and_assemble_cohort(entries, target_bin_size=100)

            se_path = tmp_path / "single_end.ccnv"
            starts = [i * 200 for i in range(6)] * 2 + [i * 200 for i in range(4)] * 2
            cl = {"chr1": 1200, "chr2": 1200, "chrX": 800, "chrY": 800}
            write_synthetic_sample(
                se_path,
                [20] * 20,
                contigs=["chr1"] * 6 + ["chr2"] * 6 + ["chrX"] * 4 + ["chrY"] * 4,
                starts=starts,
                ends=[s + 200 for s in starts],
                contig_lengths=cl,
                bin_size=200,
                fragment_lengths=None,
            )
            mixed = [SamplesheetEntry(paths[0]), SamplesheetEntry(se_path)]
            with self.assertRaisesRegex(ReferenceError, "mixed"):
                load_and_assemble_cohort(mixed, target_bin_size=200)

            corrupt = tmp_path / "corrupt.ccnv"
            corrupt.write_bytes(b"not a parquet file")
            with self.assertRaises(ReferenceError):
                load_and_assemble_cohort([SamplesheetEntry(corrupt)], 200)

    def test_defect5_alloc_cohort_counts_memmap_threshold_exact_bytes(self) -> None:
        in_ram = alloc_cohort_counts(262_144, 1, max_batch_memory_mb=1)
        self.assertFalse(isinstance(in_ram, np.memmap))
        spilled = alloc_cohort_counts(262_145, 1, max_batch_memory_mb=1)
        self.assertIsInstance(spilled, np.memmap)


class TestCohortDepthAndRebinning(unittest.TestCase):
    def test_normalize_depth_median_of_ratios_matches_deseq2_and_resists_trisomy(
        self,
    ) -> None:
        base = np.full((100, 4), 100.0, dtype=np.float64)
        base[:, 0] = 200.0
        base[0:30, 0] = 2000.0
        contigs = ["chr1"] * 100
        starts = np.arange(100, dtype=np.int64) * 100_000
        ends = starts + 100_000

        log_gm = np.mean(np.log(base), axis=1, keepdims=True)
        expected_sf = np.median(base / np.exp(log_gm), axis=0)
        sf = estimate_size_factors(base, contigs=contigs, starts=starts, ends=ends)
        np.testing.assert_allclose(
            sf / np.median(sf), expected_sf / np.median(expected_sf), rtol=1e-12
        )
        normed = base / sf
        np.testing.assert_allclose(normed[30:, 0], normed[30:, 1], rtol=1e-12)

    def test_normalize_depth_sparse_zero_fallback(self) -> None:
        sparse = np.array([[10.0, 0.0], [0.0, 20.0]], dtype=np.float64)
        sf = estimate_size_factors(
            sparse,
            contigs=["chr1", "chr2"],
            starts=np.array([0, 0]),
            ends=np.array([100_000, 100_000]),
        )
        normed = sparse / sf
        np.testing.assert_allclose(normed.sum(axis=0), [15.0, 15.0], rtol=1e-12)

    def test_rebin_cohort_polars_fast_path_coarsening(self) -> None:
        cohort = Cohort(
            np.arange(1, 9, dtype=np.float64).reshape(4, 2),
            ("chr1",) * 4,
            np.array([0, 50, 100, 150], dtype=np.int64),
            np.array([50, 100, 150, 180], dtype=np.int64),
            {"chr1": 180},
            50,
            ({"id": "a"}, {"id": "b"}),
            (Path("a.ccnv"), Path("b.ccnv")),
        )
        rebinned = rebin_cohort(cohort, 100)
        np.testing.assert_array_equal(rebinned.starts, [0, 100])
        np.testing.assert_array_equal(rebinned.ends, [100, 180])
        np.testing.assert_allclose(rebinned.counts, [[4.0, 6.0], [12.0, 14.0]])

    def test_estimate_size_factors_validation(self) -> None:
        c2, s2, e2 = ["chr1", "chr2"], np.array([0, 0]), np.array([100_000, 100_000])
        m22 = np.array([[1.0, 2.0], [3.0, 4.0]])
        with self.assertRaises(InvalidInputError):
            estimate_size_factors(m22)
        with self.assertRaises(InvalidInputError):
            estimate_size_factors(np.array([1.0, 2.0]), contigs=c2, starts=s2, ends=e2)
        with self.assertRaises(InvalidInputError):
            estimate_size_factors(
                np.array([[-1.0, 2.0], [3.0, 4.0]]), contigs=c2, starts=s2, ends=e2
            )
        with self.assertRaises(InvalidInputError):
            estimate_size_factors(
                np.array([[0.0, 2.0], [0.0, 4.0]]), contigs=c2, starts=s2, ends=e2
            )
        with self.assertRaises(InvalidInputError):
            estimate_size_factors(
                m22, mask=np.array([True]), contigs=c2, starts=s2, ends=e2
            )
        with self.assertRaises(InvalidInputError):
            estimate_size_factors(
                m22, mask=np.array([False, False]), contigs=c2, starts=s2, ends=e2
            )

    def test_rebin_sample_table_polars_fast_path(self) -> None:
        table = make_sample_table(
            [10, 20, 30, 40, 0, 0],
            contigs=["chr1", "chr1", "chr1", "chr1", "chr2", "chr2"],
            starts=[0, 50, 100, 150, 0, 50],
            ends=[50, 100, 150, 180, 50, 100],
            contig_lengths={"chr1": 180, "chr2": 100},
            bin_size=50,
        )
        rebinned = _rebin_table(table, 100)
        codes, names, starts, ends = reconstruct_sample_grid(
            SampleMetadata.from_table(rebinned)
        )
        self.assertEqual(starts.tolist(), [0, 100, 0])
        self.assertEqual(ends.tolist(), [100, 180, 100])
        self.assertEqual(rebinned.column("count").to_pylist(), [30, 70, 0])
        self.assertEqual([names[int(c)] for c in codes], ["chr1", "chr1", "chr2"])

    def test_rebin_sample_table_boundary_crossing_overlap_redistribution(self) -> None:
        table = make_sample_table(
            [20, 10, 30, 0, 40, 0, 0],
            contigs=["chr1"] * 7,
            starts=[0, 20, 40, 60, 80, 100, 120],
            ends=[20, 40, 60, 80, 100, 120, 140],
            contig_lengths={"chr1": 140},
            bin_size=20,
        )
        rebinned = _rebin_table(table, 50)
        _, _, starts, ends = reconstruct_sample_grid(
            SampleMetadata.from_table(rebinned)
        )
        self.assertEqual(starts.tolist(), [0, 50, 100])
        self.assertEqual(ends.tolist(), [50, 100, 140])
        np.testing.assert_allclose(
            rebinned.column("count").to_numpy(), [45.0, 55.0, 0.0]
        )

    def test_rebin_cohort_boundary_crossing_overlap_redistribution(self) -> None:
        counts = np.array(
            [[30.0, 60.0], [30.0, 60.0], [60.0, 120.0], [30.0, 60.0], [0.0, 0.0]],
            dtype=np.float32,
        )
        cohort = Cohort(
            counts,
            ("chr1",) * 5,
            np.array([0, 30, 60, 90, 120], dtype=np.int64),
            np.array([30, 60, 90, 120, 150], dtype=np.int64),
            {"chr1": 150},
            30,
            ({"id": "a"}, {"id": "b"}),
            (Path("a.ccnv"), Path("b.ccnv")),
        )
        rebinned = rebin_cohort(cohort, 50)
        np.testing.assert_array_equal(rebinned.starts, [0, 50, 100])
        np.testing.assert_array_equal(rebinned.ends, [50, 100, 150])
        np.testing.assert_allclose(
            rebinned.counts, [[50.0, 100.0], [80.0, 160.0], [20.0, 40.0]]
        )

    def test_rebin_sample_table_preserves_fragmentomic_columns(self) -> None:
        fl = [[120, 130], [140], [150, 160], [170]]
        table = make_sample_table(
            [10, 20, 30, 40],
            contigs=["chr1"] * 4,
            starts=[0, 50, 100, 150],
            ends=[50, 100, 150, 200],
            contig_lengths={"chr1": 200},
            bin_size=50,
            fragment_lengths=fl,
            endpoint_counts=[2, 1, 2, 1],
            frag_gc_sums=[0.8, 0.4, 0.9, 0.5],
        )
        rebinned = _rebin_table(table, 100)
        self.assertEqual(rebinned.num_rows, 2)
        _, _, starts, ends = reconstruct_sample_grid(
            SampleMetadata.from_table(rebinned)
        )
        self.assertEqual(starts.tolist(), [0, 100])
        self.assertEqual(ends.tolist(), [100, 200])
        np.testing.assert_allclose(rebinned.column("count").to_pylist(), [30, 70])
        self.assertEqual(rebinned.column("paired_count").to_pylist(), [3, 3])
        self.assertEqual(rebinned.column("frag_len_sum").to_pylist(), [390, 480])
        bands = fixed_size_list_to_numpy(
            rebinned.column("frag_bands"), NUM_FRAG_BANDS, np.uint32
        )
        self.assertEqual(bands.sum(axis=1).tolist(), [3, 3])
        self.assertEqual(rebinned.column("endpoint_count").to_pylist(), [3, 3])
        np.testing.assert_allclose(
            rebinned.column("frag_gc_sum").to_pylist(), [1.2, 1.4], rtol=1e-5
        )
        bufs = rebin_sample_buffers(SampleBuffers.from_table(table), 100)
        self.assertEqual(bufs.count.tolist(), [30, 70])

    def test_rebin_sample_table_non_nested_drops_fragmentomic_metadata_with_warning(
        self,
    ) -> None:
        table = make_sample_table(
            [10, 20, 30],
            contigs=["chr1"] * 3,
            starts=[0, 20, 40],
            ends=[20, 40, 60],
            contig_lengths={"chr1": 60},
            bin_size=20,
            fragment_lengths=[[150], [150], [150]],
        )
        vals = (True, 3, 166.0, 143.0, 0.0, 0.35, 0.05, 0.01, 0.04, 1.0, 0.90)
        frag_meta = FragmentomicsMetadata(
            *vals, [0] * 571, [0] * 571, [0] * 571, [1] * 256
        )
        meta_dict = dict(table.schema.metadata or {})
        meta_dict[FRAGMENTOMICS_METADATA_KEY] = json.dumps(asdict(frag_meta)).encode()
        table_with_frag = table.replace_schema_metadata(meta_dict)

        with self.assertLogs("ccnv.features.rebinning", level="WARNING") as cm:
            rebinned = _rebin_table(table_with_frag, 50)
        self.assertTrue(any("clears fragmentomic channels" in msg for msg in cm.output))
        self.assertNotIn(FRAGMENTOMICS_METADATA_KEY, rebinned.schema.metadata or {})

    def test_cohort_rejects_mixed_single_and_paired_end(self) -> None:
        pe_table = make_sample_table(
            [10, 20],
            contig_lengths={"chr1": 200},
            bin_size=100,
            fragment_lengths=[[150], [150]],
        )
        se_table = make_sample_table(
            [10, 20], contig_lengths={"chr1": 200}, bin_size=100
        )

        with self.assertRaises(ReferenceError) as tie_ctx:
            validate_cohort(
                [pe_table, se_table],
                paths=[Path("pe.ccnv"), Path("se.ccnv")],
            )
        self.assertIn(
            "mixed single-end and paired-end cohort controls are not supported: "
            "paired-end files (1/2): pe.ccnv; single-end files (1/2): se.ccnv",
            str(tie_ctx.exception),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            se_first = write_synthetic_sample(
                tmp_path / "se_first.ccnv",
                [10, 20],
                contig_lengths={"chr1": 200},
                bin_size=100,
            )
            pe_files = [
                write_synthetic_sample(
                    tmp_path / f"pe_{i}.ccnv",
                    [10, 20],
                    contig_lengths={"chr1": 200},
                    bin_size=100,
                    fragment_lengths=[[150], [150]],
                )
                for i in range(1, 4)
            ]
            se_last = write_synthetic_sample(
                tmp_path / "se_last.ccnv",
                [10, 20],
                contig_lengths={"chr1": 200},
                bin_size=100,
            )
            entries = [SamplesheetEntry(p) for p in (se_first, *pe_files, se_last)]
            with self.assertRaises(ReferenceError) as pe_ctx:
                load_and_assemble_cohort(entries, target_bin_size=100)
            pe_msg = str(pe_ctx.exception)
            self.assertIn("cohort is paired-end (3/5), single-end files (2/5):", pe_msg)
            self.assertIn(str(se_first), pe_msg)
            self.assertIn(str(se_last), pe_msg)
            self.assertNotIn("pe_1.ccnv", pe_msg)

        with self.assertRaises(ReferenceError) as se_ctx:
            validate_cohort(
                [se_table, pe_table, se_table],
                paths=[
                    Path("/tmp/se_1.ccnv"),
                    Path("/tmp/pe_bad.ccnv"),
                    Path("/tmp/se_2.ccnv"),
                ],
            )
        se_msg = str(se_ctx.exception)
        self.assertIn(
            "cohort is single-end (2/3), paired-end files (1/3): /tmp/pe_bad.ccnv",
            se_msg,
        )
        self.assertNotIn("se_1.ccnv", se_msg)

    def test_read_sample_buffers_and_extract_cram(self) -> None:
        table = make_sample_table([5, 10], contig_lengths={"chr1": 200}, bin_size=100)
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "sample.parquet"
            write_sample_parquet(table, p)
            loaded = read_sample_buffers(p)
            self.assertEqual(loaded.num_rows, table.num_rows)
            self.assertEqual(loaded.count.tolist(), [5, 10])

        cram = ensure_published_cram()
        fasta = ensure_published_fasta()
        cram_table = extract_sample_table(
            ExtractConfig(cram, Path("unused"), PUBLISHED_BIN_SIZE, fasta=fasta)
        )
        self.assertEqual(cram_table.num_rows, PUBLISHED_CANONICAL_BINS)
        self.assertIn("frag_bands", cram_table.column_names)
        self.assertIn("endpoint_count", cram_table.column_names)
        self.assertIn("frag_gc_sum", cram_table.column_names)

    def test_sample_sketch_v2_roundtrip_nested_coarsening_and_upsample_rejection(
        self,
    ) -> None:
        self.assertEqual(SCHEMA_VERSION, "2.0.0")
        self.assertEqual(SUPPORTED_SCHEMA_VERSIONS, frozenset({"2.0.0"}))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bam_path = tmp / "paired.bam"
            _write_bam(bam_path, [("chr1", 250_000), ("chr2", 105_000)], paired=True)

            table_5k = extract_sample_table(
                ExtractConfig(bam_path, tmp / "s", 5_000, threads=2)
            )
            self.assertNotIn("contig", table_5k.column_names)
            self.assertNotIn("start", table_5k.column_names)
            self.assertNotIn("end", table_5k.column_names)
            self.assertTrue(table_5k.schema.equals(SAMPLE_SCHEMA, check_metadata=False))
            self.assertEqual(
                table_5k.schema.field("frag_bands").type,
                pa.list_(pa.uint32(), NUM_FRAG_BANDS),
            )
            self.assertEqual(table_5k["frag_bands"].null_count, 0)

            out_pq = write_sample_parquet(table_5k, parquet_path(tmp / "s"))
            reloaded_bufs = read_sample_buffers(out_pq)
            self.assertEqual(reloaded_bufs.num_rows, 50 + 21)
            reloaded = pq.read_table(out_pq)
            meta_5k = validate_sample_table(reloaded)
            self.assertIsInstance(meta_5k, SampleMetadata)
            self.assertEqual(meta_5k.contig_order, ("chr1", "chr2"))
            codes, names, starts, ends = reconstruct_sample_grid(meta_5k)
            self.assertEqual(names, ("chr1", "chr2"))
            self.assertEqual(codes.dtype, np.int16)
            self.assertEqual((starts.dtype, ends.dtype), (np.uint32, np.uint32))
            self.assertEqual(codes.size, table_5k.num_rows)
            self.assertEqual(table_5k.num_rows, 50 + 21)

            table_100k = _rebin_table(table_5k, 100_000)
            meta_100k = validate_sample_table(table_100k)
            self.assertEqual(meta_100k.bin_size, 100_000)
            self.assertEqual(table_100k.num_rows, 3 + 2)
            self.assertIn(FRAGMENTOMICS_METADATA_KEY, table_100k.schema.metadata or {})
            for col_name in ("count", "paired_count", "frag_len_sum"):
                self.assertEqual(
                    int(pc.sum(table_100k[col_name]).as_py()),
                    int(pc.sum(table_5k[col_name]).as_py()),
                )
            for col_name in ("short_frag_count", "endpoint_count"):
                self.assertEqual(
                    int(pc.sum(table_100k[col_name]).as_py()),
                    int(pc.sum(table_5k[col_name]).as_py()),
                )
            np.testing.assert_allclose(
                float(pc.sum(table_100k["frag_gc_sum"]).as_py()),
                float(pc.sum(table_5k["frag_gc_sum"]).as_py()),
                rtol=1e-5,
            )
            bands_5k = fixed_size_list_to_numpy(
                table_5k["frag_bands"], NUM_FRAG_BANDS, np.uint32
            )
            bands_100k = fixed_size_list_to_numpy(
                table_100k["frag_bands"], NUM_FRAG_BANDS, np.uint32
            )
            np.testing.assert_array_equal(bands_100k.sum(axis=0), bands_5k.sum(axis=0))

            with self.assertLogs("ccnv.features.rebinning", level=logging.WARNING):
                table_12k = _rebin_table(table_5k, 12_000)
            validate_sample_table(table_12k)
            self.assertEqual(
                int(pc.sum(table_12k["count"]).as_py()),
                int(pc.sum(table_5k["count"]).as_py()),
            )
            self.assertEqual(table_12k["frag_bands"].null_count, table_12k.num_rows)
            self.assertEqual(int(pc.sum(table_12k["paired_count"]).as_py()), 0)
            self.assertNotIn(
                FRAGMENTOMICS_METADATA_KEY, table_12k.schema.metadata or {}
            )

            with self.assertRaises(BinCompatibilityError):
                _rebin_table(table_5k, 1_000)

            se_bam = tmp / "se.bam"
            _write_bam(se_bam, [("chr1", 25_000)], paired=False)
            se_table = extract_sample_table(
                ExtractConfig(alignment=se_bam, prefix=tmp / "se", bin_size=5_000)
            )
            self.assertEqual(se_table["frag_bands"].null_count, se_table.num_rows)
            se_zero_ep = se_table.set_column(
                se_table.schema.get_field_index("endpoint_count"),
                "endpoint_count",
                pa.array(np.zeros(se_table.num_rows, dtype=np.uint32)),
            )
            with self.assertNoLogs("ccnv.features.rebinning", level=logging.WARNING):
                se_12k = _rebin_table(se_zero_ep, 12_000)
            self.assertEqual(
                int(pc.sum(se_12k["count"]).as_py()),
                int(pc.sum(se_zero_ep["count"]).as_py()),
            )

    def test_rebin_counts_matrix_rejects_non_contiguous_contig_runs(self) -> None:
        s0, e0 = np.array([0, 0, 5000]), np.array([5000, 5000, 10000])
        c0, c1 = ["chr1", "chr2", "chr1"], ["chr1", "chr2"]
        with self.assertRaises(ValueError):
            rebin_counts_matrix(
                np.ones((3, 2)),
                c0,
                s0,
                e0,
                c1,
                np.array([0, 0]),
                np.array([10000, 5000]),
            )

    def test_ccnv_error_contracts_in_binning_rebinning_and_cohort(self) -> None:
        with self.assertRaises(InvalidInputError) as ctx1:
            bin_intervals(0, 5000)
        self.assertIsInstance(ctx1.exception, CcnvError)
        z0 = np.array([0])
        with self.assertRaises(InvalidInputError):
            rebin_counts(z0, np.array([5000]), np.array([10.0]), z0, np.array([10_000]))
        bufs = SampleBuffers.from_table(
            make_sample_table([10, 20], contig_lengths={"chr1": 10_000}, bin_size=5000)
        )
        with self.assertRaises(InvalidInputError):
            rebin_sample_buffers(bufs, 0)
        with self.assertRaises(InvalidInputError):
            rebin_cohort(validate_cohort((bufs.to_table(),)), 0)


if __name__ == "__main__":
    unittest.main()
