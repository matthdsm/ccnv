"""Unit tests for cohort validation and parallel cohort loading."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from helpers import make_sample_table

from ccnv.features.cohort import (
    Cohort,
    InvalidInputError,
    estimate_size_factors,
    load_and_assemble_cohort,
    rebin_cohort,
    validate_cohort,
)
from ccnv.features.rebinning import rebin_sample_buffers, rebin_sample_sketch
from ccnv.models import SamplesheetEntry
from ccnv.models.exceptions import ReferenceError


class TestCohortCompatibility(unittest.TestCase):
    def test_rejects_bin_size_mismatch(self) -> None:
        left = make_sample_table([10, 10], bin_size=100)
        right = make_sample_table([5, 5, 5, 5], bin_size=50)
        with self.assertRaisesRegex(ReferenceError, "bin_size"):
            validate_cohort((left, right))

    def test_rejects_coordinate_mismatch(self) -> None:
        left = make_sample_table(
            [10, 10],
            contigs=["chr1", "chr2"],
            starts=[0, 0],
            ends=[100, 100],
            contig_lengths={"chr1": 100, "chr2": 100},
            bin_size=100,
        )
        right = make_sample_table(
            [10, 10],
            contigs=["chr2", "chr1"],
            starts=[0, 0],
            ends=[100, 100],
            contig_lengths={"chr1": 100, "chr2": 100},
            bin_size=100,
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
                identity={"id": f"s{index}"},
            )
            for index in range(3)
        ]
        cohort = validate_cohort(tuple(tables))
        self.assertEqual(cohort.counts.shape, (3, 3))
        self.assertEqual(cohort.contigs, ("scaffold-A", "scaffold-A", "lgX"))


class TestCohortLoading(unittest.TestCase):
    def test_load_cohort_parallel_equivalence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
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


class TestCohortDepthAndRebinning(unittest.TestCase):
    def test_normalize_depth_median_of_ratios_matches_deseq2_and_resists_trisomy(
        self,
    ) -> None:
        # 10 bins x 4 samples; sample 0 has 2x library depth and a 10x amplification in bins 0..2
        base = np.full((10, 4), 100.0, dtype=np.float64)
        base[:, 0] = 200.0
        base[0:3, 0] = 2000.0  # 30% of genome amplified 10x in sample 0

        # Reference DESeq2 formula: geometric mean per row, then median of ratios per sample
        log_gm = np.mean(np.log(base), axis=1, keepdims=True)
        expected_sf = np.median(base / np.exp(log_gm), axis=0)
        sf = estimate_size_factors(base)
        np.testing.assert_allclose(
            sf / np.median(sf),
            expected_sf / np.median(expected_sf),
            rtol=1e-12,
        )

        normed = base / sf
        # Unamplified diploid bins (3..9) in sample 0 must equal diploid bins in samples 1..3
        np.testing.assert_allclose(normed[3:, 0], normed[3:, 1], rtol=1e-12)

    def test_normalize_depth_sparse_zero_fallback(self) -> None:
        sparse = np.array([[10.0, 0.0], [0.0, 20.0]], dtype=np.float64)
        sf = estimate_size_factors(sparse)
        normed = sparse / sf
        np.testing.assert_allclose(normed.sum(axis=0), [15.0, 15.0], rtol=1e-12)

    def test_rebin_cohort_polars_fast_path_coarsening(self) -> None:
        cohort = Cohort(
            counts=np.arange(1, 9, dtype=np.float64).reshape(4, 2),
            contigs=("chr1", "chr1", "chr1", "chr1"),
            starts=np.array([0, 50, 100, 150], dtype=np.int64),
            ends=np.array([50, 100, 150, 180], dtype=np.int64),
            contig_lengths={"chr1": 180},
            bin_size=50,
            identities=({"id": "a"}, {"id": "b"}),
            paths=(Path("a.ccnv"), Path("b.ccnv")),
        )
        rebinned = rebin_cohort(cohort, 100)
        np.testing.assert_array_equal(rebinned.starts, [0, 100])
        np.testing.assert_array_equal(rebinned.ends, [100, 180])
        np.testing.assert_allclose(rebinned.counts, [[4.0, 6.0], [12.0, 14.0]])

    def test_estimate_size_factors_validation(self) -> None:
        with self.assertRaises(InvalidInputError):
            estimate_size_factors(np.array([1.0, 2.0]))

        with self.assertRaises(InvalidInputError):
            estimate_size_factors(np.array([[-1.0, 2.0], [3.0, 4.0]]))

        with self.assertRaises(InvalidInputError):
            estimate_size_factors(np.array([[0.0, 2.0], [0.0, 4.0]]))

        with self.assertRaises(InvalidInputError):
            estimate_size_factors(
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                mask=np.array([True]),
            )

        with self.assertRaises(InvalidInputError):
            estimate_size_factors(
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                mask=np.array([False, False]),
            )

    def test_rebin_sample_table_polars_fast_path(self) -> None:
        from ccnv.models import SampleMetadata, reconstruct_sample_grid

        table = make_sample_table(
            [10, 20, 30, 40, 0, 0],
            contigs=["chr1", "chr1", "chr1", "chr1", "chr2", "chr2"],
            starts=[0, 50, 100, 150, 0, 50],
            ends=[50, 100, 150, 180, 50, 100],
            contig_lengths={"chr1": 180, "chr2": 100},
            bin_size=50,
        )
        rebinned = rebin_sample_sketch(table, 100)
        codes, names, starts, ends = reconstruct_sample_grid(
            SampleMetadata.from_table(rebinned)
        )
        self.assertEqual(starts.tolist(), [0, 100, 0])
        self.assertEqual(ends.tolist(), [100, 180, 100])
        self.assertEqual(rebinned.column("count").to_pylist(), [30, 70, 0])
        self.assertEqual([names[int(c)] for c in codes], ["chr1", "chr1", "chr2"])

    def test_rebin_sample_table_boundary_crossing_overlap_redistribution(self) -> None:
        from ccnv.models import SampleMetadata, reconstruct_sample_grid

        # Non-nested coarsening 20 bp -> 50 bp across 140 bp contig (7 bins -> 3 bins)
        table = make_sample_table(
            [20, 10, 30, 0, 40, 0, 0],
            contigs=["chr1"] * 7,
            starts=[0, 20, 40, 60, 80, 100, 120],
            ends=[20, 40, 60, 80, 100, 120, 140],
            contig_lengths={"chr1": 140},
            bin_size=20,
        )
        rebinned = rebin_sample_sketch(table, 50)
        _, _, starts, ends = reconstruct_sample_grid(
            SampleMetadata.from_table(rebinned)
        )
        self.assertEqual(starts.tolist(), [0, 50, 100])
        self.assertEqual(ends.tolist(), [50, 100, 140])
        np.testing.assert_allclose(
            rebinned.column("count").to_numpy(), [45.0, 55.0, 0.0]
        )

    def test_rebin_cohort_boundary_crossing_overlap_redistribution(self) -> None:
        cohort = Cohort(
            counts=np.array(
                [
                    [30.0, 60.0],
                    [30.0, 60.0],
                    [60.0, 120.0],
                    [30.0, 60.0],
                    [0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            contigs=("chr1", "chr1", "chr1", "chr1", "chr1"),
            starts=np.array([0, 30, 60, 90, 120], dtype=np.int64),
            ends=np.array([30, 60, 90, 120, 150], dtype=np.int64),
            contig_lengths={"chr1": 150},
            bin_size=30,
            identities=({"id": "a"}, {"id": "b"}),
            paths=(Path("a.ccnv"), Path("b.ccnv")),
        )
        rebinned = rebin_cohort(cohort, 50)
        np.testing.assert_array_equal(rebinned.starts, [0, 50, 100])
        np.testing.assert_array_equal(rebinned.ends, [50, 100, 150])
        np.testing.assert_allclose(
            rebinned.counts,
            [[50.0, 100.0], [80.0, 160.0], [20.0, 40.0]],
        )

    def test_rebin_sample_table_preserves_fragmentomic_columns(self) -> None:
        from ccnv.models import (
            NUM_FRAG_BANDS,
            SampleBuffers,
            SampleMetadata,
            fixed_size_list_to_numpy,
            reconstruct_sample_grid,
        )

        # Coarsen 50 bp -> 100 bp
        table = make_sample_table(
            [10, 20, 30, 40],
            contigs=["chr1", "chr1", "chr1", "chr1"],
            starts=[0, 50, 100, 150],
            ends=[50, 100, 150, 200],
            contig_lengths={"chr1": 200},
            bin_size=50,
            fragment_lengths=[[120, 130], [140], [150, 160], [170]],
            endpoint_counts=[2, 1, 2, 1],
            frag_gc_sums=[0.8, 0.4, 0.9, 0.5],
        )

        rebinned = rebin_sample_sketch(table, 100)
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
        import json
        from dataclasses import asdict

        from ccnv.models.schemas import (
            FRAGMENTOMICS_METADATA_KEY,
            FragmentomicsMetadata,
        )

        table = make_sample_table(
            [10, 20, 30],
            contigs=["chr1", "chr1", "chr1"],
            starts=[0, 20, 40],
            ends=[20, 40, 60],
            contig_lengths={"chr1": 60},
            bin_size=20,
            fragment_lengths=[[150], [150], [150]],
        )
        frag_meta = FragmentomicsMetadata(
            paired_end=True,
            total_paired_fragments=3,
            chromatosome_mode_bp=166.0,
            core_shoulder_bp=143.0,
            sizing_shift_bp=0.0,
            self_referenced_peak_ratio=0.35,
            dinucleosome_fraction=0.05,
            high_mw_fraction=0.01,
            helical_10bp_amplitude=0.04,
            helical_10bp_phase=1.0,
            motif_diversity_score=0.90,
            hist_autosomal=[0] * 571,
            hist_chry=[0] * 571,
            hist_chrx=[0] * 571,
            end_motif_counts=[1] * 256,
        )
        meta_dict = dict(table.schema.metadata or {})
        meta_dict[FRAGMENTOMICS_METADATA_KEY] = json.dumps(asdict(frag_meta)).encode()
        table_with_frag = table.replace_schema_metadata(meta_dict)

        with self.assertLogs("ccnv.features.rebinning", level="WARNING") as cm:
            rebinned = rebin_sample_sketch(table_with_frag, 50)

        self.assertTrue(any("clears fragmentomic channels" in msg for msg in cm.output))
        self.assertNotIn(FRAGMENTOMICS_METADATA_KEY, rebinned.schema.metadata or {})

    def test_cohort_rejects_mixed_single_and_paired_end(self) -> None:
        from ccnv.features.cohort import validate_cohort

        # Sample 1: paired-end
        s1 = make_sample_table(
            [10, 20],
            contig_lengths={"chr1": 200},
            bin_size=100,
            fragment_lengths=[[150], [150]],
        )
        # Sample 2: single-end
        s2 = make_sample_table(
            [10, 20],
            contig_lengths={"chr1": 200},
            bin_size=100,
            fragment_lengths=None,
        )

        with self.assertRaisesRegex(
            ReferenceError,
            "mixed single-end and paired-end cohort controls are not supported",
        ):
            validate_cohort([s1, s2])

    def test_read_sample_parquet_and_extract_cram(self) -> None:
        from published_fixtures import (
            PUBLISHED_BIN_SIZE,
            PUBLISHED_CRAM,
            PUBLISHED_CRAM_URL,
            ensure_published_fasta,
            ensure_published_file,
        )

        from ccnv.extract import extract_sample_table
        from ccnv.io.parquet import read_sample_parquet, write_sample_parquet
        from ccnv.models import ExtractConfig

        table = make_sample_table([5, 10], contig_lengths={"chr1": 200}, bin_size=100)
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "sample.parquet"
            write_sample_parquet(table, p)
            loaded = read_sample_parquet(p)
            self.assertEqual(loaded.num_rows, table.num_rows)
            self.assertEqual(loaded.column("count").to_pylist(), [5, 10])

        cram = ensure_published_file(PUBLISHED_CRAM, PUBLISHED_CRAM_URL)
        fasta = ensure_published_fasta()
        cram_table = extract_sample_table(
            ExtractConfig(cram, Path("unused"), PUBLISHED_BIN_SIZE, fasta=fasta)
        )
        self.assertEqual(cram_table.num_rows, 468)
        self.assertIn("frag_bands", cram_table.column_names)
        self.assertIn("endpoint_count", cram_table.column_names)
        self.assertIn("frag_gc_sum", cram_table.column_names)


if __name__ == "__main__":
    unittest.main()
