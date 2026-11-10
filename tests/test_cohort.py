"""Unit tests for cohort validation and parallel cohort loading."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from helpers import make_sample_table

from ccnv.features.cohort import rebin_cohort, validate_cohort
from ccnv.io.parquet import read_sample_tables
from ccnv.models.exceptions import ReferenceError


class TestCohortCompatibility(unittest.TestCase):
    def test_rejects_bin_size_mismatch(self) -> None:
        left = make_sample_table([10, 10], bin_size=100)
        right = make_sample_table([5, 5, 5, 5], bin_size=50)
        with self.assertRaisesRegex(ReferenceError, "bin_size"):
            validate_cohort((left, right))

    def test_rejects_coordinate_mismatch(self) -> None:
        left = make_sample_table(
            [10, 10], ends=[100, 200], contig_lengths={"chr1": 200}
        )
        right = make_sample_table(
            [10, 10],
            starts=[0, 80],
            ends=[80, 200],
            contig_lengths={"chr1": 200},
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

            cohort_seq = validate_cohort(read_sample_tables(tuple(paths), threads=1))
            cohort_par = validate_cohort(read_sample_tables(tuple(paths), threads=2))

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
        from ccnv.features.cohort import estimate_size_factors, normalize_depth

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

        normed = normalize_depth(base)
        # Unamplified diploid bins (3..9) in sample 0 must equal diploid bins in samples 1..3
        np.testing.assert_allclose(normed[3:, 0], normed[3:, 1], rtol=1e-12)

    def test_normalize_depth_sparse_zero_fallback(self) -> None:
        from ccnv.features.cohort import normalize_depth

        sparse = np.array([[10.0, 0.0], [0.0, 20.0]], dtype=np.float64)
        normed = normalize_depth(sparse)
        np.testing.assert_allclose(normed.sum(axis=0), [1.0, 1.0], rtol=1e-12)

    def test_rebin_cohort_polars_fast_path_coarsening(self) -> None:
        from ccnv.features.cohort import Cohort, rebin_cohort

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
        from ccnv.features.cohort import InvalidInputError, estimate_size_factors

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
        from ccnv.features.cohort import rebin_sample_table

        table = make_sample_table(
            [10.0, 20.0, 30.0, 40.0],
            contigs=["chr1", "chr1", "chr1", "chr1"],
            starts=[0, 50, 100, 150],
            ends=[50, 100, 150, 180],
            contig_lengths={"chr1": 180, "chr2": 100},
            bin_size=50,
        )
        rebinned = rebin_sample_table(table, 100)
        self.assertEqual(rebinned.column("start").to_pylist(), [0, 100, 0])
        self.assertEqual(rebinned.column("end").to_pylist(), [100, 180, 100])
        self.assertEqual(rebinned.column("count").to_pylist(), [30.0, 70.0, 0.0])
        self.assertEqual(
            rebinned.column("contig").to_pylist(), ["chr1", "chr1", "chr2"]
        )

    def test_rebin_sample_table_boundary_crossing_overlap_redistribution(self) -> None:
        from ccnv.features.cohort import rebin_sample_table

        # Interval [45, 75) crosses target boundary at 50
        table = make_sample_table(
            [20.0, 30.0, 40.0],
            contigs=["chr1", "chr1", "chr1"],
            starts=[10, 45, 90],
            ends=[30, 75, 110],
            contig_lengths={"chr1": 150},
            bin_size=20,
        )
        rebinned = rebin_sample_table(table, 50)
        self.assertEqual(rebinned.column("start").to_pylist(), [0, 50, 100])
        self.assertEqual(rebinned.column("end").to_pylist(), [50, 100, 150])
        # Bin 0..50 gets [10, 30) (len 20, count 20) + 5/30 of [45, 75) (count 5) = 25.0
        # Bin 50..100 gets 25/30 of [45, 75) (count 25) + 10/20 of [90, 110) (count 20) = 45.0
        # Bin 100..150 gets 10/20 of [90, 110) (count 20) = 20.0
        np.testing.assert_allclose(
            rebinned.column("count").to_numpy(), [25.0, 45.0, 20.0]
        )

    def test_rebin_cohort_boundary_crossing_overlap_redistribution(self) -> None:
        from ccnv.features.cohort import Cohort, rebin_cohort

        cohort = Cohort(
            counts=np.array([[20.0, 40.0], [30.0, 60.0], [40.0, 80.0]]),
            contigs=("chr1", "chr1", "chr1"),
            starts=np.array([10, 45, 90], dtype=np.int64),
            ends=np.array([30, 75, 110], dtype=np.int64),
            contig_lengths={"chr1": 150},
            bin_size=20,
            identities=({"id": "a"}, {"id": "b"}),
            paths=(Path("a.ccnv"), Path("b.ccnv")),
        )
        rebinned = rebin_cohort(cohort, 50)
        np.testing.assert_array_equal(rebinned.starts, [0, 50, 100])
        np.testing.assert_array_equal(rebinned.ends, [50, 100, 150])
        np.testing.assert_allclose(
            rebinned.counts,
            [[25.0, 50.0], [45.0, 90.0], [20.0, 40.0]],
        )

    def test_rebin_sample_table_preserves_fragmentomic_columns(self) -> None:
        from ccnv.features.cohort import rebin_sample_table

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

        rebinned = rebin_sample_table(table, 100)
        self.assertEqual(rebinned.num_rows, 2)
        self.assertEqual(rebinned.column("start").to_pylist(), [0, 100])
        self.assertEqual(rebinned.column("end").to_pylist(), [100, 200])

        # Counts summed
        np.testing.assert_allclose(rebinned.column("count").to_pylist(), [30.0, 70.0])
        # Fragment lengths concatenated
        self.assertEqual(
            rebinned.column("fragment_lengths").to_pylist(),
            [[120, 130, 140], [150, 160, 170]],
        )
        # Endpoint counts summed
        self.assertEqual(rebinned.column("endpoint_count").to_pylist(), [3, 3])
        # Frag GC sums summed
        np.testing.assert_allclose(
            rebinned.column("frag_gc_sum").to_pylist(), [1.2, 1.4], rtol=1e-5
        )

    def test_rebin_sample_table_drops_null_elements_in_fragment_lengths(self) -> None:
        from ccnv.features.cohort import rebin_sample_table

        # Coarsen 50 bp -> 100 bp with null element inside list
        table = make_sample_table(
            [10, 20],
            contigs=["chr1", "chr1"],
            starts=[0, 50],
            ends=[50, 100],
            contig_lengths={"chr1": 100},
            bin_size=50,
            fragment_lengths=[[120, 130], [140]],
        )
        rebinned = rebin_sample_table(table, 100)
        self.assertEqual(
            rebinned.column("fragment_lengths").to_pylist(),
            [[120, 130, 140]],
        )

    def test_rebin_sample_table_non_nested_drops_fragmentomic_metadata_with_warning(
        self,
    ) -> None:
        import json

        from ccnv.features.cohort import rebin_sample_table
        from ccnv.models.schemas import FRAGMENTOMICS_METADATA_KEY

        table = make_sample_table(
            [10, 20, 30],
            contigs=["chr1", "chr1", "chr1"],
            starts=[0, 20, 40],
            ends=[20, 40, 60],
            contig_lengths={"chr1": 60},
            bin_size=20,
        )
        meta_dict = dict(table.schema.metadata)
        meta_dict[FRAGMENTOMICS_METADATA_KEY] = json.dumps(
            {"paired_end": True}
        ).encode()
        table_with_frag = table.replace_schema_metadata(meta_dict)

        with self.assertLogs("ccnv.features.cohort", level="WARNING") as cm:
            rebinned = rebin_sample_table(table_with_frag, 50)

        self.assertTrue(
            any("dropping fragmentomic metadata" in msg for msg in cm.output)
        )
        self.assertNotIn(FRAGMENTOMICS_METADATA_KEY, rebinned.schema.metadata or {})

    def test_cohort_rejects_mixed_single_and_paired_end(self) -> None:
        import json

        from ccnv.features.cohort import validate_cohort
        from ccnv.models.schemas import FRAGMENTOMICS_METADATA_KEY

        # Sample 1: paired-end
        s1 = make_sample_table([10, 20], contig_lengths={"chr1": 200}, bin_size=100)
        s1_meta = dict(s1.schema.metadata)
        s1_meta[FRAGMENTOMICS_METADATA_KEY] = json.dumps({"paired_end": True}).encode(
            "utf-8"
        )
        s1 = s1.replace_schema_metadata(s1_meta)

        # Sample 2: single-end
        s2 = make_sample_table([10, 20], contig_lengths={"chr1": 200}, bin_size=100)
        s2_meta = dict(s2.schema.metadata)
        s2_meta[FRAGMENTOMICS_METADATA_KEY] = json.dumps({"paired_end": False}).encode(
            "utf-8"
        )
        s2 = s2.replace_schema_metadata(s2_meta)

        with self.assertRaisesRegex(
            ReferenceError,
            "mixed single-end and paired-end cohort controls are not supported",
        ):
            validate_cohort([s1, s2])

    def test_load_sample_or_extract_parquet_and_bam(self) -> None:
        import tempfile

        from published_fixtures import (
            PUBLISHED_BIN_SIZE,
            PUBLISHED_CRAM,
            PUBLISHED_CRAM_URL,
            ensure_published_fasta,
            ensure_published_file,
        )

        from ccnv.io.parquet import load_sample_or_extract, write_sample_parquet

        table = make_sample_table([5, 10], contig_lengths={"chr1": 200}, bin_size=100)
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "sample.parquet"
            write_sample_parquet(table, p)
            loaded = load_sample_or_extract(p)
            self.assertEqual(loaded.num_rows, table.num_rows)
            self.assertEqual(loaded.column("count").to_pylist(), [5, 10])

        cram = ensure_published_file(PUBLISHED_CRAM, PUBLISHED_CRAM_URL)
        fasta = ensure_published_fasta()
        cram_table = load_sample_or_extract(
            cram, fasta=fasta, bin_size=PUBLISHED_BIN_SIZE
        )
        self.assertEqual(cram_table.num_rows, 468)
        self.assertIn("fragment_lengths", cram_table.column_names)
        self.assertIn("endpoint_count", cram_table.column_names)
        self.assertIn("frag_gc_sum", cram_table.column_names)


if __name__ == "__main__":
    unittest.main()
