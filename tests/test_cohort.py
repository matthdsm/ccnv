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


if __name__ == "__main__":
    unittest.main()
