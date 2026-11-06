"""Unit tests for nearest-bin selection and parallel reference kernels."""

import unittest

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ccnv.models import ReferenceConfig
from ccnv.reference_compute import compute_null_models, select_nearest_bins


class TestNearestBinSelection(unittest.TestCase):
    def test_selects_closest_off_contig_bins(self) -> None:
        profiles = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.95, 0.05, 0.0],
                [0.05, 0.95, 0.0],
            ]
        )
        contig_ids = np.array([0, 0, 1, 1])
        indices, distances = select_nearest_bins(
            profiles, contig_ids, k=1, max_batch_memory_mb=1
        )
        self.assertEqual(indices[0], [2])
        self.assertEqual(indices[1], [3])
        self.assertEqual(indices[2], [0])
        self.assertEqual(indices[3], [1])
        self.assertTrue(all(len(row) == 1 for row in distances))

    def test_argpartition_matches_nearest_neighbors_on_small_matrix(self) -> None:
        rng = np.random.default_rng(1)
        profiles = rng.normal(size=(8, 5))
        contig_ids = np.array([0, 0, 0, 1, 1, 1, 2, 2])
        indices, distances = select_nearest_bins(
            profiles, contig_ids, k=2, max_batch_memory_mb=1
        )
        model = NearestNeighbors(n_neighbors=8, algorithm="brute", metric="euclidean")
        model.fit(profiles)
        brute_dist, brute_idx = model.kneighbors(profiles, return_distance=True)
        for query, neighbor_idx, neighbor_dist in zip(
            range(len(profiles)), brute_idx, brute_dist, strict=True
        ):
            expected = [
                (int(index), float(dist) ** 2)
                for index, dist in zip(neighbor_idx, neighbor_dist, strict=True)
                if contig_ids[index] != contig_ids[query]
            ][:2]
            self.assertEqual(indices[query], [index for index, _ in expected])
            np.testing.assert_allclose(
                distances[query],
                [dist for _, dist in expected],
                rtol=1e-6,
                atol=1e-8,
            )


class TestParallelReferenceCompute(unittest.TestCase):
    def test_reference_config_threads_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "threads must be a positive integer"):
            ReferenceConfig(threads=0)
        with self.assertRaisesRegex(ValueError, "threads must be a positive integer"):
            ReferenceConfig(threads=-1)

    def test_select_nearest_bins_parallel_equivalence(self) -> None:
        rng = np.random.default_rng(42)
        n_bins = 2500
        profiles = rng.normal(size=(n_bins, 5)).astype(np.float64)
        contig_ids = np.repeat(np.arange(5), n_bins // 5)

        idx_seq, dst_seq = select_nearest_bins(
            profiles, contig_ids, k=10, max_batch_memory_mb=16, threads=1
        )
        idx_par, dst_par = select_nearest_bins(
            profiles, contig_ids, k=10, max_batch_memory_mb=16, threads=2
        )

        self.assertEqual(idx_seq, idx_par)
        np.testing.assert_allclose(dst_seq, dst_par, rtol=1e-12, atol=1e-12)

    def test_compute_null_models_parallel_equivalence(self) -> None:
        rng = np.random.default_rng(123)
        n_bins = 6000
        corrected = rng.uniform(0.5, 2.0, size=(n_bins, 10)).astype(np.float64)
        indices = [
            rng.choice(n_bins, size=20, replace=False).tolist() for _ in range(n_bins)
        ]

        mean_seq, std_seq = compute_null_models(corrected, indices, threads=1)
        mean_par, std_par = compute_null_models(corrected, indices, threads=2)

        np.testing.assert_allclose(mean_seq, mean_par, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(std_seq, std_par, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
