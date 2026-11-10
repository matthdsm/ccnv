"""Unit tests for nearest-bin selection and parallel reference kernels."""

import unittest

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ccnv.models import ReferenceConfig
from ccnv.stats.neighbors import compute_null_models, select_nearest_bins


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

    def test_augmented_joint_covariate_basis_with_fragmentomics(self) -> None:
        from ccnv.stats.regression import build_joint_covariate_basis

        rng = np.random.default_rng(42)
        n = 500
        gc = rng.uniform(0.3, 0.6, size=n)
        mappability = rng.uniform(0.7, 1.0, size=n)
        mean_frag_len = rng.uniform(140.0, 180.0, size=n)
        endpoint_ratio = rng.uniform(0.5, 3.0, size=n)

        # Baseline GC + Mappability basis
        basis_base = build_joint_covariate_basis(gc, mappability)
        self.assertIsNotNone(basis_base)
        n_cols_base = basis_base.shape[1]

        # Augmented with mean_frag_len and endpoint_ratio
        basis_aug = build_joint_covariate_basis(
            gc,
            mappability,
            mean_frag_len=mean_frag_len,
            endpoint_ratio=endpoint_ratio,
        )
        self.assertIsNotNone(basis_aug)
        self.assertGreater(basis_aug.shape[1], n_cols_base)
        self.assertEqual(basis_aug.shape[0], n)

        # Constant fragmentomics fall back gracefully to base number of columns
        basis_const = build_joint_covariate_basis(
            gc,
            mappability,
            mean_frag_len=np.full(n, 166.0),
            endpoint_ratio=np.zeros(n),
        )
        self.assertIsNotNone(basis_const)
        self.assertEqual(basis_const.shape[1], n_cols_base)

    def test_covariate_features_and_r_p_plus_4_kdtree_features(self) -> None:
        from ccnv.features.covariates import covariate_features
        from ccnv.stats.neighbors import build_signal_space_features

        rng = np.random.default_rng(42)
        n = 500
        usable = np.ones(n, dtype=bool)
        gc = rng.uniform(0.3, 0.6, size=n)
        mappability = rng.uniform(0.7, 1.0, size=n)
        mean_frag_len = rng.uniform(140.0, 180.0, size=n)
        endpoint_ratio = rng.uniform(0.5, 3.0, size=n)

        # 4 covariates -> 4 columns
        cov_mat_4 = covariate_features(
            usable,
            gc=gc,
            mappability=mappability,
            mean_frag_len=mean_frag_len,
            endpoint_ratio=endpoint_ratio,
        )
        self.assertIsNotNone(cov_mat_4)
        self.assertEqual(cov_mat_4.shape, (n, 4))

        # PCA loadings with P=5 components
        P = 5
        pca_loadings = rng.standard_normal((P, n))
        s_vals = np.linspace(10.0, 1.0, P)

        # Build R^{P+4} features for KDTree
        feats_p4 = build_signal_space_features(
            pca_loadings,
            singular_values=s_vals,
            covariate_mat=cov_mat_4,
        )
        self.assertEqual(feats_p4.shape, (n, P + 4))


if __name__ == "__main__":
    unittest.main()
