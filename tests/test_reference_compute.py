"""Unit tests for exogenous covariate basis, signal-space features, and reference config validation."""

import unittest

import numpy as np

from ccnv.features.covariates import covariate_features
from ccnv.models import ReferenceConfig
from ccnv.stats.neighbors import build_whitened_subspace_features
from ccnv.stats.regression import build_exogenous_covariate_basis


class TestReferenceComputeKernels(unittest.TestCase):
    def test_reference_config_threads_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "threads must be a positive integer"):
            ReferenceConfig(threads=0)
        with self.assertRaisesRegex(ValueError, "threads must be a positive integer"):
            ReferenceConfig(threads=-1)

    def test_exogenous_covariate_basis_with_fragmentomics(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        usable = np.ones(n, dtype=bool)
        gc = rng.uniform(0.3, 0.6, size=n)
        mappability = rng.uniform(0.7, 1.0, size=n)
        ref_frag_len = rng.uniform(140.0, 180.0, size=n).astype(np.float32)
        ref_ep_rate = rng.uniform(0.5, 3.0, size=n).astype(np.float32)

        basis_base = build_exogenous_covariate_basis(
            usable,
            bin_size=5_000,
            gc=gc,
            mappability=mappability,
        )
        n_cols_base = basis_base.shape[1]
        self.assertGreater(n_cols_base, 0)

        basis_aug = build_exogenous_covariate_basis(
            usable,
            bin_size=5_000,
            gc=gc,
            mappability=mappability,
            ref_frag_len=ref_frag_len,
            ref_ep_rate=ref_ep_rate,
        )
        self.assertGreater(basis_aug.shape[1], n_cols_base)
        self.assertEqual(basis_aug.shape[0], n)

        basis_const = build_exogenous_covariate_basis(
            usable,
            bin_size=5_000,
            gc=gc,
            mappability=mappability,
            ref_frag_len=np.full(n, 166.0, dtype=np.float32),
            ref_ep_rate=np.zeros(n, dtype=np.float32),
        )
        self.assertEqual(basis_const.shape[1], n_cols_base)

    def test_covariate_features_and_signal_space_features(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        usable = np.ones(n, dtype=bool)
        gc = rng.uniform(0.3, 0.6, size=n)
        mappability = rng.uniform(0.7, 1.0, size=n)
        mean_frag_len = rng.uniform(140.0, 180.0, size=n)
        endpoint_ratio = rng.uniform(0.5, 3.0, size=n)

        cov_mat_4 = covariate_features(
            usable,
            gc=gc,
            mappability=mappability,
            ref_frag_len=mean_frag_len,
            ref_ep_rate=endpoint_ratio,
        )
        self.assertIsNotNone(cov_mat_4)
        assert cov_mat_4 is not None
        self.assertEqual(cov_mat_4.shape, (n, 4))

        p_rank = 5
        pca_loadings = rng.standard_normal((p_rank, n))
        s_vals = np.linspace(10.0, 1.0, p_rank)

        feats_p4 = build_whitened_subspace_features(
            pca_loadings,
            s_vals,
            0.5,
            covariate_mat=cov_mat_4,
        )
        self.assertEqual(feats_p4.shape, (n, p_rank + 4))


if __name__ == "__main__":
    unittest.main()
