"""Unit tests for variance-whitened PCA and legacy PCA helpers."""

from __future__ import annotations

import unittest

import numpy as np

from ccnv.models.exceptions import ReferenceError
from ccnv.stats.pca import fit_whitened_pca, pca_irlba_numpy


class TestPCA(unittest.TestCase):
    def test_small_sample_counts_select_rank_zero_cleanly(self) -> None:
        usable = np.array([True, True, True, False, True], dtype=np.bool_)
        for n_samples in (1, 2):
            counts = np.full((5, n_samples), 12.0, dtype=np.float32)
            expected = np.full((5, n_samples), 10.0, dtype=np.float32)
            res = fit_whitened_pca(counts, expected, usable)
            self.assertEqual(res.loadings.shape, (0, 4))
            self.assertEqual(res.singular_values.shape, (0,))
            self.assertTrue(np.all(res.mean_rate[usable] > 0.0))
            self.assertEqual(float(res.mean_rate[3]), 0.0)

    def test_max_components_zero_forces_rank_zero(self) -> None:
        rng = np.random.default_rng(19)
        usable = np.ones(120, dtype=np.bool_)
        grid = np.linspace(0.0, 2.0 * np.pi, 120)
        expected = np.full((120, 12), 25.0, dtype=np.float32)
        counts = rng.poisson(
            25.0 * np.exp(np.outer(np.sin(grid), rng.normal(0.0, 0.3, size=12)))
        ).astype(np.float32)
        res = fit_whitened_pca(counts, expected, usable, max_components=0)
        self.assertEqual(res.loadings.shape, (0, 120))
        self.assertEqual(res.singular_values.shape, (0,))

    def test_deterministic_sign_canonicalization(self) -> None:
        rng = np.random.default_rng(23)
        usable = np.ones(300, dtype=np.bool_)
        grid = np.linspace(0.0, 2.0 * np.pi, 300)
        expected = np.full((300, 20), 30.0, dtype=np.float32)
        counts = rng.poisson(
            30.0 * np.exp(np.outer(np.cos(grid), rng.normal(0.0, 0.25, size=20)))
        ).astype(np.float32)
        r1 = fit_whitened_pca(counts, expected, usable)
        r2 = fit_whitened_pca(counts.copy(), expected.copy(), usable.copy())
        self.assertEqual(r1.loadings.shape, (1, 300))
        np.testing.assert_allclose(r1.loadings, r2.loadings, atol=1e-7)

    def test_invalid_inputs_raise_reference_error(self) -> None:
        with self.assertRaises(ReferenceError):
            fit_whitened_pca(
                np.ones((4, 3), dtype=np.float32),
                np.ones((5, 3), dtype=np.float32),
                np.ones(4, dtype=np.bool_),
            )
        with self.assertRaises(ReferenceError):
            fit_whitened_pca(
                np.ones((4, 3), dtype=np.float32),
                np.ones((4, 3), dtype=np.float32),
                np.zeros(4, dtype=np.bool_),
            )

    def test_pca_irlba_numpy_preserved_for_prenatal(self) -> None:
        rng = np.random.default_rng(5)
        x = rng.normal(size=(16, 40))
        scores, center, rotation = pca_irlba_numpy(x, n_components=3)
        self.assertEqual(scores.shape, (16, 3))
        self.assertEqual(center.shape, (40,))
        self.assertEqual(rotation.shape, (40, 3))


if __name__ == "__main__":
    unittest.main()
