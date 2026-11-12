"""Regression tests for Prototype Graph fitting, LOCO baselines, and null dispersion."""

from __future__ import annotations

import numpy as np
import pytest

from ccnv.stats.neighbors import (
    calibrate_prototype_dispersion,
    evaluate_loco_prototype_baseline,
    fit_prototype_graph,
)
from ccnv.stats.regression import hsm_biweight_mode


def test_single_contig_cluster_on_aneuploid_contig_uses_secondary_off_contig_prototypes() -> (
    None
):
    pids = np.array(
        [
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=np.uint16,
    )
    weights = np.array(
        [
            [0.8, 0.2, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
        ],
        dtype=np.float16,
    )
    contigs = np.array([0, 0, 1, 1, 2, 2, 2], dtype=np.int32)
    vals = np.array([1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5], dtype=np.float64)

    baselines, support = evaluate_loco_prototype_baseline(vals, contigs, pids, weights)
    np.testing.assert_allclose(baselines[4:], 1.0)
    assert np.all(support[4:] == 4)


def test_singleton_float16_zero_secondary_weights_preserve_cnv_and_null_variance() -> (
    None
):
    rng = np.random.default_rng(42)
    feats = np.vstack(
        [
            rng.normal(0.0, 0.05, size=(16, 3)),
            np.array([[5000.0, -5000.0, 5000.0]]),
        ]
    ).astype(np.float32)
    centers, pids, pweights, _ = fit_prototype_graph(feats, max_prototypes=8, top_r=4)

    assert pweights[16, 0] == np.float16(1.0)
    np.testing.assert_array_equal(pweights[16, 1:], np.float16(0.0))

    contigs = np.array([0] * 8 + [1] * 8 + [2], dtype=np.int32)
    vals = np.array([1.0] * 16 + [2.0], dtype=np.float64)
    baselines, support = evaluate_loco_prototype_baseline(vals, contigs, pids, pweights)
    assert baselines[16] == pytest.approx(1.0)
    assert support[16] > 0

    exp = np.full((17, 12), 100.0, dtype=np.float32)
    eta = rng.normal(0.0, 0.12, size=(17, 12))
    counts = rng.poisson(exp * np.exp(eta)).astype(np.float32)
    null_mean, bio_disp = calibrate_prototype_dispersion(
        counts,
        exp,
        contigs.astype(np.int16),
        np.ones(17, dtype=bool),
        pids,
        pweights,
        pca_rank=0,
        n_prototypes=int(centers.shape[0]),
    )
    assert np.isfinite(null_mean[16])
    assert bio_disp[16] >= 0.0


def test_all_four_candidates_same_contig_falls_back_to_global_off_contig_median() -> (
    None
):
    pids = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 2, 3, 4],
            [2, 1, 3, 4],
            [3, 1, 2, 4],
            [4, 1, 2, 3],
        ],
        dtype=np.uint16,
    )
    weights = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float16), (6, 1))
    contigs = np.array([0, 1, 2, 2, 2, 2], dtype=np.int32)
    vals = np.array([1.0, 1.0, 1.5, 1.5, 1.5, 1.5], dtype=np.float64)

    baselines, support = evaluate_loco_prototype_baseline(vals, contigs, pids, weights)
    np.testing.assert_allclose(baselines[2:], 1.0)
    assert np.all(support[2:] == 2)


def test_single_contig_branch_excludes_self_singleton() -> None:
    pids = np.array([[0, 1, 0, 0], [1, 0, 0, 0]], dtype=np.uint16)
    weights = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float16)
    contigs = np.array([0, 0], dtype=np.int32)
    vals = np.array([1.0, 1.8], dtype=np.float64)

    baselines, support = evaluate_loco_prototype_baseline(vals, contigs, pids, weights)
    assert baselines[0] == pytest.approx(1.8)
    assert baselines[1] == pytest.approx(1.0)
    np.testing.assert_array_equal(support, [1, 1])

    pids_multi = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=np.uint16,
    )
    w_multi = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float16), (5, 1))
    c_multi = np.zeros(5, dtype=np.int32)
    v_multi = np.array([1.0, 1.0, 1.0, 1.0, 2.0], dtype=np.float64)
    b_multi, s_multi = evaluate_loco_prototype_baseline(
        v_multi, c_multi, pids_multi, w_multi
    )
    assert b_multi[4] == pytest.approx(1.0)
    assert s_multi[4] == 4


def test_calibrate_prototype_dispersion_small_cohort_dof_scaling() -> None:
    """Verify residual DoF variance rescaling (d_eb = S - 1 - P) on small cohorts."""
    rng = np.random.default_rng(42)
    n_bins = 200
    n_samples = 10
    pca_rank = 3
    d_eb = n_samples - 1 - pca_rank
    ddof = 1.0 + pca_rank / n_samples
    expected_scale = (n_samples - ddof) / d_eb
    assert expected_scale == pytest.approx(1.45)

    feats = rng.normal(size=(n_bins, 2)).astype(np.float32)
    centers, pids, pw, _ = fit_prototype_graph(feats, max_prototypes=16, top_r=4)
    contigs = np.array([0] * 100 + [1] * 100, dtype=np.int16)
    usable = np.ones(n_bins, dtype=bool)

    true_phi = 0.04
    lam = 100.0
    eta = rng.normal(-0.5 * true_phi, np.sqrt(true_phi), size=(n_bins, n_samples))
    exp = np.full((n_bins, n_samples), lam, dtype=np.float32)
    counts = rng.poisson(exp * np.exp(eta)).astype(np.float32)

    null_mean, disp = calibrate_prototype_dispersion(
        counts,
        exp,
        contigs,
        usable,
        pids,
        pw,
        pca_rank=pca_rank,
        n_prototypes=int(centers.shape[0]),
    )

    assert np.mean(disp[usable] > 0.0) > 0.90
    med_disp = float(np.median(disp[usable]))
    assert 0.025 < med_disp < 0.080
    assert float(np.median(np.abs(null_mean[usable]))) < 0.15


def test_hsm_biweight_mode_finite_guard_and_empty() -> None:
    """Verify hsm_biweight_mode safely handles NaNs, infinities, and empty inputs."""
    assert hsm_biweight_mode(np.array([], dtype=np.float64)) == 0.0
    assert hsm_biweight_mode(np.array([np.nan, np.nan], dtype=np.float64)) == 0.0
    assert hsm_biweight_mode(np.array([np.inf, -np.inf], dtype=np.float64)) == 0.0

    vals = np.array(
        [np.nan, 1.0, 1.02, 0.98, 1.01, 0.99, 1.0, np.nan, 5.0], dtype=np.float64
    )
    weights = np.ones_like(vals)
    m = hsm_biweight_mode(vals, weights)
    assert 0.98 < m < 1.02
