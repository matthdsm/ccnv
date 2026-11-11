"""Regression tests for Prototype Graph fitting, LOCO baselines, and null dispersion."""

from __future__ import annotations

import numpy as np
import pytest

from ccnv.stats.neighbors import (
    compute_prototype_null_models,
    evaluate_loco_prototype_baseline,
    fit_prototype_graph,
)


def test_single_contig_cluster_on_aneuploid_contig_uses_secondary_off_contig_prototypes() -> (
    None
):
    # Prototype 1 has multiple bins (3 bins), all on trisomic contig 2 (1.5).
    # Prototype 0 spans diploid contigs 0 and 1 (1.0).
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
    _, pids, pweights, _ = fit_prototype_graph(feats, max_prototypes=8, top_r=4)

    # Outlier row 16 is an exact singleton center (dists == 0.0), rounding secondary float16 weights to 0.0
    assert pweights[16, 0] == np.float16(1.0)
    np.testing.assert_array_equal(pweights[16, 1:], np.float16(0.0))

    contigs = np.array([0] * 8 + [1] * 8 + [2], dtype=np.int32)
    vals = np.array([1.0] * 16 + [2.0], dtype=np.float64)
    baselines, support = evaluate_loco_prototype_baseline(vals, contigs, pids, pweights)
    assert baselines[16] == pytest.approx(1.0)
    assert support[16] > 0

    # Null model variance on singleton row 16 does not collapse to 0.0 across noisy samples
    cohort_corrected = rng.lognormal(mean=0.0, sigma=0.12, size=(17, 12))
    null_mean, bio_disp = compute_prototype_null_models(
        cohort_corrected, contigs, pids, pweights, n_pca_components=0
    )
    assert np.isfinite(null_mean[16])
    assert bio_disp[16] > 0.0


def test_all_four_candidates_same_contig_falls_back_to_global_off_contig_median() -> (
    None
):
    # Degenerate fixture: all 4 candidate prototypes for contig 2 bins reside 100% on contig 2
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
    # Single-contig branch (e.g., female_sex chrX) with 2 singleton prototypes
    pids = np.array([[0, 1, 0, 0], [1, 0, 0, 0]], dtype=np.uint16)
    weights = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float16)
    contigs = np.array([0, 0], dtype=np.int32)
    vals = np.array([1.0, 1.8], dtype=np.float64)

    baselines, support = evaluate_loco_prototype_baseline(vals, contigs, pids, weights)
    # Leave-self-out: bin 0 gets 1.8, bin 1 gets 1.0 (never self-canceling vals[i] / vals[i])
    assert baselines[0] == pytest.approx(1.8)
    assert baselines[1] == pytest.approx(1.0)
    np.testing.assert_array_equal(support, [1, 1])

    # Single-contig branch where singleton bin 4 has a multi-bin secondary prototype 0
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
