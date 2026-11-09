"""Stateless numerical, statistical, and machine learning kernels for ccnv."""

from __future__ import annotations

from ccnv.stats.neighbors import (
    build_signal_space_features,
    compute_null_models,
    distance_cutoff,
    select_nearest_bins,
    squeeze_var,
)
from ccnv.stats.pca import (
    fit_pca,
    optimal_svht_rank,
    pca_irlba_numpy,
)
from ccnv.stats.prenatal_math import (
    fit_lasso_fista,
    fit_rprop_mlp,
    infer_sex_gmm_1d,
    ols_fit,
    rlm_huber,
    train_olm,
)
from ccnv.stats.regression import (
    apply_1d_correction,
    build_1d_basis,
    build_joint_covariate_basis,
    fit_sample_trend,
)
from ccnv.stats.segmentation import (
    DEFAULT_MAX_BLACKLIST_GAP,
    Algorithm,
    Segment,
    WeightedL2Cost,
    segment_contig,
    segment_genome,
)

__all__ = [
    "DEFAULT_MAX_BLACKLIST_GAP",
    "Algorithm",
    "Segment",
    "WeightedL2Cost",
    "apply_1d_correction",
    "build_1d_basis",
    "build_joint_covariate_basis",
    "build_signal_space_features",
    "compute_null_models",
    "distance_cutoff",
    "fit_lasso_fista",
    "fit_pca",
    "fit_rprop_mlp",
    "fit_sample_trend",
    "infer_sex_gmm_1d",
    "ols_fit",
    "optimal_svht_rank",
    "pca_irlba_numpy",
    "rlm_huber",
    "segment_contig",
    "segment_genome",
    "select_nearest_bins",
    "squeeze_var",
    "train_olm",
]
