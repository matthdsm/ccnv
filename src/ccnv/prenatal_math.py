"""Transitional re-export stub for ccnv.prenatal_math.

Deprecated in favor of ccnv.stats.prenatal_math and ccnv.stats.pca.
"""

from __future__ import annotations

from sklearn.linear_model import HuberRegressor as HuberRegressor

from ccnv.stats.pca import pca_irlba_numpy
from ccnv.stats.prenatal_math import (
    fit_lasso_fista,
    fit_rprop_mlp,
    infer_sex_gmm_1d,
    ols_fit,
    rlm_huber,
    train_olm,
)

__all__ = [
    "fit_lasso_fista",
    "fit_rprop_mlp",
    "infer_sex_gmm_1d",
    "ols_fit",
    "pca_irlba_numpy",
    "rlm_huber",
    "train_olm",
]
