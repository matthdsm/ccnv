"""Transitional re-export stub for ccnv.reference_compute.

Deprecated in favor of ccnv.stats.neighbors and ccnv.stats.pca.
"""

from __future__ import annotations

from ccnv.stats.neighbors import (
    _MAX_NULL_SAMPLES,
    _NearestContext,
    _NullContext,
    compute_null_models,
    distance_cutoff,
    select_nearest_bins,
)
from ccnv.stats.pca import (
    fit_pca,
    pca_irlba_numpy,
)

__all__ = [
    "_MAX_NULL_SAMPLES",
    "_NearestContext",
    "_NullContext",
    "compute_null_models",
    "distance_cutoff",
    "fit_pca",
    "pca_irlba_numpy",
    "select_nearest_bins",
]
