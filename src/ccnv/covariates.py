"""Genomic covariate track loading, bin masking, and bias correction stub."""

from __future__ import annotations

from ccnv.features.covariates import (
    DEFAULT_MIN_MAPPABILITY,
    correct_covariate_bias,
    covariate_features,
    mask_covariates,
    validate_min_mappability,
)
from ccnv.io.tracks import (
    _bin_weighted_mean,
    load_cytobands_track,
    load_gc_track,
    load_mappability_track,
)
from ccnv.models import (
    CovariateError,
)

__all__ = [
    "DEFAULT_MIN_MAPPABILITY",
    "CovariateError",
    "_bin_weighted_mean",
    "correct_covariate_bias",
    "covariate_features",
    "load_cytobands_track",
    "load_gc_track",
    "load_mappability_track",
    "mask_covariates",
    "validate_min_mappability",
]
