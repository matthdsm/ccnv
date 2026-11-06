"""Cohort sample loading and layout validation (transitional stub).

Deprecated in favor of ccnv.features.cohort and ccnv.models.
"""

from __future__ import annotations

from ccnv.features.cohort import (
    Cohort,
    SexInference,
    SexLabel,
    cohort_from_fasta,
    infer_sample_sexes,
    load_cohort,
    mask_low_coverage,
    normalize_depth,
    rebin_sample_table,
    validate_cohort,
)
from ccnv.models import ReferenceError

__all__ = [
    "Cohort",
    "ReferenceError",
    "SexInference",
    "SexLabel",
    "cohort_from_fasta",
    "infer_sample_sexes",
    "load_cohort",
    "mask_low_coverage",
    "normalize_depth",
    "rebin_sample_table",
    "validate_cohort",
]
