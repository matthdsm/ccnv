"""Feature extraction, binning, cohort transformations, and prenatal models."""

from ccnv.features.binning import (
    bin_counts,
    bin_intervals,
    rebin_counts,
)
from ccnv.features.cohort import (
    Cohort,
    SexInference,
    SexLabel,
    cohort_from_fasta,
    infer_sample_sexes,
    mask_low_coverage,
    normalize_depth,
    rebin_cohort,
    rebin_sample_table,
    validate_cohort,
)
from ccnv.features.covariates import (
    DEFAULT_MIN_MAPPABILITY,
    correct_covariate_bias,
    covariate_features,
    mask_covariates,
    validate_min_mappability,
)
from ccnv.features.fragmentomics import (
    evaluate_weighted_coverage,
    extract_bin_fragmentomic_covariates,
    extract_sample_fragmentomics,
    extract_segment_fragment_lengths,
    fit_cohort_fragmentomics_model,
    fuse_active_fetal_fraction,
    score_segment_origin,
)
from ccnv.features.prenatal import (
    annotate_prenatal_segments,
    classify_sca,
    fit_prenatal_models,
    predict_prenatal,
)

__all__ = [
    "DEFAULT_MIN_MAPPABILITY",
    "Cohort",
    "SexInference",
    "SexLabel",
    "annotate_prenatal_segments",
    "bin_counts",
    "bin_intervals",
    "classify_sca",
    "cohort_from_fasta",
    "correct_covariate_bias",
    "covariate_features",
    "evaluate_weighted_coverage",
    "extract_bin_fragmentomic_covariates",
    "extract_sample_fragmentomics",
    "extract_segment_fragment_lengths",
    "fit_cohort_fragmentomics_model",
    "fit_prenatal_models",
    "fuse_active_fetal_fraction",
    "infer_sample_sexes",
    "mask_covariates",
    "mask_low_coverage",
    "normalize_depth",
    "predict_prenatal",
    "rebin_cohort",
    "rebin_counts",
    "rebin_sample_table",
    "score_segment_origin",
    "validate_cohort",
    "validate_min_mappability",
]
