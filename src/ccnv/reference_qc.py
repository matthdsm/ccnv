"""Quality control metrics and usability verdict for reference files (stub)."""

from __future__ import annotations

from ccnv.qc.reference import (
    CHRY_MEAN_DIST_FAIL,
    CHRY_MEAN_DIST_WARN,
    CHRY_MIN_USABLE_PCT_FAIL,
    CHRY_MIN_USABLE_PCT_WARN,
    DEFAULT_MIN_REF_BINS,
    OUTLIER_N_SIGMA,
    SEVERITY_TO_VERDICT,
    VERDICT_SEVERITY,
    ReferenceQCConfig,
    ReferenceQCError,
    compute_branch_qc,
    compute_signal_contig_metrics,
    evaluate_autosomal_or_female_verdict,
    evaluate_male_verdict,
    run_reference_qc,
    run_reference_qc_on_table,
    write_reference_qc_json,
)

__all__ = [
    "CHRY_MEAN_DIST_FAIL",
    "CHRY_MEAN_DIST_WARN",
    "CHRY_MIN_USABLE_PCT_FAIL",
    "CHRY_MIN_USABLE_PCT_WARN",
    "DEFAULT_MIN_REF_BINS",
    "OUTLIER_N_SIGMA",
    "SEVERITY_TO_VERDICT",
    "VERDICT_SEVERITY",
    "ReferenceQCConfig",
    "ReferenceQCError",
    "compute_branch_qc",
    "compute_signal_contig_metrics",
    "evaluate_autosomal_or_female_verdict",
    "evaluate_male_verdict",
    "run_reference_qc",
    "run_reference_qc_on_table",
    "write_reference_qc_json",
]
