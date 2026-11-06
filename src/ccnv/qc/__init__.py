"""Quality control metrics, diagnostics, verdicts, and JSON reporting."""

from ccnv.qc.predict import (
    build_predict_qc_payload,
    write_predict_qc_json,
)
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
    run_reference_qc_on_table,
    write_reference_qc_json,
)
from ccnv.qc.sample import (
    QC_COUNTERS,
    build_sample_qc,
    empty_qc,
)

__all__ = [
    "CHRY_MEAN_DIST_FAIL",
    "CHRY_MEAN_DIST_WARN",
    "CHRY_MIN_USABLE_PCT_FAIL",
    "CHRY_MIN_USABLE_PCT_WARN",
    "DEFAULT_MIN_REF_BINS",
    "OUTLIER_N_SIGMA",
    "QC_COUNTERS",
    "SEVERITY_TO_VERDICT",
    "VERDICT_SEVERITY",
    "ReferenceQCConfig",
    "ReferenceQCError",
    "build_predict_qc_payload",
    "build_sample_qc",
    "compute_branch_qc",
    "compute_signal_contig_metrics",
    "empty_qc",
    "evaluate_autosomal_or_female_verdict",
    "evaluate_male_verdict",
    "run_reference_qc_on_table",
    "write_predict_qc_json",
    "write_reference_qc_json",
]
