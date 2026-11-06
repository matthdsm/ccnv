"""Quality control metrics and usability verdict for reference files."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from numpy.typing import NDArray

from ccnv.models import ReferenceQCConfig, ReferenceQCError
from ccnv.results import ReferenceQCOutputs
from ccnv.schemas import (
    BRANCH_AUTOSOMAL,
    BRANCH_FEMALE_SEX,
    BRANCH_MALE_SEX,
    BRANCH_PRENATAL_AUTOSOMAL,
    BRANCH_PRENATAL_FFY,
    ArtifactKind,
    SchemaError,
    decode_metadata,
    validate_table,
)

DEFAULT_MIN_REF_BINS: Final[int] = 150
OUTLIER_N_SIGMA: Final[float] = 3.0
CHRY_MIN_USABLE_PCT_WARN: Final[float] = 50.0
CHRY_MIN_USABLE_PCT_FAIL: Final[float] = 20.0
CHRY_MEAN_DIST_WARN: Final[float] = 5.0
CHRY_MEAN_DIST_FAIL: Final[float] = 100.0

logger = logging.getLogger(__name__)


VERDICT_SEVERITY: Final[dict[str, int]] = {
    "PASS": 0,
    "WARN": 1,
    "FAIL": 2,
}
SEVERITY_TO_VERDICT: Final[dict[int, str]] = {
    0: "PASS",
    1: "WARN",
    2: "FAIL",
}


def _escalate(current: tuple[str, str], candidate: tuple[str, str]) -> tuple[str, str]:
    """Escalate verdict to candidate if candidate has higher severity."""
    if VERDICT_SEVERITY.get(candidate[0], 0) > VERDICT_SEVERITY.get(current[0], 0):
        return candidate
    return current


def compute_signal_contig_metrics(
    contigs: list[str] | NDArray[np.str_],
    signal_contig: str,
    mean_d: NDArray[np.float64],
    n_refs: NDArray[np.int64],
    cutoff_outlier: float,
    min_ref_bins: int,
) -> dict[str, object] | None:
    """Compute sex-signal contig (e.g. chrY) specific metrics for male branch."""

    mask = np.asarray(contigs) == signal_contig
    if not np.any(mask):
        return None

    m = mean_d[mask]
    r = n_refs[mask]
    valid = np.isfinite(m)
    n_valid = int(np.sum(valid))
    n_bins = int(m.size)
    if n_valid == 0:
        return {
            "signal_contig": signal_contig,
            "n_bins": n_bins,
            "n_valid": 0,
            "mean_of_means": None,
            "std_of_means": None,
            "n_mean_outlier": 0,
            "n_low_refs": 0,
            "n_usable": 0,
            "usable_pct": 0.0,
        }

    usable = valid & (r >= min_ref_bins)
    n_usable = int(np.sum(usable))
    usable_pct = float(100.0 * n_usable / max(n_bins, 1))

    return {
        "signal_contig": signal_contig,
        "n_bins": n_bins,
        "n_valid": n_valid,
        "mean_of_means": float(np.mean(m[valid])),
        "std_of_means": float(np.std(m[valid])),
        "n_mean_outlier": int(np.sum(m[valid] >= cutoff_outlier)),
        "n_low_refs": int(np.sum(r < min_ref_bins)),
        "n_usable": n_usable,
        "usable_pct": usable_pct,
    }


def evaluate_autosomal_or_female_verdict(
    m: dict[str, object],
    min_ref_bins: int,
) -> tuple[str, str]:
    """Evaluate quality verdict for autosomal or female_sex branch."""

    if m.get("n_valid", 0) == 0:
        return "FAIL", "no unmasked bins"

    result = ("PASS", "")
    n_low_refs = int(m["n_low_refs"])
    if n_low_refs > 0:
        result = _escalate(
            result, ("WARN", f"n_refs<{min_ref_bins} in {n_low_refs} bins")
        )

    outlier_pct = float(m["outlier_pct"])
    if outlier_pct > 1.0:
        result = _escalate(result, ("WARN", f"outlier bins = {outlier_pct:.2f}%"))

    std_of_means = float(m["std_of_means"])
    if std_of_means > 10.0:
        result = _escalate(
            result, ("FAIL", f"std(per-bin mean dist) = {std_of_means:.2f} (high)")
        )
    elif std_of_means > 2.0:
        result = _escalate(
            result, ("WARN", f"std(per-bin mean dist) = {std_of_means:.2f}")
        )

    return result


def evaluate_male_verdict(
    m: dict[str, object],
    min_ref_bins: int,
) -> tuple[str, str]:
    """Evaluate quality verdict for male_sex branch."""

    if m.get("n_valid", 0) == 0:
        return "FAIL", "no unmasked bins"

    result = ("PASS", "")
    n_low_refs = int(m["n_low_refs"])
    if n_low_refs > 0:
        result = _escalate(
            result, ("WARN", f"n_refs<{min_ref_bins} in {n_low_refs} bins")
        )

    mean_of_means = float(m["mean_of_means"])
    if mean_of_means > 10.0:
        result = _escalate(
            result,
            ("FAIL", f"mean(per-bin mean dist) = {mean_of_means:.2f} (heavy tail)"),
        )
    elif mean_of_means > 2.0:
        result = _escalate(
            result, ("WARN", f"mean(per-bin mean dist) = {mean_of_means:.2f}")
        )

    cy = m.get("signal_contig_metrics")
    if isinstance(cy, dict) and cy.get("n_valid", 0) > 0:
        usable_pct = float(cy.get("usable_pct", float("nan")))
        sig_name = cy.get("signal_contig", "chrY")
        if np.isfinite(usable_pct):
            if usable_pct < CHRY_MIN_USABLE_PCT_FAIL:
                result = _escalate(
                    result,
                    (
                        "FAIL",
                        f"usable {sig_name} bins = {usable_pct:.1f}% (<{CHRY_MIN_USABLE_PCT_FAIL:.0f}%)",  # noqa
                    ),
                )
            elif usable_pct < CHRY_MIN_USABLE_PCT_WARN:
                result = _escalate(
                    result,
                    (
                        "WARN",
                        f"usable {sig_name} bins = {usable_pct:.1f}% (<{CHRY_MIN_USABLE_PCT_WARN:.0f}%)",  # noqa
                    ),
                )
        ym = cy.get("mean_of_means")
        if ym is not None and np.isfinite(ym):
            if float(ym) > CHRY_MEAN_DIST_FAIL:
                result = _escalate(
                    result,
                    (
                        "FAIL",
                        f"{sig_name} mean distance = {float(ym):.1f} (very poor {sig_name})",  # noqa
                    ),
                )
            elif float(ym) > CHRY_MEAN_DIST_WARN:
                result = _escalate(
                    result, ("WARN", f"{sig_name} mean distance = {float(ym):.1f}")
                )

    outlier_pct = float(m["outlier_pct"])
    if outlier_pct > 1.0:
        result = _escalate(result, ("WARN", f"outlier bins = {outlier_pct:.2f}%"))

    return result


def compute_branch_qc(
    table: pa.Table,
    branch_name: str,
    signal_contig: str | None,
    min_ref_bins: int,
) -> dict[str, object] | None:
    """Extract metrics and verdict for one reference branch."""

    branch_table = table.filter(pc.equal(table["branch"], branch_name))
    total_branch_bins = branch_table.num_rows
    if total_branch_bins == 0:
        return None

    masked_arr = branch_table.column("masked").to_numpy(zero_copy_only=False)
    n_masked = int(np.count_nonzero(masked_arr))
    mask_pct = float(100.0 * n_masked / max(total_branch_bins, 1))

    unmasked_table = branch_table.filter(pc.invert(branch_table["masked"]))
    n_unmasked = unmasked_table.num_rows
    if n_unmasked == 0:
        return {
            "n_bins": total_branch_bins,
            "n_valid": 0,
            "n_masked": n_masked,
            "mask_pct": mask_pct,
            "verdict": "FAIL",
            "message": "all bins masked",
        }

    mean_d = (
        unmasked_table.column("mean_distance")
        .to_numpy(zero_copy_only=False)
        .astype(np.float64)
    )
    indices_col = unmasked_table.column("nearest_ref_indices").combine_chunks()
    offsets = indices_col.offsets.to_numpy()
    n_refs = (offsets[1:] - offsets[:-1]).astype(np.int64)
    contigs = np.asarray(unmasked_table.column("contig"))
    valid = np.isfinite(mean_d)
    n_valid = int(np.sum(valid))
    if n_valid == 0:
        return {
            "n_bins": total_branch_bins,
            "n_valid": 0,
            "n_masked": n_masked,
            "mask_pct": mask_pct,
            "verdict": "FAIL",
            "message": "no valid bin distances",
        }

    mean_of_means = float(np.mean(mean_d[valid]))
    std_of_means = float(np.std(mean_d[valid]))
    if std_of_means > 0.0:
        cutoff_outlier = float(mean_of_means + OUTLIER_N_SIGMA * std_of_means)
        n_mean_outlier = int(np.sum(mean_d[valid] >= cutoff_outlier))
    else:
        cutoff_outlier = mean_of_means
        n_mean_outlier = 0
    outlier_pct = float(100.0 * n_mean_outlier / n_valid)
    n_low_refs = int(np.sum(n_refs < min_ref_bins))

    sig_metrics = None
    if branch_name == BRANCH_MALE_SEX and signal_contig:
        sig_metrics = compute_signal_contig_metrics(
            contigs, signal_contig, mean_d, n_refs, cutoff_outlier, min_ref_bins
        )

    metrics: dict[str, object] = {
        "n_bins": total_branch_bins,
        "n_valid": n_valid,
        "n_masked": n_masked,
        "mask_pct": mask_pct,
        "mean_of_means": mean_of_means,
        "std_of_means": std_of_means,
        "n_mean_outlier": n_mean_outlier,
        "outlier_pct": outlier_pct,
        "n_low_refs": n_low_refs,
        "signal_contig_metrics": sig_metrics,
    }

    if branch_name == BRANCH_MALE_SEX:
        verdict, message = evaluate_male_verdict(metrics, min_ref_bins)
    else:
        verdict, message = evaluate_autosomal_or_female_verdict(metrics, min_ref_bins)

    metrics["verdict"] = verdict
    metrics["message"] = message
    return metrics


def run_reference_qc_on_table(
    table: pa.Table,
    prefix: Path,
    *,
    reference_path: Path | None = None,
    write_json: bool = True,
    emit_stderr: bool = True,
) -> ReferenceQCOutputs:
    """Analyze an in-memory or loaded reference table, evaluate verdict, and write JSON."""  # noqa

    metadata = decode_metadata(table.schema.metadata)
    model = metadata.get("model", {})
    if not isinstance(model, dict):
        raise ReferenceQCError(
            "reference metadata does not contain a valid model object"
        )

    is_flat = bool(model.get("flat", False))
    nearest_bins = int(model.get("nearest_bins", 300))
    min_ref_bins = min(DEFAULT_MIN_REF_BINS, max(1, nearest_bins // 2))
    signal_contig = model.get("sex_signal_contig")
    if signal_contig is not None:
        signal_contig = str(signal_contig)

    branches_qc: dict[str, object] = {}
    severities: list[int] = []
    messages: list[str] = []

    cnv_branch_names = tuple(
        b
        for b in dict.fromkeys(table.column("branch").to_pylist())
        if b in (BRANCH_AUTOSOMAL, BRANCH_FEMALE_SEX, BRANCH_MALE_SEX)
    )
    logger.debug(
        "evaluating reference QC (branches=%s, flat=%s, %d rows)",
        cnv_branch_names,
        is_flat,
        table.num_rows,
    )
    for branch_name in cnv_branch_names:
        b_qc = compute_branch_qc(
            table,
            branch_name=branch_name,
            signal_contig=signal_contig,
            min_ref_bins=min_ref_bins,
        )
        if b_qc is not None:
            branches_qc[branch_name] = b_qc
            v = str(b_qc.get("verdict", "PASS"))
            logger.debug(
                "branch %s QC: %s (masked=%.1f%%, outlier=%.1f%%, low_refs=%d)",
                branch_name,
                v,
                float(b_qc.get("mask_pct", 0.0)),
                float(b_qc.get("outlier_pct", 0.0)),
                int(b_qc.get("n_low_refs", 0)),
            )
            severities.append(VERDICT_SEVERITY.get(v, 0))
            if b_qc.get("message"):
                messages.append(f"[{branch_name}] {b_qc['message']}")

    if is_flat:
        severities.append(1)
        messages.append(
            "flat FASTA reference lacks empirical cohort noise models and sex branches"
        )

    worst = max(severities) if severities else 0
    overall_verdict = SEVERITY_TO_VERDICT[worst]
    logger.debug(
        "reference QC overall: %s (worst_severity=%d, messages=%s)",
        overall_verdict,
        worst,
        messages,
    )

    sample_identities = metadata.get("sample_identities", [])
    sample_count = len(sample_identities) if isinstance(sample_identities, list) else 0

    mode = model.get("mode", "germline")
    prenatal_qc: dict[str, object] | None = None
    if mode == "prenatal":
        prenatal_meta_dict = metadata.get("prenatal")
        if not isinstance(prenatal_meta_dict, dict):
            prenatal_meta_dict = {}

        auto_rows = table.filter(pc.equal(table["branch"], BRANCH_PRENATAL_AUTOSOMAL))
        usable_auto = (
            int(pc.sum(pc.invert(auto_rows["masked"])).as_py() or 0)
            if auto_rows.num_rows > 0
            else 0
        )

        ffy_rows = table.filter(pc.equal(table["branch"], BRANCH_PRENATAL_FFY))
        informative_ffy = (
            int(pc.sum(pc.invert(ffy_rows["masked"])).as_py() or 0)
            if ffy_rows.num_rows > 0
            else 0
        )

        yfrac_threshold = float(prenatal_meta_dict.get("yfrac_threshold", 0.0))
        female_chrx_ratio_std = float(
            prenatal_meta_dict.get("female_chrx_ratio_std", 0.0)
        )

        prenatal_qc = {
            "usable_autosomal_bins": usable_auto,
            "informative_ffy_bins": informative_ffy,
            "yfrac_threshold": yfrac_threshold,
            "female_chrx_ratio_std": female_chrx_ratio_std,
        }

    json_path = Path(f"{prefix}_qc.json")
    qc_data: dict[str, object] = {
        "overall_verdict": overall_verdict,
        "worst_severity": worst,
        "messages": messages,
        "parameters": {
            "bin_size": model.get("bin_size"),
            "mode": mode,
            "prenatal_bin_size": model.get("prenatal_bin_size"),
            "pca_components": model.get("pca_components"),
            "nearest_bins": nearest_bins,
            "min_ref_bins": min_ref_bins,
            "flat": is_flat,
            "autosomal_contigs": model.get("autosomal_contigs"),
            "sex_contigs": model.get("sex_contigs"),
            "sex_signal_contig": signal_contig,
            "sex_cutoff": model.get("sex_cutoff"),
            "sex_source": model.get("sex_source"),
            "gc_corrected": bool(model.get("gc_corrected", False)),
            "mappability_corrected": bool(model.get("mappability_corrected", False)),
            "min_mappability": model.get("min_mappability"),
        },
        "cohort": {
            "sample_count": sample_count,
            "sample_identities": metadata.get("sample_identities"),
            "sample_sexes": model.get("sample_sexes"),
            "sex_inference": model.get("sex_inference"),
        },
        "branches": branches_qc,
    }
    if prenatal_qc is not None:
        qc_data["prenatal"] = prenatal_qc
    if reference_path is not None:
        qc_data["reference_path"] = str(reference_path)

    if write_json:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(qc_data, f, indent=2)
        logger.debug("wrote reference QC report to %s", json_path)

    if emit_stderr:
        auto_qc = branches_qc.get("autosomal")
        mask_info = ""
        if isinstance(auto_qc, dict) and "mask_pct" in auto_qc:
            mask_info = f", {float(auto_qc['mask_pct']):.1f}% masked"
        cohort_info = "flat FASTA" if is_flat else f"{sample_count} samples"
        summary_line = (
            f"Reference QC: {overall_verdict} ({cohort_info}{mask_info}) -> {json_path}"
        )
        sys.stderr.write(f"{summary_line}\n")
        sys.stderr.flush()

    return ReferenceQCOutputs(
        json_path=json_path,
        overall_verdict=overall_verdict,
        worst_severity=worst,
        data=qc_data,
    )


def run_reference_qc(config: ReferenceQCConfig) -> ReferenceQCOutputs:
    """Validate and run reference QC against a reference Parquet file."""

    ref_path = config.reference.expanduser().resolve()
    if not ref_path.is_file():
        raise ReferenceQCError(f"reference file not found: {ref_path}")

    try:
        table = pq.read_table(ref_path)
    except Exception as error:
        raise ReferenceQCError(f"failed to read {ref_path}: {error}") from error

    try:
        validate_table(table, ArtifactKind.REFERENCE)
    except SchemaError as error:
        raise ReferenceQCError(f"{ref_path}: {error}") from error

    prefix = config.prefix if config.prefix is not None else ref_path.with_suffix("")

    return run_reference_qc_on_table(
        table,
        prefix=prefix,
        reference_path=ref_path,
        write_json=True,
        emit_stderr=True,
    )
