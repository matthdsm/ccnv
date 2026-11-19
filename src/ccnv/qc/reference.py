"""Quality control metrics, usability verdicts, and JSON reporting for references."""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from numpy.typing import NDArray

from ccnv.models import (
    BRANCH_AUTOSOMAL,
    BRANCH_FEMALE_SEX,
    BRANCH_MALE_SEX,
    BRANCH_PRENATAL_AUTOSOMAL,
    BRANCH_PRENATAL_FFY,
    NUM_PROTOTYPES_PER_BIN,
    ReferenceMetadata,
    ReferenceQCError,
    ReferenceQCOutputs,
    SchemaError,
    canonical_dictionary_array,
    decode_dictionary_contigs,
    fixed_size_list_to_numpy,
)
from ccnv.qc.sample import write_qc_json
from ccnv.stats.neighbors import compute_loco_prototype_support

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

__all__ = [
    "CHRY_MEAN_DIST_FAIL",
    "CHRY_MEAN_DIST_WARN",
    "CHRY_MIN_USABLE_PCT_FAIL",
    "CHRY_MIN_USABLE_PCT_WARN",
    "DEFAULT_MIN_REF_BINS",
    "OUTLIER_N_SIGMA",
    "SEVERITY_TO_VERDICT",
    "VERDICT_SEVERITY",
    "ReferenceQCError",
    "compute_branch_qc",
    "compute_signal_contig_metrics",
    "evaluate_autosomal_or_female_verdict",
    "evaluate_male_verdict",
    "run_reference_qc_on_table",
    "write_reference_qc_json",
]


def _metric_int(m: Mapping[str, object], key: str, default: int = 0) -> int:
    val = m.get(key, default)
    if isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(
        val, bool
    ):
        return int(val)
    return default


def _metric_float(m: Mapping[str, object], key: str, default: float = 0.0) -> float:
    val = m.get(key, default)
    if isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(
        val, bool
    ):
        return float(val)
    return default


def _escalate(current: tuple[str, str], candidate: tuple[str, str]) -> tuple[str, str]:
    """Escalate verdict to candidate if candidate has higher severity."""
    if VERDICT_SEVERITY.get(candidate[0], 0) > VERDICT_SEVERITY.get(current[0], 0):
        return candidate
    return current


def _escalate_threshold(
    current: tuple[str, str],
    value: float,
    *,
    warn_above: float,
    fail_above: float,
    warn_msg: str,
    fail_msg: str,
) -> tuple[str, str]:
    if value > fail_above:
        return _escalate(current, ("FAIL", fail_msg))
    if value > warn_above:
        return _escalate(current, ("WARN", warn_msg))
    return current


def _evaluate_base_support(
    m: Mapping[str, object],
    min_ref_bins: int,
) -> tuple[int, tuple[str, str]]:
    """Evaluate unmasked bin presence and LOCO prototype reference support."""
    n_valid = _metric_int(m, "n_valid")
    if n_valid == 0:
        return 0, ("FAIL", "no unmasked bins")

    result = ("PASS", "")
    n_low_refs = _metric_int(m, "n_low_refs")
    n_unmasked = (
        _metric_int(m, "n_bins") - _metric_int(m, "n_masked")
        if "n_bins" in m and "n_masked" in m
        else n_valid
    )
    p_low = n_low_refs / max(n_unmasked, n_valid, 1)
    if p_low > 0.005:
        result = _escalate(
            result,
            (
                "WARN",
                f"n_refs<{min_ref_bins} in {n_low_refs} bins ({p_low * 100:.2f}%)",
            ),
        )
    return n_valid, result


def _escalate_outlier_pct(
    current: tuple[str, str], m: Mapping[str, object]
) -> tuple[str, str]:
    outlier_pct = _metric_float(m, "outlier_pct")
    if outlier_pct > 1.0:
        return _escalate(current, ("WARN", f"outlier bins = {outlier_pct:.2f}%"))
    return current


def compute_signal_contig_metrics(
    contigs: Sequence[str] | NDArray[np.str_],
    signal_contig: str,
    mean_d: NDArray[np.float64],
    n_refs: NDArray[np.int64],
    cutoff_outlier: float,
    min_ref_bins: int,
    *,
    total_bins: int | None = None,
) -> dict[str, object] | None:
    """Compute sex-signal contig (e.g. chrY) specific metrics for male branch."""
    mask = np.asarray(contigs) == signal_contig
    if not np.any(mask):
        if total_bins is None or total_bins <= 0:
            return None
        return {
            "signal_contig": signal_contig,
            "n_bins": total_bins,
            "n_valid": 0,
            "mean_of_means": None,
            "std_of_means": None,
            "n_mean_outlier": 0,
            "n_low_refs": 0,
            "n_usable": 0,
            "usable_pct": 0.0,
        }

    m = mean_d[mask]
    r = n_refs[mask]
    valid = np.isfinite(m)
    n_valid = int(np.sum(valid))
    denom = total_bins if total_bins is not None else int(m.size)
    if n_valid == 0:
        return {
            "signal_contig": signal_contig,
            "n_bins": denom,
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
    usable_pct = float(100.0 * n_usable / max(denom, 1))

    return {
        "signal_contig": signal_contig,
        "n_bins": denom,
        "n_valid": n_valid,
        "mean_of_means": float(np.mean(m[valid])),
        "std_of_means": float(np.std(m[valid])),
        "n_mean_outlier": int(np.sum(m[valid] >= cutoff_outlier)),
        "n_low_refs": int(np.sum(r < min_ref_bins)),
        "n_usable": n_usable,
        "usable_pct": usable_pct,
    }


def evaluate_autosomal_or_female_verdict(
    m: Mapping[str, object],
    min_ref_bins: int,
    dist_scale: float = 1.0,
) -> tuple[str, str]:
    """Evaluate quality verdict for autosomal or female_sex branch."""
    n_valid, result = _evaluate_base_support(m, min_ref_bins)
    if n_valid == 0:
        return result

    result = _escalate_outlier_pct(result, m)
    std_of_means = _metric_float(m, "std_of_means")
    return _escalate_threshold(
        result,
        std_of_means,
        warn_above=2.0 * dist_scale,
        fail_above=10.0 * dist_scale,
        warn_msg=f"std(per-bin mean dist) = {std_of_means:.2f}",
        fail_msg=f"std(per-bin mean dist) = {std_of_means:.2f} (high)",
    )


def evaluate_male_verdict(
    m: Mapping[str, object],
    min_ref_bins: int,
    dist_scale: float = 1.0,
) -> tuple[str, str]:
    """Evaluate quality verdict for male_sex branch."""
    n_valid, result = _evaluate_base_support(m, min_ref_bins)
    if n_valid == 0:
        return result

    mean_of_means = _metric_float(m, "mean_of_means")
    result = _escalate_threshold(
        result,
        mean_of_means,
        warn_above=2.0 * dist_scale,
        fail_above=10.0 * dist_scale,
        warn_msg=f"mean(per-bin mean dist) = {mean_of_means:.2f}",
        fail_msg=f"mean(per-bin mean dist) = {mean_of_means:.2f} (heavy tail)",
    )

    cy = m.get("signal_contig_metrics")
    if isinstance(cy, Mapping):
        usable_pct = _metric_float(cy, "usable_pct", float("nan"))
        sig_name = str(cy.get("signal_contig", "chrY"))
        n_usable = cy.get("n_usable")
        if (n_usable == 0 and (usable_pct == 0.0 or not np.isfinite(usable_pct))) or (
            np.isfinite(usable_pct) and usable_pct < CHRY_MIN_USABLE_PCT_FAIL
        ):
            result = _escalate(
                result,
                (
                    "FAIL",
                    f"usable {sig_name} bins = {usable_pct:.1f}% (<{CHRY_MIN_USABLE_PCT_FAIL:.0f}%)",
                ),
            )
        elif np.isfinite(usable_pct) and usable_pct < CHRY_MIN_USABLE_PCT_WARN:
            result = _escalate(
                result,
                (
                    "WARN",
                    f"usable {sig_name} bins = {usable_pct:.1f}% (<{CHRY_MIN_USABLE_PCT_WARN:.0f}%)",
                ),
            )

        ym_raw = cy.get("mean_of_means")
        if isinstance(ym_raw, (int, float, np.integer, np.floating)) and not isinstance(
            ym_raw, bool
        ):
            ym = float(ym_raw)
            if np.isfinite(ym):
                result = _escalate_threshold(
                    result,
                    ym,
                    warn_above=CHRY_MEAN_DIST_WARN * dist_scale,
                    fail_above=CHRY_MEAN_DIST_FAIL * dist_scale,
                    warn_msg=f"{sig_name} mean distance = {ym:.1f}",
                    fail_msg=f"{sig_name} mean distance = {ym:.1f} (very poor {sig_name})",
                )

    return _escalate_outlier_pct(result, m)


def compute_branch_qc(
    table: pa.Table,
    branch_name: str,
    signal_contig: str | None,
    min_ref_bins: int,
    dist_scale: float = 1.0,
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
    pids = fixed_size_list_to_numpy(
        unmasked_table.column("proto_ids"), NUM_PROTOTYPES_PER_BIN, np.uint16
    )
    pweights = fixed_size_list_to_numpy(
        unmasked_table.column("proto_weights"), NUM_PROTOTYPES_PER_BIN, np.float16
    )
    c_dict = canonical_dictionary_array(unmasked_table.column("contig"), "contig")
    c_codes = c_dict.indices.to_numpy(zero_copy_only=False)
    contigs = list(decode_dictionary_contigs(c_dict))

    n_refs, _ = compute_loco_prototype_support(c_codes, pids, pweights)
    eff_min_refs = min(min_ref_bins, max(1, n_unmasked // 2))

    sum_proto_w = np.sum(pweights, axis=1)
    zero_weight = (sum_proto_w <= 0.0) | (n_refs == 0)
    mean_d = np.where(zero_weight, np.nan, mean_d)

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
    n_low_refs = int(np.sum(n_refs < eff_min_refs))

    sig_metrics = None
    if branch_name == BRANCH_MALE_SEX and signal_contig:
        all_contigs = decode_dictionary_contigs(branch_table.column("contig"))
        total_sig = sum(1 for c in all_contigs if c == signal_contig)
        if total_sig > 0:
            sig_metrics = compute_signal_contig_metrics(
                contigs,
                signal_contig,
                mean_d,
                n_refs,
                cutoff_outlier,
                eff_min_refs,
                total_bins=total_sig,
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

    evaluator = (
        evaluate_male_verdict
        if branch_name == BRANCH_MALE_SEX
        else evaluate_autosomal_or_female_verdict
    )
    verdict, message = evaluator(metrics, eff_min_refs, dist_scale=dist_scale)
    metrics["verdict"] = verdict
    metrics["message"] = message
    return metrics


def write_reference_qc_json(path: Path | str, data: Mapping[str, object]) -> Path:
    """Atomically serialize reference quality metrics to a JSON file."""
    target = write_qc_json(path, data)
    logger.debug("wrote reference QC report to %s", target)
    return target


def run_reference_qc_on_table(
    table: pa.Table,
    prefix: Path | str | None = None,
    *,
    reference_path: Path | None = None,
    write_json: bool = True,
    emit_stderr: bool = True,
) -> ReferenceQCOutputs:
    """Analyze an in-memory or loaded reference table, evaluate verdict, and write JSON."""
    raw_meta = table.schema.metadata
    if raw_meta is None or b"ccnv.model" not in raw_meta:
        raise ReferenceQCError(
            "reference metadata does not contain a valid model object"
        )
    try:
        ref_meta = ReferenceMetadata.from_table(table)
    except SchemaError as err:
        raise ReferenceQCError(str(err)) from err

    is_flat = ref_meta.flat
    min_ref_bins = DEFAULT_MIN_REF_BINS
    signal_contig = ref_meta.sex_signal_contig

    n_cov = (
        int(ref_meta.gc_corrected)
        + int(ref_meta.mappability_corrected)
        + (2 if ref_meta.fragmentomics_corrected else 0)
    )

    branches_qc: dict[str, object] = {}
    severities: list[int] = []
    messages: list[str] = []

    cnv_branch_names = tuple(
        str(b)
        for b in pc.unique(table.column("branch")).to_pylist()
        if b in (BRANCH_AUTOSOMAL, BRANCH_FEMALE_SEX, BRANCH_MALE_SEX)
    )
    logger.debug(
        "evaluating reference QC (branches=%s, flat=%s, %d rows)",
        cnv_branch_names,
        is_flat,
        table.num_rows,
    )
    for branch_name in cnv_branch_names:
        branch_rank = (
            0
            if is_flat
            else (
                ref_meta.branch_pca_ranks.get(branch_name, ref_meta.pca_components)
                if ref_meta.branch_pca_ranks
                else ref_meta.pca_components
            )
        )
        dist_scale = float(np.sqrt(max(int(branch_rank) + n_cov, 1)))
        b_qc = compute_branch_qc(
            table,
            branch_name=branch_name,
            signal_contig=signal_contig,
            min_ref_bins=min_ref_bins,
            dist_scale=dist_scale,
        )
        if b_qc is not None:
            branches_qc[branch_name] = b_qc
            v = str(b_qc.get("verdict", "PASS"))
            logger.debug(
                "branch %s QC: %s (masked=%.1f%%, outlier=%.1f%%, low_refs=%d)",
                branch_name,
                v,
                _metric_float(b_qc, "mask_pct"),
                _metric_float(b_qc, "outlier_pct"),
                _metric_int(b_qc, "n_low_refs"),
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

    sample_count = ref_meta.sample_count
    mode = str(ref_meta.mode)

    prenatal_qc: dict[str, object] | None = None
    if ref_meta.mode == "prenatal":
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

        p_meta = ref_meta.prenatal
        prenatal_qc = {
            "usable_autosomal_bins": usable_auto,
            "informative_ffy_bins": informative_ffy,
            "yfrac_threshold": (
                float(p_meta.yfrac_threshold) if p_meta is not None else 0.0
            ),
            "female_chrx_ratio_std": (
                float(p_meta.female_chrx_ratio_std) if p_meta is not None else 0.0
            ),
        }

    json_path: Path | None = (
        Path(f"{prefix}_qc.json") if (prefix is not None and write_json) else None
    )

    qc_data: dict[str, object] = {
        "overall_verdict": overall_verdict,
        "worst_severity": worst,
        "messages": messages,
        "parameters": {
            "bin_size": ref_meta.bin_size,
            "mode": mode,
            "prenatal_bin_size": ref_meta.prenatal_bin_size,
            "pca_components": ref_meta.pca_components,
            "min_ref_bins": min_ref_bins,
            "flat": is_flat,
            "autosomal_contigs": list(ref_meta.autosomal_contigs),
            "sex_contigs": list(ref_meta.sex_contigs),
            "sex_signal_contig": signal_contig,
            "sex_cutoff": ref_meta.sex_cutoff,
            "sex_source": ref_meta.sex_source,
            "gc_corrected": ref_meta.gc_corrected,
            "mappability_corrected": ref_meta.mappability_corrected,
            "fragmentomics_corrected": ref_meta.fragmentomics_corrected,
            "min_mappability": ref_meta.min_mappability,
        },
        "cohort": {
            "sample_count": sample_count,
            "sample_sexes": (
                list(ref_meta.sample_sexes)
                if ref_meta.sample_sexes is not None
                else None
            ),
            "sex_inference": ref_meta.sex_inference,
        },
        "branches": branches_qc,
    }
    if prenatal_qc is not None:
        qc_data["prenatal"] = prenatal_qc
    if reference_path is not None:
        qc_data["reference_path"] = str(reference_path)

    if json_path is not None:
        write_reference_qc_json(json_path, qc_data)

    if emit_stderr:
        auto_qc = branches_qc.get("autosomal")
        mask_info = ""
        if isinstance(auto_qc, Mapping) and "mask_pct" in auto_qc:
            mask_info = f", {_metric_float(auto_qc, 'mask_pct'):.1f}% masked"
        cohort_info = "flat FASTA" if is_flat else f"{sample_count} samples"
        suffix = f" -> {json_path}" if json_path is not None else ""
        summary_line = (
            f"Reference QC: {overall_verdict} ({cohort_info}{mask_info}){suffix}"
        )
        sys.stderr.write(f"{summary_line}\n")
        sys.stderr.flush()

    return ReferenceQCOutputs(
        json_path=json_path,
        overall_verdict=overall_verdict,
        worst_severity=worst,
        data=qc_data,
    )
