"""Quality control metrics and JSON generation for prediction runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa

from ccnv.models import (
    BRANCH_FEMALE_SEX,
    BRANCH_MALE_SEX,
    SCHEMA_VERSION,
    AnalysisMode,
    BinResult,
    FetalFractionResult,
    SampleBuffers,
    SegmentResult,
)
from ccnv.qc.sample import write_qc_json

__all__ = [
    "build_predict_qc_payload",
    "write_predict_qc_json",
]


def _int_metric(
    mapping: Mapping[str, object],
    key: str,
    *,
    fallback_key: str | None = None,
) -> int:
    val = mapping.get(key) if key in mapping else None
    if val is None and fallback_key is not None:
        val = mapping.get(fallback_key)
    if isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(
        val, bool
    ):
        return int(val)
    return 0


def build_predict_qc_payload(
    mode: AnalysisMode | Literal["germline", "prenatal"],
    sample_table: SampleBuffers | pa.Table,
    segments: Sequence[SegmentResult],
    prenatal_result: FetalFractionResult | None = None,
    bin_results: Sequence[BinResult] | None = None,
    sex: Literal["female", "male"] | None = None,
) -> dict[str, object]:
    """Construct structured QC metrics for a prediction invocation."""
    resolved_sex: Literal["female", "male"] | None = None
    if sex is not None:
        if sex not in ("female", "male"):
            raise ValueError(f"sex must be 'female' or 'male', got {sex!r}")
        resolved_sex = sex
    elif mode == "prenatal":
        if prenatal_result is not None:
            resolved_sex = "male" if prenatal_result.fetal_sex == "M" else "female"
    elif bin_results:
        branches = {b.branch for b in bin_results}
        if BRANCH_MALE_SEX in branches:
            resolved_sex = "male"
        elif BRANCH_FEMALE_SEX in branches:
            resolved_sex = "female"

    bufs = SampleBuffers.coerce(sample_table)
    qc_meta: Mapping[str, object] = bufs.meta.qc or {}
    counts = np.asarray(bufs.count, dtype=np.float64)
    source_identity: object = bufs.meta.source_identity
    frag_meta = bufs.meta.fragmentomics

    if bin_results is not None and len(bin_results) > 0:
        total_bins = sum(int(b.masked.size) for b in bin_results)
        usable_bins = sum(int(np.count_nonzero(~b.masked)) for b in bin_results)
        masked_bins = total_bins - usable_bins
        usable_zs = [b.zscores[~b.masked] for b in bin_results if np.any(~b.masked)]
        if usable_zs:
            concat_z = np.concatenate(usable_zs)
            finite_z = concat_z[np.isfinite(concat_z)]
            lambda_gc: float | None = (
                float(np.median(finite_z**2) / 0.4549364231195728)
                if finite_z.size > 0
                else 1.0
            )
        else:
            lambda_gc = 1.0
    else:
        total_bins = int(counts.size)
        usable_bins = int(np.count_nonzero(counts > 0))
        masked_bins = total_bins - usable_bins
        lambda_gc = None

    mean_bin_count = float(np.mean(counts)) if counts.size > 0 else 0.0
    median_bin_count = float(np.median(counts)) if counts.size > 0 else 0.0

    effective_reads_examined = _int_metric(
        qc_meta, "effective_reads_examined", fallback_key="reads_seen"
    )
    effective_reads_kept = _int_metric(
        qc_meta, "effective_reads_kept", fallback_key="reads_kept"
    )
    autosomal_reads_kept = _int_metric(qc_meta, "autosomal_reads_kept")
    sex_reads_kept = _int_metric(qc_meta, "sex_reads_kept")

    filter_drops = {
        "unmapped": _int_metric(qc_meta, "filter_unmapped"),
        "mapq": _int_metric(qc_meta, "filter_mapq"),
        "duplicate": _int_metric(qc_meta, "filter_duplicate"),
        "secondary": _int_metric(qc_meta, "filter_secondary"),
        "supplementary": _int_metric(qc_meta, "filter_supplementary"),
        "improper_pair": _int_metric(qc_meta, "filter_improper_pair"),
        "no_start": _int_metric(qc_meta, "filter_no_start"),
    }

    read_metrics: dict[str, object] = {
        "effective_reads_examined": effective_reads_examined,
        "effective_reads_kept": effective_reads_kept,
        "autosomal_reads_kept": autosomal_reads_kept,
        "sex_reads_kept": sex_reads_kept,
        "filter_drops": filter_drops,
        "total_bins": total_bins,
        "usable_bins": usable_bins,
        "masked_bins": masked_bins,
        "mean_bin_count": mean_bin_count,
        "median_bin_count": median_bin_count,
        "lambda_gc": lambda_gc,
    }

    sample_name = "SAMPLE"
    if isinstance(source_identity, Mapping):
        sample_path = source_identity.get("path")
        if isinstance(sample_path, str) and sample_path:
            sample_name = Path(sample_path).stem
        elif isinstance(source_identity.get("sample_name"), str):
            sample_name = str(source_identity["sample_name"])

    total_segments = len(segments)
    gain_calls = sum(1 for s in segments if s.call == "gain")
    loss_calls = sum(1 for s in segments if s.call == "loss")
    discordant_segments = sum(1 for s in segments if s.discordant)

    cnv_summary: dict[str, object] = {
        "total_segments": total_segments,
        "gain_calls": gain_calls,
        "loss_calls": loss_calls,
        "discordant_segments": discordant_segments,
    }

    prenatal_data: dict[str, object] | None = None
    if mode == "prenatal" and prenatal_result is not None:
        origin_counts: dict[str, int] = {}
        for s in segments:
            if s.origin_call:
                origin_counts[s.origin_call] = origin_counts.get(s.origin_call, 0) + 1

        eff_ff = (
            float(prenatal_result.effective_ff)
            if prenatal_result.effective_ff is not None
            else float(prenatal_result.active_ff)
        )
        prenatal_data = {
            "fetal_sex": prenatal_result.fetal_sex,
            "active_ff": float(prenatal_result.active_ff),
            "effective_ff": eff_ff,
            "active_ff_se": (
                float(prenatal_result.active_ff_se)
                if prenatal_result.active_ff_se is not None
                else None
            ),
            "active_ff_source": prenatal_result.active_ff_source,
            "ffy": (
                float(prenatal_result.ffy) if prenatal_result.ffy is not None else None
            ),
            "ffx": float(prenatal_result.ffx),
            "ffa": float(prenatal_result.ffa),
            "ff_size": (
                float(prenatal_result.ff_size)
                if prenatal_result.ff_size is not None
                else None
            ),
            "chrx_ratio": float(prenatal_result.chrx_ratio),
            "chry_score": float(prenatal_result.chry_score),
            "yfrac_threshold": float(prenatal_result.yfrac_threshold),
            "sca": {
                "call": prenatal_result.sca.call,
                "is_aneuploid": bool(prenatal_result.sca.is_aneuploid),
                "confidence": float(prenatal_result.sca.confidence),
                "chrx_zscore": float(prenatal_result.sca.chrx_zscore),
                "ff_delta": (
                    float(prenatal_result.sca.ff_delta)
                    if prenatal_result.sca.ff_delta is not None
                    else None
                ),
            },
            "low_ff_warning": bool(prenatal_result.low_ff_warning),
            "warnings": list(prenatal_result.warnings),
            "origin_summary": origin_counts,
            "lod_ff_35mb": (
                float(prenatal_result.lod_metrics.lod_ff_35mb)
                if prenatal_result.lod_metrics is not None
                else None
            ),
            "lod_ff_3mb": (
                float(prenatal_result.lod_metrics.lod_ff_3mb)
                if prenatal_result.lod_metrics is not None
                else None
            ),
        }

    frag_data: dict[str, object] | None = None
    if frag_meta is not None:
        frag_data = {
            "paired_end": frag_meta.paired_end,
            "total_paired_fragments": int(frag_meta.total_paired_fragments),
            "chromatosome_mode_bp": float(frag_meta.chromatosome_mode_bp),
            "core_shoulder_bp": float(frag_meta.core_shoulder_bp),
            "sizing_shift_bp": float(frag_meta.sizing_shift_bp),
            "self_referenced_peak_ratio": float(frag_meta.self_referenced_peak_ratio),
            "dinucleosome_fraction": float(frag_meta.dinucleosome_fraction),
            "high_mw_fraction": float(frag_meta.high_mw_fraction),
            "helical_10bp_amplitude": float(frag_meta.helical_10bp_amplitude),
            "motif_diversity_score": float(frag_meta.motif_diversity_score),
            "warnings": (
                ["WARN_HIGH_LEUKOCYTE_DNA"] if frag_meta.high_mw_fraction > 0.05 else []
            ),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "mode": str(mode),
        "sample": sample_name,
        "sex": resolved_sex,
        "read_metrics": read_metrics,
        "cnv_summary": cnv_summary,
        "fragmentomics": frag_data,
        "prenatal": prenatal_data,
    }


def write_predict_qc_json(path: Path | str, payload: Mapping[str, object]) -> Path:
    """Atomically write formatted predict QC metrics to a JSON file."""
    return write_qc_json(path, payload)
