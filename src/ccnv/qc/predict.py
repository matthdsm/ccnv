"""Quality control metrics and JSON generation for prediction runs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa

from ccnv.features.fragmentomics import extract_sample_fragmentomics
from ccnv.models import (
    SCHEMA_VERSION,
    AnalysisMode,
    FetalFractionResult,
    SampleBuffers,
    SegmentResult,
    decode_metadata,
)
from ccnv.qc.sample import sanitize_qc_payload

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
    val = mapping.get(key)
    if val is None and fallback_key is not None:
        val = mapping.get(fallback_key)
    if isinstance(val, int | float) and not isinstance(val, bool):
        return int(val)
    return 0


def build_predict_qc_payload(
    mode: AnalysisMode,
    sample_table: pa.Table | SampleBuffers,
    segments: Sequence[SegmentResult],
    prenatal_result: FetalFractionResult | None = None,
) -> dict[str, object]:
    """Construct structured QC metrics for a prediction invocation."""
    if isinstance(sample_table, SampleBuffers):
        qc_meta: Mapping[str, object] = sample_table.meta.qc or {}
        counts = np.asarray(sample_table.count, dtype=np.float64)
        source_identity: object = sample_table.meta.source_identity
        frag_meta = sample_table.meta.fragmentomics
    else:
        sample_meta = decode_metadata(sample_table.schema.metadata)
        qc_meta_raw = sample_meta.get("qc")
        qc_meta = qc_meta_raw if isinstance(qc_meta_raw, Mapping) else {}
        counts = (
            sample_table.column("count")
            .to_numpy(zero_copy_only=False)
            .astype(np.float64)
        )
        source_identity = sample_meta.get("source_identity")
        frag_meta = extract_sample_fragmentomics(sample_meta)
    total_bins = int(counts.size)
    usable_bins = int(np.count_nonzero(counts > 0))
    masked_bins = total_bins - usable_bins
    mean_bin_count = float(np.mean(counts)) if total_bins > 0 else 0.0
    median_bin_count = float(np.median(counts)) if total_bins > 0 else 0.0

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
        prenatal_data = {
            "fetal_sex": prenatal_result.fetal_sex,
            "active_ff": float(prenatal_result.active_ff),
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
        "read_metrics": read_metrics,
        "cnv_summary": cnv_summary,
        "fragmentomics": frag_data,
        "prenatal": prenatal_data,
    }


def write_predict_qc_json(path: Path | str, payload: Mapping[str, object]) -> Path:
    """Write formatted predict QC metrics to a JSON file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sanitized = sanitize_qc_payload(dict(payload))
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(sanitized, handle, indent=2)
    return target
