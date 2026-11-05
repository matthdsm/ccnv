"""Quality control metrics and JSON generation for prediction runs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from ccnv.models import AnalysisMode
from ccnv.results import FetalFractionResult, SegmentResult
from ccnv.schemas import SCHEMA_VERSION, decode_metadata

__all__ = (
    "build_predict_qc_payload",
    "write_predict_qc_json",
)


def build_predict_qc_payload(
    mode: AnalysisMode,
    sample_table: pa.Table,
    segments: Sequence[SegmentResult],
    prenatal_result: FetalFractionResult | None = None,
) -> dict[str, object]:
    """Construct structured QC metrics for a prediction invocation."""

    sample_meta = decode_metadata(sample_table.schema.metadata)
    qc_meta_raw = sample_meta.get("qc")
    qc_meta: Mapping[str, Any] = qc_meta_raw if isinstance(qc_meta_raw, Mapping) else {}

    counts = (
        sample_table.column("count").to_numpy(zero_copy_only=False).astype(np.float64)
    )
    total_bins = int(counts.size)
    usable_bins = int(np.count_nonzero(counts > 0))
    masked_bins = total_bins - usable_bins
    mean_bin_count = float(np.mean(counts)) if total_bins > 0 else 0.0
    median_bin_count = float(np.median(counts)) if total_bins > 0 else 0.0

    effective_reads_examined = int(
        qc_meta.get(
            "effective_reads_examined",
            qc_meta.get("reads_seen", 0),
        )
    )
    effective_reads_kept = int(
        qc_meta.get(
            "effective_reads_kept",
            qc_meta.get("reads_kept", 0),
        )
    )
    autosomal_reads_kept = int(qc_meta.get("autosomal_reads_kept", 0))
    sex_reads_kept = int(qc_meta.get("sex_reads_kept", 0))

    filter_drops = {
        "unmapped": int(qc_meta.get("filter_unmapped", 0)),
        "mapq": int(qc_meta.get("filter_mapq", 0)),
        "duplicate": int(qc_meta.get("filter_duplicate", 0)),
        "secondary": int(qc_meta.get("filter_secondary", 0)),
        "supplementary": int(qc_meta.get("filter_supplementary", 0)),
        "improper_pair": int(qc_meta.get("filter_improper_pair", 0)),
        "no_start": int(qc_meta.get("filter_no_start", 0)),
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

    source_identity = sample_meta.get("source_identity")
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
    discordant_segments = sum(1 for s in segments if getattr(s, "discordant", False))

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
            "active_method": prenatal_result.active_ff_source,
            "ffy": (
                float(prenatal_result.ffy) if prenatal_result.ffy is not None else None
            ),
            "ffx": (
                float(prenatal_result.ffx) if prenatal_result.ffx is not None else None
            ),
            "preface": (
                float(prenatal_result.preface)
                if prenatal_result.preface is not None
                else None
            ),
            "chrx_ratio": float(prenatal_result.chrx_ratio),
            "chry_score": float(prenatal_result.chry_score),
            "yfrac_threshold": float(prenatal_result.yfrac_threshold),
            "discordant_segments_count": discordant_segments,
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

    return {
        "schema_version": SCHEMA_VERSION,
        "mode": str(mode),
        "sample": sample_name,
        "read_metrics": read_metrics,
        "cnv_summary": cnv_summary,
        "prenatal": prenatal_data,
    }


def _sanitize(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 6)
    if isinstance(value, (np.floating,)):
        f = float(value)
        return round(f, 6) if math.isfinite(f) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_predict_qc_json(path: Path, payload: Mapping[str, object]) -> Path:
    """Write formatted predict QC metrics to a JSON file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sanitized = _sanitize(dict(payload))
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(sanitized, handle, indent=2)
    return target
