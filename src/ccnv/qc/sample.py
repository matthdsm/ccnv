"""Sample-level read counting and quality control summary assembly."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import numpy as np

QC_COUNTERS: Final[tuple[str, ...]] = (
    "header_mapped",
    "header_unmapped",
    "header_no_coordinate",
    "reads_seen",
    "reads_kept",
    "mate_dedup_skipped",
    "reads_filtered",
    "effective_reads_examined",
    "effective_reads_kept",
    "autosomal_reads_kept",
    "sex_reads_kept",
    "filter_unmapped",
    "filter_no_start",
    "filter_secondary",
    "filter_supplementary",
    "filter_duplicate",
    "filter_improper_pair",
    "filter_mapq",
)

__all__ = [
    "QC_COUNTERS",
    "build_sample_qc",
    "empty_qc",
    "sanitize_qc_payload",
]


def sanitize_qc_payload(value: object) -> object:
    """Recursively convert NumPy scalars and special floats to JSON-safe types."""
    if isinstance(value, Mapping):
        return {str(k): sanitize_qc_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_qc_payload(v) for v in value]
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


def empty_qc() -> dict[str, int]:
    """Return a complete QC counter map with deterministic key order."""
    return {name: 0 for name in QC_COUNTERS}


def build_sample_qc(
    counters: Mapping[str, int] | None = None,
    *,
    effective_reads_examined: int | None = None,
    effective_reads_kept: int | None = None,
    **kwargs: int,
) -> dict[str, int]:
    """Assemble deterministic sample QC counter dictionary with effective reads."""
    result = empty_qc()
    if counters is not None:
        for k, v in counters.items():
            if k in result:
                result[k] = int(v)
    for k, v in kwargs.items():
        if k in result:
            result[k] = int(v)

    if effective_reads_examined is not None:
        result["effective_reads_examined"] = int(effective_reads_examined)
    elif result["effective_reads_examined"] == 0:
        result["effective_reads_examined"] = result["reads_seen"]

    if effective_reads_kept is not None:
        result["effective_reads_kept"] = int(effective_reads_kept)
    elif result["effective_reads_kept"] == 0:
        result["effective_reads_kept"] = result["reads_kept"]

    return result
