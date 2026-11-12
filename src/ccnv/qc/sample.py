"""Sample-level read counting, quality control summary assembly, and atomic JSON writing."""

from __future__ import annotations

import json
import math
import os
import tempfile
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
    "write_qc_json",
]


def sanitize_qc_payload(value: object) -> object:
    """Recursively convert NumPy scalars and non-finite floats to JSON-safe types."""
    if isinstance(value, Mapping):
        return {str(k): sanitize_qc_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_qc_payload(v) for v in value]
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else None
    if isinstance(value, np.floating):
        f = float(value)
        return round(f, 6) if math.isfinite(f) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_qc_json(path: Path | str, payload: Mapping[str, object]) -> Path:
    """Atomically write a sanitized QC mapping to a JSON file via sibling temp replacement."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sanitized = sanitize_qc_payload(dict(payload))
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            json.dump(sanitized, handle, indent=2)
            handle.write("\n")
        os.replace(tmp_path, target)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise
    return target


def empty_qc() -> dict[str, int]:
    """Return a complete QC counter map with deterministic key order."""
    return {name: 0 for name in QC_COUNTERS}


def build_sample_qc(
    counters: Mapping[str, int] | None = None,
    *,
    reads_seen: int | None = None,
    reads_kept: int | None = None,
    effective_reads_examined: int | None = None,
    effective_reads_kept: int | None = None,
) -> dict[str, int]:
    """Assemble deterministic sample QC counter dictionary with effective reads."""
    result = empty_qc()
    has_eff_examined = False
    has_eff_kept = False
    if counters is not None:
        has_eff_examined = (
            "effective_reads_examined" in counters
            and counters["effective_reads_examined"] is not None
        )
        has_eff_kept = (
            "effective_reads_kept" in counters
            and counters["effective_reads_kept"] is not None
        )
        for k, v in counters.items():
            if k in result and v is not None:
                result[k] = int(v)

    if reads_seen is not None:
        result["reads_seen"] = int(reads_seen)
    if reads_kept is not None:
        result["reads_kept"] = int(reads_kept)

    if effective_reads_examined is not None:
        result["effective_reads_examined"] = int(effective_reads_examined)
    elif not has_eff_examined:
        result["effective_reads_examined"] = result["reads_seen"]

    if effective_reads_kept is not None:
        result["effective_reads_kept"] = int(effective_reads_kept)
    elif not has_eff_kept:
        result["effective_reads_kept"] = result["reads_kept"]

    return result
