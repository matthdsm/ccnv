"""Sample-level read counting and quality control summary assembly."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

QC_COUNTERS: Final[tuple[str, ...]] = (
    "header_mapped",
    "header_unmapped",
    "header_no_coordinate",
    "reads_seen",
    "reads_kept",
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
]


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
