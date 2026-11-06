"""Transitional re-export stub for ccnv.segment.

Deprecated in favor of ccnv.stats.segmentation.
"""

from __future__ import annotations

from ccnv.stats.segmentation import (
    DEFAULT_MAX_BLACKLIST_GAP,
    Algorithm,
    Segment,
    WeightedL2Cost,
    segment_contig,
    segment_genome,
)

__all__ = [
    "DEFAULT_MAX_BLACKLIST_GAP",
    "Algorithm",
    "Segment",
    "WeightedL2Cost",
    "segment_contig",
    "segment_genome",
]
