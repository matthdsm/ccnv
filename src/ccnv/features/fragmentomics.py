"""Feature transformation kernels for cell-free DNA fragmentomic signals."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from ccnv.stats.fragmentomics import (
    CANONICAL_CHROMATOSOME_BP,
    FRAG_MAX_LEN,
    FRAG_MIN_LEN,
)

__all__ = [
    "evaluate_weighted_coverage",
    "extract_bin_fragmentomic_covariates",
]


def evaluate_weighted_coverage(
    table: pa.Table,
    weights_571: NDArray[np.floating],
    shift_bp: float = 0.0,
) -> NDArray[np.float64]:
    """Map fragment lengths through a 65,536-entry lookup table in <20 ms.

    Falls back to unweighted table counts when fragment_lengths is absent or empty.
    """
    counts = np.asarray(
        table.column("count").to_numpy(zero_copy_only=False), dtype=np.float64
    )
    if "fragment_lengths" not in table.column_names:
        return counts

    frag_col = table.column("fragment_lengths").combine_chunks()
    if not isinstance(frag_col, pa.ListArray):
        return counts

    values = np.asarray(frag_col.values)
    if values.size == 0:
        return counts

    offsets = np.asarray(frag_col.offsets)
    std_grid = np.arange(FRAG_MIN_LEN, FRAG_MAX_LEN + 1, dtype=np.float64)
    all_l = np.arange(65536, dtype=np.float64)
    w_lut = np.ones(65536, dtype=np.float32)

    valid = (all_l - shift_bp >= FRAG_MIN_LEN) & (all_l - shift_bp <= FRAG_MAX_LEN)
    w_lut[valid] = np.interp(all_l[valid] - shift_bp, std_grid, weights_571).astype(
        np.float32
    )

    cum = np.empty(values.size + 1, dtype=np.float64)
    cum[0] = 0.0
    np.cumsum(w_lut[values], out=cum[1:])
    return cum[offsets[1:]] - cum[offsets[:-1]]


def extract_bin_fragmentomic_covariates(
    table: pa.Table,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute per-bin mean fragment length and cleavage endpoint protection ratio.

    Imputes zero-coverage bins with the sample-wide median fragment length.
    """
    n_rows = table.num_rows
    counts = (
        np.asarray(
            table.column("count").to_numpy(zero_copy_only=False), dtype=np.float64
        )
        if "count" in table.column_names
        else np.zeros(n_rows, dtype=np.float64)
    )

    # 1. Per-bin mean fragment length
    mean_lens = np.full(n_rows, CANONICAL_CHROMATOSOME_BP, dtype=np.float64)
    if "fragment_lengths" in table.column_names:
        frag_col = table.column("fragment_lengths").combine_chunks()
        if isinstance(frag_col, pa.ListArray):
            values = np.asarray(frag_col.values, dtype=np.float64)
            if values.size > 0:
                offsets = np.asarray(frag_col.offsets)
                cum = np.empty(values.size + 1, dtype=np.float64)
                cum[0] = 0.0
                np.cumsum(values, out=cum[1:])
                sum_lens = cum[offsets[1:]] - cum[offsets[:-1]]
                bin_counts = offsets[1:] - offsets[:-1]
                has_reads = bin_counts > 0
                if np.any(has_reads):
                    computed = sum_lens[has_reads] / bin_counts[has_reads]
                    mean_lens[has_reads] = computed
                    sample_median = float(np.median(computed))
                    mean_lens[~has_reads] = sample_median

    # 2. Cleavage endpoint ratio: E_i / max(C_i, 1)
    if "endpoint_count" in table.column_names:
        ep_counts = np.asarray(
            table.column("endpoint_count").to_numpy(zero_copy_only=False),
            dtype=np.float64,
        )
        ep_ratios = ep_counts / np.maximum(counts, 1.0)
    else:
        ep_ratios = np.zeros(n_rows, dtype=np.float64)

    return mean_lens, ep_ratios
