"""Canonical synthetic data builders for test suites."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa

from ccnv._core import FRAG_BAND_EDGES
from ccnv.io.parquet import write_sample_parquet
from ccnv.models import NUM_FRAG_BANDS, ReadFilters
from ccnv.models.schemas import (
    SAMPLE_SCHEMA,
    ArtifactKind,
    metadata_for,
)


def make_sample_table(
    counts: Sequence[int] | np.ndarray,
    *,
    contigs: Sequence[str] | None = None,
    starts: Sequence[int] | np.ndarray | None = None,
    ends: Sequence[int] | np.ndarray | None = None,
    contig_lengths: Mapping[str, int] | None = None,
    bin_size: int = 100,
    identity: object | None = None,
    fragment_lengths: Sequence[Sequence[int]] | None = None,
    endpoint_counts: Sequence[int] | np.ndarray | None = None,
    frag_gc_sums: Sequence[float] | np.ndarray | None = None,
) -> pa.Table:
    """Build a Schema v2.0.0 sample sketch table from per-bin arrays."""
    n_bins = len(counts)
    starts_arr = (
        np.arange(n_bins, dtype=np.int64) * bin_size
        if starts is None
        else np.asarray(starts, dtype=np.int64)
    )
    ends_arr = (
        starts_arr + bin_size if ends is None else np.asarray(ends, dtype=np.int64)
    )
    if contigs is None:
        contig_order = ["chr1"]
        lengths_dict = (
            {"chr1": int(ends_arr[-1]) if n_bins > 0 else bin_size}
            if contig_lengths is None
            else dict(contig_lengths)
        )
    else:
        contig_list = list(contigs)
        contig_order = list(dict.fromkeys(contig_list))
        if contig_lengths is None:
            lengths_dict = {
                name: max(
                    int(end)
                    for c, end in zip(contig_list, ends_arr, strict=True)
                    if c == name
                )
                for name in contig_order
            }
        else:
            lengths_dict = dict(contig_lengths)

    metadata = metadata_for(
        ArtifactKind.SAMPLE,
        contig_lengths=lengths_dict,
        values={
            "bin_size": bin_size,
            "contig_order": contig_order,
            "source_identity": (
                identity if identity is not None else {"id": "synthetic_sample"}
            ),
            "reference_identity": {
                "name": "synthetic_ref",
                "n_contigs": len(contig_order),
                "contig_order": contig_order,
            },
            "qc": {},
            "filters": asdict(ReadFilters()),
        },
    )

    paired_arr = np.zeros(n_bins, dtype=np.uint32)
    flen_sum_arr = np.zeros(n_bins, dtype=np.uint64)
    short_arr = np.zeros(n_bins, dtype=np.uint32)
    if fragment_lengths is None:
        bands_col: pa.Array = pa.nulls(
            n_bins, type=pa.list_(pa.uint32(), NUM_FRAG_BANDS)
        )
    else:
        bands_2d = np.zeros((n_bins, NUM_FRAG_BANDS), dtype=np.uint32)
        edges = np.asarray(FRAG_BAND_EDGES, dtype=np.int64)
        for idx, bin_flens in enumerate(fragment_lengths):
            arr = np.asarray(bin_flens, dtype=np.int64)
            arr_in = arr[(arr >= 30) & (arr <= 600)]
            if arr_in.size > 0:
                paired_arr[idx] = np.uint32(arr_in.size)
                flen_sum_arr[idx] = np.uint64(np.sum(arr_in))
                short_arr[idx] = np.uint32(
                    np.count_nonzero((arr_in >= 80) & (arr_in <= 150))
                )
                hist, _ = np.histogram(arr_in, bins=edges)
                bands_2d[idx] = hist.astype(np.uint32)
        bands_col = pa.FixedSizeListArray.from_arrays(
            pa.array(bands_2d.ravel(), type=pa.uint32()),
            NUM_FRAG_BANDS,
        )

    ep_arr = (
        np.zeros(n_bins, dtype=np.uint32)
        if endpoint_counts is None
        else np.asarray(endpoint_counts, dtype=np.uint32)
    )
    gc_arr = (
        np.zeros(n_bins, dtype=np.float32)
        if frag_gc_sums is None
        else np.asarray(frag_gc_sums, dtype=np.float32)
    )

    return pa.table(
        {
            "count": np.asarray(counts, dtype=np.uint32),
            "paired_count": paired_arr,
            "frag_len_sum": flen_sum_arr,
            "short_frag_count": short_arr,
            "endpoint_count": ep_arr,
            "frag_gc_sum": gc_arr,
            "frag_bands": bands_col,
        },
        schema=SAMPLE_SCHEMA.with_metadata(metadata),
    )


def write_synthetic_sample(
    path: Path,
    counts: Sequence[int] | np.ndarray,
    *,
    contigs: Sequence[str] | None = None,
    starts: Sequence[int] | np.ndarray | None = None,
    ends: Sequence[int] | np.ndarray | None = None,
    contig_lengths: Mapping[str, int] | None = None,
    bin_size: int = 100,
    identity: object | None = None,
    fragment_lengths: Sequence[Sequence[int]] | None = None,
    endpoint_counts: Sequence[int] | np.ndarray | None = None,
    frag_gc_sums: Sequence[float] | np.ndarray | None = None,
) -> Path:
    table = make_sample_table(
        counts,
        contigs=contigs,
        starts=starts,
        ends=ends,
        contig_lengths=contig_lengths,
        bin_size=bin_size,
        identity=identity,
        fragment_lengths=fragment_lengths,
        endpoint_counts=endpoint_counts,
        frag_gc_sums=frag_gc_sums,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_sample_parquet(table, path)
    return path


def write_tiny_fasta(path: Path, contigs: Mapping[str, str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for name, seq in contigs.items():
        lines.append(f">{name}")
        for i in range(0, len(seq), 60):
            lines.append(seq[i : i + 60])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
