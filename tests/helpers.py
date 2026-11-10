"""Canonical synthetic data builders for test suites."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa

from ccnv.io.parquet import write_sample_parquet
from ccnv.models import ReadFilters
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
    n_bins = len(counts)
    if starts is None:
        starts_arr = np.arange(n_bins, dtype=np.int64) * bin_size
    else:
        starts_arr = np.asarray(starts, dtype=np.int64)
    if ends is None:
        ends_arr = starts_arr + bin_size
    else:
        ends_arr = np.asarray(ends, dtype=np.int64)
    if contigs is None:
        contig_list = ["chr1"] * n_bins
        if contig_lengths is None:
            lengths_dict = {"chr1": int(ends_arr[-1]) if n_bins > 0 else bin_size}
        else:
            lengths_dict = dict(contig_lengths)
    else:
        contig_list = list(contigs)
        if contig_lengths is None:
            lengths_dict = {}
            for name in dict.fromkeys(contig_list):
                lengths_dict[name] = max(
                    int(end)
                    for c, end in zip(contig_list, ends_arr, strict=True)
                    if c == name
                )
        else:
            lengths_dict = dict(contig_lengths)

    metadata = metadata_for(
        ArtifactKind.SAMPLE,
        contig_lengths=lengths_dict,
        values={
            "bin_size": bin_size,
            "source_identity": (
                identity if identity is not None else {"id": "synthetic_sample"}
            ),
            "reference_identity": {"name": "synthetic_ref"},
            "qc": {},
            "filters": asdict(ReadFilters()),
        },
    )
    if fragment_lengths is None:
        frag_arr = pa.array([[]] * n_bins, type=pa.list_(pa.uint16()))
    else:
        frag_arr = pa.array(fragment_lengths, type=pa.list_(pa.uint16()))

    if endpoint_counts is None:
        ep_arr = pa.array(np.zeros(n_bins, dtype=np.uint32), type=pa.uint32())
    else:
        ep_arr = pa.array(
            np.asarray(endpoint_counts, dtype=np.uint32), type=pa.uint32()
        )

    if frag_gc_sums is None:
        gc_arr = pa.array(np.zeros(n_bins, dtype=np.float32), type=pa.float32())
    else:
        gc_arr = pa.array(np.asarray(frag_gc_sums, dtype=np.float32), type=pa.float32())

    return pa.table(
        {
            "contig": pa.array(contig_list, type=pa.string()),
            "start": pa.array(starts_arr, type=pa.int64()),
            "end": pa.array(ends_arr, type=pa.int64()),
            "count": pa.array(np.asarray(counts, dtype=np.uint64), type=pa.uint64()),
            "fragment_lengths": frag_arr,
            "endpoint_count": ep_arr,
            "frag_gc_sum": gc_arr,
        },
        schema=SAMPLE_SCHEMA.with_metadata(metadata),
    )


def write_synthetic_sample(
    path: Path,
    counts: Sequence[int] | np.ndarray,
    **kwargs: object,
) -> Path:
    table = make_sample_table(counts, **kwargs)
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
