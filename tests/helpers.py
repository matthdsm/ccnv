"""Canonical synthetic data builders for test suites."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa

from ccnv._core import FRAG_BAND_EDGES
from ccnv.io.parquet import write_reference_parquet, write_sample_parquet
from ccnv.models import (
    COORDINATE_SYSTEM,
    NUM_FRAG_BANDS,
    NUM_GC_QUANTILES,
    NUM_PROTOTYPES_PER_BIN,
    SAMPLE_SCHEMA,
    SCHEMA_VERSION,
    ArtifactKind,
    ReadFilters,
    build_reference_schema,
    encode_metadata,
    validate_metadata,
)


def metadata_for(
    kind: ArtifactKind,
    *,
    contig_lengths: Mapping[str, int],
    values: Mapping[str, object],
) -> dict[bytes, bytes]:
    """Build complete metadata for a test artifact, preserving sample contig order."""
    extra: dict[str, object] = {}
    if kind is ArtifactKind.SAMPLE and "contig_order" not in values:
        extra["contig_order"] = list(contig_lengths.keys())
    metadata: dict[str, object] = {
        "artifact_kind": kind.value,
        "schema_version": SCHEMA_VERSION,
        "coordinate_system": COORDINATE_SYSTEM,
        "contig_lengths": dict(contig_lengths),
        **extra,
        **values,
    }
    validate_metadata(metadata, kind)
    return encode_metadata(metadata)


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


def make_reference_table(
    *,
    branches: Sequence[str],
    contigs: Sequence[str],
    starts: Sequence[int] | np.ndarray,
    ends: Sequence[int] | np.ndarray,
    contig_lengths: Mapping[str, int],
    masked: Sequence[bool] | np.ndarray | None = None,
    gc: Sequence[float] | np.ndarray | None = None,
    mappability: Sequence[float] | np.ndarray | None = None,
    gc_quantiles: np.ndarray | None = None,
    ref_frag_len: Sequence[float] | np.ndarray | None = None,
    ref_ep_rate: Sequence[float] | np.ndarray | None = None,
    pca_mean: Sequence[float] | np.ndarray | None = None,
    pca_loadings: Sequence[Sequence[float]] | np.ndarray | None = None,
    proto_ids: Sequence[Sequence[int]] | np.ndarray | None = None,
    proto_weights: Sequence[Sequence[float]] | np.ndarray | None = None,
    mean_distance: Sequence[float] | np.ndarray | None = None,
    null_mean: Sequence[float] | np.ndarray | None = None,
    bio_dispersion: Sequence[float] | np.ndarray | None = None,
    null_std: Sequence[float] | np.ndarray | None = None,
    metadata: dict[bytes, bytes] | None = None,
    model: Mapping[str, object] | None = None,
    sample_identities: Sequence[object] = (),
    extra_metadata: Mapping[str, object] | None = None,
) -> pa.Table:
    """Build a Schema v2.0.0 reference table from per-bin arrays."""
    n_rows = len(branches)
    b_list = [str(b) for b in branches]
    c_list = [str(c) for c in contigs]
    b_vocab = list(dict.fromkeys(b_list)) if b_list else ["autosomal"]
    c_vocab = list(dict.fromkeys(c_list)) if c_list else list(contig_lengths.keys())
    b_idx = {name: i for i, name in enumerate(b_vocab)}
    c_idx = {name: i for i, name in enumerate(c_vocab)}
    branch_col = pa.DictionaryArray.from_arrays(
        pa.array([b_idx[b] for b in b_list], type=pa.int8()),
        pa.array(b_vocab, type=pa.string()),
    )
    contig_col = pa.DictionaryArray.from_arrays(
        pa.array([c_idx[c] for c in c_list], type=pa.int16()),
        pa.array(c_vocab, type=pa.string()),
    )

    if pca_loadings is None:
        pca_rank = 0
        pca_col: pa.Array = pa.nulls(n_rows, type=pa.list_(pa.float32(), 0))
    else:
        rows_list = [list(r) for r in pca_loadings]
        pca_rank = max((len(r) for r in rows_list), default=0)
        if pca_rank == 0:
            pca_col = pa.nulls(n_rows, type=pa.list_(pa.float32(), 0))
        else:
            ld_arr = np.zeros((n_rows, pca_rank), dtype=np.float32)
            for r_i, row_vals in enumerate(rows_list):
                if row_vals:
                    ld_arr[r_i, : len(row_vals)] = row_vals
            pca_col = pa.FixedSizeListArray.from_arrays(
                pa.array(ld_arr.ravel(), type=pa.float32()), pca_rank
            )

    if gc_quantiles is None:
        gc_q_col: pa.Array = pa.nulls(
            n_rows, type=pa.list_(pa.float32(), NUM_GC_QUANTILES)
        )
    else:
        q_arr = np.asarray(gc_quantiles, dtype=np.float32).reshape(
            n_rows, NUM_GC_QUANTILES
        )
        gc_q_col = pa.FixedSizeListArray.from_arrays(
            pa.array(q_arr.ravel(), type=pa.float32()), NUM_GC_QUANTILES
        )

    if proto_ids is None:
        pids_arr = np.zeros((n_rows, NUM_PROTOTYPES_PER_BIN), dtype=np.uint16)
    else:
        pids_arr = np.zeros((n_rows, NUM_PROTOTYPES_PER_BIN), dtype=np.uint16)
        for r_i, p_row in enumerate(proto_ids):
            vals = list(p_row)[:NUM_PROTOTYPES_PER_BIN]
            if vals:
                pids_arr[r_i, : len(vals)] = vals
    pids_col = pa.FixedSizeListArray.from_arrays(
        pa.array(pids_arr.ravel(), type=pa.uint16()), NUM_PROTOTYPES_PER_BIN
    )

    if proto_weights is None:
        pw_arr = np.zeros((n_rows, NUM_PROTOTYPES_PER_BIN), dtype=np.float16)
        if n_rows > 0:
            pw_arr[:, 0] = np.float16(1.0)
    else:
        pw_arr = np.zeros((n_rows, NUM_PROTOTYPES_PER_BIN), dtype=np.float16)
        for r_i, w_row in enumerate(proto_weights):
            vals = list(w_row)[:NUM_PROTOTYPES_PER_BIN]
            if vals:
                pw_arr[r_i, : len(vals)] = vals
    pw_col = pa.FixedSizeListArray.from_arrays(
        pa.array(pw_arr.ravel(), type=pa.float16()), NUM_PROTOTYPES_PER_BIN
    )

    if bio_dispersion is not None:
        disp_arr = np.asarray(bio_dispersion, dtype=np.float32)
    elif null_std is not None:
        disp_arr = np.square(np.asarray(null_std, dtype=np.float64)).astype(np.float32)
    else:
        disp_arr = np.full(n_rows, 0.01, dtype=np.float32)

    if metadata is None:
        values: dict[str, object] = {
            "sample_identities": list(sample_identities),
            "model": dict(model) if model is not None else {"branches": b_vocab},
        }
        if extra_metadata is not None:
            values.update(extra_metadata)
        metadata = metadata_for(
            ArtifactKind.REFERENCE,
            contig_lengths={str(k): int(v) for k, v in contig_lengths.items()},
            values=values,
        )

    schema = build_reference_schema(pca_rank).with_metadata(metadata)
    return pa.table(
        {
            "branch": branch_col,
            "contig": contig_col,
            "start": np.asarray(starts, dtype=np.uint32),
            "end": np.asarray(ends, dtype=np.uint32),
            "masked": (
                np.zeros(n_rows, dtype=np.bool_)
                if masked is None
                else np.asarray(masked, dtype=np.bool_)
            ),
            "gc": (
                np.full(n_rows, -1.0, dtype=np.float32)
                if gc is None
                else np.asarray(gc, dtype=np.float32)
            ),
            "mappability": (
                np.full(n_rows, -1.0, dtype=np.float32)
                if mappability is None
                else np.asarray(mappability, dtype=np.float32)
            ),
            "gc_quantiles": gc_q_col,
            "ref_frag_len": (
                np.full(n_rows, -1.0, dtype=np.float32)
                if ref_frag_len is None
                else np.asarray(ref_frag_len, dtype=np.float32)
            ),
            "ref_ep_rate": (
                np.full(n_rows, -1.0, dtype=np.float32)
                if ref_ep_rate is None
                else np.asarray(ref_ep_rate, dtype=np.float32)
            ),
            "pca_mean": (
                np.ones(n_rows, dtype=np.float32)
                if pca_mean is None
                else np.asarray(pca_mean, dtype=np.float32)
            ),
            "pca_loadings": pca_col,
            "proto_ids": pids_col,
            "proto_weights": pw_col,
            "mean_distance": (
                np.full(n_rows, 0.5, dtype=np.float32)
                if mean_distance is None
                else np.asarray(mean_distance, dtype=np.float32)
            ),
            "null_mean": (
                np.zeros(n_rows, dtype=np.float32)
                if null_mean is None
                else np.asarray(null_mean, dtype=np.float32)
            ),
            "bio_dispersion": disp_arr,
        },
        schema=schema,
    )


def write_synthetic_reference(path: Path, table: pa.Table) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_reference_parquet(table, path)
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
