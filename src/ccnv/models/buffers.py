"""In-memory sample and reference branch array containers and Arrow table assembly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from ccnv.models.exceptions import SchemaError
from ccnv.models.metadata import FRAGMENTOMICS_METADATA_KEY, SampleMetadata
from ccnv.models.schemas import (
    NUM_FRAG_BANDS,
    NUM_GC_QUANTILES,
    NUM_PROTOTYPES_PER_BIN,
    REFERENCE_SCHEMA,
    SAMPLE_SCHEMA,
    build_reference_schema,
    fixed_size_list_to_numpy,
)
from ccnv.models.validation import validate_sample_table

__all__ = [
    "BranchFitArrays",
    "SampleBuffers",
    "assemble_reference_table",
]


@dataclass(frozen=True, slots=True)
class SampleBuffers:
    """In-memory sample sketch channels and metadata across pipeline stages."""

    meta: SampleMetadata
    count: NDArray[np.uint32] | NDArray[np.float64]
    paired_count: NDArray[np.uint32]
    frag_len_sum: NDArray[np.uint64]
    short_frag_count: NDArray[np.uint32]
    endpoint_count: NDArray[np.uint32]
    frag_gc_sum: NDArray[np.float32]
    frag_bands: NDArray[np.uint32] | None = None
    kappa_frag: NDArray[np.float32] | None = None

    @property
    def num_rows(self) -> int:
        return int(self.count.shape[0])

    @classmethod
    def from_table(cls, table: pa.Table, *, validate: bool = True) -> SampleBuffers:
        """Extract zero-copy NumPy buffers and typed metadata from a sample Arrow table."""
        meta = (
            validate_sample_table(table)
            if validate
            else SampleMetadata.from_table(table)
        )
        if not validate and table.num_rows != meta.expected_num_bins:
            raise SchemaError(
                f"sample row count {table.num_rows} does not match expected "
                f"{meta.expected_num_bins}"
            )
        count_raw = (
            table.column("count").combine_chunks().to_numpy(zero_copy_only=False)
        )
        count_arr: NDArray[np.uint32] | NDArray[np.float64] = (
            np.asarray(count_raw, dtype=np.uint32)
            if table.schema.field("count").type == pa.uint32()
            else np.asarray(count_raw, dtype=np.float64)
        )
        raw_bands_col = table.column("frag_bands")
        frag_bands = (
            None
            if raw_bands_col.null_count == table.num_rows
            else fixed_size_list_to_numpy(raw_bands_col, NUM_FRAG_BANDS, np.uint32)
        )
        return cls(
            meta=meta,
            count=count_arr,
            paired_count=np.asarray(
                table.column("paired_count")
                .combine_chunks()
                .to_numpy(zero_copy_only=False),
                dtype=np.uint32,
            ),
            frag_len_sum=np.asarray(
                table.column("frag_len_sum")
                .combine_chunks()
                .to_numpy(zero_copy_only=False),
                dtype=np.uint64,
            ),
            short_frag_count=np.asarray(
                table.column("short_frag_count")
                .combine_chunks()
                .to_numpy(zero_copy_only=False),
                dtype=np.uint32,
            ),
            endpoint_count=np.asarray(
                table.column("endpoint_count")
                .combine_chunks()
                .to_numpy(zero_copy_only=False),
                dtype=np.uint32,
            ),
            frag_gc_sum=np.asarray(
                table.column("frag_gc_sum")
                .combine_chunks()
                .to_numpy(zero_copy_only=False),
                dtype=np.float32,
            ),
            frag_bands=frag_bands,
        )

    @classmethod
    def coerce(
        cls, sample: SampleBuffers | pa.Table, *, validate: bool = False
    ) -> SampleBuffers:
        """Return SampleBuffers directly or convert an Arrow sample table once at boundaries."""
        if isinstance(sample, cls):
            if sample.num_rows != sample.meta.expected_num_bins:
                raise SchemaError(
                    f"sample buffer row count {sample.num_rows} does not match "
                    f"expected {sample.meta.expected_num_bins}"
                )
            return sample
        return cls.from_table(sample, validate=validate)

    def to_table(
        self, *, base_metadata: Mapping[bytes, bytes] | None = None
    ) -> pa.Table:
        """Materialize canonical Schema v2.0.0 Arrow sample table."""
        n_bins = self.num_rows
        if np.issubdtype(self.count.dtype, np.floating):
            u32_max = float(np.iinfo(np.uint32).max)
            cleaned = np.nan_to_num(self.count, nan=0.0, posinf=u32_max, neginf=0.0)
            count_u32 = np.rint(np.clip(cleaned, 0.0, u32_max)).astype(np.uint32)
        else:
            count_u32 = np.asarray(self.count, dtype=np.uint32)

        if self.frag_bands is not None:
            bands_col: pa.Array = pa.FixedSizeListArray.from_arrays(
                pa.array(
                    self.frag_bands.astype(np.uint32, copy=False).ravel(),
                    type=pa.uint32(),
                ),
                NUM_FRAG_BANDS,
            )
        else:
            bands_col = pa.nulls(n_bins, type=pa.list_(pa.uint32(), NUM_FRAG_BANDS))

        meta_bytes = dict(base_metadata) if base_metadata is not None else {}
        if self.meta.fragmentomics is None:
            meta_bytes.pop(FRAGMENTOMICS_METADATA_KEY, None)
        meta_bytes.update(self.meta.build_arrow_metadata())
        return pa.table(
            {
                "count": count_u32,
                "paired_count": np.asarray(self.paired_count, dtype=np.uint32),
                "frag_len_sum": np.asarray(self.frag_len_sum, dtype=np.uint64),
                "short_frag_count": np.asarray(self.short_frag_count, dtype=np.uint32),
                "endpoint_count": np.asarray(self.endpoint_count, dtype=np.uint32),
                "frag_gc_sum": np.asarray(self.frag_gc_sum, dtype=np.float32),
                "frag_bands": bands_col,
            },
            schema=SAMPLE_SCHEMA.with_metadata(meta_bytes),
        )


def _sentinel_f32(
    values: NDArray[np.floating] | None, n_bins: int
) -> NDArray[np.float32]:
    if values is None:
        return np.full(n_bins, -1.0, dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32)
    return np.where(np.isfinite(arr), arr, np.float32(-1.0))


@dataclass(frozen=True, slots=True)
class BranchFitArrays:
    """Full-length per-bin arrays for one fitted reference branch."""

    branch: str
    contigs: tuple[str, ...]
    starts: NDArray[np.uint32] | NDArray[np.int64]
    ends: NDArray[np.uint32] | NDArray[np.int64]
    masked: NDArray[np.bool_]
    gc: NDArray[np.floating] | None = None
    mappability: NDArray[np.floating] | None = None
    gc_quantiles: NDArray[np.float32] | None = None
    ref_frag_len: NDArray[np.floating] | None = None
    ref_ep_rate: NDArray[np.floating] | None = None
    pca_mean: NDArray[np.floating] | None = None
    pca_loadings: NDArray[np.floating] | None = None
    proto_ids: NDArray[np.uint16] | None = None
    proto_weights: NDArray[np.float16] | None = None
    mean_distance: NDArray[np.floating] | None = None
    null_mean: NDArray[np.floating] | None = None
    bio_dispersion: NDArray[np.floating] | None = None

    @property
    def num_rows(self) -> int:
        return len(self.contigs)

    @property
    def pca_rank(self) -> int:
        if self.pca_loadings is None:
            return 0
        ld = np.asarray(self.pca_loadings)
        if ld.ndim == 1:
            return 1 if ld.size > 0 else 0
        return int(ld.shape[1]) if ld.size > 0 else 0

    def to_table(
        self,
        *,
        target_pca_rank: int | None = None,
        branch_code: int = 0,
        branch_vocab: pa.Array | None = None,
        contig_vocab: pa.Array | None = None,
        contig_to_code: Mapping[str, int] | None = None,
        metadata: Mapping[bytes, bytes] | None = None,
    ) -> pa.Table:
        """Serialize branch arrays into a 17-column Schema v2.0.0 Arrow table."""
        n_bins = self.num_rows
        own_rank = self.pca_rank
        p_rank = own_rank if target_pca_rank is None else max(target_pca_rank, own_rank)

        b_vocab = (
            branch_vocab
            if branch_vocab is not None
            else pa.array([self.branch], type=pa.string())
        )
        branch_col = pa.DictionaryArray.from_arrays(
            pa.array(np.full(n_bins, branch_code, dtype=np.int8), type=pa.int8()),
            b_vocab,
        )

        max_int16_contigs = int(np.iinfo(np.int16).max) + 1
        if contig_vocab is not None and contig_to_code is not None:
            if len(contig_vocab) > max_int16_contigs:
                raise SchemaError(
                    f"contig count {len(contig_vocab)} exceeds int16 dictionary bounds"
                )
            c_codes = np.array(
                [contig_to_code[c] for c in self.contigs], dtype=np.int16
            )
            c_vocab = contig_vocab
        else:
            uniq_c = list(dict.fromkeys(self.contigs))
            if len(uniq_c) > max_int16_contigs:
                raise SchemaError(
                    f"contig count {len(uniq_c)} exceeds int16 dictionary bounds"
                )
            local_map = {name: idx for idx, name in enumerate(uniq_c)}
            c_codes = np.array([local_map[c] for c in self.contigs], dtype=np.int16)
            c_vocab = pa.array(uniq_c, type=pa.string())
        contig_col = pa.DictionaryArray.from_arrays(
            pa.array(c_codes, type=pa.int16()), c_vocab
        )

        gc_q_col: pa.Array = (
            pa.FixedSizeListArray.from_arrays(
                pa.array(
                    np.asarray(self.gc_quantiles, dtype=np.float32).ravel(),
                    type=pa.float32(),
                ),
                NUM_GC_QUANTILES,
            )
            if self.gc_quantiles is not None
            else pa.nulls(n_bins, type=pa.list_(pa.float32(), NUM_GC_QUANTILES))
        )

        if p_rank == 0:
            loadings_col: pa.Array = pa.nulls(n_bins, type=pa.list_(pa.float32(), 0))
        else:
            padded = np.zeros((n_bins, p_rank), dtype=np.float32)
            if self.pca_loadings is not None and own_rank > 0 and n_bins > 0:
                ld_arr = np.asarray(self.pca_loadings, dtype=np.float32)
                if ld_arr.ndim == 1:
                    ld_arr = ld_arr[:, np.newaxis]
                padded[:, :own_rank] = np.nan_to_num(ld_arr, nan=0.0)
            loadings_col = pa.FixedSizeListArray.from_arrays(
                pa.array(padded.ravel(), type=pa.float32()),
                p_rank,
            )

        pids = (
            np.asarray(self.proto_ids, dtype=np.uint16)
            if self.proto_ids is not None
            else np.zeros((n_bins, NUM_PROTOTYPES_PER_BIN), dtype=np.uint16)
        )
        pweights = (
            np.asarray(self.proto_weights, dtype=np.float16)
            if self.proto_weights is not None
            else np.zeros((n_bins, NUM_PROTOTYPES_PER_BIN), dtype=np.float16)
        )

        def _zeros_or_f32(arr: NDArray[np.floating] | None) -> NDArray[np.float32]:
            return (
                np.asarray(arr, dtype=np.float32)
                if arr is not None
                else np.zeros(n_bins, dtype=np.float32)
            )

        schema = build_reference_schema(p_rank)
        if metadata is not None:
            schema = schema.with_metadata(metadata)

        return pa.table(
            {
                "branch": branch_col,
                "contig": contig_col,
                "start": np.asarray(self.starts, dtype=np.uint32),
                "end": np.asarray(self.ends, dtype=np.uint32),
                "masked": np.asarray(self.masked, dtype=np.bool_),
                "gc": _sentinel_f32(self.gc, n_bins),
                "mappability": _sentinel_f32(self.mappability, n_bins),
                "gc_quantiles": gc_q_col,
                "ref_frag_len": _sentinel_f32(self.ref_frag_len, n_bins),
                "ref_ep_rate": _sentinel_f32(self.ref_ep_rate, n_bins),
                "pca_mean": _zeros_or_f32(self.pca_mean),
                "pca_loadings": loadings_col,
                "proto_ids": pa.FixedSizeListArray.from_arrays(
                    pa.array(pids.ravel(), type=pa.uint16()),
                    NUM_PROTOTYPES_PER_BIN,
                ),
                "proto_weights": pa.FixedSizeListArray.from_arrays(
                    pa.array(pweights.ravel(), type=pa.float16()),
                    NUM_PROTOTYPES_PER_BIN,
                ),
                "mean_distance": _zeros_or_f32(self.mean_distance),
                "null_mean": _zeros_or_f32(self.null_mean),
                "bio_dispersion": _zeros_or_f32(self.bio_dispersion),
            },
            schema=schema,
        )


def assemble_reference_table(
    branches: Sequence[BranchFitArrays],
    *,
    contig_order: Sequence[str] | None = None,
    metadata: Mapping[bytes, bytes] | None = None,
) -> pa.Table:
    """Assemble multiple fitted reference branches into one unified Schema v2.0.0 table."""
    if not branches:
        schema = REFERENCE_SCHEMA
        return (
            schema.with_metadata(metadata).empty_table()
            if metadata is not None
            else schema.empty_table()
        )
    p_max = max(b.pca_rank for b in branches)
    branch_vocab = pa.array([b.branch for b in branches], type=pa.string())

    ordered_contigs: list[str] = (
        list(dict.fromkeys(contig_order)) if contig_order is not None else []
    )
    seen_contigs = set(ordered_contigs)
    for b in branches:
        for c in b.contigs:
            if c not in seen_contigs:
                seen_contigs.add(c)
                ordered_contigs.append(c)
    if len(ordered_contigs) > int(np.iinfo(np.int16).max) + 1:
        raise SchemaError(
            f"contig count {len(ordered_contigs)} exceeds int16 dictionary bounds"
        )
    contig_to_code = {name: idx for idx, name in enumerate(ordered_contigs)}
    contig_vocab = pa.array(ordered_contigs, type=pa.string())

    tables = [
        b.to_table(
            target_pca_rank=p_max,
            branch_code=idx,
            branch_vocab=branch_vocab,
            contig_vocab=contig_vocab,
            contig_to_code=contig_to_code,
        )
        for idx, b in enumerate(branches)
    ]
    combined = pa.concat_tables(tables)
    return (
        combined.replace_schema_metadata(metadata) if metadata is not None else combined
    )
