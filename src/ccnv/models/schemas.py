"""Versioned Arrow contracts and artifact metadata for ccnv Parquet tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from numpy.typing import NDArray

from ccnv.models.exceptions import SchemaError, SchemaVersionError
from ccnv.models.metadata import (
    COORDINATE_SYSTEM,
    FRAGMENTOMICS_METADATA_KEY,
    METADATA_PREFIX,
    PRENATAL_METADATA_KEY,
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    ArtifactKind,
    FragmentomicsMetadata,
    PrenatalModelMetadata,
    ReferenceMetadata,
    SampleMetadata,
    decode_metadata,
    encode_metadata,
)

NUM_FRAG_BANDS: Final[int] = 24
NUM_PROTOTYPES_PER_BIN: Final[int] = 4
NUM_GC_QUANTILES: Final[int] = 8

BRANCH_AUTOSOMAL: Final[str] = "autosomal"
BRANCH_FEMALE_SEX: Final[str] = "female_sex"
BRANCH_MALE_SEX: Final[str] = "male_sex"
BRANCH_PRENATAL_AUTOSOMAL: Final[str] = "prenatal_autosomal"
BRANCH_PRENATAL_FFY: Final[str] = "prenatal_ffy"

KNOWN_REFERENCE_BRANCHES: Final[frozenset[str]] = frozenset(
    {
        BRANCH_AUTOSOMAL,
        BRANCH_FEMALE_SEX,
        BRANCH_MALE_SEX,
        BRANCH_PRENATAL_AUTOSOMAL,
        BRANCH_PRENATAL_FFY,
    }
)
CNV_REFERENCE_BRANCHES: Final[frozenset[str]] = frozenset(
    {BRANCH_AUTOSOMAL, BRANCH_FEMALE_SEX, BRANCH_MALE_SEX}
)

SAMPLE_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field("count", pa.uint32(), nullable=False),
        pa.field("paired_count", pa.uint32(), nullable=False),
        pa.field("frag_len_sum", pa.uint64(), nullable=False),
        pa.field("short_frag_count", pa.uint32(), nullable=False),
        pa.field("endpoint_count", pa.uint32(), nullable=False),
        pa.field("frag_gc_sum", pa.float32(), nullable=False),
        pa.field("frag_bands", pa.list_(pa.uint32(), NUM_FRAG_BANDS), nullable=True),
    ]
)


def build_reference_schema(pca_rank: int) -> pa.Schema:
    """Build Schema v2.0.0 reference Arrow schema parameterized by PCA rank."""
    if not isinstance(pca_rank, int) or isinstance(pca_rank, bool) or pca_rank < 0:
        raise ValueError(f"pca_rank must be a non-negative integer, got {pca_rank!r}")
    return pa.schema(
        [
            pa.field("branch", pa.dictionary(pa.int8(), pa.string()), nullable=False),
            pa.field("contig", pa.dictionary(pa.int16(), pa.string()), nullable=False),
            pa.field("start", pa.uint32(), nullable=False),
            pa.field("end", pa.uint32(), nullable=False),
            pa.field("masked", pa.bool_(), nullable=False),
            pa.field("gc", pa.float32(), nullable=False),
            pa.field("mappability", pa.float32(), nullable=False),
            pa.field(
                "gc_quantiles", pa.list_(pa.float32(), NUM_GC_QUANTILES), nullable=True
            ),
            pa.field("ref_frag_len", pa.float32(), nullable=False),
            pa.field("ref_ep_rate", pa.float32(), nullable=False),
            pa.field("pca_mean", pa.float32(), nullable=False),
            pa.field(
                "pca_loadings",
                pa.list_(pa.float32(), pca_rank),
                nullable=(pca_rank == 0),
            ),
            pa.field(
                "proto_ids",
                pa.list_(pa.uint16(), NUM_PROTOTYPES_PER_BIN),
                nullable=False,
            ),
            pa.field(
                "proto_weights",
                pa.list_(pa.float16(), NUM_PROTOTYPES_PER_BIN),
                nullable=False,
            ),
            pa.field("mean_distance", pa.float32(), nullable=False),
            pa.field("null_mean", pa.float32(), nullable=False),
            pa.field("bio_dispersion", pa.float32(), nullable=False),
        ]
    )


REFERENCE_SCHEMA: Final[pa.Schema] = build_reference_schema(0)


def reconstruct_sample_grid(
    meta: SampleMetadata,
) -> tuple[NDArray[np.int16], tuple[str, ...], NDArray[np.uint32], NDArray[np.uint32]]:
    """Reconstruct implicit (contig_codes, contig_names, starts, ends) bin grid."""
    names = meta.contig_order
    if len(names) > np.iinfo(np.int16).max:
        raise SchemaError(f"contig count {len(names)} exceeds int16 dictionary bounds")
    bin_size_u64 = np.uint64(meta.bin_size)
    max_u32 = int(np.iinfo(np.uint32).max)

    counts_per_contig = np.empty(len(names), dtype=np.int64)
    starts_parts: list[NDArray[np.uint32]] = []
    ends_parts: list[NDArray[np.uint32]] = []
    for idx, name in enumerate(names):
        length = meta.contig_lengths[name]
        if length > max_u32:
            raise SchemaError(f"contig {name!r} length {length} exceeds uint32 bounds")
        n_bins = (length + meta.bin_size - 1) // meta.bin_size
        counts_per_contig[idx] = n_bins
        s64 = np.arange(n_bins, dtype=np.uint64) * bin_size_u64
        e64 = np.minimum(s64 + bin_size_u64, np.uint64(length))
        starts_parts.append(s64.astype(np.uint32))
        ends_parts.append(e64.astype(np.uint32))

    codes = np.repeat(np.arange(len(names), dtype=np.int16), counts_per_contig)
    starts = (
        np.concatenate(starts_parts) if starts_parts else np.empty(0, dtype=np.uint32)
    )
    ends = np.concatenate(ends_parts) if ends_parts else np.empty(0, dtype=np.uint32)
    return codes, names, starts, ends


def canonical_dictionary_array(
    col: pa.Array | pa.ChunkedArray,
    col_name: str,
    *,
    index_type: pa.DataType | None = None,
) -> pa.DictionaryArray:
    """Encode, unify, and deduplicate a string or dictionary column into one DictionaryArray."""
    if col.null_count > 0:
        raise SchemaError(f"column {col_name!r} contains null entries")
    try:
        encoded = pc.dictionary_encode(col)
        if isinstance(encoded, pa.ChunkedArray):
            encoded = encoded.unify_dictionaries().combine_chunks()
    except (pa.ArrowInvalid, pa.ArrowTypeError) as err:
        raise SchemaError(f"column {col_name!r} dictionary is invalid: {err}") from err
    if not isinstance(encoded, pa.DictionaryArray):
        raise SchemaError(
            f"column {col_name!r} expected DictionaryArray, found {type(encoded).__name__}"
        )
    if encoded.dictionary.null_count > 0:
        raise SchemaError(f"column {col_name!r} dictionary contains null entries")

    vocab = [str(v) for v in encoded.dictionary.to_pylist()]
    target_idx_type = index_type if index_type is not None else encoded.type.index_type
    if len(set(vocab)) < len(vocab) or encoded.type.index_type != target_idx_type:
        unique_vocab: list[str] = []
        seen: dict[str, int] = {}
        remap = np.empty(len(vocab), dtype=np.int64)
        for idx, name in enumerate(vocab):
            canon = seen.get(name)
            if canon is None:
                canon = len(unique_vocab)
                seen[name] = canon
                unique_vocab.append(name)
            remap[idx] = canon
        if target_idx_type == pa.int8() and len(unique_vocab) > np.iinfo(np.int8).max:
            raise SchemaError(
                f"column {col_name!r} cardinality {len(unique_vocab)} exceeds int8 dictionary bounds"
            )
        if target_idx_type == pa.int16() and len(unique_vocab) > np.iinfo(np.int16).max:
            raise SchemaError(
                f"column {col_name!r} cardinality {len(unique_vocab)} exceeds int16 dictionary bounds"
            )
        raw_codes = encoded.indices.to_numpy(zero_copy_only=False)
        mapped_codes = (
            remap[raw_codes] if raw_codes.size > 0 else np.empty(0, dtype=np.int64)
        )
        return pa.DictionaryArray.from_arrays(
            pa.array(mapped_codes, type=target_idx_type),
            pa.array(unique_vocab, type=pa.string()),
        )
    return encoded


def decode_dictionary_contigs(
    contig_col: pa.Array | pa.ChunkedArray,
) -> tuple[str, ...]:
    """Decode a dictionary-encoded or string contig column into a tuple of names."""
    c_dict = canonical_dictionary_array(contig_col, "contig")
    vocab = tuple(str(x) for x in c_dict.dictionary.to_pylist())
    codes = c_dict.indices.to_numpy(zero_copy_only=False)
    return tuple(vocab[int(c)] for c in codes)


def unique_dictionary_contigs(contig_col: pa.Array | pa.ChunkedArray) -> list[str]:
    """Return unique contig names present in a contig column in code order."""
    c_dict = canonical_dictionary_array(contig_col, "contig")
    vocab = [str(x) for x in c_dict.dictionary.to_pylist()]
    codes = c_dict.indices.to_numpy(zero_copy_only=False)
    return [vocab[int(c)] for c in np.unique(codes)]


def fixed_size_list_to_numpy[ScalarT: np.generic](
    col: pa.Array | pa.ChunkedArray,
    list_size: int,
    dtype: type[ScalarT],
) -> NDArray[ScalarT]:
    """Extract a FixedSizeListArray or ChunkedArray into a 2D (N, list_size) NumPy array."""
    col_type = col.type
    if not pa.types.is_fixed_size_list(col_type) or not isinstance(
        col_type, pa.FixedSizeListType
    ):
        raise SchemaError(f"expected FixedSizeListType, found {col_type}")
    if col_type.list_size != list_size:
        raise SchemaError(
            f"expected FixedSizeList[{list_size}], found FixedSizeList[{col_type.list_size}]"
        )
    arr = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    n = len(arr)
    np_dtype = np.dtype(dtype)
    if n == 0 or list_size == 0:
        return np.empty((n, list_size), dtype=np_dtype)
    is_float = bool(np.issubdtype(np_dtype, np.floating))
    if arr.null_count == n:
        return np.full((n, list_size), np.nan if is_float else 0, dtype=np.dtype(dtype))
    if not isinstance(arr, pa.FixedSizeListArray):
        raise SchemaError(f"expected FixedSizeListArray, found {type(arr).__name__}")
    child = arr.values.slice(arr.offset * list_size, n * list_size)
    if child.null_count > 0 and not is_float:
        child = pc.fill_null(child, 0)
    flat = child.to_numpy(zero_copy_only=False)
    out: NDArray[ScalarT] = np.asarray(flat, dtype=np_dtype).reshape(n, list_size)
    if arr.null_count > 0:
        out = out.copy()
        null_mask = arr.is_null().to_numpy(zero_copy_only=False)
        out[null_mask] = np.nan if is_float else 0
    return out


from ccnv.models.validation import (
    metadata_for,
    validate_metadata,
    validate_reference_table,
    validate_sample_table,
    validate_schema,
    validate_table,
)


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
                "count": np.asarray(self.count, dtype=np.uint32),
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

        if contig_vocab is not None and contig_to_code is not None:
            c_codes = np.array(
                [contig_to_code[c] for c in self.contigs], dtype=np.int16
            )
            c_vocab = contig_vocab
        else:
            uniq_c = list(dict.fromkeys(self.contigs))
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


__all__ = [
    "BRANCH_AUTOSOMAL",
    "BRANCH_FEMALE_SEX",
    "BRANCH_MALE_SEX",
    "BRANCH_PRENATAL_AUTOSOMAL",
    "BRANCH_PRENATAL_FFY",
    "CNV_REFERENCE_BRANCHES",
    "COORDINATE_SYSTEM",
    "FRAGMENTOMICS_METADATA_KEY",
    "KNOWN_REFERENCE_BRANCHES",
    "METADATA_PREFIX",
    "NUM_FRAG_BANDS",
    "NUM_GC_QUANTILES",
    "NUM_PROTOTYPES_PER_BIN",
    "PRENATAL_METADATA_KEY",
    "REFERENCE_SCHEMA",
    "SAMPLE_SCHEMA",
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "ArtifactKind",
    "BranchFitArrays",
    "FragmentomicsMetadata",
    "PrenatalModelMetadata",
    "ReferenceMetadata",
    "SampleBuffers",
    "SampleMetadata",
    "SchemaError",
    "SchemaVersionError",
    "assemble_reference_table",
    "build_reference_schema",
    "canonical_dictionary_array",
    "decode_dictionary_contigs",
    "decode_metadata",
    "encode_metadata",
    "fixed_size_list_to_numpy",
    "metadata_for",
    "reconstruct_sample_grid",
    "unique_dictionary_contigs",
    "validate_metadata",
    "validate_reference_table",
    "validate_sample_table",
    "validate_schema",
    "validate_table",
]
