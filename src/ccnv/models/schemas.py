"""Versioned Arrow contracts and column helpers for ccnv Parquet tables."""

from __future__ import annotations

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
    if len(names) > int(np.iinfo(np.int16).max) + 1:
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
    target_dict_type = pa.dictionary(target_idx_type, pa.string())
    if len(set(vocab)) < len(vocab) or encoded.type != target_dict_type:
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
        if (
            target_idx_type == pa.int8()
            and len(unique_vocab) > int(np.iinfo(np.int8).max) + 1
        ):
            raise SchemaError(
                f"column {col_name!r} cardinality {len(unique_vocab)} exceeds int8 dictionary bounds"
            )
        if (
            target_idx_type == pa.int16()
            and len(unique_vocab) > int(np.iinfo(np.int16).max) + 1
        ):
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


__all__ = [
    "BRANCH_AUTOSOMAL",
    "BRANCH_FEMALE_SEX",
    "BRANCH_MALE_SEX",
    "BRANCH_PRENATAL_AUTOSOMAL",
    "BRANCH_PRENATAL_FFY",
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
    "FragmentomicsMetadata",
    "PrenatalModelMetadata",
    "ReferenceMetadata",
    "SampleMetadata",
    "SchemaError",
    "SchemaVersionError",
    "build_reference_schema",
    "canonical_dictionary_array",
    "decode_dictionary_contigs",
    "decode_metadata",
    "encode_metadata",
    "fixed_size_list_to_numpy",
    "reconstruct_sample_grid",
    "unique_dictionary_contigs",
]
