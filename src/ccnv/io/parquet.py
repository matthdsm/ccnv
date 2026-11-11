"""Arrow Parquet table reading, writing, and contract validation for ccnv."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from numpy.typing import NDArray

from ccnv.models import (
    BRANCH_PRENATAL_AUTOSOMAL,
    NUM_GC_QUANTILES,
    NUM_PROTOTYPES_PER_BIN,
    ArtifactKind,
    PredictError,
    ReferenceError,
    ReferenceMetadata,
    SampleBuffers,
    SampleMetadata,
    SchemaError,
    canonical_dictionary_array,
    decode_metadata,
    fixed_size_list_to_numpy,
    validate_reference_table,
    validate_sample_table,
    validate_schema,
)

__all__ = [
    "ReferenceBranchArrays",
    "align_sample_grid_indices",
    "decode_dictionary_contigs",
    "extract_reference_branch_arrays",
    "extract_sample_meta_and_counts",
    "is_sample_parquet",
    "load_cohort",
    "load_sample_or_extract",
    "read_reference_parquet",
    "read_sample_parquet",
    "read_sample_tables",
    "unique_dictionary_contigs",
    "write_reference_parquet",
    "write_sample_parquet",
]


@dataclass(frozen=True, slots=True)
class ReferenceBranchArrays:
    """Zero-copy extracted NumPy arrays for one reference branch."""

    contigs: tuple[str, ...]
    starts: NDArray[np.int64]
    ends: NDArray[np.int64]
    masked: NDArray[np.bool_]
    gc: NDArray[np.float32] | None
    mappability: NDArray[np.float32] | None
    pca_mean: NDArray[np.float64]
    pca_loadings: NDArray[np.float32]
    proto_ids: NDArray[np.uint16]
    proto_weights: NDArray[np.float16]
    mean_distance: NDArray[np.float32]
    null_mean: NDArray[np.float64]
    null_std: NDArray[np.float64]
    gc_quantiles: NDArray[np.float32] | None = None
    ref_frag_len: NDArray[np.float32] | None = None
    ref_ep_rate: NDArray[np.float32] | None = None
    bio_dispersion: NDArray[np.float64] | None = None
    bin_size: int = 5000
    is_flat: bool = False


def is_sample_parquet(path: Path | str) -> bool:
    """Return True if path is a readable Parquet file with sample artifact metadata."""
    try:
        p = Path(path)
        metadata = decode_metadata(pq.read_schema(p).metadata)
    except OSError, TypeError, ValueError, SchemaError, pa.ArrowException:
        return False
    return metadata.get("artifact_kind") == ArtifactKind.SAMPLE.value


def read_sample_parquet(path: Path | str) -> pa.Table:
    """Read a sample Parquet table from disk and validate its schema contract."""
    p = Path(path)
    table = pq.read_table(p)
    validate_sample_table(table)
    return table


def load_sample_or_extract(
    path: Path | str,
    *,
    fasta: Path | str | None = None,
    bin_size: int = 5000,
    threads: int = 1,
) -> pa.Table:
    """Load a sample Parquet table from disk or extract directly in memory from BAM/CRAM."""
    p = Path(path)
    if p.suffix.lower() in {".bam", ".cram"}:
        from ccnv.extract import extract_sample_table
        from ccnv.models import ExtractConfig

        fasta_path = Path(fasta) if fasta is not None else None
        config = ExtractConfig(
            alignment=p,
            prefix=p.with_suffix(""),
            fasta=fasta_path,
            bin_size=bin_size,
            threads=threads,
        )
        return extract_sample_table(config)
    return read_sample_parquet(p)


def write_sample_parquet(
    table: pa.Table,
    path: Path | str,
    *,
    compression_level: int = 3,
) -> Path:
    """Validate sample table contract and persist as a zstd-compressed Parquet table."""
    validate_sample_table(table)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        out_path,
        compression="zstd",
        compression_level=compression_level,
    )
    return out_path


def _decode_stat_str(val: object) -> str | None:
    if isinstance(val, str):
        return val
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8")
        except UnicodeDecodeError:
            return None
    return None


def _select_matching_row_groups(
    pf: pq.ParquetFile,
    target_branches: set[str],
) -> tuple[list[int], bool]:
    """Identify row group indices matching target_branches via metadata or branch column."""
    matching: list[int] = []
    needs_post_filter = False
    branch_col_idx = pf.schema_arrow.get_field_index("branch")

    for rg_idx in range(pf.num_row_groups):
        rg_meta = pf.metadata.row_group(rg_idx)
        if rg_meta.num_rows == 0:
            continue
        col_meta = rg_meta.column(branch_col_idx)
        stats = col_meta.statistics if col_meta.is_stats_set else None
        if stats is not None and stats.has_min_max:
            b_min = _decode_stat_str(stats.min)
            b_max = _decode_stat_str(stats.max)
            if b_min is not None and b_max is not None:
                if b_min == b_max:
                    if b_min in target_branches:
                        matching.append(rg_idx)
                    continue
                if any(b_min <= target <= b_max for target in target_branches):
                    matching.append(rg_idx)
                    needs_post_filter = True
                continue

        # Dictionary-encoded column chunks may omit Parquet min/max statistics;
        # inspect only the compact RLE-encoded branch column for this row group.
        rg_branch_col = pf.read_row_group(rg_idx, columns=["branch"]).column(0)
        unique_vals = {
            str(v) for v in pc.unique(rg_branch_col).to_pylist() if v is not None
        }
        if unique_vals & target_branches:
            matching.append(rg_idx)
            if not unique_vals.issubset(target_branches):
                needs_post_filter = True

    return matching, needs_post_filter


def read_reference_parquet(
    path: Path | str,
    *,
    branches: Sequence[str] | None = None,
    check_rows: bool = False,
) -> tuple[pa.Table, ReferenceMetadata]:
    """Read a reference Parquet table (optionally branch-pruned) and return (table, metadata)."""
    p = Path(path)
    pf = pq.ParquetFile(p)
    validate_schema(pf.schema_arrow, ArtifactKind.REFERENCE)

    if branches is None:
        table = pf.read()
    else:
        target_set = set(branches)
        if not target_set or pf.num_row_groups == 0:
            table = pf.schema_arrow.empty_table()
        else:
            matching_rgs, needs_filter = _select_matching_row_groups(pf, target_set)
            if not matching_rgs:
                table = pf.schema_arrow.empty_table()
            else:
                table = pf.read_row_groups(matching_rgs)
                if needs_filter:
                    mask = pc.is_in(
                        table.column("branch"),
                        pa.array(list(target_set), type=pa.string()),
                    )
                    table = table.filter(mask)

    meta = validate_reference_table(table, check_rows=check_rows)
    return table, meta


def write_reference_parquet(
    table: pa.Table,
    path: Path | str,
    *,
    compression_level: int = 3,
) -> Path:
    """Validate reference table contract and write one ZSTD Parquet row group per branch."""
    for col_name, idx_type in (("branch", pa.int8()), ("contig", pa.int16())):
        if col_name in table.schema.names:
            col_idx = table.schema.get_field_index(col_name)
            col_field = table.schema.field(col_idx)
            if (
                pa.types.is_string(col_field.type)
                or pa.types.is_large_string(col_field.type)
            ) and table.column(col_idx).null_count == 0:
                canon = canonical_dictionary_array(
                    table.column(col_idx), col_name, index_type=idx_type
                )
                table = table.set_column(
                    col_idx,
                    pa.field(
                        col_name,
                        pa.dictionary(idx_type, pa.string()),
                        nullable=False,
                    ),
                    canon,
                )

    validate_reference_table(table, check_rows=True)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer_kwargs: dict[str, object] = {
        "compression": "zstd",
        "compression_level": compression_level,
        "use_dictionary": ["branch", "contig"],
        "column_encoding": {
            "start": "DELTA_BINARY_PACKED",
            "end": "DELTA_BINARY_PACKED",
        },
    }
    if table.num_rows == 0:
        pq.write_table(table, out_path, **writer_kwargs)
        return out_path

    branch_idx = table.schema.get_field_index("branch")
    branch_dict = canonical_dictionary_array(
        table.column(branch_idx), "branch", index_type=pa.int8()
    )
    table = table.set_column(branch_idx, table.schema.field(branch_idx), branch_dict)
    b_codes = branch_dict.indices.to_numpy(zero_copy_only=False)
    boundaries = (np.flatnonzero(b_codes[1:] != b_codes[:-1]) + 1).tolist()
    slice_starts = [0, *boundaries]
    slice_ends = [*boundaries, table.num_rows]

    with pq.ParquetWriter(out_path, table.schema, **writer_kwargs) as writer:
        for s_idx, e_idx in zip(slice_starts, slice_ends, strict=True):
            branch_slice = table.slice(s_idx, e_idx - s_idx)
            writer.write_table(
                branch_slice, row_group_size=max(branch_slice.num_rows, 1)
            )

    return out_path


def _read_sample_for_cohort(path: Path | str) -> pa.Table:
    """Read and validate a single sample Parquet table for cohort aggregation."""
    p = Path(path)
    try:
        table = pq.read_table(p)
    except (FileNotFoundError, OSError, pa.ArrowInvalid, pa.ArrowException) as error:
        raise ReferenceError(f"{p}: cannot read sample Parquet ({error})") from error
    try:
        validate_sample_table(table)
    except SchemaError as error:
        raise ReferenceError(f"{p}: {error}") from error
    return table


def read_sample_tables(
    paths: Sequence[Path | str],
    *,
    threads: int = 1,
) -> tuple[pa.Table, ...]:
    """Read and validate multiple sample Parquet files into memory."""
    if not paths:
        raise ReferenceError("at least one sample Parquet path is required")
    resolved = tuple(Path(p) for p in paths)
    if threads > 1 and len(resolved) > 1:
        max_workers = min(threads, len(resolved))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tables = list(executor.map(_read_sample_for_cohort, resolved))
    else:
        tables = [_read_sample_for_cohort(p) for p in resolved]
    return tuple(tables)


load_cohort = read_sample_tables


def decode_dictionary_contigs(contig_col: pa.ChunkedArray) -> tuple[str, ...]:
    """Decode a dictionary-encoded or string contig column into a tuple of names."""
    c_dict = canonical_dictionary_array(contig_col, "contig")
    vocab = tuple(str(x) for x in c_dict.dictionary.to_pylist())
    codes = c_dict.indices.to_numpy(zero_copy_only=False)
    return tuple(vocab[int(c)] for c in codes)


def unique_dictionary_contigs(contig_col: pa.ChunkedArray) -> list[str]:
    """Return unique contig names present in a contig column in appearance/code order."""
    c_dict = canonical_dictionary_array(contig_col, "contig")
    vocab = [str(x) for x in c_dict.dictionary.to_pylist()]
    codes = c_dict.indices.to_numpy(zero_copy_only=False)
    return [vocab[int(c)] for c in np.unique(codes)]


def extract_sample_meta_and_counts(
    sample: pa.Table | SampleBuffers,
    *,
    validate: bool = False,
) -> tuple[SampleMetadata, NDArray[np.float64]]:
    """Extract SampleMetadata and float64 count array from SampleBuffers or pa.Table."""
    if isinstance(sample, SampleBuffers):
        if sample.num_rows != sample.meta.expected_num_bins:
            raise PredictError("sample buffer row count mismatch")
        return sample.meta, np.asarray(sample.count, dtype=np.float64)
    if (
        "count" in sample.schema.names
        and sample.schema.field("count").type == pa.float64()
    ):
        s_meta = SampleMetadata.from_table(sample)
        if sample.num_rows != s_meta.expected_num_bins:
            raise PredictError("weighted sample row count mismatch")
    elif validate:
        s_meta = validate_sample_table(sample)
    else:
        s_meta = SampleMetadata.from_table(sample)
    counts = np.asarray(
        sample.column("count").combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )
    return s_meta, counts


def align_sample_grid_indices(
    s_meta: SampleMetadata,
    contigs: tuple[str, ...],
    starts: NDArray[np.int64],
    ends: NDArray[np.int64],
    branch: str,
) -> NDArray[np.int64]:
    """Map reference branch coordinates onto implicit sample grid indices in O(runs)."""
    spans: dict[str, tuple[int, int, int]] = {}
    offset = 0
    b_size = s_meta.bin_size
    for name in s_meta.contig_order:
        length = s_meta.contig_lengths[name]
        n_c = (length + b_size - 1) // b_size
        spans[name] = (offset, length, n_c)
        offset += n_c

    n_rows = starts.size
    row_idx = np.empty(n_rows, dtype=np.int64)
    if n_rows == 0:
        return row_idx

    contig_arr = np.asarray(contigs, dtype=str)
    changes = np.flatnonzero(contig_arr[1:] != contig_arr[:-1]) + 1
    for r_lo, r_hi in zip(
        [0, *changes.tolist()], [*changes.tolist(), n_rows], strict=True
    ):
        c_name = str(contig_arr[r_lo])
        span = spans.get(c_name)
        if span is None:
            raise PredictError(f"branch {branch!r} bin count mismatch")
        s_off, c_len, c_nbins = span
        r_s, r_e = starts[r_lo:r_hi], ends[r_lo:r_hi]
        if np.any(r_s < 0) or np.any(r_s % b_size != 0):
            raise PredictError(f"branch {branch!r} bin count mismatch")
        bin_k = r_s // b_size
        if np.any(bin_k >= c_nbins) or not np.array_equal(
            r_e, np.minimum(r_s + b_size, c_len)
        ):
            raise PredictError(f"branch {branch!r} bin count mismatch")
        row_idx[r_lo:r_hi] = s_off + bin_k
    return row_idx


def extract_reference_branch_arrays(
    ref_table: pa.Table,
    branch: str,
    resolved_meta: ReferenceMetadata,
) -> ReferenceBranchArrays:
    """Filter one reference branch and extract zero-copy NumPy arrays."""
    ref_b = ref_table.filter(pc.equal(ref_table.column("branch"), branch))
    if ref_b.num_rows == 0:
        raise PredictError(f"branch {branch!r} has no reference rows")

    def _f32(name: str) -> NDArray[np.float32]:
        return (
            ref_b.column(name)
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
            .astype(np.float32)
        )

    contigs = decode_dictionary_contigs(ref_b.column("contig"))
    starts = (
        ref_b.column("start")
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
        .astype(np.int64)
    )
    ends = (
        ref_b.column("end")
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
        .astype(np.int64)
    )
    masked = (
        ref_b.column("masked")
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
        .astype(np.bool_)
    )
    gc_col, map_col = _f32("gc"), _f32("mappability")
    gc = gc_col if np.any(np.isfinite(gc_col) & (gc_col >= 0)) else None
    map_ = map_col if np.any(np.isfinite(map_col) & (map_col >= 0)) else None

    gc_quantiles: NDArray[np.float32] | None = None
    if "gc_quantiles" in ref_b.column_names:
        gc_q_col = ref_b.column("gc_quantiles")
        if gc_q_col.null_count < ref_b.num_rows:
            raw_gc_q = fixed_size_list_to_numpy(gc_q_col, NUM_GC_QUANTILES, np.float32)
            if np.any(np.isfinite(raw_gc_q)):
                gc_quantiles = raw_gc_q

    frag_len_col = (
        _f32("ref_frag_len") if "ref_frag_len" in ref_b.column_names else None
    )
    ref_frag_len = (
        frag_len_col
        if (
            frag_len_col is not None
            and np.any(np.isfinite(frag_len_col) & (frag_len_col >= 0))
        )
        else None
    )
    ep_rate_col = _f32("ref_ep_rate") if "ref_ep_rate" in ref_b.column_names else None
    ref_ep_rate = (
        ep_rate_col
        if (
            ep_rate_col is not None
            and np.any(np.isfinite(ep_rate_col) & (ep_rate_col >= 0))
        )
        else None
    )

    unmasked = ~masked
    u_count = int(np.count_nonzero(unmasked))
    pca_m = _f32("pca_mean").astype(np.float64)[unmasked]

    ld_col = ref_b.column("pca_loadings")
    ld_type = ld_col.type
    if not isinstance(ld_type, pa.FixedSizeListType):
        raise SchemaError(
            f"expected FixedSizeListType for 'pca_loadings', found {ld_type}"
        )
    p_max = ld_type.list_size
    p_branch = (
        resolved_meta.branch_pca_ranks.get(branch, p_max)
        if resolved_meta.branch_pca_ranks is not None
        else p_max
    )
    p_branch = min(max(int(p_branch), 0), p_max)
    if p_branch == 0 or u_count == 0:
        pca_ld = np.empty((0, u_count), dtype=np.float32)
    else:
        raw_ld = fixed_size_list_to_numpy(ld_col, p_max, np.float32)
        pca_ld = (
            raw_ld[:, :p_branch].T
            if bool(np.all(unmasked))
            else raw_ld[unmasked, :p_branch].T
        )
        if np.isnan(pca_ld).any():
            pca_ld = np.nan_to_num(pca_ld, nan=0.0)

    proto_ids = fixed_size_list_to_numpy(
        ref_b.column("proto_ids"), NUM_PROTOTYPES_PER_BIN, np.uint16
    )
    proto_weights = fixed_size_list_to_numpy(
        ref_b.column("proto_weights"), NUM_PROTOTYPES_PER_BIN, np.float16
    )
    if np.isnan(proto_weights).any():
        proto_weights = np.nan_to_num(proto_weights, nan=0.0).astype(np.float16)

    mean_dist = _f32("mean_distance")[unmasked]
    null_m = _f32("null_mean").astype(np.float64)[unmasked]
    bio_disp = np.maximum(
        np.nan_to_num(_f32("bio_dispersion").astype(np.float64)[unmasked], nan=0.0),
        0.0,
    )
    null_s = np.sqrt(bio_disp) / float(np.log(2.0))
    eff_bin_size = (
        resolved_meta.prenatal_bin_size
        if (
            branch == BRANCH_PRENATAL_AUTOSOMAL
            and resolved_meta.prenatal_bin_size is not None
        )
        else resolved_meta.bin_size
    )

    return ReferenceBranchArrays(
        contigs=contigs,
        starts=starts,
        ends=ends,
        masked=masked,
        gc=gc,
        mappability=map_,
        pca_mean=pca_m,
        pca_loadings=pca_ld,
        proto_ids=proto_ids,
        proto_weights=proto_weights,
        mean_distance=mean_dist,
        null_mean=null_m,
        null_std=null_s,
        gc_quantiles=gc_quantiles,
        ref_frag_len=ref_frag_len,
        ref_ep_rate=ref_ep_rate,
        bio_dispersion=bio_disp,
        bin_size=eff_bin_size,
        is_flat=resolved_meta.flat,
    )
