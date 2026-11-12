"""Canonical synthetic data builders for test suites."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyBigWig
import pysam
from numpy.typing import NDArray

import ccnv.models as M
from ccnv._core import FRAG_BAND_EDGES
from ccnv.features.normalization import BranchModel
from ccnv.io.parquet import write_reference_parquet, write_sample_parquet


def _u32_or_zero(vals: Sequence[int] | np.ndarray | None, n: int) -> NDArray[np.uint32]:
    return np.zeros(n, np.uint32) if vals is None else np.asarray(vals, np.uint32)


def _f32_or_fill(
    vals: Sequence[float] | np.ndarray | None, n: int, fill: float
) -> NDArray[np.float32]:
    return (
        np.full(n, fill, np.float32) if vals is None else np.asarray(vals, np.float32)
    )


def _f64_or_fill(
    vals: float | Sequence[float] | np.ndarray | None, n: int, fill: float
) -> NDArray[np.float64]:
    if vals is None or isinstance(vals, (int, float)):
        return np.full(n, fill if vals is None else float(vals), np.float64)
    return np.asarray(vals, np.float64)


def metadata_for(
    kind: M.ArtifactKind,
    *,
    contig_lengths: Mapping[str, int],
    values: Mapping[str, object],
) -> dict[bytes, bytes]:
    """Build complete metadata for a test artifact, preserving sample contig order."""
    extra = (
        {"contig_order": list(contig_lengths.keys())}
        if kind is M.ArtifactKind.SAMPLE and "contig_order" not in values
        else {}
    )
    metadata: dict[str, object] = {
        "artifact_kind": kind.value,
        "schema_version": M.SCHEMA_VERSION,
        "coordinate_system": M.COORDINATE_SYSTEM,
        "contig_lengths": dict(contig_lengths),
        **extra,
        **values,
    }
    M.validate_metadata(metadata, kind)
    return M.encode_metadata(metadata)


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
    s_default = np.arange(n_bins, dtype=np.int64) * bin_size
    starts_arr = s_default if starts is None else np.asarray(starts, dtype=np.int64)
    ends_arr = (
        starts_arr + bin_size if ends is None else np.asarray(ends, dtype=np.int64)
    )
    if contigs is None:
        contig_order = ["chr1"]
        d_len = int(ends_arr[-1]) if n_bins > 0 else bin_size
        lengths_dict = (
            {"chr1": d_len} if contig_lengths is None else dict(contig_lengths)
        )
    else:
        c_list = list(contigs)
        contig_order = list(dict.fromkeys(c_list))
        lengths_dict = (
            {
                c: max(int(e) for k, e in zip(c_list, ends_arr, strict=True) if k == c)
                for c in contig_order
            }
            if contig_lengths is None
            else dict(contig_lengths)
        )

    src = identity if identity is not None else {"id": "synthetic_sample"}
    ref = {
        "name": "synthetic_ref",
        "n_contigs": len(contig_order),
        "contig_order": contig_order,
    }
    metadata = metadata_for(
        M.ArtifactKind.SAMPLE,
        contig_lengths=lengths_dict,
        values={
            "bin_size": bin_size,
            "contig_order": contig_order,
            "source_identity": src,
            "reference_identity": ref,
            "qc": {},
            "filters": asdict(M.ReadFilters()),
        },
    )

    paired_arr = np.zeros(n_bins, dtype=np.uint32)
    flen_sum_arr = np.zeros(n_bins, dtype=np.uint64)
    short_arr = np.zeros(n_bins, dtype=np.uint32)
    if fragment_lengths is None:
        bands_col: pa.Array = pa.nulls(
            n_bins, type=pa.list_(pa.uint32(), M.NUM_FRAG_BANDS)
        )
    else:
        bands_2d = np.zeros((n_bins, M.NUM_FRAG_BANDS), dtype=np.uint32)
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
                bands_2d[idx] = np.histogram(arr_in, bins=edges)[0].astype(np.uint32)
        bands_col = pa.FixedSizeListArray.from_arrays(
            pa.array(bands_2d.ravel(), type=pa.uint32()), M.NUM_FRAG_BANDS
        )

    return pa.table(
        {
            "count": np.asarray(counts, dtype=np.uint32),
            "paired_count": paired_arr,
            "frag_len_sum": flen_sum_arr,
            "short_frag_count": short_arr,
            "endpoint_count": _u32_or_zero(endpoint_counts, n_bins),
            "frag_gc_sum": _f32_or_fill(frag_gc_sums, n_bins, 0.0),
            "frag_bands": bands_col,
        },
        schema=M.SAMPLE_SCHEMA.with_metadata(metadata),
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
    path.parent.mkdir(parents=True, exist_ok=True)
    write_sample_parquet(
        make_sample_table(
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
        ),
        path,
    )
    return path


def make_sample_buffers(
    contig_lengths: Mapping[str, int],
    bin_size: int,
    counts: Sequence[int | float] | NDArray[np.uint32] | NDArray[np.float64],
    *,
    frag_bands: NDArray[np.uint32] | None = None,
    paired_counts: Sequence[int] | NDArray[np.uint32] | None = None,
    frag_len_sums: Sequence[int] | NDArray[np.uint64] | None = None,
    short_frag_counts: Sequence[int] | NDArray[np.uint32] | None = None,
    endpoint_counts: Sequence[int] | NDArray[np.uint32] | None = None,
    frag_gc_sums: Sequence[float] | NDArray[np.float32] | None = None,
    fragmentomics: M.FragmentomicsMetadata | None = None,
    sample_id: str = "synthetic_sample",
    qc: Mapping[str, object] | None = None,
    filters: Mapping[str, object] | None = None,
    kappa_frag: NDArray[np.float32] | None = None,
) -> M.SampleBuffers:
    """Build in-memory SampleBuffers with optional paired-end 24-band histograms."""
    raw = np.asarray(counts)
    dtype = np.float64 if np.issubdtype(raw.dtype, np.floating) else np.uint32
    count_arr: NDArray[np.uint32] | NDArray[np.float64] = np.asarray(raw, dtype=dtype)
    n_bins = int(count_arr.shape[0])
    b_u32 = np.asarray(frag_bands, np.uint32) if frag_bands is not None else None
    def_p = np.sum(b_u32, axis=1, dtype=np.uint32) if b_u32 is not None else None
    paired_arr = _u32_or_zero(
        paired_counts if paired_counts is not None else def_p, n_bins
    )
    flen_arr = (
        paired_arr.astype(np.uint64) * np.uint64(166)
        if frag_len_sums is None
        else np.asarray(frag_len_sums, dtype=np.uint64)
    )
    def_s = (
        np.sum(b_u32[:, :10], axis=1, dtype=np.uint32) if b_u32 is not None else None
    )
    short_arr = _u32_or_zero(
        short_frag_counts if short_frag_counts is not None else def_s, n_bins
    )

    kept = int(np.rint(np.sum(count_arr)))
    lengths_dict = {str(k): int(v) for k, v in contig_lengths.items()}
    c_order = tuple(lengths_dict.keys())
    def_qc = {"effective_reads_examined": kept, "effective_reads_kept": kept}
    meta = M.SampleMetadata(
        schema_version=M.SCHEMA_VERSION,
        coordinate_system=M.COORDINATE_SYSTEM,
        bin_size=bin_size,
        contig_order=c_order,
        contig_lengths=lengths_dict,
        source_identity={"id": sample_id},
        reference_identity={
            "name": "synthetic_ref",
            "n_contigs": len(c_order),
            "contig_order": list(c_order),
        },
        filters=dict(filters) if filters is not None else asdict(M.ReadFilters()),
        qc=dict(qc) if qc is not None else def_qc,
        fragmentomics=fragmentomics,
    )
    return M.SampleBuffers(
        meta=meta,
        count=count_arr,
        paired_count=paired_arr,
        frag_len_sum=flen_arr,
        short_frag_count=short_arr,
        endpoint_count=_u32_or_zero(endpoint_counts, n_bins),
        frag_gc_sum=_f32_or_fill(frag_gc_sums, n_bins, 0.0),
        frag_bands=b_u32,
        kappa_frag=kappa_frag,
    )


def make_branch_model(
    contigs: Sequence[str],
    counts: Sequence[float] | NDArray[np.floating],
    *,
    branch: str = M.BRANCH_AUTOSOMAL,
    bin_size: int = 5000,
    starts: Sequence[int] | NDArray[np.int64] | None = None,
    ends: Sequence[int] | NDArray[np.int64] | None = None,
    masked: Sequence[bool] | NDArray[np.bool_] | None = None,
    pca_mean: float | Sequence[float] | NDArray[np.floating] = 1.0,
    pca_loadings: NDArray[np.float32] | None = None,
    proto_ids: NDArray[np.uint16] | None = None,
    proto_weights: NDArray[np.float16] | None = None,
    mean_distance: Sequence[float] | NDArray[np.float32] | None = None,
    null_mean: Sequence[float] | NDArray[np.floating] | None = None,
    bio_dispersion: float | Sequence[float] | NDArray[np.floating] = 0.01,
    gc: Sequence[float] | NDArray[np.floating] | None = None,
    mappability: Sequence[float] | NDArray[np.floating] | None = None,
    gc_quantiles: NDArray[np.float32] | None = None,
    ref_frag_len: NDArray[np.float32] | None = None,
    ref_ep_rate: NDArray[np.float32] | None = None,
    is_flat: bool = False,
    kappa_frag: NDArray[np.float32] | None = None,
) -> BranchModel:
    """Build an in-memory BranchModel with off-contig prototype graph defaults."""
    c_tup = tuple(str(c) for c in contigs)
    n_bins = len(c_tup)
    if starts is None:
        offsets: dict[str, int] = {}
        s_list: list[int] = []
        for name in c_tup:
            s_list.append(offsets.get(name, 0))
            offsets[name] = s_list[-1] + bin_size
        starts_arr = np.asarray(s_list, dtype=np.int64)
    else:
        starts_arr = np.asarray(starts, dtype=np.int64)
    ends_arr = (
        starts_arr + bin_size if ends is None else np.asarray(ends, dtype=np.int64)
    )

    k_p = M.NUM_PROTOTYPES_PER_BIN
    uniq = list(dict.fromkeys(c_tup))
    if proto_ids is None:
        pids_arr = np.zeros((n_bins, k_p), dtype=np.uint16)
        if len(uniq) > 1:
            c_idx = {c: idx for idx, c in enumerate(uniq)}
            for r_i, c_name in enumerate(c_tup):
                pids_arr[r_i, 0] = np.uint16((c_idx[c_name] + 1) % len(uniq))
    else:
        pids_arr = np.asarray(proto_ids, dtype=np.uint16)

    if proto_weights is None:
        pw_arr = np.zeros((n_bins, k_p), dtype=np.float16)
        if n_bins > 0:
            pw_arr[:, 0] = np.float16(1.0)
    else:
        pw_arr = np.asarray(proto_weights, dtype=np.float16)

    ld_arr = (
        np.empty((0, n_bins), dtype=np.float32)
        if pca_loadings is None
        else np.asarray(pca_loadings, dtype=np.float32)
    )
    m_arr = (
        np.zeros(n_bins, dtype=np.bool_)
        if masked is None
        else np.asarray(masked, dtype=np.bool_)
    )
    return BranchModel(
        branch=branch,
        contigs=c_tup,
        starts=starts_arr,
        ends=ends_arr,
        counts=np.asarray(counts, dtype=np.float64),
        masked=m_arr,
        pca_mean=_f64_or_fill(pca_mean, n_bins, 1.0),
        pca_loadings=ld_arr,
        proto_ids=pids_arr,
        proto_weights=pw_arr,
        mean_distance=_f32_or_fill(mean_distance, n_bins, 0.0),
        null_mean=_f64_or_fill(null_mean, n_bins, 0.0),
        bio_dispersion=_f64_or_fill(bio_dispersion, n_bins, 0.01),
        gc=np.asarray(gc, dtype=np.float64) if gc is not None else None,
        mappability=np.asarray(mappability, float) if mappability is not None else None,
        gc_quantiles=gc_quantiles,
        ref_frag_len=ref_frag_len,
        ref_ep_rate=ref_ep_rate,
        bin_size=bin_size,
        is_flat=is_flat,
        kappa_frag=kappa_frag,
    )


def make_prenatal_metadata(
    *,
    prenatal_bin_size: int = 100_000,
    yfrac_threshold: float = 0.0041,
    ffy_slope: float = 0.985,
    olm_weights: Sequence[float] | None = None,
    olm_bias: float = 0.005,
    female_chrx_ratio_mean: float = 1.0003,
    female_chrx_ratio_std: float = 0.0112,
    alpha_1: float = -0.5,
    p_fetal_density: list[float] | None = None,
    p_maternal_density: list[float] | None = None,
    ff_frag_weights: list[float] | None = None,
    ff_frag_bias: float | None = None,
    ff_frag_feature_mean: list[float] | None = None,
    ff_frag_feature_scale: list[float] | None = None,
    ff_size_std: float = 0.018,
    band_fisher_weights: list[float] | None = None,
    mu_fetal_w: float | None = None,
    mu_maternal_w: float | None = None,
) -> M.PrenatalModelMetadata:
    """Build a synthetic PrenatalModelMetadata instance."""
    return M.PrenatalModelMetadata(
        prenatal_bin_size=prenatal_bin_size,
        yfrac_threshold=yfrac_threshold,
        ffy_slope=ffy_slope,
        olm_weights=list(olm_weights) if olm_weights is not None else [0.02] * 50,
        olm_bias=olm_bias,
        female_chrx_ratio_mean=female_chrx_ratio_mean,
        female_chrx_ratio_std=female_chrx_ratio_std,
        alpha_1=alpha_1,
        p_fetal_density=p_fetal_density,
        p_maternal_density=p_maternal_density,
        ff_frag_weights=ff_frag_weights,
        ff_frag_bias=ff_frag_bias,
        ff_frag_feature_mean=ff_frag_feature_mean,
        ff_frag_feature_scale=ff_frag_feature_scale,
        ff_size_std=ff_size_std,
        band_fisher_weights=band_fisher_weights,
        mu_fetal_w=mu_fetal_w,
        mu_maternal_w=mu_maternal_w,
    )


def _dict_col(
    items: Sequence[str], fallback: list[str], idx_type: pa.DataType
) -> tuple[pa.DictionaryArray, list[str]]:
    str_list = [str(v) for v in items]
    vocab = list(dict.fromkeys(str_list)) if str_list else fallback
    v_map = {name: i for i, name in enumerate(vocab)}
    indices = pa.array([v_map[v] for v in str_list], type=idx_type)
    return pa.DictionaryArray.from_arrays(indices, pa.array(vocab, pa.string())), vocab


def _fixed_2d(
    rows: Sequence[Sequence[int | float]] | np.ndarray | None,
    n_rows: int,
    width: int,
    np_dtype: type[np.uint16 | np.float16 | np.float32],
    pa_type: pa.DataType,
    default_first: float = 0.0,
) -> pa.FixedSizeListArray:
    arr = np.zeros((n_rows, width), dtype=np_dtype)
    if rows is None and n_rows > 0 and default_first != 0.0:
        arr[:, 0] = np_dtype(default_first)
    for r_i, row in enumerate(rows or ()):
        vals = list(row)[:width]
        arr[r_i, : len(vals)] = vals
    return pa.FixedSizeListArray.from_arrays(pa.array(arr.ravel(), type=pa_type), width)


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
    branch_col, b_vocab = _dict_col(branches, ["autosomal"], pa.int8())
    contig_col, _ = _dict_col(contigs, list(contig_lengths.keys()), pa.int16())

    rows_list = [list(r) for r in pca_loadings] if pca_loadings is not None else []
    pca_rank = max((len(r) for r in rows_list), default=0)
    pca_col: pa.Array = (
        pa.nulls(n_rows, pa.list_(pa.float32(), 0))
        if pca_rank == 0
        else _fixed_2d(rows_list, n_rows, pca_rank, np.float32, pa.float32())
    )
    n_gcq = M.NUM_GC_QUANTILES
    gc_q_col: pa.Array = (
        pa.nulls(n_rows, pa.list_(pa.float32(), n_gcq))
        if gc_quantiles is None
        else _fixed_2d(gc_quantiles, n_rows, n_gcq, np.float32, pa.float32())
    )
    k_p = M.NUM_PROTOTYPES_PER_BIN
    pids_col = _fixed_2d(proto_ids, n_rows, k_p, np.uint16, pa.uint16())
    pw_col = _fixed_2d(proto_weights, n_rows, k_p, np.float16, pa.float16(), 1.0)
    def_disp = (
        np.square(np.asarray(null_std, np.float64)).astype(np.float32)
        if null_std is not None
        else None
    )
    disp_arr = _f32_or_fill(
        bio_dispersion if bio_dispersion is not None else def_disp, n_rows, 0.01
    )

    if metadata is None:
        vals_meta: dict[str, object] = {
            "sample_identities": list(sample_identities),
            "model": dict(model) if model is not None else {"branches": b_vocab},
            **(extra_metadata or {}),
        }
        metadata = metadata_for(
            M.ArtifactKind.REFERENCE,
            contig_lengths={str(k): int(v) for k, v in contig_lengths.items()},
            values=vals_meta,
        )

    m_arr = (
        np.zeros(n_rows, dtype=np.bool_)
        if masked is None
        else np.asarray(masked, dtype=np.bool_)
    )
    return pa.table(
        {
            "branch": branch_col,
            "contig": contig_col,
            "start": np.asarray(starts, dtype=np.uint32),
            "end": np.asarray(ends, dtype=np.uint32),
            "masked": m_arr,
            "gc": _f32_or_fill(gc, n_rows, -1.0),
            "mappability": _f32_or_fill(mappability, n_rows, -1.0),
            "gc_quantiles": gc_q_col,
            "ref_frag_len": _f32_or_fill(ref_frag_len, n_rows, -1.0),
            "ref_ep_rate": _f32_or_fill(ref_ep_rate, n_rows, -1.0),
            "pca_mean": _f32_or_fill(pca_mean, n_rows, 1.0),
            "pca_loadings": pca_col,
            "proto_ids": pids_col,
            "proto_weights": pw_col,
            "mean_distance": _f32_or_fill(mean_distance, n_rows, 0.5),
            "null_mean": _f32_or_fill(null_mean, n_rows, 0.0),
            "bio_dispersion": disp_arr,
        },
        schema=M.build_reference_schema(pca_rank).with_metadata(metadata),
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
        lines.extend(seq[i : i + 60] for i in range(0, len(seq), 60))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_test_bam(
    path: Path,
    reads: Sequence[pysam.AlignedSegment] = (),
    *,
    contigs: Sequence[tuple[str, int]] = (("chr1", 250),),
    header: Mapping[str, object] | pysam.AlignmentHeader | None = None,
    index: bool = True,
) -> Path:
    """Write coordinate-sorted synthetic BAM file with optional BAI index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    def_hdr: dict[str, object] = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": name, "LN": length} for name, length in contigs],
    }
    sorted_reads = sorted(
        reads,
        key=lambda r: (
            r.reference_id if r.reference_id >= 0 else 1_000_000,
            max(r.reference_start, 0),
        ),
    )
    bam_hdr = def_hdr if header is None else header
    with pysam.AlignmentFile(str(path), "wb", header=bam_hdr) as bam:
        for read in sorted_reads:
            bam.write(read)
    if index:
        pysam.index(str(path))
    return path


def write_test_bigwig(
    path: Path,
    chroms: Mapping[str, int],
    entries: Sequence[tuple[str, int, int, float]] = (),
) -> Path:
    """Write synthetic BigWig file from 0-based half-open interval tuples."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bw = pyBigWig.open(str(path), "w")
    if bw is None:
        raise RuntimeError(f"failed to open BigWig file for writing: {path}")
    try:
        bw.addHeader([(str(k), int(v)) for k, v in chroms.items()])
        if entries:
            bw.addEntries(
                [str(e[0]) for e in entries],
                [int(e[1]) for e in entries],
                ends=[int(e[2]) for e in entries],
                values=[float(e[3]) for e in entries],
            )
    finally:
        bw.close()
    return path
