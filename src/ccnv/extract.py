"""Stream BAM/CRAM alignments into compact Schema v2.0.0 sample sketch Parquet."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Final

import numpy as np
import pyarrow as pa

from ccnv.io.alignments import (
    read_alignment_header,
    scan_alignment_bins,
    validate_alignment_inputs,
)
from ccnv.io.parquet import write_sample_parquet
from ccnv.models import (
    FRAGMENTOMICS_METADATA_KEY,
    NUM_FRAG_BANDS,
    SAMPLE_SCHEMA,
    AlignmentError,
    ArtifactKind,
    ExtractConfig,
    ExtractError,
    FragmentomicsMetadata,
    PredictError,
    ReadFilters,
    ReferenceMetadata,
    SampleBuffers,
    SampleMetadata,
    metadata_for,
    parquet_path,
    select_layout,
)
from ccnv.qc.sample import build_sample_qc
from ccnv.stats.fragmentomics import estimate_sample_landmarks

logger = logging.getLogger(__name__)

__all__ = [
    "ExtractError",
    "extract_for_predict",
    "extract_sample_table",
    "extracted_to_sample_buffers",
    "extracted_to_sample_table",
    "run",
    "run_extract",
]


def _scan_to_extracted(
    scan: Mapping[str, object],
    selected: Sequence[tuple[str, int]],
    *,
    bin_size: int,
    fmt: str,
    resolved_alignment: Path,
    resolved_fasta: Path | None,
    filters: ReadFilters,
) -> dict[str, object]:
    counts = np.asarray(scan["count"], dtype=np.uint32)
    paired_counts = np.asarray(scan["paired_count"], dtype=np.uint32)
    frag_len_sums = np.asarray(scan["frag_len_sum"], dtype=np.uint64)
    short_frag_counts = np.asarray(scan["short_frag_count"], dtype=np.uint32)
    endpoint_counts = np.asarray(scan["endpoint_count"], dtype=np.uint32)
    frag_gc_sums = np.asarray(scan["frag_gc_sum"], dtype=np.float32)
    paired_end = bool(scan["paired_end"])
    raw_bands = scan["frag_bands"]
    bands_arr = (
        np.asarray(raw_bands, dtype=np.uint32)
        if paired_end and raw_bands is not None
        else None
    )
    raw_qc_obj = scan["qc_counts"]
    raw_qc = (
        {
            str(k): int(v)
            for k, v in raw_qc_obj.items()
            if isinstance(v, int) and not isinstance(v, bool)
        }
        if isinstance(raw_qc_obj, Mapping)
        else {}
    )
    qc = build_sample_qc(raw_qc)
    meta_frag = estimate_sample_landmarks(
        hist_autosomal=np.asarray(scan["hist_autosomal"], dtype=np.int64),
        hist_chry=np.asarray(scan["hist_chry"], dtype=np.int64),
        hist_chrx=np.asarray(scan["hist_chrx"], dtype=np.int64),
        end_motif_counts=np.asarray(scan["motif_counts"], dtype=np.int64),
        paired_end=paired_end,
    )
    contig_order = [name for name, _ in selected]
    contig_lengths = dict(selected)
    source_id: dict[str, object] = {
        "path": str(resolved_alignment.resolve()),
        "format": fmt,
    }
    ref_id: dict[str, object] = {
        "n_contigs": len(contig_order),
        "contig_order": contig_order,
        **(
            {"fasta": str(resolved_fasta.resolve())}
            if resolved_fasta is not None
            else {}
        ),
    }
    qc_dict = dict(qc)
    filters_dict = dict(asdict(filters))
    sample_meta = SampleMetadata(
        bin_size=bin_size,
        contig_order=tuple(contig_order),
        contig_lengths=contig_lengths,
        source_identity=source_id,
        reference_identity=ref_id,
        qc=qc_dict,
        filters=filters_dict,
        fragmentomics=meta_frag,
    )
    return {
        "counts": counts,
        "count": counts,
        "paired_count": paired_counts,
        "frag_len_sum": frag_len_sums,
        "short_frag_count": short_frag_counts,
        "endpoint_count": endpoint_counts,
        "frag_gc_sum": frag_gc_sums,
        "frag_bands": bands_arr,
        "paired_end": paired_end,
        "contig_lengths": contig_lengths,
        "contig_order": contig_order,
        "bin_size": bin_size,
        "format": fmt,
        "resolved_alignment": resolved_alignment,
        "resolved_fasta": resolved_fasta,
        "filters": filters,
        "qc": qc_dict,
        "fragmentomics": meta_frag,
        "sample_metadata": sample_meta,
    }


def extracted_to_sample_buffers(extracted: Mapping[str, object]) -> SampleBuffers:
    """Wrap extracted scan arrays and metadata directly without building a PyArrow Table."""
    meta = extracted["sample_metadata"]
    assert isinstance(meta, SampleMetadata)
    raw_bands = extracted.get("frag_bands")
    bands = (
        np.asarray(raw_bands, dtype=np.uint32)
        if bool(extracted["paired_end"]) and isinstance(raw_bands, np.ndarray)
        else None
    )
    return SampleBuffers(
        meta=meta,
        count=np.asarray(extracted["counts"], dtype=np.uint32),
        paired_count=np.asarray(extracted["paired_count"], dtype=np.uint32),
        frag_len_sum=np.asarray(extracted["frag_len_sum"], dtype=np.uint64),
        short_frag_count=np.asarray(extracted["short_frag_count"], dtype=np.uint32),
        endpoint_count=np.asarray(extracted["endpoint_count"], dtype=np.uint32),
        frag_gc_sum=np.asarray(extracted["frag_gc_sum"], dtype=np.float32),
        frag_bands=bands,
    )


def extracted_to_sample_table(extracted: Mapping[str, object]) -> pa.Table:
    """Assemble a validated Schema v2.0.0 Arrow sample table from extracted scan arrays."""
    counts = np.asarray(extracted["counts"], dtype=np.uint32)
    n_bins = int(counts.shape[0])
    paired_end = bool(extracted["paired_end"])
    raw_bands = extracted.get("frag_bands")
    if paired_end and isinstance(raw_bands, np.ndarray):
        frag_bands_col: pa.Array = pa.FixedSizeListArray.from_arrays(
            pa.array(raw_bands.astype(np.uint32, copy=False).ravel(), type=pa.uint32()),
            NUM_FRAG_BANDS,
        )
    else:
        frag_bands_col = pa.nulls(n_bins, type=pa.list_(pa.uint32(), NUM_FRAG_BANDS))

    contig_order_raw = extracted["contig_order"]
    assert isinstance(contig_order_raw, list)
    contig_order = [str(x) for x in contig_order_raw]
    contig_lengths_raw = extracted["contig_lengths"]
    assert isinstance(contig_lengths_raw, Mapping)
    contig_lengths = {str(k): int(v) for k, v in contig_lengths_raw.items()}
    resolved_alignment = extracted["resolved_alignment"]
    assert isinstance(resolved_alignment, Path)
    resolved_fasta = extracted.get("resolved_fasta")
    filters = extracted["filters"]
    assert isinstance(filters, ReadFilters)
    qc = extracted["qc"]
    assert isinstance(qc, Mapping)
    meta_frag = extracted["fragmentomics"]
    assert isinstance(meta_frag, FragmentomicsMetadata)

    ref_id: dict[str, object] = {
        "n_contigs": len(contig_order),
        "contig_order": contig_order,
        **(
            {"fasta": str(resolved_fasta.resolve())}
            if isinstance(resolved_fasta, Path)
            else {}
        ),
    }
    metadata = metadata_for(
        ArtifactKind.SAMPLE,
        contig_lengths=contig_lengths,
        values={
            "bin_size": int(str(extracted["bin_size"])),
            "contig_order": contig_order,
            "source_identity": {
                "path": str(resolved_alignment.resolve()),
                "format": str(extracted["format"]),
            },
            "reference_identity": ref_id,
            "qc": dict(qc),
            "filters": dict(asdict(filters)),
        },
    )
    metadata[FRAGMENTOMICS_METADATA_KEY] = json.dumps(
        asdict(meta_frag), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    return pa.table(
        {
            "count": counts,
            "paired_count": np.asarray(extracted["paired_count"], dtype=np.uint32),
            "frag_len_sum": np.asarray(extracted["frag_len_sum"], dtype=np.uint64),
            "short_frag_count": np.asarray(
                extracted["short_frag_count"], dtype=np.uint32
            ),
            "endpoint_count": np.asarray(extracted["endpoint_count"], dtype=np.uint32),
            "frag_gc_sum": np.asarray(extracted["frag_gc_sum"], dtype=np.float32),
            "frag_bands": frag_bands_col,
        },
        schema=SAMPLE_SCHEMA.with_metadata(metadata),
    )


def _filters_from_ref_meta(ref_meta: ReferenceMetadata) -> ReadFilters:
    raw = ref_meta.filters
    if not isinstance(raw, Mapping):
        return ReadFilters()

    def _bool(key: str, default: bool) -> bool:
        v = raw.get(key)
        return v if isinstance(v, bool) else default

    mq = raw.get("min_mapping_quality")
    min_mq = int(mq) if isinstance(mq, int) and not isinstance(mq, bool) else 20
    return ReadFilters(
        min_mapping_quality=min_mq,
        exclude_duplicates=_bool("exclude_duplicates", True),
        exclude_secondary=_bool("exclude_secondary", True),
        exclude_supplementary=_bool("exclude_supplementary", True),
        exclude_unmapped=_bool("exclude_unmapped", True),
        require_proper_pair=_bool("require_proper_pair", True),
    )


def extract_for_predict(
    alignment: Path,
    ref_meta: ReferenceMetadata,
    *,
    threads: int = 1,
    fasta: Path | None = None,
    collect_bands: bool = False,
) -> dict[str, object]:
    """Stream BAM/CRAM directly into reference-aligned bin arrays via ccnv._core."""
    try:
        fmt, resolved_alignment, resolved_fasta = validate_alignment_inputs(
            alignment, fasta
        )
        header = read_alignment_header(resolved_alignment, resolved_fasta)
    except (AlignmentError, ExtractError, ValueError, OSError) as err:
        raise PredictError(f"failed to read sample {alignment}: {err}") from err

    if not header or any(length <= 0 for _, length in header):
        raise PredictError(f"{resolved_alignment} header declares invalid contigs")
    header_map = dict(header)
    if len(header_map) != len(header):
        raise PredictError(f"{resolved_alignment} header declares duplicate contigs")

    selected: list[tuple[str, int]] = []
    for name, expected_len in ref_meta.contig_lengths.items():
        actual_len = header_map.get(name)
        if actual_len is None:
            continue
        if actual_len != expected_len:
            raise PredictError(f"contig length mismatch for {name!r}")
        selected.append((name, expected_len))
    if not selected:
        raise PredictError(
            f"{resolved_alignment} declares none of the reference contigs"
        )

    filters = _filters_from_ref_meta(ref_meta)
    try:
        scan = scan_alignment_bins(
            resolved_alignment,
            selected,
            ref_meta.bin_size,
            filters,
            threads=threads,
            fasta=resolved_fasta,
            collect_bands=collect_bands,
        )
    except (AlignmentError, ExtractError, ValueError, OSError) as err:
        raise PredictError(f"failed to stream alignment {alignment}: {err}") from err

    return _scan_to_extracted(
        scan,
        selected,
        bin_size=ref_meta.bin_size,
        fmt=fmt,
        resolved_alignment=resolved_alignment,
        resolved_fasta=resolved_fasta,
        filters=filters,
    )


def extract_sample_table(config: ExtractConfig) -> pa.Table:
    """Extract a Schema v2.0.0 sample sketch table in memory from one BAM or CRAM."""
    fmt, resolved_alignment, resolved_fasta = validate_alignment_inputs(
        config.alignment, config.fasta
    )
    logger.debug("opening alignment %s as %s", resolved_alignment, fmt)

    header = read_alignment_header(resolved_alignment, resolved_fasta)
    if not header or any(length <= 0 for _, length in header):
        raise ExtractError(f"{resolved_alignment} header declares invalid contigs")

    selected = select_layout(header, all_contigs=config.all_contigs)
    if not selected:
        raise ExtractError(
            f"{resolved_alignment} header declares no canonical chromosomes; "
            "pass --all-contigs to bin every contig"
        )
    if len(dict(selected)) != len(selected):
        raise ExtractError(f"{resolved_alignment} header declares duplicate contigs")

    scan = scan_alignment_bins(
        resolved_alignment,
        selected,
        config.bin_size,
        config.filters,
        threads=config.threads,
        fasta=resolved_fasta,
        collect_bands=True,
    )
    extracted = _scan_to_extracted(
        scan,
        selected,
        bin_size=config.bin_size,
        fmt=fmt,
        resolved_alignment=resolved_alignment,
        resolved_fasta=resolved_fasta,
        filters=config.filters,
    )
    qc = extracted["qc"]
    if isinstance(qc, Mapping):
        logger.debug(
            "extract QC summary: %s reads seen, %s kept, %s mate_dedup_skipped, %s filtered",
            qc.get("reads_seen"),
            qc.get("reads_kept"),
            qc.get("mate_dedup_skipped"),
            qc.get("reads_filtered"),
        )
    return extracted_to_sample_table(extracted)


def run(config: ExtractConfig) -> None:
    """Extract binned counts from one BAM/CRAM into canonical sample Parquet."""
    table = extract_sample_table(config)
    output = parquet_path(config.prefix)
    logger.debug("writing sample parquet %s (%d bins)", output, table.num_rows)
    write_sample_parquet(table, output)
    logger.debug(
        "wrote sample parquet %s (%.1f KB)",
        output,
        output.stat().st_size / 1024,
    )


run_extract: Final = run
