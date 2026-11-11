"""Stream BAM/CRAM alignments into compact Schema v2.0.0 sample sketch Parquet."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import asdict
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
    ArtifactKind,
    ExtractConfig,
    ExtractError,
    metadata_for,
    parquet_path,
    select_layout,
)
from ccnv.qc.sample import build_sample_qc
from ccnv.stats.fragmentomics import estimate_sample_landmarks

logger = logging.getLogger(__name__)

__all__ = [
    "ExtractError",
    "extract_sample_table",
    "run",
    "run_extract",
]


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

    contig_lengths = dict(selected)
    if len(contig_lengths) != len(selected):
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

    counts = np.asarray(scan["count"], dtype=np.uint32)
    paired_counts = np.asarray(scan["paired_count"], dtype=np.uint32)
    frag_len_sums = np.asarray(scan["frag_len_sum"], dtype=np.uint64)
    short_frag_counts = np.asarray(scan["short_frag_count"], dtype=np.uint32)
    endpoint_counts = np.asarray(scan["endpoint_count"], dtype=np.uint32)
    frag_gc_sums = np.asarray(scan["frag_gc_sum"], dtype=np.float32)
    paired_end = bool(scan["paired_end"])
    n_bins = int(counts.shape[0])

    raw_bands = scan["frag_bands"]
    if paired_end and raw_bands is not None:
        bands_arr = np.asarray(raw_bands, dtype=np.uint32)
        frag_bands_col: pa.Array = pa.FixedSizeListArray.from_arrays(
            pa.array(bands_arr.ravel(), type=pa.uint32()),
            NUM_FRAG_BANDS,
        )
    else:
        frag_bands_col = pa.nulls(n_bins, type=pa.list_(pa.uint32(), NUM_FRAG_BANDS))

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
    logger.debug(
        "extract QC summary: %d reads seen, %d kept, %d mate_dedup_skipped, %d filtered",
        qc["reads_seen"],
        qc["reads_kept"],
        qc["mate_dedup_skipped"],
        qc["reads_filtered"],
    )

    meta_frag = estimate_sample_landmarks(
        hist_autosomal=np.asarray(scan["hist_autosomal"], dtype=np.int64),
        hist_chry=np.asarray(scan["hist_chry"], dtype=np.int64),
        hist_chrx=np.asarray(scan["hist_chrx"], dtype=np.int64),
        end_motif_counts=np.asarray(scan["motif_counts"], dtype=np.int64),
        paired_end=paired_end,
    )

    contig_names = [name for name, _ in selected]
    ref_id: dict[str, object] = {
        "n_contigs": len(contig_names),
        "contig_order": contig_names,
        **(
            {"fasta": str(resolved_fasta.resolve())}
            if resolved_fasta is not None
            else {}
        ),
    }

    metadata = metadata_for(
        ArtifactKind.SAMPLE,
        contig_lengths=contig_lengths,
        values={
            "bin_size": config.bin_size,
            "contig_order": contig_names,
            "source_identity": {
                "path": str(resolved_alignment.resolve()),
                "format": fmt,
            },
            "reference_identity": ref_id,
            "qc": dict(qc),
            "filters": dict(asdict(config.filters)),
        },
    )
    metadata[FRAGMENTOMICS_METADATA_KEY] = json.dumps(
        asdict(meta_frag), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    return pa.table(
        {
            "count": counts,
            "paired_count": paired_counts,
            "frag_len_sum": frag_len_sums,
            "short_frag_count": short_frag_counts,
            "endpoint_count": endpoint_counts,
            "frag_gc_sum": frag_gc_sums,
            "frag_bands": frag_bands_col,
        },
        schema=SAMPLE_SCHEMA.with_metadata(metadata),
    )


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
