"""Formatted BED generation with bgzip compression and tabix indexing."""

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Final

import numpy as np
import pysam

from ccnv.models import BinResult, SegmentResult, natural_sort_key

__all__ = [
    "BINS_BED_HEADER",
    "CALLS_BED_HEADER",
    "PRENATAL_CALLS_BED_HEADER",
    "PRENATAL_SEGMENTS_BED_HEADER",
    "SEGMENTS_BED_HEADER",
    "bin_bed_rows",
    "calls_bed_rows",
    "format_bed_float",
    "natural_sort_key",
    "prenatal_bed_fields",
    "segment_bed_rows",
    "sort_bin_result",
    "sort_bin_results",
    "sort_segments",
    "write_compressed_bed",
]

BINS_BED_HEADER: Final[tuple[str, ...]] = (
    "#contig",
    "start",
    "end",
    "id",
    "ratio",
    "zscore",
    "weight",
    "reference_size",
    "masked",
    "branch",
)
SEGMENTS_BED_HEADER: Final[tuple[str, ...]] = (
    "#contig",
    "start",
    "end",
    "ratio",
    "zscore",
    "weight",
    "bin_count",
    "call",
)
CALLS_BED_HEADER: Final[tuple[str, ...]] = (
    "#contig",
    "start",
    "end",
    "ratio",
    "zscore",
    "type",
)
PRENATAL_SEGMENTS_BED_HEADER: Final[tuple[str, ...]] = (
    "#contig",
    "start",
    "end",
    "ratio",
    "zscore",
    "weight",
    "bin_count",
    "call",
    "expected_ratio",
    "implied_ff",
    "discordant",
    "discordance_reason",
)
PRENATAL_CALLS_BED_HEADER: Final[tuple[str, ...]] = (
    "#contig",
    "start",
    "end",
    "ratio",
    "zscore",
    "type",
    "expected_ratio",
    "implied_ff",
    "discordant",
    "discordance_reason",
)


def format_bed_float(value: float) -> str:
    """Format floating point numbers for BED output."""
    if not np.isfinite(value):
        return "NaN"
    return f"{value:.8g}"


def sort_bin_result(result: BinResult) -> BinResult:
    """Sort BinResult bins in natural chromosome order, preserving coords."""
    if len(result.contigs) <= 1:
        return result
    unique_contigs = list(dict.fromkeys(result.contigs))
    sorted_contigs = sorted(unique_contigs, key=natural_sort_key)
    if unique_contigs == sorted_contigs:
        return result

    contig_indices: dict[str, list[int]] = {}
    for i, c in enumerate(result.contigs):
        contig_indices.setdefault(c, []).append(i)

    idx_list: list[int] = []
    new_contigs: list[str] = []
    for c in sorted_contigs:
        indices = contig_indices[c]
        idx_list.extend(indices)
        new_contigs.extend([c] * len(indices))

    idx = np.array(idx_list, dtype=np.intp)
    return BinResult(
        branch=result.branch,
        contigs=tuple(new_contigs),
        starts=result.starts[idx],
        ends=result.ends[idx],
        ratios=result.ratios[idx],
        zscores=result.zscores[idx],
        weights=result.weights[idx],
        reference_sizes=result.reference_sizes[idx],
        masked=result.masked[idx],
    )


def sort_bin_results(results: tuple[BinResult, ...]) -> tuple[BinResult, ...]:
    """Sort a collection of BinResult objects by natural chromosome order."""
    sorted_branches = tuple(sort_bin_result(res) for res in results)
    return tuple(
        sorted(
            sorted_branches,
            key=lambda res: (
                natural_sort_key(res.contigs[0]) if res.contigs else (),
                res.branch,
            ),
        )
    )


def sort_segments(segments: Iterable[SegmentResult]) -> tuple[SegmentResult, ...]:
    """Sort SegmentResult records naturally by chromosome and start position."""
    return tuple(
        sorted(
            segments,
            key=lambda s: (natural_sort_key(s.contig), s.start, s.end),
        )
    )


def bin_bed_rows(results: tuple[BinResult, ...]) -> Iterator[tuple[object, ...]]:
    """Format bin records into BED tuples."""
    for result in results:
        for index in range(len(result.contigs)):
            contig = result.contigs[index]
            start = int(result.starts[index])
            end = int(result.ends[index])
            yield (
                contig,
                start,
                end,
                f"{contig}:{start}-{end}",
                format_bed_float(float(result.ratios[index])),
                format_bed_float(float(result.zscores[index])),
                format_bed_float(float(result.weights[index])),
                int(result.reference_sizes[index]),
                "true" if bool(result.masked[index]) else "false",
                result.branch,
            )


def prenatal_bed_fields(item: SegmentResult) -> tuple[object, ...]:
    """Return prenatal discordance annotations for a segment."""
    ratio_str = (
        format_bed_float(item.expected_ratio) if item.expected_ratio is not None else ""
    )
    ff_str = format_bed_float(item.implied_ff) if item.implied_ff is not None else ""
    return (
        ratio_str,
        ff_str,
        "true" if item.discordant else "false",
        item.discordance_reason or "",
    )


def segment_bed_rows(
    results: tuple[SegmentResult, ...],
    *,
    is_prenatal: bool = False,
) -> tuple[tuple[object, ...], ...]:
    """Format segment records into BED tuples."""
    return tuple(
        (
            item.contig,
            item.start,
            item.end,
            format_bed_float(item.ratio),
            format_bed_float(item.zscore),
            format_bed_float(item.weight),
            item.bin_count,
            item.call or "",
            *(prenatal_bed_fields(item) if is_prenatal else ()),
        )
        for item in results
    )


def calls_bed_rows(
    calls: Sequence[SegmentResult],
    *,
    is_prenatal: bool = False,
) -> tuple[tuple[object, ...], ...]:
    """Format call records into BED tuples."""
    return tuple(
        (
            item.contig,
            item.start,
            item.end,
            format_bed_float(item.ratio),
            format_bed_float(item.zscore),
            item.call,
            *(prenatal_bed_fields(item) if is_prenatal else ()),
        )
        for item in calls
    )


def write_compressed_bed(
    gz_path: Path, header: tuple[str, ...], rows: Iterable[tuple[object, ...]]
) -> Path:
    """Write tab-delimited records to a bgzipped file with tabix index."""
    with pysam.BGZFile(str(gz_path), "wb") as bgz:
        bgz.write(("\t".join(header) + "\n").encode("utf-8"))
        for row in rows:
            bgz.write(("\t".join(str(value) for value in row) + "\n").encode("utf-8"))
    pysam.tabix_index(str(gz_path), preset="bed", force=True)
    return gz_path
