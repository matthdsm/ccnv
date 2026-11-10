"""Structural variant VCFv4.2 writer with bgzip compression and tabix indexing."""

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import numpy as np
import pysam

from ccnv.models import (
    BinResult,
    FetalFractionResult,
    SegmentResult,
    natural_sort_key,
)

__all__ = [
    "build_vcf_header",
    "vcf_float",
    "write_bins_vcf",
    "write_segments_vcf",
]


def vcf_float(value: float) -> float | None:
    """Return finite float or None for missing VCF values."""
    return float(value) if np.isfinite(value) else None


def build_vcf_header(
    contig_lengths: dict[str, int],
    sample_name: str,
    record_type: Literal["bins", "segments"],
    extra_contigs: Iterable[str] = (),
    prenatal_result: FetalFractionResult | None = None,
) -> pysam.VariantHeader:
    """Construct a VCFv4.2 header declared with contigs and format fields."""
    header = pysam.VariantHeader()
    header.add_line("##source=ccnv_predict")
    all_contigs = set(contig_lengths.keys()) | set(extra_contigs)
    for contig in sorted(all_contigs, key=natural_sort_key):
        length = contig_lengths.get(contig)
        if length is not None:
            header.contigs.add(contig, length=length)
        else:
            header.contigs.add(contig)

    header.add_line(
        "##INFO=<ID=END,Number=1,Type=Integer,"
        'Description="End coordinate of structural variant">'
    )
    header.add_line(
        "##INFO=<ID=SVTYPE,Number=1,Type=String,"
        'Description="Type of structural variant">'
    )
    header.add_line(
        "##FORMAT=<ID=LOG2R,Number=1,Type=Float,"
        'Description="Log2 copy ratio against within-sample reference">'
    )
    header.add_line(
        "##FORMAT=<ID=ZSCORE,Number=1,Type=Float,"
        'Description="Z-score relative to reference cohort null model">'
    )
    header.add_line(
        "##FORMAT=<ID=WEIGHT,Number=1,Type=Float,"
        'Description="Precision weight for segmentation">'
    )

    if record_type == "bins":
        header.add_line('##ALT=<ID=CNV,Description="Copy number variation">')
        header.add_line(
            '##INFO=<ID=BRANCH,Number=1,Type=String,Description="Model branch">'
        )
        header.add_line(
            "##FILTER=<ID=MASKED,"
            'Description="Bin masked due to low coverage or quality filters">'
        )
        header.add_line(
            "##FORMAT=<ID=REF_SIZE,Number=1,Type=Integer,"
            'Description="Number of within-sample reference bins used">'
        )
    else:
        header.add_line('##ALT=<ID=DUP,Description="Duplication / copy number gain">')
        header.add_line('##ALT=<ID=DEL,Description="Deletion / copy number loss">')
        header.add_line('##ALT=<ID=CNV,Description="Copy number variation">')
        header.add_line(
            "##INFO=<ID=IMPRECISE,Number=0,Type=Flag,"
            'Description="Imprecise structural variation">'
        )
        header.add_line(
            "##FORMAT=<ID=BINS,Number=1,Type=Integer,"
            'Description="Number of bins in segment">'
        )
        if prenatal_result is not None:
            header.add_line(f"##fetalSex={prenatal_result.fetal_sex}")
            header.add_line(f"##fetalFraction={prenatal_result.active_ff:.4f}")
            header.add_line(f"##scaCall={prenatal_result.sca.call}")
            header.add_line(
                "##INFO=<ID=DISCORDANT,Number=0,Type=Flag,"
                'Description="Segment ratio is discordant with fetal '
                'fraction or karyotype">'
            )
            header.add_line(
                "##INFO=<ID=DISC_REASON,Number=1,Type=String,"
                'Description="Reason for discordance flag">'
            )
            header.add_line(
                "##FORMAT=<ID=EXP_LOG2R,Number=1,Type=Float,"
                'Description="Expected log2 ratio based on active fetal fraction">'
            )
            header.add_line(
                "##FORMAT=<ID=IMPLIED_FF,Number=1,Type=Float,"
                'Description="Implied fetal fraction based on segment ratio">'
            )
            header.add_line(
                "##INFO=<ID=FRAG_SHORT_R,Number=1,Type=Float,"
                'Description="Fraction of short cell-free DNA fragments <= 150 bp">'
            )
            header.add_line(
                "##INFO=<ID=FRAG_LLR,Number=1,Type=Float,"
                'Description="Fragment length log-likelihood ratio for fetal vs maternal origin">'
            )
            header.add_line(
                "##INFO=<ID=ORIGIN,Number=1,Type=String,"
                'Description="Fragmentomic origin call: FETAL_CONFIRMED or MATERNAL_CONFIRMED">'
            )

    header.add_sample(sample_name)
    return header


def write_bins_vcf(
    gz_path: Path,
    header: pysam.VariantHeader,
    results: tuple[BinResult, ...],
    sample_name: str,
) -> Path:
    """Write bin-level copy ratios to a compressed, indexed VCF."""
    with pysam.VariantFile(str(gz_path), "wz", header=header) as vf:
        for result in results:
            branch = result.branch
            for index in range(len(result.contigs)):
                contig = result.contigs[index]
                start = int(result.starts[index])
                end = int(result.ends[index])
                is_masked = bool(result.masked[index])
                filt = "MASKED" if is_masked else "PASS"
                rec = vf.new_record(
                    contig=contig,
                    start=start,
                    stop=end,
                    alleles=("N", "<CNV>"),
                    id=f"{contig}:{start}-{end}",
                    filter=filt,
                )
                rec.info["SVTYPE"] = "CNV"
                rec.info["BRANCH"] = branch
                sample = rec.samples[sample_name]
                sample["LOG2R"] = vcf_float(float(result.ratios[index]))
                sample["ZSCORE"] = vcf_float(float(result.zscores[index]))
                sample["WEIGHT"] = vcf_float(float(result.weights[index]))
                sample["REF_SIZE"] = int(result.reference_sizes[index])
                vf.write(rec)
    pysam.tabix_index(str(gz_path), preset="vcf", force=True)
    return gz_path


def write_segments_vcf(
    gz_path: Path,
    header: pysam.VariantHeader,
    results: tuple[SegmentResult, ...],
    sample_name: str,
    *,
    calls_only: bool = False,
    is_prenatal: bool = False,
) -> Path:
    """Write segment-level calls and metrics to a compressed, indexed VCF."""
    with pysam.VariantFile(str(gz_path), "wz", header=header) as vf:
        for item in results:
            if calls_only and item.call is None:
                continue
            contig = item.contig
            start = item.start
            end = item.end
            if item.call == "gain":
                alt = "<DUP>"
                svtype = "DUP"
                filt = "PASS"
            elif item.call == "loss":
                alt = "<DEL>"
                svtype = "DEL"
                filt = "PASS"
            else:
                alt = "<CNV>"
                svtype = "CNV"
                filt = None
            rec = vf.new_record(
                contig=contig,
                start=start,
                stop=end,
                alleles=("N", alt),
                id=".",
                filter=filt,
            )
            rec.info["SVTYPE"] = svtype
            rec.info["IMPRECISE"] = True
            if is_prenatal:
                if item.discordant:
                    rec.info["DISCORDANT"] = True
                    if item.discordance_reason:
                        rec.info["DISC_REASON"] = item.discordance_reason
                if item.frag_short_ratio is not None:
                    rec.info["FRAG_SHORT_R"] = vcf_float(item.frag_short_ratio)
                if item.frag_llr_size is not None:
                    rec.info["FRAG_LLR"] = vcf_float(item.frag_llr_size)
                if item.origin_call is not None:
                    rec.info["ORIGIN"] = item.origin_call
            sample = rec.samples[sample_name]
            sample["LOG2R"] = vcf_float(item.ratio)
            sample["ZSCORE"] = vcf_float(item.zscore)
            sample["WEIGHT"] = vcf_float(item.weight)
            sample["BINS"] = int(item.bin_count)
            if is_prenatal:
                if item.expected_ratio is not None:
                    sample["EXP_LOG2R"] = vcf_float(item.expected_ratio)
                if item.implied_ff is not None:
                    sample["IMPLIED_FF"] = vcf_float(item.implied_ff)
            vf.write(rec)

    pysam.tabix_index(str(gz_path), preset="vcf", force=True)
    return gz_path
