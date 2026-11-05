import base64
import gzip
import json
import logging
import re
from collections.abc import Iterable, Sequence
from importlib.resources import files
from pathlib import Path
from typing import Final, Literal

import jinja2
import numpy as np
import pysam

from ccnv.models import AnalysisMode, Cytoband
from ccnv.results import (
    BinResult,
    FetalFractionResult,
    PredictionOutputs,
    SegmentResult,
)

logger = logging.getLogger(__name__)

BINS_BED_HEADER = (
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
SEGMENTS_BED_HEADER = (
    "#contig",
    "start",
    "end",
    "ratio",
    "zscore",
    "weight",
    "bin_count",
    "call",
)
CALLS_BED_HEADER = ("#contig", "start", "end", "ratio", "zscore", "type")
PRENATAL_SEGMENTS_BED_HEADER = (
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
PRENATAL_CALLS_BED_HEADER = (
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
PLOTLY_CDN_URL: Final[str] = "https://cdn.plot.ly/plotly-2.35.2.min.js"
_JINJA_ENV: Final[jinja2.Environment] = jinja2.Environment(
    autoescape=jinja2.select_autoescape(["html", "j2"]),
    keep_trailing_newline=True,
)


def natural_sort_key(name: str) -> tuple[tuple[int, int | str], ...]:
    """Return key for natural sort order treating embedded numbers as integers."""

    return tuple(
        (0, int(chunk)) if chunk.isdigit() else (1, chunk.casefold())
        for chunk in re.split(r"(\d+)", name)
        if chunk
    )


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


def _strip_chr(name: str) -> str:
    return name[3:] if name.lower().startswith("chr") else name


def read_cytobands(path: Path, contig_lengths: dict[str, int]) -> tuple[Cytoband, ...]:
    """Parse a UCSC cytoband BED/TSV file (.txt, .bed, or .gz) aligned to contigs."""

    contig_lookup: dict[str, str] = {}
    for c in contig_lengths:
        contig_lookup[c.casefold()] = c
        contig_lookup[_strip_chr(c).casefold()] = c

    open_fn = gzip.open if path.suffix.lower() == ".gz" else open
    bands: list[Cytoband] = []
    with open_fn(path, "rt", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("track"):
                continue
            fields = line.split("\t") if "\t" in line else line.split()
            if len(fields) < 4:
                continue
            raw_contig = fields[0]
            canonical = contig_lookup.get(raw_contig.casefold()) or contig_lookup.get(
                _strip_chr(raw_contig).casefold()
            )
            if canonical is None:
                continue
            try:
                start = int(fields[1])
                end = int(fields[2])
            except ValueError:
                continue
            max_len = contig_lengths[canonical]
            start = max(0, min(start, max_len))
            end = max(0, min(end, max_len))
            if end <= start:
                continue
            name = fields[3]
            stain = fields[4] if len(fields) >= 5 else "gneg"
            bands.append(
                Cytoband(
                    contig=canonical,
                    start=start,
                    end=end,
                    name=name,
                    stain=stain,
                )
            )
    return tuple(
        sorted(bands, key=lambda b: (natural_sort_key(b.contig), b.start, b.end))
    )


def write_outputs(
    prefix: Path,
    bins: tuple[BinResult, ...],
    segments: tuple[SegmentResult, ...],
    contig_lengths: dict[str, int],
    plot: bool,
    vcf: bool = True,
    *,
    cytobands: Path | Sequence[Cytoband] | None = None,
    z_threshold: float = 3.0,
    ratio_threshold: float | None = None,
    prenatal_result: FetalFractionResult | None = None,
    mode: AnalysisMode = "germline",
    qc_json_path: Path | None = None,
) -> PredictionOutputs:
    """Write compressed BED evidence, VCFs, and an interactive HTML plot."""

    prefix.parent.mkdir(parents=True, exist_ok=True)
    bins = sort_bin_results(bins)
    segments = sort_segments(segments)
    bins_bed = Path(f"{prefix}_bins.bed.gz")
    segments_bed = Path(f"{prefix}_segments.bed.gz")
    calls_bed = Path(f"{prefix}_calls.bed.gz")
    plot_path = Path(f"{prefix}_plot.html") if plot else None
    sample_name = prefix.name if prefix.name else "SAMPLE"
    calls = [item for item in segments if item.call is not None]
    total_bins = sum(b.starts.size for b in bins)
    is_prenatal = mode == "prenatal"
    seg_header = PRENATAL_SEGMENTS_BED_HEADER if is_prenatal else SEGMENTS_BED_HEADER
    call_header = PRENATAL_CALLS_BED_HEADER if is_prenatal else CALLS_BED_HEADER

    logger.debug(
        "writing prediction outputs to prefix %s "
        "(%d bins, %d segments, %d calls, mode=%s)",
        prefix,
        total_bins,
        len(segments),
        len(calls),
        mode,
    )
    _write_compressed_bed(bins_bed, BINS_BED_HEADER, _bin_bed_rows(bins))
    _write_compressed_bed(
        segments_bed, seg_header, _segment_bed_rows(segments, is_prenatal=is_prenatal)
    )
    _write_compressed_bed(
        calls_bed,
        call_header,
        _calls_bed_rows(calls, is_prenatal=is_prenatal),
    )
    bins_vcf: Path | None = None
    segments_vcf: Path | None = None
    calls_vcf: Path | None = None
    if vcf:
        bins_vcf = Path(f"{prefix}_bins.vcf.gz")
        segments_vcf = Path(f"{prefix}_segments.vcf.gz")
        calls_vcf = Path(f"{prefix}_calls.vcf.gz")
        _write_bins_vcf(
            bins_vcf,
            _build_vcf_header(
                contig_lengths,
                sample_name,
                "bins",
                extra_contigs=(c for r in bins for c in r.contigs),
            ),
            bins,
            sample_name,
        )
        _write_segments_vcf(
            segments_vcf,
            _build_vcf_header(
                contig_lengths,
                sample_name,
                "segments",
                extra_contigs=(s.contig for s in segments),
                prenatal_result=prenatal_result if is_prenatal else None,
            ),
            segments,
            sample_name,
            is_prenatal=is_prenatal,
        )
        _write_segments_vcf(
            calls_vcf,
            _build_vcf_header(
                contig_lengths,
                sample_name,
                "segments",
                extra_contigs=(s.contig for s in segments),
                prenatal_result=prenatal_result if is_prenatal else None,
            ),
            segments,
            sample_name,
            calls_only=True,
            is_prenatal=is_prenatal,
        )
    logger.debug(
        "wrote %s, %s, %s",
        bins_bed.name,
        segments_bed.name,
        calls_bed.name,
    )
    if plot_path is not None:
        write_genome_plot(
            plot_path,
            bins,
            segments,
            contig_lengths,
            cytobands=cytobands,
            z_threshold=z_threshold,
            ratio_threshold=ratio_threshold,
            sample_name=sample_name,
            prenatal_result=prenatal_result if is_prenatal else None,
        )
    return PredictionOutputs(
        bins_bed=bins_bed,
        segments_bed=segments_bed,
        calls_bed=calls_bed,
        plot_html=plot_path,
        bins_vcf=bins_vcf,
        segments_vcf=segments_vcf,
        calls_vcf=calls_vcf,
        bins=bins,
        segments=segments,
        qc_json_path=qc_json_path,
        prenatal_result=prenatal_result,
    )


def _encode_b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def write_genome_plot(
    path: Path,
    bins: tuple[BinResult, ...],
    segments: tuple[SegmentResult, ...],
    contig_lengths: dict[str, int],
    *,
    cytobands: Path | Sequence[Cytoband] | None = None,
    z_threshold: float = 3.0,
    ratio_threshold: float | None = None,
    sample_name: str | None = None,
    prenatal_result: FetalFractionResult | None = None,
) -> None:
    """Write a minimal interactive Plotly HTML viewer with binary-packed bins."""

    parsed_bands: tuple[Cytoband, ...] = ()
    if isinstance(cytobands, Path):
        parsed_bands = read_cytobands(cytobands, contig_lengths)
    elif cytobands is not None:
        parsed_bands = tuple(cytobands)

    raw_contigs = dict.fromkeys(
        [contig for result in bins for contig in result.contigs]
        + [item.contig for item in segments]
        + [b.contig for b in parsed_bands]
    )
    ordered = tuple(
        sorted(
            (c for c in raw_contigs if c in contig_lengths and contig_lengths[c] > 0),
            key=natural_sort_key,
        )
    )
    contig_to_idx = {c: i for i, c in enumerate(ordered)}

    bin_c_list: list[np.ndarray] = []
    bin_s_list: list[np.ndarray] = []
    bin_e_list: list[np.ndarray] = []
    bin_r_list: list[np.ndarray] = []
    bin_z_list: list[np.ndarray] = []

    for result in bins:
        valid = (
            ~result.masked
            & np.isfinite(result.ratios)
            & np.array([c in contig_to_idx for c in result.contigs], dtype=bool)
        )
        if not np.any(valid):
            continue
        valid_idx = np.flatnonzero(valid)
        c_idx = np.array(
            [contig_to_idx[result.contigs[i]] for i in valid_idx], dtype="<u2"
        )
        bin_c_list.append(c_idx)
        bin_s_list.append(result.starts[valid_idx].astype("<i4"))
        bin_e_list.append(result.ends[valid_idx].astype("<i4"))
        bin_r_list.append(result.ratios[valid_idx].astype("<f4"))
        z_arr = np.nan_to_num(result.zscores[valid_idx], nan=0.0).astype("<f4")
        bin_z_list.append(z_arr)

    if bin_c_list:
        c_arr = np.concatenate(bin_c_list)
        s_arr = np.concatenate(bin_s_list)
        e_arr = np.concatenate(bin_e_list)
        r_arr = np.concatenate(bin_r_list)
        z_arr = np.concatenate(bin_z_list)
    else:
        c_arr = np.empty(0, dtype="<u2")
        s_arr = np.empty(0, dtype="<i4")
        e_arr = np.empty(0, dtype="<i4")
        r_arr = np.empty(0, dtype="<f4")
        z_arr = np.empty(0, dtype="<f4")

    if prenatal_result is not None:
        seg_rows: list[list[object]] = [
            [
                contig_to_idx[s.contig],
                int(s.start),
                int(s.end),
                round(float(s.ratio), 5) if np.isfinite(s.ratio) else 0.0,
                round(float(s.zscore), 4) if np.isfinite(s.zscore) else 0.0,
                int(s.bin_count),
                bool(s.discordant),
                round(float(s.implied_ff), 4)
                if s.implied_ff is not None and np.isfinite(s.implied_ff)
                else None,
                round(float(s.expected_ratio), 5)
                if s.expected_ratio is not None and np.isfinite(s.expected_ratio)
                else None,
                s.discordance_reason,
            ]
            for s in segments
            if s.contig in contig_to_idx and np.isfinite(s.ratio)
        ]
    else:
        seg_rows = [
            [
                contig_to_idx[s.contig],
                int(s.start),
                int(s.end),
                round(float(s.ratio), 5) if np.isfinite(s.ratio) else 0.0,
                round(float(s.zscore), 4) if np.isfinite(s.zscore) else 0.0,
                int(s.bin_count),
            ]
            for s in segments
            if s.contig in contig_to_idx and np.isfinite(s.ratio)
        ]

    band_rows = [
        [contig_to_idx[b.contig], int(b.start), int(b.end), b.name, b.stain]
        for b in parsed_bands
        if b.contig in contig_to_idx
    ]

    title = sample_name or path.stem.removesuffix("_plot")
    payload: dict[str, object] = {
        "sample": title,
        "contigs": list(ordered),
        "lengths": [int(contig_lengths[c]) for c in ordered],
        "z_thresh": float(z_threshold),
        "r_thresh": float(ratio_threshold) if ratio_threshold is not None else 0.2,
        "n_bins": int(c_arr.size),
        "bins": {
            "c": _encode_b64(c_arr),
            "s": _encode_b64(s_arr),
            "e": _encode_b64(e_arr),
            "r": _encode_b64(r_arr),
            "z": _encode_b64(z_arr),
        },
        "segments": seg_rows,
        "cytobands": band_rows,
    }
    if prenatal_result is not None:
        payload["prenatal"] = {
            "fetal_sex": prenatal_result.fetal_sex,
            "active_ff": float(prenatal_result.active_ff),
            "active_ff_source": prenatal_result.active_ff_source,
            "ffy": (
                float(prenatal_result.ffy) if prenatal_result.ffy is not None else None
            ),
            "ffx": (
                float(prenatal_result.ffx) if prenatal_result.ffx is not None else None
            ),
            "preface": (
                float(prenatal_result.preface)
                if prenatal_result.preface is not None
                else None
            ),
            "sca_call": prenatal_result.sca.call,
            "is_aneuploid": bool(prenatal_result.sca.is_aneuploid),
            "low_ff_warning": bool(prenatal_result.low_ff_warning),
            "warnings": list(prenatal_result.warnings),
        }
    payload_json = json.dumps(payload, separators=(",", ":"))
    template_source = (
        files("ccnv.templates")
        .joinpath("genome_plot.html.j2")
        .read_text(encoding="utf-8")
    )
    template = _JINJA_ENV.from_string(template_source)
    html = template.render(
        title=title,
        plotly_cdn_url=PLOTLY_CDN_URL,
        payload_json=payload_json,
    )
    path.write_text(html, encoding="utf-8")
    logger.debug(
        "wrote interactive genome HTML plot %s "
        "(%.1f KB, %d bins, %d segments, %d cytobands)",
        path,
        path.stat().st_size / 1024,
        int(c_arr.size),
        len(seg_rows),
        len(band_rows),
    )


def _format_bed_float(value: float) -> str:
    if not np.isfinite(value):
        return "NaN"
    return f"{value:.8g}"


def _vcf_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _write_compressed_bed(
    gz_path: Path, header: tuple[str, ...], rows: Iterable[tuple[object, ...]]
) -> Path:
    with pysam.BGZFile(str(gz_path), "wb") as bgz:
        bgz.write(("\t".join(header) + "\n").encode("utf-8"))
        for row in rows:
            bgz.write(("\t".join(str(value) for value in row) + "\n").encode("utf-8"))
    pysam.tabix_index(str(gz_path), preset="bed", force=True)
    return gz_path


def _build_vcf_header(
    contig_lengths: dict[str, int],
    sample_name: str,
    record_type: Literal["bins", "segments"],
    extra_contigs: Iterable[str] = (),
    prenatal_result: FetalFractionResult | None = None,
) -> pysam.VariantHeader:
    header = pysam.VariantHeader()
    header.add_line("##source=ccnv_predict")
    all_contigs = set(contig_lengths.keys()) | set(extra_contigs)
    for contig in sorted(all_contigs, key=natural_sort_key):
        length = contig_lengths.get(contig)
        if length is not None:
            header.contigs.add(contig, length=length)
        else:
            header.contigs.add(contig)

    if record_type == "bins":
        header.add_line('##ALT=<ID=CNV,Description="Copy number variation">')
        header.add_line(
            "##INFO=<ID=END,Number=1,Type=Integer,"
            'Description="End coordinate of structural variant">'
        )
        header.add_line(
            "##INFO=<ID=SVTYPE,Number=1,Type=String,"
            'Description="Type of structural variant">'
        )
        header.add_line(
            '##INFO=<ID=BRANCH,Number=1,Type=String,Description="Model branch">'
        )
        header.add_line(
            "##FILTER=<ID=MASKED,"
            'Description="Bin masked due to low coverage or quality filters">'
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
        header.add_line(
            "##FORMAT=<ID=REF_SIZE,Number=1,Type=Integer,"
            'Description="Number of within-sample reference bins used">'
        )
    else:
        header.add_line('##ALT=<ID=DUP,Description="Duplication / copy number gain">')
        header.add_line('##ALT=<ID=DEL,Description="Deletion / copy number loss">')
        header.add_line('##ALT=<ID=CNV,Description="Copy number variation">')
        header.add_line(
            "##INFO=<ID=END,Number=1,Type=Integer,"
            'Description="End coordinate of structural variant">'
        )
        header.add_line(
            "##INFO=<ID=SVTYPE,Number=1,Type=String,"
            'Description="Type of structural variant">'
        )
        header.add_line(
            "##INFO=<ID=IMPRECISE,Number=0,Type=Flag,"
            'Description="Imprecise structural variation">'
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

    header.add_sample(sample_name)
    return header


def _write_bins_vcf(
    gz_path: Path,
    header: pysam.VariantHeader,
    results: tuple[BinResult, ...],
    sample_name: str,
) -> Path:
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
                sample["LOG2R"] = _vcf_float(float(result.ratios[index]))
                sample["ZSCORE"] = _vcf_float(float(result.zscores[index]))
                sample["WEIGHT"] = _vcf_float(float(result.weights[index]))
                sample["REF_SIZE"] = int(result.reference_sizes[index])
                vf.write(rec)
    pysam.tabix_index(str(gz_path), preset="vcf", force=True)
    return gz_path


def _write_segments_vcf(
    gz_path: Path,
    header: pysam.VariantHeader,
    results: tuple[SegmentResult, ...],
    sample_name: str,
    *,
    calls_only: bool = False,
    is_prenatal: bool = False,
) -> Path:
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
            if is_prenatal and item.discordant:
                rec.info["DISCORDANT"] = True
                if item.discordance_reason:
                    rec.info["DISC_REASON"] = item.discordance_reason
            sample = rec.samples[sample_name]
            sample["LOG2R"] = _vcf_float(item.ratio)
            sample["ZSCORE"] = _vcf_float(item.zscore)
            sample["WEIGHT"] = _vcf_float(item.weight)
            sample["BINS"] = int(item.bin_count)
            if is_prenatal and item.discordant:
                if item.expected_ratio is not None:
                    sample["EXP_LOG2R"] = _vcf_float(item.expected_ratio)
                if item.implied_ff is not None:
                    sample["IMPLIED_FF"] = _vcf_float(item.implied_ff)
            vf.write(rec)
    pysam.tabix_index(str(gz_path), preset="vcf", force=True)
    return gz_path


def _bin_bed_rows(results: tuple[BinResult, ...]) -> tuple[tuple[object, ...], ...]:
    rows: list[tuple[object, ...]] = []
    for result in results:
        for index in range(len(result.contigs)):
            contig = result.contigs[index]
            start = int(result.starts[index])
            end = int(result.ends[index])
            rows.append(
                (
                    contig,
                    start,
                    end,
                    f"{contig}:{start}-{end}",
                    _format_bed_float(float(result.ratios[index])),
                    _format_bed_float(float(result.zscores[index])),
                    _format_bed_float(float(result.weights[index])),
                    int(result.reference_sizes[index]),
                    "true" if bool(result.masked[index]) else "false",
                    result.branch,
                )
            )
    return tuple(rows)


def _segment_bed_rows(
    results: tuple[SegmentResult, ...],
    *,
    is_prenatal: bool = False,
) -> tuple[tuple[object, ...], ...]:
    if is_prenatal:
        return tuple(
            (
                item.contig,
                item.start,
                item.end,
                _format_bed_float(item.ratio),
                _format_bed_float(item.zscore),
                _format_bed_float(item.weight),
                item.bin_count,
                item.call or "",
                (
                    _format_bed_float(item.expected_ratio)
                    if item.expected_ratio is not None
                    else ""
                ),
                (
                    _format_bed_float(item.implied_ff)
                    if item.implied_ff is not None
                    else ""
                ),
                "true" if item.discordant else "false",
                item.discordance_reason or "",
            )
            for item in results
        )
    return tuple(
        (
            item.contig,
            item.start,
            item.end,
            _format_bed_float(item.ratio),
            _format_bed_float(item.zscore),
            _format_bed_float(item.weight),
            item.bin_count,
            item.call or "",
        )
        for item in results
    )


def _calls_bed_rows(
    calls: Sequence[SegmentResult],
    *,
    is_prenatal: bool = False,
) -> tuple[tuple[object, ...], ...]:
    if is_prenatal:
        return tuple(
            (
                item.contig,
                item.start,
                item.end,
                _format_bed_float(item.ratio),
                _format_bed_float(item.zscore),
                item.call,
                (
                    _format_bed_float(item.expected_ratio)
                    if item.expected_ratio is not None
                    else ""
                ),
                (
                    _format_bed_float(item.implied_ff)
                    if item.implied_ff is not None
                    else ""
                ),
                "true" if item.discordant else "false",
                item.discordance_reason or "",
            )
            for item in calls
        )
    return tuple(
        (
            item.contig,
            item.start,
            item.end,
            _format_bed_float(item.ratio),
            _format_bed_float(item.zscore),
            item.call,
        )
        for item in calls
    )
