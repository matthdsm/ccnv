"""Interactive Plotly HTML generator and cytoband ideogram renderers."""

import base64
import json
import logging
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from typing import Final

import jinja2
import numpy as np

from ccnv.models import (
    BinResult,
    Cytoband,
    FetalFractionResult,
    SegmentResult,
    natural_sort_key,
)

logger = logging.getLogger(__name__)

PLOTLY_CDN_URL: Final[str] = "https://cdn.plot.ly/plotly-2.35.2.min.js"
_JINJA_ENV: Final[jinja2.Environment] = jinja2.Environment(
    autoescape=jinja2.select_autoescape(["html", "j2"]),
    keep_trailing_newline=True,
)

__all__ = [
    "PLOTLY_CDN_URL",
    "encode_b64",
    "render_genome_plot",
    "write_genome_plot",
]


def encode_b64(array: np.ndarray) -> str:
    """Encode a contiguous numpy array as base64 ascii string."""
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


_encode_b64 = encode_b64


def _load_template_source() -> str:
    """Load the genome plot Jinja2 template with transitional fallback."""
    try:
        res = files("ccnv.report.templates").joinpath("genome_plot.html.j2")
        if res.is_file():
            return res.read_text(encoding="utf-8")
    except ModuleNotFoundError, TypeError, FileNotFoundError:
        pass
    res = files("ccnv.templates").joinpath("genome_plot.html.j2")
    return res.read_text(encoding="utf-8")


def render_genome_plot(
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
    """Write an interactive Plotly HTML viewer with binary-packed bins and segments."""
    parsed_bands: tuple[Cytoband, ...] = ()
    if isinstance(cytobands, Path):
        from ccnv.io.tracks import load_cytobands_track

        parsed_bands = load_cytobands_track(cytobands, contig_lengths)
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

    seg_rows: list[list[object]] = []
    for s in segments:
        if s.contig not in contig_to_idx or not np.isfinite(s.ratio):
            continue
        row: list[object] = [
            contig_to_idx[s.contig],
            int(s.start),
            int(s.end),
            round(float(s.ratio), 5),
            round(float(s.zscore), 4) if np.isfinite(s.zscore) else 0.0,
            int(s.bin_count),
        ]
        if prenatal_result is not None:
            row.extend(
                [
                    bool(s.discordant),
                    (
                        round(float(s.implied_ff), 4)
                        if s.implied_ff is not None and np.isfinite(s.implied_ff)
                        else None
                    ),
                    (
                        round(float(s.expected_ratio), 5)
                        if (
                            s.expected_ratio is not None
                            and np.isfinite(s.expected_ratio)
                        )
                        else None
                    ),
                    s.discordance_reason,
                ]
            )
        seg_rows.append(row)

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
            "c": encode_b64(c_arr),
            "s": encode_b64(s_arr),
            "e": encode_b64(e_arr),
            "r": encode_b64(r_arr),
            "z": encode_b64(z_arr),
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
            "ffx": float(prenatal_result.ffx),
            "preface": float(prenatal_result.preface),
            "sca_call": prenatal_result.sca.call,
            "is_aneuploid": bool(prenatal_result.sca.is_aneuploid),
            "low_ff_warning": bool(prenatal_result.low_ff_warning),
            "warnings": list(prenatal_result.warnings),
        }
    payload_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    template_source = _load_template_source()
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


write_genome_plot = render_genome_plot
