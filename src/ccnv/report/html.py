"""Interactive Plotly WebGL HTML generator with M4 min/max bin decimation."""

from __future__ import annotations

import base64
import functools
import json
import logging
import os
import tempfile
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from typing import Final

import jinja2
import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from ccnv.models import (
    BinResult,
    Cytoband,
    FetalFractionResult,
    SegmentResult,
    natural_sort_key,
)

logger = logging.getLogger(__name__)

PLOTLY_CDN_URL: Final[str] = "https://cdn.plot.ly/plotly-2.35.2.min.js"
DEFAULT_MAX_PLOT_BINS: Final[int] = 48_000
_JINJA_ENV: Final[jinja2.Environment] = jinja2.Environment(
    autoescape=True,
    keep_trailing_newline=True,
)

__all__ = [
    "DEFAULT_MAX_PLOT_BINS",
    "PLOTLY_CDN_URL",
    "decimate_bins_m4",
    "encode_b64",
    "render_genome_plot",
]


def encode_b64(array: NDArray[np.generic]) -> str:
    """Encode a contiguous NumPy array as a base64 ASCII string."""
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


@functools.lru_cache(maxsize=1)
def _get_template() -> jinja2.Template:
    """Load and compile the genome plot Jinja2 template once per process."""
    res = files("ccnv.report.templates").joinpath("genome_plot.html.j2")
    return _JINJA_ENV.from_string(res.read_text(encoding="utf-8"))


def _first_per_bucket(
    hit_indices: NDArray[np.intp], bucket_num: NDArray[np.intp]
) -> NDArray[np.intp]:
    hit_buckets = bucket_num[hit_indices]
    first_mask = np.empty(hit_buckets.size, dtype=np.bool_)
    first_mask[0] = True
    first_mask[1:] = hit_buckets[1:] != hit_buckets[:-1]
    return hit_indices[first_mask]


def decimate_bins_m4(
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    z: NDArray[np.float32],
    *,
    max_points: int = DEFAULT_MAX_PLOT_BINS,
    z_preserve: float = 3.0,
) -> NDArray[np.int32]:
    """Select sorted unique M4 (first, argmin, argmax, last) + |z|-outlier indices."""
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    z_arr = np.asarray(z)
    if x_arr.ndim != 1 or y_arr.shape != x_arr.shape or z_arr.shape != x_arr.shape:
        raise ValueError("x, y, and z must be 1D arrays of identical length")
    if isinstance(max_points, bool) or max_points < 1:
        raise ValueError("max_points must be a positive integer")
    if not np.isfinite(z_preserve) or z_preserve < 0.0:
        raise ValueError("z_preserve must be a non-negative finite float")

    finite_x = np.flatnonzero(np.isfinite(x_arr))
    if finite_x.size == 0:
        return np.empty(0, dtype=np.int32)
    if finite_x.size <= max_points:
        return finite_x.astype(np.int32)

    fx = x_arr[finite_x].astype(np.float64, copy=False)
    fy = y_arr[finite_x]
    if bool(np.all(fx[1:] >= fx[:-1])):
        order = finite_x
        sx = fx
        sy = fy
    else:
        perm = np.argsort(fx, kind="stable")
        order = finite_x[perm]
        sx = fx[perm]
        sy = fy[perm]

    n_buckets = max(1, max_points // 4)
    x_min = float(sx[0])
    x_max = float(sx[-1])
    if x_max <= x_min:
        starts = np.array([0], dtype=np.intp)
        ends = np.array([sx.size - 1], dtype=np.intp)
        bucket_num = np.zeros(sx.size, dtype=np.intp)
    else:
        scale = n_buckets / (x_max - x_min)
        b_idx = np.clip(
            np.floor((sx - x_min) * scale).astype(np.int64), 0, n_buckets - 1
        )
        change = np.empty(b_idx.size, dtype=np.bool_)
        change[0] = True
        change[1:] = b_idx[1:] != b_idx[:-1]
        starts = np.flatnonzero(change)
        ends = np.empty_like(starts)
        ends[:-1] = starts[1:] - 1
        ends[-1] = sx.size - 1
        bucket_num = np.cumsum(change) - 1

    y_finite = np.isfinite(sy)
    y_min_in = np.where(y_finite, sy, np.inf)
    b_min = np.minimum.reduceat(y_min_in, starts)
    argmin_local = _first_per_bucket(
        np.flatnonzero(y_min_in == b_min[bucket_num]), bucket_num
    )

    y_max_in = np.where(y_finite, sy, -np.inf)
    b_max = np.maximum.reduceat(y_max_in, starts)
    argmax_local = _first_per_bucket(
        np.flatnonzero(y_max_in == b_max[bucket_num]), bucket_num
    )

    z_fin = z_arr[finite_x]
    outliers = finite_x[np.isfinite(z_fin) & (np.abs(z_fin) >= z_preserve)]
    combined = np.concatenate(
        [
            order[starts],
            order[argmin_local],
            order[argmax_local],
            order[ends],
            outliers,
        ]
    )
    return np.unique(combined).astype(np.int32)


def _map_contigs_to_indices(
    contigs: tuple[str, ...], contig_to_idx: dict[str, int]
) -> NDArray[np.int32]:
    if not contigs:
        return np.empty(0, dtype=np.int32)
    dict_arr = pa.array(contigs, type=pa.string()).dictionary_encode()
    dict_vals = dict_arr.dictionary.to_pylist()
    lut = np.array([contig_to_idx.get(c, -1) for c in dict_vals], dtype=np.int32)
    indices_np = dict_arr.indices.to_numpy(zero_copy_only=False)
    return lut[indices_np]


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(text)
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def _finite_or_none(val: float | None, ndigits: int | None = None) -> float | None:
    if val is None:
        return None
    f_val = float(val)
    if not np.isfinite(f_val):
        return None
    return round(f_val, ndigits) if ndigits is not None else f_val


def _finite_or_zero(val: float | None) -> float:
    if val is None:
        return 0.0
    f_val = float(val)
    return f_val if np.isfinite(f_val) else 0.0


def render_genome_plot(
    path: Path,
    bins: tuple[BinResult, ...],
    segments: tuple[SegmentResult, ...],
    contig_lengths: dict[str, int],
    *,
    cytobands: Sequence[Cytoband] | None = None,
    z_threshold: float = 3.0,
    ratio_threshold: float | None = None,
    sample_name: str | None = None,
    prenatal_result: FetalFractionResult | None = None,
    max_plot_bins: int = DEFAULT_MAX_PLOT_BINS,
) -> None:
    """Write an interactive Plotly WebGL HTML viewer with M4-decimated typed buffers."""
    parsed_bands: tuple[Cytoband, ...] = (
        tuple(cytobands) if cytobands is not None else ()
    )

    raw_contigs: dict[str, None] = {}
    for result in bins:
        for contig in dict.fromkeys(result.contigs):
            raw_contigs[contig] = None
    for item in segments:
        raw_contigs[item.contig] = None
    for band in parsed_bands:
        raw_contigs[band.contig] = None

    ordered = tuple(
        sorted(
            (c for c in raw_contigs if c in contig_lengths and contig_lengths[c] > 0),
            key=natural_sort_key,
        )
    )
    contig_to_idx = {c: i for i, c in enumerate(ordered)}

    bin_c_list: list[NDArray[np.uint16]] = []
    bin_s_list: list[NDArray[np.uint32]] = []
    bin_e_list: list[NDArray[np.uint32]] = []
    bin_r_list: list[NDArray[np.float32]] = []
    bin_z_list: list[NDArray[np.float32]] = []

    for result in bins:
        mapped_c = _map_contigs_to_indices(result.contigs, contig_to_idx)
        valid = ~result.masked & np.isfinite(result.ratios) & (mapped_c >= 0)
        if not np.any(valid):
            continue
        valid_idx = np.flatnonzero(valid)
        bin_c_list.append(mapped_c[valid_idx].astype("<u2"))
        bin_s_list.append(result.starts[valid_idx].astype("<u4"))
        bin_e_list.append(result.ends[valid_idx].astype("<u4"))
        bin_r_list.append(result.ratios[valid_idx].astype("<f4"))
        z_clean = np.nan_to_num(result.zscores[valid_idx], nan=0.0).astype("<f4")
        bin_z_list.append(z_clean)

    if bin_c_list:
        c_arr = np.concatenate(bin_c_list)
        s_arr = np.concatenate(bin_s_list)
        e_arr = np.concatenate(bin_e_list)
        r_arr = np.concatenate(bin_r_list)
        z_arr = np.concatenate(bin_z_list)
    else:
        c_arr = np.empty(0, dtype="<u2")
        s_arr = np.empty(0, dtype="<u4")
        e_arr = np.empty(0, dtype="<u4")
        r_arr = np.empty(0, dtype="<f4")
        z_arr = np.empty(0, dtype="<f4")

    total_valid_bins = int(c_arr.size)
    if total_valid_bins > max_plot_bins and ordered:
        lengths_f64 = np.array([contig_lengths[c] for c in ordered], dtype=np.float64)
        offsets_f64 = np.zeros(len(ordered), dtype=np.float64)
        if len(ordered) > 1:
            offsets_f64[1:] = np.cumsum(lengths_f64[:-1])
        x_f32 = (
            offsets_f64[c_arr]
            + 0.5 * (s_arr.astype(np.float64) + e_arr.astype(np.float64))
        ).astype(np.float32)
        z_keep = min(3.0, float(z_threshold)) if z_threshold > 0.0 else 3.0
        keep = decimate_bins_m4(
            x_f32, r_arr, z_arr, max_points=max_plot_bins, z_preserve=z_keep
        )
        c_arr = c_arr[keep].astype("<u2", copy=False)
        s_arr = s_arr[keep].astype("<u4", copy=False)
        e_arr = e_arr[keep].astype("<u4", copy=False)
        r_arr = r_arr[keep].astype("<f4", copy=False)
        z_arr = z_arr[keep].astype("<f4", copy=False)

    seg_rows: list[list[object]] = []
    segment_calls: list[str | None] = []
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
                    (
                        round(float(s.frag_short_ratio), 4)
                        if (
                            s.frag_short_ratio is not None
                            and np.isfinite(s.frag_short_ratio)
                        )
                        else None
                    ),
                    (
                        round(float(s.frag_llr_size), 3)
                        if (
                            s.frag_llr_size is not None and np.isfinite(s.frag_llr_size)
                        )
                        else None
                    ),
                    s.origin_call,
                ]
            )
        seg_rows.append(row)
        segment_calls.append(s.call)

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
        "z_thresh": _finite_or_zero(z_threshold),
        "r_thresh": _finite_or_zero(ratio_threshold),
        "n_bins": int(c_arr.size),
        "total_bins": total_valid_bins,
        "bins": {
            "c": encode_b64(c_arr),
            "s": encode_b64(s_arr),
            "e": encode_b64(e_arr),
            "r": encode_b64(r_arr),
            "z": encode_b64(z_arr),
        },
        "segments": seg_rows,
        "segment_calls": segment_calls,
        "cytobands": band_rows,
    }
    if prenatal_result is not None:
        raw_eff_ff = (
            prenatal_result.effective_ff
            if prenatal_result.effective_ff is not None
            else prenatal_result.active_ff
        )
        payload["prenatal"] = {
            "fetal_sex": prenatal_result.fetal_sex,
            "active_ff": _finite_or_none(prenatal_result.active_ff),
            "effective_ff": _finite_or_none(raw_eff_ff, 6),
            "active_ff_source": prenatal_result.active_ff_source,
            "ffy": _finite_or_none(prenatal_result.ffy),
            "ffx": _finite_or_none(prenatal_result.ffx),
            "ffa": _finite_or_none(prenatal_result.ffa),
            "ff_size": _finite_or_none(prenatal_result.ff_size),
            "sca_call": prenatal_result.sca.call,
            "is_aneuploid": bool(prenatal_result.sca.is_aneuploid),
            "low_ff_warning": bool(prenatal_result.low_ff_warning),
            "warnings": list(prenatal_result.warnings),
        }
    payload_json = (
        json.dumps(payload, separators=(",", ":"), allow_nan=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    html = _get_template().render(
        title=title,
        plotly_cdn_url=PLOTLY_CDN_URL,
        payload_json=payload_json,
    )
    _write_text_atomic(path, html)
    logger.debug(
        "wrote interactive genome HTML plot %s "
        "(%.1f KB, %d/%d bins, %d segments, %d cytobands)",
        path,
        path.stat().st_size / 1024,
        int(c_arr.size),
        total_valid_bins,
        len(seg_rows),
        len(band_rows),
    )
