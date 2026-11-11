"""Deterministic 1D and 2D germline sex karyotyping and outlier quarantine."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from ccnv.features.binning import compute_bin_exposure
from ccnv.models import (
    BRANCH_FEMALE_SEX,
    BRANCH_MALE_SEX,
    InvalidInputError,
    PredictError,
    ReferenceError,
    SampleBuffers,
    SampleMetadata,
    is_autosome_name,
    normalize_chr,
    reconstruct_sample_grid,
)

if TYPE_CHECKING:
    from ccnv.features.cohort import Cohort

logger = logging.getLogger(__name__)
SexLabel = Literal["female", "male"]

_SEPARATION_MAD_MULTIPLIER: Final[float] = 4.0
_QUARANTINE_SIGMA_MULTIPLIER: Final[float] = 4.5
_HUMAN_PAR_MIN_CONTIG_BP: Final[int] = 10_000_000
_HUMAN_PAR1_END_BP: Final[int] = 3_000_000
_HUMAN_PAR2_TAIL_BP: Final[int] = 1_000_000

__all__ = [
    "SexInference",
    "SexLabel",
    "infer_germline_sex_2d",
    "infer_sample_sexes",
    "infer_single_sample_sex",
    "nonpar_contig_mask",
]


@dataclass(frozen=True, slots=True)
class SexInference:
    """Deterministic sex assignments, 2D ploidy dosages, and outlier quarantine."""

    labels: tuple[SexLabel, ...] | None
    cutoff: float | None
    fractions: tuple[float, ...]
    passed: bool
    reason: str
    quarantined_mask: NDArray[np.bool_] = field(
        default_factory=lambda: np.zeros(0, dtype=np.bool_)
    )
    d_x: tuple[float, ...] = ()
    d_y: tuple[float, ...] = ()

    @property
    def sample_sexes(self) -> tuple[SexLabel, ...] | None:
        return self.labels

    @property
    def chry_cutoff(self) -> float | None:
        return self.cutoff


def nonpar_contig_mask(
    on_contig: NDArray[np.bool_],
    starts: NDArray[np.integer],
    ends: NDArray[np.integer],
    contig_length: int,
) -> NDArray[np.bool_]:
    """Exclude human-scale PAR1/PAR2 terminal intervals while preserving short contigs."""
    if contig_length >= _HUMAN_PAR_MIN_CONTIG_BP:
        interior = (
            on_contig
            & (starts >= _HUMAN_PAR1_END_BP)
            & (ends <= contig_length - _HUMAN_PAR2_TAIL_BP)
        )
        if np.any(interior):
            return interior
    return on_contig


def _mad(values: NDArray[np.floating]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.median(np.abs(arr - np.median(arr)))) if arr.size > 0 else 0.0


def _sum_counts_by_sample(
    counts: NDArray[np.floating] | NDArray[np.integer],
    row_mask: NDArray[np.bool_] | None = None,
    *,
    chunk_rows: int | None = None,
) -> NDArray[np.float64]:
    """Sum sample columns in bounded row slices without materializing boolean row copies."""
    n_bins, n_samples = counts.shape
    totals = np.zeros(n_samples, dtype=np.float64)
    if n_bins == 0 or n_samples == 0:
        return totals
    step = (
        chunk_rows
        if chunk_rows is not None and chunk_rows > 0
        else max(1024, (64 * 1024 * 1024) // max(n_samples, 1))
    )
    for r0 in range(0, n_bins, step):
        r1 = min(r0 + step, n_bins)
        if row_mask is None:
            totals += np.sum(counts[r0:r1], axis=0, dtype=np.float64)
        else:
            m_sub = row_mask[r0:r1]
            if np.any(m_sub):
                totals += np.sum(
                    counts[r0:r1],
                    axis=0,
                    dtype=np.float64,
                    where=m_sub[:, np.newaxis],
                )
    return totals


def _ellipse_quarantine_2d(
    d_x: NDArray[np.float64],
    d_y: NDArray[np.float64],
    f_core: NDArray[np.bool_],
    m_core: NDArray[np.bool_],
    *,
    has_f: bool = True,
    has_m: bool = True,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    def _dist2(core: NDArray[np.bool_]) -> NDArray[np.float64]:
        mx = float(np.median(d_x[core]))
        my = float(np.median(d_y[core]))
        sx = _QUARANTINE_SIGMA_MULTIPLIER * float(
            np.clip(1.4826 * _mad(d_x[core]), 0.02, 0.12)
        )
        sy = _QUARANTINE_SIGMA_MULTIPLIER * float(
            np.clip(1.4826 * _mad(d_y[core]), 0.02, 0.12)
        )
        return ((d_x - mx) / sx) ** 2 + ((d_y - my) / sy) ** 2

    inf_arr = np.full(d_x.shape, np.inf, dtype=np.float64)
    df = _dist2(f_core) if has_f else inf_arr
    dm = _dist2(m_core) if has_m else inf_arr
    return (df <= 1.0) & (df < dm), (dm <= 1.0) & (dm < df)


def _quarantine_for_labels(
    labels: Sequence[SexLabel],
    in_f: NDArray[np.bool_],
    in_m: NDArray[np.bool_],
    f_core: NDArray[np.bool_],
    cand_xy: NDArray[np.bool_],
    *,
    has_x: bool,
    has_f: bool,
    has_m: bool,
) -> NDArray[np.bool_]:
    if not (has_f or has_m):
        return np.zeros(len(labels), dtype=np.bool_)
    is_male = np.asarray([lbl == "male" for lbl in labels], dtype=np.bool_)
    ok_f = in_f if has_f else (f_core if has_x else ~is_male)
    ok_m = in_m if has_m else (cand_xy if has_x else is_male)
    return ~np.where(is_male, ok_m, ok_f)


def _merge_explicit_sexes(
    base: Sequence[SexLabel], explicit: Sequence[SexLabel | None] | None
) -> tuple[SexLabel, ...]:
    if explicit is not None and len(explicit) == len(base):
        return tuple(
            e if e is not None else b for b, e in zip(base, explicit, strict=True)
        )
    return tuple(base)


def infer_sample_sexes(
    fractions: NDArray[np.floating],
    min_group_size: int,
    min_separation: float = 0.0,
    cutoff: float | None = None,
) -> SexInference:
    """Split Y-like fractions into female/male groups when separation is clear."""
    values = np.asarray(fractions, dtype=np.float64)
    recorded = tuple(float(v) for v in values)
    empty_q = np.zeros(values.size, dtype=np.bool_)
    if min_group_size < 1:
        raise ReferenceError("min_sex_group_size must be a positive integer")
    if cutoff is not None:
        if not math.isfinite(cutoff) or not (0.0 <= cutoff <= 1.0):
            raise ReferenceError("sex_cutoff must be a finite number in [0, 1]")
        labels: tuple[SexLabel, ...] = tuple(
            "male" if float(v) > cutoff else "female" for v in values
        )
        return SexInference(labels, float(cutoff), recorded, True, "cutoff", empty_q)
    if values.size < 2 * min_group_size:
        return SexInference(
            None,
            None,
            recorded,
            False,
            "cohort is too small for sex inference",
            empty_q,
        )

    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    gaps = np.diff(sorted_vals)[min_group_size - 1 : values.size - min_group_size]
    if gaps.size == 0 or float(np.max(gaps)) <= 0:
        return SexInference(
            None,
            None,
            recorded,
            False,
            "sex-signal fractions are not separable",
            empty_q,
        )
    best_rel = int(np.argmax(gaps))
    best_split = min_group_size + best_rel
    low = sorted_vals[:best_split]
    high = sorted_vals[best_split:]
    spread = max(_mad(low), _mad(high), 1e-12)
    cutoff_val = float((low[-1] + high[0]) / 2.0)
    if float(gaps[best_rel]) < max(min_separation, _SEPARATION_MAD_MULTIPLIER * spread):
        return SexInference(
            None,
            cutoff_val,
            recorded,
            False,
            "sex-signal groups are not sufficiently separated",
            empty_q,
        )
    labels_list: list[SexLabel] = ["female"] * values.size
    for idx in order[best_split:]:
        labels_list[int(idx)] = "male"
    return SexInference(
        tuple(labels_list), cutoff_val, recorded, True, "separated", empty_q
    )


def infer_germline_sex_2d(
    cohort: Cohort,
    usable: NDArray[np.bool_],
    *,
    min_group_size: int = 5,
    signal_contig: str | None = None,
    cutoff: float | None = None,
    explicit_labels: Sequence[SexLabel | None] | None = None,
) -> SexInference:
    """Infer germline sex in 2D non-PAR (d_X, d_Y) ploidy space with 4.5-sigma ellipse quarantine."""
    if min_group_size < 1:
        raise ReferenceError("min_sex_group_size must be a positive integer")
    n_bins, n_samples = cohort.counts.shape
    usable_arr = np.asarray(usable, dtype=np.bool_)
    if usable_arr.shape != (n_bins,):
        raise InvalidInputError("usable length must match cohort bin count")
    empty_q = np.zeros(n_samples, dtype=np.bool_)
    lib_totals = _sum_counts_by_sample(cohort.counts)
    if np.any(lib_totals <= 0):
        raise ReferenceError("every sample must have a positive total count")

    ordered = tuple(dict.fromkeys(cohort.contigs))
    y_name = signal_contig or next(
        (c for c in ordered if normalize_chr(c) == "Y"), None
    )
    if y_name is None or y_name not in cohort.contig_lengths:
        return SexInference(None, None, (), False, "no sex-signal contig", empty_q)
    x_names = tuple(c for c in ordered if normalize_chr(c) == "X" and c != y_name)
    sex_set = {y_name, *x_names}
    auto_names = tuple(
        c for c in ordered if is_autosome_name(c) and c not in sex_set
    ) or tuple(c for c in ordered if c not in sex_set)
    if not auto_names:
        return SexInference(None, None, (), False, "no autosomal contigs", empty_q)

    exposure = compute_bin_exposure(cohort.starts, cohort.ends, cohort.bin_size).astype(
        np.float64
    )
    c_arr = np.asarray(cohort.contigs, dtype=object)
    auto_raw = np.isin(c_arr, auto_names)
    auto_mask = auto_raw & usable_arr if np.any(auto_raw & usable_arr) else auto_raw
    exp_auto = float(np.sum(exposure[auto_mask]))
    sum_auto = _sum_counts_by_sample(cohort.counts, auto_mask)
    if exp_auto <= 0.0 or np.any(sum_auto <= 0.0):
        return SexInference(None, None, (), False, "zero autosomal depth", empty_q)
    rate_auto = sum_auto / exp_auto

    y_nonpar = nonpar_contig_mask(
        c_arr == y_name, cohort.starts, cohort.ends, cohort.contig_lengths[y_name]
    )
    y_fracs = _sum_counts_by_sample(cohort.counts, y_nonpar) / lib_totals
    recorded = tuple(float(v) for v in y_fracs)
    exp_y_nonpar = float(np.sum(exposure[y_nonpar]))
    y_usable = y_nonpar & usable_arr
    y_mask = (
        y_usable
        if exp_y_nonpar > 0.0
        and float(np.sum(exposure[y_usable])) >= 0.20 * exp_y_nonpar
        else y_nonpar
    )
    exp_y = max(float(np.sum(exposure[y_mask])), 1e-6)
    raw_dy = 2.0 * (_sum_counts_by_sample(cohort.counts, y_mask) / exp_y) / rate_auto

    has_x = len(x_names) > 0
    if has_x:
        x_nonpar = np.zeros(n_bins, dtype=np.bool_)
        for xn in x_names:
            x_nonpar |= nonpar_contig_mask(
                c_arr == xn, cohort.starts, cohort.ends, cohort.contig_lengths[xn]
            )
        x_mask = x_nonpar & usable_arr if np.any(x_nonpar & usable_arr) else x_nonpar
        exp_x = max(float(np.sum(exposure[x_mask])), 1e-6)
        d_x = 2.0 * (_sum_counts_by_sample(cohort.counts, x_mask) / exp_x) / rate_auto
        cand_xy = (np.abs(d_x - 1.0) < 0.45) & (raw_dy > 0.20)
        y_scale = (
            float(np.median(raw_dy[cand_xy]))
            if np.count_nonzero(cand_xy) >= max(2, min_group_size)
            else 0.0
        )
        d_y = raw_dy / y_scale if y_scale > 0.20 else raw_dy
    else:
        d_x = np.ones(n_samples, dtype=np.float64)
        d_y = raw_dy
        cand_xy = empty_q

    dx_tup = tuple(float(v) for v in d_x)
    dy_tup = tuple(float(v) for v in d_y)
    f_core = (np.abs(d_x - 2.0) < 0.45) & (d_y < 0.25) if has_x else empty_q
    m_init = (np.abs(d_x - 1.0) < 0.45) & (d_y > 0.25) if has_x else empty_q
    med_my0 = (
        float(np.median(d_y[m_init]))
        if np.count_nonzero(m_init) >= min_group_size
        else 0.0
    )
    m_core = (
        m_init & (np.abs(d_y - med_my0) < 0.35 * max(med_my0, 0.5))
        if med_my0 > 0.0
        else empty_q
    )
    has_f = int(np.count_nonzero(f_core)) >= min_group_size
    has_m = int(np.count_nonzero(m_core)) >= min_group_size
    in_f, in_m = (
        _ellipse_quarantine_2d(d_x, d_y, f_core, m_core, has_f=has_f, has_m=has_m)
        if (has_f or has_m)
        else (empty_q, empty_q)
    )

    if cutoff is not None:
        if not math.isfinite(cutoff) or not (0.0 <= cutoff <= 1.0):
            raise ReferenceError("sex_cutoff must be a finite number in [0, 1]")
        base_lbls: list[SexLabel] = [
            "male" if float(v) > cutoff else "female" for v in y_fracs
        ]
        merged_cut = _merge_explicit_sexes(base_lbls, explicit_labels)
        q_mask = _quarantine_for_labels(
            merged_cut,
            in_f,
            in_m,
            f_core,
            cand_xy,
            has_x=has_x,
            has_f=has_f,
            has_m=has_m,
        )
        return SexInference(
            merged_cut,
            float(cutoff),
            recorded,
            True,
            "cutoff",
            q_mask,
            dx_tup,
            dy_tup,
        )

    if (
        explicit_labels is not None
        and len(explicit_labels) == n_samples
        and all(x in ("female", "male") for x in explicit_labels)
    ):
        lbl_tup: tuple[SexLabel, ...] = tuple(
            "male" if x == "male" else "female" for x in explicit_labels
        )
        q_mask = _quarantine_for_labels(
            lbl_tup,
            in_f,
            in_m,
            f_core,
            cand_xy,
            has_x=has_x,
            has_f=has_f,
            has_m=has_m,
        )
        lbl_arr = np.asarray(lbl_tup)
        cf = (
            (lbl_arr == "female") & ~q_mask
            if np.any((lbl_arr == "female") & ~q_mask)
            else (lbl_arr == "female")
        )
        cm = (
            (lbl_arr == "male") & ~q_mask
            if np.any((lbl_arr == "male") & ~q_mask)
            else (lbl_arr == "male")
        )
        if np.any(cf) and np.any(cm):
            max_f = float(np.max(y_fracs[cf]))
            min_m = float(np.min(y_fracs[cm]))
            cut_val = (
                0.5 * (max_f + min_m)
                if max_f < min_m
                else 0.5
                * (float(np.median(y_fracs[cf])) + float(np.median(y_fracs[cm])))
            )
        elif np.any(cf):
            exp_hap = 0.5 * exp_y / max(float(np.sum(exposure[usable_arr])), 1.0)
            max_f = float(np.max(y_fracs[cf]))
            cut_val = float(
                np.clip(
                    0.5 * (max_f + max(exp_hap, max_f * 2.0 + 1e-4)), 1e-6, 0.999999
                )
            )
        else:
            cut_val = float(np.clip(0.5 * float(np.min(y_fracs[cm])), 1e-6, 0.999999))
        return SexInference(
            lbl_tup, cut_val, recorded, True, "samplesheet", q_mask, dx_tup, dy_tup
        )

    if (
        has_f
        and has_m
        and int(np.count_nonzero(in_f)) >= min_group_size
        and int(np.count_nonzero(in_m)) >= min_group_size
    ):
        cut_val = 0.5 * (float(np.max(y_fracs[in_f])) + float(np.min(y_fracs[in_m])))
        inferred_2d: list[SexLabel] = [
            "female"
            if bool(in_f[i])
            else ("male" if bool(in_m[i]) or float(y_fracs[i]) > cut_val else "female")
            for i in range(n_samples)
        ]
        merged_2d = _merge_explicit_sexes(inferred_2d, explicit_labels)
        q_mask = _quarantine_for_labels(
            merged_2d,
            in_f,
            in_m,
            f_core,
            cand_xy,
            has_x=has_x,
            has_f=has_f,
            has_m=has_m,
        )
        return SexInference(
            merged_2d,
            cut_val,
            recorded,
            True,
            "separated_2d",
            q_mask,
            dx_tup,
            dy_tup,
        )

    base = infer_sample_sexes(y_fracs, min_group_size=min_group_size)
    if not base.passed or base.labels is None:
        return SexInference(
            None, base.cutoff, recorded, False, base.reason, empty_q, dx_tup, dy_tup
        )
    return SexInference(
        _merge_explicit_sexes(base.labels, explicit_labels),
        base.cutoff,
        recorded,
        True,
        base.reason,
        empty_q,
        dx_tup,
        dy_tup,
    )


def infer_single_sample_sex(
    sample: pa.Table | SampleBuffers,
    metadata: Mapping[str, object],
    branches: set[object],
) -> SexLabel | None:
    """Infer single-sample sex from non-PAR signal coverage against persisted reference cutoff."""
    if not ({BRANCH_FEMALE_SEX, BRANCH_MALE_SEX} & branches):
        return None
    model = metadata.get("model")
    if not isinstance(model, Mapping) or not isinstance(
        model.get("sex_signal_contig"), str
    ):
        raise PredictError("cannot infer sample sex; pass --sex female or --sex male")
    raw_inf = model.get("sex_inference")
    cutoff = model.get("sex_cutoff")
    if cutoff is None and isinstance(raw_inf, Mapping):
        cutoff = raw_inf.get("cutoff")
    if (
        isinstance(cutoff, bool)
        or not isinstance(cutoff, int | float)
        or not math.isfinite(float(cutoff))
    ):
        raise PredictError("cannot infer sample sex; pass --sex female or --sex male")
    counts = (
        np.asarray(sample.count, dtype=np.float64)
        if isinstance(sample, SampleBuffers)
        else np.asarray(
            sample.column("count").combine_chunks().to_numpy(zero_copy_only=False),
            dtype=np.float64,
        )
    )
    total = float(np.sum(counts))
    if total <= 0:
        raise PredictError("sample has zero total coverage")
    signal = str(model["sex_signal_contig"])
    raw_clen = metadata.get("contig_lengths")
    sig_len = (
        int(raw_clen[signal])
        if isinstance(raw_clen, Mapping) and isinstance(raw_clen.get(signal), int)
        else 0
    )
    if not isinstance(sample, SampleBuffers) and "contig" in sample.column_names:
        contigs = np.asarray(
            sample.column("contig").combine_chunks().to_numpy(zero_copy_only=False),
            dtype=object,
        )
        on_sig = contigs == signal
        if (
            "start" in sample.column_names
            and "end" in sample.column_names
            and np.any(on_sig)
        ):
            starts = np.asarray(
                sample.column("start").combine_chunks().to_numpy(zero_copy_only=False),
                dtype=np.int64,
            )
            ends = np.asarray(
                sample.column("end").combine_chunks().to_numpy(zero_copy_only=False),
                dtype=np.int64,
            )
            on_sig = nonpar_contig_mask(
                on_sig, starts, ends, sig_len or int(np.max(ends[on_sig]))
            )
    else:
        s_meta = (
            sample.meta
            if isinstance(sample, SampleBuffers)
            else SampleMetadata.from_schema(sample.schema)
        )
        codes, names, starts_u32, ends_u32 = reconstruct_sample_grid(s_meta)
        on_sig = (
            nonpar_contig_mask(
                codes == names.index(signal),
                starts_u32,
                ends_u32,
                s_meta.contig_lengths.get(signal, sig_len),
            )
            if signal in names
            else np.zeros(counts.size, dtype=np.bool_)
        )
    fraction = float(np.sum(counts[on_sig])) / total
    sex: SexLabel = "male" if fraction > float(cutoff) else "female"
    logger.debug(
        "inferred sample sex as %s (signal %s fraction=%.4f vs cutoff=%.4f)",
        sex,
        signal,
        fraction,
        float(cutoff),
    )
    return sex
