"""Deterministic 1D and 2D germline sex karyotyping and outlier quarantine."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
from numpy.typing import NDArray

from ccnv.features.binning import compute_bin_exposure
from ccnv.models import (
    BRANCH_FEMALE_SEX,
    BRANCH_MALE_SEX,
    InvalidInputError,
    PredictError,
    ReferenceError,
    ReferenceMetadata,
    SampleBuffers,
    is_autosome_name,
    normalize_chr,
    reconstruct_sample_grid,
)
from ccnv.stats import mad_scale

if TYPE_CHECKING:
    from ccnv.features.cohort import Cohort

logger = logging.getLogger(__name__)
SexLabel = Literal["female", "male"]

_SEPARATION_SIGMA_MULTIPLIER: Final[float] = 4.0 / 1.4826
_QUARANTINE_SIGMA_MULTIPLIER: Final[float] = 4.5
_HUMAN_PAR_MIN_CONTIG_BP: Final[int] = 10_000_000
_HUMAN_PAR1_END_BP: Final[int] = 3_000_000
_HUMAN_PAR2_TAIL_BP: Final[int] = 1_000_000
_HUMAN_Y_XTR_START_BP: Final[int] = 2_900_000
_HUMAN_Y_XTR_END_BP: Final[int] = 6_600_000
_HUMAN_X_XTR_MIN_CONTIG_BP: Final[int] = 100_000_000
_HUMAN_X_XTR_START_BP: Final[int] = 88_400_000
_HUMAN_X_XTR_END_BP: Final[int] = 93_300_000

__all__ = [
    "SexInference",
    "SexLabel",
    "infer_germline_sex_2d",
    "infer_sample_sexes",
    "infer_single_sample_sex",
    "par_xtr_mask",
]


@dataclass(frozen=True, slots=True)
class SexInference:
    """Deterministic sex assignments, signal cutoff, and outlier quarantine."""

    labels: tuple[SexLabel, ...] | None
    cutoff: float | None
    fractions: tuple[float, ...]
    passed: bool
    reason: str
    quarantined_mask: NDArray[np.bool_] = field(
        default_factory=lambda: np.zeros(0, dtype=np.bool_)
    )


def par_xtr_mask(
    on_contig: NDArray[np.bool_],
    starts: NDArray[np.integer],
    ends: NDArray[np.integer],
    contig_length: int,
    *,
    contig_name: str | None = None,
) -> NDArray[np.bool_]:
    """Exclude human-scale PAR1, PAR2, and X/Y XTR homology intervals while preserving short contigs."""
    if contig_length >= _HUMAN_PAR_MIN_CONTIG_BP:
        interior = (
            on_contig
            & (starts >= _HUMAN_PAR1_END_BP)
            & (ends <= contig_length - _HUMAN_PAR2_TAIL_BP)
        )
        if contig_name is not None:
            norm = normalize_chr(contig_name)
            if norm == "Y":
                interior = interior & ~(
                    (starts < _HUMAN_Y_XTR_END_BP) & (ends > _HUMAN_Y_XTR_START_BP)
                )
            elif norm == "X" and contig_length >= _HUMAN_X_XTR_MIN_CONTIG_BP:
                interior = interior & ~(
                    (starts < _HUMAN_X_XTR_END_BP) & (ends > _HUMAN_X_XTR_START_BP)
                )
        if np.any(interior):
            return interior
    return on_contig


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
            np.clip(mad_scale(d_x[core]), 0.02, 0.12)
        )
        sy = _QUARANTINE_SIGMA_MULTIPLIER * float(
            np.clip(mad_scale(d_y[core]), 0.02, 0.12)
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


def _calibrate_labeled_cutoff(
    labels: tuple[SexLabel, ...],
    q_mask: NDArray[np.bool_],
    y_fracs: NDArray[np.float64],
    exp_y: float,
    denom_exp: float,
) -> float:
    lbl_arr = np.asarray(labels)
    cf_clean = (lbl_arr == "female") & ~q_mask
    cm_clean = (lbl_arr == "male") & ~q_mask
    cf, cm = (
        (cf_clean, cm_clean)
        if (np.any(cf_clean) or np.any(cm_clean))
        else (lbl_arr == "female", lbl_arr == "male")
    )
    if np.any(cf) and np.any(cm):
        max_f = float(np.max(y_fracs[cf]))
        min_m = float(np.min(y_fracs[cm]))
        return (
            0.5 * (max_f + min_m)
            if max_f < min_m
            else 0.5 * (float(np.median(y_fracs[cf])) + float(np.median(y_fracs[cm])))
        )
    if np.any(cf):
        exp_hap = 0.5 * exp_y / max(denom_exp, 1.0)
        max_f = float(np.max(y_fracs[cf]))
        return float(
            np.clip(
                0.5 * (max_f + max(exp_hap, max_f * 4.0 + 1e-4)),
                1e-6,
                0.999999,
            )
        )
    return float(np.clip(0.5 * float(np.min(y_fracs[cm])), 1e-6, 0.999999))


def infer_sample_sexes(
    fractions: NDArray[np.floating],
    min_group_size: int,
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
    spread = max(mad_scale(low), mad_scale(high), 1e-12)
    cutoff_val = float((low[-1] + high[0]) / 2.0)
    if float(gaps[best_rel]) < _SEPARATION_SIGMA_MULTIPLIER * spread:
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
    """Infer germline sex in 2D non-PAR/XTR (d_X, d_Y) ploidy space with 4.5-sigma ellipse quarantine."""
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

    y_clean = par_xtr_mask(
        c_arr == y_name,
        cohort.starts,
        cohort.ends,
        cohort.contig_lengths[y_name],
        contig_name=y_name,
    )
    exp_y_clean = float(np.sum(exposure[y_clean]))
    y_usable = y_clean & usable_arr
    y_mask = (
        y_usable
        if exp_y_clean > 0.0 and float(np.sum(exposure[y_usable])) >= 0.20 * exp_y_clean
        else y_clean
    )
    denom_mask = usable_arr | y_mask
    denom_totals = _sum_counts_by_sample(cohort.counts, denom_mask)
    denom_safe = np.where(denom_totals > 0.0, denom_totals, lib_totals)
    y_fracs = _sum_counts_by_sample(cohort.counts, y_mask) / denom_safe
    recorded = tuple(float(v) for v in y_fracs)
    exp_y = max(float(np.sum(exposure[y_mask])), 1e-6)
    raw_dy = 2.0 * (_sum_counts_by_sample(cohort.counts, y_mask) / exp_y) / rate_auto

    has_x = len(x_names) > 0
    if has_x:
        x_clean = np.zeros(n_bins, dtype=np.bool_)
        for xn in x_names:
            x_clean |= par_xtr_mask(
                c_arr == xn,
                cohort.starts,
                cohort.ends,
                cohort.contig_lengths[xn],
                contig_name=xn,
            )
        x_mask = x_clean & usable_arr if np.any(x_clean & usable_arr) else x_clean
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

    all_explicit = (
        explicit_labels is not None
        and len(explicit_labels) == n_samples
        and all(x in ("female", "male") for x in explicit_labels)
    )
    if cutoff is not None:
        if not math.isfinite(cutoff) or not (0.0 <= cutoff <= 1.0):
            raise ReferenceError("sex_cutoff must be a finite number in [0, 1]")
        raw_labels: Sequence[SexLabel] = [
            "male" if float(v) > cutoff else "female" for v in y_fracs
        ]
        cut_val: float | None = float(cutoff)
        reason = "cutoff"
        apply_q = True
    elif all_explicit and explicit_labels is not None:
        raw_labels = ["male" if x == "male" else "female" for x in explicit_labels]
        cut_val = None
        reason = "samplesheet"
        apply_q = True
    elif (
        has_f
        and has_m
        and int(np.count_nonzero(in_f)) >= min_group_size
        and int(np.count_nonzero(in_m)) >= min_group_size
    ):
        cut_val = 0.5 * (float(np.max(y_fracs[in_f])) + float(np.min(y_fracs[in_m])))
        raw_labels = [
            "female"
            if bool(in_f[i])
            else ("male" if bool(in_m[i]) or float(y_fracs[i]) > cut_val else "female")
            for i in range(n_samples)
        ]
        reason = "separated_2d"
        apply_q = True
    else:
        base = infer_sample_sexes(y_fracs, min_group_size=min_group_size)
        if not base.passed or base.labels is None:
            return SexInference(
                None, base.cutoff, recorded, False, base.reason, empty_q
            )
        raw_labels = base.labels
        cut_val = base.cutoff
        reason = base.reason
        apply_q = False

    merged = _merge_explicit_sexes(raw_labels, explicit_labels)
    q_mask = (
        _quarantine_for_labels(
            merged,
            in_f,
            in_m,
            f_core,
            cand_xy,
            has_x=has_x,
            has_f=has_f,
            has_m=has_m,
        )
        if apply_q
        else empty_q
    )
    if cut_val is None:
        cut_val = _calibrate_labeled_cutoff(
            merged,
            q_mask,
            y_fracs,
            exp_y,
            float(np.sum(exposure[denom_mask])),
        )
    return SexInference(merged, cut_val, recorded, True, reason, q_mask)


def infer_single_sample_sex(
    sample: SampleBuffers,
    ref_meta: ReferenceMetadata,
    sex: SexLabel | None = None,
    unmasked_sample_bins: NDArray[np.bool_] | None = None,
) -> SexLabel | None:
    """Infer single-sample sex from non-PAR/XTR signal coverage over unmasked reference-aligned bins."""
    if not isinstance(sample, SampleBuffers):
        raise PredictError("sample must be a SampleBuffers instance")
    if not isinstance(ref_meta, ReferenceMetadata):
        raise PredictError("ref_meta must be a ReferenceMetadata instance")
    if sex is not None:
        if sex not in ("female", "male"):
            raise PredictError(
                f"invalid sex label {sex!r}; expected 'female' or 'male'"
            )
        return sex

    if not ({BRANCH_FEMALE_SEX, BRANCH_MALE_SEX} & set(ref_meta.branches)):
        return None
    signal = ref_meta.sex_signal_contig
    cutoff = ref_meta.sex_cutoff
    if not signal or cutoff is None or not math.isfinite(cutoff):
        raise PredictError("cannot infer sample sex; pass --sex female or --sex male")

    counts = np.asarray(sample.count, dtype=np.float64)
    total = float(np.sum(counts))
    if total <= 0.0:
        raise PredictError("sample has zero total coverage")

    codes, names, starts_u32, ends_u32 = reconstruct_sample_grid(sample.meta)
    sig_len = sample.meta.contig_lengths.get(
        signal, ref_meta.contig_lengths.get(signal, 0)
    )
    y_clean = (
        par_xtr_mask(
            codes == names.index(signal),
            starts_u32,
            ends_u32,
            sig_len,
            contig_name=signal,
        )
        if signal in names
        else np.zeros(counts.size, dtype=np.bool_)
    )

    if unmasked_sample_bins is not None:
        unmasked_arr = np.asarray(unmasked_sample_bins, dtype=np.bool_)
        if unmasked_arr.shape != (counts.size,):
            raise PredictError(
                "unmasked_sample_bins length must match sample bin count"
            )
        y_usable = y_clean & unmasked_arr
        n_y_clean = int(np.count_nonzero(y_clean))
        y_mask = (
            y_usable
            if (
                n_y_clean > 0
                and int(np.count_nonzero(y_usable)) >= max(1, int(0.20 * n_y_clean))
            )
            else y_clean
        )
        denom = float(np.sum(counts[unmasked_arr | y_mask]))
        if denom <= 0.0:
            raise PredictError("sample has zero coverage on unmasked reference bins")
    else:
        y_mask = y_clean
        denom = total

    fraction = float(np.sum(counts[y_mask])) / denom
    resolved: SexLabel = "male" if fraction > float(cutoff) else "female"
    logger.debug(
        "inferred sample sex as %s (signal %s fraction=%.4f vs cutoff=%.4f)",
        resolved,
        signal,
        fraction,
        float(cutoff),
    )
    return resolved
