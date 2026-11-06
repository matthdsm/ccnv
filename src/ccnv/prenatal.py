"""Prenatal model training, inference, SCA calling, and annotation (stub)."""

from __future__ import annotations

from ccnv.features.prenatal import (
    annotate_prenatal_segments,
    classify_sca,
    fit_prenatal_models,
    predict_prenatal,
)

__all__ = [
    "annotate_prenatal_segments",
    "classify_sca",
    "fit_prenatal_models",
    "predict_prenatal",
]
