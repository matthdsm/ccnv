"""Copy-number variation analysis for shallow whole-genome sequencing."""

from __future__ import annotations

from ccnv.extract import ExtractError, run_extract
from ccnv.models import (
    ExtractConfig,
    PredictConfig,
    ReadFilters,
    ReferenceConfig,
    ReferenceQCConfig,
)
from ccnv.predict import PredictError, run_predict
from ccnv.ref_qc import ReferenceQCError, run_reference_qc
from ccnv.reference import ReferenceError, run_reference

__version__ = "0.1.0dev"

__all__ = [
    "ExtractConfig",
    "ExtractError",
    "PredictConfig",
    "PredictError",
    "ReadFilters",
    "ReferenceConfig",
    "ReferenceError",
    "ReferenceQCConfig",
    "ReferenceQCError",
    "__version__",
    "run_extract",
    "run_predict",
    "run_reference",
    "run_reference_qc",
]
