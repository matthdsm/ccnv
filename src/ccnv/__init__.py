"""Copy-number variation analysis for whole-genome sequencing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ccnv.models import (
    AlignmentError,
    ExtractConfig,
    ExtractError,
    PredictConfig,
    PredictError,
    ReadFilters,
    ReferenceConfig,
    ReferenceError,
    ReferenceQCConfig,
    ReferenceQCError,
)

if TYPE_CHECKING:
    from ccnv.extract import run_extract as run_extract
    from ccnv.predict import run_predict as run_predict
    from ccnv.ref_qc import run_reference_qc as run_reference_qc
    from ccnv.reference import run_reference as run_reference

__version__ = "1.0.0"

__all__ = [
    "AlignmentError",
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


def __getattr__(name: str) -> object:
    """Lazily resolve top-level pipeline orchestrators on demand."""
    if name == "run_extract":
        from ccnv.extract import run_extract

        return run_extract
    if name == "run_predict":
        from ccnv.predict import run_predict

        return run_predict
    if name == "run_reference":
        from ccnv.reference import run_reference

        return run_reference
    if name == "run_reference_qc":
        from ccnv.ref_qc import run_reference_qc

        return run_reference_qc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
