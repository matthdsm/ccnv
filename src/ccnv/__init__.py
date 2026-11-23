"""Copy-number variation analysis for whole-genome sequencing."""

from __future__ import annotations

import contextlib
import ctypes
import importlib.util
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from ccnv.models.config import (
    ExtractConfig,
    PredictConfig,
    ReadFilters,
    ReferenceConfig,
    ReferenceQCConfig,
)
from ccnv.models.exceptions import (
    AlignmentError,
    ExtractError,
    PredictError,
    ReferenceError,
    ReferenceQCError,
)


def _preload_aarch64_gomp() -> None:
    """Reserve libgomp static TLS on linux-aarch64 before Arrow or Polars load."""
    if sys.platform != "linux" or os.uname().machine != "aarch64":
        return
    loaded = False
    spec = importlib.util.find_spec("sklearn")
    if spec is not None and spec.origin is not None:
        libs_dir = Path(spec.origin).resolve().parent.parent / "scikit_learn.libs"
        if libs_dir.is_dir():
            for candidate in sorted(libs_dir.glob("libgomp*.so*")):
                with contextlib.suppress(OSError):
                    ctypes.CDLL(str(candidate))
                    loaded = True
    for prefix in dict.fromkeys((sys.prefix, sys.base_prefix)):
        candidate = Path(prefix) / "lib" / "libgomp.so.1"
        if candidate.is_file():
            with contextlib.suppress(OSError):
                ctypes.CDLL(str(candidate))
                loaded = True
    if not loaded:
        with contextlib.suppress(OSError):
            ctypes.CDLL("libgomp.so.1")


_preload_aarch64_gomp()

if TYPE_CHECKING:
    from ccnv.extract import run_extract as run_extract
    from ccnv.predict import run_predict as run_predict
    from ccnv.ref_qc import run_reference_qc as run_reference_qc
    from ccnv.reference import run_reference as run_reference

__version__ = "1.1.0"

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
