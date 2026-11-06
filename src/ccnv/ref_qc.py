"""Quality control orchestrator for reference models."""

from __future__ import annotations

import logging
from pathlib import Path

from ccnv.io.parquet import read_reference_parquet
from ccnv.models import ReferenceQCConfig, ReferenceQCError, ReferenceQCOutputs
from ccnv.qc.reference import run_reference_qc_on_table

logger = logging.getLogger(__name__)

__all__ = ["ReferenceQCError", "run_reference_qc"]


def run_reference_qc(
    config_or_path: ReferenceQCConfig | Path,
    prefix: Path | None = None,
) -> ReferenceQCOutputs:
    """Validate and run reference QC against a reference Parquet file."""
    if isinstance(config_or_path, ReferenceQCConfig):
        ref_path = config_or_path.reference.expanduser().resolve()
        out_prefix = (
            config_or_path.prefix
            if config_or_path.prefix is not None
            else ref_path.with_suffix("")
        )
    else:
        ref_path = Path(config_or_path).expanduser().resolve()
        out_prefix = prefix if prefix is not None else ref_path.with_suffix("")

    if not ref_path.is_file():
        raise ReferenceQCError(f"reference file not found: {ref_path}")

    try:
        table = read_reference_parquet(ref_path)
    except Exception as error:
        raise ReferenceQCError(f"failed to read {ref_path}: {error}") from error

    return run_reference_qc_on_table(
        table,
        prefix=out_prefix,
        reference_path=ref_path,
        write_json=True,
        emit_stderr=True,
    )
