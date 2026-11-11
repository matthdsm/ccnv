"""Cohort samplesheet validation and parsing."""

from __future__ import annotations

import functools
import importlib.resources
import json
from pathlib import Path

import jsonschema
import yaml

from ccnv.models import ReferenceError, SamplesheetEntry

__all__ = [
    "_get_samplesheet_schema",
    "_validate_sample_file",
    "load_samplesheet",
]


@functools.cache
def _get_samplesheet_schema() -> dict[str, object]:
    """Load and cache the JSON schema for samplesheet validation."""
    schema_ref = importlib.resources.files("ccnv.io").joinpath(
        "samplesheet.schema.json"
    )
    with schema_ref.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ReferenceError("samplesheet JSON schema root must be an object")
    return loaded


def _validate_sample_file(path: Path) -> None:
    """Validate that a referenced sample file exists on disk."""
    if not path.is_file():
        raise ReferenceError(f"{path}: sample file not found")


def load_samplesheet(path: Path | str) -> tuple[SamplesheetEntry, ...]:
    """Load, validate, and parse a cohort samplesheet from YAML or JSON."""
    samplesheet_path = Path(path)
    if not samplesheet_path.is_file():
        raise ReferenceError(f"samplesheet not found: {samplesheet_path}")

    try:
        content = samplesheet_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ReferenceError(
            f"{samplesheet_path}: cannot read samplesheet ({error})"
        ) from error

    try:
        if samplesheet_path.suffix.lower() == ".json":
            raw_data = json.loads(content)
        else:
            loader_cls = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
            raw_data = yaml.load(content, Loader=loader_cls)
    except (json.JSONDecodeError, yaml.YAMLError) as error:
        raise ReferenceError(
            f"{samplesheet_path}: failed to parse YAML/JSON ({error})"
        ) from error

    schema = _get_samplesheet_schema()
    try:
        jsonschema.validate(instance=raw_data, schema=schema)
    except jsonschema.ValidationError as error:
        path_elements = list(error.path)
        if path_elements and isinstance(path_elements[0], int):
            entry_idx = path_elements[0]
            field = ".".join(str(p) for p in path_elements[1:])
            loc = f"entry {entry_idx}" + (f", field '{field}'" if field else "")
        else:
            field = ".".join(str(p) for p in path_elements)
            loc = f"field '{field}'" if field else "root"
        raise ReferenceError(
            f"{samplesheet_path}: schema violation at {loc}: {error.message}"
        ) from error

    if not isinstance(raw_data, list):
        raise ReferenceError(f"{samplesheet_path}: samplesheet root must be a list")

    samplesheet_dir = samplesheet_path.parent.resolve()
    seen_paths: set[Path] = set()
    entries: list[SamplesheetEntry] = []

    for idx, item in enumerate(raw_data):
        raw_filepath = item["filepath"]
        entry_path = Path(raw_filepath)
        resolved_path = (
            entry_path.resolve()
            if entry_path.is_absolute()
            else (samplesheet_dir / entry_path).resolve()
        )

        if resolved_path in seen_paths:
            raise ReferenceError(
                f"{samplesheet_path}: duplicate sample path at entry "
                f"{idx}: {resolved_path}"
            )
        seen_paths.add(resolved_path)

        _validate_sample_file(resolved_path)

        raw_sex = item.get("sex")
        raw_ff = item.get("ff")
        ff_val = float(raw_ff) if raw_ff is not None else None

        entries.append(
            SamplesheetEntry(
                filepath=resolved_path,
                sex=raw_sex,
                ff=ff_val,
            )
        )

    return tuple(entries)
