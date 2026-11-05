"""Cohort samplesheet validation and parsing."""

import functools
import importlib.resources
import json
from pathlib import Path
from typing import Any

import jsonschema
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from ccnv.cohort import ReferenceError
from ccnv.models import SamplesheetEntry
from ccnv.schemas import ArtifactKind, SchemaError, validate_schema


@functools.cache
def _get_samplesheet_schema() -> dict[str, Any]:
    schema_ref = importlib.resources.files("ccnv").joinpath("samplesheet.schema.json")
    with schema_ref.open("r", encoding="utf-8") as f:
        return json.load(f)


def _validate_sample_file(path: Path) -> None:
    if not path.is_file():
        raise ReferenceError(f"{path}: sample file not found")
    try:
        schema = pq.read_schema(path)
        validate_schema(schema, ArtifactKind.SAMPLE)
    except (FileNotFoundError, OSError, pa.ArrowInvalid, pa.ArrowException) as error:
        raise ReferenceError(f"{path}: cannot read sample Parquet ({error})") from error
    except SchemaError as error:
        raise ReferenceError(f"{path}: invalid sample Parquet ({error})") from error


def load_samplesheet(path: Path | str) -> tuple[SamplesheetEntry, ...]:
    """Load, validate, and parse a cohort samplesheet from YAML or JSON.

    Parameters
    ----------
    path:
        Path to the YAML or JSON samplesheet file.

    Returns
    -------
    tuple[SamplesheetEntry, ...]
        Validated samplesheet entries with resolved filepaths in declared order.

    Raises
    ------
    ReferenceError
        If the samplesheet cannot be found, contains invalid YAML/JSON,
        violates the JSON schema, references missing or invalid sample
        Parquet files, or contains duplicate resolved sample paths.
    """
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
        raw_data = yaml.safe_load(content)
    except yaml.YAMLError as error:
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

    assert isinstance(raw_data, list)

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
