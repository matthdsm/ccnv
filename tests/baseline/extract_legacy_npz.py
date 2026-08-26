"""Extract legacy NPZ values for baseline capture only.

Legacy ccnv archives can contain pickled Python objects. This script is
intentionally outside the runtime package and requires an explicit opt-in.
Review the source archive before use.
"""

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import numpy as np


def to_json_value(value: Any) -> object:
    """Convert NumPy values into JSON-compatible values."""

    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return [to_json_value(item) for item in value.tolist()]
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_value(item) for item in value]
    return value


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--i-understand-this-may-unpickle-untrusted-data",
        action="store_true",
        dest="allow_pickle",
    )
    args = parser.parse_args()
    if not args.allow_pickle:
        parser.error("explicit unsafe-pickle acknowledgement is required")

    with np.load(args.archive, allow_pickle=True) as archive:
        extracted = {key: to_json_value(archive[key]) for key in archive.files}
    args.output.write_text(
        json.dumps(extracted, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
