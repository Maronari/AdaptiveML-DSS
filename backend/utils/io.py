from __future__ import annotations

import json
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def dataframe_from_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert API-style records into a DataFrame."""
    if not records:
        raise ValueError("At least one record is required.")
    return pd.DataFrame.from_records(records)


def pythonize(value: Any) -> Any:
    """Convert numpy/pandas scalar values into plain Python types."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime, date, time)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def pythonize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert every record value into a JSON-safe Python representation."""
    return {str(key): pythonize(value) for key, value in record.items()}


def dataframe_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame into JSON-safe records."""
    return [pythonize_record(record) for record in frame.to_dict(orient="records")]


def read_json(path: Path) -> list[dict[str, Any]]:
    """Read a JSON list from disk, returning an empty list when absent."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: list[dict[str, Any]]) -> None:
    """Write a JSON list to disk, creating parent directories when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
