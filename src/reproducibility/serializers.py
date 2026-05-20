from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd


BLOCKED_TEXT_MARKERS = [
    "api_key",
    "secret",
    "token",
    "password",
    "traceback",
    "streamlit.session_state",
    "environment variable",
    "provider secret",
    "C:\\",
    "C:/Users/",
    "/Users/",
    "/home/",
]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [to_jsonable(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return to_jsonable(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return to_jsonable(value.tolist())
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    return value


def to_json_bytes(value: Any) -> bytes:
    return json.dumps(to_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def assert_no_blocked_markers(bundle_text: str) -> None:
    lowered = bundle_text.lower()
    for marker in BLOCKED_TEXT_MARKERS:
        if marker.lower() in lowered:
            raise ValueError(f"Reproducibility export contains blocked marker: {marker}")
