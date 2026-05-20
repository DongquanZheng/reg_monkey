from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def is_numeric_series(series: pd.Series) -> bool:
    """Return True when a pandas Series has a numeric dtype."""
    return pd.api.types.is_numeric_dtype(series)


def significance_stars(p_value: float | int | None) -> str:
    """Convert a p-value into conventional academic significance stars."""
    if p_value is None:
        return ""
    try:
        p = float(p_value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Encode a DataFrame as UTF-8 CSV bytes for Streamlit downloads."""
    return df.to_csv(index=False).encode("utf-8")


def safe_float(value: Any) -> float | None:
    """Return a regular float when possible, otherwise None."""
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(output) or np.isinf(output):
        return None
    return output
