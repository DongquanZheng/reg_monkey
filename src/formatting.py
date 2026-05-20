from __future__ import annotations

from numbers import Number
from typing import Any

import pandas as pd


def format_p_value(value: Any, language: str = "en") -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A" if language != "zh" else "不适用"
    if pd.isna(numeric):
        return "N/A" if language != "zh" else "不适用"
    if numeric < 0.001:
        return "<0.001"
    return f"{numeric:.3f}"


def format_p_narrative(value: Any, language: str = "en") -> str:
    formatted = format_p_value(value, language)
    if formatted == "<0.001":
        return "p < 0.001"
    if formatted in {"N/A", "不适用"}:
        return "p = " + formatted
    return "p = " + formatted


def format_number(value: Any, digits: int = 4, language: str = "en") -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(numeric):
        return "N/A" if language != "zh" else "不适用"
    return f"{numeric:.{digits}f}"


def format_percentage(value: Any, digits: int = 1, language: str = "en") -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A" if language != "zh" else "不适用"
    if pd.isna(numeric):
        return "N/A" if language != "zh" else "不适用"
    return f"{numeric:.{digits}f}%"


def prepare_display_table(table: pd.DataFrame | None, language: str = "en") -> pd.DataFrame:
    if table is None or table.empty:
        return pd.DataFrame()

    prepared = table.copy()
    placeholder = "N/A" if language != "zh" else "不适用"
    for column in prepared.columns:
        series = prepared[column]
        non_missing = series.dropna()
        has_text = any(isinstance(value, str) for value in non_missing)
        has_numeric = any(isinstance(value, Number) and not isinstance(value, bool) for value in non_missing)
        if has_text and has_numeric:
            prepared[column] = series.apply(lambda value: placeholder if pd.isna(value) else str(value))
    return prepared


def format_regression_table(table: pd.DataFrame, language: str = "en") -> pd.DataFrame:
    if table is None or table.empty:
        return pd.DataFrame()

    formatted = table.copy()
    for column in [
        "coefficient",
        "std_error",
        "t_value",
        "z_value",
        "conf_int_low",
        "conf_int_high",
        "odds_ratio",
        "marginal_effect",
    ]:
        if column in formatted.columns:
            formatted[column] = formatted[column].apply(lambda value: format_number(value, 4, language))
    if "p_value" in formatted.columns:
        formatted["p_value"] = formatted["p_value"].apply(lambda value: format_p_value(value, language))
    return formatted


def format_diagnostic_table(table: pd.DataFrame, language: str = "en", digits: int = 2) -> pd.DataFrame:
    if table is None or table.empty:
        return pd.DataFrame()

    formatted = table.copy()
    for column in formatted.columns:
        if pd.api.types.is_numeric_dtype(formatted[column]):
            formatted[column] = formatted[column].apply(lambda value: format_number(value, digits, language))
    return formatted
