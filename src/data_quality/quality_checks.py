from __future__ import annotations

from typing import Any

import pandas as pd

from src.data_quality.missingness import HIGH_MISSING_THRESHOLD, VERY_HIGH_MISSING_THRESHOLD, build_missingness_profile
from src.data_quality.profiles import DataQualityProfile, VariableQualitySummary
from src.variable_roles import is_binary_like


HIGH_CARDINALITY_UNIQUE_THRESHOLD = 20
HIGH_CARDINALITY_SHARE_THRESHOLD = 0.5
ID_LIKE_UNIQUE_SHARE_THRESHOLD = 0.95
NEAR_CONSTANT_TOP_SHARE_THRESHOLD = 0.95
TEXT_NUMERIC_CONVERSION_THRESHOLD = 0.8
DATETIME_CONVERSION_THRESHOLD = 0.8

ID_NAME_TOKENS = ("id", "_id", "code", "uuid", "identifier", "编号", "代码")
DATETIME_NAME_TOKENS = ("date", "time", "year", "month", "quarter", "日期", "时间", "年份", "年度")


def build_variable_quality_summaries(df: pd.DataFrame) -> list[VariableQualitySummary]:
    """Return per-variable quality hints without changing role assignments."""
    _require_dataframe(df)
    row_count = int(len(df))
    return [_variable_summary(df, column, row_count) for column in df.columns]


def build_data_quality_profile(df: pd.DataFrame) -> DataQualityProfile:
    """Build a language-neutral, JSON-friendly data quality profile."""
    _require_dataframe(df)
    summaries = build_variable_quality_summaries(df)
    missingness = build_missingness_profile(df)
    warnings = _profile_warnings(summaries, missingness)

    return DataQualityProfile(
        row_count=int(len(df)),
        column_count=int(len(df.columns)),
        duplicate_row_count=int(df.duplicated().sum()),
        numeric_columns=[str(column) for column in df.select_dtypes(include="number").columns],
        categorical_columns=[str(column) for column in df.columns if not pd.api.types.is_numeric_dtype(df[column])],
        binary_columns=[item.variable for item in summaries if item.is_binary_like],
        datetime_like_columns=[item.variable for item in summaries if item.is_datetime_like],
        id_like_columns=[item.variable for item in summaries if item.is_id_like],
        high_cardinality_columns=[item.variable for item in summaries if item.is_high_cardinality],
        constant_columns=[item.variable for item in summaries if item.is_constant],
        near_constant_columns=[item.variable for item in summaries if item.is_near_constant and not item.is_constant],
        text_numeric_columns=[item.variable for item in summaries if item.is_text_numeric_like],
        mixed_type_columns=[item.variable for item in summaries if item.is_mixed_type],
        warnings=warnings,
    )


def _variable_summary(df: pd.DataFrame, column: Any, row_count: int) -> VariableQualitySummary:
    variable = str(column)
    series = df[column]
    non_missing = series.dropna()
    missing_count = int(series.isna().sum())
    missing_percentage = round(float(missing_count / row_count * 100), 2) if row_count else 0.0
    unique_count = int(non_missing.astype(str).nunique(dropna=True)) if not non_missing.empty else 0
    unique_percentage = round(float(unique_count / row_count * 100), 2) if row_count else 0.0
    binary_like = bool(is_binary_like(series))
    id_like = _is_id_like(variable, unique_count, row_count)
    constant = unique_count <= 1
    near_constant = constant or _is_near_constant(non_missing)
    high_cardinality = _is_high_cardinality(series, unique_count, row_count, id_like)
    text_numeric_like = _is_text_numeric_like(series)
    datetime_like = _is_datetime_like(variable, series)
    mixed_type = _is_mixed_type(series)
    return VariableQualitySummary(
        variable=variable,
        dtype=str(series.dtype),
        inferred_role_hint=_role_hint(binary_like, id_like, datetime_like, high_cardinality, text_numeric_like, series),
        missing_count=missing_count,
        missing_percentage=missing_percentage,
        unique_count=unique_count,
        unique_percentage=unique_percentage,
        is_binary_like=binary_like,
        is_id_like=id_like,
        is_constant=constant,
        is_near_constant=near_constant,
        is_high_cardinality=high_cardinality,
        is_text_numeric_like=text_numeric_like,
        is_datetime_like=datetime_like,
        is_mixed_type=mixed_type,
    )


def _profile_warnings(summaries: list[VariableQualitySummary], missingness) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    if missingness.high_missing_variables:
        warnings.append(
            {
                "code": "high_missing_variables",
                "severity": "warning",
                "variables": list(missingness.high_missing_variables),
                "threshold": HIGH_MISSING_THRESHOLD,
            }
        )
    very_high = [
        item["variable"]
        for item in missingness.missing_by_variable
        if float(item.get("missing_percentage", 0.0)) > VERY_HIGH_MISSING_THRESHOLD
    ]
    if very_high:
        warnings.append(
            {
                "code": "very_high_missing_variables",
                "severity": "warning",
                "variables": very_high,
                "threshold": VERY_HIGH_MISSING_THRESHOLD,
            }
        )
    if missingness.all_missing_variables:
        warnings.append(
            {
                "code": "all_missing_variables",
                "severity": "error",
                "variables": list(missingness.all_missing_variables),
            }
        )
    constant = [item.variable for item in summaries if item.is_constant]
    if constant:
        warnings.append({"code": "constant_variables", "severity": "warning", "variables": constant})
    high_cardinality = [item.variable for item in summaries if item.is_high_cardinality]
    if high_cardinality:
        warnings.append({"code": "high_cardinality_variables", "severity": "info", "variables": high_cardinality})
    return warnings


def _is_id_like(variable: str, unique_count: int, row_count: int) -> bool:
    lower = variable.lower()
    if any(token in lower for token in ID_NAME_TOKENS):
        return True
    return row_count >= 10 and unique_count >= max(8, int(row_count * ID_LIKE_UNIQUE_SHARE_THRESHOLD))


def _is_high_cardinality(series: pd.Series, unique_count: int, row_count: int, id_like: bool) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return False
    if id_like:
        return True
    if unique_count > HIGH_CARDINALITY_UNIQUE_THRESHOLD:
        return True
    return row_count >= 10 and unique_count / row_count > HIGH_CARDINALITY_SHARE_THRESHOLD


def _is_near_constant(non_missing: pd.Series) -> bool:
    if non_missing.empty:
        return True
    top_share = float(non_missing.astype(str).value_counts(normalize=True).iloc[0])
    return top_share >= NEAR_CONSTANT_TOP_SHARE_THRESHOLD


def _is_text_numeric_like(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return False
    cleaned = series.dropna().astype(str).str.strip()
    if cleaned.empty:
        return False
    converted = pd.to_numeric(cleaned.str.replace(",", "", regex=False).str.replace("%", "", regex=False), errors="coerce")
    return float(converted.notna().sum() / len(cleaned)) >= TEXT_NUMERIC_CONVERSION_THRESHOLD


def _is_datetime_like(variable: str, series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    lower = variable.lower()
    name_hint = any(token in lower for token in DATETIME_NAME_TOKENS)
    if pd.api.types.is_numeric_dtype(series):
        return name_hint
    if not name_hint:
        return False
    cleaned = series.dropna().astype(str).str.strip()
    if cleaned.empty:
        return False
    parsed = pd.to_datetime(cleaned, errors="coerce")
    return name_hint and float(parsed.notna().sum() / len(cleaned)) >= DATETIME_CONVERSION_THRESHOLD


def _is_mixed_type(series: pd.Series) -> bool:
    non_missing = series.dropna()
    if non_missing.empty or not pd.api.types.is_object_dtype(series):
        return False
    types = {type(value).__name__ for value in non_missing}
    return len(types) > 1


def _role_hint(
    binary_like: bool,
    id_like: bool,
    datetime_like: bool,
    high_cardinality: bool,
    text_numeric_like: bool,
    series: pd.Series,
) -> str:
    if binary_like:
        return "binary_like"
    if id_like:
        return "id_like"
    if datetime_like:
        return "datetime_like"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric_like"
    if text_numeric_like:
        return "text_numeric_like"
    if high_cardinality:
        return "high_cardinality_categorical"
    return "categorical_like"


def _require_dataframe(df: pd.DataFrame) -> None:
    if df is None:
        raise ValueError("A DataFrame is required for data quality profiling.")
