from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from src.data_quality.missingness import build_missingness_profile
from src.data_quality.quality_checks import build_variable_quality_summaries


LARGE_FILE_SIZE_MB_WARNING = 25.0
MANY_ROWS_WARNING = 50_000
MANY_ROWS_HIGH = 200_000
MANY_COLUMNS_WARNING = 100
MANY_COLUMNS_HIGH = 300
VERY_WIDE_DATA_COLUMNS = 500
HIGH_TOTAL_MISSINGNESS_WARNING = 20.0
MANY_HIGH_MISSING_VARIABLES_WARNING = 10
MANY_HIGH_CARDINALITY_VARIABLES_WARNING = 10
MANY_TEXT_LIKE_COLUMNS_WARNING = 10


@dataclass(frozen=True)
class ResourceWarningItem:
    code: str
    severity: str
    affected_variables: list[str] = field(default_factory=list)
    observed_value: int | float | None = None
    threshold: int | float | None = None
    show_in_ui: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResourceWarningProfile:
    row_count: int
    column_count: int
    file_size_mb: float | None
    public_demo_mode: bool
    warning_items: list[ResourceWarningItem] = field(default_factory=list)
    overall_warning_level: str = "none"
    thresholds: dict[str, int | float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_resource_warning_profile(
    df: pd.DataFrame,
    *,
    file_size_mb: float | None = None,
    public_demo_mode: bool = False,
) -> ResourceWarningProfile:
    """Build advisory resource and data-size warnings without mutating data."""
    _require_dataframe(df)
    row_count = int(len(df))
    column_count = int(len(df.columns))
    missingness = build_missingness_profile(df)
    summaries = build_variable_quality_summaries(df)
    warning_items: list[ResourceWarningItem] = []

    if file_size_mb is not None and float(file_size_mb) > LARGE_FILE_SIZE_MB_WARNING:
        warning_items.append(
            ResourceWarningItem(
                code="large_file_size",
                severity="warning",
                observed_value=round(float(file_size_mb), 2),
                threshold=LARGE_FILE_SIZE_MB_WARNING,
            )
        )
    if row_count > MANY_ROWS_HIGH:
        warning_items.append(ResourceWarningItem("many_rows", "warning", observed_value=row_count, threshold=MANY_ROWS_HIGH))
    elif row_count > MANY_ROWS_WARNING:
        warning_items.append(ResourceWarningItem("many_rows", "info", observed_value=row_count, threshold=MANY_ROWS_WARNING))

    if column_count > VERY_WIDE_DATA_COLUMNS:
        warning_items.append(
            ResourceWarningItem("very_wide_data", "warning", observed_value=column_count, threshold=VERY_WIDE_DATA_COLUMNS)
        )
    elif column_count > MANY_COLUMNS_HIGH:
        warning_items.append(
            ResourceWarningItem("many_columns", "warning", observed_value=column_count, threshold=MANY_COLUMNS_HIGH)
        )
    elif column_count > MANY_COLUMNS_WARNING:
        warning_items.append(ResourceWarningItem("many_columns", "info", observed_value=column_count, threshold=MANY_COLUMNS_WARNING))

    if float(missingness.total_missing_percentage) > HIGH_TOTAL_MISSINGNESS_WARNING:
        warning_items.append(
            ResourceWarningItem(
                "high_total_missingness",
                "warning",
                affected_variables=list(missingness.columns_with_any_missing),
                observed_value=float(missingness.total_missing_percentage),
                threshold=HIGH_TOTAL_MISSINGNESS_WARNING,
            )
        )

    high_missing_variables = list(missingness.high_missing_variables)
    if len(high_missing_variables) > MANY_HIGH_MISSING_VARIABLES_WARNING:
        warning_items.append(
            ResourceWarningItem(
                "many_high_missing_variables",
                "warning",
                affected_variables=high_missing_variables,
                observed_value=len(high_missing_variables),
                threshold=MANY_HIGH_MISSING_VARIABLES_WARNING,
            )
        )

    high_cardinality_variables = [item.variable for item in summaries if item.is_high_cardinality]
    if len(high_cardinality_variables) > MANY_HIGH_CARDINALITY_VARIABLES_WARNING:
        warning_items.append(
            ResourceWarningItem(
                "high_cardinality_categorical_variables",
                "info",
                affected_variables=high_cardinality_variables,
                observed_value=len(high_cardinality_variables),
                threshold=MANY_HIGH_CARDINALITY_VARIABLES_WARNING,
            )
        )

    text_like_variables = [item.variable for item in summaries if item.is_text_numeric_like or item.is_mixed_type]
    if len(text_like_variables) > MANY_TEXT_LIKE_COLUMNS_WARNING:
        warning_items.append(
            ResourceWarningItem(
                "many_text_like_columns",
                "info",
                affected_variables=text_like_variables,
                observed_value=len(text_like_variables),
                threshold=MANY_TEXT_LIKE_COLUMNS_WARNING,
            )
        )

    if _expensive_experimental_path_possible(row_count, column_count, summaries):
        warning_items.append(
            ResourceWarningItem(
                "expensive_experimental_path_possible",
                "info",
                observed_value=row_count,
                threshold=MANY_ROWS_WARNING,
            )
        )

    return ResourceWarningProfile(
        row_count=row_count,
        column_count=column_count,
        file_size_mb=round(float(file_size_mb), 2) if file_size_mb is not None else None,
        public_demo_mode=bool(public_demo_mode),
        warning_items=warning_items,
        overall_warning_level=_overall_warning_level(warning_items),
        thresholds=_thresholds(),
    )


def _expensive_experimental_path_possible(row_count: int, column_count: int, summaries: list[Any]) -> bool:
    if row_count <= MANY_ROWS_WARNING and column_count <= MANY_COLUMNS_WARNING:
        return False
    binary_like_count = sum(1 for item in summaries if item.is_binary_like)
    id_or_time_count = sum(1 for item in summaries if item.is_id_like or item.is_datetime_like)
    return binary_like_count >= 1 or id_or_time_count >= 2


def _overall_warning_level(items: list[ResourceWarningItem]) -> str:
    if any(item.severity == "warning" for item in items):
        return "warning"
    if any(item.severity == "info" for item in items):
        return "info"
    return "none"


def _thresholds() -> dict[str, int | float]:
    return {
        "large_file_size_mb_warning": LARGE_FILE_SIZE_MB_WARNING,
        "many_rows_warning": MANY_ROWS_WARNING,
        "many_rows_high": MANY_ROWS_HIGH,
        "many_columns_warning": MANY_COLUMNS_WARNING,
        "many_columns_high": MANY_COLUMNS_HIGH,
        "very_wide_data_columns": VERY_WIDE_DATA_COLUMNS,
        "high_total_missingness_warning": HIGH_TOTAL_MISSINGNESS_WARNING,
        "many_high_missing_variables_warning": MANY_HIGH_MISSING_VARIABLES_WARNING,
        "many_high_cardinality_variables_warning": MANY_HIGH_CARDINALITY_VARIABLES_WARNING,
        "many_text_like_columns_warning": MANY_TEXT_LIKE_COLUMNS_WARNING,
    }


def _require_dataframe(df: pd.DataFrame) -> None:
    if df is None:
        raise ValueError("A DataFrame is required for resource warning profiling.")
