from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.data_quality.profiles import MissingnessProfile, ModelSampleImpact


HIGH_MISSING_THRESHOLD = 20.0
VERY_HIGH_MISSING_THRESHOLD = 50.0
SAMPLE_LOSS_MEDIUM_THRESHOLD = 20.0
SAMPLE_LOSS_HIGH_THRESHOLD = 50.0


def build_missingness_profile(df: pd.DataFrame) -> MissingnessProfile:
    """Summarize missingness without mutating or modifying the data."""
    _require_dataframe(df)
    row_count = int(len(df))
    column_count = int(len(df.columns))
    total_cells = row_count * column_count
    missing_counts = df.isna().sum()
    missing_by_variable: list[dict] = []
    high_missing_variables: list[str] = []
    all_missing_variables: list[str] = []
    columns_with_any_missing: list[str] = []

    for column in df.columns:
        variable = str(column)
        missing_count = int(missing_counts[column])
        missing_percentage = round(float(missing_count / row_count * 100), 2) if row_count else 0.0
        row = {
            "variable": variable,
            "missing_count": missing_count,
            "missing_percentage": missing_percentage,
        }
        missing_by_variable.append(row)
        if missing_count > 0:
            columns_with_any_missing.append(variable)
        if missing_percentage > HIGH_MISSING_THRESHOLD:
            high_missing_variables.append(variable)
        if row_count > 0 and missing_count == row_count:
            all_missing_variables.append(variable)

    total_missing = int(missing_counts.sum())
    complete_case_rows = int(df.dropna().shape[0]) if column_count else row_count
    return MissingnessProfile(
        total_missing_cells=total_missing,
        total_missing_percentage=round(float(total_missing / total_cells * 100), 2) if total_cells else 0.0,
        complete_case_rows=complete_case_rows,
        complete_case_percentage=round(float(complete_case_rows / row_count * 100), 2) if row_count else 0.0,
        missing_by_variable=missing_by_variable,
        high_missing_variables=high_missing_variables,
        all_missing_variables=all_missing_variables,
        rows_with_any_missing=int(df.isna().any(axis=1).sum()) if column_count else 0,
        columns_with_any_missing=columns_with_any_missing,
    )


def estimate_model_sample_impact(df: pd.DataFrame, selected_variables: Iterable[str] | None) -> ModelSampleImpact:
    """Estimate complete-case sample loss for selected modeling variables only."""
    _require_dataframe(df)
    variables = _unique_existing_names(selected_variables or [])
    original_rows = int(len(df))
    if not variables:
        return ModelSampleImpact(
            selected_variables=[],
            original_rows=original_rows,
            usable_rows_after_dropna=original_rows,
            dropped_rows=0,
            dropped_percentage=0.0,
            variables_causing_missing_loss=[],
            warning_level="none",
        )

    existing_variables = [variable for variable in variables if variable in df.columns]
    usable_rows = int(df[existing_variables].dropna().shape[0]) if existing_variables else 0
    dropped_rows = max(0, original_rows - usable_rows)
    dropped_percentage = round(float(dropped_rows / original_rows * 100), 2) if original_rows else 0.0
    variables_causing_missing_loss = [
        variable for variable in existing_variables if int(df[variable].isna().sum()) > 0
    ]
    return ModelSampleImpact(
        selected_variables=variables,
        original_rows=original_rows,
        usable_rows_after_dropna=usable_rows,
        dropped_rows=dropped_rows,
        dropped_percentage=dropped_percentage,
        variables_causing_missing_loss=variables_causing_missing_loss,
        warning_level=_sample_loss_warning_level(dropped_percentage),
    )


def _sample_loss_warning_level(dropped_percentage: float) -> str:
    if dropped_percentage >= SAMPLE_LOSS_HIGH_THRESHOLD:
        return "high"
    if dropped_percentage >= SAMPLE_LOSS_MEDIUM_THRESHOLD:
        return "medium"
    if dropped_percentage > 0:
        return "low"
    return "none"


def _unique_existing_names(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        name = str(value)
        if not name or name in seen:
            continue
        seen.add(name)
        result.append(name)
    return result


def _require_dataframe(df: pd.DataFrame) -> None:
    if df is None:
        raise ValueError("A DataFrame is required for data quality profiling.")
