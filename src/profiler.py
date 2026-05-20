from __future__ import annotations

from typing import Any

import pandas as pd

from src.variable_roles import BINARY_FALSE, BINARY_TRUE, is_binary_like


MISSING_SYMBOLS = {
    "",
    "NA",
    "N/A",
    "na",
    "n/a",
    "NULL",
    "null",
    "None",
    "none",
    "--",
    "-",
    "—",
    "无",
    "缺失",
}


def _normalize_text_series(series: pd.Series) -> pd.Series:
    """Strip common whitespace and normalize blank strings to missing values."""
    return series.astype("string").str.strip().replace({symbol: pd.NA for symbol in MISSING_SYMBOLS})


def _has_leading_zero_codes(series: pd.Series) -> bool:
    cleaned_text = _normalize_text_series(series).dropna()
    return bool(cleaned_text.str.fullmatch(r"0\d+").any())


def _is_code_like_column(column: Any) -> bool:
    name = str(column).lower()
    return any(token in name for token in ["code", "id", "编码", "代码"])


def _is_year_like_column(column: Any) -> bool:
    name = str(column).lower()
    return "year" in name or "年度" in name or "年份" in name


def _numeric_conversion_candidate(series: pd.Series) -> tuple[pd.Series, float]:
    """Return a numeric conversion attempt and its successful conversion ratio."""
    cleaned_text = _normalize_text_series(series)
    without_commas = cleaned_text.str.replace(",", "", regex=False)
    without_percent = without_commas.str.replace("%", "", regex=False)
    converted = pd.to_numeric(without_percent, errors="coerce")
    non_missing_count = int(cleaned_text.notna().sum())
    if non_missing_count == 0:
        return converted, 0.0
    conversion_ratio = float(converted.notna().sum() / non_missing_count)
    return converted, conversion_ratio


def _year_conversion_candidate(series: pd.Series) -> tuple[pd.Series, float]:
    cleaned_text = _normalize_text_series(series)
    extracted = cleaned_text.str.extract(r"((?:19|20)\d{2})", expand=False)
    converted = pd.to_numeric(extracted, errors="coerce")
    non_missing_count = int(cleaned_text.notna().sum())
    if non_missing_count == 0:
        return converted, 0.0
    conversion_ratio = float(converted.notna().sum() / non_missing_count)
    return converted, conversion_ratio


def _choose_numeric_conversion(series: pd.Series, column: Any) -> tuple[pd.Series, float, str]:
    numeric_converted, numeric_ratio = _numeric_conversion_candidate(series)
    if _is_year_like_column(column):
        year_converted, year_ratio = _year_conversion_candidate(series)
        if year_ratio >= numeric_ratio:
            return year_converted, year_ratio, "year_extraction"
    return numeric_converted, numeric_ratio, "numeric"


def _count_mostly_numeric_columns(df: pd.DataFrame, threshold: float = 0.70) -> int:
    count = 0
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            count += 1
            continue
        _, ratio, _ = _choose_numeric_conversion(df[column], column)
        if ratio >= threshold:
            count += 1
    return count


def _numeric_convertibility_score(df: pd.DataFrame, threshold: float = 0.70) -> float:
    """Score how numeric-like a DataFrame is after a potential row skip."""
    if df.empty:
        return 0.0

    mostly_numeric_count = 0
    ratio_sum = 0.0
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            mostly_numeric_count += 1
            ratio_sum += 1.0
            continue

        _, ratio, _ = _choose_numeric_conversion(df[column], column)
        ratio_sum += ratio
        if ratio >= threshold:
            mostly_numeric_count += 1

    # Column count matters most; ratio sum breaks ties such as label rows vs unit rows.
    return mostly_numeric_count * 100.0 + ratio_sum


def detect_metadata_rows(df: pd.DataFrame, max_rows: int = 5) -> int:
    """Detect top metadata/unit rows by maximizing numeric convertibility.

    The heuristic tries skip values from 0 to 5 and chooses the smallest skip
    value that produces the best number of mostly numeric columns. This keeps
    automatic behavior conservative while still handling common unit rows.
    """
    if df is None or df.empty:
        return 0

    best_skip = 0
    best_score = _numeric_convertibility_score(df)
    max_skip = min(max_rows, max(0, len(df) - 1))

    for skip in range(1, max_skip + 1):
        candidate = df.iloc[skip:].reset_index(drop=True)
        if len(candidate) < 2:
            continue
        numeric_score = _numeric_convertibility_score(candidate)
        remaining_ratio = len(candidate) / len(df)
        score = numeric_score + (remaining_ratio * 0.01)
        best_adjusted_score = best_score + ((len(df) - best_skip) / len(df) * 0.01)
        if score > best_adjusted_score + 0.05:
            best_skip = skip
            best_score = numeric_score

    return best_skip


def preprocess_dataframe(
    df: pd.DataFrame,
    skip_rows: int = 0,
    auto_detect_metadata_rows: bool = False,
    coerce_numeric: bool = True,
    use_first_row_as_column_names: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prepare uploaded data for profiling, visualization, and regression.

    This handles common statistical yearbook exports where the first rows after
    the header contain labels, units, or notes that make numeric columns look
    like text to pandas.
    """
    if df is None:
        raise ValueError("A DataFrame is required for preprocessing.")

    original_columns = [str(column) for column in df.columns]
    generated_temporary_variable_names: list[str] = []
    if use_first_row_as_column_names:
        working_df = df.copy()
    else:
        generated_temporary_variable_names = [f"Var_{index + 1}" for index in range(df.shape[1])]
        header_row = pd.DataFrame([list(df.columns)], columns=generated_temporary_variable_names)
        data_rows = df.copy()
        data_rows.columns = generated_temporary_variable_names
        working_df = pd.concat([header_row, data_rows], ignore_index=True)

    requested_skip = max(0, int(skip_rows or 0))
    detected_skip = detect_metadata_rows(working_df) if auto_detect_metadata_rows else 0
    effective_skip = min(max(requested_skip, detected_skip), len(working_df))
    processed = working_df.iloc[effective_skip:].reset_index(drop=True).copy()

    converted_columns: list[str] = []
    categorical_columns: list[str] = []
    removed_empty_columns: list[str] = []
    percent_columns: list[str] = []
    year_extracted_columns: list[str] = []
    protected_code_columns: list[str] = []
    conversion_ratios: dict[str, float] = {}

    for column in list(processed.columns):
        column_name = str(column)
        is_unnamed = column_name.startswith("Unnamed") or column_name.strip() == ""
        if is_unnamed and processed[column].isna().all():
            processed = processed.drop(columns=[column])
            removed_empty_columns.append(column_name)

    for column in processed.columns:
        if not str(column).startswith("Unnamed") and processed[column].isna().all():
            categorical_columns.append(column)

    if coerce_numeric:
        for column in processed.columns:
            if pd.api.types.is_numeric_dtype(processed[column]):
                converted_columns.append(column)
                conversion_ratios[column] = 1.0
                continue

            if _has_leading_zero_codes(processed[column]) and _is_code_like_column(column):
                processed[column] = _normalize_text_series(processed[column])
                protected_code_columns.append(column)
                categorical_columns.append(column)
                conversion_ratios[column] = 0.0
                continue

            converted, ratio, conversion_kind = _choose_numeric_conversion(processed[column], column)
            conversion_ratios[column] = round(ratio, 4)
            if ratio >= 0.70:
                processed[column] = converted
                converted_columns.append(column)
                text_values = _normalize_text_series(df[column] if column in df.columns else processed[column]).dropna()
                if bool(text_values.astype("string").str.contains("%", regex=False).any()):
                    percent_columns.append(column)
                if conversion_kind == "year_extraction":
                    year_extracted_columns.append(column)
            else:
                processed[column] = _normalize_text_series(processed[column])
                if column not in categorical_columns:
                    categorical_columns.append(column)
    else:
        categorical_columns = [
            column for column in processed.columns if not pd.api.types.is_numeric_dtype(processed[column])
        ]

    preprocessing_log = {
        "requested_skip_rows": requested_skip,
        "detected_skip_rows": detected_skip,
        "rows_skipped": effective_skip,
        "use_first_row_as_column_names": bool(use_first_row_as_column_names),
        "generated_temporary_variable_names": generated_temporary_variable_names,
        "original_column_names": original_columns,
        "auto_detect_metadata_rows": bool(auto_detect_metadata_rows),
        "coerce_numeric": bool(coerce_numeric),
        "columns_converted_to_numeric": converted_columns,
        "columns_kept_as_categorical": categorical_columns,
        "columns_removed_empty": removed_empty_columns,
        "percent_columns_converted": percent_columns,
        "year_columns_extracted": year_extracted_columns,
        "protected_code_columns": protected_code_columns,
        "numeric_conversion_ratios": conversion_ratios,
        "rows_before_preprocessing": int(working_df.shape[0]),
        "columns_before_preprocessing": int(working_df.shape[1]),
        "remaining_rows": int(processed.shape[0]),
        "remaining_columns": int(processed.shape[1]),
    }

    return processed, preprocessing_log


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Create a compact profile dictionary for a DataFrame."""
    if df is None:
        raise ValueError("A DataFrame is required for profiling.")

    missing_counts = df.isna().sum()
    row_count = len(df)
    missing_percentages = (
        (missing_counts / row_count * 100).round(2) if row_count else missing_counts.astype(float)
    )
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = [col for col in df.columns if col not in numeric_columns]

    numeric_descriptive_statistics = compute_numeric_descriptive_stats(df)
    binary_descriptive_statistics = compute_binary_descriptive_stats(df)
    categorical_descriptive_statistics = compute_categorical_descriptive_stats(df)

    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": missing_counts.astype(int).to_dict(),
        "missing_percentages": missing_percentages.to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "descriptive_statistics": numeric_descriptive_statistics,
        "numeric_descriptive_statistics": numeric_descriptive_statistics,
        "binary_descriptive_statistics": binary_descriptive_statistics,
        "categorical_descriptive_statistics": categorical_descriptive_statistics,
    }


def compute_numeric_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return expanded descriptive statistics for numeric variables."""
    if df is None:
        raise ValueError("A DataFrame is required for descriptive statistics.")

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    rows: list[dict[str, Any]] = []
    row_count = len(df)

    for column in numeric_columns:
        series = df[column]
        if is_binary_like(series):
            continue
        non_missing = series.dropna()
        rows.append(
            {
                "variable": column,
                "count": int(non_missing.count()),
                "mean": round(float(non_missing.mean()), 4) if not non_missing.empty else pd.NA,
                "std": round(float(non_missing.std()), 4) if len(non_missing) > 1 else pd.NA,
                "min": round(float(non_missing.min()), 4) if not non_missing.empty else pd.NA,
                "25%": round(float(non_missing.quantile(0.25)), 4) if not non_missing.empty else pd.NA,
                "median": round(float(non_missing.median()), 4) if not non_missing.empty else pd.NA,
                "75%": round(float(non_missing.quantile(0.75)), 4) if not non_missing.empty else pd.NA,
                "max": round(float(non_missing.max()), 4) if not non_missing.empty else pd.NA,
                "missing_count": int(series.isna().sum()),
                "missing_percentage": round(float(series.isna().mean() * 100), 2) if row_count else 0.0,
                "skewness": round(float(non_missing.skew()), 4) if len(non_missing) > 2 else pd.NA,
                "kurtosis": round(float(non_missing.kurtosis()), 4) if len(non_missing) > 3 else pd.NA,
                "unique_values": int(series.nunique(dropna=True)),
            }
        )

    return pd.DataFrame(rows)


def compute_categorical_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for categorical/text variables."""
    if df is None:
        raise ValueError("A DataFrame is required for descriptive statistics.")

    numeric_columns = set(df.select_dtypes(include="number").columns.tolist())
    categorical_columns = [column for column in df.columns if column not in numeric_columns]
    rows: list[dict[str, Any]] = []
    row_count = len(df)

    for column in categorical_columns:
        series = df[column]
        non_missing = series.dropna()
        value_counts = non_missing.astype(str).value_counts()
        top_category = value_counts.index[0] if not value_counts.empty else pd.NA
        top_frequency = int(value_counts.iloc[0]) if not value_counts.empty else 0
        rows.append(
            {
                "variable": column,
                "count": int(non_missing.count()),
                "missing_count": int(series.isna().sum()),
                "missing_percentage": round(float(series.isna().mean() * 100), 2) if row_count else 0.0,
                "unique_categories": int(non_missing.astype(str).nunique(dropna=True)),
                "top_category": top_category,
                "top_frequency": top_frequency,
                "top_percentage": round(float(top_frequency / len(non_missing) * 100), 2) if len(non_missing) else 0.0,
            }
        )

    return pd.DataFrame(rows)


def compute_binary_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for columns with two binary-like values."""
    if df is None:
        raise ValueError("A DataFrame is required for descriptive statistics.")

    rows: list[dict[str, Any]] = []
    row_count = len(df)
    for column in df.columns:
        series = df[column]
        if not is_binary_like(series):
            continue

        normalized = series.dropna().astype(str).str.strip().str.lower()
        is_one = normalized.isin(BINARY_TRUE | {"1.0"})
        is_zero = normalized.isin(BINARY_FALSE | {"0.0"})
        ones = int(is_one.sum())
        zeros = int(is_zero.sum())
        non_missing_count = int(series.notna().sum())
        rows.append(
            {
                "variable": column,
                "count": non_missing_count,
                "zeros": zeros,
                "ones": ones,
                "percentage_one": round(float(ones / non_missing_count * 100), 2) if non_missing_count else 0.0,
                "missing_count": int(series.isna().sum()),
                "missing_percentage": round(float(series.isna().mean() * 100), 2) if row_count else 0.0,
            }
        )

    return pd.DataFrame(rows)


def clean_for_regression(
    df: pd.DataFrame, y_col: str, x_cols: list[str]
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Keep selected regression variables and drop rows with missing values."""
    if df is None:
        raise ValueError("A DataFrame is required for cleaning.")
    if not y_col:
        raise ValueError("Please select a dependent variable.")
    if not x_cols:
        raise ValueError("Please select at least one independent variable.")

    selected_columns = [y_col] + list(x_cols)
    missing_columns = [col for col in selected_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Selected columns not found in data: {', '.join(missing_columns)}")

    original_rows = len(df)
    cleaned = df[selected_columns].dropna().copy()
    final_rows = len(cleaned)
    dropped_rows = original_rows - final_rows
    dropped_pct = round((dropped_rows / original_rows * 100), 2) if original_rows else 0.0

    cleaning_log = {
        "original_row_count": original_rows,
        "final_row_count": final_rows,
        "dropped_row_count": dropped_rows,
        "dropped_row_percentage": dropped_pct,
    }
    return cleaned, cleaning_log
