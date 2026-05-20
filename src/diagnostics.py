from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def _looks_like_panel_data(df: pd.DataFrame) -> bool:
    lower_columns = {str(col).lower(): col for col in df.columns}
    has_year = any("year" in name or "年度" in str(col) for name, col in lower_columns.items())
    has_entity = any(
        token in name
        for name in lower_columns
        for token in ["firm_id", "company_id", "entity_id", "id", "firm", "企业"]
    )
    return has_year and has_entity


def _outlier_columns(df: pd.DataFrame) -> list[str]:
    outlier_cols: list[str] = []
    numeric_df = df.select_dtypes(include="number")
    for column in numeric_df.columns:
        series = numeric_df[column].dropna()
        if len(series) < 8 or series.nunique() <= 1:
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        share = float(((series < lower) | (series > upper)).mean())
        if share > 0:
            outlier_cols.append(str(column))
    return outlier_cols


def calculate_vif(df: pd.DataFrame, x_cols: list[str]) -> pd.DataFrame:
    """Calculate VIF values for numeric independent variables."""
    if not x_cols:
        return pd.DataFrame(columns=["variable", "VIF"])

    missing_cols = [col for col in x_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Independent variable(s) not found: {', '.join(missing_cols)}")

    non_numeric = [col for col in x_cols if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric:
        raise ValueError(f"VIF requires numeric variables. Non-numeric variable(s): {', '.join(non_numeric)}")

    x = df[x_cols].dropna().astype(float)
    if x.empty:
        return pd.DataFrame({"variable": x_cols, "VIF": [np.nan] * len(x_cols)})

    vif_values: list[float] = []
    values = x.to_numpy()
    for idx, col in enumerate(x_cols):
        if x[col].nunique(dropna=True) <= 1:
            vif_values.append(np.inf)
            continue
        try:
            vif = float(variance_inflation_factor(values, idx))
        except Exception:
            vif = np.inf
        vif_values.append(vif)

    return pd.DataFrame({"variable": x_cols, "VIF": vif_values})


def generate_diagnostic_warnings(
    df_original: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    vif_df: pd.DataFrame,
) -> list[str]:
    """Return human-readable warnings about common OLS risks."""
    warnings: list[str] = []
    n_obs = len(df_cleaned)

    duplicate_rows = int(df_original.duplicated().sum())
    if duplicate_rows > 0:
        warnings.append(f"Duplicate rows detected: {duplicate_rows} duplicate row(s) found.")

    all_missing_cols = [str(col) for col in df_original.columns if df_original[col].isna().all()]
    if all_missing_cols:
        warnings.append("Columns with all values missing detected: " + ", ".join(all_missing_cols) + ".")

    empty_named_cols = [
        str(col)
        for col in df_original.columns
        if str(col).startswith("Unnamed") or str(col).strip() == ""
    ]
    if empty_named_cols:
        warnings.append("Empty or unnamed columns detected: " + ", ".join(empty_named_cols) + ".")

    if _looks_like_panel_data(df_original):
        warnings.append(
            "Potential panel data detected from firm/entity and year columns. Reg Monkey currently runs pooled OLS only."
        )

    if y_col in df_original.columns and not pd.api.types.is_numeric_dtype(df_original[y_col]):
        warnings.append("Dependent variable is non-numeric. OLS requires a numeric dependent variable.")

    non_numeric_x = [
        col for col in x_cols if col in df_original.columns and not pd.api.types.is_numeric_dtype(df_original[col])
    ]
    if non_numeric_x:
        warnings.append(
            "Some independent variables are non-numeric and cannot be used in OLS: "
            + ", ".join(non_numeric_x)
        )

    if n_obs < 30:
        warnings.append("Sample size is below 30 observations. Regression estimates may be unstable.")

    original_rows = len(df_original)
    if original_rows:
        dropped_pct = (original_rows - n_obs) / original_rows * 100
        if dropped_pct > 30:
            warnings.append(
                f"Missing values reduced the regression sample by {dropped_pct:.1f}%, which may bias results."
            )

    if x_cols and n_obs <= len(x_cols) * 10:
        warnings.append(
            "There are many predictors relative to the sample size. Consider a simpler model or more data."
        )

    constant_x = [
        col
        for col in x_cols
        if col in df_cleaned.columns and df_cleaned[col].nunique(dropna=True) <= 1
    ]
    if constant_x:
        warnings.append(
            "Zero-variance independent variable(s) detected: "
            + ", ".join(constant_x)
            + ". These variables do not help identify regression relationships."
        )

    outlier_cols = _outlier_columns(df_original)
    if outlier_cols:
        warnings.append(
            "Extreme outliers detected in numeric column(s): "
            + ", ".join(outlier_cols)
            + ". Consider inspecting distributions and robustness."
        )

    if not vif_df.empty and "VIF" in vif_df.columns:
        high_vif = vif_df[vif_df["VIF"] > 5]
        serious_vif = vif_df[vif_df["VIF"] > 10]
        if not high_vif.empty:
            warnings.append(
                "High multicollinearity detected: VIF above 5 for "
                + ", ".join(high_vif["variable"].astype(str).tolist())
                + "."
            )
        if not serious_vif.empty:
            warnings.append(
                "Serious multicollinearity detected: VIF above 10 for "
                + ", ".join(serious_vif["variable"].astype(str).tolist())
                + "."
            )

    return warnings
