from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.models.ols import validate_ols_inputs
from src.profiler import clean_for_regression
from src.utils import significance_stars


def _validate_numeric_columns(df: pd.DataFrame, y_col: str, x_cols: list[str]) -> None:
    errors = validate_ols_inputs(
        df,
        {
            "dependent_variable": y_col,
            "independent_variables": x_cols,
        },
    )
    if errors:
        raise ValueError(errors[0])


def run_ols_regression(
    df: pd.DataFrame, y_col: str, x_cols: list[str], robust: bool = True, robust_cov_type: str = "HC3"
) -> tuple[Any, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Run OLS and return model, tidy table, summary metadata, and cleaned data."""
    if not x_cols:
        raise ValueError("Please select at least one independent variable.")
    _validate_numeric_columns(df, y_col, x_cols)

    cleaned_df, _ = clean_for_regression(df, y_col, x_cols)
    if cleaned_df.empty:
        raise ValueError("No complete observations remain after dropping missing values.")
    if len(cleaned_df) <= len(x_cols) + 1:
        raise ValueError("Not enough observations to estimate this regression model.")

    y = cleaned_df[y_col].astype(float)
    x = cleaned_df[x_cols].astype(float)
    x = sm.add_constant(x, has_constant="add")

    fitted = sm.OLS(y, x).fit()
    cov_type = (robust_cov_type or "HC3").upper()
    result = fitted.get_robustcov_results(cov_type=cov_type) if robust else fitted

    params = pd.Series(result.params, index=x.columns)
    std_errors = pd.Series(result.bse, index=x.columns)
    t_values = pd.Series(result.tvalues, index=x.columns)
    p_values = pd.Series(result.pvalues, index=x.columns)

    regression_table = pd.DataFrame(
        {
            "variable": x.columns,
            "coefficient": params.values,
            "std_error": std_errors.values,
            "t_value": t_values.values,
            "p_value": p_values.values,
        }
    )
    conf_int = pd.DataFrame(result.conf_int(), index=x.columns)
    regression_table["conf_int_low"] = conf_int.iloc[:, 0].values
    regression_table["conf_int_high"] = conf_int.iloc[:, 1].values
    regression_table["significance"] = regression_table["p_value"].apply(significance_stars)
    f_statistic, f_pvalue, model_wide_test_warning = _safe_model_wide_test_metrics(result)

    model_summary = {
        "r_squared": float(result.rsquared),
        "adj_r_squared": float(result.rsquared_adj),
        "n_obs": int(result.nobs),
        "f_statistic": f_statistic,
        "f_pvalue": f_pvalue,
        "model_wide_test_available": model_wide_test_warning is None,
        "dependent_variable": y_col,
        "independent_variables": list(x_cols),
        "robust_standard_errors": bool(robust),
        "standard_errors": cov_type.lower() if robust else "conventional",
        "robust_cov_type": cov_type if robust else "",
        "model_wide_test_warning": model_wide_test_warning,
    }
    return result, regression_table, model_summary, cleaned_df


def _safe_model_wide_test_metrics(result: Any) -> tuple[float | None, float | None, str | None]:
    try:
        f_value = result.fvalue
        f_pvalue = result.f_pvalue
        if f_value is None or f_pvalue is None:
            return None, None, None
        f_float = float(f_value)
        p_float = float(f_pvalue)
        if not np.isfinite(f_float) or not np.isfinite(p_float):
            raise ValueError("non-finite model-wide F statistic")
        return f_float, p_float, None
    except Exception:
        return (
            None,
            None,
            "Model-wide F test could not be computed reliably because of numerical instability.",
        )
