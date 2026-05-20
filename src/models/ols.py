from __future__ import annotations

from typing import Any

import pandas as pd
from statsmodels.stats.diagnostic import het_breuschpagan

from src.diagnostics import calculate_vif, generate_diagnostic_warnings
from src.models.base import ModelDefinition


HIGH_CARDINALITY_THRESHOLD = 20


def _main_x_cols(config: dict[str, Any]) -> list[str]:
    return list(config.get("main_independent_variables") or config.get("independent_variables") or [])


def _numeric_control_cols(config: dict[str, Any]) -> list[str]:
    return list(config.get("numeric_control_variables") or [])


def _categorical_control_cols(config: dict[str, Any]) -> list[str]:
    return list(config.get("categorical_control_variables") or [])


def _all_numeric_predictors(config: dict[str, Any]) -> list[str]:
    return _main_x_cols(config) + _numeric_control_cols(config)


def validate_ols_inputs(df: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    """Validate OLS variable roles and return human-readable error strings."""
    errors: list[str] = []
    y_col = config.get("dependent_variable")
    x_cols = _all_numeric_predictors(config)
    categorical_controls = _categorical_control_cols(config)
    encode_categoricals = bool(config.get("encode_categorical_controls", False))

    if not y_col:
        errors.append("Please select a dependent variable.")
        return errors
    if y_col not in df.columns:
        errors.append(f"Dependent variable '{y_col}' was not found in the dataset.")
        return errors
    if not pd.api.types.is_numeric_dtype(df[y_col]):
        errors.append("The dependent variable is non-numeric. OLS regression requires a numeric dependent variable.")

    if not x_cols:
        errors.append("Please select at least one main independent variable.")
        return errors

    missing_x = [col for col in x_cols if col not in df.columns]
    if missing_x:
        errors.append(f"Independent variable(s) not found: {', '.join(missing_x)}")

    non_numeric_x = [
        col for col in x_cols if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric_x:
        errors.append(
            "OLS regression requires numeric independent variables. "
            f"Non-numeric variable(s): {', '.join(non_numeric_x)}"
        )

    if y_col in x_cols:
        errors.append("The dependent variable cannot also be used as an independent variable.")

    if y_col in categorical_controls:
        errors.append("The dependent variable cannot also be used as a categorical control variable.")

    missing_categorical = [col for col in categorical_controls if col not in df.columns]
    if missing_categorical:
        errors.append(f"Categorical control variable(s) not found: {', '.join(missing_categorical)}")

    if categorical_controls and not encode_categoricals:
        pass

    constant_x = [
        col for col in x_cols if col in df.columns and df[col].nunique(dropna=True) <= 1
    ]
    if constant_x:
        errors.append(
            "OLS regression cannot use constant-only independent variable(s): "
            + ", ".join(constant_x)
        )

    selected_columns = [y_col] + [col for col in x_cols if col in df.columns]
    if encode_categoricals:
        selected_columns += [col for col in categorical_controls if col in df.columns]
    complete_rows = df[selected_columns].dropna()
    if x_cols and len(complete_rows) <= len(x_cols) + 1:
        errors.append("Not enough observations to estimate this regression model.")

    return errors


def prepare_ols_dataframe(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, str, list[str], dict[str, Any], dict[str, Any]]:
    y_col = config["dependent_variable"]
    numeric_predictors = _all_numeric_predictors(config)
    categorical_controls = _categorical_control_cols(config)
    encode_categoricals = bool(config.get("encode_categorical_controls", False))

    selected_columns = [y_col] + numeric_predictors
    if encode_categoricals:
        selected_columns += categorical_controls

    selected_columns = list(dict.fromkeys(selected_columns))
    original_rows = len(df)
    working = df[selected_columns].dropna().copy()

    encoding_info = {
        "enabled": encode_categoricals,
        "selected_categorical_controls": categorical_controls,
        "encoded_categorical_controls": [],
        "reference_categories": {},
        "dummy_variables": [],
        "ignored_categorical_controls": [] if encode_categoricals else categorical_controls,
        "high_cardinality_warnings": [],
    }

    x_cols = list(numeric_predictors)
    if encode_categoricals and categorical_controls:
        for column in categorical_controls:
            levels = sorted(working[column].dropna().astype(str).unique().tolist())
            if len(levels) > HIGH_CARDINALITY_THRESHOLD:
                encoding_info["high_cardinality_warnings"].append(
                    f"Categorical control '{column}' has {len(levels)} levels; dummy encoding may create a large model."
                )
            if len(levels) <= 1:
                continue
            reference = levels[0]
            encoding_info["reference_categories"][column] = reference
            dummies = pd.get_dummies(working[column].astype(str), prefix=column, drop_first=True, dtype=float)
            working = pd.concat([working.drop(columns=[column]), dummies], axis=1)
            dummy_names = dummies.columns.tolist()
            x_cols.extend(dummy_names)
            encoding_info["dummy_variables"].extend(dummy_names)
            encoding_info["encoded_categorical_controls"].append(column)

    final_rows = len(working)
    cleaning_log = {
        "original_row_count": original_rows,
        "final_row_count": final_rows,
        "dropped_row_count": original_rows - final_rows,
        "dropped_row_percentage": round(((original_rows - final_rows) / original_rows * 100), 2) if original_rows else 0.0,
    }

    return working, y_col, x_cols, cleaning_log, encoding_info


def fit_ols(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Fit OLS and return a model-result payload used by the app/reporting."""
    errors = validate_ols_inputs(df, config)
    if errors:
        raise ValueError(errors[0])

    from src.regression import run_ols_regression

    robust = bool(config.get("robust_standard_errors", True))
    robust_cov_type = str(config.get("robust_cov_type") or "HC3")
    model_df, y_col, x_cols, cleaning_log, encoding_info = prepare_ols_dataframe(df, config)
    fitted_model, regression_table, model_summary, cleaned_df = run_ols_regression(
        model_df, y_col, x_cols, robust=robust, robust_cov_type=robust_cov_type
    )
    model_summary["main_independent_variables"] = _main_x_cols(config)
    model_summary["numeric_control_variables"] = _numeric_control_cols(config)
    model_summary["categorical_control_variables"] = _categorical_control_cols(config)
    model_summary["encoded_categorical_controls"] = encoding_info["encoded_categorical_controls"]
    model_summary["dummy_variables"] = encoding_info["dummy_variables"]
    model_summary["reference_categories"] = encoding_info["reference_categories"]
    warnings = []
    if model_summary.get("model_wide_test_warning"):
        warnings.append(str(model_summary["model_wide_test_warning"]))

    return {
        "fitted_model": fitted_model,
        "regression_table": regression_table,
        "model_summary": model_summary,
        "cleaned_df": cleaned_df,
        "cleaning_log": cleaning_log,
        "encoding_info": encoding_info,
        "warnings": warnings,
    }


def diagnose_ols(
    df_original: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    config: dict[str, Any],
    fit_result: Any,
) -> dict[str, Any]:
    """Run OLS-specific diagnostics."""
    y_col = config["dependent_variable"]
    x_cols = list(fit_result["model_summary"].get("independent_variables", _all_numeric_predictors(config)))

    if len(x_cols) >= 2:
        vif_df = calculate_vif(df_cleaned, x_cols)
    else:
        vif_df = pd.DataFrame(columns=["variable", "VIF"])

    warnings = list(fit_result.get("warnings", []))
    warnings.extend(generate_diagnostic_warnings(df_original, df_cleaned, y_col, x_cols, vif_df))
    warnings.extend(fit_result.get("encoding_info", {}).get("high_cardinality_warnings", []))
    heteroskedasticity = _breusch_pagan_diagnostic(fit_result.get("fitted_model"))
    return {"vif_df": vif_df, "warnings": warnings, "heteroskedasticity": heteroskedasticity}


def _breusch_pagan_diagnostic(fitted_model: Any) -> dict[str, Any]:
    if fitted_model is None or not hasattr(fitted_model, "resid") or not hasattr(fitted_model, "model"):
        return {}
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(fitted_model.resid, fitted_model.model.exog)
    return {
        "test": "Breusch-Pagan",
        "statistic": float(lm_stat),
        "p_value": float(lm_pvalue),
        "f_statistic": float(f_stat),
        "f_p_value": float(f_pvalue),
    }


OLS_MODEL = ModelDefinition(
    model_id="ols",
    display_name_en="OLS Regression",
    display_name_zh="OLS 回归",
    description_en="Estimate a linear regression model for a numeric dependent variable.",
    description_zh="用于数值型因变量的线性回归模型。",
    required_roles=["dependent_variable", "independent_variables"],
    validate=validate_ols_inputs,
    fit=fit_ols,
    diagnostics=diagnose_ols,
    report_label_en="OLS Regression",
    report_label_zh="OLS 回归",
    limitations_en=[
        "Requires a numeric dependent variable and numeric independent variables.",
        "Does not automatically prove causality.",
        "Categorical variables enter the model only when the user explicitly enables dummy encoding.",
    ],
    limitations_zh=[
        "要求因变量和自变量均为数值变量。",
        "回归结果本身不能自动证明因果关系。",
        "分类变量必须由用户明确选择并启用虚拟变量编码后，才会进入模型。",
    ],
)
