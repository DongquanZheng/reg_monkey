from __future__ import annotations

from typing import Any, Callable
import warnings as py_warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationError

from src.diagnostics import calculate_vif
from src.models.ols import HIGH_CARDINALITY_THRESHOLD
from src.utils import significance_stars


BINARY_STRING_MAP = {
    "1": 1,
    "0": 0,
    "true": 1,
    "false": 0,
    "yes": 1,
    "no": 0,
    "y": 1,
    "n": 0,
    "是": 1,
    "否": 0,
    "是的": 1,
    "不是": 0,
}


def _main_x_cols(config: dict[str, Any]) -> list[str]:
    return list(config.get("main_independent_variables") or config.get("independent_variables") or [])


def _numeric_control_cols(config: dict[str, Any]) -> list[str]:
    return list(config.get("numeric_control_variables") or [])


def _categorical_control_cols(config: dict[str, Any]) -> list[str]:
    return list(config.get("categorical_control_variables") or [])


def _numeric_predictors(config: dict[str, Any]) -> list[str]:
    return _main_x_cols(config) + _numeric_control_cols(config)


def coerce_binary_series(series: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    """Convert common binary labels to 0/1 while rejecting variables with >2 classes."""
    non_missing = series.dropna()
    info = {
        "converted_binary_y": False,
        "binary_mapping": {},
        "unique_values": sorted(non_missing.astype(str).unique().tolist()),
    }
    if non_missing.empty:
        return pd.Series(np.nan, index=series.index, dtype="float"), info

    if pd.api.types.is_bool_dtype(series):
        info["converted_binary_y"] = True
        info["binary_mapping"] = {"False": 0, "True": 1}
        return series.astype("float"), info

    if pd.api.types.is_numeric_dtype(series):
        unique = sorted(pd.Series(non_missing).astype(float).unique().tolist())
        if set(unique).issubset({0.0, 1.0}):
            return series.astype("float"), info
        return series, info

    cleaned = series.astype("string").str.strip()
    lowered = cleaned.str.lower()
    converted = lowered.map(BINARY_STRING_MAP)
    if converted.notna().sum() == non_missing.shape[0]:
        info["converted_binary_y"] = True
        used = sorted(cleaned.dropna().unique().tolist())
        info["binary_mapping"] = {value: BINARY_STRING_MAP[str(value).strip().lower()] for value in used}
        return converted.astype("float"), info

    unique = sorted(non_missing.astype(str).str.strip().unique().tolist())
    if len(unique) == 2:
        mapping = {unique[0]: 0, unique[1]: 1}
        info["converted_binary_y"] = True
        info["binary_mapping"] = mapping
        return cleaned.map(mapping).astype("float"), info

    return series, info


def validate_binary_choice_inputs(df: pd.DataFrame, config: dict[str, Any], model_label: str) -> list[str]:
    errors: list[str] = []
    y_col = config.get("dependent_variable")
    x_cols = _numeric_predictors(config)
    categorical_controls = _categorical_control_cols(config)
    encode_categoricals = bool(config.get("encode_categorical_controls", False))

    if not y_col:
        errors.append("Please select a dependent variable.")
        return errors
    if y_col not in df.columns:
        errors.append(f"Dependent variable '{y_col}' was not found in the dataset.")
        return errors

    y_binary, _ = coerce_binary_series(df[y_col])
    y_complete = y_binary.dropna()
    if y_complete.empty or set(pd.Series(y_complete).unique()).difference({0, 1, 0.0, 1.0}):
        errors.append(f"{model_label} requires a binary dependent variable such as 0/1, Yes/No, Y/N, or 是/否.")
    elif y_complete.nunique(dropna=True) != 2:
        errors.append("The dependent variable has only one outcome class after removing missing values.")

    if not x_cols:
        errors.append("Please select at least one main independent variable.")
        return errors

    missing_x = [col for col in x_cols if col not in df.columns]
    if missing_x:
        errors.append(f"Independent variable(s) not found: {', '.join(missing_x)}")

    non_numeric_x = [col for col in x_cols if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric_x:
        errors.append(f"{model_label} requires numeric main predictors and numeric controls. Non-numeric variable(s): {', '.join(non_numeric_x)}")

    if y_col in x_cols:
        errors.append("The dependent variable cannot also be used as an independent variable.")
    if y_col in categorical_controls:
        errors.append("The dependent variable cannot also be used as a categorical control variable.")

    missing_categorical = [col for col in categorical_controls if col not in df.columns]
    if missing_categorical:
        errors.append(f"Categorical control variable(s) not found: {', '.join(missing_categorical)}")

    constant_x = [col for col in x_cols if col in df.columns and df[col].nunique(dropna=True) <= 1]
    if constant_x:
        errors.append(f"{model_label} cannot use constant-only predictor(s): " + ", ".join(constant_x))

    selected_columns = [y_col] + [col for col in x_cols if col in df.columns]
    if encode_categoricals:
        selected_columns += [col for col in categorical_controls if col in df.columns]
    complete_rows = df[selected_columns].dropna()
    if len(complete_rows) <= len(x_cols) + 1:
        errors.append("Not enough observations to estimate this model.")

    return errors


def prepare_binary_choice_dataframe(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, str, list[str], dict[str, Any], dict[str, Any]]:
    y_col = config["dependent_variable"]
    numeric_predictors = _numeric_predictors(config)
    categorical_controls = _categorical_control_cols(config)
    encode_categoricals = bool(config.get("encode_categorical_controls", False))

    selected_columns = [y_col] + numeric_predictors
    if encode_categoricals:
        selected_columns += categorical_controls
    selected_columns = list(dict.fromkeys(selected_columns))

    original_rows = len(df)
    working = df[selected_columns].copy()
    working[y_col], binary_info = coerce_binary_series(working[y_col])
    working = working.dropna().copy()

    encoding_info = {
        "enabled": encode_categoricals,
        "selected_categorical_controls": categorical_controls,
        "encoded_categorical_controls": [],
        "reference_categories": {},
        "dummy_variables": [],
        "ignored_categorical_controls": [] if encode_categoricals else categorical_controls,
        "high_cardinality_warnings": [],
        "binary_y_info": binary_info,
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

    cleaning_log = {
        "original_row_count": original_rows,
        "final_row_count": len(working),
        "dropped_row_count": original_rows - len(working),
        "dropped_row_percentage": round(((original_rows - len(working)) / original_rows * 100), 2) if original_rows else 0.0,
    }
    return working, y_col, x_cols, cleaning_log, encoding_info


def fit_binary_choice_model(
    df: pd.DataFrame,
    config: dict[str, Any],
    model_label: str,
    model_type: str,
    model_factory: Callable[[pd.Series, pd.DataFrame], Any],
) -> dict[str, Any]:
    errors = validate_binary_choice_inputs(df, config, model_label)
    if errors:
        raise ValueError(errors[0])

    model_df, y_col, x_cols, cleaning_log, encoding_info = prepare_binary_choice_dataframe(df, config)
    if model_df[y_col].nunique(dropna=True) != 2:
        raise ValueError("The dependent variable has only one outcome class after dropping missing values.")

    y = model_df[y_col].astype(float)
    x = sm.add_constant(model_df[x_cols].astype(float), has_constant="add")
    model_warnings: list[str] = []

    try:
        with py_warnings.catch_warnings(record=True) as caught:
            py_warnings.simplefilter("always", ConvergenceWarning)
            fitted = model_factory(y, x).fit(disp=False, maxiter=200)
        for warning in caught:
            if issubclass(warning.category, ConvergenceWarning):
                model_warnings.append("The model did not fully converge. Interpret the estimates cautiously.")
    except PerfectSeparationError as exc:
        raise ValueError("Perfect separation detected. One or more predictors may perfectly predict the outcome.") from exc
    except np.linalg.LinAlgError as exc:
        raise ValueError("The model could not be estimated because the design matrix is singular.") from exc

    conf_int = fitted.conf_int()
    params = pd.Series(fitted.params, index=x.columns)
    regression_table = pd.DataFrame(
        {
            "variable": x.columns,
            "coefficient": params.values,
            "std_error": pd.Series(fitted.bse, index=x.columns).values,
            "z_value": pd.Series(fitted.tvalues, index=x.columns).values,
            "p_value": pd.Series(fitted.pvalues, index=x.columns).values,
            "conf_int_low": conf_int[0].values,
            "conf_int_high": conf_int[1].values,
        }
    )
    regression_table["significance"] = regression_table["p_value"].apply(significance_stars)
    advanced_outputs: dict[str, Any] = {}
    if model_type == "logit" and bool(config.get("include_odds_ratios", True)):
        advanced_outputs["odds_ratio_table"] = _odds_ratio_table(regression_table)
    if bool(config.get("include_marginal_effects", True)):
        marginal_effects_type = str(config.get("marginal_effects_type") or "average")
        try:
            advanced_outputs["marginal_effects_table"] = _marginal_effects_table(fitted, marginal_effects_type)
            advanced_outputs["marginal_effects_type"] = "average" if marginal_effects_type == "average" else "at_means"
        except Exception as exc:
            advanced_outputs["marginal_effects_error"] = str(exc)
            model_warnings.append(f"Marginal effects could not be computed: {exc}")

    model_summary = {
        "model_type": model_type,
        "dependent_variable": y_col,
        "independent_variables": list(x_cols),
        "main_independent_variables": _main_x_cols(config),
        "numeric_control_variables": _numeric_control_cols(config),
        "categorical_control_variables": _categorical_control_cols(config),
        "encoded_categorical_controls": encoding_info["encoded_categorical_controls"],
        "dummy_variables": encoding_info["dummy_variables"],
        "reference_categories": encoding_info["reference_categories"],
        "n_obs": int(fitted.nobs),
        "num_predictors": len(x_cols),
        "log_likelihood": float(fitted.llf) if fitted.llf is not None else None,
        "pseudo_r_squared": float(fitted.prsquared) if hasattr(fitted, "prsquared") else None,
        "aic": float(fitted.aic) if fitted.aic is not None else None,
        "bic": float(fitted.bic) if fitted.bic is not None else None,
        "converged": bool(fitted.mle_retvals.get("converged", True)) if hasattr(fitted, "mle_retvals") else True,
        "robust_standard_errors": False,
        "marginal_effects_type": advanced_outputs.get("marginal_effects_type"),
    }
    if not model_summary["converged"]:
        model_warnings.append("The model did not fully converge. Interpret the estimates cautiously.")

    return {
        "fitted_model": fitted,
        "regression_table": regression_table,
        "model_summary": model_summary,
        "cleaned_df": model_df,
        "cleaning_log": cleaning_log,
        "encoding_info": encoding_info,
        "warnings": model_warnings,
        "advanced_outputs": advanced_outputs,
    }


def _odds_ratio_table(regression_table: pd.DataFrame) -> pd.DataFrame:
    table = regression_table.copy()
    return pd.DataFrame(
        {
            "variable": table["variable"],
            "odds_ratio": np.exp(table["coefficient"].astype(float)),
            "conf_int_low": np.exp(table["conf_int_low"].astype(float)),
            "conf_int_high": np.exp(table["conf_int_high"].astype(float)),
            "p_value": table["p_value"],
            "significance": table["significance"],
        }
    )


def _marginal_effects_table(fitted: Any, marginal_effects_type: str) -> pd.DataFrame:
    at = "overall" if marginal_effects_type == "average" else "mean"
    frame = fitted.get_margeff(at=at).summary_frame().reset_index(names="variable")
    rename_map = {
        "dy/dx": "marginal_effect",
        "Std. Err.": "std_error",
        "z": "z_value",
        "Pr(>|z|)": "p_value",
        "Conf. Int. Low": "conf_int_low",
        "Cont. Int. Hi.": "conf_int_high",
        "Conf. Int. Hi.": "conf_int_high",
    }
    frame = frame.rename(columns=rename_map)
    columns = [
        column
        for column in [
            "variable",
            "marginal_effect",
            "std_error",
            "z_value",
            "p_value",
            "conf_int_low",
            "conf_int_high",
        ]
        if column in frame.columns
    ]
    table = frame[columns].copy()
    if "p_value" in table.columns:
        table["significance"] = table["p_value"].apply(significance_stars)
    return table


def diagnose_binary_choice_model(
    df_original: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    config: dict[str, Any],
    fit_result: dict[str, Any],
) -> dict[str, Any]:
    warnings = list(fit_result.get("warnings", []))
    summary = fit_result["model_summary"]
    x_cols = list(summary.get("independent_variables", []))
    y_col = summary["dependent_variable"]
    n_obs = len(df_cleaned)

    if df_cleaned[y_col].nunique(dropna=True) < 2:
        warnings.append("Only one outcome class remains after dropping missing values.")
    if n_obs < 30:
        warnings.append("Sample size is below 30 observations. Binary response model estimates may be unstable.")

    original_rows = len(df_original)
    if original_rows:
        dropped_pct = (original_rows - n_obs) / original_rows * 100
        if dropped_pct > 30:
            warnings.append(f"Missing values reduced the model sample by {dropped_pct:.1f}%, which may bias results.")

    if x_cols and n_obs <= len(x_cols) * 10:
        warnings.append("There are many predictors relative to the sample size. Consider a simpler model or more data.")

    if not summary.get("converged", True):
        warnings.append("The model did not fully converge. Interpret the estimates cautiously.")

    warnings.extend(fit_result.get("encoding_info", {}).get("high_cardinality_warnings", []))

    if len(x_cols) >= 2:
        vif_df = calculate_vif(df_cleaned, x_cols)
        high_vif = vif_df[vif_df["VIF"] > 5]
        serious_vif = vif_df[vif_df["VIF"] > 10]
        if not high_vif.empty:
            warnings.append("High multicollinearity detected: VIF above 5 for " + ", ".join(high_vif["variable"].astype(str).tolist()) + ".")
        if not serious_vif.empty:
            warnings.append("Serious multicollinearity detected: VIF above 10 for " + ", ".join(serious_vif["variable"].astype(str).tolist()) + ".")
    else:
        vif_df = pd.DataFrame(columns=["variable", "VIF"])

    return {"vif_df": vif_df, "warnings": warnings}
