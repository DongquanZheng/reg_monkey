from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

from src.models.base import ModelDefinition
from src.models.diagnostics import DiagnosticCode, DiagnosticSeverity, ModelDiagnostic
from src.utils import significance_stars


IV_MODEL_ID = "iv_2sls"
IV_FITTED_PREFIX = "fitted_"
WEAK_FIRST_STAGE_F_THRESHOLD = 10.0
LOW_FIRST_STAGE_R2_THRESHOLD = 0.05


def validate_iv_inputs(df: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    from src.models.execution import ModelSpec
    from src.models.iv_contract import validate_iv_spec_contract

    spec = ModelSpec.from_config(IV_MODEL_ID, config)
    validation = validate_iv_spec_contract(df, spec)
    errors = list(validation.errors)
    if errors:
        return errors

    selected = _selected_columns(config)
    working = df[selected].dropna().copy()
    if working.empty:
        return ["No complete observations remain after dropping missing values for the IV/2SLS specification."]

    non_numeric = [column for column in selected if column in working.columns and not pd.api.types.is_numeric_dtype(working[column])]
    if non_numeric:
        errors.append("Minimal IV/2SLS requires numeric variables. Non-numeric variable(s): " + ", ".join(non_numeric) + ".")

    instruments = _instrument_cols(config)
    controls = _control_cols(config)
    first_stage_predictors = [*instruments, *controls]
    second_stage_predictors = [_iv_term(config), *controls]
    if len(working) <= max(len(first_stage_predictors), len(second_stage_predictors)) + 1:
        errors.append("Not enough observations to estimate this IV/2SLS model.")

    if not errors:
        first_stage_design = _design_matrix(working, first_stage_predictors)
        second_stage_probe = working[[config["endogenous_variable"], *controls]].copy()
        second_stage_probe[_iv_term(config)] = pd.to_numeric(working[config["endogenous_variable"]], errors="coerce")
        second_stage_design = _design_matrix(second_stage_probe, second_stage_predictors)
        if _is_rank_deficient(first_stage_design):
            errors.append("The IV/2SLS first-stage design matrix is collinear or rank deficient.")
        if _is_rank_deficient(second_stage_design):
            errors.append("The IV/2SLS second-stage design matrix is collinear or rank deficient.")
        if working[config["endogenous_variable"]].nunique(dropna=True) <= 1:
            errors.append("The IV/2SLS endogenous variable has no usable variation.")

    return errors


def prepare_iv_dataframe(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, str, str, list[str], list[str], dict[str, Any]]:
    y_col = config["dependent_variable"]
    endogenous_col = config["endogenous_variable"]
    instruments = _instrument_cols(config)
    controls = _control_cols(config)
    selected = list(dict.fromkeys([y_col, endogenous_col, *instruments, *controls]))
    original_rows = len(df)
    working = df[selected].dropna().copy()
    for column in selected:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.dropna().copy()
    cleaning_log = {
        "original_row_count": original_rows,
        "final_row_count": len(working),
        "dropped_row_count": original_rows - len(working),
        "dropped_row_percentage": round(((original_rows - len(working)) / original_rows * 100), 2) if original_rows else 0.0,
    }
    return working, y_col, endogenous_col, instruments, controls, cleaning_log


def fit_iv_2sls(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    errors = validate_iv_inputs(df, config)
    if errors:
        raise ValueError(errors[0])

    model_df, y_col, endogenous_col, instruments, controls, cleaning_log = prepare_iv_dataframe(df, config)
    first_stage_x_cols = [*instruments, *controls]
    first_stage_x = _design_matrix(model_df, first_stage_x_cols)
    first_stage_y = model_df[endogenous_col].astype(float)
    first_stage = sm.OLS(first_stage_y, first_stage_x).fit()

    iv_term = _iv_term(config)
    model_df[iv_term] = first_stage.fittedvalues.astype(float)
    if not _has_usable_variation(model_df[iv_term]):
        raise ValueError("The fitted endogenous variable has no usable variation after the first stage.")
    second_stage_x_cols = [iv_term, *controls]
    second_stage_x = _design_matrix(model_df, second_stage_x_cols)
    second_stage_y = model_df[y_col].astype(float)
    second_stage = sm.OLS(second_stage_y, second_stage_x).fit()
    iv_fit = _fit_linearmodels_iv_2sls(model_df, y_col, endogenous_col, instruments, controls)

    regression_table = _iv_regression_table(iv_fit, endogenous_col, iv_term, controls)
    iv_row = regression_table[regression_table["variable"] == iv_term].iloc[0]
    first_stage_f = _first_stage_f_statistic(first_stage, first_stage_x.columns, instruments)
    weak_flag = first_stage_f is None or pd.isna(first_stage_f) or float(first_stage_f) < WEAK_FIRST_STAGE_F_THRESHOLD
    assumption_notes = [
        "instrument_relevance_required",
        "exclusion_restriction_required_not_tested",
        "conditional_iv_estimate_no_automatic_causal_claim",
        "single_endogenous_variable_minimal_runner",
    ]
    model_summary = {
        "model_type": IV_MODEL_ID,
        "dependent_variable": y_col,
        "endogenous_variable": endogenous_col,
        "instrument_variable": instruments[0] if instruments else "",
        "instruments": list(instruments),
        "exogenous_controls": list(controls),
        "numeric_control_variables": list(controls),
        "main_independent_variables": [iv_term],
        "independent_variables": list(second_stage_x_cols),
        "standard_errors": "2sls_unadjusted",
        "covariance_estimator": "linearmodels_iv2sls_unadjusted",
        "r_squared": float(second_stage.rsquared),
        "adj_r_squared": float(second_stage.rsquared_adj),
        "n_obs": int(second_stage.nobs),
        "num_predictors": len(second_stage_x_cols),
        "iv_term": iv_term,
        "iv_estimate": float(iv_row["coefficient"]),
        "iv_p_value": float(iv_row["p_value"]),
        "first_stage_r_squared": float(first_stage.rsquared),
        "first_stage_f_statistic": first_stage_f,
        "first_stage_nobs": int(first_stage.nobs),
        "second_stage_nobs": int(second_stage.nobs),
        "instrument_count": len(instruments),
        "weak_instrument_rule_of_thumb": bool(weak_flag),
        "assumption_notes": list(assumption_notes),
    }
    return {
        "fitted_model": iv_fit,
        "second_stage_model": second_stage,
        "first_stage_model": first_stage,
        "regression_table": regression_table,
        "model_summary": model_summary,
        "cleaned_df": model_df,
        "cleaning_log": cleaning_log,
        "advanced_outputs": {
            "iv_summary": {
                "endogenous_variable": endogenous_col,
                "instruments": list(instruments),
                "exogenous_controls": list(controls),
                "iv_term": iv_term,
                "iv_estimate": float(iv_row["coefficient"]),
                "standard_error": float(iv_row["std_error"]),
                "p_value": float(iv_row["p_value"]),
                "first_stage_r_squared": float(first_stage.rsquared),
                "first_stage_f_statistic": first_stage_f,
                "first_stage_nobs": int(first_stage.nobs),
                "second_stage_nobs": int(second_stage.nobs),
                "observations_used": int(second_stage.nobs),
                "instrument_count": len(instruments),
                "weak_instrument_rule_of_thumb": bool(weak_flag),
                "assumption_notes": list(assumption_notes),
            }
        },
    }


def diagnose_iv(
    df_original: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    config: dict[str, Any],
    fit_result: Any,
) -> dict[str, Any]:
    summary = dict(fit_result.get("model_summary") or {})
    diagnostics = [
        ModelDiagnostic(
            code=DiagnosticCode.IV_IDENTIFICATION_ASSUMPTION,
            severity=DiagnosticSeverity.CONSTRAINT,
            title="IV/2SLS identification assumptions",
            message="IV/2SLS interpretation depends on instrument relevance and the exclusion restriction; the estimate is not automatically causal.",
            recommendation="Assess instrument validity and research design before interpreting the IV estimate causally.",
            show_in_ui=True,
            show_in_report=True,
            llm_instruction="Explain that IV estimates require valid instruments and do not automatically establish causality.",
        )
    ]
    first_stage_f = summary.get("first_stage_f_statistic")
    first_stage_r2 = summary.get("first_stage_r_squared")
    instruments = list(summary.get("instruments") or [])
    controls = list(summary.get("exogenous_controls") or summary.get("numeric_control_variables") or [])
    diagnostics.extend(
        [
            ModelDiagnostic(
                code=DiagnosticCode.IV_INSTRUMENT_COUNT,
                severity=DiagnosticSeverity.INFO,
                title="IV instrument count",
                message=f"The minimal IV/2SLS run uses {len(instruments)} instrument(s): " + (", ".join(instruments) if instruments else "none") + ".",
                affected_variables=instruments,
                recommendation="Confirm that each instrument is relevant and satisfies the exclusion restriction.",
                show_in_ui=False,
                show_in_report=True,
                llm_instruction="State the number of instruments as setup metadata, not as proof of validity.",
            ),
            ModelDiagnostic(
                code=DiagnosticCode.IV_CONTROLS_INCLUDED,
                severity=DiagnosticSeverity.INFO,
                title="Controls included in both stages",
                message="The same exogenous control variable(s) are included in both IV/2SLS stages: " + (", ".join(controls) if controls else "none") + ".",
                affected_variables=controls,
                recommendation="Confirm that these controls match the research design.",
                show_in_ui=False,
                show_in_report=True,
                llm_instruction="Mention controls only as model setup context.",
            ),
            ModelDiagnostic(
                code=DiagnosticCode.IV_SINGLE_ENDOGENOUS_LIMITATION,
                severity=DiagnosticSeverity.CONSTRAINT,
                title="Single endogenous regressor limitation",
                message="This minimal IV/2SLS runner supports one endogenous regressor.",
                recommendation="Use this result only for the current single-endogenous-variable specification.",
                show_in_ui=False,
                show_in_report=True,
                llm_instruction="Mention the single-endogenous-variable limitation if relevant.",
            ),
        ]
    )
    if first_stage_f is None or pd.isna(first_stage_f):
        diagnostics.append(
            ModelDiagnostic(
                code=DiagnosticCode.IV_FIRST_STAGE_UNAVAILABLE,
                severity=DiagnosticSeverity.WARNING,
                title="First-stage strength unavailable",
                message="The first-stage F-statistic could not be computed for this minimal IV/2SLS run.",
                recommendation="Review first-stage relevance before interpreting the IV estimate.",
                show_in_ui=True,
                show_in_report=True,
                llm_instruction="Mention that first-stage strength is unavailable and IV interpretation should be cautious.",
            )
        )
    elif float(first_stage_f) < WEAK_FIRST_STAGE_F_THRESHOLD:
        diagnostics.append(
            ModelDiagnostic(
                code=DiagnosticCode.IV_FIRST_STAGE_STRENGTH,
                severity=DiagnosticSeverity.WARNING,
                title="Weak first-stage indication",
                message=f"The first-stage F-statistic is {float(first_stage_f):.3f}, below the conventional rule-of-thumb threshold of {WEAK_FIRST_STAGE_F_THRESHOLD:.0f}.",
                recommendation="Treat the IV estimate cautiously and review instrument relevance.",
                show_in_ui=True,
                show_in_report=True,
                llm_instruction="Warn that the instrument may be weak and the IV estimate may be unstable.",
            )
        )
    if first_stage_r2 is not None and not pd.isna(first_stage_r2) and float(first_stage_r2) < LOW_FIRST_STAGE_R2_THRESHOLD:
        diagnostics.append(
            ModelDiagnostic(
                code=DiagnosticCode.IV_LOW_FIRST_STAGE_R2,
                severity=DiagnosticSeverity.WARNING,
                title="Low first-stage R-squared",
                message=f"The first-stage R-squared is {float(first_stage_r2):.3f}, indicating limited explained variation in the endogenous variable.",
                recommendation="Review instrument relevance before interpreting the IV estimate.",
                show_in_ui=True,
                show_in_report=True,
                llm_instruction="Warn that low first-stage fit can weaken IV interpretation.",
            )
        )
    else:
        diagnostics.append(
            ModelDiagnostic(
                code=DiagnosticCode.IV_FIRST_STAGE_STRENGTH,
                severity=DiagnosticSeverity.INFO,
                title="First-stage strength note",
                message=f"The first-stage F-statistic is {float(first_stage_f):.3f}.",
                recommendation="Still review instrument validity and exclusion restriction assumptions.",
                show_in_ui=False,
                show_in_report=True,
                llm_instruction="Mention first-stage strength only as supporting diagnostic context, not proof of instrument validity.",
            )
        )
    return {"structured_diagnostics": diagnostics, "warnings": []}


def _instrument_cols(config: dict[str, Any]) -> list[str]:
    instruments = list(config.get("instruments") or [])
    instrument_variable = str(config.get("instrument_variable") or "")
    if instrument_variable:
        instruments.insert(0, instrument_variable)
    return list(dict.fromkeys([str(item) for item in instruments if str(item or "").strip()]))


def _control_cols(config: dict[str, Any]) -> list[str]:
    controls = list(config.get("exogenous_controls") or config.get("numeric_control_variables") or [])
    return list(dict.fromkeys([str(item) for item in controls if str(item or "").strip()]))


def _selected_columns(config: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys([config.get("dependent_variable"), config.get("endogenous_variable"), *_instrument_cols(config), *_control_cols(config)]))


def _design_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return sm.add_constant(df[columns].astype(float), has_constant="add")


def _fit_linearmodels_iv_2sls(
    df: pd.DataFrame,
    y_col: str,
    endogenous_col: str,
    instruments: list[str],
    controls: list[str],
) -> Any:
    exog = sm.add_constant(df[controls].astype(float), has_constant="add") if controls else pd.DataFrame({"const": 1.0}, index=df.index)
    return IV2SLS(
        dependent=df[y_col].astype(float),
        exog=exog,
        endog=df[endogenous_col].astype(float),
        instruments=df[instruments].astype(float),
    ).fit(cov_type="unadjusted")


def _is_rank_deficient(design: pd.DataFrame) -> bool:
    matrix = design.to_numpy(dtype=float)
    return bool(np.linalg.matrix_rank(matrix) < matrix.shape[1])


def _has_usable_variation(series: pd.Series, tolerance: float = 1e-10) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) <= 1:
        return False
    return bool(np.nanmax(values) - np.nanmin(values) > tolerance)


def _iv_term(config: dict[str, Any]) -> str:
    return f"{IV_FITTED_PREFIX}{config['endogenous_variable']}"


def _first_stage_f_statistic(result: Any, columns: pd.Index, instruments: list[str]) -> float | None:
    if not instruments:
        return None
    try:
        restrictions = np.zeros((len(instruments), len(columns)))
        for row_index, instrument in enumerate(instruments):
            restrictions[row_index, list(columns).index(instrument)] = 1.0
        f_test = result.f_test(restrictions)
        return float(np.asarray(f_test.fvalue).reshape(-1)[0])
    except Exception:
        return None


def _regression_table(result: Any, columns: pd.Index) -> pd.DataFrame:
    params = pd.Series(result.params, index=columns)
    std_errors = pd.Series(result.bse, index=columns)
    t_values = pd.Series(result.tvalues, index=columns)
    p_values = pd.Series(result.pvalues, index=columns)
    table = pd.DataFrame(
        {
            "variable": columns,
            "coefficient": params.values,
            "std_error": std_errors.values,
            "t_value": t_values.values,
            "p_value": p_values.values,
        }
    )
    conf_int = pd.DataFrame(result.conf_int(), index=columns)
    table["conf_int_low"] = conf_int.iloc[:, 0].values
    table["conf_int_high"] = conf_int.iloc[:, 1].values
    table["significance"] = table["p_value"].apply(significance_stars)
    return table


def _iv_regression_table(result: Any, endogenous_col: str, iv_term: str, controls: list[str]) -> pd.DataFrame:
    ordered_variables = ["const", iv_term, *controls]
    conf_int = result.conf_int()
    rows: list[dict[str, Any]] = []
    for output_variable in ordered_variables:
        source_variable = endogenous_col if output_variable == iv_term else output_variable
        if source_variable not in result.params.index:
            continue
        rows.append(
            {
                "variable": output_variable,
                "coefficient": float(result.params[source_variable]),
                "std_error": float(result.std_errors[source_variable]),
                "t_value": float(result.tstats[source_variable]),
                "p_value": float(result.pvalues[source_variable]),
                "conf_int_low": float(conf_int.loc[source_variable].iloc[0]),
                "conf_int_high": float(conf_int.loc[source_variable].iloc[1]),
            }
        )
    table = pd.DataFrame(rows)
    table["significance"] = table["p_value"].apply(significance_stars)
    return table


IV_2SLS_MODEL = ModelDefinition(
    model_id=IV_MODEL_ID,
    display_name_en="IV / 2SLS",
    display_name_zh="工具变量 / 两阶段最小二乘",
    description_en="Estimate a minimal two-stage least squares model with one endogenous regressor and one or more instruments.",
    description_zh="估计包含一个内生变量和一个或多个工具变量的最小两阶段最小二乘模型。",
    required_roles=["dependent_variable", "endogenous_variable", "instruments"],
    validate=validate_iv_inputs,
    fit=fit_iv_2sls,
    diagnostics=diagnose_iv,
    report_label_en="IV / 2SLS",
    report_label_zh="工具变量 / 两阶段最小二乘",
    limitations_en=[
        "This minimal IV/2SLS runner does not test the exclusion restriction.",
        "First-stage strength should be reviewed before interpreting the IV estimate.",
        "The IV estimate does not automatically establish causality.",
    ],
    limitations_zh=[
        "当前最小 IV/2SLS 运行器不会检验排除限制。",
        "解释 IV 估计前应检查第一阶段强度。",
        "IV 估计不能自动证明因果关系。",
    ],
)
