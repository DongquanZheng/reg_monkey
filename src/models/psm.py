from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from src.models.base import ModelDefinition
from src.models.diagnostics import DiagnosticCode, DiagnosticSeverity, ModelDiagnostic


PSM_MODEL_ID = "psm"
PSM_ATT_TERM = "att_estimate"
PSM_SMALL_MATCHED_SAMPLE_THRESHOLD = 10
PSM_POOR_OVERLAP_UNMATCHED_SHARE = 0.20
PSM_BALANCE_TOLERANCE = 1e-9
PSM_HIGH_RESIDUAL_SMD_THRESHOLD = 0.10


def validate_psm_inputs(df: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    from src.models.execution import ModelSpec
    from src.models.psm_contract import validate_psm_spec_contract

    spec = ModelSpec.from_config(PSM_MODEL_ID, config)
    validation = validate_psm_spec_contract(df, spec)
    errors = list(validation.errors)
    if errors:
        return errors

    estimand = _estimand(config)
    if estimand != "ATT":
        errors.append("The minimal PSM runner currently supports ATT only.")

    selected = _selected_columns(config)
    working = df[selected].dropna().copy()
    if working.empty:
        errors.append("No complete observations remain after dropping missing values for the PSM specification.")
        return errors

    non_numeric = _non_numeric_columns(working, selected)
    if non_numeric:
        errors.append("Minimal PSM requires numeric outcome, treatment, and matching covariates. Non-numeric variable(s): " + ", ".join(non_numeric) + ".")
        return errors

    numeric = _numeric_frame(working, selected).dropna().copy()
    if numeric.empty:
        errors.append("No numeric complete observations remain for the PSM specification.")
        return errors

    treatment_col = config["treatment_variable"]
    treatment = numeric[treatment_col]
    if not _is_binary_numeric(treatment):
        errors.append(f"PSM treatment variable must be binary 0/1 after numeric conversion: {treatment_col}.")
        return errors

    treated_count = int((treatment == 1).sum())
    control_count = int((treatment == 0).sum())
    covariate_count = len(_matching_covariates(config))
    if treated_count <= 0:
        errors.append("PSM requires at least one treated observation.")
    if control_count <= 0:
        errors.append("PSM requires at least one control observation.")
    if len(numeric) <= covariate_count + 2:
        errors.append("Not enough observations to estimate the PSM propensity score model.")

    if not errors:
        design = _design_matrix(numeric, _matching_covariates(config))
        if _is_rank_deficient(design):
            errors.append("The PSM propensity score design matrix is collinear or rank deficient.")

    return errors


def prepare_psm_dataframe(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, str, str, list[str], dict[str, Any]]:
    y_col = config["dependent_variable"]
    treatment_col = config["treatment_variable"]
    covariates = _matching_covariates(config)
    selected = _selected_columns(config)
    original_rows = len(df)
    working = df[selected].dropna().copy()
    working = _numeric_frame(working, selected).dropna().copy()
    working[treatment_col] = working[treatment_col].astype(float)
    cleaning_log = {
        "original_row_count": original_rows,
        "final_row_count": len(working),
        "dropped_row_count": original_rows - len(working),
        "dropped_row_percentage": round(((original_rows - len(working)) / original_rows * 100), 2) if original_rows else 0.0,
    }
    return working, y_col, treatment_col, covariates, cleaning_log


def fit_psm(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    errors = validate_psm_inputs(df, config)
    if errors:
        raise ValueError(errors[0])

    model_df, y_col, treatment_col, covariates, cleaning_log = prepare_psm_dataframe(df, config)
    treatment = model_df[treatment_col].astype(float)
    propensity_model = _fit_propensity_model(model_df, treatment_col, covariates)
    propensity_scores = pd.Series(propensity_model.predict(_design_matrix(model_df, covariates)), index=model_df.index, name="propensity_score")
    if not np.isfinite(propensity_scores.to_numpy(dtype=float)).all():
        raise ValueError("PSM propensity scores are unavailable or non-finite.")

    model_df["propensity_score"] = propensity_scores.astype(float)
    pairs = _nearest_neighbor_att_pairs(model_df, y_col, treatment_col, propensity_scores, _caliper(config))
    if pairs.empty:
        if _caliper(config) is not None:
            raise ValueError("The PSM caliper eliminated all treated observations; no valid control matches remain.")
        raise ValueError("PSM could not find any valid control matches.")

    att_estimate = float(pairs["att_component"].mean())
    matched_treated_count = int(len(pairs))
    matched_control_count = int(len(pairs))
    unmatched_treated_count = int((model_df[treatment_col] == 1).sum() - matched_treated_count)
    treated_outcome = float(pairs["treated_outcome"].mean())
    matched_control_outcome = float(pairs["control_outcome"].mean())
    balance_summary = _balance_summary(model_df, pairs, treatment_col, covariates)
    balance_overview = _balance_overview(balance_summary)
    propensity_summary = _propensity_score_summary(model_df, treatment_col)
    assumption_notes = [
        "observed_covariates_only",
        "does_not_address_unobserved_confounding",
        "att_only_minimal_runner",
        "nearest_neighbor_with_replacement",
    ]

    psm_summary = {
        "treatment_variable": treatment_col,
        "outcome_variable": y_col,
        "matching_covariates": list(covariates),
        "estimand": _estimand(config),
        "matching_method": _matching_method(config),
        "caliper": _caliper(config),
        "replacement": True,
        "replacement_matching": True,
        "observations_used": int(len(model_df)),
        "treated_count": int((model_df[treatment_col] == 1).sum()),
        "control_count": int((model_df[treatment_col] == 0).sum()),
        "matched_treated_count": matched_treated_count,
        "matched_control_count": matched_control_count,
        "unique_matched_control_count": int(pairs["control_index"].nunique()),
        "unmatched_treated_count": unmatched_treated_count,
        "att_estimate": att_estimate,
        "mean_outcome_treated": treated_outcome,
        "mean_outcome_matched_control": matched_control_outcome,
        "assumption_notes": assumption_notes,
        **balance_overview,
    }
    regression_table = pd.DataFrame(
        [
            {
                "variable": PSM_ATT_TERM,
                "coefficient": att_estimate,
                "std_error": None,
                "t_value": None,
                "p_value": None,
                "conf_int_low": None,
                "conf_int_high": None,
                "significance": "",
            }
        ]
    )
    model_summary = {
        "model_type": PSM_MODEL_ID,
        "dependent_variable": y_col,
        "treatment_variable": treatment_col,
        "matching_covariates": list(covariates),
        "psm_estimand": _estimand(config),
        "matching_method": _matching_method(config),
        "caliper": _caliper(config),
        "main_independent_variables": [PSM_ATT_TERM],
        "independent_variables": list(covariates),
        "numeric_control_variables": list(covariates),
        "standard_errors": "not_estimated",
        "n_obs": int(len(model_df)),
        "num_predictors": len(covariates),
        "treated_count": psm_summary["treated_count"],
        "control_count": psm_summary["control_count"],
        "matched_treated_count": matched_treated_count,
        "matched_control_count": matched_control_count,
        "unmatched_treated_count": unmatched_treated_count,
        "replacement_matching": True,
        "att_estimate": att_estimate,
        "mean_outcome_treated": treated_outcome,
        "mean_outcome_matched_control": matched_control_outcome,
        "balance_summary_available": bool(balance_summary),
        "assumption_notes": assumption_notes,
        **balance_overview,
    }
    return {
        "fitted_model": propensity_model,
        "regression_table": regression_table,
        "model_summary": model_summary,
        "cleaned_df": model_df,
        "cleaning_log": cleaning_log,
        "advanced_outputs": {
            "psm_summary": psm_summary,
            "propensity_score_summary": propensity_summary,
            "balance_summary": balance_summary,
            "psm_balance_overview": balance_overview,
        },
    }


def diagnose_psm(df_original: pd.DataFrame, df_cleaned: pd.DataFrame, config: dict[str, Any], fit_result: Any) -> dict[str, Any]:
    summary = dict(fit_result.get("model_summary") or {})
    advanced = dict(fit_result.get("advanced_outputs") or {})
    psm_summary = dict(advanced.get("psm_summary") or {})
    propensity_summary = dict(advanced.get("propensity_score_summary") or {})
    balance = list(advanced.get("balance_summary") or [])
    balance_overview = dict(advanced.get("psm_balance_overview") or {})
    diagnostics: list[ModelDiagnostic] = [
        ModelDiagnostic(
            code=DiagnosticCode.PSM_IDENTIFICATION_ASSUMPTION_OBSERVED_COVARIATES,
            severity=DiagnosticSeverity.CONSTRAINT,
            title="PSM observed-covariate assumption",
            message="PSM improves comparability only on observed matching covariates; it does not address unobserved confounding.",
            recommendation="Interpret the ATT as conditional on the selected observed covariates and research design.",
            show_in_ui=True,
            show_in_report=True,
            llm_instruction="Explain that PSM does not remove unobserved confounding and should not be presented as automatic causal proof.",
        ),
        ModelDiagnostic(
            code=DiagnosticCode.PSM_NO_UNOBSERVED_CONFOUNDING_CAUTION,
            severity=DiagnosticSeverity.CONSTRAINT,
            title="No automatic causal interpretation",
            message="The PSM ATT estimate is not automatically causal; it depends on selection-on-observables and design assumptions.",
            recommendation="Justify the covariate set and compare pre/post-match balance before using causal language.",
            show_in_ui=True,
            show_in_report=True,
            llm_instruction="Do not claim a causal effect unless the design assumptions are explicitly justified by the user.",
        ),
    ]

    treated_count = int(psm_summary.get("treated_count") or 0)
    control_count = int(psm_summary.get("control_count") or 0)
    matched_count = int(psm_summary.get("matched_treated_count") or 0)
    unmatched_count = int(psm_summary.get("unmatched_treated_count") or 0)
    if psm_summary.get("replacement_matching") or psm_summary.get("replacement"):
        diagnostics.append(
            ModelDiagnostic(
                code=DiagnosticCode.PSM_REPLACEMENT_MATCHING_NOTICE,
                severity=DiagnosticSeverity.INFO,
                title="Replacement matching used",
                message="This minimal PSM run uses 1:1 nearest-neighbor matching with replacement.",
                recommendation="Interpret matched-control counts with replacement matching in mind.",
                show_in_ui=False,
                show_in_report=True,
                llm_instruction="Mention replacement matching when explaining the PSM matched sample.",
            )
        )
    if str(psm_summary.get("estimand") or _estimand(config)).upper() == "ATT":
        diagnostics.append(
            ModelDiagnostic(
                code=DiagnosticCode.PSM_ATT_ONLY_NOTICE,
                severity=DiagnosticSeverity.CONSTRAINT,
                title="ATT-only minimal PSM",
                message="This minimal PSM runner reports ATT only; ATE and other matching estimands are not estimated.",
                recommendation="Do not describe this result as ATE or a population-wide treatment effect.",
                show_in_ui=False,
                show_in_report=True,
                llm_instruction="Explain that the PSM result is ATT under the selected matching specification.",
            )
        )
    if propensity_summary:
        diagnostics.append(
            ModelDiagnostic(
                code=DiagnosticCode.PSM_COMMON_SUPPORT_SUMMARY,
                severity=DiagnosticSeverity.INFO,
                title="PSM common-support summary",
                message=(
                    "Propensity scores range from "
                    f"{_fmt(propensity_summary.get('min'))} to {_fmt(propensity_summary.get('max'))}; "
                    f"treated mean {_fmt(propensity_summary.get('treated_mean'))}, control mean {_fmt(propensity_summary.get('control_mean'))}."
                ),
                recommendation="Review overlap before interpreting matched comparisons.",
                show_in_ui=False,
                show_in_report=True,
                llm_instruction="Use propensity-score overlap as a design diagnostic, not as proof of causal identification.",
            )
        )
    if balance:
        diagnostics.append(
            ModelDiagnostic(
                code=DiagnosticCode.PSM_BALANCE_SUMMARY_AVAILABLE,
                severity=DiagnosticSeverity.INFO,
                title="Balance diagnostics available",
                message="Before/after standardized mean-difference balance diagnostics are available for the matching covariates.",
                recommendation="Review balance diagnostics before interpreting the ATT.",
                show_in_ui=False,
                show_in_report=True,
                llm_instruction="Use the balance diagnostics to qualify PSM interpretation.",
            )
        )
    high_residual = list(balance_overview.get("high_residual_imbalance_variables") or [])
    if high_residual:
        diagnostics.append(
            ModelDiagnostic(
                code=DiagnosticCode.PSM_HIGH_RESIDUAL_IMBALANCE,
                severity=DiagnosticSeverity.WARNING,
                title="Residual imbalance after matching",
                message=(
                    "Post-match absolute standardized mean differences remain above "
                    f"{PSM_HIGH_RESIDUAL_SMD_THRESHOLD:.2f} for: " + ", ".join(high_residual) + "."
                ),
                affected_variables=high_residual,
                recommendation="Review the matching covariates and consider whether the matched comparison is sufficiently balanced.",
                show_in_ui=True,
                show_in_report=True,
                llm_instruction="Mention high residual imbalance as a PSM limitation.",
            )
        )
    if treated_count and unmatched_count / treated_count > PSM_POOR_OVERLAP_UNMATCHED_SHARE:
        diagnostics.append(
            ModelDiagnostic(
                code=DiagnosticCode.PSM_POOR_OVERLAP_WARNING,
                severity=DiagnosticSeverity.WARNING,
                title="Limited propensity-score overlap",
                message=f"{unmatched_count} of {treated_count} treated observation(s) were not matched under the selected caliper/support.",
                recommendation="Review common support, covariates, and caliper before interpreting the ATT.",
                show_in_ui=True,
                show_in_report=True,
                llm_instruction="Mention limited overlap as a PSM limitation.",
            )
        )
    if control_count < treated_count:
        diagnostics.append(
            ModelDiagnostic(
                code=DiagnosticCode.PSM_SPARSE_CONTROL_POOL,
                severity=DiagnosticSeverity.WARNING,
                title="Sparse control pool",
                message=f"PSM uses {control_count} control observation(s) for {treated_count} treated observation(s).",
                recommendation="Check whether there are enough comparable controls for nearest-neighbor ATT matching.",
                show_in_ui=True,
                show_in_report=True,
                llm_instruction="Mention sparse controls as a match-quality limitation.",
            )
        )
    if matched_count and matched_count < PSM_SMALL_MATCHED_SAMPLE_THRESHOLD:
        diagnostics.append(
            ModelDiagnostic(
                code=DiagnosticCode.PSM_SMALL_MATCHED_SAMPLE,
                severity=DiagnosticSeverity.WARNING,
                title="Small matched sample",
                message=f"Only {matched_count} treated observation(s) were matched, so the ATT may be unstable.",
                recommendation="Use more data or review the matching specification.",
                show_in_ui=True,
                show_in_report=True,
                llm_instruction="Mention small matched sample size as a stability limitation.",
            )
        )
    if _caliper(config) is not None and unmatched_count > 0:
        diagnostics.append(
            ModelDiagnostic(
                code=DiagnosticCode.PSM_CALIPER_DROPS_TREATED_UNITS,
                severity=DiagnosticSeverity.WARNING,
                title="Caliper dropped treated units",
                message=f"The selected caliper left {unmatched_count} treated observation(s) unmatched.",
                recommendation="Review whether the caliper is too restrictive or whether common support is limited.",
                show_in_ui=True,
                show_in_report=True,
                llm_instruction="Explain that the caliper changed the matched treated sample.",
            )
        )
    worsened = [str(row["variable"]) for row in balance if _balance_worsened(row)]
    if worsened:
        diagnostics.append(
            ModelDiagnostic(
                code=DiagnosticCode.PSM_BALANCE_NOT_IMPROVED,
                severity=DiagnosticSeverity.WARNING,
                title="Balance did not improve for all covariates",
                message="Post-match standardized mean differences did not improve for: " + ", ".join(worsened) + ".",
                affected_variables=worsened,
                recommendation="Review balance diagnostics and consider a different covariate set or matching specification.",
                show_in_ui=True,
                show_in_report=True,
                llm_instruction="Mention that matching did not improve balance for all observed covariates.",
            )
        )
    return {"structured_diagnostics": diagnostics, "warnings": []}


def _fit_propensity_model(df: pd.DataFrame, treatment_col: str, covariates: list[str]) -> Any:
    try:
        return sm.Logit(df[treatment_col].astype(float), _design_matrix(df, covariates)).fit(disp=False, maxiter=100)
    except (np.linalg.LinAlgError, PerfectSeparationError, ValueError) as exc:
        raise ValueError(_propensity_failure_message()) from exc
    except Exception as exc:
        raise ValueError(_propensity_failure_message()) from exc


def _propensity_failure_message() -> str:
    return (
        "PSM propensity score model could not be estimated reliably. "
        "Poor treatment-control overlap, sparse support, or collinear matching covariates may be present. "
        "Simplify matching covariates or review propensity-score overlap before interpreting PSM."
    )


def _nearest_neighbor_att_pairs(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    scores: pd.Series,
    caliper: float | None,
) -> pd.DataFrame:
    treated = df[df[treatment_col] == 1]
    controls = df[df[treatment_col] == 0]
    rows: list[dict[str, Any]] = []
    for treated_index, treated_row in treated.iterrows():
        distances = (scores.loc[controls.index] - scores.loc[treated_index]).abs()
        if distances.empty:
            continue
        control_index = distances.sort_values(kind="mergesort").index[0]
        distance = float(distances.loc[control_index])
        if caliper is not None and distance > caliper:
            continue
        control_row = df.loc[control_index]
        treated_outcome = float(treated_row[outcome_col])
        control_outcome = float(control_row[outcome_col])
        rows.append(
            {
                "treated_index": str(treated_index),
                "control_index": str(control_index),
                "distance": distance,
                "treated_outcome": treated_outcome,
                "control_outcome": control_outcome,
                "att_component": treated_outcome - control_outcome,
            }
        )
    return pd.DataFrame(rows)


def _balance_summary(df: pd.DataFrame, pairs: pd.DataFrame, treatment_col: str, covariates: list[str]) -> list[dict[str, Any]]:
    treated = df[df[treatment_col] == 1]
    controls = df[df[treatment_col] == 0]
    matched_treated = df.loc[pairs["treated_index"].astype(str).map(_restore_index_type(df.index)).tolist()] if not pairs.empty else pd.DataFrame()
    matched_controls = df.loc[pairs["control_index"].astype(str).map(_restore_index_type(df.index)).tolist()] if not pairs.empty else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for variable in covariates:
        before = _smd(treated[variable], controls[variable])
        after = _smd(matched_treated[variable], matched_controls[variable]) if not matched_treated.empty and not matched_controls.empty else None
        rows.append(
            {
                "variable": variable,
                "treated_mean_before": _float_or_none(treated[variable].mean()),
                "control_mean_before": _float_or_none(controls[variable].mean()),
                "standardized_mean_difference_before": before,
                "treated_mean_after": _float_or_none(matched_treated[variable].mean()) if not matched_treated.empty else None,
                "control_mean_after": _float_or_none(matched_controls[variable].mean()) if not matched_controls.empty else None,
                "standardized_mean_difference_after": after,
                "absolute_smd_before": abs(before) if before is not None else None,
                "absolute_smd_after": abs(after) if after is not None else None,
                "balance_improved": _balance_improved(before, after),
            }
        )
    return rows


def _balance_overview(balance_summary: list[dict[str, Any]]) -> dict[str, Any]:
    before_values = [_float_or_none(row.get("absolute_smd_before")) for row in balance_summary]
    after_values = [_float_or_none(row.get("absolute_smd_after")) for row in balance_summary]
    before_clean = [value for value in before_values if value is not None]
    after_clean = [value for value in after_values if value is not None]
    improved = [row for row in balance_summary if row.get("balance_improved") is True]
    worsened = [row for row in balance_summary if row.get("balance_improved") is False]
    high_residual = [
        str(row.get("variable"))
        for row in balance_summary
        if _float_or_none(row.get("absolute_smd_after")) is not None
        and float(row.get("absolute_smd_after")) > PSM_HIGH_RESIDUAL_SMD_THRESHOLD
    ]
    return {
        "balance_summary_available": bool(balance_summary),
        "max_absolute_smd_before": max(before_clean) if before_clean else None,
        "max_absolute_smd_after": max(after_clean) if after_clean else None,
        "mean_absolute_smd_before": float(np.mean(before_clean)) if before_clean else None,
        "mean_absolute_smd_after": float(np.mean(after_clean)) if after_clean else None,
        "covariates_improved_count": len(improved),
        "covariates_worsened_count": len(worsened),
        "high_residual_imbalance_variables": high_residual,
    }


def _restore_index_type(index: pd.Index):
    lookup = {str(item): item for item in index}
    return lambda value: lookup[value]


def _smd(left: pd.Series, right: pd.Series) -> float | None:
    if left.empty or right.empty:
        return None
    left_values = left.astype(float)
    right_values = right.astype(float)
    pooled = np.sqrt((float(left_values.var(ddof=1)) + float(right_values.var(ddof=1))) / 2)
    if not np.isfinite(pooled) or pooled <= 0:
        return None
    return float((float(left_values.mean()) - float(right_values.mean())) / pooled)


def _propensity_score_summary(df: pd.DataFrame, treatment_col: str) -> dict[str, float | None]:
    scores = df["propensity_score"].astype(float)
    treated_scores = df.loc[df[treatment_col] == 1, "propensity_score"].astype(float)
    control_scores = df.loc[df[treatment_col] == 0, "propensity_score"].astype(float)
    return {
        "min": _float_or_none(scores.min()),
        "max": _float_or_none(scores.max()),
        "mean": _float_or_none(scores.mean()),
        "treated_mean": _float_or_none(treated_scores.mean()),
        "control_mean": _float_or_none(control_scores.mean()),
    }


def _selected_columns(config: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys([config.get("dependent_variable"), config.get("treatment_variable"), *_matching_covariates(config)]))


def _matching_covariates(config: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys([str(item) for item in config.get("matching_covariates", []) if str(item or "").strip()]))


def _numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    numeric = df[columns].copy()
    for column in columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    return numeric


def _non_numeric_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns and pd.to_numeric(df[column], errors="coerce").isna().any()]


def _is_binary_numeric(series: pd.Series) -> bool:
    unique = set(pd.to_numeric(series, errors="coerce").dropna().astype(float).unique().tolist())
    return bool(unique) and unique.issubset({0.0, 1.0})


def _design_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return sm.add_constant(df[columns].astype(float), has_constant="add")


def _is_rank_deficient(design: pd.DataFrame) -> bool:
    matrix = design.to_numpy(dtype=float)
    return bool(np.linalg.matrix_rank(matrix) < matrix.shape[1])


def _estimand(config: dict[str, Any]) -> str:
    return str(config.get("psm_estimand") or "ATT").upper()


def _matching_method(config: dict[str, Any]) -> str:
    return str(config.get("matching_method") or "nearest_neighbor")


def _caliper(config: dict[str, Any]) -> float | None:
    value = config.get("caliper")
    if value is None or value == "":
        return None
    return float(value)


def _balance_worsened(row: dict[str, Any]) -> bool:
    before = row.get("absolute_smd_before")
    after = row.get("absolute_smd_after")
    if before is None or after is None:
        return False
    return float(after) > float(before) + PSM_BALANCE_TOLERANCE


def _balance_improved(before: float | None, after: float | None) -> bool | None:
    if before is None or after is None:
        return None
    return float(after) <= float(before) + PSM_BALANCE_TOLERANCE


def _fmt(value: Any) -> str:
    converted = _float_or_none(value)
    return "N/A" if converted is None else f"{converted:.3f}"


def _float_or_none(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(converted):
        return None
    return converted


PSM_MODEL = ModelDefinition(
    model_id=PSM_MODEL_ID,
    display_name_en="Propensity Score Matching",
    display_name_zh="倾向得分匹配",
    description_en="Estimate a minimal nearest-neighbor ATT using propensity scores from a Logit model.",
    description_zh="使用 Logit 倾向得分估计最小最近邻 ATT。",
    required_roles=["dependent_variable", "treatment_variable", "matching_covariates"],
    validate=validate_psm_inputs,
    fit=fit_psm,
    diagnostics=diagnose_psm,
    report_label_en="Propensity Score Matching",
    report_label_zh="倾向得分匹配",
    limitations_en=[
        "This minimal PSM runner estimates ATT with nearest-neighbor matching only.",
        "PSM improves balance on observed covariates only and does not address unobserved confounding.",
        "No bootstrap standard errors, IPW, kernel matching, or doubly robust estimator is implemented.",
    ],
    limitations_zh=[
        "当前最小 PSM 运行器仅估计最近邻匹配 ATT。",
        "PSM 只能改善观测协变量的平衡，不能处理未观测混杂。",
        "当前未实现 bootstrap 标准误、IPW、核匹配或双重稳健估计。",
    ],
)
