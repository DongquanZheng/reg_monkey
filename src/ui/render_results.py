from __future__ import annotations

import pandas as pd
import streamlit as st

from src.formatting import format_diagnostic_table, format_number, format_p_narrative, format_regression_table
from src.i18n import get_text


def model_display_name(model_id: str, language: str) -> str:
    key = {
        "ols": "model_label_ols",
        "logit": "model_label_logit",
        "probit": "model_label_probit",
        "panel_fe": "model_label_panel_fe",
        "did": "model_label_did",
        "iv_2sls": "model_label_iv_2sls",
        "psm": "model_label_psm",
    }.get(str(model_id or "").lower())
    return get_text(language, key) if key else str(model_id or "")


def independent_variable_count(model_summary: dict) -> int:
    variables = model_summary.get("independent_variables")
    if isinstance(variables, list):
        return len(variables)
    if isinstance(variables, tuple):
        return len(variables)
    variables = model_summary.get("main_independent_variables")
    if isinstance(variables, list):
        return len(variables)
    if isinstance(variables, tuple):
        return len(variables)
    value = model_summary.get("num_predictors")
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def standard_error_display_label(model_summary: dict, language: str) -> str:
    t = lambda key: get_text(language, key)
    standard_errors = str(model_summary.get("standard_errors") or "").strip()
    standard_errors_lower = standard_errors.lower()
    if standard_errors_lower in {"cluster_entity", "clustered_entity"}:
        return t("cluster_entity")
    if standard_errors_lower.startswith("cluster"):
        return t("clustered_se")
    if standard_errors_lower in {"hc3", "robust"} or model_summary.get("robust_standard_errors") is True:
        return t("hc3")
    if standard_errors_lower in {"conventional", "none"} or model_summary.get("robust_standard_errors") is False:
        return t("conventional")
    return t("model_default_se")


def fit_metric_summary(model_summary: dict, language: str) -> str:
    model_type = str(model_summary.get("model_type") or "ols").lower()
    if model_type == "panel_fe":
        return f"{get_text(language, 'within_r_squared')}: {format_number(model_summary.get('r_squared_within'), 3, language)}"
    if model_type in {"logit", "probit"}:
        return f"{get_text(language, 'pseudo_r_squared')}: {format_number(model_summary.get('pseudo_r_squared'), 3, language)}"
    if model_type == "did":
        return f"{get_text(language, 'regression_r_squared')}: {format_number(model_summary.get('r_squared'), 3, language)}"
    if model_type == "iv_2sls":
        return f"{get_text(language, 'second_stage_r_squared')}: {format_number(model_summary.get('r_squared'), 3, language)}"
    if model_type == "psm":
        return f"{get_text(language, 'att_estimate_label')}: {format_number(model_summary.get('att_estimate'), 4, language)}"
    return f"R²: {format_number(model_summary.get('r_squared'), 3, language)}"


def main_effect_display(regression_table: pd.DataFrame, model_summary: dict, language: str) -> str:
    model_type = str(model_summary.get("model_type") or "ols").lower()
    if model_type == "psm":
        direction = _direction_text(model_summary.get("att_estimate"), language)
        att = format_number(model_summary.get("att_estimate"), 4, language)
        matched = model_summary.get("matched_treated_count", get_text(language, "not_available"))
        return f"{get_text(language, 'att_estimate_label')}: {att} ({direction}); {get_text(language, 'matched_treated_observations')}: {matched}"
    main_vars = list(model_summary.get("main_independent_variables") or [])
    main_table = regression_table[regression_table["variable"].isin(main_vars)]
    if main_table.empty:
        return get_text(language, "not_available")
    row = main_table.iloc[0]
    direction = _direction_text(row.get("coefficient"), language)
    return f"{row['variable']}: {direction}, {format_p_narrative(row.get('p_value'), language)}"


def result_summary_metrics(model_summary: dict, language: str) -> list[tuple[str, str]]:
    model_type = str(model_summary.get("model_type") or "ols").lower()
    n_obs = str(model_summary.get("n_obs", get_text(language, "not_available")))
    if model_type == "panel_fe":
        fe_parts = []
        if model_summary.get("entity_effects"):
            fe_parts.append(get_text(language, "entity_fixed_effects"))
        if model_summary.get("time_effects"):
            fe_parts.append(get_text(language, "time_fixed_effects"))
        return [
            (get_text(language, "available_observations"), n_obs),
            (get_text(language, "entities"), str(model_summary.get("entities", "N/A"))),
            (get_text(language, "time_periods"), str(model_summary.get("time_periods", "N/A"))),
            (get_text(language, "fixed_effects_summary"), " + ".join(fe_parts) if fe_parts else "N/A"),
            (get_text(language, "within_r_squared"), format_number(model_summary.get("r_squared_within"), 3, language)),
        ]
    if model_type in {"logit", "probit"}:
        return [
            (get_text(language, "available_observations"), n_obs),
            (get_text(language, "pseudo_r_squared"), format_number(model_summary.get("pseudo_r_squared"), 3, language)),
            (get_text(language, "log_likelihood"), format_number(model_summary.get("log_likelihood"), 3, language)),
            (get_text(language, "aic"), format_number(model_summary.get("aic"), 2, language)),
            (get_text(language, "bic"), format_number(model_summary.get("bic"), 2, language)),
        ]
    if model_type == "psm":
        return [
            (get_text(language, "available_observations"), n_obs),
            (get_text(language, "att_estimate_label"), format_number(model_summary.get("att_estimate"), 4, language)),
            (get_text(language, "matched_treated_observations"), str(model_summary.get("matched_treated_count", "N/A"))),
            (get_text(language, "matched_control_observations"), str(model_summary.get("matched_control_count", "N/A"))),
            (get_text(language, "balance_status"), _psm_balance_status(model_summary, language)),
        ]
    if model_type == "iv_2sls":
        return [
            (get_text(language, "available_observations"), n_obs),
            (get_text(language, "second_stage_r_squared"), format_number(model_summary.get("r_squared"), 3, language)),
            (get_text(language, "first_stage_f_statistic"), format_number(model_summary.get("first_stage_f_statistic"), 3, language)),
            (get_text(language, "instrument_count"), str(model_summary.get("instrument_count", "N/A"))),
            (get_text(language, "standard_error_type"), standard_error_display_label(model_summary, language)),
        ]
    return [
        (get_text(language, "available_observations"), n_obs),
        (get_text(language, "regression_r_squared"), format_number(model_summary.get("r_squared"), 3, language)),
        (get_text(language, "adjusted_r_squared"), format_number(model_summary.get("adj_r_squared"), 3, language)),
        (get_text(language, "num_independent_variables"), str(independent_variable_count(model_summary))),
        (get_text(language, "standard_error_type"), standard_error_display_label(model_summary, language)),
    ]


def psm_key_findings(model_summary: dict, warnings: list[str], language: str) -> list[str]:
    att = format_number(model_summary.get("att_estimate"), 4, language)
    matched_treated = model_summary.get("matched_treated_count", "N/A")
    matched_control = model_summary.get("matched_control_count", "N/A")
    lines = [
        get_text(language, "psm_key_finding_att").format(att=att),
        get_text(language, "psm_key_finding_matched_sample").format(
            treated=matched_treated,
            controls=matched_control,
        ),
        get_text(language, "psm_key_finding_balance"),
        get_text(language, "psm_key_finding_caution"),
    ]
    if _psm_balance_needs_review(model_summary):
        lines.append(get_text(language, "psm_balance_warning"))
    if warnings:
        lines.append(warnings[0])
    return lines


def _psm_balance_status(model_summary: dict, language: str) -> str:
    if not model_summary.get("balance_summary_available"):
        return get_text(language, "not_available")
    if _psm_balance_needs_review(model_summary):
        return get_text(language, "diagnostic_attention")
    return get_text(language, "diagnostic_clear")


def _psm_balance_needs_review(model_summary: dict) -> bool:
    high_residual = model_summary.get("high_residual_imbalance_variables") or []
    if high_residual:
        return True
    try:
        return float(model_summary.get("max_absolute_smd_after")) > 0.1
    except (TypeError, ValueError):
        return False


def _direction_text(value: object, language: str) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return get_text(language, "not_available")
    if language == "zh":
        return "正向" if numeric > 0 else "负向"
    return "Positive" if numeric > 0 else "Negative"


def render_advanced_outputs(model_results: dict | None, language: str) -> None:
    outputs = dict((model_results or {}).get("advanced_outputs") or {})
    diagnostics = dict((model_results or {}).get("diagnostics") or {})
    odds_ratio_table = outputs.get("odds_ratio_table")
    marginal_effects_table = outputs.get("marginal_effects_table")
    heteroskedasticity = diagnostics.get("heteroskedasticity")
    if not any(
        [
            isinstance(odds_ratio_table, pd.DataFrame) and not odds_ratio_table.empty,
            isinstance(marginal_effects_table, pd.DataFrame) and not marginal_effects_table.empty,
            isinstance(heteroskedasticity, dict) and bool(heteroskedasticity),
        ]
    ):
        return

    t = lambda key: get_text(language, key)
    with st.expander(t("advanced_model_outputs"), expanded=False):
        if isinstance(odds_ratio_table, pd.DataFrame) and not odds_ratio_table.empty:
            st.subheader(t("odds_ratios"))
            st.caption(t("odds_ratios_help"))
            st.dataframe(format_regression_table(odds_ratio_table, language), width="stretch")
        if isinstance(marginal_effects_table, pd.DataFrame) and not marginal_effects_table.empty:
            st.subheader(t("marginal_effects"))
            st.caption(t("marginal_effects_help"))
            st.dataframe(format_regression_table(marginal_effects_table, language), width="stretch")
        if isinstance(heteroskedasticity, dict) and heteroskedasticity:
            st.subheader(t("heteroskedasticity_diagnostics"))
            st.caption(t("heteroskedasticity_diagnostics_help"))
            st.dataframe(format_diagnostic_table(pd.DataFrame([heteroskedasticity]), language, digits=3), width="stretch")
