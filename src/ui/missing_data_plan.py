from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from src.data_quality import MissingnessProfile, VariableQualitySummary, build_missingness_profile, build_variable_quality_summaries
from src.i18n import get_text
from src.preprocessing import (
    DROP_ROWS,
    MEAN_IMPUTE,
    MEDIAN_IMPUTE,
    MISSING_INDICATOR,
    MODE_IMPUTE,
    NO_ACTION,
    MISSING_DATA_STRATEGIES,
    MissingDataAction,
    MissingDataHandlingResult,
    MissingDataPlan,
    missing_data_handling_result_to_dict,
    missing_data_plan_to_dict,
    validate_missing_data_plan,
)


MISSING_DATA_STATE_KEYS = [
    "missing_data_handled_df",
    "missing_data_plan",
    "missing_data_handling_result",
    "missing_data_base_roles",
    "missing_data_confirm_apply",
]


@dataclass(frozen=True)
class MissingDataPlanUiEvent:
    plan_to_apply: MissingDataPlan | None = None
    reset_requested: bool = False


def missing_data_state_reset_keys() -> list[str]:
    return list(MISSING_DATA_STATE_KEYS)


def build_missing_data_state_update(
    handled_df: pd.DataFrame,
    plan: MissingDataPlan,
    handling_result: MissingDataHandlingResult,
    base_roles: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "missing_data_handled_df": handled_df.copy(deep=True),
        "missing_data_plan": missing_data_plan_to_dict(plan),
        "missing_data_handling_result": missing_data_handling_result_to_dict(handling_result),
        "missing_data_base_roles": dict(base_roles or {}),
    }


def build_missing_data_plan_from_selections(
    selections: list[dict[str, Any]],
    user_confirmed: bool,
    plan_id: str = "ui_missing_data_plan",
    target_scope: str = "selected_variables",
) -> MissingDataPlan:
    actions: list[MissingDataAction] = []
    for selection in selections:
        variable = str(selection.get("variable") or "")
        strategy_id = str(selection.get("strategy_id") or NO_ACTION)
        if not variable:
            continue
        strategy = MISSING_DATA_STRATEGIES.get(strategy_id)
        changes_data = bool(strategy and strategy.changes_data)
        parameters = {"target_scope": target_scope} if strategy_id == DROP_ROWS else {}
        actions.append(
            MissingDataAction(
                variable=variable,
                strategy_id=strategy_id,
                parameters=parameters,
                reason="user_selected_missing_data_handling",
                user_confirmed=bool(user_confirmed and changes_data),
            )
        )
    return MissingDataPlan(
        plan_id=plan_id,
        actions=actions,
        created_from_profile="data_quality_missingness_profile",
        target_scope=target_scope,
        requires_user_confirmation=any(_strategy_changes_data(action.strategy_id) for action in actions),
        status="approved" if user_confirmed else "draft",
    )


def allowed_strategy_ids_for_summary(summary: VariableQualitySummary | dict[str, Any]) -> list[str]:
    payload = summary.to_dict() if hasattr(summary, "to_dict") else dict(summary)
    dtype = str(payload.get("dtype") or "").lower()
    unique_count = int(payload.get("unique_count") or 0)
    is_binary = bool(payload.get("is_binary_like"))
    is_numeric = is_binary or any(token in dtype for token in ["int", "float", "number"])
    is_mode_ready = is_binary or unique_count <= 20 or any(token in dtype for token in ["object", "category", "bool"])
    strategies = [NO_ACTION, DROP_ROWS]
    if is_numeric:
        strategies.extend([MEAN_IMPUTE, MEDIAN_IMPUTE])
    if is_mode_ready:
        strategies.append(MODE_IMPUTE)
    strategies.append(MISSING_INDICATOR)
    return strategies


def render_missing_data_plan_builder(
    df: pd.DataFrame,
    language: str,
    applied_plan: dict[str, Any] | None = None,
    handling_result: dict[str, Any] | None = None,
) -> MissingDataPlanUiEvent:
    t = lambda key: get_text(language, key)
    missingness = build_missingness_profile(df)
    summaries = build_variable_quality_summaries(df)
    summary_map = {summary.variable: summary for summary in summaries}

    with st.expander(t("missing_data_plan_builder"), expanded=False):
        st.caption(t("missing_data_plan_caption"))
        if handling_result:
            _render_handling_summary(handling_result, language)
            if st.button(t("missing_data_undo"), key="missing_data_undo_button", width="stretch"):
                return MissingDataPlanUiEvent(reset_requested=True)

        missing_rows = [row for row in missingness.missing_by_variable if int(row.get("missing_count") or 0) > 0]
        if not missing_rows:
            st.success(t("missing_data_no_missing_variables"))
            return MissingDataPlanUiEvent()

        selections: list[dict[str, Any]] = []
        for row in missing_rows:
            variable = str(row.get("variable") or "")
            options = allowed_strategy_ids_for_summary(summary_map.get(variable, {}))
            labels = [_strategy_label(strategy_id, language) for strategy_id in options]
            choice = st.selectbox(
                f"{variable} · {t('dq_missing_count')}: {row.get('missing_count')}",
                options=labels,
                index=0,
                key=f"missing_data_strategy_{variable}",
            )
            selections.append({"variable": variable, "strategy_id": options[labels.index(choice)]})

        plan_summary = _plan_summary_frame(selections, missingness, language)
        st.dataframe(plan_summary, width="stretch", hide_index=True)

        confirmation = st.checkbox(
            t("missing_data_confirm_apply"),
            value=False,
            key="missing_data_confirm_apply",
        )
        plan = build_missing_data_plan_from_selections(selections, confirmation)
        validation = validate_missing_data_plan(plan, missingness_profile=missingness, variable_summaries=summaries)
        if validation.errors:
            for error in validation.errors[:3]:
                st.error(error)
        elif _plan_changes_data(plan):
            st.warning(t("missing_data_changes_data_warning"))
        else:
            st.info(t("missing_data_no_data_changes"))

        apply_clicked = st.button(
            f"{t('missing_data_apply')} ↓",
            key="missing_data_apply_button",
            type="secondary",
            width="stretch",
            disabled=bool(validation.errors),
        )
        if apply_clicked:
            if _plan_changes_data(plan) and not confirmation:
                st.warning(t("missing_data_confirmation_required"))
                return MissingDataPlanUiEvent()
            return MissingDataPlanUiEvent(plan_to_apply=plan)

    return MissingDataPlanUiEvent()


def _render_handling_summary(handling_result: dict[str, Any], language: str) -> None:
    t = lambda key: get_text(language, key)
    action_results = [dict(item) for item in handling_result.get("action_results", []) if isinstance(item, dict)]
    rows_dropped = sum(int(item.get("rows_dropped") or 0) for item in action_results)
    values_filled = sum(int(item.get("values_filled") or 0) for item in action_results)
    indicator_count = sum(1 for item in action_results if item.get("indicator_variable_created"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("missing_data_original_rows"), handling_result.get("original_row_count"))
    c2.metric(t("missing_data_final_rows"), handling_result.get("final_row_count"))
    c3.metric(t("missing_data_rows_dropped"), rows_dropped)
    c4.metric(t("missing_data_values_filled"), values_filled)
    st.caption(f"{t('missing_data_indicators_created')}: {indicator_count}")
    if action_results:
        st.dataframe(pd.DataFrame(action_results), width="stretch", hide_index=True)


def _plan_summary_frame(selections: list[dict[str, Any]], missingness: MissingnessProfile, language: str) -> pd.DataFrame:
    t = lambda key: get_text(language, key)
    missing_counts = {
        str(row.get("variable") or ""): int(row.get("missing_count") or 0)
        for row in missingness.missing_by_variable
    }
    rows = []
    for selection in selections:
        strategy_id = str(selection["strategy_id"])
        rows.append(
            {
                t("table_variable"): selection["variable"],
                t("dq_missing_count"): missing_counts.get(selection["variable"], 0),
                t("missing_data_strategy"): _strategy_label(strategy_id, language),
                t("missing_data_changes_data"): get_text(language, "yes") if _strategy_changes_data(strategy_id) else get_text(language, "no"),
                t("missing_data_expected_impact"): _expected_impact(strategy_id, selection["variable"], missing_counts, language),
            }
        )
    return pd.DataFrame(rows)


def _expected_impact(strategy_id: str, variable: str, missing_counts: dict[str, int], language: str) -> str:
    count = missing_counts.get(variable, 0)
    if strategy_id == DROP_ROWS:
        return f"{get_text(language, 'missing_data_expected_rows_affected')}: {count}"
    if strategy_id in {MEAN_IMPUTE, MEDIAN_IMPUTE, MODE_IMPUTE}:
        return f"{get_text(language, 'missing_data_expected_values_filled')}: {count}"
    if strategy_id == MISSING_INDICATOR:
        return get_text(language, "missing_data_expected_indicator")
    return get_text(language, "missing_data_expected_no_change")


def _strategy_label(strategy_id: str, language: str) -> str:
    strategy = MISSING_DATA_STRATEGIES.get(strategy_id)
    return get_text(language, strategy.display_name_key) if strategy else strategy_id


def _strategy_changes_data(strategy_id: str) -> bool:
    strategy = MISSING_DATA_STRATEGIES.get(strategy_id)
    return bool(strategy and strategy.changes_data)


def _plan_changes_data(plan: MissingDataPlan) -> bool:
    return any(_strategy_changes_data(action.strategy_id) for action in plan.actions)
