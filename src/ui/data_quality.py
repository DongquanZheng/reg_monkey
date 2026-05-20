from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.data_quality import (
    build_data_quality_profile,
    build_missingness_profile,
    build_pre_model_risk_profile,
    build_resource_warning_profile,
    build_variable_quality_summaries,
    estimate_model_sample_impact,
)
from src.models.execution import ModelSpec
from src.i18n import get_text
from src.ui.components import render_callout, render_diagnostic_card, render_metric_card, render_section_header


def render_data_quality_section(
    df: pd.DataFrame,
    language: str,
    model_config: dict[str, Any] | None = None,
) -> None:
    t = lambda key: get_text(language, key)
    quality = build_data_quality_profile(df)
    missingness = build_missingness_profile(df)
    variable_summaries = build_variable_quality_summaries(df)
    selected_variables = selected_variables_from_model_config(model_config or {})
    sample_impact = estimate_model_sample_impact(df, selected_variables)

    render_section_header(t("data_quality"), t("data_quality_caption"))
    cols = st.columns(3)
    metrics = [
        (t("dq_rows"), quality.row_count),
        (t("dq_columns"), quality.column_count),
        (t("dq_complete_case_rows"), missingness.complete_case_rows),
        (t("dq_variables_with_missing"), len(missingness.columns_with_any_missing)),
        (t("dq_high_missing_variables"), len(missingness.high_missing_variables)),
        (t("dq_constant_near_constant"), len(quality.constant_columns) + len(quality.near_constant_columns)),
    ]
    for index, (label, value) in enumerate(metrics):
        with cols[index % len(cols)]:
            render_metric_card(label, value, tone="info")

    with st.expander(t("dq_missingness_by_variable"), expanded=False):
        st.dataframe(_missingness_frame(missingness.missing_by_variable, language), width="stretch")

    with st.expander(t("dq_variable_quality_summary"), expanded=False):
        st.dataframe(_variable_quality_frame(variable_summaries, language), width="stretch")

    with st.expander(t("dq_model_sample_impact"), expanded=False):
        if selected_variables:
            st.dataframe(_sample_impact_frame(sample_impact.to_dict(), language), width="stretch")
        else:
            st.caption(t("dq_model_sample_impact_empty"))


def render_resource_warning_section(
    df: pd.DataFrame,
    language: str,
    *,
    file_size_mb: float | None = None,
    public_demo_mode: bool = False,
) -> None:
    t = lambda key: get_text(language, key)
    profile = build_resource_warning_profile(df, file_size_mb=file_size_mb, public_demo_mode=public_demo_mode)
    visible_items = [item for item in profile.warning_items if item.show_in_ui]
    if not visible_items:
        return

    caption_key = "resource_warning_public_demo_caption" if public_demo_mode else "resource_warning_caption"
    render_section_header(t("resource_warning_title"), t(caption_key))
    top_items = _top_resource_items(visible_items, limit=3)
    for item in top_items:
        severity = "warning" if item.severity == "warning" else "info"
        render_diagnostic_card(
            t(f"resource_warning_{item.code}_title"),
            t(f"resource_warning_{item.code}_message"),
            severity=severity,
            affected_variables=item.affected_variables[:8],
            recommendation=t(f"resource_warning_{item.code}_recommendation"),
        )
    with st.expander(t("resource_warning_details"), expanded=False):
        st.dataframe(_resource_warning_frame([item.to_dict() for item in visible_items], language), width="stretch")


def render_pre_model_risk_check(
    df: pd.DataFrame,
    spec: ModelSpec,
    language: str,
) -> None:
    t = lambda key: get_text(language, key)
    profile = build_pre_model_risk_profile(df, spec)
    render_section_header(t("pre_model_risk_check"), t("pre_model_risk_caption"))
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card(t("pre_model_risk_level"), t(f"risk_level_{profile.overall_risk_level}"), tone=_risk_tone(profile.overall_risk_level))
    with c2:
        render_metric_card(t("dq_usable_rows_after_dropna"), profile.usable_rows, tone="info")
    with c3:
        render_metric_card(t("dq_dropped_rows"), profile.dropped_rows, tone="neutral")
    with c4:
        render_metric_card(t("dq_dropped_percentage"), profile.dropped_percentage, tone="neutral")

    visible_items = [item for item in profile.risk_items if item.show_in_ui]
    prominent_items = [item for item in visible_items if item.severity in {"error", "warning"}]
    top_items = _top_risk_items(prominent_items, limit=3)
    if top_items:
        for item in top_items:
            severity = "critical" if item.severity == "error" else ("warning" if item.severity == "warning" else "info")
            render_diagnostic_card(
                t(f"risk_{item.code}_title"),
                t(f"risk_{item.code}_message"),
                severity=severity,
                affected_variables=item.affected_variables,
                recommendation=t(f"risk_{item.code}_recommendation"),
            )
    else:
        render_callout(t("diagnostic_clear"), t("pre_model_risk_no_major_items"), tone="success")

    with st.expander(t("pre_model_risk_details"), expanded=False):
        if visible_items:
            st.dataframe(_risk_items_frame([item.to_dict() for item in visible_items], language), width="stretch")
        else:
            st.caption(t("pre_model_risk_no_major_items"))


def _top_risk_items(items: list[Any], limit: int) -> list[Any]:
    rank = {"error": 0, "warning": 1, "info": 2}
    return sorted(items, key=lambda item: (rank.get(item.severity, 3), item.code))[:limit]


def _top_resource_items(items: list[Any], limit: int) -> list[Any]:
    rank = {"warning": 0, "info": 1}
    return sorted(items, key=lambda item: (rank.get(item.severity, 2), item.code))[:limit]


def _risk_item_text(item: dict[str, Any], language: str, *, include_title: bool = True) -> str:
    code = item.get("code", "")
    affected = ", ".join(item.get("affected_variables") or [])
    title = get_text(language, f"risk_{code}_title")
    message = get_text(language, f"risk_{code}_message")
    recommendation = get_text(language, f"risk_{code}_recommendation")
    parts = [f"**{title}**", message] if include_title else [message]
    if affected:
        parts.append(f"{get_text(language, 'affected_variables')}: {affected}")
    if recommendation:
        parts.append(recommendation)
    return " ".join(parts)


def _risk_tone(level: str) -> str:
    if level == "error":
        return "danger"
    if level == "warning":
        return "warning"
    if level == "info":
        return "info"
    return "success"


def _risk_items_frame(rows: list[dict[str, Any]], language: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                get_text(language, "pre_model_risk_code"): row.get("code"),
                get_text(language, "pre_model_risk_severity"): get_text(language, f"risk_level_{row.get('severity', 'info')}"),
                get_text(language, "pre_model_risk_title"): get_text(language, f"risk_{row.get('code')}_title"),
                get_text(language, "affected_variables"): ", ".join(row.get("affected_variables") or []),
                get_text(language, "pre_model_risk_recommendation"): get_text(language, f"risk_{row.get('code')}_recommendation"),
            }
            for row in rows
        ]
    )
    return frame


def _resource_warning_frame(rows: list[dict[str, Any]], language: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                get_text(language, "resource_warning_code"): row.get("code"),
                get_text(language, "resource_warning_severity"): get_text(language, f"risk_level_{row.get('severity', 'info')}"),
                get_text(language, "resource_warning_observed"): row.get("observed_value"),
                get_text(language, "resource_warning_threshold"): row.get("threshold"),
                get_text(language, "affected_variables"): ", ".join((row.get("affected_variables") or [])[:12]),
            }
            for row in rows
        ]
    )


def selected_variables_from_model_config(config: dict[str, Any]) -> list[str]:
    keys = [
        "dependent_variable",
        "outcome_variable",
        "main_independent_variables",
        "numeric_control_variables",
        "categorical_control_variables",
        "entity_id",
        "time_id",
        "time_variable",
        "treatment_variable",
        "post_variable",
        "group_variable",
        "cluster_variable",
        "endogenous_variable",
        "instrument_variable",
        "instruments",
        "exogenous_controls",
        "matching_covariates",
    ]
    result: list[str] = []
    seen: set[str] = set()
    for key in keys:
        value = config.get(key)
        values = value if isinstance(value, list | tuple | set) else [value]
        for item in values:
            if item is None:
                continue
            name = str(item)
            if not name or name in seen:
                continue
            seen.add(name)
            result.append(name)
    return result


def _missingness_frame(rows: list[dict[str, Any]], language: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.rename(
        columns={
            "variable": get_text(language, "table_variable"),
            "missing_count": get_text(language, "dq_missing_count"),
            "missing_percentage": get_text(language, "table_missing"),
        }
    )


def _variable_quality_frame(rows: list[Any], language: str) -> pd.DataFrame:
    frame = pd.DataFrame([row.to_dict() for row in rows])
    if frame.empty:
        return frame
    columns = [
        "variable",
        "dtype",
        "inferred_role_hint",
        "missing_count",
        "missing_percentage",
        "unique_count",
        "is_binary_like",
        "is_id_like",
        "is_constant",
        "is_near_constant",
        "is_high_cardinality",
        "is_text_numeric_like",
        "is_datetime_like",
        "is_mixed_type",
    ]
    labels = {
        "variable": get_text(language, "table_variable"),
        "dtype": get_text(language, "dq_dtype"),
        "inferred_role_hint": get_text(language, "dq_role_hint"),
        "missing_count": get_text(language, "dq_missing_count"),
        "missing_percentage": get_text(language, "table_missing"),
        "unique_count": get_text(language, "unique_values"),
        "is_binary_like": get_text(language, "dq_binary_like"),
        "is_id_like": get_text(language, "dq_id_like"),
        "is_constant": get_text(language, "dq_constant"),
        "is_near_constant": get_text(language, "dq_near_constant"),
        "is_high_cardinality": get_text(language, "dq_high_cardinality"),
        "is_text_numeric_like": get_text(language, "dq_text_numeric_like"),
        "is_datetime_like": get_text(language, "dq_datetime_like"),
        "is_mixed_type": get_text(language, "dq_mixed_type"),
    }
    return frame[[column for column in columns if column in frame.columns]].rename(columns=labels)


def _sample_impact_frame(payload: dict[str, Any], language: str) -> pd.DataFrame:
    rows = [
        (get_text(language, "dq_selected_variables"), ", ".join(payload.get("selected_variables") or [])),
        (get_text(language, "dq_original_rows"), payload.get("original_rows")),
        (get_text(language, "dq_usable_rows_after_dropna"), payload.get("usable_rows_after_dropna")),
        (get_text(language, "dq_dropped_rows"), payload.get("dropped_rows")),
        (get_text(language, "dq_dropped_percentage"), payload.get("dropped_percentage")),
        (get_text(language, "dq_variables_causing_loss"), ", ".join(payload.get("variables_causing_missing_loss") or [])),
        (get_text(language, "dq_warning_level"), payload.get("warning_level")),
    ]
    return pd.DataFrame(rows, columns=[get_text(language, "dq_metric"), get_text(language, "dq_value")])
