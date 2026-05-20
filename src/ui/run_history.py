from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.formatting import format_number
from src.i18n import get_text
from src.model_comparison import ModelComparisonResult, compare_run_records
from src.run_history import run_history_to_dict


def run_history_records(history: dict[str, Any] | None) -> list[dict[str, Any]]:
    payload = run_history_to_dict(history or {})
    return [dict(record) for record in payload.get("records", []) if isinstance(record, dict)]


def run_history_table(history: dict[str, Any] | None, language: str) -> pd.DataFrame:
    t = lambda key: get_text(language, key)
    rows = []
    for record in run_history_records(history):
        diagnostics = dict(record.get("diagnostics_summary") or {})
        advanced = dict(record.get("advanced_outputs_summary") or {})
        rows.append(
            {
                t("run_history_short_id"): _short_id(record.get("run_id")),
                t("run_history_model"): _model_label(record, language),
                t("run_history_source"): _source_label(record.get("source"), language),
                t("run_history_status"): _status_label(record.get("status"), language),
                t("run_history_created_at"): str(record.get("created_at") or ""),
                t("run_history_key_result"): _key_result_label(record, language),
                t("run_history_diagnostics"): _diagnostics_label(diagnostics, language),
                t("run_history_advanced_outputs"): _advanced_label(advanced, language),
            }
        )
    return pd.DataFrame(rows)


def run_history_detail_sections(record: dict[str, Any], language: str) -> dict[str, pd.DataFrame]:
    t = lambda key: get_text(language, key)
    return {
        t("run_history_model_setup"): _mapping_frame(record.get("model_spec"), language),
        t("run_history_key_result_summary"): _mapping_frame(record.get("key_result_summary"), language),
        t("run_history_diagnostics_summary"): _mapping_frame(record.get("diagnostics_summary"), language),
        t("run_history_advanced_outputs_summary"): _mapping_frame(record.get("advanced_outputs_summary"), language),
    }


def model_comparison_table(comparison: ModelComparisonResult, language: str) -> pd.DataFrame:
    t = lambda key: get_text(language, key)
    rows = []
    for item in comparison.items:
        fit_metric = t("not_available")
        if item.fit_metric_label and item.fit_metric_label != "not_applicable":
            fit_metric = f"{_fit_metric_label(item.fit_metric_label, language)}: {_display_value(item.fit_metric_value, language)}"
        rows.append(
            {
                t("run_history_model"): get_text(language, item.model_display_name_key),
                t("run_history_source"): _source_label(item.source, language),
                t("run_history_status"): _status_label(item.status, language),
                t("model_comparison_sample_size"): _display_value(item.sample_size, language),
                t("model_comparison_dependent_variable"): item.dependent_variable or t("not_available"),
                t("model_comparison_key_estimate"): _display_value(item.key_estimate, language),
                t("model_comparison_p_value"): _display_value(item.p_value, language),
                t("model_comparison_fit_metric"): fit_metric,
                t("model_comparison_diagnostics_count"): item.diagnostics_count,
                t("model_comparison_warning_count"): item.warning_count,
                t("run_history_advanced_outputs"): ", ".join(item.advanced_outputs_available) if item.advanced_outputs_available else t("not_available"),
            }
        )
    return pd.DataFrame(rows)


def model_comparison_warnings_table(comparison: ModelComparisonResult, language: str) -> pd.DataFrame:
    t = lambda key: get_text(language, key)
    rows = []
    for warning in comparison.warnings:
        rows.append(
            {
                t("model_comparison_warning_code"): warning.code,
                t("model_comparison_severity"): _comparison_severity_label(warning.severity, language),
                t("model_comparison_warning_message"): _comparison_warning_message(warning.code, language),
                t("model_comparison_affected_runs"): ", ".join(_short_id(run_id) for run_id in warning.affected_run_ids),
            }
        )
    return pd.DataFrame(rows)


def render_run_history_preview(history: dict[str, Any] | None, language: str) -> None:
    t = lambda key: get_text(language, key)
    records = run_history_records(history)
    with st.expander(t("run_history"), expanded=False):
        st.caption(t("run_history_caption"))
        if not records:
            st.info(t("run_history_empty"))
            return
        st.dataframe(run_history_table(history, language), width="stretch", hide_index=True)
        st.caption(t("run_history_no_raw_data"))
        labels = [_record_option_label(record, language) for record in records]
        selected = st.selectbox(
            t("run_history_details"),
            labels,
            index=max(len(labels) - 1, 0),
            key="run_history_detail_selector",
        )
        record = records[labels.index(selected)]
        for title, frame in run_history_detail_sections(record, language).items():
            st.markdown(f"**{title}**")
            st.dataframe(frame, width="stretch", hide_index=True)
        _render_model_comparison_preview(records, labels, language)


def _render_model_comparison_preview(records: list[dict[str, Any]], labels: list[str], language: str) -> None:
    t = lambda key: get_text(language, key)
    st.markdown(f"**{t('model_comparison')}**")
    st.caption(t("model_comparison_caution"))
    if len(records) < 2:
        st.info(t("model_comparison_need_two_runs"))
        return

    default = labels[-2:] if len(labels) >= 2 else []
    selected_labels = st.multiselect(
        t("model_comparison_select_runs"),
        labels,
        default=default,
        key="model_comparison_selected_runs",
        help=t("model_comparison_select_help"),
    )
    if len(selected_labels) > 3:
        st.warning(t("model_comparison_max_three_runs"))
    can_compare = 2 <= len(selected_labels) <= 3
    if st.button(t("model_comparison_compare_selected"), key="model_comparison_compare_button", disabled=not can_compare):
        selected_records = [records[labels.index(label)] for label in selected_labels]
        comparison = compare_run_records(selected_records)
        st.dataframe(model_comparison_table(comparison, language), width="stretch", hide_index=True)
        if comparison.warnings:
            st.markdown(f"**{t('model_comparison_warnings')}**")
            st.dataframe(model_comparison_warnings_table(comparison, language), width="stretch", hide_index=True)
        else:
            st.info(t("model_comparison_no_warnings"))


def _short_id(value: object) -> str:
    return str(value or "")[:8]


def _model_label(record: dict[str, Any], language: str) -> str:
    key = str(record.get("model_display_name_key") or "")
    if key:
        return get_text(language, key)
    return str(record.get("model_id") or "")


def _source_label(source: object, language: str) -> str:
    key = f"run_history_source_{source or 'manual_configuration'}"
    return get_text(language, key)


def _status_label(status: object, language: str) -> str:
    key = "run_history_status_success" if status == "success" else "run_history_status_failed"
    return get_text(language, key)


def _key_result_label(record: dict[str, Any], language: str) -> str:
    t = lambda key: get_text(language, key)
    summary = dict(record.get("key_result_summary") or {})
    for key in ["att_estimate", "did_estimate", "iv_estimate", "r_squared", "pseudo_r_squared", "within_r_squared"]:
        if summary.get(key) is not None:
            return f"{_display_key(key)}: {format_number(summary.get(key), 3, language)}"
    if summary.get("observations_used") is not None:
        return f"{t('available_observations')}: {summary.get('observations_used')}"
    return t("not_available")


def _diagnostics_label(summary: dict[str, Any], language: str) -> str:
    diagnostics = get_text(language, "diagnostics")
    warnings = get_text(language, "warnings")
    return f"{diagnostics}: {summary.get('diagnostics_count', 0)} / {warnings}: {summary.get('warning_count', 0)}"


def _advanced_label(summary: dict[str, Any], language: str) -> str:
    outputs = summary.get("advanced_outputs_available") or []
    if not outputs:
        return get_text(language, "not_available")
    return ", ".join(str(item) for item in outputs)


def _record_option_label(record: dict[str, Any], language: str) -> str:
    return f"{_short_id(record.get('run_id'))} - {_model_label(record, language)} - {_status_label(record.get('status'), language)}"


def _mapping_frame(payload: object, language: str) -> pd.DataFrame:
    rows = []
    for key, value in _flatten_mapping(payload).items():
        rows.append(
            {
                get_text(language, "field"): _display_key(key),
                get_text(language, "value"): _display_value(value, language),
            }
        )
    return pd.DataFrame(
        rows
        or [
            {
                get_text(language, "field"): get_text(language, "not_available"),
                get_text(language, "value"): "",
            }
        ]
    )


def _flatten_mapping(payload: object, prefix: str = "") -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    rows: dict[str, Any] = {}
    for key, value in payload.items():
        current = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            rows.update(_flatten_mapping(value, current))
        elif isinstance(value, list):
            rows[current] = ", ".join(str(item) for item in value[:8])
        else:
            rows[current] = value
    return rows


def _display_value(value: object, language: str) -> str:
    if isinstance(value, float):
        return format_number(value, 3, language)
    if value is None:
        return get_text(language, "not_available")
    if isinstance(value, bool):
        return get_text(language, "yes") if value else get_text(language, "no")
    return str(value)


def _display_key(key: object) -> str:
    return str(key or "").replace("_", " ")


def _fit_metric_label(key: str, language: str) -> str:
    return get_text(language, f"model_comparison_fit_{key}")


def _comparison_severity_label(severity: str, language: str) -> str:
    return get_text(language, f"model_comparison_severity_{severity}")


def _comparison_warning_message(code: str, language: str) -> str:
    key = f"model_comparison_warning_{code}"
    label = get_text(language, key)
    return label if label != key else code.replace("_", " ")
