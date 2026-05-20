from __future__ import annotations

import html
import os
from io import BytesIO
import json

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.agent import AnalysisPlan, generate_analysis_plan, generate_narrative, run_guided_workflow
from src.agent.workflow_summary import workflow_result_frame, workflow_summary_markdown
from src.config import is_public_demo_mode
from src.data_quality import (
    build_data_quality_profile,
    build_missingness_profile,
    build_pre_model_risk_profile,
    build_resource_warning_profile,
    build_variable_quality_summaries,
    data_quality_to_jsonable,
)
from src.data_loader import load_dataset
from src.demo_data import demo_dataset_upload_cache, get_demo_dataset, list_demo_datasets
from src.demo_guides import DemoWorkflowGuide, build_demo_workflow_guide
from src.diagnostic_rendering import diagnostic_dicts
from src.file_errors import FriendlyFileError, build_file_quality_warnings
from src.formatting import format_diagnostic_table, format_number, format_p_narrative, format_regression_table, prepare_display_table
from src.i18n import get_text, normalize_language, translate_diagnostic_field, translate_warning
from src.llm.contracts import LLMExplanationMode
from src.models.binary_choice import prepare_binary_choice_dataframe
from src.models.did import prepare_did_dataframe
from src.models.iv import prepare_iv_dataframe
from src.models.ols import prepare_ols_dataframe
from src.models.panel_fe import check_panel_structure, prepare_panel_fe_dataframe
from src.models.psm import prepare_psm_dataframe
from src.models.execution import ModelSpec as ExecutableModelSpec, run_model_spec
from src.models.registry import get_available_models, get_model
from src.models.runners.registry import get_model_runner
from src.profiler import preprocess_dataframe, profile_dataframe
from src.report import build_categorical_control_summary, generate_markdown_report, generate_simple_report
from src.reproducibility.serializers import to_jsonable
from src.result_explainer import BeginnerResultGuide, build_beginner_result_guide
from src.run_history import add_run_to_history, create_analysis_run_record, run_history_to_dict
from src.session_reset import reset_analysis_session_state
from src.ui.did_config import (
    DID_EXPERIMENTAL_MODEL_ID,
    build_did_model_config,
    did_group_candidates,
    did_indicator_candidates,
    did_model_label,
    did_required_fields_present,
)
from src.ui.components import (
    render_action_card,
    render_callout,
    render_diagnostic_card,
    render_empty_state,
    render_metric_card,
    render_page_header,
    render_selection_card,
    render_section_header,
    render_status_card,
)
from src.ui.iv_config import (
    IV_EXPERIMENTAL_MODEL_ID,
    build_iv_model_config,
    iv_model_label,
    iv_numeric_candidates,
    iv_required_fields_present,
)
from src.preprocessing import apply_missing_data_plan
from src.ui.psm_config import (
    PSM_EXPERIMENTAL_MODEL_ID,
    PSM_UI_ESTIMANDS,
    PSM_UI_MATCHING_METHODS,
    build_psm_model_config,
    psm_matching_covariate_candidates,
    psm_model_label,
    psm_outcome_candidates,
    psm_required_fields_present,
    psm_treatment_candidates,
)
from src.ui.data_quality import render_data_quality_section, render_pre_model_risk_check, render_resource_warning_section
from src.ui.missing_data_plan import build_missing_data_state_update, missing_data_state_reset_keys, render_missing_data_plan_builder
from src.ui.render_diagnostics import structured_diagnostics_frame as _structured_diagnostics_frame
from src.ui.render_diagnostics import warning_lines_for_ui as _warning_lines_for_ui
from src.ui.render_export import render_reproducibility_pack_download
from src.ui.render_interpretation import explanation_layer_label as _explanation_layer_label
from src.ui.render_interpretation import explanation_mode_options as _explanation_mode_options
from src.ui.render_interpretation import render_mock_llm_interpretation as _render_mock_llm_interpretation
from src.ui.render_interpretation import render_research_interpretation as _render_research_interpretation
from src.ui.research_design_candidates import render_research_design_candidate_preview as _render_research_design_candidate_preview
from src.ui.run_history import render_run_history_preview
from src.ui.render_results import fit_metric_summary as _ui_fit_metric_summary
from src.ui.render_results import independent_variable_count as _independent_variable_count
from src.ui.render_results import main_effect_display as _ui_main_effect_display
from src.ui.render_results import model_display_name as _model_display_name
from src.ui.render_results import psm_key_findings as _psm_key_findings
from src.ui.render_results import render_advanced_outputs as _render_advanced_outputs
from src.ui.render_results import result_summary_metrics as _result_summary_metrics
from src.ui.theme import inject_regmonkey_theme
from src.utils import dataframe_to_csv_bytes
from src.variable_metadata import build_variable_reference
from src.variable_roles import (
    ROLE_CATEGORICAL,
    ROLE_CODE,
    ROLE_ENTITY,
    ROLE_EXCLUDE,
    ROLE_NUMERIC,
    ROLE_BINARY,
    ROLE_TIME,
    build_variable_role_table,
    get_binary_dependent_candidates,
    get_binary_variables,
    get_categorical_variables,
    get_code_identifier_variables,
    get_correlation_variables,
    get_entity_id_variables,
    get_numeric_measure_variables,
    get_ols_selector_variables,
    get_scatter_variables,
    get_time_variables,
    infer_variable_roles,
    role_from_label,
    role_label,
    role_options,
)
from src.visualization import (
    plot_boxplot,
    plot_categorical_bar,
    plot_correlation_heatmap,
    plot_histogram,
    plot_missing_values,
    plot_numeric_by_category,
    plot_pairwise_scatter,
    plot_scatter,
    plot_time_trend,
)


st.set_page_config(page_title="Reg Monkey", page_icon="🐒", layout="wide")


def _init_state() -> None:
    defaults = {
        "analysis_ran": False,
        "regression_table": None,
        "model_summary": None,
        "cleaned_df": None,
        "cleaning_log": None,
        "vif_df": None,
        "warnings": [],
        "structured_diagnostics": [],
        "model_results": None,
        "model_run_result": None,
        "explanation_mode": "rule_based",
        "simple_report": "",
        "full_report": "",
        "analysis_error": "",
        "data_signature": None,
        "uploaded_file_cache": None,
        "dataset_uploader_version": 0,
        "demo_dataset_id": "",
        "data_source_metadata": None,
        "confirm_preprocessing": False,
        "confirm_variable_roles": False,
        "variable_roles": None,
        "explored_data": False,
        "analysis_plan": None,
        "analysis_plan_generated": False,
        "analysis_plan_applied": False,
        "applied_plan_model_id": "",
        "active_model_id": "",
        "active_model_config": None,
        "model_setup_source": "",
        "model_setup_version": 0,
        "guided_workflow_result": None,
        "guided_workflow_has_run": False,
        "guided_workflow_status": "",
        "workflow_page": "setup",
        "plan_model_path": "recommended",
        "plan_manual_requested_model_id": "",
        "missing_data_handled_df": None,
        "missing_data_plan": None,
        "missing_data_handling_result": None,
        "missing_data_base_roles": None,
        "run_history": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _reset_analysis() -> None:
    for key in [
        "analysis_ran",
        "regression_table",
        "model_summary",
        "cleaned_df",
        "cleaning_log",
        "vif_df",
        "warnings",
        "structured_diagnostics",
        "model_results",
        "model_run_result",
        "simple_report",
        "full_report",
        "analysis_error",
    ]:
        st.session_state.pop(key, None)
    _init_state()


def _reset_planner() -> None:
    for key in [
        "analysis_plan",
        "analysis_plan_generated",
        "analysis_plan_applied",
        "applied_plan_model_id",
        "active_model_id",
        "active_model_config",
        "model_setup_source",
        "model_setup_version",
        "guided_workflow_result",
        "guided_workflow_has_run",
        "guided_workflow_status",
        "plan_model_path",
        "plan_manual_requested_model_id",
    ]:
        st.session_state.pop(key, None)
    _init_state()


def _reset_missing_data_handling() -> None:
    for key in missing_data_state_reset_keys():
        st.session_state.pop(key, None)
    _init_state()


def _reset_run_history() -> None:
    st.session_state.pop("run_history", None)
    _init_state()


def _reset_for_new_data_source() -> None:
    _reset_analysis()
    st.session_state.confirm_preprocessing = False
    st.session_state.confirm_variable_roles = False
    st.session_state.variable_roles = None
    st.session_state.explored_data = False
    _reset_missing_data_handling()
    _reset_planner()
    _reset_run_history()


def _invalidate_after_missing_data_change() -> None:
    _reset_analysis()
    _reset_planner()
    st.session_state.explored_data = False


def _roles_with_new_columns(df: pd.DataFrame, existing_roles: dict[str, str] | None) -> dict[str, str]:
    roles = dict(existing_roles or {})
    inferred = infer_variable_roles(df, profile_dataframe(df))
    for column in df.columns:
        if column not in roles:
            roles[column] = inferred.get(column, ROLE_BINARY if str(column).endswith("_missing") else ROLE_NUMERIC)
    return roles


def _stable_uploaded_file(uploaded_file: object | None) -> object | None:
    if uploaded_file is not None:
        data = uploaded_file.getvalue()
        source_metadata = {
            "kind": "upload",
            "filename": getattr(uploaded_file, "name", "uploaded_data"),
        }
        st.session_state.uploaded_file_cache = {
            "name": getattr(uploaded_file, "name", "uploaded_data"),
            "type": getattr(uploaded_file, "type", None),
            "data": data,
            "source_metadata": source_metadata,
        }
        st.session_state.data_source_metadata = source_metadata
        st.session_state.demo_dataset_id = ""
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        return uploaded_file

    cache = st.session_state.get("uploaded_file_cache")
    if not isinstance(cache, dict) or not cache.get("data"):
        return None
    restored = BytesIO(cache["data"])
    restored.name = str(cache.get("name") or "uploaded_data")
    restored.type = cache.get("type")
    restored.size = len(cache["data"])
    st.session_state.data_source_metadata = cache.get("source_metadata") or {
        "kind": "upload",
        "filename": restored.name,
    }
    return restored


def _uploaded_file_size_mb(uploaded_file: object | None) -> float | None:
    if uploaded_file is None:
        return None
    size = getattr(uploaded_file, "size", None)
    if size is None:
        try:
            size = len(uploaded_file.getvalue())
        except Exception:
            return None
    return round(float(size) / (1024 * 1024), 2)


def _load_demo_dataset_into_session(dataset_id: str) -> None:
    cache = demo_dataset_upload_cache(dataset_id)
    st.session_state.uploaded_file_cache = cache
    st.session_state.data_source_metadata = cache["source_metadata"]
    st.session_state.demo_dataset_id = dataset_id
    st.session_state.dataset_uploader_version = int(st.session_state.get("dataset_uploader_version") or 0) + 1
    st.session_state.skip_rows_input = 0
    st.session_state.use_first_row_as_header_checkbox = True
    st.session_state.auto_detect_metadata_checkbox = False
    st.session_state.coerce_numeric_checkbox = True
    st.session_state.data_signature = None
    st.session_state.workflow_page = "setup"
    _reset_for_new_data_source()


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        html, body, [class*="css"], [data-testid="stAppViewContainer"] {
            font-family: "Microsoft YaHei", "Segoe UI", "PingFang SC", "Noto Sans CJK SC", "SimHei", Arial, sans-serif;
        }
        div[data-testid="stTabs"] button p { font-size: 1.05rem; font-weight: 800; }
        div[data-testid="stTabs"] button[aria-selected="true"] { border-bottom: 3px solid #ff8c42; }
        .section-title { font-size: 1.6rem; font-weight: 850; margin-top: 0.3rem; }
        .section-guide { color: #5f6368; font-size: 1rem; margin-bottom: 1rem; }
        .rm-card {
            border: 1px solid #e8ebef;
            border-radius: 10px;
            background: #ffffff;
            padding: 0.9rem 1rem;
            margin: 0.35rem 0 0.75rem 0;
        }
        .rm-card-title { font-size: 0.96rem; font-weight: 780; color: #202734; margin-bottom: 0.35rem; }
        .rm-card-body { color: #68707c; font-size: 0.9rem; line-height: 1.45; }
        .rm-card-selected { border-color: #b84a3f; background: #fffafa; }
        .rm-summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 0.65rem;
            margin: 0.4rem 0 0.9rem 0;
        }
        .rm-summary-card {
            border: 1px solid #e9edf2;
            border-radius: 9px;
            background: #fbfcfd;
            padding: 0.75rem 0.85rem;
        }
        .rm-summary-label { color: #75808f; font-size: 0.78rem; margin-bottom: 0.2rem; }
        .rm-summary-value { color: #1f2937; font-size: 0.98rem; font-weight: 760; overflow-wrap: anywhere; }
        .rm-chip-row { margin: 0.25rem 0 0.55rem 0; color: #68707c; font-size: 0.86rem; }
        .rm-chip {
            display: inline-block;
            border: 1px solid #e4e8ee;
            border-radius: 999px;
            padding: 0.12rem 0.48rem;
            margin: 0.12rem 0.18rem 0.12rem 0;
            background: #fafbfc;
            color: #2f3846;
            font-size: 0.82rem;
        }
        .rm-callout {
            border-left: 3px solid #b84a3f;
            background: #fffafa;
            padding: 0.65rem 0.8rem;
            border-radius: 7px;
            margin: 0.4rem 0 0.75rem 0;
            color: #47505d;
            font-size: 0.9rem;
        }
        .workflow-stage-strip {
            color: #5f6368;
            font-size: 0.92rem;
            margin: 0.15rem 0 1.2rem 0;
            padding: 0.45rem 0;
            border: 0;
            border-radius: 0;
            background: transparent;
        }
        .workflow-stage-label {
            display: inline-block;
            margin: 0 0.45rem 0.35rem 0;
            color: #6f655c;
        }
        .workflow-stage-chip {
            display: inline-block;
            border: 1px solid #e4dad0;
            border-radius: 999px;
            padding: 0.16rem 0.56rem;
            margin: 0 0.24rem 0.35rem 0;
            background: #fffcf7;
            color: #6f655c;
            font-size: 0.82rem;
            font-weight: 600;
        }
        .workflow-stage-chip-complete {
            border-color: #d2dfd6;
            background: #f7fbf8;
            color: #4f745a;
        }
        .workflow-stage-chip-active {
            border-color: #a84a3f;
            background: #f4ddd8;
            color: #8f3e35;
        }
        .workflow-stage-index {
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 800;
            margin-right: 0.28rem;
            opacity: 0.76;
        }
        .workflow-stage-separator {
            color: #c7b9ad;
            display: inline-block;
            margin: 0 0.08rem 0.35rem 0;
        }
        .rm-app-shell {
            align-items: center;
            border-bottom: 1px solid #eee5dc;
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem 1rem;
            margin: 0 0 1.2rem 0;
            padding: 1.4rem 0 1.05rem 0;
        }
        .rm-app-logo {
            align-items: center;
            background: #fff7ec;
            border: 1px solid #ead9c8;
            border-radius: 14px;
            box-shadow: 0 4px 14px rgba(37, 33, 29, 0.06);
            display: inline-flex;
            font-size: 2.05rem;
            height: 3.25rem;
            justify-content: center;
            line-height: 1;
            width: 3.25rem;
        }
        .rm-app-identity {
            align-items: center;
            display: flex;
            gap: 0.78rem;
        }
        .rm-app-brand {
            color: #25211d;
            font-size: 1.22rem;
            font-weight: 850;
            letter-spacing: 0;
            line-height: 1.15;
        }
        .rm-app-subtitle {
            color: #6f655c;
            font-size: 0.86rem;
            margin-top: 0.2rem;
        }
        .rm-app-tagline {
            background: #fffcf7;
            border: 1px solid #e8ddd1;
            border-radius: 999px;
            color: #7a6d62;
            font-size: 0.82rem;
            font-weight: 650;
            padding: 0.22rem 0.58rem;
        }
        .setup-flow-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            margin-top: 0.55rem;
        }
        .setup-flow-chip {
            border: 1px solid #e7e9ee;
            border-radius: 999px;
            background: #fffcf7;
            color: #5f6368;
            font-size: 0.82rem;
            font-weight: 650;
            padding: 0.18rem 0.55rem;
        }
        .setup-entry-card {
            border: 1px solid #e4dad0;
            border-radius: 8px;
            background: #ffffff;
            padding: 1rem;
            min-height: 16rem;
        }
        .setup-entry-card-primary {
            border-color: #b9d0bf;
            background: #fbfffc;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
            background: transparent;
            border: 1px solid transparent;
            box-shadow: none;
            justify-content: flex-start;
            color: #4f5663;
            padding: 0.34rem 0.35rem;
            min-height: 2rem;
            font-weight: 520;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
            background: #f7f8fa;
            border-color: #edf0f4;
            color: #24292f;
        }
        div[data-testid="stButton"] > button[kind="primary"] {
            background: #9b3a32;
            border-color: #9b3a32;
        }
        div[data-testid="stButton"] > button[kind="primary"]:hover {
            background: #842f29;
            border-color: #842f29;
        }
        section[data-testid="stSidebar"] [data-testid="stExpander"] {
            border-color: #efe5db;
            background: #fffaf3;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _section_header(title: str, guide: str) -> None:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-guide'>{guide}</div>", unsafe_allow_html=True)


def _workflow_pages(language: str) -> list[tuple[str, str, str, str]]:
    t = lambda key: get_text(language, key)
    return [
        ("setup", t("tab_setup"), t("tab_setup_title"), t("guide_setup")),
        ("understand", t("tab_understand"), t("tab_understand_title"), t("guide_understand")),
        ("plan", t("tab_plan"), t("tab_plan_title"), t("guide_plan")),
        ("run", t("tab_run"), t("tab_run_title"), t("guide_run")),
        ("interpret", t("tab_interpret"), t("tab_interpret_title"), t("guide_interpret")),
        ("export", t("tab_export"), t("tab_export_title"), t("guide_export")),
    ]


def _recommended_page_id() -> str:
    if st.session_state.full_report:
        return ""
    if not st.session_state.confirm_preprocessing or not st.session_state.confirm_variable_roles:
        return "setup"
    if not st.session_state.explored_data:
        return "understand"
    analysis_complete = bool(st.session_state.analysis_ran or st.session_state.guided_workflow_has_run)
    setup_ready = bool(st.session_state.get("active_model_id"))
    if not st.session_state.analysis_plan_generated and not setup_ready and not analysis_complete:
        return "plan"
    if not setup_ready and not analysis_complete:
        return "plan"
    if not analysis_complete:
        return "run"
    if not st.session_state.full_report:
        return "export"
    return "interpret"


def _go_to_page(page_id: str, anchor_id: str | None = None) -> None:
    if anchor_id:
        _request_scroll_anchor(anchor_id)
    st.session_state.workflow_page = page_id
    st.rerun()


def _next_action(label: str) -> str:
    text = str(label).rstrip()
    return text if text.endswith("→") else f"{text} →"


def _down_action(label: str) -> str:
    text = str(label).rstrip()
    return text if text.endswith("↓") else f"{text} ↓"


def _request_scroll_anchor(anchor_id: str) -> None:
    st.session_state.scroll_anchor_requested = anchor_id


def _scroll_anchor(anchor_id: str) -> None:
    safe_anchor = html.escape(anchor_id, quote=True)
    st.markdown(f"<span id='{safe_anchor}'></span>", unsafe_allow_html=True)
    if st.session_state.get("scroll_anchor_requested") != anchor_id:
        return
    st.session_state.pop("scroll_anchor_requested", None)
    components.html(
        f"""
        <script>
        try {{
          const doc = window.parent.document;
          const target = doc.getElementById("{safe_anchor}");
          if (target) {{
            target.scrollIntoView({{ block: "start", inline: "nearest", behavior: "auto" }});
          }}
        }} catch (error) {{}}
        </script>
        """,
        height=0,
        width=0,
    )


def _workflow_stepper(
    pages: list[tuple[str, str, str, str]],
    current_page: str,
    completed: dict[str, bool],
    next_page: str,
    language: str,
) -> None:
    t = lambda key: get_text(language, key)
    label_by_id = {page_id: label for page_id, label, _, _ in pages}
    for page_id, label, _, _ in pages:
        if completed.get(page_id, False):
            marker = "✓"
        elif page_id == current_page or page_id == next_page:
            marker = "→"
        else:
            marker = "·"
        if st.button(f"{marker} {label}", key=f"sidebar_workflow_step_{page_id}", type="secondary", width="stretch"):
            _go_to_page(page_id)
    if next_page:
        st.caption(f"{t('next_recommended_step')}: {label_by_id.get(next_page, '')}")
    else:
        st.caption(t("next_action_done"))


def _render_app_shell(pages: list[tuple[str, str, str, str]], current_page: str, language: str) -> None:
    t = lambda key: get_text(language, key)
    label_by_id = {page_id: label for page_id, label, _, _ in pages}
    current_label = label_by_id.get(current_page, "")
    st.markdown(
        "<div class='rm-app-shell'>"
        "<div class='rm-app-identity'>"
        "<div class='rm-app-logo' aria-hidden='true'>🐒</div>"
        "<div>"
        f"<div class='rm-app-brand'>{html.escape(t('app_title'))}</div>"
        f"<div class='rm-app-subtitle'>{html.escape(t('app_subtitle'))}</div>"
        "</div>"
        "</div>"
        f"<div class='rm-app-tagline'>{html.escape(t('app_shell_caption').format(step=current_label))}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_reset_session_control(language: str) -> None:
    t = lambda key: get_text(language, key)
    version = int(st.session_state.get("reset_session_version") or 0)
    pending_key = f"reset_session_pending_{version}"
    st.divider()
    st.subheader(t("reset_session"))
    if not st.session_state.get(pending_key):
        if st.button(
            _down_action(t("reset_session")),
            key=f"reset_session_begin_button_{version}",
            type="secondary",
            width="stretch",
        ):
            st.session_state[pending_key] = True
            st.rerun()
        return

    st.caption(t("reset_session_help"))
    st.caption(t("reset_session_language_preserved"))
    if is_public_demo_mode():
        st.caption(t("reset_session_public_demo_hint"))
    confirmed = st.checkbox(t("reset_session_confirm"), key=f"reset_session_confirm_{version}")
    if st.button(
        t("reset_session_apply"),
        key=f"reset_session_button_{version}",
        type="secondary",
        width="stretch",
        disabled=not confirmed,
    ):
        reset_analysis_session_state(st.session_state, preserve_keys={"language_selector"})
        st.session_state.reset_session_version = version + 1
        st.rerun()
    if st.button(
        t("reset_session_cancel"),
        key=f"reset_session_cancel_button_{version}",
        type="secondary",
        width="stretch",
    ):
        st.session_state[pending_key] = False
        st.rerun()


def _render_friendly_file_error(error: FriendlyFileError, language: str) -> None:
    t = lambda key: get_text(language, key)
    render_callout(t(error.title_key), t(error.message_key), tone="warning")
    st.caption(t(error.action_key))
    if error.affected_columns:
        st.caption(f"{t('friendly_file_error_affected_columns')}: {', '.join(error.affected_columns[:8])}")


def _render_file_quality_warnings(df: pd.DataFrame, language: str) -> None:
    t = lambda key: get_text(language, key)
    for warning in build_file_quality_warnings(df):
        render_callout(t(warning.title_key), t(warning.message_key), tone="warning")
        st.caption(t(warning.action_key))


def _stage_strip(pages: list[tuple[str, str, str, str]], current_page: str, language: str) -> None:
    t = lambda key: get_text(language, key)
    label_by_id = {page_id: label for page_id, label, _, _ in pages}
    current_index = next((index for index, (page_id, _, _, _) in enumerate(pages) if page_id == current_page), 0)
    chips: list[str] = []
    for index, (page_id, label, _, _) in enumerate(pages, start=1):
        classes = ["workflow-stage-chip"]
        if page_id == current_page:
            classes.append("workflow-stage-chip-active")
        elif index - 1 < current_index:
            classes.append("workflow-stage-chip-complete")
        chip = (
            f"<span class='{' '.join(classes)}'>"
            f"<span class='workflow-stage-index'>{index}</span>{html.escape(label)}"
            "</span>"
        )
        chips.append(chip)
        if index < len(pages):
            chips.append("<span class='workflow-stage-separator'>›</span>")
    chip_markup = "".join(chips)
    st.markdown(
        "<div class='workflow-stage-strip'>"
        f"<span class='workflow-stage-label'>{html.escape(t('current_stage'))}: "
        f"<strong>{html.escape(label_by_id.get(current_page, ''))}</strong></span>"
        f"{chip_markup}"
        "</div>",
        unsafe_allow_html=True,
    )


def _status_strip(items: list[tuple[str, str]]) -> None:
    if not items:
        return
    cards = "".join(
        "<div class='rm-summary-card'>"
        f"<div class='rm-summary-label'>{html.escape(str(label))}</div>"
        f"<div class='rm-summary-value'>{html.escape(str(value))}</div>"
        "</div>"
        for label, value in items
    )
    st.markdown(f"<div class='rm-summary-grid'>{cards}</div>", unsafe_allow_html=True)


def _card(title: str, body: str, selected: bool = False) -> None:
    selected_class = " rm-card-selected" if selected else ""
    st.markdown(
        f"<div class='rm-card{selected_class}'>"
        f"<div class='rm-card-title'>{html.escape(str(title))}</div>"
        f"<div class='rm-card-body'>{html.escape(str(body))}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _action_card(title: str, body: str, button_label: str | None = None, target_page: str | None = None, key: str | None = None) -> None:
    _card(title, body)
    if button_label and target_page and key:
        if st.button(_next_action(button_label), type="primary", key=key):
            _go_to_page(target_page)


def _decision_cards(
    left_title: str,
    left_body: str,
    right_title: str,
    right_body: str,
    selected: str,
) -> None:
    left, right = st.columns(2)
    with left:
        _card(left_title, left_body, selected=selected == "recommended")
    with right:
        _card(right_title, right_body, selected=selected == "manual")


def _empty_state_card(title: str, body: str, button_label: str, target_page: str, key: str) -> None:
    _action_card(title, body, button_label, target_page, key)


def _model_summary_card(language: str, title: str, rows: list[tuple[str, str]]) -> None:
    st.markdown(f"**{title}**")
    _status_strip(rows)


def _chip_row(label: str, values: list[str] | tuple[str, ...]) -> None:
    shown = [str(value) for value in values if str(value)]
    if shown:
        chips = "".join(f"<span class='rm-chip'>{html.escape(value)}</span>" for value in shown)
        st.markdown(
            f"<div class='rm-chip-row'><strong>{html.escape(str(label))}</strong>: {chips}</div>",
            unsafe_allow_html=True,
        )


def _callout(text: str) -> None:
    st.markdown(f"<div class='rm-callout'>{html.escape(str(text))}</div>", unsafe_allow_html=True)


def _dataframe_block(title: str, caption: str, frame: pd.DataFrame, expanded: bool = False) -> None:
    with st.expander(title, expanded=expanded):
        if caption:
            st.caption(caption)
        st.dataframe(frame, width="stretch")


def _text_value_examples(df: pd.DataFrame, limit: int = 8) -> str:
    examples: list[str] = []
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            continue
        for value in df[column].dropna().astype(str).str.strip().unique().tolist():
            if value and value not in examples:
                examples.append(value)
            if len(examples) >= limit:
                return ", ".join(examples)
    return ", ".join(examples)


def _workflow_item(done: bool, label: str) -> None:
    marker = "✓" if done else "·"
    st.caption(f"{marker} {label}")


def _next_action_text(language: str) -> str:
    t = lambda key: get_text(language, key)
    if "data_signature" not in st.session_state or st.session_state.data_signature is None:
        return t("next_action_upload")
    if not st.session_state.confirm_preprocessing:
        return t("next_action_clean")
    if not st.session_state.confirm_variable_roles:
        return t("next_action_roles")
    if not st.session_state.explored_data:
        return t("next_action_explore")
    if not st.session_state.analysis_ran:
        return t("next_action_model")
    if not st.session_state.full_report:
        return t("next_action_report")
    return t("next_action_done")


def _disabled_reason(language: str, selected_model: object | None = None, y_col: str = "", main_x: list[str] | None = None) -> str:
    t = lambda key: get_text(language, key)
    if not st.session_state.confirm_preprocessing:
        return t("disabled_confirm_cleaned")
    if not st.session_state.confirm_variable_roles:
        return t("disabled_confirm_roles")
    if selected_model is None or not y_col or not main_x:
        return t("disabled_select_variables")
    return t("run_disabled_reason")


def _model_label(model_id: str, language: str) -> str:
    if not model_id:
        return ""
    return _model_definition(model_id).display_name(normalize_language(language))


def _model_definition(model_id: str) -> object:
    try:
        return get_model(model_id)
    except ValueError:
        return get_model_runner(model_id).model_definition


def _planner_warning_frame(plan: AnalysisPlan) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "severity": warning.severity,
                "code": warning.code,
                "message": warning.message,
                "variables": ", ".join(warning.variables),
            }
            for warning in plan.warnings
        ]
    )


def _planner_candidate_frame(candidates: list[object]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variable": candidate.name,
                "role": candidate.role,
                "confidence": round(float(candidate.confidence), 2),
                "reason": candidate.reason,
            }
            for candidate in candidates
        ]
    )


def _apply_analysis_plan_to_state(plan: AnalysisPlan, language: str) -> None:
    model = plan.recommended_main_model
    if model is None:
        return

    st.session_state.analysis_plan_applied = True
    st.session_state.applied_plan_model_id = model.model_id
    config = _config_from_analysis_plan(plan)
    _store_model_setup(model.model_id, config, "recommended")
    st.session_state.model_setup_version += 1


def _applied_plan_for_model(model_id: str) -> AnalysisPlan | None:
    if not st.session_state.analysis_plan_applied:
        return None
    if st.session_state.applied_plan_model_id != model_id:
        return None
    return st.session_state.analysis_plan


def _config_from_analysis_plan(plan: AnalysisPlan) -> dict:
    model_id = plan.recommended_main_model.model_id if plan.recommended_main_model else ""
    standard_errors = "cluster_entity" if model_id == "panel_fe" else "hc3"
    return {
        "dependent_variable": plan.recommended_dependent_variable,
        "main_independent_variables": list(plan.recommended_main_explanatory_variables),
        "numeric_control_variables": list(plan.numeric_controls),
        "categorical_control_variables": list(plan.categorical_controls if model_id == "ols" else []),
        "encode_categorical_controls": False,
        "robust_standard_errors": standard_errors != "conventional",
        "entity_id": plan.entity_id if model_id == "panel_fe" else "",
        "time_id": plan.time_id if model_id == "panel_fe" else "",
        "entity_effects": bool(plan.fixed_effects.get("entity", True)) if model_id == "panel_fe" else False,
        "time_effects": bool(plan.fixed_effects.get("time", True)) if model_id == "panel_fe" else False,
        "standard_errors": standard_errors,
    }


def _store_model_setup(model_id: str, config: dict, source: str) -> None:
    st.session_state.active_model_id = model_id
    st.session_state.active_model_config = dict(config)
    st.session_state.model_setup_source = source


def _active_model_setup() -> tuple[str, dict]:
    return st.session_state.get("active_model_id", ""), dict(st.session_state.get("active_model_config") or {})


def _setup_source_label(language: str, source: str) -> str:
    t = lambda key: get_text(language, key)
    if source == "recommended":
        return t("setup_source_recommended")
    if source == "manual":
        return t("setup_source_manual")
    return t("status_pending")


def _run_history_source(model_id: str) -> str:
    if model_id in {DID_EXPERIMENTAL_MODEL_ID, IV_EXPERIMENTAL_MODEL_ID, PSM_EXPERIMENTAL_MODEL_ID}:
        return "experimental_manual"
    if st.session_state.get("model_setup_source") == "recommended":
        return "recommended_plan"
    return "manual_configuration"


def _current_preprocessing_signature() -> str:
    payload = {
        "data_signature": st.session_state.get("data_signature"),
        "missing_data_plan_applied": bool(st.session_state.get("missing_data_plan")),
        "missing_data_plan": st.session_state.get("missing_data_plan") or {},
        "missing_data_result": st.session_state.get("missing_data_handling_result") or {},
    }
    return json.dumps(to_jsonable(payload), ensure_ascii=False, sort_keys=True)


def _record_run_history(run_result: object, model_id: str) -> None:
    record = create_analysis_run_record(
        run_result,
        data_signature=json.dumps(to_jsonable(st.session_state.get("data_signature")), ensure_ascii=False, sort_keys=True),
        preprocessing_signature=_current_preprocessing_signature(),
        missing_data_plan_applied=bool(st.session_state.get("missing_data_plan")),
        source=_run_history_source(model_id),
    )
    history = add_run_to_history(st.session_state.get("run_history"), record)
    st.session_state.run_history = run_history_to_dict(history)


def _model_setup_rows(model_id: str, config: dict, language: str) -> list[tuple[str, str]]:
    t = lambda key: get_text(language, key)
    model_name = _model_label(model_id, language) if model_id else t("select_model_placeholder")
    if model_id == DID_EXPERIMENTAL_MODEL_ID:
        standard_errors = t("clustered_se") if config.get("cluster_variable") else t("robust_se").replace("Use HC3 ", "").replace("使用 HC3 ", "")
        return [
            (t("run_status_model"), model_name),
            (t("run_status_y"), config.get("dependent_variable") or t("status_pending")),
            (t("treatment_variable"), config.get("treatment_variable") or t("status_pending")),
            (t("post_variable"), config.get("post_variable") or t("status_pending")),
            (t("numeric_controls"), ", ".join(config.get("numeric_control_variables") or []) or t("not_available")),
            (t("categorical_controls"), ", ".join(config.get("categorical_control_variables") or []) or t("not_available")),
            (t("group_variable"), config.get("group_variable") or t("not_available")),
            (t("cluster_variable"), config.get("cluster_variable") or t("not_available")),
            (t("standard_errors"), standard_errors),
        ]
    if model_id == IV_EXPERIMENTAL_MODEL_ID:
        return [
            (t("run_status_model"), model_name),
            (t("run_status_y"), config.get("dependent_variable") or t("status_pending")),
            (t("endogenous_variable"), config.get("endogenous_variable") or t("status_pending")),
            (t("instrument_variables"), ", ".join(config.get("instruments") or []) or t("status_pending")),
            (t("exogenous_controls"), ", ".join(config.get("exogenous_controls") or config.get("numeric_control_variables") or []) or t("not_available")),
            (t("standard_errors"), t("conventional")),
        ]
    if model_id == PSM_EXPERIMENTAL_MODEL_ID:
        return [
            (t("run_status_model"), model_name),
            (t("run_status_y"), config.get("dependent_variable") or t("status_pending")),
            (t("treatment_variable"), config.get("treatment_variable") or t("status_pending")),
            (t("matching_covariates"), ", ".join(config.get("matching_covariates") or []) or t("status_pending")),
            (t("psm_estimand"), config.get("psm_estimand") or "ATT"),
            (t("matching_method"), t("nearest_neighbor")),
            (t("replacement_matching"), t("yes") if config.get("matching_method", "nearest_neighbor") == "nearest_neighbor" else t("not_available")),
            (t("caliper"), str(config.get("caliper")) if config.get("caliper") is not None else t("not_available")),
        ]
    rows = [
        (t("run_status_model"), model_name),
        (t("run_status_y"), config.get("dependent_variable") or t("status_pending")),
        (t("run_status_main_x"), ", ".join(config.get("main_independent_variables") or []) or t("status_pending")),
        (t("numeric_controls"), ", ".join(config.get("numeric_control_variables") or []) or t("not_available")),
    ]
    if model_id == "panel_fe":
        fe_parts = []
        if config.get("entity_effects"):
            fe_parts.append(t("entity_fixed_effects"))
        if config.get("time_effects"):
            fe_parts.append(t("time_fixed_effects"))
        rows.extend(
            [
                (t("entity_id"), config.get("entity_id") or t("status_pending")),
                (t("time_id"), config.get("time_id") or t("status_pending")),
                (t("fixed_effects_summary"), " + ".join(fe_parts) or t("status_pending")),
                (t("standard_errors"), t("cluster_entity") if config.get("standard_errors") == "cluster_entity" else config.get("standard_errors", "")),
            ]
        )
    return rows


def _model_category_label(model_id: str, language: str) -> str:
    t = lambda key: get_text(language, key)
    if model_id in {DID_EXPERIMENTAL_MODEL_ID, IV_EXPERIMENTAL_MODEL_ID, PSM_EXPERIMENTAL_MODEL_ID}:
        return t("experimental_manual_category")
    if model_id:
        return t("basic_model_category")
    return t("status_pending")


def _manual_family_focus(model_id: str, language: str) -> str:
    t = lambda key: get_text(language, key)
    if model_id == DID_EXPERIMENTAL_MODEL_ID:
        return t("did_workspace_focus")
    if model_id == IV_EXPERIMENTAL_MODEL_ID:
        return t("iv_workspace_focus")
    if model_id == PSM_EXPERIMENTAL_MODEL_ID:
        return t("psm_workspace_focus")
    if model_id:
        return t("basic_workspace_focus")
    return t("manual_workspace_empty")


def _manual_setup_summary_title(model_id: str, language: str) -> str:
    t = lambda key: get_text(language, key)
    if model_id == DID_EXPERIMENTAL_MODEL_ID:
        return t("did_setup_summary")
    if model_id == IV_EXPERIMENTAL_MODEL_ID:
        return t("iv_setup_summary")
    if model_id == PSM_EXPERIMENTAL_MODEL_ID:
        return t("psm_setup_summary")
    return t("model_specific_setup_summary")


def _render_manual_family_banner(model_id: str, language: str) -> None:
    if not model_id:
        return
    t = lambda key: get_text(language, key)
    category = _model_category_label(model_id, language)
    tone = "experimental" if model_id in {DID_EXPERIMENTAL_MODEL_ID, IV_EXPERIMENTAL_MODEL_ID, PSM_EXPERIMENTAL_MODEL_ID} else "info"
    body = _manual_family_focus(model_id, language)
    st.markdown(
        "<div class='rm-family-summary "
        f"{'rm-family-summary-experimental' if tone == 'experimental' else ''}'>"
        f"<div class='rm-section-title-row'><strong>{html.escape(_model_label(model_id, language))}</strong>"
        f"<span class='rm-badge rm-badge-{tone}'>{html.escape(category)}</span></div>"
        f"<div class='rm-card-body'>{html.escape(body)}</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    if tone == "experimental":
        render_callout(t("experimental_confirmation_required"), t("experimental_manual_model_family_body"), tone="warning")


def _render_model_setup_form(
    df: pd.DataFrame,
    confirmed_roles: dict[str, str],
    language: str,
    key_prefix: str,
    initial_model_id: str = "",
    initial_config: dict | None = None,
    applied_plan: AnalysisPlan | None = None,
    show_ui_test_defaults: bool = False,
) -> dict:
    t = lambda key: get_text(language, key)
    initial_config = initial_config or {}
    role_selectors = get_ols_selector_variables(confirmed_roles)
    numeric_candidates = role_selectors["main_independent_variables"]
    categorical_candidates = role_selectors["categorical_controls"]
    model_options = [""] + [model.model_id for model in get_available_models()] + [
        DID_EXPERIMENTAL_MODEL_ID,
        IV_EXPERIMENTAL_MODEL_ID,
        PSM_EXPERIMENTAL_MODEL_ID,
    ]
    labels = {model.model_id: model.display_name(normalize_language(language)) for model in get_available_models()}
    labels[DID_EXPERIMENTAL_MODEL_ID] = did_model_label(language)
    labels[IV_EXPERIMENTAL_MODEL_ID] = iv_model_label(language)
    labels[PSM_EXPERIMENTAL_MODEL_ID] = psm_model_label(language)
    setup_version = st.session_state.model_setup_version
    model_index = model_options.index(initial_model_id) if initial_model_id in model_options else 0
    selected_model_id = st.selectbox(
        t("select_model"),
        options=model_options,
        index=model_index,
        format_func=lambda value: t("select_model_placeholder") if value == "" else labels[value],
        help=t("selector_help"),
        key=f"{key_prefix}_model_selector_{setup_version}",
    )
    selected_model = _model_definition(selected_model_id) if selected_model_id else None

    y_col = ""
    main_x: list[str] = []
    numeric_controls: list[str] = []
    categorical_controls: list[str] = []
    encode_categoricals = False
    use_robust = True
    entity_col = ""
    time_col = ""
    treatment_col = ""
    post_col = ""
    group_col = ""
    cluster_col = ""
    endogenous_col = ""
    instrument_cols: list[str] = []
    exogenous_controls: list[str] = []
    matching_covariates: list[str] = []
    psm_estimand = "ATT"
    matching_method = "nearest_neighbor"
    caliper: float | None = None
    entity_effects = False
    time_effects = False
    standard_errors = "hc3"

    if selected_model is None:
        st.info(t("select_model_placeholder"))
        return {"model_id": "", "model": None, "config": {}, "run_valid": False, "errors": []}

    model_key = selected_model.model_id
    _render_manual_family_banner(selected_model.model_id, language)
    if selected_model.model_id == "ols":
        st.info(t("ols_numeric_only"))
        y_options = role_selectors["dependent_variables"]
    elif selected_model.model_id == "panel_fe":
        st.info(t("panel_model_hint"))
        y_options = numeric_candidates
    elif selected_model.model_id == DID_EXPERIMENTAL_MODEL_ID:
        st.info(t("did_experimental_body"))
        st.caption(t("did_assumption_caution"))
        y_options = numeric_candidates
    elif selected_model.model_id == IV_EXPERIMENTAL_MODEL_ID:
        st.info(t("iv_experimental_body"))
        st.caption(t("iv_assumption_caution"))
        y_options = numeric_candidates
    elif selected_model.model_id == PSM_EXPERIMENTAL_MODEL_ID:
        st.info(t("psm_experimental_body"))
        st.caption(t("psm_assumption_caution"))
        y_options = psm_outcome_candidates(confirmed_roles)
    else:
        st.info(t("binary_model_hint"))
        y_options = get_binary_dependent_candidates(df, confirmed_roles)
    st.caption(selected_model.description(normalize_language(language)))

    if show_ui_test_defaults and os.environ.get("REG_MONKEY_UI_TEST_MODE") == "1":
        if st.button("应用 UI 测试默认变量", key=f"{key_prefix}_ui_test_defaults_{model_key}"):
            st.session_state.ui_test_defaults_model = selected_model.model_id
            st.rerun()

    y_default = initial_config.get("dependent_variable") or (applied_plan.recommended_dependent_variable if applied_plan else "")
    y_index = ([""] + list(y_options)).index(y_default) if y_default in y_options else 0
    y_col = st.selectbox(
        t("dependent_variable"),
        [""] + list(y_options),
        index=y_index,
        format_func=lambda value: t("select_dependent") if value == "" else value,
        help=t("dependent_variable_help"),
        key=f"{key_prefix}_{model_key}_dependent_variable_select_{setup_version}",
    )
    if selected_model.model_id == "ols" and y_col and confirmed_roles.get(y_col) == ROLE_BINARY:
        st.warning(t("ols_binary_y_warning"))

    x_pool = [col for col in numeric_candidates if col != y_col]
    if selected_model.model_id == DID_EXPERIMENTAL_MODEL_ID:
        st.subheader(t("did_variable_settings"))
        indicator_options = [col for col in did_indicator_candidates(confirmed_roles) if col != y_col]
        treatment_default = initial_config.get("treatment_variable") or ""
        treatment_index = ([""] + indicator_options).index(treatment_default) if treatment_default in indicator_options else 0
        treatment_col = st.selectbox(
            t("treatment_variable"),
            [""] + indicator_options,
            index=treatment_index,
            format_func=lambda value: t("select_treatment_variable") if value == "" else value,
            help=t("did_indicator_help"),
            key=f"{key_prefix}_{model_key}_treatment_variable_select_{setup_version}",
        )
        post_options = [col for col in indicator_options if col != treatment_col]
        post_default = initial_config.get("post_variable") or ""
        post_index = ([""] + post_options).index(post_default) if post_default in post_options else 0
        post_col = st.selectbox(
            t("post_variable"),
            [""] + post_options,
            index=post_index,
            format_func=lambda value: t("select_post_variable") if value == "" else value,
            help=t("did_indicator_help"),
            key=f"{key_prefix}_{model_key}_post_variable_select_{setup_version}",
        )
        did_excluded = {y_col, treatment_col, post_col}
        control_defaults = initial_config.get("numeric_control_variables") or []
        numeric_controls_default = [col for col in control_defaults if col in x_pool and col not in did_excluded]
        numeric_controls = st.multiselect(
            t("numeric_controls"),
            [col for col in x_pool if col not in did_excluded],
            default=numeric_controls_default,
            help=t("numeric_controls_help"),
            key=f"{key_prefix}_{model_key}_numeric_controls_select_{setup_version}",
        )
        categorical_defaults = initial_config.get("categorical_control_variables") or []
        categorical_default = [col for col in categorical_defaults if col in categorical_candidates and col not in did_excluded]
        categorical_controls = st.multiselect(
            t("categorical_controls"),
            [col for col in categorical_candidates if col not in did_excluded],
            default=categorical_default,
            help=t("categorical_controls_help"),
            key=f"{key_prefix}_{model_key}_categorical_controls_select_{setup_version}",
        )
        encode_categoricals = st.checkbox(
            t("encode_categorical_controls"),
            value=bool(initial_config.get("encode_categorical_controls", False)),
            help=t("dummy_encoding_help"),
            key=f"{key_prefix}_{model_key}_encode_categorical_controls_checkbox_{setup_version}",
        )
        st.subheader(t("did_structure_settings"))
        group_options = [col for col in did_group_candidates(confirmed_roles) if col not in did_excluded]
        group_default = initial_config.get("group_variable") or ""
        group_index = ([""] + group_options).index(group_default) if group_default in group_options else 0
        group_col = st.selectbox(
            t("group_variable"),
            [""] + group_options,
            index=group_index,
            format_func=lambda value: t("optional_not_selected") if value == "" else value,
            help=t("group_variable_help"),
            key=f"{key_prefix}_{model_key}_group_variable_select_{setup_version}",
        )
        cluster_default = initial_config.get("cluster_variable") or ""
        cluster_index = ([""] + group_options).index(cluster_default) if cluster_default in group_options else 0
        cluster_col = st.selectbox(
            t("cluster_variable"),
            [""] + group_options,
            index=cluster_index,
            format_func=lambda value: t("optional_not_selected") if value == "" else value,
            help=t("cluster_variable_help"),
            key=f"{key_prefix}_{model_key}_cluster_variable_select_{setup_version}",
        )
        standard_errors = "cluster" if cluster_col else "hc3"
        use_robust = True
        _advanced_model_options(selected_model.model_id, language)
    elif selected_model.model_id == IV_EXPERIMENTAL_MODEL_ID:
        st.subheader(t("iv_variable_settings"))
        iv_candidates = [col for col in iv_numeric_candidates(confirmed_roles) if col != y_col]
        endogenous_default = initial_config.get("endogenous_variable") or ""
        endogenous_index = ([""] + iv_candidates).index(endogenous_default) if endogenous_default in iv_candidates else 0
        endogenous_col = st.selectbox(
            t("endogenous_variable"),
            [""] + iv_candidates,
            index=endogenous_index,
            format_func=lambda value: t("select_endogenous_variable") if value == "" else value,
            help=t("iv_numeric_variable_help"),
            key=f"{key_prefix}_{model_key}_endogenous_variable_select_{setup_version}",
        )
        iv_excluded = {y_col, endogenous_col}
        instrument_options = [col for col in iv_candidates if col not in iv_excluded]
        instrument_defaults = [col for col in (initial_config.get("instruments") or []) if col in instrument_options]
        instrument_cols = st.multiselect(
            t("instrument_variables"),
            instrument_options,
            default=instrument_defaults,
            placeholder=t("select_instruments"),
            help=t("iv_instruments_help"),
            key=f"{key_prefix}_{model_key}_instrument_variables_select_{setup_version}",
        )
        control_options = [col for col in x_pool if col not in {*iv_excluded, *instrument_cols}]
        control_defaults = initial_config.get("exogenous_controls") or initial_config.get("numeric_control_variables") or []
        exogenous_controls = [
            col for col in control_defaults if col in control_options
        ]
        exogenous_controls = st.multiselect(
            t("exogenous_controls"),
            control_options,
            default=exogenous_controls,
            help=t("iv_exogenous_controls_help"),
            key=f"{key_prefix}_{model_key}_exogenous_controls_select_{setup_version}",
        )
        numeric_controls = list(exogenous_controls)
        use_robust = False
        standard_errors = "conventional"
        _advanced_model_options(selected_model.model_id, language)
    elif selected_model.model_id == PSM_EXPERIMENTAL_MODEL_ID:
        st.subheader(t("psm_variable_settings"))
        treatment_options = [col for col in psm_treatment_candidates(confirmed_roles) if col != y_col]
        treatment_default = initial_config.get("treatment_variable") or ""
        treatment_index = ([""] + treatment_options).index(treatment_default) if treatment_default in treatment_options else 0
        treatment_col = st.selectbox(
            t("treatment_variable"),
            [""] + treatment_options,
            index=treatment_index,
            format_func=lambda value: t("select_treatment_variable") if value == "" else value,
            help=t("psm_treatment_help"),
            key=f"{key_prefix}_{model_key}_treatment_variable_select_{setup_version}",
        )
        psm_excluded = {y_col, treatment_col}
        covariate_options = [col for col in psm_matching_covariate_candidates(confirmed_roles) if col not in psm_excluded]
        covariate_defaults = [col for col in (initial_config.get("matching_covariates") or []) if col in covariate_options]
        matching_covariates = st.multiselect(
            t("matching_covariates"),
            covariate_options,
            default=covariate_defaults,
            placeholder=t("select_matching_covariates"),
            help=t("psm_matching_covariates_help"),
            key=f"{key_prefix}_{model_key}_matching_covariates_select_{setup_version}",
        )
        psm_estimand = st.selectbox(
            t("psm_estimand"),
            PSM_UI_ESTIMANDS,
            index=0,
            format_func=lambda value: t("att_estimand") if value == "ATT" else value,
            help=t("psm_estimand_help"),
            key=f"{key_prefix}_{model_key}_psm_estimand_select_{setup_version}",
        )
        matching_method = st.selectbox(
            t("matching_method"),
            PSM_UI_MATCHING_METHODS,
            index=0,
            format_func=lambda value: t("nearest_neighbor") if value == "nearest_neighbor" else value,
            help=t("psm_matching_method_help"),
            key=f"{key_prefix}_{model_key}_matching_method_select_{setup_version}",
        )
        st.caption(t("psm_replacement_notice"))
        use_caliper = st.checkbox(
            t("use_caliper"),
            value=initial_config.get("caliper") is not None,
            help=t("caliper_help"),
            key=f"{key_prefix}_{model_key}_use_caliper_checkbox_{setup_version}",
        )
        if use_caliper:
            caliper = st.number_input(
                t("caliper"),
                min_value=0.0001,
                value=float(initial_config.get("caliper") or 0.05),
                step=0.01,
                format="%.4f",
                key=f"{key_prefix}_{model_key}_caliper_input_{setup_version}",
            )
        else:
            caliper = None
        use_robust = False
        standard_errors = "not_estimated"
        _advanced_model_options(selected_model.model_id, language)
    else:
        main_defaults = initial_config.get("main_independent_variables") or (applied_plan.recommended_main_explanatory_variables if applied_plan else [])
        main_x_default = [col for col in main_defaults if col in x_pool]
        main_x = st.multiselect(
            t("main_independent_variables"),
            x_pool,
            default=main_x_default,
            placeholder=t("select_independent"),
            help=t("main_independent_variables_help"),
            key=f"{key_prefix}_{model_key}_main_independent_variables_select_{setup_version}",
        )
        control_defaults = initial_config.get("numeric_control_variables") or (applied_plan.numeric_controls if applied_plan else [])
        numeric_controls_default = [col for col in control_defaults if col in x_pool and col not in main_x]
        numeric_controls = st.multiselect(
            t("numeric_controls"),
            [col for col in x_pool if col not in main_x],
            default=numeric_controls_default,
            help=t("numeric_controls_help"),
            key=f"{key_prefix}_{model_key}_numeric_controls_select_{setup_version}",
        )

    if selected_model.model_id == "panel_fe":
        st.subheader(t("panel_structure"))
        entity_options = get_entity_id_variables(confirmed_roles) + get_code_identifier_variables(confirmed_roles)
        time_options = get_time_variables(confirmed_roles)
        entity_default = initial_config.get("entity_id") or (applied_plan.entity_id if applied_plan else "")
        entity_index = ([""] + entity_options).index(entity_default) if entity_default in entity_options else 0
        entity_col = st.selectbox(
            t("entity_id"),
            [""] + entity_options,
            index=entity_index,
            format_func=lambda value: "Select entity ID" if value == "" and language == "en" else ("请选择个体 ID" if value == "" else value),
            help=t("selector_help"),
            key=f"{key_prefix}_{model_key}_entity_id_select_{setup_version}",
        )
        time_default = initial_config.get("time_id") or (applied_plan.time_id if applied_plan else "")
        time_index = ([""] + time_options).index(time_default) if time_default in time_options else 0
        time_col = st.selectbox(
            t("time_id"),
            [""] + time_options,
            index=time_index,
            format_func=lambda value: "Select time ID" if value == "" and language == "en" else ("请选择时间变量" if value == "" else value),
            help=t("selector_help"),
            key=f"{key_prefix}_{model_key}_time_id_select_{setup_version}",
        )
        entity_effects = st.checkbox(
            t("entity_fixed_effects"),
            value=bool(initial_config.get("entity_effects", True)),
            key=f"{key_prefix}_{model_key}_entity_effects_checkbox_{setup_version}",
        )
        time_effects = st.checkbox(
            t("time_fixed_effects"),
            value=bool(initial_config.get("time_effects", True)),
            key=f"{key_prefix}_{model_key}_time_effects_checkbox_{setup_version}",
        )
        se_labels = [t("cluster_entity"), t("robust_se").replace("Use HC3 ", "").replace("使用 HC3 ", ""), t("conventional")]
        se_map = {
            t("cluster_entity"): "cluster_entity",
            t("robust_se").replace("Use HC3 ", "").replace("使用 HC3 ", ""): "robust",
            t("conventional"): "conventional",
        }
        reverse_se_map = {value: label for label, value in se_map.items()}
        se_default = reverse_se_map.get(initial_config.get("standard_errors", "cluster_entity"), t("cluster_entity"))
        se_label = st.radio(
            t("standard_error_option"),
            se_labels,
            index=se_labels.index(se_default) if se_default in se_labels else 0,
            horizontal=True,
            key=f"{key_prefix}_{model_key}_se_type_select_{setup_version}",
        )
        standard_errors = se_map.get(se_label, "cluster_entity")
        use_robust = standard_errors != "conventional"
        _advanced_model_options(selected_model.model_id, language)
    elif selected_model.model_id not in {DID_EXPERIMENTAL_MODEL_ID, IV_EXPERIMENTAL_MODEL_ID, PSM_EXPERIMENTAL_MODEL_ID}:
        categorical_defaults = initial_config.get("categorical_control_variables") or (
            applied_plan.categorical_controls if applied_plan and model_key == "ols" else []
        )
        categorical_default = [col for col in categorical_defaults if col in categorical_candidates]
        categorical_controls = st.multiselect(
            t("categorical_controls"),
            categorical_candidates,
            default=categorical_default,
            help=t("categorical_controls_help"),
            key=f"{key_prefix}_{model_key}_categorical_controls_select_{setup_version}",
        )
        encode_categoricals = st.checkbox(
            t("encode_categorical_controls"),
            value=bool(initial_config.get("encode_categorical_controls", False)),
            help=t("dummy_encoding_help"),
            key=f"{key_prefix}_{model_key}_encode_categorical_controls_checkbox_{setup_version}",
        )
        use_robust = _advanced_model_options(selected_model.model_id, language)
        standard_errors = "hc3" if use_robust else "conventional"

    if show_ui_test_defaults and os.environ.get("REG_MONKEY_UI_TEST_MODE") == "1" and st.session_state.get("ui_test_defaults_model") == selected_model.model_id:
        y_col = "pollution_intensity" if selected_model.model_id in {"ols", "panel_fe"} else "export_dummy"
        if selected_model.model_id == DID_EXPERIMENTAL_MODEL_ID:
            y_col = "outcome" if "outcome" in df.columns else y_col
        main_x = ["digital_index"]
        numeric_controls = ["leverage", "rd_intensity"]
        categorical_controls = []
        encode_categoricals = False
        if selected_model.model_id == DID_EXPERIMENTAL_MODEL_ID:
            treatment_col = "treatment" if "treatment" in df.columns else treatment_col
            post_col = "post" if "post" in df.columns else post_col
            numeric_controls = ["control"] if "control" in df.columns else []
        if selected_model.model_id == IV_EXPERIMENTAL_MODEL_ID:
            y_col = "outcome" if "outcome" in df.columns else y_col
            endogenous_col = "endogenous" if "endogenous" in df.columns else endogenous_col
            instrument_cols = ["instrument"] if "instrument" in df.columns else instrument_cols
            exogenous_controls = ["control"] if "control" in df.columns else []
            numeric_controls = list(exogenous_controls)
        if selected_model.model_id == PSM_EXPERIMENTAL_MODEL_ID:
            y_col = "outcome" if "outcome" in df.columns else y_col
            treatment_col = "treatment" if "treatment" in df.columns else treatment_col
            matching_covariates = [
                col
                for col in ["covariate_age", "covariate_size", "covariate_margin"]
                if col in df.columns
            ]
            psm_estimand = "ATT"
            matching_method = "nearest_neighbor"
            caliper = None
        group_col = "firm_id" if selected_model.model_id == DID_EXPERIMENTAL_MODEL_ID else group_col
        cluster_col = "firm_id" if selected_model.model_id == DID_EXPERIMENTAL_MODEL_ID else cluster_col
        entity_col = "firm_id" if selected_model.model_id == "panel_fe" else entity_col
        time_col = "year" if selected_model.model_id == "panel_fe" else time_col
        entity_effects = True if selected_model.model_id == "panel_fe" else entity_effects
        time_effects = True if selected_model.model_id == "panel_fe" else time_effects
        standard_errors = "cluster_entity" if selected_model.model_id == "panel_fe" else standard_errors

    if selected_model.model_id == DID_EXPERIMENTAL_MODEL_ID:
        config = build_did_model_config(
            dependent_variable=y_col,
            treatment_variable=treatment_col,
            post_variable=post_col,
            numeric_control_variables=numeric_controls,
            categorical_control_variables=categorical_controls,
            group_variable=group_col,
            cluster_variable=cluster_col,
            encode_categorical_controls=encode_categoricals,
            variable_roles=confirmed_roles,
        )
    elif selected_model.model_id == IV_EXPERIMENTAL_MODEL_ID:
        config = build_iv_model_config(
            dependent_variable=y_col,
            endogenous_variable=endogenous_col,
            instruments=instrument_cols,
            exogenous_controls=exogenous_controls,
            variable_roles=confirmed_roles,
        )
    elif selected_model.model_id == PSM_EXPERIMENTAL_MODEL_ID:
        config = build_psm_model_config(
            dependent_variable=y_col,
            treatment_variable=treatment_col,
            matching_covariates=matching_covariates,
            psm_estimand=psm_estimand,
            matching_method=matching_method,
            caliper=caliper,
            variable_roles=confirmed_roles,
        )
    else:
        config = {
            "dependent_variable": y_col,
            "main_independent_variables": main_x,
            "numeric_control_variables": numeric_controls,
            "categorical_control_variables": categorical_controls,
            "encode_categorical_controls": encode_categoricals,
            "robust_standard_errors": use_robust,
            "entity_id": entity_col,
            "time_id": time_col,
            "entity_effects": entity_effects,
            "time_effects": time_effects,
            "standard_errors": standard_errors,
            "robust_cov_type": "HC3",
            "include_odds_ratios": True,
            "include_marginal_effects": True,
            "marginal_effects_type": "average",
            "variable_roles": confirmed_roles,
        }
    errors = selected_model.validate(df, config)
    if selected_model.model_id == DID_EXPERIMENTAL_MODEL_ID:
        run_valid = did_required_fields_present(config)
    elif selected_model.model_id == IV_EXPERIMENTAL_MODEL_ID:
        run_valid = iv_required_fields_present(config)
    elif selected_model.model_id == PSM_EXPERIMENTAL_MODEL_ID:
        run_valid = psm_required_fields_present(config)
    else:
        run_valid = bool(y_col and main_x)
    return {
        "model_id": selected_model.model_id,
        "model": selected_model,
        "config": config,
        "run_valid": bool(run_valid and not errors),
        "errors": errors,
    }


def _is_panel_fe_sample_too_small(error: object) -> bool:
    return "too few observations remain after dropping missing values" in str(error).lower()


def _translate_did_validation_error(language: str, error: object) -> str:
    text = str(error)
    lower = text.lower()
    t = lambda key: get_text(language, key)
    if "missing required field" in lower or "references a column" in lower:
        return t("did_error_incomplete")
    if "binary or numeric indicator" in lower:
        return t("did_error_indicator")
    if "insufficient variation" in lower:
        return t("did_error_variation")
    if "not enough observations" in lower or "no complete observations" in lower:
        return t("did_error_sample")
    if "cluster" in lower:
        return t("did_error_cluster")
    return text


def _translate_iv_validation_error(language: str, error: object) -> str:
    text = str(error)
    lower = text.lower()
    t = lambda key: get_text(language, key)
    if "missing required field" in lower or "requires at least one instrument" in lower or "references a column" in lower:
        return t("iv_error_incomplete")
    if "cannot also be used" in lower or "must not duplicate" in lower or "distinct outcome" in lower:
        return t("iv_error_role_conflict")
    if "numeric variables" in lower or "non-numeric" in lower:
        return t("iv_error_numeric")
    if "not enough observations" in lower or "no complete observations" in lower:
        return t("iv_error_sample")
    if "collinear" in lower or "rank deficient" in lower:
        return t("iv_error_collinearity")
    if "no usable variation" in lower:
        return t("iv_error_variation")
    return text


def _translate_psm_validation_error(language: str, error: object) -> str:
    text = str(error)
    lower = text.lower()
    t = lambda key: get_text(language, key)
    if "missing required field" in lower or "matching covariate" in lower:
        return t("psm_error_incomplete")
    if "cannot also" in lower or "must not duplicate" in lower or "duplicate" in lower:
        return t("psm_error_role_conflict")
    if "binary" in lower:
        return t("psm_error_treatment")
    if "numeric" in lower or "non-numeric" in lower:
        return t("psm_error_numeric")
    if "no complete observations" in lower or "not enough observations" in lower:
        return t("psm_error_sample")
    if "caliper" in lower:
        return t("psm_error_caliper")
    if "control observation" in lower or "valid control matches" in lower:
        return t("psm_error_controls")
    if "treated observation" in lower:
        return t("psm_error_treated")
    if "collinear" in lower or "rank deficient" in lower:
        return t("psm_error_collinearity")
    return text


def _render_model_error(language: str, model_id: str, error: object) -> None:
    t = lambda key: get_text(language, key)
    if model_id == "panel_fe" and _is_panel_fe_sample_too_small(error):
        st.error(t("panel_fe_sample_too_small"))
        with st.expander(t("technical_error_details"), expanded=False):
            st.code(str(error))
    elif model_id == DID_EXPERIMENTAL_MODEL_ID:
        st.error(_translate_did_validation_error(language, error))
        with st.expander(t("technical_error_details"), expanded=False):
            st.code(str(error))
    elif model_id == IV_EXPERIMENTAL_MODEL_ID:
        st.error(_translate_iv_validation_error(language, error))
        with st.expander(t("technical_error_details"), expanded=False):
            st.code(str(error))
    elif model_id == PSM_EXPERIMENTAL_MODEL_ID:
        st.error(_translate_psm_validation_error(language, error))
        with st.expander(t("technical_error_details"), expanded=False):
            st.code(str(error))
    else:
        st.error(str(error))


def _render_model_run_failure(language: str, model_id: str, run_result: object) -> None:
    diagnostics = getattr(run_result, "structured_diagnostics", []) or []
    errors = list(getattr(run_result, "errors", []) or [])
    rendered = False
    for item in diagnostic_dicts(diagnostics, ui_only=True):
        if item.get("severity") != "error":
            continue
        code = str(item.get("code") or "")
        message = translate_diagnostic_field(language, code, "message", str(item.get("message") or ""))
        recommendation = translate_diagnostic_field(language, code, "recommendation", str(item.get("recommendation") or ""))
        st.error(message)
        if recommendation:
            st.caption(recommendation)
        rendered = True
    if not rendered:
        for error in errors:
            _render_model_error(language, model_id, error)
    if errors:
        with st.expander(get_text(language, "technical_error_details"), expanded=False):
            st.code("\n".join(str(error) for error in errors))


def _execute_model_setup(
    df: pd.DataFrame,
    profile: dict,
    confirmed_roles: dict[str, str],
    language: str,
    selected_model: object,
    config: dict,
) -> bool:
    t = lambda key: get_text(language, key)
    _reset_analysis()
    config = dict(config)
    config["variable_roles"] = confirmed_roles
    model_spec = ExecutableModelSpec.from_config(selected_model.model_id, config)
    run_result = run_model_spec(df, model_spec)
    if not run_result.success:
        _record_run_history(run_result, selected_model.model_id)
        _render_model_run_failure(language, selected_model.model_id, run_result)
        return False

    fit_payload = run_result.to_legacy_payload()
    data_quality_profile = data_quality_to_jsonable(build_data_quality_profile(df))
    missingness_profile = data_quality_to_jsonable(build_missingness_profile(df))
    variable_quality_summary = data_quality_to_jsonable(build_variable_quality_summaries(df))
    resource_warning_profile = data_quality_to_jsonable(
        build_resource_warning_profile(df, public_demo_mode=is_public_demo_mode())
    )
    pre_model_risk_profile = data_quality_to_jsonable(build_pre_model_risk_profile(df, model_spec))
    fit_payload["data_quality_profile"] = data_quality_profile
    fit_payload["missingness_profile"] = missingness_profile
    fit_payload["variable_quality_summary"] = variable_quality_summary
    fit_payload["resource_warning_profile"] = resource_warning_profile
    fit_payload["pre_model_risk_profile"] = pre_model_risk_profile
    if st.session_state.get("missing_data_plan"):
        fit_payload["missing_data_plan"] = st.session_state.get("missing_data_plan")
    if st.session_state.get("missing_data_handling_result"):
        fit_payload["missing_data_handling_result"] = st.session_state.get("missing_data_handling_result")
    diagnostics = run_result.diagnostics
    warnings = diagnostics.get("warnings", [])
    fit_metric = fit_payload["model_summary"].get(
        "r_squared",
        fit_payload["model_summary"].get(
            "pseudo_r_squared",
            fit_payload["model_summary"].get("r_squared_within"),
        ),
    )
    if fit_metric is not None and fit_metric < 0.05:
        warnings.append(t("low_r2_warning"))

    simple_report = generate_simple_report(
        fit_payload["regression_table"],
        fit_payload["model_summary"],
        warnings,
        language=normalize_language(language),
        model_metadata=selected_model,
        model_results=fit_payload,
        profile=profile,
        variable_roles=confirmed_roles,
        guided_workflow_result=st.session_state.guided_workflow_result,
    )
    full_report = generate_markdown_report(
        profile,
        fit_payload["cleaning_log"],
        fit_payload["regression_table"],
        fit_payload["model_summary"],
        diagnostics.get("vif_df", pd.DataFrame()),
        warnings,
        language=normalize_language(language),
        model_metadata=selected_model,
        model_results=fit_payload,
        variable_roles=confirmed_roles,
        guided_workflow_result=st.session_state.guided_workflow_result,
    )

    st.session_state.analysis_ran = True
    st.session_state.regression_table = fit_payload["regression_table"]
    st.session_state.model_summary = fit_payload["model_summary"]
    st.session_state.cleaned_df = fit_payload["cleaned_df"]
    st.session_state.cleaning_log = fit_payload["cleaning_log"]
    st.session_state.vif_df = diagnostics.get("vif_df", pd.DataFrame())
    st.session_state.warnings = warnings
    st.session_state.structured_diagnostics = fit_payload.get("structured_diagnostics", [])
    st.session_state.model_results = fit_payload
    st.session_state.model_run_result = run_result
    st.session_state.simple_report = simple_report
    st.session_state.full_report = full_report
    _record_run_history(run_result, selected_model.model_id)
    return True


def _show_plot(plot_result: tuple[object, str | None]) -> None:
    fig, message = plot_result
    if message:
        st.info(message)
    elif fig is not None:
        st.pyplot(fig, clear_figure=True)


def _render_analysis_planner_tab(
    df: pd.DataFrame,
    profile: dict,
    confirmed_roles: dict[str, str],
    language: str,
) -> None:
    t = lambda key: get_text(language, key)
    render_page_header(t("tab_plan_title"), t("guide_plan"))
    if not st.session_state.confirm_variable_roles:
        _action_card(t("planner_confirm_roles_first"), t("planner_confirm_roles_subtext"), t("tab_setup"), "setup", "planner_go_setup_button")
        return

    if st.button(_down_action(t("generate_analysis_plan")), key="planner_generate_plan_button", type="secondary", width="stretch"):
        st.session_state.analysis_plan = generate_analysis_plan(df, confirmed_roles, language=normalize_language(language))
        st.session_state.analysis_plan_generated = True
        st.session_state.analysis_plan_applied = False
        _request_scroll_anchor("analysis_plan_anchor")
        st.rerun()

    if (
        not st.session_state.analysis_plan_generated
        or st.session_state.analysis_plan is None
        or st.session_state.analysis_plan.language != normalize_language(language)
    ):
        st.session_state.analysis_plan = generate_analysis_plan(df, confirmed_roles, language=normalize_language(language))
        st.session_state.analysis_plan_generated = True

    plan: AnalysisPlan = st.session_state.analysis_plan
    can_apply = plan.recommended_main_model is not None and bool(plan.recommended_dependent_variable) and bool(plan.recommended_main_explanatory_variables)

    main_model_name = plan.recommended_main_model.model_name if plan.recommended_main_model else t("not_available")

    _scroll_anchor("analysis_plan_anchor")
    render_section_header(t("plan_section_title"), t("plan_section_body"))
    if st.session_state.plan_model_path == "recommended":
        st.session_state.plan_manual_requested_model_id = ""

    active_model_id, active_config = _active_model_setup()
    requested_model_id = st.session_state.get("plan_manual_requested_model_id", "")
    manual_status_model_id = active_model_id if st.session_state.model_setup_source == "manual" else requested_model_id
    manual_status_label = _model_label(manual_status_model_id, language) if manual_status_model_id else t("status_pending")
    manual_status_body = t("manual_workspace_status_ready") if st.session_state.plan_model_path == "manual" else t("manual_workspace_status_available")

    plan_cols = st.columns([1.05, 1], gap="large")
    with plan_cols[0]:
        with st.container(border=True):
            render_selection_card(
                t("use_recommended_plan"),
                t("recommended_plan_card_body"),
                selected=st.session_state.plan_model_path == "recommended",
                badge=(t("active_path") if st.session_state.plan_model_path == "recommended" else t("inactive_path"), "success" if st.session_state.plan_model_path == "recommended" else "neutral"),
                metadata={
                    t("planner_main_model"): main_model_name,
                    t("configuration_status"): t("recommended_path_status"),
                },
            )
            if st.session_state.plan_model_path == "recommended":
                st.caption(t("plan_path_current"))
            elif st.button(_down_action(t("choose_recommended_path")), key="plan_choose_recommended_path_button", type="secondary", width="stretch"):
                st.session_state.plan_model_path = "recommended"
                st.session_state.plan_manual_requested_model_id = ""
                _request_scroll_anchor("analysis_plan_anchor")
                st.rerun()
            _model_summary_card(
                language,
                t("recommended_setup_summary"),
                [
                    (t("planner_main_model"), main_model_name),
                    (t("dependent_variable"), plan.recommended_dependent_variable or t("not_available")),
                    (t("main_independent_variables"), " · ".join(plan.recommended_main_explanatory_variables) or t("not_available")),
                    (t("numeric_controls"), " · ".join(plan.numeric_controls) if plan.numeric_controls else t("not_available")),
                    (t("panel_structure"), " · ".join([value for value in [plan.entity_id, plan.time_id] if value]) or t("not_available")),
                ],
            )
            with st.expander(t("planner_rationale"), expanded=False):
                for item in plan.rationale:
                    st.markdown(f"- {item}")
            if st.button(_next_action(t("plan_continue_run")), type="primary", disabled=not can_apply, key="planner_apply_and_continue_run_button", width="stretch"):
                _apply_analysis_plan_to_state(plan, language)
                _go_to_page("run", "run_primary_action_anchor")
            if not can_apply:
                st.caption(t("planner_apply_disabled"))
            elif st.session_state.analysis_plan_applied:
                st.success(f"{t('planner_applied_model')}: {_model_label(st.session_state.applied_plan_model_id, language)}")
    with plan_cols[1]:
        with st.container(border=True):
            render_selection_card(
                t("configure_manually"),
                t("manual_configuration_body"),
                selected=st.session_state.plan_model_path == "manual",
                badge=(t("active_path") if st.session_state.plan_model_path == "manual" else t("inactive_path"), "info" if st.session_state.plan_model_path == "manual" else "neutral"),
                metadata={
                    t("selected_model_family"): manual_status_label,
                    t("configuration_status"): manual_status_body,
                    t("available_model_families"): f"{t('basic_model_family')} / {t('experimental_manual_model_family')}",
                },
            )
            if st.session_state.plan_model_path == "manual":
                st.caption(t("plan_path_current"))
            elif st.button(_down_action(t("choose_manual_path")), type="secondary", key="plan_choose_manual_path_button", width="stretch"):
                st.session_state.plan_model_path = "manual"
                _request_scroll_anchor("manual_config_anchor")
                st.rerun()

    def _open_manual_candidate(model_id: str) -> None:
        st.session_state.plan_model_path = "manual"
        st.session_state.plan_manual_requested_model_id = model_id
        st.session_state.model_setup_version += 1
        _request_scroll_anchor("manual_config_anchor")

    with st.container(border=True):
        _render_research_design_candidate_preview(
            df,
            confirmed_roles,
            language,
            on_go_manual=_open_manual_candidate,
        )

    render_section_header(t("configure_section_title"), t("configure_section_body"))
    if st.session_state.plan_model_path == "recommended":
        render_callout(t("manual_configuration_title"), t("configure_section_recommended_hint"), tone="info")
    else:
        _scroll_anchor("manual_config_anchor")
        initial_manual_model_id = active_model_id if st.session_state.model_setup_source == "manual" else requested_model_id
        initial_manual_config = active_config if st.session_state.model_setup_source == "manual" else {}
        with st.container(border=True):
            render_section_header(
                t("manual_configuration_workspace"),
                t("manual_configuration_body"),
                badge=(t("active_path"), "info"),
            )
            model_family_cols = st.columns(2, gap="large")
            with model_family_cols[0]:
                render_status_card(
                    t("basic_model_family"),
                    ", ".join(model.display_name(normalize_language(language)) for model in get_available_models()),
                    status="info",
                )
            with model_family_cols[1]:
                render_status_card(
                    t("experimental_manual_model_family"),
                    " · ".join([did_model_label(language), iv_model_label(language), psm_model_label(language)]),
                    status="warning",
                )
            manual_result = _render_model_setup_form(
                df,
                confirmed_roles,
                language,
                key_prefix="plan_manual",
                initial_model_id=initial_manual_model_id,
                initial_config=initial_manual_config,
                show_ui_test_defaults=True,
            )
            if manual_result["model_id"]:
                _model_summary_card(
                    language,
                    _manual_setup_summary_title(manual_result["model_id"], language),
                    [
                        (t("model_category"), _model_category_label(manual_result["model_id"], language)),
                    ]
                    + _model_setup_rows(manual_result["model_id"], manual_result["config"], language),
                )
            has_attempted_manual_spec = bool(
                manual_result["config"].get("dependent_variable")
                and (
                    manual_result["config"].get("main_independent_variables")
                    or manual_result["model_id"] == DID_EXPERIMENTAL_MODEL_ID
                    or manual_result["model_id"] == IV_EXPERIMENTAL_MODEL_ID
                    or manual_result["model_id"] == PSM_EXPERIMENTAL_MODEL_ID
                )
            )
            if manual_result["errors"] and has_attempted_manual_spec:
                for error in manual_result["errors"]:
                    _render_model_error(language, manual_result["model_id"], error)
            manual_valid = bool(manual_result["model_id"] and manual_result["run_valid"])
            render_section_header(t("manual_workspace_actions"), None)
            manual_col_1, manual_col_2 = st.columns(2)
            if manual_col_1.button(_down_action(t("use_manual_settings")), type="secondary", disabled=not manual_valid, key="plan_use_manual_settings_button"):
                _store_model_setup(manual_result["model_id"], manual_result["config"], "manual")
                st.session_state.analysis_plan_applied = False
                st.session_state.applied_plan_model_id = ""
                st.session_state.model_setup_version += 1
                _request_scroll_anchor("manual_config_anchor")
                st.success(t("manual_settings_applied"))
                st.rerun()
            if manual_col_2.button(_next_action(t("use_manual_settings_continue")), type="primary", disabled=not manual_valid, key="plan_use_manual_settings_continue_button"):
                _store_model_setup(manual_result["model_id"], manual_result["config"], "manual")
                st.session_state.analysis_plan_applied = False
                st.session_state.applied_plan_model_id = ""
                st.session_state.model_setup_version += 1
                _go_to_page("run", "run_primary_action_anchor")

    current_model_id, current_config = _active_model_setup()
    with st.expander(t("current_selected_setup"), expanded=False):
        if current_model_id:
            _model_summary_card(language, t("model_summary"), [(t("setup_source"), _setup_source_label(language, st.session_state.model_setup_source))] + _model_setup_rows(current_model_id, current_config, language))
            if st.button(_next_action(t("continue_to_run_analysis")), type="primary", key="plan_current_continue_run_button"):
                _go_to_page("run", "run_primary_action_anchor")
        else:
            st.caption(t("no_model_setup_selected"))

    with st.expander(t("planner_details"), expanded=False):
        st.subheader(t("planner_data_structure"))
        panel_cols = st.columns(5)
        panel_cols[0].metric(t("planner_structure_type"), t(f"planner_structure_{plan.data_structure}"))
        panel_cols[1].metric(t("entity_id"), plan.entity_id or t("not_available"))
        panel_cols[2].metric(t("time_id"), plan.time_id or t("not_available"))
        panel_cols[3].metric(t("entities"), _planner_panel_value(df, plan.entity_id))
        panel_cols[4].metric(t("time_periods"), _planner_panel_value(df, plan.time_id))
        st.info(plan.summary)

        st.subheader(t("planner_recommended_variables"))
        v1, v2 = st.columns(2)
        with v1:
            st.markdown(f"**{t('dependent_variable')}:** {plan.recommended_dependent_variable or t('not_available')}")
            st.markdown(
                f"**{t('main_independent_variables')}:** "
                f"{', '.join(plan.recommended_main_explanatory_variables) if plan.recommended_main_explanatory_variables else t('not_available')}"
            )
        with v2:
            st.markdown(f"**{t('numeric_controls')}:** {', '.join(plan.numeric_controls) if plan.numeric_controls else t('not_available')}")
            st.markdown(f"**{t('categorical_controls')}:** {', '.join(plan.categorical_controls) if plan.categorical_controls else t('not_available')}")
        st.dataframe(_planner_candidate_frame(plan.dependent_variable_candidates), width="stretch")
        st.dataframe(_planner_candidate_frame(plan.main_explanatory_candidates), width="stretch")

        st.subheader(t("planner_recommended_path"))
        st.markdown(f"**{t('planner_main_model')}:** {main_model_name}")
        st.markdown(
            f"**{t('planner_baseline_models')}:** "
            f"{', '.join(model.model_name for model in plan.baseline_models) if plan.baseline_models else t('not_available')}"
        )
        st.markdown(
            f"**{t('planner_alternative_models')}:** "
            f"{', '.join(model.model_name for model in plan.alternative_models) if plan.alternative_models else t('not_available')}"
        )

        st.subheader(t("planner_warnings"))
        warning_frame = _planner_warning_frame(plan)
        if warning_frame.empty:
            st.success(t("no_warnings"))
        else:
            for warning in plan.warnings:
                st.warning(warning.message)
            st.dataframe(warning_frame, width="stretch")

        st.subheader(t("planner_next_actions"))
        for action in plan.user_confirmable_actions:
            st.caption(f"- {action}")


def _render_guided_workflow_block(
    df: pd.DataFrame,
    profile: dict,
    confirmed_roles: dict[str, str],
    language: str,
) -> None:
    t = lambda key: get_text(language, key)
    st.subheader(t("guided_workflow"))
    if not st.session_state.analysis_plan_generated or st.session_state.analysis_plan is None:
        st.info(t("guided_workflow_needs_plan"))
        return
    plan: AnalysisPlan = st.session_state.analysis_plan
    if plan.recommended_main_model is None:
        st.info(t("guided_workflow_needs_plan"))
        return
    st.caption(t("guided_workflow_caption"))
    _model_summary_card(
        language,
        t("guided_workflow_steps"),
        [
            (t("guided_step_baseline"), ", ".join(model.model_name for model in plan.baseline_models) or t("not_available")),
            (t("guided_step_main"), plan.recommended_main_model.model_name),
            (t("guided_step_compare"), t("result_comparison")),
        ],
    )
    preview_rows = [
        {t("table_variable"): t("planner_baseline_models"), "value": ", ".join(model.model_name for model in plan.baseline_models) or t("not_available")},
        {t("table_variable"): t("planner_main_model"), "value": plan.recommended_main_model.model_name},
        {t("table_variable"): t("dependent_variable"), "value": plan.recommended_dependent_variable or t("not_available")},
        {t("table_variable"): t("main_independent_variables"), "value": ", ".join(plan.recommended_main_explanatory_variables) or t("not_available")},
        {t("table_variable"): t("numeric_controls"), "value": ", ".join(plan.numeric_controls) or t("not_available")},
        {t("table_variable"): t("entity_id"), "value": plan.entity_id or t("not_available")},
        {t("table_variable"): t("time_id"), "value": plan.time_id or t("not_available")},
    ]
    st.dataframe(pd.DataFrame(preview_rows), width="stretch")

    workflow_result = st.session_state.guided_workflow_result
    if st.session_state.guided_workflow_has_run and workflow_result is not None:
        _scroll_anchor("guided_workflow_result_anchor")
        st.subheader(t("workflow_summary"))
        st.markdown(workflow_summary_markdown(workflow_result, language, detailed=False))
        st.subheader(t("result_comparison"))
        st.dataframe(workflow_result_frame(workflow_result, language), width="stretch")
        if workflow_result.warnings:
            for warning in workflow_result.warnings:
                st.warning(translate_warning(language, warning))
        with st.expander(t("view_detailed_diagnostics"), expanded=False):
            if workflow_result.comparison:
                st.dataframe(prepare_display_table(pd.DataFrame(workflow_result.comparison.coefficient_comparison), language), width="stretch")
            st.write(workflow_result.user_next_steps)

    if st.button(_down_action(t("run_guided_analysis")), type="secondary", key="planner_run_guided_workflow_button", width="stretch"):
        workflow_result = run_guided_workflow(df, confirmed_roles, plan, language=normalize_language(language))
        st.session_state.guided_workflow_result = workflow_result
        st.session_state.guided_workflow_has_run = True
        st.session_state.guided_workflow_status = workflow_result.status
        if st.session_state.analysis_ran and st.session_state.model_summary is not None:
            current_model_id = str(st.session_state.model_summary.get("model_type") or "ols").lower()
            current_model = _model_definition(current_model_id)
            st.session_state.simple_report = generate_simple_report(
                st.session_state.regression_table,
                st.session_state.model_summary,
                st.session_state.warnings,
                language=normalize_language(language),
                model_metadata=current_model,
                model_results=st.session_state.model_results,
                profile=profile,
                variable_roles=confirmed_roles,
                guided_workflow_result=workflow_result,
            )
            st.session_state.full_report = generate_markdown_report(
                profile,
                st.session_state.cleaning_log,
                st.session_state.regression_table,
                st.session_state.model_summary,
                st.session_state.vif_df,
                st.session_state.warnings,
                language=normalize_language(language),
                model_metadata=current_model,
                model_results=st.session_state.model_results,
                variable_roles=confirmed_roles,
                guided_workflow_result=workflow_result,
            )
        _request_scroll_anchor("guided_workflow_result_anchor")
        st.rerun()


def _planner_panel_value(df: pd.DataFrame, column: str | None) -> int | str:
    if not column or column not in df.columns:
        return "-"
    return int(df[column].nunique(dropna=True))


def _preprocessing_log_frame(raw_df: pd.DataFrame, df: pd.DataFrame, log: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "use_first_row_as_variable_names": log.get("use_first_row_as_column_names", True),
                "manual_skip_rows": log["requested_skip_rows"],
                "auto_detected_skip_rows": log["detected_skip_rows"],
                "applied_skip_rows": log["rows_skipped"],
                "rows_before": log.get("rows_before_preprocessing", len(raw_df)),
                "rows_after": len(df),
                "columns_before": log.get("columns_before_preprocessing", raw_df.shape[1]),
                "columns_after": df.shape[1],
                "temporary_variable_names": ", ".join(log.get("generated_temporary_variable_names", [])[:5]),
            }
        ]
    )


def _variable_reference(df: pd.DataFrame, profile: dict, language: str, query: str) -> pd.DataFrame:
    t = lambda key: get_text(language, key)
    reference = build_variable_reference(df, profile, query)
    return reference.rename(
        columns={
            "Variable": t("table_variable"),
            "Type": t("table_type"),
            "Missing %": t("table_missing"),
            "Examples": t("table_examples"),
            "Direct numeric use": t("table_direct_numeric_use"),
            "Available as categorical control": t("table_categorical_control_use"),
            "Required handling / notes": t("table_required_handling"),
        }
    )


def _role_table_for_editor(df: pd.DataFrame, profile: dict, roles: dict[str, str], language: str, editable: bool = True) -> pd.DataFrame:
    table = build_variable_role_table(df, profile, roles).copy()
    table["recommended_role"] = table["recommended_role"].apply(lambda role: role_label(role, normalize_language(language)))
    table["confirmed_role"] = table["confirmed_role"].apply(lambda role: role_label(role, normalize_language(language)))
    confirmed_label = get_text(language, "click_to_choose_role") if editable else get_text(language, "confirmed_role")
    return table.rename(
        columns={
            "variable": get_text(language, "table_variable"),
            "inferred_type": get_text(language, "inferred_type"),
            "missing_percentage": get_text(language, "table_missing"),
            "unique_values": get_text(language, "unique_values"),
            "examples": get_text(language, "table_examples"),
            "recommended_role": get_text(language, "recommended_role"),
            "confirmed_role": confirmed_label,
        }
    )


def _roles_from_editor(editor_df: pd.DataFrame, language: str) -> dict[str, str]:
    variable_col = get_text(language, "table_variable")
    role_col = get_text(language, "click_to_choose_role")
    return {
        row[variable_col]: role_from_label(row[role_col], normalize_language(language))
        for _, row in editor_df.iterrows()
    }


def _role_group_frame(roles: dict[str, str], language: str) -> pd.DataFrame:
    rows = [
        (get_text(language, "role_group_numeric"), get_numeric_measure_variables(roles)),
        (get_text(language, "role_group_binary"), get_binary_variables(roles)),
        (get_text(language, "role_group_categorical"), get_categorical_variables(roles)),
        (get_text(language, "role_group_code"), get_code_identifier_variables(roles)),
        (get_text(language, "role_group_time"), get_time_variables(roles)),
        (get_text(language, "role_group_entity"), get_entity_id_variables(roles)),
        (get_text(language, "role_group_exclude"), [col for col, role in roles.items() if role == ROLE_EXCLUDE]),
    ]
    return pd.DataFrame(
        {
            "Role": [label for label, _ in rows],
            "Count": [len(values) for _, values in rows],
            "Variables": [", ".join(values) if values else "" for _, values in rows],
        }
    )


def _panel_structure_frame(panel_info: dict, language: str) -> pd.DataFrame:
    keys = [
        "observations",
        "entities",
        "time_periods",
        "balanced_panel",
        "duplicate_entity_time_rows",
        "missing_entity_ids",
        "missing_time_ids",
        "singleton_entities",
        "min_observations_per_entity",
        "max_observations_per_entity",
        "average_observations_per_entity",
    ]
    return pd.DataFrame(
        {
            get_text(language, "table_variable"): [get_text(language, key) for key in keys],
            "value": [str(panel_info.get(key, "")) for key in keys],
        }
    )


def _model_fit_label(r_squared: float, language: str) -> str:
    t = lambda key: get_text(language, key)
    if r_squared < 0.05:
        return t("fit_weak")
    if r_squared < 0.30:
        return t("fit_moderate")
    return t("fit_strong")


def _fit_value_for_label(model_summary: dict) -> float:
    if "r_squared" in model_summary:
        return float(model_summary.get("r_squared") or 0)
    if "r_squared_within" in model_summary:
        return float(model_summary.get("r_squared_within") or 0)
    return float(model_summary.get("pseudo_r_squared") or 0)


def _main_effect_summary(regression_table: pd.DataFrame, model_summary: dict, language: str) -> str:
    return _ui_main_effect_display(regression_table, model_summary, language)


def _fit_metric_summary(model_summary: dict, language: str) -> str:
    return _ui_fit_metric_summary(model_summary, language)


def _key_findings(regression_table: pd.DataFrame, model_summary: dict, warnings: list[str], language: str) -> None:
    t = lambda key: get_text(language, key)
    model_type = str(model_summary.get("model_type") or "ols").lower()
    if model_type == "psm":
        for line in _psm_key_findings(model_summary, warnings, language):
            st.write(f"- {line}")
        st.caption(t("psm_interpretation_caution"))
        return
    st.write(_model_fit_label(_fit_value_for_label(model_summary), language))
    main_vars = list(model_summary.get("main_independent_variables") or [])
    main_table = regression_table[regression_table["variable"].isin(main_vars)]
    significant = main_table[main_table["p_value"] < 0.10] if not main_table.empty else pd.DataFrame()
    if main_table.empty:
        st.write("No main explanatory variables are available in the coefficient table." if language == "en" else "系数表中没有可展示的核心解释变量。")
    elif significant.empty:
        st.write("No main explanatory variables are statistically significant at the 10% level." if language == "en" else "在 10% 显著性水平下，没有核心解释变量达到统计显著。")
    else:
        for _, row in significant.iterrows():
            direction = "positive" if row["coefficient"] > 0 else "negative"
            if model_type == "panel_fe":
                if language == "zh":
                    direction_text = "正相关" if row["coefficient"] > 0 else "负相关"
                    st.write(
                        f"- 在控制固定效应后，{row['variable']} 与同一个体内部随时间变化的 {model_summary.get('dependent_variable', 'Y')} 呈{direction_text}，{format_p_narrative(row['p_value'], language)}。"
                    )
                else:
                    direction_text = "positively" if row["coefficient"] > 0 else "negatively"
                    st.write(
                        f"- After controlling for fixed effects, {row['variable']} is {direction_text} associated with within-entity changes over time in {model_summary.get('dependent_variable', 'Y')}, {format_p_narrative(row['p_value'], language)}."
                    )
            elif language == "zh":
                direction = "正向" if row["coefficient"] > 0 else "负向"
                st.write(f"- {row['variable']}: {direction}相关，{format_p_narrative(row['p_value'], language)}")
            else:
                st.write(f"- {row['variable']}: {direction} association, {format_p_narrative(row['p_value'], language)}")
    controls = set(model_summary.get("numeric_control_variables") or [])
    control_table = regression_table[regression_table["variable"].isin(controls)]
    if not control_table.empty and (control_table["p_value"] < 0.10).any():
        st.caption("Some controls are statistically significant." if language == "en" else "部分控制变量在统计上显著。")
    if model_type == "panel_fe":
        st.caption(t("panel_fe_interpretation_note"))
    elif model_type == "logit":
        st.caption(
            "Logit coefficients are log-odds units, not direct percentage-point probability changes."
            if language == "en"
            else "Logit 系数是对数胜算单位，不能直接解释为概率变化的百分点。"
        )
    elif model_type == "probit":
        st.caption(
            "Probit coefficients are latent-index units, not direct percentage-point probability changes."
            if language == "en"
            else "Probit 系数是潜变量指数单位，不能直接解释为概率变化的百分点。"
        )
    if warnings:
        _callout(translate_warning(language, warnings[0]))


def _render_result_reading_path(
    regression_table: pd.DataFrame,
    model_summary: dict,
    display_warnings: list[str],
    language: str,
) -> None:
    t = lambda key: get_text(language, key)
    diagnostics_status = t("diagnostic_attention") if display_warnings else t("diagnostic_clear")
    diagnostics_tone = "warning" if display_warnings else "success"
    with st.container(border=True):
        render_section_header(t("result_reading_path"), t("result_reading_path_caption"))
        path_columns = st.columns(3)
        with path_columns[0]:
            render_status_card(
                t("result_reading_main_estimate"),
                _main_effect_summary(regression_table, model_summary, language),
                status="success",
            )
        with path_columns[1]:
            render_status_card(
                t("result_reading_trust_checks"),
                diagnostics_status,
                status=diagnostics_tone,
            )
        with path_columns[2]:
            render_status_card(
                t("result_reading_details"),
                t("result_reading_details_body"),
                status="info",
            )


def _render_explanation_controls(language: str) -> str:
    t = lambda key: get_text(language, key)
    mode_options = _explanation_mode_options(language)
    current_mode = st.session_state.get("explanation_mode", LLMExplanationMode.RULE_BASED.value)
    labels = list(mode_options.keys())
    current_label = next(
        (label for label, value in mode_options.items() if value == current_mode),
        labels[0],
    )
    with st.expander(t("explanation_mode"), expanded=False):
        st.caption(t("explanation_mode_help"))
        selected_label = st.radio(
            t("explanation_mode"),
            labels,
            index=labels.index(current_label),
            help=t("explanation_mode_help"),
            horizontal=True,
            key="interpret_explanation_mode_radio",
            label_visibility="collapsed",
        )
    selected_mode = mode_options[selected_label]
    st.session_state.explanation_mode = selected_mode
    st.caption(_explanation_layer_label(language, selected_mode))
    return selected_mode


def _preview_model_dataframe(model_id: str, df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, str, list[str], dict, dict]:
    if model_id == "ols":
        return prepare_ols_dataframe(df, config)
    if model_id == DID_EXPERIMENTAL_MODEL_ID:
        model_df, y_col, x_cols, cleaning_log, encoding_info, cell_counts = prepare_did_dataframe(df, config)
        encoding_info = dict(encoding_info)
        encoding_info["did_cell_counts"] = cell_counts
        return model_df, y_col, x_cols, cleaning_log, encoding_info
    if model_id == IV_EXPERIMENTAL_MODEL_ID:
        model_df, y_col, endogenous_col, instruments, controls, cleaning_log = prepare_iv_dataframe(df, config)
        return model_df, y_col, [endogenous_col, *instruments, *controls], cleaning_log, {
            "reference_categories": {},
            "dummy_variables": [],
        }
    if model_id == PSM_EXPERIMENTAL_MODEL_ID:
        model_df, y_col, treatment_col, covariates, cleaning_log = prepare_psm_dataframe(df, config)
        return model_df, y_col, [treatment_col, *covariates], cleaning_log, {
            "reference_categories": {},
            "dummy_variables": [],
        }
    if model_id == "panel_fe":
        model_df, y_col, x_cols, entity_col, time_col, cleaning_log = prepare_panel_fe_dataframe(df, config)
        panel_info = check_panel_structure(model_df, entity_col, time_col)
        return model_df, y_col, x_cols, cleaning_log, {
            "panel_structure": panel_info,
            "reference_categories": {},
            "dummy_variables": [],
        }
    return prepare_binary_choice_dataframe(df, config)


def _advanced_model_options(model_id: str, language: str) -> bool:
    t = lambda key: get_text(language, key)
    use_robust = True
    with st.expander(t("advanced_model_options"), expanded=False):
        st.caption(t("advanced_model_options_help"))
        if model_id == "ols":
            use_robust = st.checkbox(t("robust_se"), value=True, help=t("robust_se_help"), key=f"{model_id}_robust_se_checkbox")
            st.caption(t("heteroskedasticity_diagnostics"))
        elif model_id == "logit":
            st.caption(t("advanced_outputs_computed"))
            st.caption(t("classification_diagnostics_not_generated"))
        elif model_id == "probit":
            st.caption(t("advanced_outputs_computed"))
            st.caption(t("classification_diagnostics_not_generated"))
        elif model_id == "panel_fe":
            st.caption(t("panel_current_outputs_note"))
        elif model_id == DID_EXPERIMENTAL_MODEL_ID:
            st.caption(t("did_assumption_caution"))
        elif model_id == IV_EXPERIMENTAL_MODEL_ID:
            st.caption(t("iv_assumption_caution"))
        elif model_id == PSM_EXPERIMENTAL_MODEL_ID:
            st.caption(t("psm_assumption_caution"))
        else:
            st.caption(t("select_model_placeholder"))
    return use_robust


def _render_demo_workflow_guide(guide: DemoWorkflowGuide, language: str, *, expanded: bool = False) -> None:
    t = lambda key: get_text(language, key)
    with st.expander(t("sample_workflow_guide"), expanded=expanded):
        st.markdown(f"**{guide.sample_title}**")
        st.caption(f"{t('sample_workflow_goal')}: {guide.suitable_user_goal}")
        st.markdown(f"**{t('sample_workflow_expected_model')}**")
        st.write(guide.expected_model_family)
        st.markdown(f"**{t('sample_workflow_steps')}**")
        for step in guide.steps:
            st.markdown(f"- {step.text}")
        st.markdown(f"**{t('sample_workflow_result_focus')}**")
        for item in guide.expected_result_focus:
            st.markdown(f"- {item}")
        st.markdown(f"**{t('sample_workflow_cautions')}**")
        for caution in guide.interpretation_cautions:
            st.markdown(f"- {caution.text}")
        st.markdown(f"**{t('sample_workflow_not_conclude')}**")
        for item in guide.what_not_to_conclude:
            st.markdown(f"- {item}")


def _render_first_run_intro(language: str, *, public_demo_mode: bool = False) -> None:
    t = lambda key: get_text(language, key)
    render_section_header(t("first_run_intro_title"), t("first_run_intro_body"))
    chips = [
        t("first_run_intro_sample_title"),
        t("first_run_intro_upload_title"),
        t("first_run_intro_flow_understand"),
        t("first_run_intro_flow_export"),
    ]
    st.markdown(
        "<div class='setup-flow-chips'>"
        + "".join(f"<span class='setup-flow-chip'>{html.escape(chip)}</span>" for chip in chips)
        + "</div>",
        unsafe_allow_html=True,
    )
    if public_demo_mode:
        st.caption(t("first_run_intro_public_demo_caution"))


def _render_beginner_result_guide(guide: BeginnerResultGuide, language: str) -> None:
    t = lambda key: get_text(language, key)
    with st.container(border=True):
        render_section_header(
            t("beginner_result_guide_title"),
            t("beginner_result_guide_caption"),
            badge=(t("beginner_result_guide_badge"), "info"),
        )
        st.caption(guide.summary)
        item_columns = st.columns(2)
        visible_items = guide.items[:2]
        secondary_items = guide.items[2:]
        for index, item in enumerate(visible_items):
            with item_columns[index % 2]:
                render_status_card(item.title, item.description, status="info")
        if secondary_items:
            with st.expander(t("result_reading_more_guidance"), expanded=False):
                for item in secondary_items:
                    render_status_card(item.title, item.description, status="info")
        with st.expander(t("beginner_result_guide_cautions"), expanded=False):
            for caution in guide.cautions:
                render_callout(caution.title, caution.description, tone="warning")


def main() -> None:
    _init_state()
    _inject_css()
    inject_regmonkey_theme()

    with st.sidebar:
        language_label = st.selectbox("Language / 语言", ["English", "中文"], index=0, help=get_text("en", "language_help"), key="language_selector")
        language = "zh" if language_label == "中文" else "en"
        t = lambda key: get_text(language, key)
        pages = _workflow_pages(language)
        page_ids = [page_id for page_id, _, _, _ in pages]
        pending_demo_dataset_id = st.session_state.pop("pending_demo_dataset_id", "")
        if pending_demo_dataset_id:
            _load_demo_dataset_into_session(str(pending_demo_dataset_id))
            _request_scroll_anchor("preprocessing_anchor")
        if st.session_state.workflow_page not in page_ids:
            st.session_state.workflow_page = "setup"
        current_page = st.session_state.workflow_page

        st.caption(t("sidebar_control_rail"))
        uploaded_file_input = None
        if current_page != "setup":
            with st.expander(t("sidebar_change_data"), expanded=False):
                st.caption(t("upload_sensitive_data_warning"))
                uploaded_file_input = st.file_uploader(
                    t("upload_file"),
                    type=["csv", "xlsx", "xls"],
                    help=t("upload_help"),
                    key=f"dataset_uploader_{st.session_state.dataset_uploader_version}",
                )
        if current_page == "setup":
            st.subheader(t("preprocessing_settings"))
            skip_rows = st.number_input(t("skip_rows"), min_value=0, max_value=50, value=0, step=1, help=t("skip_rows_help"), key="skip_rows_input")
            use_first_row_as_header = st.checkbox(
                t("use_first_row_as_header"),
                value=True,
                help=t("use_first_row_as_header_help"),
                key="use_first_row_as_header_checkbox",
            )
            auto_detect_metadata = st.checkbox(t("auto_detect"), value=False, help=t("auto_detect_help"), key="auto_detect_metadata_checkbox")
            coerce_numeric = st.checkbox(t("coerce_numeric"), value=True, help=t("coerce_numeric_help"), key="coerce_numeric_checkbox")
        else:
            with st.expander(t("preprocessing_quick_settings"), expanded=False):
                skip_rows = st.number_input(t("skip_rows"), min_value=0, max_value=50, value=0, step=1, help=t("skip_rows_help"), key="skip_rows_input")
                use_first_row_as_header = st.checkbox(
                    t("use_first_row_as_header"),
                    value=True,
                    help=t("use_first_row_as_header_help"),
                    key="use_first_row_as_header_checkbox",
                )
                auto_detect_metadata = st.checkbox(t("auto_detect"), value=False, help=t("auto_detect_help"), key="auto_detect_metadata_checkbox")
                coerce_numeric = st.checkbox(t("coerce_numeric"), value=True, help=t("coerce_numeric_help"), key="coerce_numeric_checkbox")
        _render_reset_session_control(language)

    t = lambda key: get_text(language, key)
    _render_app_shell(pages, current_page, language)
    active_source_metadata = st.session_state.get("data_source_metadata") or {}
    if active_source_metadata.get("kind") == "demo":
        dataset_id = str(active_source_metadata.get("dataset_id") or "")
        try:
            demo_dataset = get_demo_dataset(dataset_id)
            st.info(f"{t('sample_data_loaded_source')}: {t(demo_dataset.display_name_key)}. {t('sample_data_notice')}")
        except ValueError:
            st.info(t("sample_data_notice"))

    if current_page == "setup":
        render_page_header(
            t("setup_page_task_title"),
            t("guide_setup"),
            badges=[
                (
                    t("status_done") if st.session_state.confirm_preprocessing else t("status_pending"),
                    "success" if st.session_state.confirm_preprocessing else "warning",
                )
            ],
        )
        _render_first_run_intro(language, public_demo_mode=is_public_demo_mode())
        if is_public_demo_mode():
            render_callout(t("public_demo_mode_notice_title"), t("public_demo_mode_notice_body"), tone="warning")
        entry_cols = st.columns([1, 1], gap="large")
        with entry_cols[0]:
            with st.container(border=True):
                render_action_card(
                    t("setup_upload_card_title"),
                    t("setup_upload_card_body"),
                    action_label=t("setup_upload_action_label"),
                    action_tone="secondary",
                )
                render_callout(t("upload_sensitive_data_warning"), tone="warning")
                st.caption(t("upload_or_sample_help"))
                uploaded_file_input = st.file_uploader(
                    t("upload_file"),
                    type=["csv", "xlsx", "xls"],
                    help=t("upload_help"),
                    key=f"dataset_uploader_{st.session_state.dataset_uploader_version}",
                )
        with entry_cols[1]:
            with st.container(border=True):
                render_action_card(
                    t("setup_sample_card_title"),
                    t("setup_sample_card_body"),
                    action_label=t("setup_sample_action_label"),
                    action_tone="primary",
                )
                render_callout(t("use_sample_data"), t("sample_data_notice"), tone="success")
                demo_datasets = list_demo_datasets()
                demo_options = [dataset.dataset_id for dataset in demo_datasets]
                selected_demo_dataset_id = st.selectbox(
                    t("sample_dataset_select"),
                    demo_options,
                    format_func=lambda dataset_id: t(get_demo_dataset(dataset_id).display_name_key),
                    key="sample_dataset_selector",
                )
                selected_demo_dataset = get_demo_dataset(selected_demo_dataset_id)
                st.caption(t(selected_demo_dataset.description_key))
                _render_demo_workflow_guide(
                    build_demo_workflow_guide(selected_demo_dataset_id, language),
                    language,
                    expanded=is_public_demo_mode(),
                )
                if st.button(_down_action(t("load_sample_dataset")), key="demo_dataset_load_button", type="secondary", width="stretch"):
                    st.session_state.pending_demo_dataset_id = selected_demo_dataset_id
                    st.rerun()
        with st.expander(t("upload_privacy_notice"), expanded=False):
            st.caption(t("upload_privacy_notice_intro"))
            for key in [
                "upload_privacy_notice_private_data",
                "upload_privacy_notice_session_processing",
                "upload_privacy_notice_deployment",
                "upload_privacy_notice_exports",
                "upload_privacy_notice_review_exports",
                "upload_privacy_notice_sample_data",
            ]:
                st.markdown(f"- {t(key)}")

    uploaded_file = _stable_uploaded_file(uploaded_file_input)
    source_metadata = st.session_state.get("data_source_metadata") or {}

    signature = (
        source_metadata.get("kind", "upload"),
        source_metadata.get("dataset_id", ""),
        getattr(uploaded_file, "name", None),
        getattr(uploaded_file, "size", None),
        int(skip_rows),
        bool(use_first_row_as_header),
        bool(auto_detect_metadata),
        bool(coerce_numeric),
    )
    if st.session_state.data_signature != signature:
        _reset_for_new_data_source()
        st.session_state.data_signature = signature
        if uploaded_file is not None:
            st.session_state.workflow_page = "setup"
            current_page = "setup"

    if uploaded_file is None:
        with st.sidebar:
            st.divider()
            st.subheader(t("workflow_status"))
            _workflow_stepper(pages, "setup", {}, "setup", language)
            st.caption(_next_action_text(language))
            st.divider()
            st.caption(t("api_free_caption"))
        if current_page != "setup":
            st.info(t("upload_or_sample_help"))
        st.stop()

    try:
        raw_df = load_dataset(uploaded_file)
    except FriendlyFileError as error:
        st.session_state.workflow_page = "setup"
        current_page = "setup"
        _reset_for_new_data_source()
        _stage_strip(pages, "setup", language)
        _render_friendly_file_error(error, language)
        st.stop()
    raw_file_quality_warnings = build_file_quality_warnings(raw_df)
    df, preprocessing_log = preprocess_dataframe(
        raw_df,
        skip_rows=int(skip_rows),
        auto_detect_metadata_rows=auto_detect_metadata,
        coerce_numeric=coerce_numeric,
        use_first_row_as_column_names=use_first_row_as_header,
    )
    if df.empty:
        _stage_strip(pages, "setup", language)
        _render_friendly_file_error(FriendlyFileError("no_usable_rows"), language)
        st.stop()
    for warning in raw_file_quality_warnings:
        if warning.code == "blank_column_names":
            _render_friendly_file_error(warning, language)
    _render_file_quality_warnings(df, language)
    preprocessing_confirmed_df = df.copy(deep=True)
    if isinstance(st.session_state.get("missing_data_handled_df"), pd.DataFrame):
        df = st.session_state.missing_data_handled_df.copy(deep=True)

    profile = profile_dataframe(df)
    inferred_roles = infer_variable_roles(df, profile)
    confirmed_roles = st.session_state.variable_roles or inferred_roles
    numeric_measure_columns = get_numeric_measure_variables(confirmed_roles)
    categorical_role_columns = get_categorical_variables(confirmed_roles)
    time_role_columns = get_time_variables(confirmed_roles)

    with st.sidebar:
        st.divider()
        st.subheader(t("workflow_status"))
        setup_complete = bool(st.session_state.confirm_preprocessing and st.session_state.confirm_variable_roles)
        analysis_complete = bool(st.session_state.analysis_ran or st.session_state.guided_workflow_has_run)
        plan_ready = bool((st.session_state.analysis_plan_generated and st.session_state.analysis_plan is not None) or st.session_state.get("active_model_id") or analysis_complete)
        workflow_complete = bool(st.session_state.full_report)
        completed_steps = {
            "setup": workflow_complete or setup_complete,
            "understand": workflow_complete or bool(st.session_state.explored_data or analysis_complete),
            "plan": workflow_complete or plan_ready,
            "run": workflow_complete or analysis_complete,
            "interpret": workflow_complete or analysis_complete,
            "export": workflow_complete,
        }
        _workflow_stepper(pages, current_page, completed_steps, _recommended_page_id(), language)
        if not workflow_complete:
            st.caption(_next_action_text(language))
        st.divider()
        st.caption(t("api_free_caption"))

    _stage_strip(pages, current_page, language)

    if current_page == "setup":
        render_section_header(t("setup_dataset_status_title"), t("setup_dataset_status_body"))
        setup_status_columns = st.columns(4)
        with setup_status_columns[0]:
            render_status_card(
                t("setup_status_cleaned"),
                t("status_done") if st.session_state.confirm_preprocessing else t("status_pending"),
                status="success" if st.session_state.confirm_preprocessing else "warning",
            )
        with setup_status_columns[1]:
            render_status_card(
                t("setup_status_roles"),
                t("status_done") if st.session_state.confirm_variable_roles else t("status_pending"),
                status="success" if st.session_state.confirm_variable_roles else "warning",
            )
        with setup_status_columns[2]:
            render_metric_card(t("setup_status_variables"), len(df.columns), tone="info")
        with setup_status_columns[3]:
            source_label = (
                t("use_sample_data")
                if (st.session_state.get("data_source_metadata") or {}).get("kind") == "demo"
                else t("upload_file")
            )
            render_status_card(t("setup_status_source"), source_label, status="info")
        render_resource_warning_section(
            df,
            language,
            file_size_mb=_uploaded_file_size_mb(uploaded_file),
            public_demo_mode=is_public_demo_mode(),
        )
        if st.session_state.confirm_preprocessing and st.session_state.confirm_variable_roles:
            _action_card(
                t("setup_action_ready_title"),
                t("setup_action_ready_body"),
                t("setup_continue_understand"),
                "understand",
                "setup_continue_understand_button",
            )
        elif not st.session_state.confirm_preprocessing:
            render_action_card(t("setup_action_clean_title"), t("setup_action_clean_body"))
        else:
            render_action_card(t("setup_action_roles_title"), t("setup_action_roles_body"))
        with st.container(border=True):
            render_section_header(t("preprocessing_settings"), t("preprocessing_settings_caption"))
            setting_cols = st.columns(4)
            with setting_cols[0]:
                render_metric_card(t("skip_rows"), int(skip_rows), tone="neutral")
            with setting_cols[1]:
                render_metric_card(t("use_first_row_as_header"), t("yes") if use_first_row_as_header else t("no"), tone="neutral")
            with setting_cols[2]:
                render_metric_card(t("auto_detect"), t("yes") if auto_detect_metadata else t("no"), tone="neutral")
            with setting_cols[3]:
                render_metric_card(t("coerce_numeric"), t("yes") if coerce_numeric else t("no"), tone="neutral")

        with st.container(border=True):
            _scroll_anchor("preprocessing_anchor")
            render_section_header(t("setup_data_preview_title"), t("setup_data_preview_body"))
            raw_tab, processed_tab = st.tabs([t("raw_tab"), t("processed_tab")])
            with raw_tab:
                st.subheader(t("raw_uploaded_data"))
                st.caption(t("raw_uploaded_caption"))
                st.dataframe(raw_df.head(20), width="stretch")
            with processed_tab:
                st.subheader(t("cleaned_data_used"))
                st.caption(t("cleaned_data_caption"))
                st.dataframe(df.head(20), width="stretch")
                text_examples = _text_value_examples(df)
                if text_examples:
                    st.caption(f"{t('text_value_examples')}: {text_examples}")
                if preprocessing_log.get("use_first_row_as_column_names", True):
                    st.caption(t("first_row_used_as_names"))
                elif preprocessing_log.get("generated_temporary_variable_names"):
                    st.caption(t("temporary_names_generated"))

            with st.expander(t("view_preprocessing_details"), expanded=False):
                st.dataframe(_preprocessing_log_frame(raw_df, df, preprocessing_log), width="stretch")
                st.write(f"**{t('converted_numeric')}:**", preprocessing_log["columns_converted_to_numeric"] or "None")
                st.write(f"**{t('kept_categorical')}:**", preprocessing_log["columns_kept_as_categorical"] or "None")
            if st.button(_down_action(t("confirm_cleaned_data")), type="primary", width="stretch", help=t("confirm_cleaned_data_help")):
                st.session_state.confirm_preprocessing = True
                _request_scroll_anchor("variable_roles_anchor")
                st.rerun()
            if st.session_state.confirm_preprocessing:
                st.success(t("preprocessing_confirmed"))

        with st.container(border=True):
            _scroll_anchor("variable_roles_anchor")
            render_section_header(t("variable_roles"), t("guide_variables"))
            if not st.session_state.confirm_preprocessing:
                st.warning(t("variables_confirm_preprocess_first"))
                st.caption(t("variables_confirm_preprocess_subtext"))
            else:
                st.caption(t("variable_roles_help"))
                st.caption(f"{t('supported_roles')}: {', '.join(role_options(normalize_language(language)))}")
                st.caption(t("editable_table_hint"))
                st.caption(t("editable_role_column_hint"))
                role_table = _role_table_for_editor(df, profile, confirmed_roles, language)
                edited_roles = st.data_editor(
                    role_table,
                    width="stretch",
                    hide_index=True,
                    key="variable_role_editor",
                    column_order=[
                        t("table_variable"),
                        t("click_to_choose_role"),
                        t("recommended_role"),
                        t("inferred_type"),
                        t("table_missing"),
                        t("unique_values"),
                        t("table_examples"),
                    ],
                    column_config={
                        t("click_to_choose_role"): st.column_config.SelectboxColumn(
                            t("click_to_choose_role"),
                            options=role_options(normalize_language(language)),
                            help=t("selector_help"),
                            required=True,
                        )
                    },
                    disabled=[
                        t("table_variable"),
                        t("inferred_type"),
                        t("table_missing"),
                        t("unique_values"),
                        t("table_examples"),
                        t("recommended_role"),
                    ],
                )
                st.caption(t("role_summary_caption"))
                st.dataframe(_role_group_frame(_roles_from_editor(edited_roles, language), language), width="stretch")
                if st.button(_next_action(t("confirm_variable_roles")), type="primary", width="stretch"):
                    st.session_state.variable_roles = _roles_from_editor(edited_roles, language)
                    st.session_state.confirm_variable_roles = True
                    _reset_planner()
                    _reset_analysis()
                    _go_to_page("understand")
                if st.session_state.confirm_variable_roles:
                    st.success(t("variable_roles_saved"))
                    if st.button(_next_action(t("setup_continue_understand")), type="primary", width="stretch", key="setup_bottom_continue_understand"):
                        _go_to_page("understand")

    elif current_page == "understand":
        render_page_header(t("tab_understand_title"), t("guide_understand"))
        if not st.session_state.confirm_variable_roles:
            render_empty_state(t("confirm_roles_first"), t("confirm_roles_first_subtext"), t("tab_setup"))
            if st.button(_next_action(t("tab_setup")), type="primary", key="understand_go_setup_button", width="stretch"):
                _go_to_page("setup")
        else:
            missing_counts = pd.Series(profile["missing_counts"])
            missing_variable_count = int((missing_counts > 0).sum()) if not missing_counts.empty else 0
            render_section_header(t("understand_dashboard_title"), t("understand_dashboard_body"))
            first_metric_row = st.columns(3)
            with first_metric_row[0]:
                render_metric_card(t("dq_rows"), len(df), tone="info")
            with first_metric_row[1]:
                render_metric_card(t("dq_columns"), len(df.columns), tone="info")
            with first_metric_row[2]:
                render_metric_card(t("dq_variables_with_missing"), missing_variable_count, tone="warning" if missing_variable_count else "success")
            second_metric_row = st.columns(3)
            with second_metric_row[0]:
                render_metric_card(t("understand_status_numeric"), len(numeric_measure_columns), tone="neutral")
            with second_metric_row[1]:
                render_metric_card(t("understand_status_categorical"), len(categorical_role_columns), tone="neutral")
            with second_metric_row[2]:
                render_metric_card(t("understand_status_time"), len(time_role_columns), tone="neutral")

            summary_cols = st.columns([1.1, 1], gap="large")
            with summary_cols[0]:
                with st.container(border=True):
                    render_data_quality_section(df, language, model_config=st.session_state.get("active_model_config"))
            with summary_cols[1]:
                with st.container(border=True):
                    render_section_header(t("missing_data_plan_builder"), t("missing_data_plan_caption"))
                    missing_data_event = render_missing_data_plan_builder(
                        df,
                        language,
                        applied_plan=st.session_state.get("missing_data_plan"),
                        handling_result=st.session_state.get("missing_data_handling_result"),
                    )
                _action_card(
                    t("understand_action_title"),
                    t("understand_action_body"),
                    t("understand_continue_plan"),
                    "plan",
                    "understand_continue_plan_button",
                )

            with st.expander(t("variable_reference"), expanded=False):
                st.caption(t("full_variable_table_caption"))
                st.dataframe(_role_group_frame(confirmed_roles, language), width="stretch")
                query = st.text_input(t("search_variables"), value="", key="exploration_variable_search")
                role_reference = _role_table_for_editor(df, profile, confirmed_roles, language, editable=False)
                if query:
                    role_reference = role_reference[
                        role_reference[t("table_variable")].astype(str).str.lower().str.contains(query.lower(), na=False)
                    ]
                st.dataframe(role_reference, width="stretch")
            if missing_data_event.reset_requested:
                base_roles = st.session_state.get("missing_data_base_roles")
                _reset_missing_data_handling()
                if isinstance(base_roles, dict):
                    st.session_state.variable_roles = dict(base_roles)
                    st.session_state.confirm_variable_roles = True
                _invalidate_after_missing_data_change()
                st.session_state.workflow_page = "understand"
                st.rerun()
            if missing_data_event.plan_to_apply is not None:
                st.session_state.missing_data_base_roles = dict(st.session_state.get("variable_roles") or {})
                handled_df, handling_result = apply_missing_data_plan(df, missing_data_event.plan_to_apply)
                if handling_result.action_results:
                    state_update = build_missing_data_state_update(
                        handled_df,
                        missing_data_event.plan_to_apply,
                        handling_result,
                        base_roles=st.session_state.get("missing_data_base_roles"),
                    )
                    for key, value in state_update.items():
                        st.session_state[key] = value
                    st.session_state.variable_roles = _roles_with_new_columns(handled_df, st.session_state.get("variable_roles"))
                    st.session_state.confirm_variable_roles = True
                    _invalidate_after_missing_data_change()
                    st.session_state.workflow_page = "understand"
                    st.rerun()
                else:
                    for warning in handling_result.warnings[:3]:
                        st.error(warning)

            if not st.session_state.explored_data:
                st.session_state.explored_data = True
                st.rerun()

            st.caption(
                f"{t('numeric_descriptive_statistics')} · "
                f"{t('binary_descriptive_statistics')} · "
                f"{t('categorical_descriptive_statistics')}"
            )
            with st.expander(t("numeric_descriptive_statistics"), expanded=False):
                stats_tab_numeric, stats_tab_binary, stats_tab_categorical = st.tabs(
                    [
                        t("numeric_descriptive_statistics"),
                        t("binary_descriptive_statistics"),
                        t("categorical_descriptive_statistics"),
                    ]
                )
                with stats_tab_numeric:
                    numeric_stats = profile.get("numeric_descriptive_statistics", pd.DataFrame())
                    if isinstance(numeric_stats, pd.DataFrame) and not numeric_stats.empty:
                        st.dataframe(numeric_stats, width="stretch")
                    else:
                        st.info(t("no_numeric_after_preprocess"))
                with stats_tab_binary:
                    binary_stats = profile.get("binary_descriptive_statistics", pd.DataFrame())
                    if isinstance(binary_stats, pd.DataFrame) and not binary_stats.empty:
                        st.dataframe(binary_stats, width="stretch")
                    else:
                        st.info(t("no_binary_variables"))
                with stats_tab_categorical:
                    categorical_stats = profile.get("categorical_descriptive_statistics", pd.DataFrame())
                    if isinstance(categorical_stats, pd.DataFrame) and not categorical_stats.empty:
                        st.dataframe(categorical_stats, width="stretch")
                    else:
                        st.info(t("no_categorical_variables"))

            with st.expander(t("missing_values_overview"), expanded=False):
                missing_counts = pd.Series(profile["missing_counts"], name="missing_count").reset_index()
                missing_counts.columns = [t("table_variable"), "missing_count"]
                st.dataframe(missing_counts, width="stretch")
                _show_plot(plot_missing_values(df))

            with st.expander(t("variable_distribution"), expanded=False):
                if numeric_measure_columns:
                    distribution_col = st.selectbox(t("distribution_variable"), options=numeric_measure_columns, help=t("selector_help"), key="distribution_variable_select")
                    plot_col_1, plot_col_2 = st.columns(2)
                    with plot_col_1:
                        _show_plot(plot_histogram(df, distribution_col))
                    with plot_col_2:
                        _show_plot(plot_boxplot(df, distribution_col))
                else:
                    st.info(t("no_numeric_after_preprocess"))

            with st.expander(t("relationship_exploration"), expanded=False):
                if len(numeric_measure_columns) >= 2:
                    x_scatter = st.selectbox(t("x_variable"), options=numeric_measure_columns, index=0, key="explore_x", help=t("selector_help"))
                    y_scatter = st.selectbox(t("y_variable"), options=numeric_measure_columns, index=1, key="explore_y", help=t("selector_help"))
                    _show_plot(plot_scatter(df, x_scatter, y_scatter))
                    pairwise_cols = st.multiselect(t("pairwise_variables"), numeric_measure_columns, default=[], key="explore_pairwise")
                    if pairwise_cols:
                        _show_plot(plot_pairwise_scatter(df, pairwise_cols))
                else:
                    st.info(t("need_two_numeric"))

            with st.expander(t("correlation_matrix"), expanded=False):
                include_time = st.checkbox(
                    t("include_time_in_correlation"),
                    value=False,
                    help=t("include_time_in_correlation_help"),
                    key="include_time_in_correlation_checkbox",
                )
                corr_cols = get_correlation_variables(confirmed_roles, include_time=include_time)
                _show_plot(plot_correlation_heatmap(df[corr_cols] if corr_cols else df[[]]))

            with st.expander(t("categorical_analysis"), expanded=False):
                if categorical_role_columns:
                    category_col = st.selectbox(t("categorical_variable"), options=categorical_role_columns, help=t("selector_help"), key="categorical_variable_select")
                    frequency = df[category_col].dropna().astype(str).value_counts().head(20).reset_index()
                    frequency.columns = [category_col, "count"]
                    st.dataframe(frequency, width="stretch")
                    _show_plot(plot_categorical_bar(df, category_col))
                    if numeric_measure_columns:
                        category_numeric = st.selectbox(t("category_numeric_variable"), options=numeric_measure_columns, help=t("selector_help"), key="category_numeric_variable_select")
                        _show_plot(plot_numeric_by_category(df, category_numeric, category_col))
                else:
                    st.info(t("no_categorical_variables"))

            with st.expander(t("time_trend_analysis"), expanded=False):
                if time_role_columns and numeric_measure_columns:
                    time_col = st.selectbox(t("time_variable"), options=time_role_columns, help=t("selector_help"), key="time_variable_select")
                    trend_col = st.selectbox(t("trend_numeric_variable"), options=numeric_measure_columns, help=t("selector_help"), key="trend_numeric_variable_select")
                    agg_label = st.radio(t("trend_aggregation"), [t("mean"), t("sum")], horizontal=True, key="trend_aggregation_radio")
                    agg = "sum" if agg_label == t("sum") else "mean"
                    _show_plot(plot_time_trend(df, time_col, trend_col, agg=agg))
                elif not time_role_columns:
                    st.info(t("no_time_variables"))

            if st.session_state.explored_data:
                st.success(t("exploration_confirmed"))
                if st.button(_next_action(t("understand_continue_plan")), type="primary", width="stretch", key="understand_bottom_continue_plan"):
                    _go_to_page("plan")

    elif current_page == "plan":
        _render_analysis_planner_tab(df, profile, confirmed_roles, language)

    elif current_page == "run":
        render_page_header(t("tab_run_title"), t("guide_run"))
        if not st.session_state.confirm_preprocessing:
            _action_card(t("model_confirm_first"), t("model_confirm_first_subtext"), t("tab_setup"), "setup", "run_go_setup_clean_button")
        elif not st.session_state.confirm_variable_roles:
            _action_card(t("model_roles_first"), t("model_roles_first_subtext"), t("tab_setup"), "setup", "run_go_setup_roles_button")
        else:
            _status_strip(
                [
                    (t("status_setup_complete"), t("status_done")),
                    (t("status_plan_ready"), t("status_done") if st.session_state.get("active_model_id") else t("status_pending")),
                    (t("status_analysis_run"), t("status_done") if st.session_state.analysis_ran else t("status_pending")),
                ]
            )
            active_model_id, active_config = _active_model_setup()
            if active_model_id:
                active_model = _model_definition(active_model_id)
                run_config = dict(active_config)
                run_config["variable_roles"] = confirmed_roles
                active_spec = ExecutableModelSpec.from_config(active_model_id, run_config)
                risk_profile = build_pre_model_risk_profile(df, active_spec)
                setup_errors = active_model.validate(df, run_config)

                with st.container(border=True):
                    render_section_header(
                        t("run_current_model_setup"),
                        t("run_control_panel_caption"),
                        badge=(_setup_source_label(language, st.session_state.model_setup_source), "info"),
                    )
                    _model_summary_card(
                        language,
                        t("model_summary"),
                        [(t("setup_source"), _setup_source_label(language, st.session_state.model_setup_source))]
                        + _model_setup_rows(active_model_id, active_config, language),
                    )

                _scroll_anchor("run_primary_action_anchor")
                if setup_errors:
                    with st.container(border=True):
                        render_section_header(t("run_blocking_validation"), t("pre_model_risk_caption"))
                        for error in setup_errors:
                            _render_model_error(language, active_model_id, error)
                else:
                    preview_df, _, preview_x, preview_log, encoding_info = _preview_model_dataframe(active_model_id, df, run_config)
                    with st.container(border=True):
                        render_section_header(
                            t("run_execution_controls"),
                            t("run_action_body"),
                            badge=(t(f"risk_level_{risk_profile.overall_risk_level}"), "warning" if risk_profile.overall_risk_level in {"warning", "error"} else "success"),
                        )
                        if st.button(_down_action(t("run_analysis")), type="primary", disabled=bool(setup_errors), key="run_active_model_button", width="stretch"):
                            if _execute_model_setup(df, profile, confirmed_roles, language, active_model, run_config):
                                _request_scroll_anchor("run_result_anchor")
                                st.rerun()
                        render_action_card(
                            t("run_action_title"),
                            t("run_ready_to_execute_body"),
                            metadata={
                                t("run_status_model"): active_model.display_name(normalize_language(language)),
                                t("setup_source"): _setup_source_label(language, st.session_state.model_setup_source),
                                t("pre_model_risk_level"): t(f"risk_level_{risk_profile.overall_risk_level}"),
                            },
                        )
                        c1, c2 = st.columns(2)
                        with c1:
                            render_metric_card(t("num_independent_variables"), len(preview_x), tone="info")
                        with c2:
                            render_metric_card(t("estimated_observations"), preview_log["final_row_count"], tone="info")
                        if active_model_id == "panel_fe":
                            with st.expander(t("panel_structure_check"), expanded=False):
                                st.dataframe(_panel_structure_frame(encoding_info.get("panel_structure", {}), language), width="stretch")

                with st.container(border=True):
                    render_pre_model_risk_check(df, active_spec, language)
                if st.session_state.analysis_ran and st.session_state.get("model_run_result") is not None:
                    _scroll_anchor("run_result_anchor")
                    _action_card(
                        t("result_snapshot"),
                        t("guide_interpret"),
                        t("run_continue_interpret"),
                        "interpret",
                        "run_active_model_continue_interpret_button",
                    )
            else:
                _action_card(t("no_model_setup_selected"), t("run_needs_model_setup"), t("tab_plan"), "plan", "run_go_plan_for_setup_button")
            render_section_header(t("secondary_execution_tools"), t("secondary_execution_tools_caption"))
            if st.session_state.analysis_plan_generated or st.session_state.guided_workflow_has_run:
                with st.expander(t("guided_workflow"), expanded=False):
                    _render_guided_workflow_block(df, profile, confirmed_roles, language)
            with st.expander(t("adjust_model_settings"), expanded=False):
                setup_col, ref_col = st.columns([1, 1.05])
                with setup_col:
                    st.subheader(t("model_setup"))
                    model_options = [""] + [model.model_id for model in get_available_models()]
                    labels = {model.model_id: model.display_name(normalize_language(language)) for model in get_available_models()}
                    setup_version = st.session_state.model_setup_version
                    initial_model_id = st.session_state.applied_plan_model_id if st.session_state.analysis_plan_applied else ""
                    initial_model_index = model_options.index(initial_model_id) if initial_model_id in model_options else 0
                    selected_model_id = st.selectbox(
                        t("select_model"),
                        options=model_options,
                        index=initial_model_index,
                        format_func=lambda value: t("select_model_placeholder") if value == "" else labels[value],
                        help=t("selector_help"),
                        key=f"model_selector_{setup_version}",
                    )
                    selected_model = get_model(selected_model_id) if selected_model_id else None
    
                    y_col = ""
                    main_x = []
                    numeric_controls = []
                    categorical_controls = []
                    encode_categoricals = False
                    use_robust = True
                    run_valid = False
    
                    if selected_model is None:
                        st.info(t("select_model_placeholder"))
                    else:
                        model_key = selected_model.model_id
                        applied_plan = _applied_plan_for_model(model_key)
                        if selected_model.model_id == "ols":
                            st.info(t("ols_numeric_only"))
                        elif selected_model.model_id == "panel_fe":
                            st.info(t("panel_model_hint"))
                        else:
                            st.info(t("binary_model_hint"))
                        st.caption(selected_model.description(normalize_language(language)))
                        _status_strip([(t("run_status_model"), selected_model.display_name(normalize_language(language)))])
                        st.subheader(t("recommendations"))
                        role_selectors = get_ols_selector_variables(confirmed_roles)
                        numeric_candidates = role_selectors["main_independent_variables"]
                        categorical_candidates = role_selectors["categorical_controls"]
                        st.caption(f"{t('eligible_numeric_hint')} {t('eligible_variables')}: {len(numeric_candidates)}")
                        st.caption(t("categorical_hint"))
                        if time_role_columns:
                            st.caption(t("year_hint"))
                        if categorical_candidates:
                            st.caption(t("area_hint"))
                        if get_code_identifier_variables(confirmed_roles) or get_entity_id_variables(confirmed_roles):
                            st.caption(t("code_hint"))
    
                        if os.environ.get("REG_MONKEY_UI_TEST_MODE") == "1":
                            if st.button("应用 UI 测试默认变量", width="stretch", key=f"ui_test_defaults_{selected_model.model_id}"):
                                st.session_state.ui_test_defaults_model = selected_model.model_id
                                st.rerun()
    
                        if selected_model.model_id == "ols":
                            y_options = role_selectors["dependent_variables"]
                        elif selected_model.model_id == "panel_fe":
                            y_options = numeric_candidates
                        else:
                            y_options = get_binary_dependent_candidates(df, confirmed_roles)
                        y_default = applied_plan.recommended_dependent_variable if applied_plan else ""
                        y_index = ([""] + list(y_options)).index(y_default) if y_default in y_options else 0
                        y_col = st.selectbox(
                            t("dependent_variable"),
                            [""] + list(y_options),
                            index=y_index,
                            format_func=lambda value: t("select_dependent") if value == "" else value,
                            help=t("dependent_variable_help"),
                            key=f"{model_key}_dependent_variable_select_{setup_version}",
                        )
                        if selected_model.model_id == "ols" and y_col and confirmed_roles.get(y_col) == ROLE_BINARY:
                            st.warning(t("ols_binary_y_warning"))
                        x_pool = [col for col in numeric_candidates if col != y_col]
                        main_x_default = [col for col in (applied_plan.recommended_main_explanatory_variables if applied_plan else []) if col in x_pool]
                        main_x = st.multiselect(
                            t("main_independent_variables"),
                            x_pool,
                            default=main_x_default,
                            placeholder=t("select_independent"),
                            help=t("main_independent_variables_help"),
                            key=f"{model_key}_main_independent_variables_select_{setup_version}",
                        )
                        numeric_controls_default = [col for col in (applied_plan.numeric_controls if applied_plan else []) if col in x_pool and col not in main_x]
                        numeric_controls = st.multiselect(
                            t("numeric_controls"),
                            [col for col in x_pool if col not in main_x],
                            default=numeric_controls_default,
                            help=t("numeric_controls_help"),
                            key=f"{model_key}_numeric_controls_select_{setup_version}",
                        )
                        if selected_model.model_id == "panel_fe":
                            categorical_controls = []
                            encode_categoricals = False
                            st.subheader(t("panel_structure"))
                            entity_options = get_entity_id_variables(confirmed_roles) + get_code_identifier_variables(confirmed_roles)
                            time_options = get_time_variables(confirmed_roles)
                            entity_default = applied_plan.entity_id if applied_plan else ""
                            entity_index = ([""] + entity_options).index(entity_default) if entity_default in entity_options else 0
                            entity_col = st.selectbox(
                                t("entity_id"),
                                [""] + entity_options,
                                index=entity_index,
                                format_func=lambda value: t("select_dependent").replace("dependent variable", "entity ID") if value == "" and language == "en" else ("请选择个体 ID" if value == "" else value),
                                help=t("selector_help"),
                                key=f"{model_key}_entity_id_select_{setup_version}",
                            )
                            time_default = applied_plan.time_id if applied_plan else ""
                            time_index = ([""] + time_options).index(time_default) if time_default in time_options else 0
                            time_col = st.selectbox(
                                t("time_id"),
                                [""] + time_options,
                                index=time_index,
                                format_func=lambda value: "Select time ID" if value == "" and language == "en" else ("请选择时间变量" if value == "" else value),
                                help=t("selector_help"),
                                key=f"{model_key}_time_id_select_{setup_version}",
                            )
                            st.subheader(t("fixed_effects"))
                            entity_effects = st.checkbox(
                                t("entity_fixed_effects"),
                                value=bool(applied_plan.fixed_effects.get("entity", True)) if applied_plan else True,
                                key=f"{model_key}_entity_effects_checkbox_{setup_version}",
                            )
                            time_effects = st.checkbox(
                                t("time_fixed_effects"),
                                value=bool(applied_plan.fixed_effects.get("time", True)) if applied_plan else True,
                                key=f"{model_key}_time_effects_checkbox_{setup_version}",
                            )
                            se_labels = [t("cluster_entity"), t("robust_se").replace("Use HC3 ", "").replace("使用 HC3 ", ""), t("conventional")]
                            se_label = st.radio(t("standard_error_option"), se_labels, horizontal=True, key=f"{model_key}_se_type_select_{setup_version}")
                            se_map = {
                                t("cluster_entity"): "cluster_entity",
                                t("robust_se").replace("Use HC3 ", "").replace("使用 HC3 ", ""): "robust",
                                t("conventional"): "conventional",
                            }
                            standard_errors = se_map.get(se_label, "cluster_entity")
                            use_robust = standard_errors != "conventional"
                            _advanced_model_options(selected_model.model_id, language)
                        else:
                            categorical_default = [col for col in (applied_plan.categorical_controls if applied_plan and model_key == "ols" else []) if col in categorical_candidates]
                            categorical_controls = st.multiselect(
                                t("categorical_controls"),
                                categorical_candidates,
                                default=categorical_default,
                                help=t("categorical_controls_help"),
                                key=f"{model_key}_categorical_controls_select_{setup_version}",
                            )
                            encode_categoricals = st.checkbox(t("encode_categorical_controls"), value=False, help=t("dummy_encoding_help"), key=f"{model_key}_encode_categorical_controls_checkbox_{setup_version}")
                            use_robust = _advanced_model_options(selected_model.model_id, language)
                            entity_col = ""
                            time_col = ""
                            entity_effects = False
                            time_effects = False
                            standard_errors = "hc3" if use_robust else "conventional"
                        if (
                            os.environ.get("REG_MONKEY_UI_TEST_MODE") == "1"
                            and st.session_state.get("ui_test_defaults_model") == selected_model.model_id
                        ):
                            y_col = "pollution_intensity" if selected_model.model_id == "ols" else "export_dummy"
                            if selected_model.model_id == "panel_fe":
                                y_col = "pollution_intensity"
                            main_x = ["digital_index"]
                            numeric_controls = ["leverage", "rd_intensity"]
                            categorical_controls = []
                            encode_categoricals = False
                            entity_col = "firm_id" if selected_model.model_id == "panel_fe" else entity_col
                            time_col = "year" if selected_model.model_id == "panel_fe" else time_col
                            entity_effects = True if selected_model.model_id == "panel_fe" else entity_effects
                            time_effects = True if selected_model.model_id == "panel_fe" else time_effects
                            standard_errors = "cluster_entity" if selected_model.model_id == "panel_fe" else standard_errors
    
                        config = {
                            "dependent_variable": y_col,
                            "main_independent_variables": main_x,
                            "numeric_control_variables": numeric_controls,
                            "categorical_control_variables": categorical_controls,
                            "encode_categorical_controls": encode_categoricals,
                            "robust_standard_errors": use_robust,
                            "entity_id": entity_col,
                            "time_id": time_col,
                            "entity_effects": entity_effects,
                            "time_effects": time_effects,
                            "standard_errors": standard_errors,
                            "robust_cov_type": "HC3",
                            "include_odds_ratios": True,
                            "include_marginal_effects": True,
                            "marginal_effects_type": "average",
                            "variable_roles": confirmed_roles,
                        }
                        errors = selected_model.validate(df, config)
                        run_valid = bool(selected_model and y_col and main_x and not errors)
                        if errors and y_col and main_x:
                            for error in errors:
                                _render_model_error(language, selected_model.model_id, error)
    
                        st.subheader(t("pre_run_confirmation"))
                        if run_valid:
                            preview_df, _, preview_x, preview_log, encoding_info = _preview_model_dataframe(selected_model.model_id, df, config)
                            _action_card(t("run_action_title"), t("run_action_body"))
                            if selected_model.model_id == "panel_fe":
                                st.subheader(t("panel_structure_check"))
                                panel_info = encoding_info.get("panel_structure", {})
                                st.dataframe(_panel_structure_frame(panel_info, language), width="stretch")
                            st.markdown(f"**{t('about_to_run')}: {selected_model.display_name(normalize_language(language))}**")
                            _model_summary_card(
                                language,
                                t("model_summary"),
                                [
                                    (t("run_status_model"), selected_model.display_name(normalize_language(language))),
                                    (t("run_status_y"), y_col),
                                    (t("run_status_main_x"), ", ".join(main_x)),
                                    (t("numeric_controls"), ", ".join(numeric_controls) or t("not_available")),
                                    (
                                        t("standard_errors"),
                                        t("cluster_entity")
                                        if standard_errors == "cluster_entity"
                                        else (
                                            t("robust_se").replace("Use HC3 ", "").replace("使用 HC3 ", "")
                                            if standard_errors == "robust"
                                            else t("conventional")
                                        ),
                                    ),
                                ],
                            )
                            c1, c2 = st.columns(2)
                            c1.metric(t("num_independent_variables"), len(preview_x))
                            c2.metric(t("estimated_observations"), preview_log["final_row_count"])
                            if selected_model.model_id == "panel_fe":
                                st.caption(f"{t('entity_id')}: {entity_col}; {t('time_id')}: {time_col}")
                                st.caption(f"{t('fixed_effects_summary')}: {t('entity_fixed_effects') if entity_effects else ''} {t('time_fixed_effects') if time_effects else ''}")
                            _chip_row(t("main_independent_variables"), main_x)
                            _chip_row(t("numeric_controls"), numeric_controls)
                            if categorical_controls:
                                _chip_row(t("categorical_controls"), categorical_controls)
                            if encoding_info["reference_categories"]:
                                st.caption(f"{t('reference_categories')}: {encoding_info['reference_categories']}")
                            if encoding_info["dummy_variables"]:
                                st.caption(f"{t('generated_dummies')}: {', '.join(encoding_info['dummy_variables'])}")
                            if len(preview_x) == 1:
                                st.warning(t("one_x_warning"))
                                st.caption(t("limited_diagnostics_warning"))
                        else:
                            st.caption(_disabled_reason(language, selected_model, y_col, main_x))
    
                        if st.button(_down_action(t("run_analysis")), type="primary", disabled=not run_valid, width="stretch"):
                            _store_model_setup(selected_model.model_id, config, "manual")
                            if _execute_model_setup(df, profile, confirmed_roles, language, selected_model, config):
                                _request_scroll_anchor("run_result_anchor")
                                st.rerun()
                        if st.session_state.analysis_ran:
                            _scroll_anchor("run_result_anchor")
                            _action_card(
                                t("guided_workflow_results_summary"),
                                t("guide_interpret"),
                                t("run_continue_interpret"),
                                "interpret",
                                "run_continue_interpret_button",
                            )
                with ref_col:
                    st.subheader(t("variable_reference"))
                    st.caption(t("variable_reference_help"))
                    _status_strip(
                        [
                            (t("understand_status_numeric"), str(len(numeric_measure_columns))),
                            (t("understand_status_categorical"), str(len(categorical_role_columns))),
                            (t("understand_status_time"), str(len(time_role_columns))),
                        ]
                    )
                    st.caption(t("role_summary_caption"))
                    st.dataframe(_role_group_frame(confirmed_roles, language), width="stretch")

    elif current_page == "interpret":
        render_page_header(t("tab_interpret_title"), t("guide_interpret"))
        if st.session_state.guided_workflow_has_run and st.session_state.guided_workflow_result is not None:
            with st.expander(t("guided_workflow_results_summary"), expanded=False):
                st.markdown(workflow_summary_markdown(st.session_state.guided_workflow_result, language, detailed=False))
                st.caption(t("guided_workflow_results_caption"))
        if not st.session_state.analysis_ran:
            _action_card(t("interpret_empty_title"), t("interpret_empty_body"), t("interpret_go_run"), "run", "interpret_go_run_button")
        else:
            summary = st.session_state.model_summary
            model_type = str(summary.get("model_type") or "ols").lower()
            display_warnings = _warning_lines_for_ui(
                st.session_state.structured_diagnostics,
                st.session_state.warnings,
                language,
            )
            with st.container(border=True):
                render_section_header(
                    t("result_snapshot"),
                    t("result_dashboard_caption"),
                    badge=(_model_display_name(model_type, language), "info"),
                )
                snapshot_metrics = [
                    (t("run_status_model"), _model_display_name(model_type, language), "info"),
                    (t("available_observations"), str(summary["n_obs"]), "neutral"),
                    (t("fit_metric"), _fit_metric_summary(summary, language), "neutral"),
                    (t("key_variable_result"), _main_effect_summary(st.session_state.regression_table, summary, language), "success"),
                    (t("diagnostic_status"), t("diagnostic_attention") if display_warnings else t("diagnostic_clear"), "warning" if display_warnings else "success"),
                ]
                if model_type == "panel_fe":
                    snapshot_metrics.insert(
                        3,
                        (
                            t("panel_structure_check"),
                            f"{summary.get('entities', 'N/A')} x {summary.get('time_periods', 'N/A')}",
                            "info",
                        ),
                    )
                metric_columns = st.columns(len(snapshot_metrics))
                for column, (label, value, tone) in zip(metric_columns, snapshot_metrics):
                    with column:
                        render_metric_card(label, value, tone=tone)

            _render_result_reading_path(
                st.session_state.regression_table,
                summary,
                display_warnings,
                language,
            )

            active_model_id, active_config = _active_model_setup()
            result_guide = build_beginner_result_guide(
                st.session_state.model_run_result,
                {"model_id": active_model_id or model_type, **active_config},
                language=language,
            )
            _render_beginner_result_guide(result_guide, language)

            with st.container(border=True):
                render_section_header(t("key_findings"), t("key_findings_caption"))
                _key_findings(st.session_state.regression_table, summary, display_warnings, language)

            narrative = generate_narrative(
                st.session_state.regression_table,
                summary,
                display_warnings,
                language=normalize_language(language),
                workflow_result=st.session_state.guided_workflow_result if st.session_state.guided_workflow_has_run else None,
                variable_roles=confirmed_roles,
                structured_diagnostics=st.session_state.structured_diagnostics,
                advanced_outputs=(st.session_state.model_results or {}).get("advanced_outputs", {}),
            )
            if narrative is not None:
                with st.container(border=True):
                    render_section_header(t("research_interpretation"), t("guide_interpret"))
                    selected_mode = _render_explanation_controls(language)
                    if selected_mode == LLMExplanationMode.RULE_BASED.value:
                        _render_research_interpretation(narrative, language)
                    else:
                        _render_mock_llm_interpretation(st.session_state.model_run_result, selected_mode, narrative, language)

            with st.container(border=True):
                render_section_header(t("results_summary"), t("result_metrics_caption"))
                metric_columns = st.columns(5)
                for column, (label, value) in zip(metric_columns, _result_summary_metrics(summary, language)):
                    with column:
                        render_metric_card(label, value, tone="info")
                if str(summary.get("model_type") or "ols").lower() == "panel_fe":
                    st.caption(t("panel_fe_interpretation_note"))
                    st.caption(
                        f"{t('standard_error_type')}: "
                        f"{t('cluster_entity') if summary.get('standard_errors') == 'cluster_entity' else summary.get('standard_errors')}"
                    )
                    st.caption(
                        f"{t('within_r_squared')}: {format_number(summary.get('r_squared_within'), 3, language)}; "
                        f"{t('entities')}: {summary.get('entities', 'N/A')}; "
                        f"{t('time_periods')}: {summary.get('time_periods', 'N/A')}"
                    )
                    st.caption(t("panel_structure_check"))
                    with st.expander(t("panel_structure_check"), expanded=False):
                        st.dataframe(_panel_structure_frame(summary.get("panel_structure", {}), language), width="stretch")
                elif str(summary.get("model_type") or "ols").lower() in {"logit", "probit"}:
                    st.caption(t("pseudo_r_squared_help"))
                    st.caption(t("aic_bic_help"))
                elif str(summary.get("model_type") or "ols").lower() == "psm":
                    st.caption(t("psm_result_summary_note"))

            with st.container(border=True):
                render_section_header(t("diagnostics_and_cautions"), t("view_detailed_diagnostics"))
                if display_warnings:
                    for warning in display_warnings:
                        render_diagnostic_card(t("diagnostic_attention"), warning, severity="warning")
                else:
                    render_callout(t("diagnostic_clear"), t("no_warnings"), tone="success")

            with st.expander(t("technical_interpretation"), expanded=False):
                st.subheader(t("categorical_controls_summary"))
                st.markdown(
                    build_categorical_control_summary(
                        st.session_state.model_results,
                        st.session_state.regression_table,
                        language,
                        detailed=False,
                    )
                )
                _render_advanced_outputs(st.session_state.model_results, language)
                st.subheader(t("show_full_regression_table"))
                st.caption(t("technical_table_caption"))
                st.dataframe(format_regression_table(st.session_state.regression_table, language), width="stretch")

            with st.expander(t("view_detailed_diagnostics"), expanded=False):
                structured_frame = _structured_diagnostics_frame(st.session_state.structured_diagnostics, language)
                if not structured_frame.empty:
                    st.subheader(t("structured_diagnostics"))
                    st.dataframe(structured_frame, width="stretch")
                st.subheader(t("vif_diagnostics"))
                if _independent_variable_count(summary) < 2:
                    st.info(t("vif_skipped"))
                else:
                    st.dataframe(format_diagnostic_table(st.session_state.vif_df, language, digits=2), width="stretch")
                st.subheader(t("model_sample_cleaning_log"))
                st.dataframe(pd.DataFrame([st.session_state.cleaning_log]), width="stretch")
            _action_card(
                t("interpret_action_title"),
                t("interpret_action_body"),
                t("interpret_continue_export"),
                "export",
                "interpret_continue_export_button",
            )
        render_section_header(t("secondary_result_tools"), t("secondary_result_tools_caption"))
        render_run_history_preview(st.session_state.get("run_history"), language)

    elif current_page == "export":
        render_page_header(t("tab_export_title"), t("guide_export"))
        if not st.session_state.analysis_ran:
            _action_card(t("export_empty_title"), t("export_empty_body"), t("interpret_go_run"), "run", "export_go_run_button")
        else:
            report_modes = [t("simple_report"), t("full_report")]
            with st.container(border=True):
                render_section_header(t("export_readiness_status"), t("export_readiness_caption"))
                status_cols = st.columns(3)
                with status_cols[0]:
                    render_metric_card(t("run_status_model"), _model_display_name(str(st.session_state.model_summary.get("model_type") or "ols"), language), tone="info")
                with status_cols[1]:
                    render_metric_card(t("report_ready"), t("status_done"), tone="success")
                with status_cols[2]:
                    render_metric_card(t("reproducibility_pack"), t("status_done") if st.session_state.get("model_run_result") is not None else t("status_pending"), tone="success")

            with st.container(border=True):
                render_section_header(t("export_download_center"), t("download_caption"))
                render_callout(t("export_safe_sharing_title"), t("export_safe_sharing_body"), tone="warning")
                selected_report_mode = st.radio(
                    t("report_mode"),
                    report_modes,
                    horizontal=True,
                    key="export_report_mode_radio",
                )
                download_col_1, download_col_2 = st.columns(2)
                download_col_1.download_button(
                    t("download_simple_report"),
                    st.session_state.simple_report.encode("utf-8"),
                    "reg_monkey_simple_report.md",
                    "text/markdown",
                    width="stretch",
                )
                download_col_2.download_button(
                    t("download_full_report"),
                    st.session_state.full_report.encode("utf-8"),
                    "reg_monkey_full_report.md",
                    "text/markdown",
                    width="stretch",
                )
                run_result = st.session_state.get("model_run_result")
                if run_result is not None:
                    render_reproducibility_pack_download(
                        language,
                        run_result,
                        st.session_state.simple_report,
                        st.session_state.full_report,
                        preprocessing_summary=st.session_state.cleaning_log,
                        variable_roles=confirmed_roles,
                        explanation_mode=st.session_state.get("explanation_mode", "rule_based"),
                        data_quality_profile=data_quality_to_jsonable(build_data_quality_profile(df)),
                        missingness_profile=data_quality_to_jsonable(build_missingness_profile(df)),
                        variable_quality_summary=data_quality_to_jsonable(build_variable_quality_summaries(df)),
                        resource_warning_profile=data_quality_to_jsonable(
                            build_resource_warning_profile(
                                df,
                                file_size_mb=_uploaded_file_size_mb(uploaded_file),
                                public_demo_mode=is_public_demo_mode(),
                            )
                        ),
                        pre_model_risk_profile=data_quality_to_jsonable(build_pre_model_risk_profile(df, run_result.spec)),
                        missing_data_plan=st.session_state.get("missing_data_plan"),
                        missing_data_handling_result=st.session_state.get("missing_data_handling_result"),
                        show_safe_sharing=False,
                    )

            with st.container(border=True):
                render_section_header(t("export_report_builder"), t("export_report_builder_caption"))
                selected_body = t("simple_report_mode_body") if selected_report_mode == t("simple_report") else t("full_report_mode_body")
                mode_col_1, mode_col_2 = st.columns(2)
                with mode_col_1:
                    render_selection_card(
                        t("simple_report"),
                        t("simple_report_mode_body"),
                        selected=selected_report_mode == t("simple_report"),
                        badge=(t("selected_report_mode"), "success") if selected_report_mode == t("simple_report") else None,
                        metadata={
                            t("export_report_scope"): t("export_brief_report_scope"),
                        },
                    )
                with mode_col_2:
                    render_selection_card(
                        t("full_report"),
                        t("full_report_mode_body"),
                        selected=selected_report_mode == t("full_report"),
                        badge=(t("selected_report_mode"), "success") if selected_report_mode == t("full_report") else None,
                        metadata={
                            t("export_report_scope"): t("export_full_report_scope"),
                        },
                    )
                _model_summary_card(
                    language,
                    t("current_report_status"),
                    [
                        (t("selected_report_mode"), selected_report_mode),
                        (t("report_ready"), t("status_done")),
                        (t("export_selected_report_includes"), selected_body),
                    ],
                )
                if st.button(_down_action(t("generate_report")), type="secondary", width="stretch"):
                    st.success(t("report_generated"))

            with st.expander(t("export_technical_contents"), expanded=False):
                st.caption(t("export_reproducibility_contents"))
                st.caption(t("export_reproducibility_no_raw_data"))

            preview_text = st.session_state.simple_report if selected_report_mode == t("simple_report") else st.session_state.full_report
            with st.container(border=True):
                render_section_header(t("report_preview"), selected_report_mode)
                st.markdown(preview_text)
                with st.expander(t("view_markdown_source"), expanded=False):
                    st.text_area(selected_report_mode, preview_text, height=360)
            with st.expander(t("export_secondary_data_downloads"), expanded=False):
                c1, c2 = st.columns(2)
                c1.download_button(t("download_results"), dataframe_to_csv_bytes(st.session_state.regression_table), "regression_results.csv", "text/csv", width="stretch")
                c2.download_button(t("download_cleaned"), dataframe_to_csv_bytes(st.session_state.cleaned_df), "cleaned_regression_dataset.csv", "text/csv", width="stretch")

if __name__ == "__main__":
    main()
