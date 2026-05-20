from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


DEFAULT_PRESERVE_KEYS = {
    "language",
    "language_selector",
    "reset_session_version",
}

ANALYSIS_STATE_KEYS = {
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
    "explanation_mode",
    "simple_report",
    "full_report",
    "analysis_error",
    "data_signature",
    "data_quality_profile",
    "missingness_profile",
    "variable_quality_summary",
    "model_sample_impact",
    "pre_model_risk_profile",
    "resource_warning_profile",
    "uploaded_file_cache",
    "demo_dataset_id",
    "data_source_metadata",
    "confirm_preprocessing",
    "confirm_variable_roles",
    "variable_roles",
    "explored_data",
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
    "workflow_page",
    "plan_model_path",
    "plan_manual_requested_model_id",
    "pending_plan_model_path",
    "pending_manual_model_family",
    "pending_demo_dataset_id",
    "missing_data_handled_df",
    "missing_data_plan",
    "missing_data_handling_result",
    "missing_data_base_roles",
    "missing_data_confirm_apply",
    "run_history",
    "sample_dataset_selector",
}

ANALYSIS_STATE_PREFIXES = (
    "dataset_uploader_",
    "missing_data_",
    "planner_",
    "plan_",
    "manual_",
    "did_",
    "iv_",
    "psm_",
    "rd_",
    "research_design_",
    "run_active_model_",
    "run_history_",
    "model_comparison_",
    "export_",
    "guided_workflow_",
    "demo_dataset_",
)


def reset_analysis_session_state(
    state: MutableMapping[str, Any],
    preserve_keys: set[str] | list[str] | tuple[str, ...] | None = None,
) -> None:
    """Clear in-app analysis state while preserving language/UI preferences."""

    preserve = DEFAULT_PRESERVE_KEYS | set(preserve_keys or ())
    previous_uploader_version = _coerce_int(state.get("dataset_uploader_version"), default=0)

    for key in list(state.keys()):
        if key in preserve:
            continue
        if key in ANALYSIS_STATE_KEYS or key.startswith(ANALYSIS_STATE_PREFIXES):
            state.pop(key, None)

    state.update(
        {
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
            "dataset_uploader_version": previous_uploader_version + 1,
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
    )


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
