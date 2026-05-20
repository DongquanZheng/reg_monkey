from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4
import json

from src.models.execution import ModelRunResult
from src.reproducibility.serializers import to_jsonable


RUN_HISTORY_SOURCES = {
    "recommended_plan",
    "manual_configuration",
    "guided_workflow",
    "experimental_manual",
}
RUN_HISTORY_STATUSES = {"success", "failed"}

_BLOCKED_TEXT_MARKERS = [
    "api_key",
    "secret",
    "token",
    "password",
    "traceback",
    "streamlit.session_state",
    "session_state",
    "environment variable",
    "provider secret",
    "c:\\",
    "c:/users/",
    "\\users\\",
    "/users/",
    "/home/",
]
_BLOCKED_KEY_MARKERS = [
    "raw_data",
    "uploaded_data",
    "uploaded_file",
    "file_path",
    "filepath",
    "session_state",
    "api_key",
    "secret",
    "token",
    "password",
]


@dataclass(frozen=True)
class AnalysisRunRecord:
    run_id: str
    created_at: str
    model_id: str
    model_display_name_key: str
    model_spec: dict[str, Any]
    model_result: dict[str, Any]
    data_signature: str = ""
    preprocessing_signature: str = ""
    missing_data_plan_applied: bool = False
    source: str = "manual_configuration"
    status: str = "failed"
    key_result_summary: dict[str, Any] = field(default_factory=dict)
    diagnostics_summary: dict[str, Any] = field(default_factory=dict)
    advanced_outputs_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class RunHistory:
    records: list[AnalysisRunRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"records": [record.to_dict() for record in self.records]}


@dataclass(frozen=True)
class RunHistorySummary:
    total_runs: int
    successful_runs: int
    failed_runs: int
    counts_by_model: dict[str, int] = field(default_factory=dict)
    counts_by_status: dict[str, int] = field(default_factory=dict)
    counts_by_source: dict[str, int] = field(default_factory=dict)
    latest_run_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


def create_analysis_run_record(
    result: ModelRunResult,
    *,
    run_id: str | None = None,
    created_at: str | datetime | None = None,
    data_signature: str = "",
    preprocessing_signature: str = "",
    missing_data_plan_applied: bool = False,
    source: str = "manual_configuration",
    key_result_summary: dict[str, Any] | None = None,
    diagnostics_summary: dict[str, Any] | None = None,
    advanced_outputs_summary: dict[str, Any] | None = None,
) -> AnalysisRunRecord:
    if source not in RUN_HISTORY_SOURCES:
        raise ValueError(f"Unknown run-history source: {source}")
    status = "success" if result.success else "failed"
    if status not in RUN_HISTORY_STATUSES:
        raise ValueError(f"Unknown run-history status: {status}")
    record = AnalysisRunRecord(
        run_id=str(run_id or uuid4()),
        created_at=_created_at_string(created_at),
        model_id=str(result.model_id),
        model_display_name_key=f"model_label_{result.model_id}",
        model_spec=_sanitize(result.spec.to_dict()),
        model_result=_sanitize(result.to_dict()),
        data_signature=str(_sanitize(data_signature)),
        preprocessing_signature=str(_sanitize(preprocessing_signature)),
        missing_data_plan_applied=bool(missing_data_plan_applied),
        source=source,
        status=status,
        key_result_summary=_sanitize(key_result_summary or _build_key_result_summary(result)),
        diagnostics_summary=_sanitize(diagnostics_summary or _build_diagnostics_summary(result)),
        advanced_outputs_summary=_sanitize(advanced_outputs_summary or _build_advanced_outputs_summary(result)),
    )
    _assert_json_safe(record.to_dict())
    return record


def add_run_to_history(history: RunHistory | dict[str, Any] | None, record: AnalysisRunRecord) -> RunHistory:
    records = _history_records(history)
    return RunHistory(records=[*records, record])


def summarize_run_history(history: RunHistory | dict[str, Any] | None) -> RunHistorySummary:
    records = _history_records(history)
    counts_by_model = _counts(record.model_id for record in records)
    counts_by_status = _counts(record.status for record in records)
    counts_by_source = _counts(record.source for record in records)
    summary = RunHistorySummary(
        total_runs=len(records),
        successful_runs=counts_by_status.get("success", 0),
        failed_runs=counts_by_status.get("failed", 0),
        counts_by_model=counts_by_model,
        counts_by_status=counts_by_status,
        counts_by_source=counts_by_source,
        latest_run_id=records[-1].run_id if records else "",
    )
    _assert_json_safe(summary.to_dict())
    return summary


def run_history_to_dict(history: RunHistory | dict[str, Any] | None) -> dict[str, Any]:
    records = _history_records(history)
    payload = RunHistory(records=records).to_dict()
    _assert_json_safe(payload)
    return payload


def _history_records(history: RunHistory | dict[str, Any] | None) -> list[AnalysisRunRecord]:
    if history is None:
        return []
    if isinstance(history, RunHistory):
        return list(history.records)
    records = []
    for item in history.get("records", []) if isinstance(history, dict) else []:
        if isinstance(item, AnalysisRunRecord):
            records.append(item)
        elif isinstance(item, dict):
            records.append(
                AnalysisRunRecord(
                    run_id=str(item.get("run_id") or ""),
                    created_at=str(item.get("created_at") or ""),
                    model_id=str(item.get("model_id") or ""),
                    model_display_name_key=str(item.get("model_display_name_key") or ""),
                    model_spec=dict(item.get("model_spec") or {}),
                    model_result=dict(item.get("model_result") or {}),
                    data_signature=str(item.get("data_signature") or ""),
                    preprocessing_signature=str(item.get("preprocessing_signature") or ""),
                    missing_data_plan_applied=bool(item.get("missing_data_plan_applied")),
                    source=str(item.get("source") or "manual_configuration"),
                    status=str(item.get("status") or "failed"),
                    key_result_summary=dict(item.get("key_result_summary") or {}),
                    diagnostics_summary=dict(item.get("diagnostics_summary") or {}),
                    advanced_outputs_summary=dict(item.get("advanced_outputs_summary") or {}),
                )
            )
    return records


def _created_at_string(created_at: str | datetime | None) -> str:
    if isinstance(created_at, datetime):
        value = created_at
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    if created_at:
        return str(created_at)
    return datetime.now(timezone.utc).isoformat()


def _build_key_result_summary(result: ModelRunResult) -> dict[str, Any]:
    fit = dict(result.fit_metrics or {})
    sample = dict(result.sample_info or {})
    coefficients = result.coefficients
    summary: dict[str, Any] = {
        "model_id": result.model_id,
        "status": "success" if result.success else "failed",
        "dependent_variable": result.spec.dependent_variable,
        "observations_used": sample.get("final_rows") or fit.get("n_obs") or fit.get("observations_used"),
        "coefficient_count": int(len(coefficients)) if coefficients is not None else 0,
        "fit_metric_keys": sorted(str(key) for key in fit.keys()),
    }
    for key in [
        "r_squared",
        "pseudo_r_squared",
        "within_r_squared",
        "did_estimate",
        "iv_estimate",
        "att_estimate",
    ]:
        if key in fit:
            summary[key] = fit[key]
    if result.model_id == "did":
        did_summary = _dict_output(result, "did_summary")
        summary.update({key: did_summary.get(key) for key in ["did_term", "did_estimate", "observations_used"] if key in did_summary})
    elif result.model_id == "iv_2sls":
        iv_summary = _dict_output(result, "iv_summary")
        summary.update({key: iv_summary.get(key) for key in ["iv_term", "iv_estimate", "first_stage_f_statistic", "observations_used"] if key in iv_summary})
    elif result.model_id == "psm":
        psm_summary = _dict_output(result, "psm_summary")
        summary.update({key: psm_summary.get(key) for key in ["att_estimate", "matched_treated_count", "matched_control_count"] if key in psm_summary})
    return summary


def _build_diagnostics_summary(result: ModelRunResult) -> dict[str, Any]:
    diagnostics = result.to_dict().get("structured_diagnostics", [])
    severity_counts = _counts(str(item.get("severity") or "unknown") for item in diagnostics if isinstance(item, dict))
    return {
        "diagnostics_count": len(diagnostics),
        "warning_count": len(result.warnings),
        "error_count": len(result.errors),
        "severity_counts": severity_counts,
        "diagnostic_codes": [str(item.get("code") or "") for item in diagnostics if isinstance(item, dict) and item.get("code")],
    }


def _build_advanced_outputs_summary(result: ModelRunResult) -> dict[str, Any]:
    outputs = dict(result.advanced_outputs or {})
    return {
        "advanced_outputs_available": sorted(str(key) for key in outputs.keys()),
        "advanced_outputs_count": len(outputs),
        "advanced_output_shapes": {str(key): _output_shape(value) for key, value in outputs.items()},
    }


def _dict_output(result: ModelRunResult, key: str) -> dict[str, Any]:
    value = (result.advanced_outputs or {}).get(key)
    return dict(value) if isinstance(value, dict) else {}


def _output_shape(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {"type": "dict", "field_count": len(value)}
    if isinstance(value, list):
        return {"type": "list", "row_count": len(value)}
    if hasattr(value, "shape"):
        rows, columns = value.shape
        return {"type": "table", "row_count": int(rows), "column_count": int(columns)}
    return {"type": type(value).__name__}


def _counts(values) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        key = str(value or "")
        if not key:
            continue
        result[key] = result.get(key, 0) + 1
    return result


def _sanitize(value: Any) -> Any:
    value = to_jsonable(value)
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _blocked_key(key_text):
                continue
            clean[key_text] = _sanitize(item)
        return clean
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        return "[redacted]" if _blocked_text(value) else value
    return value


def _json_safe(value: Any) -> Any:
    payload = _sanitize(value)
    json.dumps(payload, ensure_ascii=False, allow_nan=False)
    return payload


def _assert_json_safe(value: Any) -> None:
    payload = _json_safe(value)
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    lowered = text.lower()
    for marker in _BLOCKED_TEXT_MARKERS:
        if marker in lowered:
            raise ValueError(f"Run history payload contains blocked marker: {marker}")


def _blocked_text(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in _BLOCKED_TEXT_MARKERS)


def _blocked_key(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in _BLOCKED_KEY_MARKERS)
