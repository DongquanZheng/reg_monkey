from __future__ import annotations

from io import BytesIO
import json
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from src.models.execution import ModelRunResult
from src.reproducibility.manifest import build_reproducibility_manifest
from src.reproducibility.serializers import assert_no_blocked_markers, dataframe_to_csv_bytes, to_json_bytes, to_jsonable


def build_reproducibility_bundle(
    result: ModelRunResult,
    brief_report: str,
    technical_report: str,
    preprocessing_summary: dict | None = None,
    variable_roles: dict | None = None,
    explanation_mode: str = "rule_based",
    data_quality_profile: dict | None = None,
    missingness_profile: dict | None = None,
    variable_quality_summary: list | None = None,
    resource_warning_profile: dict | None = None,
    pre_model_risk_profile: dict | None = None,
    missing_data_plan: dict | None = None,
    missing_data_handling_result: dict | None = None,
) -> bytes:
    manifest = build_reproducibility_manifest(result, explanation_mode=explanation_mode)
    manifest.update(
        _data_quality_manifest_fields(
            data_quality_profile=data_quality_profile,
            missingness_profile=missingness_profile,
            variable_quality_summary=variable_quality_summary,
            resource_warning_profile=resource_warning_profile,
            pre_model_risk_profile=pre_model_risk_profile,
        )
    )
    manifest.update(_missing_data_manifest_fields(missing_data_plan, missing_data_handling_result))
    result_payload = result.to_dict()
    diagnostics = {
        "structured_diagnostics": result_payload.get("structured_diagnostics", []),
        "diagnostics": result_payload.get("diagnostics", {}),
    }
    warnings = {"warnings": list(result.warnings)}
    advanced_outputs = dict(result.advanced_outputs or {})

    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("brief_report.md", brief_report or "")
        archive.writestr("technical_report.md", technical_report or "")
        archive.writestr("model_spec.json", to_json_bytes(result.spec.to_dict()))
        archive.writestr("model_result.json", to_json_bytes(result_payload))
        archive.writestr("diagnostics.json", to_json_bytes(diagnostics))
        archive.writestr("warnings.json", to_json_bytes(warnings))
        archive.writestr("advanced_outputs.json", to_json_bytes(advanced_outputs))
        archive.writestr("reproducibility_manifest.json", to_json_bytes(manifest))
        archive.writestr("fit_statistics.json", to_json_bytes(result.fit_metrics))
        archive.writestr("preprocessing_summary.json", to_json_bytes(preprocessing_summary or {}))
        archive.writestr("variable_roles.json", to_json_bytes(variable_roles or {}))
        archive.writestr("data_quality_profile.json", to_json_bytes(data_quality_profile or {}))
        archive.writestr("missingness_profile.json", to_json_bytes(missingness_profile or {}))
        archive.writestr("variable_quality_summary.json", to_json_bytes(variable_quality_summary or []))
        archive.writestr("resource_warning_profile.json", to_json_bytes(resource_warning_profile or {}))
        archive.writestr("pre_model_risk_profile.json", to_json_bytes(pre_model_risk_profile or {}))
        if missing_data_plan or missing_data_handling_result:
            handling = dict(to_jsonable(missing_data_handling_result or {}))
            archive.writestr("missing_data_plan.json", to_json_bytes(missing_data_plan or {}))
            archive.writestr("missing_data_handling_result.json", to_json_bytes(handling))
            archive.writestr("missing_data_action_log.json", to_json_bytes(handling.get("log") or []))
        archive.writestr("coefficients.csv", dataframe_to_csv_bytes(result.coefficients))
        _write_advanced_output_csvs(archive, advanced_outputs)

    bundle = output.getvalue()
    assert_no_blocked_markers(_bundle_text_for_screening(bundle))
    return bundle


def _write_advanced_output_csvs(archive: ZipFile, advanced_outputs: dict) -> None:
    for key, value in advanced_outputs.items():
        frame = _advanced_output_frame(value)
        if frame is not None and not frame.empty:
            archive.writestr(f"advanced_outputs/{key}.csv", dataframe_to_csv_bytes(frame))


def _advanced_output_frame(value) -> pd.DataFrame | None:
    if isinstance(value, pd.DataFrame):
        return _csv_safe_frame(value)
    if isinstance(value, dict) and value:
        return _csv_safe_frame(pd.DataFrame([value]))
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return _csv_safe_frame(pd.DataFrame(value))
    return None


def _csv_safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    safe = frame.copy()
    for column in safe.columns:
        safe[column] = safe[column].map(_csv_safe_value)
    return safe


def _csv_safe_value(value):
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True)
    return value


def _data_quality_manifest_fields(
    data_quality_profile: dict | None,
    missingness_profile: dict | None,
    variable_quality_summary: list | None,
    resource_warning_profile: dict | None,
    pre_model_risk_profile: dict | None,
) -> dict:
    missingness = dict(to_jsonable(missingness_profile or {}))
    quality = dict(to_jsonable(data_quality_profile or {}))
    resource = dict(to_jsonable(resource_warning_profile or {}))
    risk = dict(to_jsonable(pre_model_risk_profile or {}))
    return {
        "data_quality_profile_available": bool(data_quality_profile),
        "missingness_profile_available": bool(missingness_profile),
        "variable_quality_summary_available": bool(variable_quality_summary),
        "resource_warning_profile_available": bool(resource_warning_profile),
        "pre_model_risk_profile_available": bool(pre_model_risk_profile),
        "complete_case_rows": missingness.get("complete_case_rows"),
        "total_missing_percentage": missingness.get("total_missing_percentage"),
        "high_missing_variable_count": len(missingness.get("high_missing_variables") or quality.get("high_missing_variables") or []),
        "resource_warning_count": len(resource.get("warning_items") or []),
        "risk_item_count": len(risk.get("risk_items") or []),
    }


def _missing_data_manifest_fields(missing_data_plan: dict | None, missing_data_handling_result: dict | None) -> dict:
    plan = dict(to_jsonable(missing_data_plan or {}))
    handling = dict(to_jsonable(missing_data_handling_result or {}))
    action_results = list(handling.get("action_results") or [])
    actions = list(plan.get("actions") or handling.get("actions_applied") or [])
    rows_before = handling.get("original_row_count")
    rows_after = handling.get("final_row_count")
    rows_dropped = sum(int(item.get("rows_dropped") or 0) for item in action_results if isinstance(item, dict))
    if not rows_dropped and rows_before is not None and rows_after is not None:
        rows_dropped = max(int(rows_before) - int(rows_after), 0)
    imputed_count = sum(int(item.get("values_filled") or 0) for item in action_results if isinstance(item, dict))
    indicator_count = sum(1 for item in action_results if isinstance(item, dict) and item.get("indicator_variable_created"))
    return {
        "missing_data_plan_applied": bool(plan or handling),
        "missing_data_action_count": len(actions) if actions else len(action_results),
        "rows_before_missing_data_handling": rows_before,
        "rows_after_missing_data_handling": rows_after,
        "rows_dropped_by_missing_data_handling": rows_dropped,
        "imputed_value_count": imputed_count,
        "missing_indicator_count": indicator_count,
    }


def _bundle_text_for_screening(bundle: bytes) -> str:
    parts = []
    with ZipFile(BytesIO(bundle), "r") as archive:
        for name in archive.namelist():
            if name.endswith((".json", ".md", ".csv")):
                parts.append(archive.read(name).decode("utf-8", errors="ignore"))
    return "\n".join(parts)
