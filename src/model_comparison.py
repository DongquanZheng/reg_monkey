from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import json

from src.reproducibility.serializers import to_jsonable
from src.run_history import AnalysisRunRecord


EXPERIMENTAL_DESIGN_MODELS = {"did", "iv_2sls", "psm"}
REGRESSION_LIKE_MODELS = {"ols", "panel_fe"}
BINARY_CHOICE_MODELS = {"logit", "probit"}

_BLOCKED_TEXT_MARKERS = [
    "api_key",
    "secret",
    "token",
    "password",
    "traceback",
    "streamlit.session_state",
    "session_state",
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
class ModelComparisonItem:
    run_id: str
    model_id: str
    model_display_name_key: str
    source: str
    status: str
    sample_size: int | None
    dependent_variable: str
    key_variable: str
    key_estimate: float | None
    standard_error: float | None
    p_value: float | None
    fit_metric_label: str
    fit_metric_value: float | None
    diagnostics_count: int
    warning_count: int
    advanced_outputs_available: list[str] = field(default_factory=list)
    data_signature: str = ""
    preprocessing_signature: str = ""
    missing_data_plan_applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class ModelComparisonWarning:
    code: str
    severity: str
    message: str
    affected_run_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class ModelComparisonInput:
    run_count: int
    run_ids: list[str]
    items: list[ModelComparisonItem]

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class ModelComparisonResult:
    run_count: int
    items: list[ModelComparisonItem]
    warnings: list[ModelComparisonWarning] = field(default_factory=list)
    comparison_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


def build_model_comparison_input(run_records: list[AnalysisRunRecord | dict[str, Any]]) -> ModelComparisonInput:
    items = [_build_item(_record_to_dict(record)) for record in run_records]
    payload = ModelComparisonInput(run_count=len(items), run_ids=[item.run_id for item in items], items=items)
    _assert_json_safe(payload.to_dict())
    return payload


def compare_run_records(run_records: list[AnalysisRunRecord | dict[str, Any]]) -> ModelComparisonResult:
    comparison_input = build_model_comparison_input(run_records)
    warnings = _build_warnings(comparison_input.items)
    result = ModelComparisonResult(
        run_count=comparison_input.run_count,
        items=comparison_input.items,
        warnings=warnings,
        comparison_notes=[
            "comparison_is_descriptive_not_a_model_recommendation",
            "interpret_differences_with_research_design_and_data_context",
            "no_single_ranking_is_declared",
        ],
    )
    _assert_json_safe(result.to_dict())
    return result


def model_comparison_to_dict(value: ModelComparisonInput | ModelComparisonResult | ModelComparisonItem | ModelComparisonWarning | dict[str, Any]) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        payload = value.to_dict()
    else:
        payload = dict(value)
    _assert_json_safe(payload)
    return payload


def _build_item(record: dict[str, Any]) -> ModelComparisonItem:
    model_result = _dict(record.get("model_result"))
    model_spec = _dict(record.get("model_spec")) or _dict(model_result.get("spec"))
    fit_metrics = _dict(model_result.get("fit_metrics"))
    key_summary = _dict(record.get("key_result_summary"))
    diagnostics = _dict(record.get("diagnostics_summary"))
    advanced = _dict(record.get("advanced_outputs_summary"))
    coefficients = model_result.get("coefficients") if isinstance(model_result.get("coefficients"), list) else []
    key_variable = _key_variable(record, model_spec, key_summary, coefficients)
    coefficient_row = _coefficient_row(coefficients, key_variable)
    fit_label, fit_value = _fit_metric(record, key_summary, fit_metrics)

    return ModelComparisonItem(
        run_id=str(record.get("run_id") or ""),
        model_id=str(record.get("model_id") or key_summary.get("model_id") or ""),
        model_display_name_key=str(record.get("model_display_name_key") or ""),
        source=str(record.get("source") or ""),
        status=str(record.get("status") or ""),
        sample_size=_int_or_none(key_summary.get("observations_used") or fit_metrics.get("n_obs") or fit_metrics.get("observations_used")),
        dependent_variable=str(key_summary.get("dependent_variable") or model_spec.get("dependent_variable") or fit_metrics.get("dependent_variable") or ""),
        key_variable=key_variable,
        key_estimate=_float_or_none(_key_estimate(record, key_summary, fit_metrics, coefficient_row)),
        standard_error=_float_or_none(coefficient_row.get("std_error")),
        p_value=_float_or_none(_p_value(record, fit_metrics, coefficient_row)),
        fit_metric_label=fit_label,
        fit_metric_value=_float_or_none(fit_value),
        diagnostics_count=_int_or_zero(diagnostics.get("diagnostics_count")),
        warning_count=_int_or_zero(diagnostics.get("warning_count")),
        advanced_outputs_available=[str(item) for item in (advanced.get("advanced_outputs_available") or [])],
        data_signature=str(record.get("data_signature") or ""),
        preprocessing_signature=str(record.get("preprocessing_signature") or ""),
        missing_data_plan_applied=bool(record.get("missing_data_plan_applied")),
    )


def _build_warnings(items: list[ModelComparisonItem]) -> list[ModelComparisonWarning]:
    warnings: list[ModelComparisonWarning] = []
    if not items:
        return warnings
    if len(items) < 2:
        warnings.append(_warning("insufficient_runs_for_comparison", "info", "At least two runs are needed for comparison.", [item.run_id for item in items]))
        return warnings

    failed_run_ids = [item.run_id for item in items if item.status != "success"]
    if failed_run_ids:
        warnings.append(_warning("failed_run_included", "warning", "One or more runs failed and should be interpreted as failed records, not estimates.", failed_run_ids))
    if _distinct_values(item.data_signature for item in items):
        warnings.append(_warning("different_data_signature", "warning", "Compared runs use different data signatures.", [item.run_id for item in items]))
    if _distinct_values(item.preprocessing_signature for item in items):
        warnings.append(_warning("different_preprocessing_signature", "warning", "Compared runs use different preprocessing signatures.", [item.run_id for item in items]))
    if _distinct_values(item.dependent_variable for item in items):
        warnings.append(_warning("different_dependent_variable", "warning", "Compared runs use different dependent variables.", [item.run_id for item in items]))
    if _substantial_sample_difference(items):
        warnings.append(_warning("sample_size_difference", "warning", "Compared runs use substantially different sample sizes.", [item.run_id for item in items]))

    model_ids = {item.model_id for item in items}
    experimental = model_ids & EXPERIMENTAL_DESIGN_MODELS
    non_experimental = model_ids - EXPERIMENTAL_DESIGN_MODELS
    if experimental:
        warnings.append(
            _warning(
                "research_design_caution",
                "caution",
                "DID, IV/2SLS, and PSM comparisons are descriptive and depend on research-design assumptions; they do not identify causality automatically.",
                [item.run_id for item in items if item.model_id in EXPERIMENTAL_DESIGN_MODELS],
            )
        )
    if len(experimental) > 1:
        warnings.append(
            _warning(
                "mixed_experimental_models_not_directly_comparable",
                "caution",
                "Mixed experimental research-design models are not directly comparable without an explicit research design.",
                [item.run_id for item in items if item.model_id in EXPERIMENTAL_DESIGN_MODELS],
            )
        )
    if experimental and non_experimental:
        warnings.append(
            _warning(
                "experimental_vs_baseline_caution",
                "caution",
                "Experimental models may be compared with baseline models only as descriptive context, not as a ranking.",
                [item.run_id for item in items],
            )
        )
    if _not_directly_comparable_model_families(model_ids):
        warnings.append(
            _warning(
                "model_family_not_directly_comparable",
                "caution",
                "Compared model families use different estimands, likelihoods, or identifying assumptions and should not be ranked as a single ordered choice.",
                [item.run_id for item in items],
            )
        )
    return warnings


def _record_to_dict(record: AnalysisRunRecord | dict[str, Any]) -> dict[str, Any]:
    if isinstance(record, AnalysisRunRecord):
        return record.to_dict()
    return _json_safe(dict(record))


def _key_variable(record: dict[str, Any], model_spec: dict[str, Any], key_summary: dict[str, Any], coefficients: list[Any]) -> str:
    model_id = str(record.get("model_id") or key_summary.get("model_id") or "")
    if model_id == "did":
        return str(key_summary.get("did_term") or "treatment:post")
    if model_id == "iv_2sls":
        return str(key_summary.get("iv_term") or "fitted_endogenous")
    if model_id == "psm":
        return "att_estimate"
    main_vars = model_spec.get("main_independent_variables")
    if isinstance(main_vars, list) and main_vars:
        return str(main_vars[0])
    for row in coefficients:
        if isinstance(row, dict) and row.get("variable") not in {None, "", "const", "Intercept"}:
            return str(row.get("variable"))
    return ""


def _coefficient_row(coefficients: list[Any], key_variable: str) -> dict[str, Any]:
    for row in coefficients:
        if isinstance(row, dict) and str(row.get("variable") or "") == key_variable:
            return dict(row)
    return {}


def _key_estimate(record: dict[str, Any], key_summary: dict[str, Any], fit_metrics: dict[str, Any], coefficient_row: dict[str, Any]) -> Any:
    model_id = str(record.get("model_id") or key_summary.get("model_id") or "")
    if model_id == "did":
        return key_summary.get("did_estimate") or fit_metrics.get("did_estimate")
    if model_id == "iv_2sls":
        return key_summary.get("iv_estimate") or fit_metrics.get("iv_estimate")
    if model_id == "psm":
        return key_summary.get("att_estimate") or fit_metrics.get("att_estimate")
    return coefficient_row.get("coefficient")


def _p_value(record: dict[str, Any], fit_metrics: dict[str, Any], coefficient_row: dict[str, Any]) -> Any:
    model_id = str(record.get("model_id") or "")
    if model_id == "did":
        return fit_metrics.get("did_p_value") or coefficient_row.get("p_value")
    if model_id == "iv_2sls":
        return fit_metrics.get("iv_p_value") or coefficient_row.get("p_value")
    return coefficient_row.get("p_value")


def _fit_metric(record: dict[str, Any], key_summary: dict[str, Any], fit_metrics: dict[str, Any]) -> tuple[str, Any]:
    model_id = str(record.get("model_id") or key_summary.get("model_id") or "")
    if model_id == "panel_fe":
        return "within_r_squared", key_summary.get("within_r_squared") or fit_metrics.get("within_r_squared")
    if model_id in {"logit", "probit"}:
        return "pseudo_r_squared", key_summary.get("pseudo_r_squared") or fit_metrics.get("pseudo_r_squared")
    if model_id == "did":
        return "regression_r_squared", key_summary.get("r_squared") or fit_metrics.get("r_squared")
    if model_id == "iv_2sls":
        return "second_stage_r_squared", key_summary.get("r_squared") or fit_metrics.get("r_squared")
    if model_id == "psm":
        return "not_applicable", None
    return "r_squared", key_summary.get("r_squared") or fit_metrics.get("r_squared")


def _not_directly_comparable_model_families(model_ids: set[str]) -> bool:
    clean = {model_id for model_id in model_ids if model_id}
    if len(clean) <= 1:
        return False
    if clean <= BINARY_CHOICE_MODELS:
        return False
    if clean <= {"ols", "panel_fe"}:
        return "panel_fe" in clean and "ols" in clean
    return True


def _substantial_sample_difference(items: list[ModelComparisonItem]) -> bool:
    sizes = [item.sample_size for item in items if item.sample_size is not None and item.sample_size > 0]
    if len(sizes) < 2:
        return False
    return (max(sizes) - min(sizes)) / max(sizes) >= 0.2


def _distinct_values(values) -> bool:
    clean = {str(value) for value in values if value not in {None, ""}}
    return len(clean) > 1


def _warning(code: str, severity: str, message: str, run_ids: list[str]) -> ModelComparisonWarning:
    return ModelComparisonWarning(code=code, severity=severity, message=message, affected_run_ids=run_ids)


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _int_or_zero(value: Any) -> int:
    parsed = _int_or_none(value)
    return int(parsed or 0)


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


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
            raise ValueError(f"Model comparison payload contains blocked marker: {marker}")


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


def _blocked_text(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in _BLOCKED_TEXT_MARKERS)


def _blocked_key(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in _BLOCKED_KEY_MARKERS)
