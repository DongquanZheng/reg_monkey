from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import pandas as pd

from src.models.diagnostics import diagnostics_to_dict
from src.models.execution import ModelRunResult


class LLMExplanationMode(StrEnum):
    RULE_BASED = "rule_based"
    LLM_ASSISTED = "llm_assisted"
    RULE_BASED_LLM_POLISHED = "rule_based_llm_polished"


@dataclass(frozen=True)
class LLMExplanationInput:
    language: str
    mode: str
    report_mode: str
    model_id: str
    model_name: str
    model_setup: dict[str, Any]
    sample_info: dict[str, Any]
    fit_metrics: dict[str, Any]
    coefficients: list[dict[str, Any]]
    structured_diagnostics: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)
    advanced_outputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class LLMExplanationOutput:
    explanation_text: str
    limitations_text: str
    next_steps_text: str
    safety_flags: list[str]
    language: str
    source_mode: str

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


def build_llm_explanation_input(
    result: ModelRunResult,
    language: str = "en",
    report_mode: str = "full",
    mode: LLMExplanationMode | str = LLMExplanationMode.LLM_ASSISTED,
) -> LLMExplanationInput:
    result_payload = result.to_dict()
    spec = result.spec
    metadata = result.model_metadata or {}
    lang = "zh" if language == "zh" else "en"
    model_name = str(
        metadata.get("display_name_zh" if lang == "zh" else "display_name_en")
        or metadata.get("display_name_en")
        or result.model_id
    )

    return LLMExplanationInput(
        language=lang,
        mode=str(mode),
        report_mode=str(report_mode),
        model_id=result.model_id,
        model_name=model_name,
        model_setup={
            "dependent_variable": spec.dependent_variable,
            "main_independent_variables": list(spec.main_independent_variables),
            "numeric_control_variables": list(spec.numeric_control_variables),
            "categorical_control_variables": list(spec.categorical_control_variables),
            "entity_id": spec.entity_id,
            "time_id": spec.time_id,
            "entity_effects": spec.entity_effects,
            "time_effects": spec.time_effects,
            "standard_errors": spec.standard_errors,
        },
        sample_info=dict(result_payload.get("sample_info") or {}),
        fit_metrics=dict(result_payload.get("fit_metrics") or {}),
        coefficients=list(result_payload.get("coefficients") or []),
        structured_diagnostics=diagnostics_to_dict(result.structured_diagnostics),
        warnings=list(result.warnings),
        advanced_outputs=dict(result_payload.get("advanced_outputs") or {}),
    )


def llm_input_to_dict(payload: LLMExplanationInput) -> dict[str, Any]:
    return payload.to_dict()


def llm_output_to_dict(payload: LLMExplanationOutput) -> dict[str, Any]:
    return payload.to_dict()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and pd.isna(value):
        return None
    return value
