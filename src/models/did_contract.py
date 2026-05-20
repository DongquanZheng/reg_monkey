from __future__ import annotations

import pandas as pd

from src.models.diagnostics import DiagnosticSeverity, ModelDiagnostic
from src.models.execution import ModelSpec, ModelValidationResult


DID_MODEL_ID = "did"
DID_REQUIRED_FIELDS = [
    "dependent_variable",
    "treatment_variable",
    "post_variable",
]
DID_OPTIONAL_FIELDS = [
    "group_variable",
    "entity_id",
    "time_id",
    "numeric_control_variables",
    "categorical_control_variables",
    "entity_effects",
    "time_effects",
    "cluster_variable",
    "standard_errors",
]


def validate_did_spec_contract(df: pd.DataFrame, spec: ModelSpec) -> ModelValidationResult:
    diagnostics: list[ModelDiagnostic] = []
    errors: list[str] = []

    if spec.model_id != DID_MODEL_ID:
        errors.append("DID contract validation requires model_id='did'.")
        diagnostics.append(_diagnostic("did_contract_wrong_model", errors[-1], []))

    required_values = {
        "dependent_variable": spec.dependent_variable,
        "treatment_variable": spec.treatment_variable,
        "post_variable": spec.post_variable,
    }
    for field_name, value in required_values.items():
        if not str(value or "").strip():
            message = f"DID specification is missing required field: {field_name}."
            errors.append(message)
            diagnostics.append(_diagnostic("did_spec_missing_field", message, [field_name]))

    selected_columns = _selected_columns(spec)
    missing_columns = [column for column in selected_columns if column and column not in df.columns]
    for column in missing_columns:
        message = f"DID specification references a column that is not in the dataset: {column}."
        errors.append(message)
        diagnostics.append(_diagnostic("did_spec_missing_column", message, [column]))

    return ModelValidationResult(
        model_id=spec.model_id,
        is_valid=not errors,
        errors=errors,
        warnings=[],
        structured_diagnostics=diagnostics,
        required_fields=list(DID_REQUIRED_FIELDS),
        spec=spec,
    )


def did_spec_contract_summary() -> dict[str, object]:
    return {
        "model_id": DID_MODEL_ID,
        "is_runnable": True,
        "required_fields": list(DID_REQUIRED_FIELDS),
        "optional_fields": list(DID_OPTIONAL_FIELDS),
        "execution_contract": "DID can be run through the v2.1.1 minimal DID runner; it is not exposed in the main UI model selector.",
    }


def _selected_columns(spec: ModelSpec) -> list[str]:
    return [
        spec.dependent_variable,
        spec.treatment_variable,
        spec.post_variable,
        spec.group_variable,
        spec.entity_id,
        spec.time_id,
        spec.cluster_variable,
        *list(spec.numeric_control_variables),
        *list(spec.categorical_control_variables),
    ]


def _diagnostic(code: str, message: str, affected: list[str]) -> ModelDiagnostic:
    return ModelDiagnostic(
        code=code,
        severity=DiagnosticSeverity.ERROR,
        title="DID specification is incomplete",
        message=message,
        affected_variables=affected,
        recommendation="Complete the DID specification before implementing or running a DID model.",
        show_in_ui=False,
        show_in_report=False,
        llm_instruction="Treat this as a blocking DID specification contract issue. Do not estimate or interpret DID results.",
    )
