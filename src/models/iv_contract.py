from __future__ import annotations

import pandas as pd

from src.models.diagnostics import DiagnosticSeverity, ModelDiagnostic
from src.models.execution import ModelSpec, ModelValidationResult


IV_MODEL_ID = "iv_2sls"
IV_REQUIRED_FIELDS = [
    "dependent_variable",
    "endogenous_variable",
    "instruments",
]
IV_OPTIONAL_FIELDS = [
    "numeric_control_variables",
    "categorical_control_variables",
    "exogenous_controls",
    "standard_errors",
    "cluster_variable",
]


def validate_iv_spec_contract(df: pd.DataFrame, spec: ModelSpec) -> ModelValidationResult:
    diagnostics: list[ModelDiagnostic] = []
    errors: list[str] = []

    if spec.model_id != IV_MODEL_ID:
        message = "IV/2SLS contract validation requires model_id='iv_2sls'."
        errors.append(message)
        diagnostics.append(_diagnostic("iv_contract_wrong_model", message, []))

    instruments = _instrument_list(spec)
    required_values = {
        "dependent_variable": spec.dependent_variable,
        "endogenous_variable": spec.endogenous_variable,
    }
    for field_name, value in required_values.items():
        if not str(value or "").strip():
            message = f"IV/2SLS specification is missing required field: {field_name}."
            errors.append(message)
            diagnostics.append(_diagnostic("iv_spec_missing_field", message, [field_name]))

    if not instruments:
        message = "IV/2SLS specification requires at least one instrument."
        errors.append(message)
        diagnostics.append(_diagnostic("iv_spec_missing_instrument", message, ["instruments"]))

    selected_columns = _selected_columns(spec)
    missing_columns = [column for column in selected_columns if column and column not in df.columns]
    for column in missing_columns:
        message = f"IV/2SLS specification references a column that is not in the dataset: {column}."
        errors.append(message)
        diagnostics.append(_diagnostic("iv_spec_missing_column", message, [column]))

    role_errors = _role_conflict_errors(spec, instruments)
    for code, message, affected in role_errors:
        errors.append(message)
        diagnostics.append(_diagnostic(code, message, affected))

    distinct_core_roles = {spec.dependent_variable, spec.endogenous_variable, *instruments}
    distinct_core_roles = {role for role in distinct_core_roles if str(role or "").strip()}
    if len(distinct_core_roles) < 3:
        message = "IV/2SLS specification requires distinct outcome, endogenous, and instrument roles."
        errors.append(message)
        diagnostics.append(_diagnostic("iv_spec_insufficient_roles", message, list(distinct_core_roles)))

    return ModelValidationResult(
        model_id=spec.model_id,
        is_valid=not errors,
        errors=errors,
        warnings=[],
        structured_diagnostics=diagnostics,
        required_fields=list(IV_REQUIRED_FIELDS),
        spec=spec,
    )


def iv_spec_contract_summary() -> dict[str, object]:
    return {
        "model_id": IV_MODEL_ID,
        "is_runnable": True,
        "required_fields": list(IV_REQUIRED_FIELDS),
        "optional_fields": list(IV_OPTIONAL_FIELDS),
        "execution_contract": "IV/2SLS can be run through the v2.2.1 minimal backend runner. It is not registered in the UI-facing model metadata registry and is not exposed in the UI.",
        "future_outputs": [
            "weak_instrument_diagnostics",
            "structured_identification_warnings",
            "overidentification_tests",
        ],
    }


def _instrument_list(spec: ModelSpec) -> list[str]:
    instruments = list(spec.instruments or [])
    if spec.instrument_variable:
        instruments.insert(0, spec.instrument_variable)
    return list(dict.fromkeys([str(item) for item in instruments if str(item or "").strip()]))


def _selected_columns(spec: ModelSpec) -> list[str]:
    return [
        spec.dependent_variable,
        spec.endogenous_variable,
        *_instrument_list(spec),
        *list(spec.numeric_control_variables),
        *list(spec.categorical_control_variables),
        *list(spec.exogenous_controls),
        spec.cluster_variable,
    ]


def _role_conflict_errors(spec: ModelSpec, instruments: list[str]) -> list[tuple[str, str, list[str]]]:
    errors: list[tuple[str, str, list[str]]] = []
    y = str(spec.dependent_variable or "")
    endogenous = str(spec.endogenous_variable or "")
    controls = list(dict.fromkeys([*list(spec.numeric_control_variables), *list(spec.categorical_control_variables), *list(spec.exogenous_controls)]))

    if endogenous and endogenous in instruments:
        errors.append(
            (
                "iv_spec_role_conflict",
                f"IV/2SLS endogenous variable cannot also be used as an instrument: {endogenous}.",
                [endogenous],
            )
        )
    if y and y in instruments:
        errors.append(
            (
                "iv_spec_role_conflict",
                f"IV/2SLS dependent variable cannot also be used as an instrument: {y}.",
                [y],
            )
        )
    duplicate_controls = sorted(set(controls).intersection({y, endogenous, *instruments}))
    if duplicate_controls:
        errors.append(
            (
                "iv_spec_role_conflict",
                "IV/2SLS controls must not duplicate outcome, endogenous, or instrument roles: " + ", ".join(duplicate_controls) + ".",
                duplicate_controls,
            )
        )
    duplicate_instruments = sorted({item for item in instruments if instruments.count(item) > 1})
    if duplicate_instruments:
        errors.append(
            (
                "iv_spec_duplicate_instrument",
                "IV/2SLS instrument list contains duplicate instrument(s): " + ", ".join(duplicate_instruments) + ".",
                duplicate_instruments,
            )
        )
    return errors


def _diagnostic(code: str, message: str, affected: list[str]) -> ModelDiagnostic:
    return ModelDiagnostic(
        code=code,
        severity=DiagnosticSeverity.ERROR,
        title="IV/2SLS specification is incomplete",
        message=message,
        affected_variables=affected,
        recommendation="Complete the IV/2SLS specification contract before implementing or running a 2SLS model.",
        show_in_ui=False,
        show_in_report=False,
        llm_instruction="Treat this as a blocking IV/2SLS contract issue. No IV model has been estimated.",
    )
