from __future__ import annotations

from typing import Any

import pandas as pd

from src.models.diagnostics import DiagnosticSeverity, ModelDiagnostic
from src.models.execution import ModelSpec, ModelValidationResult


PSM_MODEL_ID = "psm"
PSM_ALLOWED_ESTIMANDS = {"ATT", "ATE"}
PSM_ALLOWED_MATCHING_METHODS = {"nearest_neighbor"}
PSM_REQUIRED_FIELDS = [
    "dependent_variable",
    "treatment_variable",
    "matching_covariates",
]
PSM_OPTIONAL_FIELDS = [
    "psm_estimand",
    "matching_method",
    "caliper",
]


def validate_psm_spec_contract(df: pd.DataFrame, spec: ModelSpec) -> ModelValidationResult:
    diagnostics: list[ModelDiagnostic] = []
    errors: list[str] = []

    if spec.model_id != PSM_MODEL_ID:
        message = "PSM contract validation requires model_id='psm'."
        errors.append(message)
        diagnostics.append(_diagnostic("psm_contract_wrong_model", message, []))

    required_values: dict[str, Any] = {
        "dependent_variable": spec.dependent_variable,
        "treatment_variable": spec.treatment_variable,
    }
    for field_name, value in required_values.items():
        if not str(value or "").strip():
            message = f"PSM specification is missing required field: {field_name}."
            errors.append(message)
            diagnostics.append(_diagnostic("psm_spec_missing_field", message, [field_name]))

    covariates = _matching_covariates(spec)
    if not covariates:
        message = "PSM specification requires at least one matching covariate."
        errors.append(message)
        diagnostics.append(_diagnostic("psm_spec_missing_covariates", message, ["matching_covariates"]))

    selected_columns = _selected_columns(spec)
    missing_columns = [column for column in selected_columns if column and column not in df.columns]
    for column in missing_columns:
        message = f"PSM specification references a column that is not in the dataset: {column}."
        errors.append(message)
        diagnostics.append(_diagnostic("psm_spec_missing_column", message, [column]))

    for code, message, affected in _role_conflict_errors(spec, covariates):
        errors.append(message)
        diagnostics.append(_diagnostic(code, message, affected))

    if spec.treatment_variable and spec.treatment_variable in df.columns and not _is_binary_like(df[spec.treatment_variable]):
        message = f"PSM treatment variable must be binary-like: {spec.treatment_variable}."
        errors.append(message)
        diagnostics.append(_diagnostic("psm_spec_nonbinary_treatment", message, [spec.treatment_variable]))

    estimand = str(spec.psm_estimand or "ATT").upper()
    if estimand not in PSM_ALLOWED_ESTIMANDS:
        message = f"PSM estimand must be one of: {', '.join(sorted(PSM_ALLOWED_ESTIMANDS))}."
        errors.append(message)
        diagnostics.append(_diagnostic("psm_spec_invalid_estimand", message, ["psm_estimand"]))

    method = str(spec.matching_method or "nearest_neighbor")
    if method not in PSM_ALLOWED_MATCHING_METHODS:
        message = f"PSM matching method must be one of: {', '.join(sorted(PSM_ALLOWED_MATCHING_METHODS))}."
        errors.append(message)
        diagnostics.append(_diagnostic("psm_spec_invalid_matching_method", message, ["matching_method"]))

    if spec.caliper is not None:
        try:
            caliper = float(spec.caliper)
        except (TypeError, ValueError):
            caliper = -1.0
        if caliper <= 0:
            message = "PSM caliper must be numeric and positive when provided."
            errors.append(message)
            diagnostics.append(_diagnostic("psm_spec_invalid_caliper", message, ["caliper"]))

    return ModelValidationResult(
        model_id=spec.model_id,
        is_valid=not errors,
        errors=errors,
        warnings=[],
        structured_diagnostics=diagnostics,
        required_fields=list(PSM_REQUIRED_FIELDS),
        spec=spec,
    )


def psm_spec_contract_summary() -> dict[str, object]:
    return {
        "model_id": PSM_MODEL_ID,
        "is_runnable": True,
        "required_fields": list(PSM_REQUIRED_FIELDS),
        "optional_fields": list(PSM_OPTIONAL_FIELDS),
        "allowed_estimands": sorted(PSM_ALLOWED_ESTIMANDS),
        "allowed_matching_methods": sorted(PSM_ALLOWED_MATCHING_METHODS),
        "execution_contract": "PSM is executable through the backend ModelRunnerRegistry in v2.3.1, but it has no UI exposure and no planner recommendation.",
        "current_outputs": [
            "propensity_score_summary",
            "psm_summary",
            "balance_summary",
            "att_estimate",
            "reproducibility_artifacts",
        ],
    }


def _matching_covariates(spec: ModelSpec) -> list[str]:
    return list(dict.fromkeys([str(item) for item in spec.matching_covariates if str(item or "").strip()]))


def _selected_columns(spec: ModelSpec) -> list[str]:
    return [spec.dependent_variable, spec.treatment_variable, *_matching_covariates(spec)]


def _role_conflict_errors(spec: ModelSpec, covariates: list[str]) -> list[tuple[str, str, list[str]]]:
    errors: list[tuple[str, str, list[str]]] = []
    y = str(spec.dependent_variable or "")
    treatment = str(spec.treatment_variable or "")

    if y and treatment and y == treatment:
        errors.append(
            (
                "psm_spec_role_conflict",
                f"PSM treatment variable cannot also be the outcome variable: {treatment}.",
                [treatment],
            )
        )
    duplicate_covariates = sorted({item for item in spec.matching_covariates if spec.matching_covariates.count(item) > 1})
    if duplicate_covariates:
        errors.append(
            (
                "psm_spec_duplicate_covariate",
                "PSM matching covariates contain duplicate variable(s): " + ", ".join(duplicate_covariates) + ".",
                duplicate_covariates,
            )
        )
    duplicate_roles = sorted(set(covariates).intersection({y, treatment}))
    if duplicate_roles:
        errors.append(
            (
                "psm_spec_role_conflict",
                "PSM matching covariates must not duplicate outcome or treatment roles: " + ", ".join(duplicate_roles) + ".",
                duplicate_roles,
            )
        )
    return errors


def _is_binary_like(series: pd.Series) -> bool:
    values = series.dropna()
    if values.empty:
        return False
    unique = values.drop_duplicates()
    if len(unique) > 2:
        return False
    numeric = pd.to_numeric(unique, errors="coerce")
    if numeric.notna().all():
        return set(numeric.astype(float).tolist()).issubset({0.0, 1.0})
    return True


def _diagnostic(code: str, message: str, affected: list[str]) -> ModelDiagnostic:
    return ModelDiagnostic(
        code=code,
        severity=DiagnosticSeverity.ERROR,
        title="PSM specification is incomplete",
        message=message,
        affected_variables=affected,
        recommendation="Complete the PSM specification contract before running the minimal PSM model.",
        show_in_ui=False,
        show_in_report=False,
        llm_instruction="Treat this as a blocking PSM specification contract issue. No PSM model has been estimated.",
    )
