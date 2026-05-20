from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from src.models.base import ModelDefinition
from src.models.diagnostics import (
    DiagnosticSeverity,
    ModelDiagnostic,
    blocking_validation_diagnostics,
    dedupe_diagnostics,
    diagnose_binary_outcome,
    diagnose_constant_variables,
    diagnose_advanced_outputs,
    diagnose_heteroskedasticity,
    diagnose_interpretation_constraints,
    diagnose_missingness,
    diagnose_multicollinearity,
    diagnose_panel_structure,
    diagnose_panel_fe_specification,
    diagnose_robust_standard_errors,
    diagnose_sample_size,
    diagnose_separation_risk,
    diagnose_within_variation,
    diagnostic_messages,
    diagnostics_from_legacy_warnings,
    diagnostics_to_dict,
    failed_run_diagnostic,
)
from src.models.registry import get_available_models, get_model
from src.models.runners.base import BaseModelRunner
from src.models.runners.registry import get_model_runner


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    dependent_variable: str
    main_independent_variables: list[str] = field(default_factory=list)
    numeric_control_variables: list[str] = field(default_factory=list)
    categorical_control_variables: list[str] = field(default_factory=list)
    encode_categorical_controls: bool = False
    treatment_variable: str = ""
    post_variable: str = ""
    group_variable: str = ""
    endogenous_variable: str = ""
    instrument_variable: str = ""
    instruments: list[str] = field(default_factory=list)
    exogenous_controls: list[str] = field(default_factory=list)
    first_stage_dependent_variable: str = ""
    matching_covariates: list[str] = field(default_factory=list)
    psm_estimand: str = ""
    matching_method: str = ""
    caliper: float | None = None
    entity_id: str = ""
    time_id: str = ""
    entity_effects: bool = True
    time_effects: bool = True
    cluster_variable: str = ""
    standard_errors: str = "hc3"
    robust_standard_errors: bool = True
    robust_cov_type: str = "HC3"
    include_odds_ratios: bool = True
    include_marginal_effects: bool = True
    marginal_effects_type: str = "average"
    preprocessing_assumptions: dict[str, Any] = field(default_factory=dict)
    variable_roles: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, model_id: str, config: dict[str, Any]) -> "ModelSpec":
        return cls(
            model_id=model_id,
            dependent_variable=str(config.get("dependent_variable") or ""),
            main_independent_variables=list(config.get("main_independent_variables") or config.get("independent_variables") or []),
            numeric_control_variables=list(config.get("numeric_control_variables") or []),
            categorical_control_variables=list(config.get("categorical_control_variables") or []),
            encode_categorical_controls=bool(config.get("encode_categorical_controls", False)),
            treatment_variable=str(config.get("treatment_variable") or ""),
            post_variable=str(config.get("post_variable") or ""),
            group_variable=str(config.get("group_variable") or ""),
            endogenous_variable=str(config.get("endogenous_variable") or ""),
            instrument_variable=str(config.get("instrument_variable") or ""),
            instruments=list(config.get("instruments") or ([config.get("instrument_variable")] if config.get("instrument_variable") else [])),
            exogenous_controls=list(config.get("exogenous_controls") or []),
            first_stage_dependent_variable=str(config.get("first_stage_dependent_variable") or ""),
            matching_covariates=list(config.get("matching_covariates") or []),
            psm_estimand=str(config.get("psm_estimand") or ""),
            matching_method=str(config.get("matching_method") or ""),
            caliper=config.get("caliper", None),
            entity_id=str(config.get("entity_id") or ""),
            time_id=str(config.get("time_id") or ""),
            entity_effects=bool(config.get("entity_effects", True)),
            time_effects=bool(config.get("time_effects", True)),
            cluster_variable=str(config.get("cluster_variable") or ""),
            standard_errors=str(config.get("standard_errors") or ("cluster_entity" if model_id == "panel_fe" else "hc3")),
            robust_standard_errors=bool(config.get("robust_standard_errors", True)),
            robust_cov_type=str(config.get("robust_cov_type") or "HC3"),
            include_odds_ratios=bool(config.get("include_odds_ratios", True)),
            include_marginal_effects=bool(config.get("include_marginal_effects", True)),
            marginal_effects_type=str(config.get("marginal_effects_type") or "average"),
            preprocessing_assumptions=dict(config.get("preprocessing_assumptions") or {}),
            variable_roles=dict(config.get("variable_roles") or {}),
            metadata=dict(config.get("metadata") or {}),
        )

    def to_config(self) -> dict[str, Any]:
        return {
            "dependent_variable": self.dependent_variable,
            "main_independent_variables": list(self.main_independent_variables),
            "independent_variables": list(self.main_independent_variables),
            "numeric_control_variables": list(self.numeric_control_variables),
            "categorical_control_variables": list(self.categorical_control_variables),
            "encode_categorical_controls": self.encode_categorical_controls,
            "treatment_variable": self.treatment_variable,
            "post_variable": self.post_variable,
            "group_variable": self.group_variable,
            "endogenous_variable": self.endogenous_variable,
            "instrument_variable": self.instrument_variable,
            "instruments": list(self.instruments),
            "exogenous_controls": list(self.exogenous_controls),
            "first_stage_dependent_variable": self.first_stage_dependent_variable,
            "matching_covariates": list(self.matching_covariates),
            "psm_estimand": self.psm_estimand,
            "matching_method": self.matching_method,
            "caliper": self.caliper,
            "entity_id": self.entity_id,
            "time_id": self.time_id,
            "entity_effects": self.entity_effects,
            "time_effects": self.time_effects,
            "cluster_variable": self.cluster_variable,
            "standard_errors": self.standard_errors,
            "robust_standard_errors": self.robust_standard_errors,
            "robust_cov_type": self.robust_cov_type,
            "include_odds_ratios": self.include_odds_ratios,
            "include_marginal_effects": self.include_marginal_effects,
            "marginal_effects_type": self.marginal_effects_type,
            "preprocessing_assumptions": dict(self.preprocessing_assumptions),
            "variable_roles": dict(self.variable_roles),
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class ModelValidationResult:
    model_id: str
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    structured_diagnostics: list[ModelDiagnostic] = field(default_factory=list)
    required_fields: list[str] = field(default_factory=list)
    spec: ModelSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["spec"] = self.spec.to_dict() if self.spec else None
        payload["structured_diagnostics"] = diagnostics_to_dict(self.structured_diagnostics)
        return _json_safe(payload)


@dataclass(frozen=True)
class ModelRunResult:
    model_id: str
    status: str
    spec: ModelSpec
    validation: ModelValidationResult
    coefficients: pd.DataFrame = field(default_factory=pd.DataFrame)
    fit_metrics: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    advanced_outputs: dict[str, Any] = field(default_factory=dict)
    structured_diagnostics: list[ModelDiagnostic] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    model_metadata: dict[str, Any] = field(default_factory=dict)
    sample_info: dict[str, Any] = field(default_factory=dict)
    legacy_payload: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def success(self) -> bool:
        return self.status == "success"

    def to_legacy_payload(self) -> dict[str, Any]:
        payload = dict(self.legacy_payload)
        payload["structured_diagnostics"] = diagnostics_to_dict(self.structured_diagnostics)
        payload["advanced_outputs"] = dict(self.advanced_outputs)
        payload["diagnostics"] = dict(self.diagnostics)
        return payload

    def to_dict(self) -> dict[str, Any]:
        diagnostics = dict(self.diagnostics)
        if isinstance(diagnostics.get("vif_df"), pd.DataFrame):
            diagnostics["vif"] = _records(diagnostics.pop("vif_df"))
        return _json_safe(
            {
                "model_id": self.model_id,
                "status": self.status,
                "spec": self.spec.to_dict(),
                "validation": self.validation.to_dict(),
                "coefficients": _records(self.coefficients),
                "fit_metrics": self.fit_metrics,
                "diagnostics": diagnostics,
                "advanced_outputs": self.advanced_outputs,
                "structured_diagnostics": diagnostics_to_dict(self.structured_diagnostics),
                "warnings": list(self.warnings),
                "errors": list(self.errors),
                "model_metadata": self.model_metadata,
                "sample_info": self.sample_info,
            }
        )


class ModelRegistry:
    def get(self, model_id: str) -> ModelDefinition:
        return get_model(model_id)

    def runner(self, model_id: str) -> BaseModelRunner:
        return get_model_runner(model_id)

    def list(self) -> list[ModelDefinition]:
        return get_available_models()

    def required_fields(self, model_id: str) -> list[str]:
        try:
            return list(self.get(model_id).required_roles)
        except ValueError:
            return list(self.runner(model_id).model_definition.required_roles)


def validate_model_spec(df: pd.DataFrame, spec: ModelSpec, registry: ModelRegistry | None = None) -> ModelValidationResult:
    active_registry = registry or ModelRegistry()
    runner = active_registry.runner(spec.model_id)
    errors = runner.validate_spec(spec, df)
    structured = blocking_validation_diagnostics(errors, spec.model_id) if errors else []
    structured.extend(_pre_execution_diagnostics(df, spec))
    structured = dedupe_diagnostics(structured)
    return ModelValidationResult(
        model_id=spec.model_id,
        is_valid=not errors,
        errors=list(errors),
        warnings=diagnostic_messages(structured, {DiagnosticSeverity.WARNING}, ui_only=True),
        structured_diagnostics=structured,
        required_fields=active_registry.required_fields(spec.model_id),
        spec=spec,
    )


def run_model_spec(df: pd.DataFrame, spec: ModelSpec, registry: ModelRegistry | None = None) -> ModelRunResult:
    active_registry = registry or ModelRegistry()
    runner = active_registry.runner(spec.model_id)
    return _run_with_runner(df, spec, runner, active_registry)


def _run_with_runner(
    df: pd.DataFrame,
    spec: ModelSpec,
    runner: BaseModelRunner,
    registry: ModelRegistry | None = None,
) -> ModelRunResult:
    active_registry = registry or ModelRegistry()
    model = runner.model_definition
    validation = validate_model_spec(df, spec, active_registry)
    metadata = _model_metadata(model)
    if not validation.is_valid:
        return ModelRunResult(
            model_id=spec.model_id,
            status="validation_failed",
            spec=spec,
            validation=validation,
            structured_diagnostics=validation.structured_diagnostics,
            warnings=diagnostic_messages(validation.structured_diagnostics, {DiagnosticSeverity.WARNING, DiagnosticSeverity.ERROR}, ui_only=True),
            errors=list(validation.errors),
            model_metadata=metadata,
            sample_info=_sample_info({}, df),
        )

    try:
        payload = runner.fit(spec, df)
        diagnostics = runner.diagnostics(df, payload["cleaned_df"], spec, payload)
    except Exception as exc:
        structured = dedupe_diagnostics(validation.structured_diagnostics + [failed_run_diagnostic(str(exc), spec.model_id)])
        return ModelRunResult(
            model_id=spec.model_id,
            status="failed",
            spec=spec,
            validation=validation,
            structured_diagnostics=structured,
            warnings=diagnostic_messages(structured, {DiagnosticSeverity.WARNING, DiagnosticSeverity.ERROR}, ui_only=True),
            errors=[str(exc)],
            model_metadata=metadata,
            sample_info=_sample_info({}, df),
        )

    structured = dedupe_diagnostics(validation.structured_diagnostics + _post_execution_diagnostics(spec, payload, diagnostics))
    warning_messages = diagnostic_messages(
        structured,
        {DiagnosticSeverity.WARNING, DiagnosticSeverity.ERROR},
        ui_only=True,
    )
    enriched_diagnostics = dict(diagnostics)
    enriched_diagnostics.pop("structured_diagnostics", None)
    enriched_diagnostics["structured_diagnostics"] = diagnostics_to_dict(structured)
    enriched_diagnostics["warnings"] = warning_messages
    advanced_outputs = dict(payload.get("advanced_outputs") or {})
    return ModelRunResult(
        model_id=spec.model_id,
        status="success",
        spec=spec,
        validation=validation,
        coefficients=payload.get("regression_table", pd.DataFrame()),
        fit_metrics=dict(payload.get("model_summary", {})),
        diagnostics=enriched_diagnostics,
        advanced_outputs=advanced_outputs,
        structured_diagnostics=structured,
        warnings=warning_messages,
        errors=[],
        model_metadata=metadata,
        sample_info=_sample_info(payload, df),
        legacy_payload=payload,
    )


def model_run_result_to_dict(result: ModelRunResult) -> dict[str, Any]:
    return result.to_dict()


def _pre_execution_diagnostics(df: pd.DataFrame, spec: ModelSpec) -> list[ModelDiagnostic]:
    diagnostics: list[ModelDiagnostic] = []
    selected = _selected_columns(spec)
    present = [column for column in selected if column in df.columns]
    if present:
        dropped = len(df) - len(df[present].dropna())
        diagnostics.extend(diagnose_missingness(len(df), len(df) - dropped))
        diagnostics.extend(diagnose_constant_variables(df[present].dropna(), _predictor_columns(spec)))
    if spec.model_id in {"logit", "probit"}:
        diagnostics.extend(diagnose_binary_outcome(df, spec.dependent_variable))
    if spec.model_id == "panel_fe" and spec.entity_id in df.columns and spec.time_id in df.columns:
        from src.models.panel_fe import check_panel_structure, detect_within_variation

        panel_columns = [column for column in present if column in df.columns]
        panel_df = df[panel_columns].dropna().copy() if panel_columns else df
        diagnostics.extend(diagnose_panel_structure(check_panel_structure(panel_df, spec.entity_id, spec.time_id)))
        diagnostics.extend(
            diagnose_within_variation(
                detect_within_variation(panel_df, spec.entity_id, spec.time_id, _predictor_columns(spec)),
                spec.entity_effects,
                spec.time_effects,
            )
        )
    return diagnostics


def _post_execution_diagnostics(spec: ModelSpec, payload: dict[str, Any], legacy_diagnostics: dict[str, Any]) -> list[ModelDiagnostic]:
    summary = dict(payload.get("model_summary") or {})
    sample_info = _sample_info(payload, pd.DataFrame(index=range(int(payload.get("cleaning_log", {}).get("original_row_count", 0)))))
    final_rows = int(summary.get("n_obs") or sample_info.get("final_rows") or 0)
    predictor_count = len(summary.get("independent_variables") or _predictor_columns(spec))
    structured: list[ModelDiagnostic] = []
    for item in legacy_diagnostics.get("structured_diagnostics", []) or []:
        if isinstance(item, ModelDiagnostic):
            structured.append(item)
    structured.extend(diagnose_missingness(int(payload.get("cleaning_log", {}).get("original_row_count") or 0), final_rows))
    structured.extend(diagnose_sample_size(final_rows, predictor_count, spec.model_id))
    cleaned_df = payload.get("cleaned_df", pd.DataFrame())
    if isinstance(cleaned_df, pd.DataFrame):
        structured.extend(diagnose_constant_variables(cleaned_df, list(summary.get("independent_variables") or [])))
        if spec.model_id in {"logit", "probit"}:
            structured.extend(diagnose_binary_outcome(cleaned_df, summary.get("dependent_variable", spec.dependent_variable)))
            structured.extend(diagnose_separation_risk(cleaned_df, summary.get("dependent_variable", spec.dependent_variable), list(summary.get("independent_variables") or [])))
    structured.extend(diagnose_multicollinearity(legacy_diagnostics.get("vif_df")))
    structured.extend(diagnose_robust_standard_errors(spec.model_id, summary))
    structured.extend(diagnose_heteroskedasticity(legacy_diagnostics.get("heteroskedasticity")))
    structured.extend(diagnose_advanced_outputs(spec.model_id, payload.get("advanced_outputs")))
    if spec.model_id == "panel_fe":
        structure = legacy_diagnostics.get("panel_structure") or summary.get("panel_structure")
        structured.extend(diagnose_panel_structure(structure))
        structured.extend(diagnose_panel_fe_specification(summary))
        structured.extend(
            diagnose_within_variation(
                legacy_diagnostics.get("within_variation") or summary.get("within_variation"),
                bool(summary.get("entity_effects")),
                bool(summary.get("time_effects")),
            )
        )
    legacy_warnings = list(payload.get("warnings", [])) + list(legacy_diagnostics.get("warnings", []))
    structured.extend(diagnostics_from_legacy_warnings(legacy_warnings))
    structured.extend(diagnose_interpretation_constraints(spec.model_id, summary, spec.standard_errors))
    return structured


def _selected_columns(spec: ModelSpec) -> list[str]:
    columns = [spec.dependent_variable]
    columns.extend(spec.main_independent_variables)
    columns.extend(spec.numeric_control_variables)
    if spec.encode_categorical_controls:
        columns.extend(spec.categorical_control_variables)
    if spec.model_id == "panel_fe":
        columns.extend([spec.entity_id, spec.time_id])
    if spec.model_id == "did":
        columns.extend([spec.treatment_variable, spec.post_variable, spec.group_variable, spec.entity_id, spec.time_id, spec.cluster_variable])
    if spec.model_id == "iv_2sls":
        columns.extend([spec.endogenous_variable, spec.instrument_variable, spec.cluster_variable])
        columns.extend(spec.instruments)
        columns.extend(spec.exogenous_controls)
    if spec.model_id == "psm":
        columns.extend([spec.treatment_variable])
        columns.extend(spec.matching_covariates)
    return [column for column in dict.fromkeys(columns) if column]


def _predictor_columns(spec: ModelSpec) -> list[str]:
    columns = list(spec.main_independent_variables)
    if spec.model_id == "did":
        columns.extend([spec.treatment_variable, spec.post_variable])
    if spec.model_id == "iv_2sls":
        columns.extend([spec.endogenous_variable, spec.instrument_variable])
        columns.extend(spec.instruments)
        columns.extend(spec.exogenous_controls)
    if spec.model_id == "psm":
        columns.extend([spec.treatment_variable])
        columns.extend(spec.matching_covariates)
    columns.extend(spec.numeric_control_variables)
    return [column for column in dict.fromkeys(columns) if column]


def _sample_info(payload: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    cleaning_log = dict(payload.get("cleaning_log") or {})
    summary = dict(payload.get("model_summary") or {})
    return {
        "input_rows": int(len(df)),
        "final_rows": int(cleaning_log.get("final_row_count") or summary.get("n_obs") or 0),
        "dropped_rows": int(cleaning_log.get("dropped_row_count") or 0),
        "dropped_row_percentage": cleaning_log.get("dropped_row_percentage", 0.0),
        "cleaning_log": cleaning_log,
    }


def _model_metadata(model: ModelDefinition) -> dict[str, Any]:
    return {
        "model_id": model.model_id,
        "display_name_en": model.display_name_en,
        "display_name_zh": model.display_name_zh,
        "description_en": model.description_en,
        "description_zh": model.description_zh,
        "required_roles": list(model.required_roles),
        "report_label_en": model.report_label_en,
        "report_label_zh": model.report_label_zh,
        "limitations_en": list(model.limitations_en),
        "limitations_zh": list(model.limitations_zh),
    }


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    return _json_safe(df.to_dict(orient="records"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return _records(value)
    if isinstance(value, pd.Series):
        return _json_safe(value.tolist())
    if pd.isna(value) if isinstance(value, int | float | str | bool | type(None)) else False:
        return None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value
