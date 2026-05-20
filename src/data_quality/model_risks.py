from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

import pandas as pd

from src.data_quality.missingness import (
    HIGH_MISSING_THRESHOLD,
    SAMPLE_LOSS_HIGH_THRESHOLD,
    SAMPLE_LOSS_MEDIUM_THRESHOLD,
    build_missingness_profile,
    estimate_model_sample_impact,
)
from src.data_quality.profiles import DataQualityProfile, MissingnessProfile, VariableQualitySummary
from src.data_quality.quality_checks import build_data_quality_profile, build_variable_quality_summaries
from src.models.execution import ModelSpec
from src.variable_roles import is_binary_like


SMALL_USABLE_ROWS = 30
PREDICTOR_ROW_RATIO_WARNING = 0.35
HIGH_CARDINALITY_DUMMY_THRESHOLD = 20
SPARSE_DID_CELL_THRESHOLD = 5
LOW_VARIATION_UNIQUE_THRESHOLD = 1


@dataclass(frozen=True)
class PreModelRiskItem:
    code: str
    severity: str
    title: str
    message: str
    affected_variables: list[str] = field(default_factory=list)
    recommendation: str = ""
    model_ids: list[str] = field(default_factory=list)
    show_in_ui: bool = True
    show_in_report: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreModelRiskProfile:
    model_id: str
    selected_variables: list[str]
    usable_rows: int
    dropped_rows: int
    dropped_percentage: float
    risk_items: list[PreModelRiskItem] = field(default_factory=list)
    overall_risk_level: str = "none"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["risk_items"] = [item.to_dict() for item in self.risk_items]
        return payload


def build_pre_model_risk_profile(
    df: pd.DataFrame,
    spec: ModelSpec,
    data_quality_profile: DataQualityProfile | dict[str, Any] | None = None,
    missingness_profile: MissingnessProfile | dict[str, Any] | None = None,
    variable_quality_summaries: list[VariableQualitySummary] | list[dict[str, Any]] | None = None,
) -> PreModelRiskProfile:
    """Build advisory risks for a model setup without estimating or modifying data."""
    if df is None:
        raise ValueError("A DataFrame is required for pre-model risk diagnostics.")
    quality = _as_dict(data_quality_profile or build_data_quality_profile(df))
    missingness = _as_dict(missingness_profile or build_missingness_profile(df))
    summaries = _summary_map(variable_quality_summaries or build_variable_quality_summaries(df))
    selected_variables = selected_variables_from_spec(spec)
    sample_impact = estimate_model_sample_impact(df, selected_variables)
    items: list[PreModelRiskItem] = []

    _add_general_risks(items, df, spec, selected_variables, sample_impact.to_dict(), quality, missingness, summaries)
    _add_model_specific_risks(items, df, spec, summaries)

    return PreModelRiskProfile(
        model_id=spec.model_id,
        selected_variables=selected_variables,
        usable_rows=sample_impact.usable_rows_after_dropna,
        dropped_rows=sample_impact.dropped_rows,
        dropped_percentage=sample_impact.dropped_percentage,
        risk_items=_dedupe_items(items),
        overall_risk_level=_overall_level(items),
    )


def selected_variables_from_spec(spec: ModelSpec) -> list[str]:
    values: list[Any] = [
        spec.dependent_variable,
        spec.main_independent_variables,
        spec.numeric_control_variables,
        spec.categorical_control_variables,
        spec.entity_id,
        spec.time_id,
        spec.treatment_variable,
        spec.post_variable,
        spec.group_variable,
        spec.cluster_variable,
        spec.endogenous_variable,
        spec.instrument_variable,
        spec.instruments,
        spec.exogenous_controls,
        spec.matching_covariates,
    ]
    return _flatten_unique(values)


def _add_general_risks(
    items: list[PreModelRiskItem],
    df: pd.DataFrame,
    spec: ModelSpec,
    selected_variables: list[str],
    sample_impact: dict[str, Any],
    quality: dict[str, Any],
    missingness: dict[str, Any],
    summaries: dict[str, dict[str, Any]],
) -> None:
    missing_columns = [variable for variable in selected_variables if variable not in df.columns]
    if missing_columns:
        items.append(_item("selected_variable_missing", "error", missing_columns, spec.model_id))

    high_missing = set(missingness.get("high_missing_variables") or [])
    all_missing = set(missingness.get("all_missing_variables") or [])
    selected_existing = [variable for variable in selected_variables if variable in df.columns]
    selected_high_missing = [variable for variable in selected_existing if variable in high_missing]
    selected_all_missing = [variable for variable in selected_existing if variable in all_missing]
    if selected_high_missing:
        items.append(_item("selected_variable_high_missing", "warning", selected_high_missing, spec.model_id))
    if selected_all_missing:
        items.append(_item("selected_variable_all_missing", "error", selected_all_missing, spec.model_id))

    for flag, code, severity in [
        ("is_constant", "selected_variable_constant", "warning"),
        ("is_near_constant", "selected_variable_near_constant", "warning"),
        ("is_text_numeric_like", "selected_variable_text_numeric_like", "warning"),
    ]:
        affected = [variable for variable in selected_existing if summaries.get(variable, {}).get(flag)]
        if affected:
            items.append(_item(code, severity, affected, spec.model_id))

    dropped_percentage = float(sample_impact.get("dropped_percentage") or 0.0)
    if dropped_percentage >= SAMPLE_LOSS_HIGH_THRESHOLD:
        items.append(_item("model_sample_loss_high", "warning", sample_impact.get("variables_causing_missing_loss") or [], spec.model_id))
    elif dropped_percentage >= SAMPLE_LOSS_MEDIUM_THRESHOLD:
        items.append(_item("model_sample_loss_medium", "warning", sample_impact.get("variables_causing_missing_loss") or [], spec.model_id))

    usable_rows = int(sample_impact.get("usable_rows_after_dropna") or 0)
    if 0 < usable_rows < SMALL_USABLE_ROWS:
        items.append(_item("usable_sample_small", "warning", [], spec.model_id))

    predictors = _predictor_variables(spec)
    if usable_rows and len(predictors) / usable_rows >= PREDICTOR_ROW_RATIO_WARNING:
        items.append(_item("predictor_row_ratio_high", "warning", predictors, spec.model_id))

    high_cardinality_controls = [
        variable
        for variable in spec.categorical_control_variables
        if summaries.get(variable, {}).get("is_high_cardinality")
        or int(summaries.get(variable, {}).get("unique_count") or 0) > HIGH_CARDINALITY_DUMMY_THRESHOLD
    ]
    if high_cardinality_controls:
        items.append(_item("high_cardinality_categorical_controls", "warning", high_cardinality_controls, spec.model_id))


def _add_model_specific_risks(items: list[PreModelRiskItem], df: pd.DataFrame, spec: ModelSpec, summaries: dict[str, dict[str, Any]]) -> None:
    model_id = spec.model_id
    if model_id == "ols":
        _dependent_constant_risks(items, spec, summaries)
        near_constant_predictors = [variable for variable in _predictor_variables(spec) if summaries.get(variable, {}).get("is_near_constant")]
        if near_constant_predictors:
            items.append(_item("ols_numeric_predictor_near_constant", "warning", near_constant_predictors, model_id))
    elif model_id in {"logit", "probit"}:
        _binary_outcome_risks(items, df, spec.dependent_variable, model_id)
    elif model_id == "panel_fe":
        _panel_fe_risks(items, df, spec, summaries)
    elif model_id == "did":
        _did_risks(items, df, spec)
    elif model_id == "iv_2sls":
        _iv_risks(items, df, spec, summaries)
    elif model_id == "psm":
        _psm_risks(items, df, spec, summaries)


def _dependent_constant_risks(items: list[PreModelRiskItem], spec: ModelSpec, summaries: dict[str, dict[str, Any]]) -> None:
    summary = summaries.get(spec.dependent_variable, {})
    if summary.get("is_constant"):
        items.append(_item("dependent_variable_constant", "warning", [spec.dependent_variable], spec.model_id))
    elif summary.get("is_near_constant"):
        items.append(_item("dependent_variable_near_constant", "warning", [spec.dependent_variable], spec.model_id))


def _binary_outcome_risks(items: list[PreModelRiskItem], df: pd.DataFrame, dependent_variable: str, model_id: str) -> None:
    if dependent_variable not in df.columns:
        return
    series = df[dependent_variable].dropna()
    if not is_binary_like(series):
        items.append(_item("binary_outcome_not_binary_like", "warning", [dependent_variable], model_id))
        return
    counts = series.astype(str).value_counts()
    if len(counts) < 2:
        items.append(_item("binary_outcome_single_class", "error", [dependent_variable], model_id))
        return
    minority = int(counts.min())
    total = int(counts.sum())
    if minority < 5:
        items.append(_item("binary_outcome_few_events", "warning", [dependent_variable], model_id))
    if total and minority / total < 0.1:
        items.append(_item("binary_outcome_severe_imbalance", "warning", [dependent_variable], model_id))


def _panel_fe_risks(items: list[PreModelRiskItem], df: pd.DataFrame, spec: ModelSpec, summaries: dict[str, dict[str, Any]]) -> None:
    missing_fields = [name for name, value in [("entity_id", spec.entity_id), ("time_id", spec.time_id)] if not value or value not in df.columns]
    if missing_fields:
        items.append(_item("panel_missing_entity_time", "error", missing_fields, spec.model_id))
        return
    entity_count = int(df[spec.entity_id].dropna().nunique())
    time_count = int(df[spec.time_id].dropna().nunique())
    if entity_count < 2:
        items.append(_item("panel_too_few_entities", "warning", [spec.entity_id], spec.model_id))
    if time_count < 2:
        items.append(_item("panel_too_few_time_periods", "warning", [spec.time_id], spec.model_id))
    weak_within = [
        variable
        for variable in [spec.dependent_variable] + spec.main_independent_variables + spec.numeric_control_variables
        if variable in df.columns and _within_entity_unique_max(df, spec.entity_id, variable) <= LOW_VARIATION_UNIQUE_THRESHOLD
    ]
    if weak_within:
        items.append(_item("panel_weak_within_variation", "warning", weak_within, spec.model_id))


def _did_risks(items: list[PreModelRiskItem], df: pd.DataFrame, spec: ModelSpec) -> None:
    for variable, code in [(spec.treatment_variable, "did_treatment_not_binary_like"), (spec.post_variable, "did_post_not_binary_like")]:
        if variable and variable in df.columns and not is_binary_like(df[variable].dropna()):
            items.append(_item(code, "warning", [variable], spec.model_id))
    if spec.treatment_variable in df.columns and spec.post_variable in df.columns:
        cells = df[[spec.treatment_variable, spec.post_variable]].dropna()
        counts = cells.groupby([spec.treatment_variable, spec.post_variable]).size()
        observed = {
            (t_norm, p_norm)
            for t, p in counts.index
            for t_norm, p_norm in [(_binary_value(t), _binary_value(p))]
            if t_norm in {0, 1} and p_norm in {0, 1}
        }
        missing = [f"{t}:{p}" for t in (0, 1) for p in (0, 1) if (t, p) not in observed]
        sparse = [
            f"{t_norm}:{p_norm}"
            for (t, p), count in counts.items()
            for t_norm, p_norm in [(_binary_value(t), _binary_value(p))]
            if t_norm in {0, 1} and p_norm in {0, 1} and int(count) < SPARSE_DID_CELL_THRESHOLD
        ]
        if missing:
            items.append(_item("did_missing_treatment_post_cells", "warning", missing, spec.model_id))
        if sparse:
            items.append(_item("did_sparse_treatment_post_cells", "warning", sparse, spec.model_id))
        if spec.group_variable and spec.group_variable in df.columns:
            treatment_by_group = df.groupby(spec.group_variable)[spec.treatment_variable].nunique(dropna=True)
            if (treatment_by_group > 1).any():
                items.append(_item("did_staggered_like_pattern", "warning", [spec.group_variable, spec.treatment_variable], spec.model_id))


def _iv_risks(items: list[PreModelRiskItem], df: pd.DataFrame, spec: ModelSpec, summaries: dict[str, dict[str, Any]]) -> None:
    instruments = list(spec.instruments or ([spec.instrument_variable] if spec.instrument_variable else []))
    missing = [variable for variable in [spec.endogenous_variable] + instruments if not variable or variable not in df.columns]
    if missing:
        items.append(_item("iv_missing_endogenous_or_instrument", "error", missing, spec.model_id))
    role_sets = [spec.dependent_variable, spec.endogenous_variable] + instruments + list(spec.exogenous_controls)
    duplicates = _duplicates([value for value in role_sets if value])
    if duplicates:
        items.append(_item("iv_overlapping_variable_roles", "error", duplicates, spec.model_id))
    low_variation = [variable for variable in instruments if summaries.get(variable, {}).get("unique_count", 0) <= LOW_VARIATION_UNIQUE_THRESHOLD]
    if low_variation:
        items.append(_item("iv_instrument_low_variation", "warning", low_variation, spec.model_id))
    items.append(_item("iv_first_stage_not_checked_pre_run", "info", instruments, spec.model_id))


def _psm_risks(items: list[PreModelRiskItem], df: pd.DataFrame, spec: ModelSpec, summaries: dict[str, dict[str, Any]]) -> None:
    treatment = spec.treatment_variable
    if treatment in df.columns:
        series = df[treatment].dropna()
        unique_values = set(series.astype(str).unique())
        if len(unique_values) < 2:
            items.append(_item("psm_no_treated_or_control_group", "error", [treatment], spec.model_id))
        elif not is_binary_like(series):
            items.append(_item("psm_treatment_not_binary_like", "warning", [treatment], spec.model_id))
        else:
            counts = series.astype(str).value_counts()
            if len(counts) < 2:
                items.append(_item("psm_no_treated_or_control_group", "error", [treatment], spec.model_id))
    high_missing_covariates = [variable for variable in spec.matching_covariates if summaries.get(variable, {}).get("missing_percentage", 0.0) > HIGH_MISSING_THRESHOLD]
    if high_missing_covariates:
        items.append(_item("psm_matching_covariates_high_missing", "warning", high_missing_covariates, spec.model_id))
    near_constant_covariates = [variable for variable in spec.matching_covariates if summaries.get(variable, {}).get("is_near_constant")]
    if near_constant_covariates:
        items.append(_item("psm_matching_covariates_near_constant", "warning", near_constant_covariates, spec.model_id))
    items.append(_item("psm_overlap_not_checked_pre_run", "info", list(spec.matching_covariates), spec.model_id))


def _item(code: str, severity: str, affected_variables: list[str], model_id: str) -> PreModelRiskItem:
    return PreModelRiskItem(
        code=code,
        severity=severity,
        title=code,
        message=code,
        affected_variables=[str(variable) for variable in affected_variables if str(variable)],
        recommendation=code,
        model_ids=[model_id],
    )


def _predictor_variables(spec: ModelSpec) -> list[str]:
    if spec.model_id == "iv_2sls":
        return _flatten_unique([spec.endogenous_variable, spec.instruments, spec.exogenous_controls])
    if spec.model_id == "psm":
        return _flatten_unique([spec.treatment_variable, spec.matching_covariates])
    if spec.model_id == "did":
        return _flatten_unique([spec.treatment_variable, spec.post_variable, spec.main_independent_variables, spec.numeric_control_variables, spec.categorical_control_variables])
    return _flatten_unique([spec.main_independent_variables, spec.numeric_control_variables, spec.categorical_control_variables])


def _summary_map(summaries: list[VariableQualitySummary] | list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in summaries:
        payload = _as_dict(item)
        variable = str(payload.get("variable") or "")
        if variable:
            result[variable] = payload
    return result


def _as_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    return dict(value or {})


def _flatten_unique(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        nested = value if isinstance(value, list | tuple | set) else [value]
        for item in nested:
            name = str(item or "")
            if name and name not in seen:
                seen.add(name)
                result.append(name)
    return result


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _binary_value(value: Any) -> int | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric == 0:
        return 0
    if numeric == 1:
        return 1
    return None


def _within_entity_unique_max(df: pd.DataFrame, entity_id: str, variable: str) -> int:
    if entity_id not in df.columns or variable not in df.columns:
        return 0
    grouped = df[[entity_id, variable]].dropna().groupby(entity_id)[variable].nunique()
    return int(grouped.max()) if not grouped.empty else 0


def _dedupe_items(items: list[PreModelRiskItem]) -> list[PreModelRiskItem]:
    result: list[PreModelRiskItem] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for item in items:
        key = (item.code, tuple(item.affected_variables))
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _overall_level(items: list[PreModelRiskItem]) -> str:
    severities = {item.severity for item in items}
    if "error" in severities:
        return "error"
    if "warning" in severities:
        return "warning"
    if "info" in severities:
        return "info"
    return "none"
