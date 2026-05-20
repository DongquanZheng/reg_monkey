from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import pandas as pd

from src.variable_roles import (
    ROLE_BINARY,
    ROLE_ENTITY,
    ROLE_NUMERIC,
    ROLE_TIME,
    is_binary_like,
)


DesignStatus = Literal["possible", "insufficient_information", "not_applicable", "manual_only"]
ConfidenceLevel = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class ResearchDesignRequirement:
    code: str
    description_key: str
    variables: list[str] = field(default_factory=list)
    status: Literal["present", "missing", "uncertain"] = "uncertain"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchDesignCaution:
    code: str
    message_key: str
    severity: Literal["info", "warning"] = "warning"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchDesignCandidate:
    design_id: str
    display_name_key: str
    status: DesignStatus
    confidence_level: ConfidenceLevel
    detected_from: list[str] = field(default_factory=list)
    required_user_confirmations: list[ResearchDesignRequirement] = field(default_factory=list)
    required_data_features: list[ResearchDesignRequirement] = field(default_factory=list)
    missing_or_uncertain_features: list[ResearchDesignRequirement] = field(default_factory=list)
    cautions: list[ResearchDesignCaution] = field(default_factory=list)
    compatible_model_ids: list[str] = field(default_factory=list)
    not_auto_recommended_reason: str = "research_design_candidates_are_not_model_recommendations"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchDesignCandidateSet:
    schema_version: str = "research_design_candidates.v1"
    generated_by: str = "structural_heuristics"
    candidates: list[ResearchDesignCandidate] = field(default_factory=list)
    source_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_research_design_candidates(
    data: pd.DataFrame,
    variable_roles: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ResearchDesignCandidateSet:
    """Detect structural research-design candidates without recommending or running models."""

    roles = variable_roles or {}
    meta = metadata or {}
    columns = [str(column) for column in data.columns]
    binary_candidates = _binary_candidates(data, roles)
    numeric_candidates = _numeric_candidates(data, roles)
    outcome_candidates = _outcome_candidates(data, roles, meta, numeric_candidates, binary_candidates)
    treatment_candidates = _role_or_name_candidates(
        data,
        roles,
        meta,
        role_tokens=("treatment", "treated"),
        name_tokens=("treatment", "treated", "treat", "intervention", "program", "policy"),
        restrict_to=binary_candidates,
    )
    post_candidates = _role_or_name_candidates(
        data,
        roles,
        meta,
        role_tokens=("post", "after", "time_after"),
        name_tokens=("post", "after", "post_period", "after_policy", "period_after"),
        restrict_to=binary_candidates,
    )
    entity_candidates = _entity_candidates(data, roles, meta)
    time_candidates = _time_candidates(data, roles, meta)
    instrument_candidates = _role_or_name_candidates(
        data,
        roles,
        meta,
        role_tokens=("instrument",),
        name_tokens=("instrument", "iv", "z_", "_z", "distance", "eligibility"),
        restrict_to=None,
    )
    endogenous_candidates = _role_or_name_candidates(
        data,
        roles,
        meta,
        role_tokens=("endogenous",),
        name_tokens=("endogenous", "endog", "x_endog"),
        restrict_to=None,
    )
    covariates = _covariate_candidates(
        numeric_candidates,
        outcome_candidates,
        treatment_candidates,
        post_candidates,
        instrument_candidates,
        endogenous_candidates,
    )

    candidates = [
        _did_candidate(outcome_candidates, treatment_candidates, post_candidates, entity_candidates, time_candidates),
        _iv_candidate(outcome_candidates, endogenous_candidates, instrument_candidates),
        _psm_candidate(outcome_candidates, treatment_candidates, covariates),
        _panel_fe_candidate(data, entity_candidates, time_candidates),
        _observational_regression_candidate(outcome_candidates, numeric_candidates, treatment_candidates, post_candidates),
    ]

    return ResearchDesignCandidateSet(
        candidates=candidates,
        source_summary={
            "row_count": int(len(data)),
            "column_count": int(len(columns)),
            "columns_scanned": columns,
            "outcome_candidates": outcome_candidates,
            "binary_candidates": binary_candidates,
            "numeric_candidates": numeric_candidates,
            "entity_candidates": entity_candidates,
            "time_candidates": time_candidates,
            "planner_integration": "not_integrated",
            "models_run": [],
        },
    )


def _did_candidate(
    outcomes: list[str],
    treatments: list[str],
    posts: list[str],
    entities: list[str],
    times: list[str],
) -> ResearchDesignCandidate:
    requirements = [
        _requirement("outcome", "rd_requirement_outcome", outcomes),
        _requirement("treatment", "rd_requirement_treatment", treatments),
        _requirement("post_indicator", "rd_requirement_post_indicator", posts),
        _requirement("group_or_entity", "rd_requirement_group_or_entity", entities),
        _requirement("time_variable", "rd_requirement_time_variable", times),
    ]
    missing = [item for item in requirements if item.status != "present"]
    status: DesignStatus = "manual_only" if len(missing) <= 1 and treatments and (posts or times) and outcomes else "insufficient_information"
    confidence = "medium" if status == "manual_only" and entities and (posts or times) else "low"
    return ResearchDesignCandidate(
        design_id="did",
        display_name_key="research_design_did",
        status=status,
        confidence_level=confidence,
        detected_from=_present_features(requirements),
        required_user_confirmations=[
            _confirmation("confirm_treatment_control", "rd_confirm_treatment_control"),
            _confirmation("confirm_pre_post_timing", "rd_confirm_pre_post_timing"),
            _confirmation("confirm_parallel_trends", "rd_confirm_parallel_trends"),
        ],
        required_data_features=requirements,
        missing_or_uncertain_features=missing,
        cautions=[
            _caution("did_parallel_trends_not_verified", "rd_caution_did_parallel_trends"),
            _caution("did_timing_requires_confirmation", "rd_caution_did_timing"),
            _caution("did_minimal_runner_no_event_study", "rd_caution_did_minimal_only", severity="info"),
            _caution("research_design_not_auto_recommended", "rd_caution_not_auto_recommended", severity="info"),
        ],
        compatible_model_ids=["did"],
    )


def _iv_candidate(outcomes: list[str], endogenous: list[str], instruments: list[str]) -> ResearchDesignCandidate:
    requirements = [
        _requirement("outcome", "rd_requirement_outcome", outcomes),
        _requirement("endogenous_variable", "rd_requirement_endogenous_variable", endogenous),
        _requirement("instrument_variable", "rd_requirement_instrument_variable", instruments),
    ]
    missing = [item for item in requirements if item.status != "present"]
    status: DesignStatus = "manual_only" if not missing else "insufficient_information"
    confidence = "medium" if status == "manual_only" else "low"
    return ResearchDesignCandidate(
        design_id="iv_2sls",
        display_name_key="research_design_iv_2sls",
        status=status,
        confidence_level=confidence,
        detected_from=_present_features(requirements),
        required_user_confirmations=[
            _confirmation("confirm_endogeneity", "rd_confirm_endogeneity"),
            _confirmation("confirm_exclusion_restriction", "rd_confirm_exclusion_restriction"),
            _confirmation("confirm_instrument_relevance", "rd_confirm_instrument_relevance"),
        ],
        required_data_features=requirements,
        missing_or_uncertain_features=missing,
        cautions=[
            _caution("iv_exclusion_restriction_not_verified", "rd_caution_iv_exclusion"),
            _caution("iv_relevance_requires_first_stage", "rd_caution_iv_relevance"),
            _caution("iv_not_automatically_causal", "rd_caution_iv_not_causal"),
            _caution("research_design_not_auto_recommended", "rd_caution_not_auto_recommended", severity="info"),
        ],
        compatible_model_ids=["iv_2sls"],
    )


def _psm_candidate(outcomes: list[str], treatments: list[str], covariates: list[str]) -> ResearchDesignCandidate:
    requirements = [
        _requirement("outcome", "rd_requirement_outcome", outcomes),
        _requirement("treatment", "rd_requirement_treatment", treatments),
        _requirement("matching_covariates", "rd_requirement_matching_covariates", covariates, minimum=2),
    ]
    missing = [item for item in requirements if item.status != "present"]
    status: DesignStatus = "manual_only" if not missing else "insufficient_information"
    confidence = "medium" if status == "manual_only" else "low"
    return ResearchDesignCandidate(
        design_id="psm",
        display_name_key="research_design_psm",
        status=status,
        confidence_level=confidence,
        detected_from=_present_features(requirements),
        required_user_confirmations=[
            _confirmation("confirm_observed_covariates", "rd_confirm_observed_covariates"),
            _confirmation("confirm_treatment_assignment", "rd_confirm_treatment_assignment"),
            _confirmation("review_balance_diagnostics", "rd_confirm_balance_diagnostics"),
        ],
        required_data_features=requirements,
        missing_or_uncertain_features=missing,
        cautions=[
            _caution("psm_observed_covariates_only", "rd_caution_psm_observed_only"),
            _caution("psm_unobserved_confounding_remains", "rd_caution_psm_unobserved"),
            _caution("psm_balance_required", "rd_caution_psm_balance"),
            _caution("research_design_not_auto_recommended", "rd_caution_not_auto_recommended", severity="info"),
        ],
        compatible_model_ids=["psm"],
    )


def _panel_fe_candidate(data: pd.DataFrame, entities: list[str], times: list[str]) -> ResearchDesignCandidate:
    repeated_entity = bool(entities and data[entities[0]].nunique(dropna=True) < len(data))
    requirements = [
        _requirement("entity_id", "rd_requirement_entity_id", entities),
        _requirement("time_variable", "rd_requirement_time_variable", times),
    ]
    missing = [item for item in requirements if item.status != "present"]
    if entities and times and repeated_entity:
        status: DesignStatus = "possible"
        confidence: ConfidenceLevel = "high"
    elif entities or times:
        status = "insufficient_information"
        confidence = "low"
    else:
        status = "not_applicable"
        confidence = "low"
    return ResearchDesignCandidate(
        design_id="panel_fe",
        display_name_key="research_design_panel_fe",
        status=status,
        confidence_level=confidence,
        detected_from=_present_features(requirements),
        required_user_confirmations=[
            _confirmation("confirm_panel_structure", "rd_confirm_panel_structure"),
            _confirmation("confirm_within_entity_variation", "rd_confirm_within_entity_variation"),
        ],
        required_data_features=requirements,
        missing_or_uncertain_features=missing,
        cautions=[
            _caution("panel_fe_within_variation_required", "rd_caution_panel_within_variation", severity="info"),
        ],
        compatible_model_ids=["panel_fe"],
        not_auto_recommended_reason="panel_fe_remains_part_of_existing_planner_not_research_design_auto_selection",
    )


def _observational_regression_candidate(
    outcomes: list[str],
    numeric: list[str],
    treatments: list[str],
    posts: list[str],
) -> ResearchDesignCandidate:
    predictors = [column for column in numeric if column not in set(outcomes[:1] + treatments + posts)]
    status: DesignStatus = "possible" if outcomes and predictors else "insufficient_information"
    confidence: ConfidenceLevel = "medium" if status == "possible" else "low"
    requirements = [
        _requirement("outcome", "rd_requirement_outcome", outcomes),
        _requirement("explanatory_variables", "rd_requirement_explanatory_variables", predictors),
    ]
    return ResearchDesignCandidate(
        design_id="observational_regression",
        display_name_key="research_design_observational_regression",
        status=status,
        confidence_level=confidence,
        detected_from=_present_features(requirements),
        required_user_confirmations=[
            _confirmation("confirm_outcome_and_predictors", "rd_confirm_outcome_predictors"),
        ],
        required_data_features=requirements,
        missing_or_uncertain_features=[item for item in requirements if item.status != "present"],
        cautions=[
            _caution("observational_regression_association_only", "rd_caution_observational_association"),
        ],
        compatible_model_ids=["ols", "logit", "probit"],
    )


def _binary_candidates(data: pd.DataFrame, roles: dict[str, str]) -> list[str]:
    candidates: list[str] = []
    for column in data.columns:
        name = str(column)
        if roles.get(name) == ROLE_BINARY or is_binary_like(data[column]):
            candidates.append(name)
    return _unique_in_order(candidates)


def _numeric_candidates(data: pd.DataFrame, roles: dict[str, str]) -> list[str]:
    candidates: list[str] = []
    for column in data.columns:
        name = str(column)
        if roles.get(name) in {ROLE_NUMERIC, ROLE_BINARY} or pd.api.types.is_numeric_dtype(data[column]):
            candidates.append(name)
    return _unique_in_order(candidates)


def _outcome_candidates(
    data: pd.DataFrame,
    roles: dict[str, str],
    metadata: dict[str, Any],
    numeric: list[str],
    binary: list[str],
) -> list[str]:
    explicit = _metadata_values(metadata, "outcome", "dependent_variable", "outcome_variable")
    role_based = [column for column, role in roles.items() if _contains_token(role, ("outcome", "dependent")) and column in data.columns]
    name_based = [column for column in data.columns if _contains_token(column, ("outcome", "dependent", "y", "sales", "revenue", "profit"))]
    return _unique_in_order([*explicit, *role_based, *name_based, *numeric, *binary])


def _entity_candidates(data: pd.DataFrame, roles: dict[str, str], metadata: dict[str, Any]) -> list[str]:
    explicit = _metadata_values(metadata, "entity", "entity_id", "group_variable", "group")
    role_based = [column for column, role in roles.items() if role == ROLE_ENTITY and column in data.columns]
    name_based = [
        str(column)
        for column in data.columns
        if _contains_token(column, ("firm_id", "company_id", "entity_id", "unit_id", "group_id"))
    ]
    return _unique_in_order([*explicit, *role_based, *name_based])


def _time_candidates(data: pd.DataFrame, roles: dict[str, str], metadata: dict[str, Any]) -> list[str]:
    explicit = _metadata_values(metadata, "time", "time_variable", "post_variable")
    role_based = [column for column, role in roles.items() if role == ROLE_TIME and column in data.columns]
    name_based = [
        str(column)
        for column in data.columns
        if _contains_token(column, ("year", "date", "month", "quarter", "time", "period"))
    ]
    return _unique_in_order([*explicit, *role_based, *name_based])


def _role_or_name_candidates(
    data: pd.DataFrame,
    roles: dict[str, str],
    metadata: dict[str, Any],
    role_tokens: tuple[str, ...],
    name_tokens: tuple[str, ...],
    restrict_to: list[str] | None,
) -> list[str]:
    allowed = set(restrict_to) if restrict_to is not None else {str(column) for column in data.columns}
    explicit = [value for value in _metadata_values(metadata, *role_tokens, *name_tokens) if value in allowed]
    role_based = [
        column
        for column, role in roles.items()
        if column in allowed and _contains_token(role, role_tokens)
    ]
    name_based = [str(column) for column in data.columns if str(column) in allowed and _contains_token(column, name_tokens)]
    return _unique_in_order([*explicit, *role_based, *name_based])


def _covariate_candidates(
    numeric: list[str],
    outcomes: list[str],
    treatments: list[str],
    posts: list[str],
    instruments: list[str],
    endogenous: list[str],
) -> list[str]:
    excluded = set(outcomes[:1] + treatments + posts + instruments + endogenous)
    return [column for column in numeric if column not in excluded]


def _metadata_values(metadata: dict[str, Any], *keys: str) -> list[str]:
    values: list[str] = []
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            values.extend(str(item) for item in value)
        else:
            values.append(str(value))
    return values


def _requirement(
    code: str,
    description_key: str,
    variables: list[str],
    minimum: int = 1,
) -> ResearchDesignRequirement:
    status: Literal["present", "missing", "uncertain"] = "present" if len(variables) >= minimum else "missing"
    return ResearchDesignRequirement(
        code=code,
        description_key=description_key,
        variables=list(variables),
        status=status,
    )


def _confirmation(code: str, description_key: str) -> ResearchDesignRequirement:
    return ResearchDesignRequirement(code=code, description_key=description_key, status="uncertain")


def _caution(code: str, message_key: str, severity: Literal["info", "warning"] = "warning") -> ResearchDesignCaution:
    return ResearchDesignCaution(code=code, message_key=message_key, severity=severity)


def _present_features(requirements: list[ResearchDesignRequirement]) -> list[str]:
    features: list[str] = []
    for requirement in requirements:
        if requirement.status == "present":
            features.append(requirement.code)
            features.extend(requirement.variables)
    return _unique_in_order(features)


def _contains_token(value: Any, tokens: tuple[str, ...]) -> bool:
    normalized = str(value).strip().lower()
    return any(token in normalized for token in tokens)


def _unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique
