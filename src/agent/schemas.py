from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Language = Literal["en", "zh"]
PlannerSeverity = Literal["info", "warning", "error"]


@dataclass(frozen=True)
class VariableCandidate:
    """A variable the planner may recommend for a specific analytical role."""

    name: str
    role: str
    reason: str
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelRecommendation:
    """A candidate model recommendation from the planner."""

    model_id: str
    model_name: str
    priority: str
    reason: str
    requirements: dict[str, str | list[str] | bool | None] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlannerWarning:
    """A transparent warning attached to an analysis plan."""

    severity: PlannerSeverity
    code: str
    message: str
    variables: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalysisPlan:
    """Structured output for the future v0.7 Analysis Planner."""

    plan_id: str
    language: Language
    data_structure: str
    confidence: float
    summary: str
    recommended_main_model: ModelRecommendation | None = None
    baseline_models: list[ModelRecommendation] = field(default_factory=list)
    alternative_models: list[ModelRecommendation] = field(default_factory=list)
    dependent_variable_candidates: list[VariableCandidate] = field(default_factory=list)
    recommended_dependent_variable: str | None = None
    main_explanatory_candidates: list[VariableCandidate] = field(default_factory=list)
    recommended_main_explanatory_variables: list[str] = field(default_factory=list)
    numeric_controls: list[str] = field(default_factory=list)
    categorical_controls: list[str] = field(default_factory=list)
    entity_id: str | None = None
    time_id: str | None = None
    fixed_effects: dict[str, bool] = field(default_factory=dict)
    standard_errors: str | None = None
    diagnostics_to_run: list[str] = field(default_factory=list)
    warnings: list[PlannerWarning] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)
    user_confirmable_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def to_dict(plan: AnalysisPlan) -> dict[str, Any]:
    return plan.to_dict()
