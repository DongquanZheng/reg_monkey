from __future__ import annotations

from typing import Any

from src.agent.schemas import ModelRecommendation, PlannerWarning, VariableCandidate
from src.agent.text import planner_text
from src.models.registry import get_model


def recommend_models(
    data_understanding: dict[str, Any],
    variable_recommendations: dict[str, Any],
    variable_roles: dict[str, str],
    language: str = "en",
) -> dict[str, Any]:
    recommended_y = variable_recommendations.get("recommended_dependent_variable")
    y_role = _candidate_role(variable_recommendations.get("dependent_variable_candidates", []), recommended_y)
    has_panel = bool(data_understanding.get("panel", {}).get("likely_panel"))
    has_binary = bool(data_understanding.get("outcomes", {}).get("binary_outcome_available"))
    has_continuous_y = y_role == "continuous_outcome"
    has_binary_y = y_role == "binary_outcome"

    baseline_models: list[ModelRecommendation] = []
    alternative_models: list[ModelRecommendation] = []
    warnings = list(variable_recommendations.get("warnings", []))
    rationale: list[str] = []
    fixed_effects: dict[str, bool] = {}
    standard_errors: str | None = None

    if has_binary_y:
        main_model = _recommendation("logit", "main", planner_text(language, "logit_model_reason"), language)
        alternative_models.append(_recommendation("probit", "alternative", planner_text(language, "probit_model_reason"), language))
        rationale.append(planner_text(language, "binary_rationale"))
        standard_errors = "default"
    elif has_continuous_y and has_panel:
        main_model = _recommendation("panel_fe", "main", planner_text(language, "panel_model_reason"), language)
        baseline_models.append(_recommendation("ols", "baseline", planner_text(language, "ols_model_reason"), language))
        fixed_effects = {"entity": True, "time": True}
        standard_errors = "cluster_entity"
        rationale += [
            planner_text(language, "panel_rationale_observations"),
            planner_text(language, "panel_rationale_fe"),
            planner_text(language, "panel_rationale_baseline"),
        ]
    elif has_continuous_y:
        main_model = _recommendation("ols", "main", planner_text(language, "ols_model_reason"), language)
        rationale.append(planner_text(language, "ols_rationale"))
        standard_errors = "hc3"
    elif has_binary:
        main_model = _recommendation("logit", "main", planner_text(language, "logit_model_reason"), language)
        alternative_models.append(_recommendation("probit", "alternative", planner_text(language, "probit_model_reason"), language))
        rationale.append(planner_text(language, "binary_rationale"))
        standard_errors = "default"
    else:
        main_model = None
        warnings.append(
            PlannerWarning(
                severity="info",
                code="unclear_structure",
                message=planner_text(language, "warn_review_roles"),
                variables=[],
            )
        )

    if has_binary and (not main_model or main_model.model_id not in {"logit", "probit"}):
        alternative_models.append(_recommendation("logit", "alternative", planner_text(language, "logit_model_reason"), language))
        alternative_models.append(_recommendation("probit", "alternative", planner_text(language, "probit_model_reason"), language))

    return {
        "recommended_main_model": main_model,
        "baseline_models": baseline_models,
        "alternative_models": _dedupe_recommendations(alternative_models),
        "fixed_effects": fixed_effects,
        "standard_errors": standard_errors,
        "diagnostics_to_run": _diagnostics_for(main_model.model_id if main_model else ""),
        "warnings": warnings,
        "rationale": rationale,
    }


def _candidate_role(candidates: list[VariableCandidate], name: str | None) -> str | None:
    for candidate in candidates:
        if candidate.name == name:
            return candidate.role
    return None


def _recommendation(model_id: str, priority: str, reason: str, language: str) -> ModelRecommendation:
    model = get_model(model_id)
    return ModelRecommendation(
        model_id=model_id,
        model_name=model.display_name("zh" if language == "zh" else "en"),
        priority=priority,
        reason=reason,
        requirements={role: None for role in model.required_roles},
        warnings=[],
    )


def _diagnostics_for(model_id: str) -> list[str]:
    if model_id == "panel_fe":
        return ["panel_structure_check", "within_variation_check", "model_sample_cleaning_log"]
    if model_id == "ols":
        return ["vif", "model_sample_cleaning_log"]
    if model_id in {"logit", "probit"}:
        return ["binary_outcome_check", "vif", "convergence_check"]
    return ["data_exploration"]


def _dedupe_recommendations(recommendations: list[ModelRecommendation]) -> list[ModelRecommendation]:
    seen: set[str] = set()
    result: list[ModelRecommendation] = []
    for recommendation in recommendations:
        if recommendation.model_id in seen:
            continue
        seen.add(recommendation.model_id)
        result.append(recommendation)
    return result
