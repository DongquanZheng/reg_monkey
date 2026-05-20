from __future__ import annotations

from uuid import uuid4

import pandas as pd

from src.agent.data_understanding import understand_data
from src.agent.model_recommender import recommend_models
from src.agent.schemas import AnalysisPlan, PlannerWarning
from src.agent.text import planner_text
from src.agent.variable_recommender import recommend_variables


def generate_analysis_plan(df: pd.DataFrame, variable_roles: dict[str, str], language: str = "en") -> AnalysisPlan:
    lang = "zh" if language == "zh" else "en"
    data_info = understand_data(df, variable_roles, lang)
    variable_recs = recommend_variables(df, variable_roles, data_info, lang)
    model_recs = recommend_models(data_info, variable_recs, variable_roles, lang)

    warnings = list(model_recs.get("warnings", []))
    warnings.append(
        PlannerWarning(
            severity="info",
            code="review_required",
            message=planner_text(lang, "warn_review_roles"),
            variables=[],
        )
    )

    return AnalysisPlan(
        plan_id=f"plan_{uuid4().hex[:10]}",
        language=lang,
        data_structure=data_info["data_structure"],
        confidence=float(data_info["confidence"]),
        summary=_summary_for(data_info, variable_recs, lang),
        recommended_main_model=model_recs.get("recommended_main_model"),
        baseline_models=model_recs.get("baseline_models", []),
        alternative_models=model_recs.get("alternative_models", []),
        dependent_variable_candidates=variable_recs["dependent_variable_candidates"],
        recommended_dependent_variable=variable_recs["recommended_dependent_variable"],
        main_explanatory_candidates=variable_recs["main_explanatory_candidates"],
        recommended_main_explanatory_variables=variable_recs["recommended_main_explanatory_variables"],
        numeric_controls=variable_recs["numeric_controls"],
        categorical_controls=variable_recs["categorical_controls"],
        entity_id=data_info["panel"]["entity_id"],
        time_id=data_info["panel"]["time_id"],
        fixed_effects=model_recs.get("fixed_effects", {}),
        standard_errors=model_recs.get("standard_errors"),
        diagnostics_to_run=model_recs.get("diagnostics_to_run", []),
        warnings=warnings,
        rationale=model_recs.get("rationale", []),
        user_confirmable_actions=_actions_for(model_recs.get("recommended_main_model"), lang),
    )


def _summary_for(data_info: dict, variable_recs: dict, language: str) -> str:
    recommended_y = variable_recs.get("recommended_dependent_variable")
    candidate_role = None
    for candidate in variable_recs.get("dependent_variable_candidates", []):
        if candidate.name == recommended_y:
            candidate_role = candidate.role
            break
    if candidate_role == "binary_outcome":
        return planner_text(language, "binary_summary")
    if data_info["data_structure"] == "panel":
        return planner_text(language, "panel_summary")
    if data_info["data_structure"] == "cross_section":
        return planner_text(language, "cross_section_summary")
    if data_info["outcomes"].get("binary_outcome_available"):
        return planner_text(language, "binary_summary")
    return planner_text(language, "unknown_summary")


def _actions_for(main_model, language: str) -> list[str]:
    actions = [planner_text(language, "action_apply_recommended")]
    if main_model is not None:
        actions.append(planner_text(language, "action_apply_main"))
    if main_model is not None and main_model.model_id == "panel_fe":
        actions.append(planner_text(language, "action_run_baseline"))
    actions.append(planner_text(language, "action_run_main"))
    return actions
