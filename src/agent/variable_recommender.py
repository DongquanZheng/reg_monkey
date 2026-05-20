from __future__ import annotations

from typing import Any

import pandas as pd

from src.agent.schemas import PlannerWarning, VariableCandidate
from src.agent.text import planner_text


BINARY_OUTCOME_TOKENS = [
    "adopt",
    "churn",
    "convert",
    "conversion",
    "default",
    "export",
    "purchase",
    "retention",
    "subscribe",
    "success",
    "yes",
]

OUTCOME_TOKENS = [
    "outcome",
    "dependent",
    "pollution",
    "intensity",
    "performance",
    "productivity",
    "risk",
    "revenue",
    "profit",
    "sales_growth",
    "roa",
    "growth",
]
MAIN_X_TOKENS = ["digital", "index", "treatment", "policy", "ai", "score", "exposure", "shock", "risk"]
CONTROL_TOKENS = ["leverage", "size", "age", "employees", "employee", "rd_intensity", "assets", "revenue", "growth", "debt"]


def recommend_variables(
    df: pd.DataFrame,
    variable_roles: dict[str, str],
    data_understanding: dict[str, Any],
    language: str = "en",
) -> dict[str, Any]:
    groups = data_understanding["variable_groups"]
    numeric_variables = list(groups.get("Numeric measure", []))
    binary_variables = list(groups.get("Binary variable", []))
    categorical_variables = list(groups.get("Categorical variable", []))

    dependent_candidates = _dependent_candidates(numeric_variables, binary_variables, language)
    recommended_y = _recommended_dependent_variable(dependent_candidates)

    main_candidates = _main_explanatory_candidates(numeric_variables, recommended_y, language)
    recommended_main_x = [candidate.name for candidate in main_candidates if candidate.confidence >= 0.55][:2]
    if not recommended_main_x and main_candidates:
        recommended_main_x = [main_candidates[0].name]

    numeric_controls = _numeric_controls(numeric_variables, recommended_y, recommended_main_x)
    categorical_controls, warnings = _categorical_controls(df, categorical_variables, language)

    return {
        "dependent_variable_candidates": dependent_candidates,
        "recommended_dependent_variable": recommended_y,
        "main_explanatory_candidates": main_candidates,
        "recommended_main_explanatory_variables": recommended_main_x,
        "numeric_controls": numeric_controls,
        "categorical_controls": categorical_controls,
        "warnings": warnings,
    }


def _dependent_candidates(numeric_variables: list[str], binary_variables: list[str], language: str) -> list[VariableCandidate]:
    candidates: list[VariableCandidate] = []
    for name in numeric_variables:
        score = 0.55 + _token_score(name, OUTCOME_TOKENS, 0.35)
        candidates.append(
            VariableCandidate(
                name=name,
                role="continuous_outcome",
                reason=planner_text(language, "reason_numeric_y"),
                confidence=min(score, 0.95),
            )
        )
    for name in binary_variables:
        treatment_penalty = 0.15 if _has_token(name, ["treated", "treatment", "post", "policy"]) else 0.0
        score = 0.55 + _token_score(name, BINARY_OUTCOME_TOKENS, 0.3) - treatment_penalty
        candidates.append(
            VariableCandidate(
                name=name,
                role="binary_outcome",
                reason=planner_text(language, "reason_binary_y"),
                confidence=min(score, 0.95),
            )
        )
    return sorted(candidates, key=lambda candidate: candidate.confidence, reverse=True)


def _recommended_dependent_variable(candidates: list[VariableCandidate]) -> str | None:
    if not candidates:
        return None
    top = candidates[0]
    if top.role == "binary_outcome" and top.confidence >= 0.8:
        return top.name
    continuous = [candidate for candidate in candidates if candidate.role == "continuous_outcome"]
    binary = [candidate for candidate in candidates if candidate.role == "binary_outcome" and candidate.confidence >= 0.8]
    strong_continuous = [candidate for candidate in continuous if candidate.confidence >= 0.8]
    if binary and not strong_continuous:
        return binary[0].name
    if continuous:
        return continuous[0].name
    return top.name


def _main_explanatory_candidates(numeric_variables: list[str], recommended_y: str | None, language: str) -> list[VariableCandidate]:
    candidates: list[VariableCandidate] = []
    for name in numeric_variables:
        if name == recommended_y:
            continue
        control_penalty = 0.25 if _has_token(name, CONTROL_TOKENS) else 0.0
        score = 0.45 + _token_score(name, MAIN_X_TOKENS, 0.4) - control_penalty
        candidates.append(
            VariableCandidate(
                name=name,
                role="main_explanatory",
                reason=planner_text(language, "reason_main_x"),
                confidence=max(0.15, min(score, 0.95)),
            )
        )
    return sorted(candidates, key=lambda candidate: candidate.confidence, reverse=True)


def _numeric_controls(numeric_variables: list[str], recommended_y: str | None, main_x: list[str]) -> list[str]:
    excluded = set(main_x)
    if recommended_y:
        excluded.add(recommended_y)
    controls = [name for name in numeric_variables if name not in excluded and _has_token(name, CONTROL_TOKENS)]
    controls += [name for name in numeric_variables if name not in excluded and name not in controls]
    return controls[:8]


def _categorical_controls(df: pd.DataFrame, categorical_variables: list[str], language: str) -> tuple[list[str], list[PlannerWarning]]:
    controls: list[str] = []
    high_cardinality: list[str] = []
    for name in categorical_variables:
        unique_count = int(df[name].nunique(dropna=True)) if name in df.columns else 0
        if unique_count <= 20:
            controls.append(name)
        else:
            high_cardinality.append(name)
    warnings = []
    if high_cardinality:
        warnings.append(
            PlannerWarning(
                severity="warning",
                code="high_cardinality_categorical",
                message=planner_text(language, "warn_dummy_explosion"),
                variables=high_cardinality,
            )
        )
    return controls, warnings


def _token_score(name: str, tokens: list[str], weight: float) -> float:
    return weight if _has_token(name, tokens) else 0.0


def _has_token(name: str, tokens: list[str]) -> bool:
    lower = name.lower()
    return any(token in lower for token in tokens)
