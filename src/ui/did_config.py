from __future__ import annotations

from typing import Any

from src.i18n import get_text
from src.models.execution import ModelSpec
from src.variable_roles import (
    get_binary_variables,
    get_categorical_variables,
    get_code_identifier_variables,
    get_entity_id_variables,
    get_numeric_measure_variables,
)


DID_EXPERIMENTAL_MODEL_ID = "did"


def did_model_label(language: str) -> str:
    return get_text(language, "did_minimal_experimental")


def did_indicator_candidates(confirmed_roles: dict[str, str]) -> list[str]:
    return _unique([*get_binary_variables(confirmed_roles), *get_numeric_measure_variables(confirmed_roles)])


def did_group_candidates(confirmed_roles: dict[str, str]) -> list[str]:
    return _unique(
        [
            *get_entity_id_variables(confirmed_roles),
            *get_code_identifier_variables(confirmed_roles),
            *get_categorical_variables(confirmed_roles),
            *get_binary_variables(confirmed_roles),
        ]
    )


def build_did_model_config(
    *,
    dependent_variable: str,
    treatment_variable: str,
    post_variable: str,
    numeric_control_variables: list[str] | None = None,
    categorical_control_variables: list[str] | None = None,
    group_variable: str = "",
    cluster_variable: str = "",
    encode_categorical_controls: bool = False,
    variable_roles: dict[str, str] | None = None,
) -> dict[str, Any]:
    standard_errors = "cluster" if cluster_variable else "hc3"
    return {
        "dependent_variable": dependent_variable,
        "main_independent_variables": [],
        "numeric_control_variables": list(numeric_control_variables or []),
        "categorical_control_variables": list(categorical_control_variables or []),
        "encode_categorical_controls": encode_categorical_controls,
        "robust_standard_errors": standard_errors != "conventional",
        "entity_id": "",
        "time_id": "",
        "entity_effects": False,
        "time_effects": False,
        "standard_errors": standard_errors,
        "robust_cov_type": "HC3",
        "include_odds_ratios": False,
        "include_marginal_effects": False,
        "marginal_effects_type": "average",
        "treatment_variable": treatment_variable,
        "post_variable": post_variable,
        "group_variable": group_variable,
        "cluster_variable": cluster_variable,
        "variable_roles": dict(variable_roles or {}),
    }


def build_did_model_spec(config: dict[str, Any]) -> ModelSpec:
    return ModelSpec.from_config(DID_EXPERIMENTAL_MODEL_ID, config)


def did_required_fields_present(config: dict[str, Any]) -> bool:
    return bool(config.get("dependent_variable") and config.get("treatment_variable") and config.get("post_variable"))


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys([value for value in values if value]))
