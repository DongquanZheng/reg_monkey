from __future__ import annotations

from typing import Any

from src.i18n import get_text
from src.models.execution import ModelSpec
from src.models.psm_contract import PSM_MODEL_ID
from src.variable_roles import get_binary_variables, get_numeric_measure_variables


PSM_EXPERIMENTAL_MODEL_ID = PSM_MODEL_ID
PSM_UI_ESTIMANDS = ["ATT"]
PSM_UI_MATCHING_METHODS = ["nearest_neighbor"]


def psm_model_label(language: str) -> str:
    return get_text(language, "psm_minimal_experimental")


def psm_outcome_candidates(confirmed_roles: dict[str, str]) -> list[str]:
    return _unique(get_numeric_measure_variables(confirmed_roles))


def psm_treatment_candidates(confirmed_roles: dict[str, str]) -> list[str]:
    return _unique(get_binary_variables(confirmed_roles))


def psm_matching_covariate_candidates(confirmed_roles: dict[str, str]) -> list[str]:
    return _unique([*get_numeric_measure_variables(confirmed_roles), *get_binary_variables(confirmed_roles)])


def build_psm_model_config(
    *,
    dependent_variable: str,
    treatment_variable: str,
    matching_covariates: list[str] | None = None,
    psm_estimand: str = "ATT",
    matching_method: str = "nearest_neighbor",
    caliper: float | None = None,
    variable_roles: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "dependent_variable": dependent_variable,
        "main_independent_variables": [],
        "numeric_control_variables": [],
        "categorical_control_variables": [],
        "encode_categorical_controls": False,
        "robust_standard_errors": False,
        "entity_id": "",
        "time_id": "",
        "entity_effects": False,
        "time_effects": False,
        "standard_errors": "not_estimated",
        "robust_cov_type": "HC3",
        "include_odds_ratios": False,
        "include_marginal_effects": False,
        "marginal_effects_type": "average",
        "treatment_variable": treatment_variable,
        "matching_covariates": list(matching_covariates or []),
        "psm_estimand": psm_estimand,
        "matching_method": matching_method,
        "caliper": caliper,
        "variable_roles": dict(variable_roles or {}),
    }


def build_psm_model_spec(config: dict[str, Any]) -> ModelSpec:
    return ModelSpec.from_config(PSM_EXPERIMENTAL_MODEL_ID, config)


def psm_required_fields_present(config: dict[str, Any]) -> bool:
    return bool(config.get("dependent_variable") and config.get("treatment_variable") and config.get("matching_covariates"))


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys([value for value in values if value]))
