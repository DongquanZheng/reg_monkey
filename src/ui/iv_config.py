from __future__ import annotations

from typing import Any

from src.i18n import get_text
from src.models.execution import ModelSpec
from src.models.iv_contract import IV_MODEL_ID
from src.variable_roles import get_binary_variables, get_numeric_measure_variables


IV_EXPERIMENTAL_MODEL_ID = IV_MODEL_ID


def iv_model_label(language: str) -> str:
    return get_text(language, "iv_minimal_experimental")


def iv_numeric_candidates(confirmed_roles: dict[str, str]) -> list[str]:
    return _unique([*get_numeric_measure_variables(confirmed_roles), *get_binary_variables(confirmed_roles)])


def build_iv_model_config(
    *,
    dependent_variable: str,
    endogenous_variable: str,
    instruments: list[str] | None = None,
    exogenous_controls: list[str] | None = None,
    variable_roles: dict[str, str] | None = None,
) -> dict[str, Any]:
    instrument_list = list(instruments or [])
    return {
        "dependent_variable": dependent_variable,
        "main_independent_variables": [],
        "numeric_control_variables": list(exogenous_controls or []),
        "categorical_control_variables": [],
        "encode_categorical_controls": False,
        "robust_standard_errors": False,
        "entity_id": "",
        "time_id": "",
        "entity_effects": False,
        "time_effects": False,
        "standard_errors": "conventional",
        "robust_cov_type": "HC3",
        "include_odds_ratios": False,
        "include_marginal_effects": False,
        "marginal_effects_type": "average",
        "endogenous_variable": endogenous_variable,
        "instrument_variable": instrument_list[0] if instrument_list else "",
        "instruments": instrument_list,
        "exogenous_controls": list(exogenous_controls or []),
        "variable_roles": dict(variable_roles or {}),
    }


def build_iv_model_spec(config: dict[str, Any]) -> ModelSpec:
    return ModelSpec.from_config(IV_EXPERIMENTAL_MODEL_ID, config)


def iv_required_fields_present(config: dict[str, Any]) -> bool:
    return bool(config.get("dependent_variable") and config.get("endogenous_variable") and config.get("instruments"))


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys([value for value in values if value]))
