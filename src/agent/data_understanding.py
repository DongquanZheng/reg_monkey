from __future__ import annotations

from typing import Any

import pandas as pd

from src.variable_roles import (
    ROLE_BINARY,
    ROLE_CATEGORICAL,
    ROLE_CODE,
    ROLE_ENTITY,
    ROLE_NUMERIC,
    ROLE_TIME,
    get_binary_variables,
    get_categorical_variables,
    get_code_identifier_variables,
    get_entity_id_variables,
    get_numeric_measure_variables,
    get_time_variables,
)


def understand_data(df: pd.DataFrame, variable_roles: dict[str, str], language: str = "en") -> dict[str, Any]:
    numeric_variables = [col for col in get_numeric_measure_variables(variable_roles) if col in df.columns]
    binary_variables = [col for col in get_binary_variables(variable_roles) if col in df.columns]
    categorical_variables = [col for col in get_categorical_variables(variable_roles) if col in df.columns]
    code_id_variables = [col for col in get_code_identifier_variables(variable_roles) if col in df.columns]
    entity_id_variables = [col for col in get_entity_id_variables(variable_roles) if col in df.columns]
    time_variables = [col for col in get_time_variables(variable_roles) if col in df.columns]

    entity_id = entity_id_variables[0] if entity_id_variables else None
    time_id = time_variables[0] if time_variables else None
    likely_panel = bool(entity_id and time_id)

    number_of_entities = int(df[entity_id].nunique(dropna=True)) if entity_id else 0
    number_of_time_periods = int(df[time_id].nunique(dropna=True)) if time_id else 0

    if likely_panel:
        data_structure = "panel"
        confidence = 0.9
    elif numeric_variables:
        data_structure = "cross_section"
        confidence = 0.65
    elif binary_variables:
        data_structure = "binary_outcome_available"
        confidence = 0.6
    else:
        data_structure = "unknown"
        confidence = 0.35

    return {
        "language": language,
        "data_structure": data_structure,
        "confidence": confidence,
        "panel": {
            "entity_id": entity_id,
            "time_id": time_id,
            "number_of_entities": number_of_entities,
            "number_of_time_periods": number_of_time_periods,
            "likely_panel": likely_panel,
        },
        "outcomes": {
            "continuous_candidates": numeric_variables,
            "binary_candidates": binary_variables,
            "binary_outcome_available": bool(binary_variables),
        },
        "variable_groups": {
            ROLE_NUMERIC: numeric_variables,
            ROLE_BINARY: binary_variables,
            ROLE_CATEGORICAL: categorical_variables,
            ROLE_CODE: code_id_variables,
            ROLE_ENTITY: entity_id_variables,
            ROLE_TIME: time_variables,
        },
    }
