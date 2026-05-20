from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

import pandas as pd

from src.data_quality import build_data_quality_profile, build_missingness_profile, build_variable_quality_summaries
from src.reproducibility.serializers import to_jsonable


NO_ACTION = "no_action"
DROP_ROWS = "drop_rows"
MEAN_IMPUTE = "mean_impute"
MEDIAN_IMPUTE = "median_impute"
MODE_IMPUTE = "mode_impute"
MISSING_INDICATOR = "missing_indicator"

ANY_VARIABLE_TYPES = ["numeric", "categorical", "binary", "low_cardinality", "datetime", "id_like"]
NUMERIC_VARIABLE_TYPES = ["numeric", "binary"]
MODE_VARIABLE_TYPES = ["categorical", "binary", "low_cardinality"]


@dataclass(frozen=True)
class MissingDataStrategy:
    strategy_id: str
    display_name_key: str
    description_key: str
    allowed_variable_types: list[str] = field(default_factory=list)
    requires_user_confirmation: bool = True
    changes_data: bool = False
    parameters_schema: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass(frozen=True)
class MissingDataAction:
    variable: str
    strategy_id: str
    parameters: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    user_confirmed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass(frozen=True)
class MissingDataPlan:
    plan_id: str
    actions: list[MissingDataAction] = field(default_factory=list)
    created_from_profile: str = ""
    target_scope: str = ""
    requires_user_confirmation: bool = True
    status: str = "draft"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["actions"] = [action.to_dict() for action in self.actions]
        return to_jsonable(payload)


@dataclass(frozen=True)
class MissingDataPlanValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass(frozen=True)
class MissingDataActionResult:
    variable: str
    strategy_id: str
    rows_affected: int = 0
    values_filled: int = 0
    rows_dropped: int = 0
    indicator_variable_created: str = ""
    fill_value: Any | None = None
    message_code: str = ""

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass(frozen=True)
class MissingDataHandlingResult:
    original_row_count: int
    final_row_count: int
    actions_applied: list[MissingDataAction] = field(default_factory=list)
    action_results: list[MissingDataActionResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    log: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["actions_applied"] = [action.to_dict() for action in self.actions_applied]
        payload["action_results"] = [result.to_dict() for result in self.action_results]
        return to_jsonable(payload)


MISSING_DATA_STRATEGIES: dict[str, MissingDataStrategy] = {
    NO_ACTION: MissingDataStrategy(
        strategy_id=NO_ACTION,
        display_name_key="missing_strategy_no_action",
        description_key="missing_strategy_no_action_description",
        allowed_variable_types=list(ANY_VARIABLE_TYPES),
        requires_user_confirmation=False,
        changes_data=False,
        parameters_schema={},
    ),
    DROP_ROWS: MissingDataStrategy(
        strategy_id=DROP_ROWS,
        display_name_key="missing_strategy_drop_rows",
        description_key="missing_strategy_drop_rows_description",
        allowed_variable_types=list(ANY_VARIABLE_TYPES),
        requires_user_confirmation=True,
        changes_data=True,
        parameters_schema={"target_scope": {"type": "string", "required": True}},
    ),
    MEAN_IMPUTE: MissingDataStrategy(
        strategy_id=MEAN_IMPUTE,
        display_name_key="missing_strategy_mean_impute",
        description_key="missing_strategy_mean_impute_description",
        allowed_variable_types=list(NUMERIC_VARIABLE_TYPES),
        requires_user_confirmation=True,
        changes_data=True,
        parameters_schema={},
    ),
    MEDIAN_IMPUTE: MissingDataStrategy(
        strategy_id=MEDIAN_IMPUTE,
        display_name_key="missing_strategy_median_impute",
        description_key="missing_strategy_median_impute_description",
        allowed_variable_types=list(NUMERIC_VARIABLE_TYPES),
        requires_user_confirmation=True,
        changes_data=True,
        parameters_schema={},
    ),
    MODE_IMPUTE: MissingDataStrategy(
        strategy_id=MODE_IMPUTE,
        display_name_key="missing_strategy_mode_impute",
        description_key="missing_strategy_mode_impute_description",
        allowed_variable_types=list(MODE_VARIABLE_TYPES),
        requires_user_confirmation=True,
        changes_data=True,
        parameters_schema={"allow_high_cardinality": {"type": "boolean", "required": False}},
    ),
    MISSING_INDICATOR: MissingDataStrategy(
        strategy_id=MISSING_INDICATOR,
        display_name_key="missing_strategy_missing_indicator",
        description_key="missing_strategy_missing_indicator_description",
        allowed_variable_types=list(ANY_VARIABLE_TYPES),
        requires_user_confirmation=True,
        changes_data=True,
        parameters_schema={"indicator_suffix": {"type": "string", "required": False}},
    ),
}


def list_missing_data_strategies() -> list[MissingDataStrategy]:
    return list(MISSING_DATA_STRATEGIES.values())


def get_missing_data_strategy(strategy_id: str) -> MissingDataStrategy | None:
    return MISSING_DATA_STRATEGIES.get(str(strategy_id or ""))


def validate_missing_data_plan(
    plan: MissingDataPlan,
    data_quality_profile: Any | None = None,
    missingness_profile: Any | None = None,
    variable_summaries: list[Any] | None = None,
) -> MissingDataPlanValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    summaries = _summary_map(variable_summaries)
    available_variables = _available_variables(data_quality_profile, missingness_profile, summaries)
    seen_by_variable: dict[str, set[str]] = {}

    if not plan.plan_id:
        errors.append("missing_data_plan_missing_plan_id")
    if not plan.actions:
        warnings.append("missing_data_plan_has_no_actions")

    for action in plan.actions:
        variable = str(action.variable or "")
        strategy_id = str(action.strategy_id or "")
        strategy = get_missing_data_strategy(strategy_id)

        if not variable:
            errors.append("missing_data_action_missing_variable")
            continue
        if available_variables and variable not in available_variables:
            errors.append(f"missing_data_action_unknown_variable:{variable}")
        if strategy is None:
            errors.append(f"missing_data_action_unknown_strategy:{strategy_id}")
            continue
        strategies_for_variable = seen_by_variable.setdefault(variable, set())
        non_indicator_strategies = {item for item in strategies_for_variable | {strategy_id} if item != MISSING_INDICATOR}
        if len(non_indicator_strategies) > 1:
            errors.append(f"missing_data_action_conflicting_duplicate:{variable}")
        strategies_for_variable.add(strategy_id)

        if strategy.changes_data and strategy.requires_user_confirmation and not action.user_confirmed:
            errors.append(f"missing_data_action_requires_user_confirmation:{variable}:{strategy_id}")

        summary = summaries.get(variable, {})
        variable_type = _variable_type(summary)
        if summary and not _strategy_supports_variable_type(strategy, variable_type):
            errors.append(f"missing_data_action_incompatible_variable_type:{variable}:{strategy_id}:{variable_type}")

        if strategy_id == MODE_IMPUTE and summary.get("is_high_cardinality") and not action.parameters.get("allow_high_cardinality"):
            warnings.append(f"missing_data_mode_high_cardinality_variable:{variable}")
        if strategy_id == DROP_ROWS and not (plan.target_scope or action.parameters.get("target_scope")):
            errors.append(f"missing_data_drop_rows_requires_target_scope:{variable}")

        _validate_parameters(action, strategy, errors)

    requires_confirmation = any(
        (get_missing_data_strategy(action.strategy_id) and get_missing_data_strategy(action.strategy_id).changes_data)
        for action in plan.actions
    )
    if requires_confirmation and not plan.requires_user_confirmation:
        errors.append("missing_data_plan_requires_user_confirmation")

    return MissingDataPlanValidationResult(is_valid=not errors, errors=errors, warnings=warnings)


def apply_missing_data_plan(
    df: pd.DataFrame,
    plan: MissingDataPlan,
    variable_summaries: list[Any] | None = None,
) -> tuple[pd.DataFrame, MissingDataHandlingResult]:
    """Apply a user-confirmed missing-data plan without mutating the input frame."""
    working = df.copy(deep=True)
    summaries = variable_summaries if variable_summaries is not None else build_variable_quality_summaries(df)
    validation = validate_missing_data_plan(
        plan,
        data_quality_profile=build_data_quality_profile(df),
        missingness_profile=build_missingness_profile(df),
        variable_summaries=summaries,
    )
    if not validation.is_valid:
        return working, MissingDataHandlingResult(
            original_row_count=int(len(df)),
            final_row_count=int(len(df)),
            actions_applied=[],
            action_results=[],
            warnings=list(validation.errors) + list(validation.warnings),
            log=[
                {
                    "message_code": "missing_data_plan_validation_failed",
                    "errors": list(validation.errors),
                    "warnings": list(validation.warnings),
                }
            ],
        )

    action_results: list[MissingDataActionResult] = []
    actions_applied: list[MissingDataAction] = []
    warnings = list(validation.warnings)
    for action in _ordered_actions(plan.actions):
        strategy_id = str(action.strategy_id or "")
        variable = str(action.variable or "")
        if strategy_id == NO_ACTION:
            result = MissingDataActionResult(variable=variable, strategy_id=strategy_id, message_code="missing_data_no_action_applied")
        elif strategy_id == MISSING_INDICATOR:
            result = _apply_missing_indicator(working, variable, action)
        elif strategy_id == DROP_ROWS:
            result = _apply_drop_rows(working, variable)
        elif strategy_id in {MEAN_IMPUTE, MEDIAN_IMPUTE, MODE_IMPUTE}:
            result = _apply_imputation(working, variable, strategy_id)
        else:
            result = MissingDataActionResult(
                variable=variable,
                strategy_id=strategy_id,
                message_code="missing_data_unknown_strategy_skipped",
            )
            warnings.append(f"missing_data_unknown_strategy_skipped:{variable}:{strategy_id}")
        actions_applied.append(action)
        action_results.append(result)

    log = [result.to_dict() for result in action_results]
    return working, MissingDataHandlingResult(
        original_row_count=int(len(df)),
        final_row_count=int(len(working)),
        actions_applied=actions_applied,
        action_results=action_results,
        warnings=warnings,
        log=log,
    )


def missing_data_strategy_to_dict(strategy: MissingDataStrategy) -> dict[str, Any]:
    return strategy.to_dict()


def missing_data_plan_to_dict(plan: MissingDataPlan) -> dict[str, Any]:
    return plan.to_dict()


def missing_data_validation_to_dict(validation: MissingDataPlanValidationResult) -> dict[str, Any]:
    return validation.to_dict()


def missing_data_handling_result_to_dict(result: MissingDataHandlingResult) -> dict[str, Any]:
    return result.to_dict()


def _validate_parameters(action: MissingDataAction, strategy: MissingDataStrategy, errors: list[str]) -> None:
    schema = strategy.parameters_schema or {}
    parameters = dict(action.parameters or {})
    allowed_keys = set(schema.keys())
    unknown = sorted(set(parameters.keys()) - allowed_keys)
    for key in unknown:
        errors.append(f"missing_data_action_unknown_parameter:{action.variable}:{strategy.strategy_id}:{key}")
    for key, definition in schema.items():
        if definition.get("required") and key not in parameters:
            if not (strategy.strategy_id == DROP_ROWS and key == "target_scope"):
                errors.append(f"missing_data_action_missing_parameter:{action.variable}:{strategy.strategy_id}:{key}")
        if key in parameters and not _parameter_matches_type(parameters[key], str(definition.get("type") or "")):
            errors.append(f"missing_data_action_invalid_parameter_type:{action.variable}:{strategy.strategy_id}:{key}")


def _parameter_matches_type(value: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    return True


def _ordered_actions(actions: list[MissingDataAction]) -> list[MissingDataAction]:
    return sorted(
        list(actions),
        key=lambda action: (0 if action.strategy_id == MISSING_INDICATOR else 1),
    )


def _apply_missing_indicator(df: pd.DataFrame, variable: str, action: MissingDataAction) -> MissingDataActionResult:
    suffix = str(action.parameters.get("indicator_suffix") or "_missing")
    indicator = _unique_indicator_name(df, f"{variable}{suffix}")
    missing_mask = df[variable].isna()
    df[indicator] = missing_mask.astype(int)
    return MissingDataActionResult(
        variable=variable,
        strategy_id=MISSING_INDICATOR,
        rows_affected=int(missing_mask.sum()),
        indicator_variable_created=indicator,
        message_code="missing_data_indicator_created",
    )


def _apply_drop_rows(df: pd.DataFrame, variable: str) -> MissingDataActionResult:
    missing_mask = df[variable].isna()
    rows_dropped = int(missing_mask.sum())
    if rows_dropped:
        df.drop(index=df.index[missing_mask], inplace=True)
    return MissingDataActionResult(
        variable=variable,
        strategy_id=DROP_ROWS,
        rows_affected=rows_dropped,
        rows_dropped=rows_dropped,
        message_code="missing_data_rows_dropped",
    )


def _apply_imputation(df: pd.DataFrame, variable: str, strategy_id: str) -> MissingDataActionResult:
    missing_mask = df[variable].isna()
    values_filled = int(missing_mask.sum())
    fill_value = _fill_value(df[variable], strategy_id)
    if values_filled and not pd.isna(fill_value):
        df.loc[missing_mask, variable] = fill_value
        message_code = "missing_data_values_filled"
    else:
        message_code = "missing_data_fill_value_unavailable" if values_filled else "missing_data_no_missing_values"
    return MissingDataActionResult(
        variable=variable,
        strategy_id=strategy_id,
        rows_affected=values_filled,
        values_filled=values_filled if not pd.isna(fill_value) else 0,
        fill_value=fill_value,
        message_code=message_code,
    )


def _fill_value(series: pd.Series, strategy_id: str) -> Any:
    non_missing = series.dropna()
    if non_missing.empty:
        return None
    if strategy_id == MEAN_IMPUTE:
        return float(pd.to_numeric(non_missing).mean())
    if strategy_id == MEDIAN_IMPUTE:
        return float(pd.to_numeric(non_missing).median())
    if strategy_id == MODE_IMPUTE:
        counts = non_missing.value_counts(dropna=True)
        max_count = counts.max()
        candidates = [value for value, count in counts.items() if count == max_count]
        return sorted(candidates, key=lambda value: str(value))[0]
    return None


def _unique_indicator_name(df: pd.DataFrame, base_name: str) -> str:
    if base_name not in df.columns:
        return base_name
    counter = 2
    while f"{base_name}_{counter}" in df.columns:
        counter += 1
    return f"{base_name}_{counter}"


def _summary_map(variable_summaries: list[Any] | None) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in variable_summaries or []:
        payload = _as_dict(item)
        variable = str(payload.get("variable") or "")
        if variable:
            result[variable] = payload
    return result


def _available_variables(data_quality_profile: Any | None, missingness_profile: Any | None, summaries: dict[str, dict[str, Any]]) -> set[str]:
    variables = set(summaries.keys())
    quality = _as_dict(data_quality_profile)
    missingness = _as_dict(missingness_profile)
    for key in [
        "numeric_columns",
        "categorical_columns",
        "binary_columns",
        "datetime_like_columns",
        "id_like_columns",
        "high_cardinality_columns",
        "constant_columns",
        "near_constant_columns",
        "text_numeric_columns",
        "mixed_type_columns",
        "columns_with_any_missing",
    ]:
        variables.update(str(item) for item in quality.get(key, []) if str(item))
        variables.update(str(item) for item in missingness.get(key, []) if str(item))
    for item in missingness.get("missing_by_variable", []) or []:
        payload = _as_dict(item)
        if payload.get("variable"):
            variables.add(str(payload["variable"]))
    return variables


def _variable_type(summary: dict[str, Any]) -> str:
    dtype = str(summary.get("dtype") or "").lower()
    hint = str(summary.get("inferred_role_hint") or "").lower()
    unique_count = int(summary.get("unique_count") or 0)
    if summary.get("is_binary_like"):
        return "binary"
    if summary.get("is_datetime_like") or "datetime" in dtype:
        return "datetime"
    if summary.get("is_id_like"):
        return "id_like"
    if any(token in dtype for token in ["int", "float", "number"]) or "numeric" in hint:
        return "numeric"
    if unique_count and unique_count <= 20:
        return "low_cardinality"
    return "categorical"


def _strategy_supports_variable_type(strategy: MissingDataStrategy, variable_type: str) -> bool:
    allowed = set(strategy.allowed_variable_types or [])
    if variable_type in allowed:
        return True
    if variable_type == "low_cardinality" and "categorical" in allowed:
        return True
    if variable_type == "binary" and ("numeric" in allowed or "categorical" in allowed):
        return True
    return False


def _as_dict(value: Any | None) -> dict[str, Any]:
    if value is None:
        return {}
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    return {}
