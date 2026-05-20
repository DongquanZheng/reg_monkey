from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from src.models.base import ModelDefinition
from src.utils import significance_stars


def _main_x_cols(config: dict[str, Any]) -> list[str]:
    return list(config.get("main_independent_variables") or config.get("independent_variables") or [])


def _numeric_control_cols(config: dict[str, Any]) -> list[str]:
    return list(config.get("numeric_control_variables") or [])


def _predictor_cols(config: dict[str, Any]) -> list[str]:
    return _main_x_cols(config) + _numeric_control_cols(config)


def _entity_col(config: dict[str, Any]) -> str:
    return str(config.get("entity_id") or "")


def _time_col(config: dict[str, Any]) -> str:
    return str(config.get("time_id") or "")


def _entity_effects(config: dict[str, Any]) -> bool:
    return bool(config.get("entity_effects", True))


def _time_effects(config: dict[str, Any]) -> bool:
    return bool(config.get("time_effects", True))


def check_panel_structure(df: pd.DataFrame, entity_col: str, time_col: str) -> dict[str, Any]:
    if not entity_col or not time_col or entity_col not in df.columns or time_col not in df.columns:
        return {
            "observations": int(len(df)),
            "entities": 0,
            "time_periods": 0,
            "duplicate_entity_time_rows": 0,
            "missing_entity_ids": int(len(df)),
            "missing_time_ids": int(len(df)),
            "balanced_panel": False,
            "min_observations_per_entity": 0,
            "max_observations_per_entity": 0,
            "average_observations_per_entity": 0.0,
            "singleton_entities": 0,
        }

    panel_keys = df[[entity_col, time_col]]
    entity_counts = df.groupby(entity_col, dropna=True)[time_col].count()
    counts = entity_counts.astype(int)
    time_periods = int(df[time_col].nunique(dropna=True))
    entities = int(df[entity_col].nunique(dropna=True))
    min_obs = int(counts.min()) if not counts.empty else 0
    max_obs = int(counts.max()) if not counts.empty else 0
    avg_obs = round(float(counts.mean()), 2) if not counts.empty else 0.0

    return {
        "observations": int(len(df)),
        "entities": entities,
        "time_periods": time_periods,
        "duplicate_entity_time_rows": int(panel_keys.duplicated().sum()),
        "missing_entity_ids": int(df[entity_col].isna().sum()),
        "missing_time_ids": int(df[time_col].isna().sum()),
        "balanced_panel": bool(entities > 0 and time_periods > 0 and min_obs == max_obs == time_periods),
        "min_observations_per_entity": min_obs,
        "max_observations_per_entity": max_obs,
        "average_observations_per_entity": avg_obs,
        "singleton_entities": int((counts <= 1).sum()) if not counts.empty else 0,
    }


def detect_within_variation(
    df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    variables: list[str],
) -> dict[str, list[str]]:
    no_within: list[str] = []
    low_within: list[str] = []
    time_only: list[str] = []

    for variable in variables:
        if variable not in df.columns:
            continue
        entity_nunique = df.groupby(entity_col, dropna=True)[variable].nunique(dropna=True)
        if not entity_nunique.empty and (entity_nunique <= 1).all():
            no_within.append(variable)
        else:
            share_changing = float((entity_nunique > 1).mean()) if not entity_nunique.empty else 0.0
            if 0 < share_changing < 0.20:
                low_within.append(variable)

        time_nunique = df.groupby(time_col, dropna=True)[variable].nunique(dropna=True)
        if not time_nunique.empty and (time_nunique <= 1).all():
            time_only.append(variable)

    return {
        "no_within_entity_variation": no_within,
        "low_within_entity_variation": low_within,
        "time_only_variation": time_only,
    }


def _prepare_panel_dataframe(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, str, list[str], str, str, dict[str, Any]]:
    y_col = config["dependent_variable"]
    x_cols = _predictor_cols(config)
    entity_col = _entity_col(config)
    time_col = _time_col(config)
    selected_columns = list(dict.fromkeys([y_col] + x_cols + [entity_col, time_col]))

    original_rows = len(df)
    model_df = df[selected_columns].dropna().copy()
    cleaning_log = {
        "original_row_count": original_rows,
        "final_row_count": len(model_df),
        "dropped_row_count": original_rows - len(model_df),
        "dropped_row_percentage": round(((original_rows - len(model_df)) / original_rows * 100), 2) if original_rows else 0.0,
    }
    return model_df, y_col, x_cols, entity_col, time_col, cleaning_log


def prepare_panel_fe_dataframe(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, str, list[str], str, str, dict[str, Any]]:
    return _prepare_panel_dataframe(df, config)


def validate_panel_fe_inputs(df: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    y_col = config.get("dependent_variable")
    x_cols = _predictor_cols(config)
    entity_col = _entity_col(config)
    time_col = _time_col(config)

    if not y_col:
        errors.append("Please select a dependent variable.")
        return errors
    if not x_cols:
        errors.append("Please select at least one main independent variable.")
        return errors
    if not entity_col:
        errors.append("Please select an entity ID.")
        return errors
    if not time_col:
        errors.append("Please select a time ID.")
        return errors
    if not _entity_effects(config) and not _time_effects(config):
        errors.append("Please select at least one fixed effect.")
        return errors

    required = [y_col] + x_cols + [entity_col, time_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        errors.append("Selected variable(s) not found: " + ", ".join(missing))
        return errors

    if not pd.api.types.is_numeric_dtype(df[y_col]):
        errors.append("Panel Fixed Effects requires a numeric dependent variable.")
    non_numeric_x = [col for col in x_cols if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric_x:
        errors.append("Panel Fixed Effects requires numeric explanatory variables: " + ", ".join(non_numeric_x))
    if y_col in x_cols:
        errors.append("The dependent variable cannot also be used as an explanatory variable.")
    if entity_col in x_cols or time_col in x_cols:
        errors.append("Entity ID and time ID should not be used as ordinary explanatory variables.")

    model_df = df[required].dropna().copy()
    structure = check_panel_structure(model_df, entity_col, time_col)
    if structure["duplicate_entity_time_rows"] > 0:
        errors.append("Duplicate entity-time observations detected. Panel Fixed Effects requires each entity-time combination to be unique.")
    if structure["entities"] <= 1:
        errors.append("Panel Fixed Effects requires more than one entity.")
    if structure["time_periods"] <= 1:
        errors.append("Panel Fixed Effects requires more than one time period.")
    if len(model_df) <= len(x_cols) + structure["entities"]:
        errors.append("Too few observations remain after dropping missing values for the selected panel model.")

    constant_x = [col for col in x_cols if col in model_df.columns and model_df[col].nunique(dropna=True) <= 1]
    if constant_x:
        errors.append("Panel Fixed Effects cannot use constant-only explanatory variable(s): " + ", ".join(constant_x))

    valid_se = {"conventional", "robust", "cluster_entity"}
    se = str(config.get("standard_errors") or "cluster_entity")
    if se not in valid_se:
        errors.append("Panel Fixed Effects supports conventional, robust, or cluster-by-entity standard errors.")

    return errors


def _covariance_config(standard_errors: str) -> dict[str, Any]:
    se = (standard_errors or "cluster_entity").lower()
    if se == "conventional":
        return {"cov_type": "unadjusted"}
    if se == "robust":
        return {"cov_type": "robust"}
    return {"cov_type": "clustered", "cluster_entity": True}


def _panel_fe_failure_message(error: Exception) -> str:
    text = str(error)
    lower = text.lower()
    if "too few observations remain after dropping missing values" in lower:
        return "Panel Fixed Effects could not be estimated because too few observations remain after dropping missing values."
    if "absorbed" in lower or "fully absorbed" in lower:
        return "Panel Fixed Effects could not estimate one or more selected variables because they may be absorbed by fixed effects."
    if "rank" in lower or "exog" in lower or "collinear" in lower:
        return "Panel Fixed Effects could not be estimated because the selected variables may have insufficient independent variation after fixed effects are applied."
    return text


def fit_panel_fe(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    errors = validate_panel_fe_inputs(df, config)
    if errors:
        raise ValueError(errors[0])

    model_df, y_col, x_cols, entity_col, time_col, cleaning_log = _prepare_panel_dataframe(df, config)
    panel_structure = check_panel_structure(model_df, entity_col, time_col)
    variation = detect_within_variation(model_df, entity_col, time_col, x_cols)

    indexed = model_df.set_index([entity_col, time_col]).sort_index()
    y = indexed[y_col].astype(float)
    x = indexed[x_cols].astype(float)

    model = PanelOLS(
        y,
        x,
        entity_effects=_entity_effects(config),
        time_effects=_time_effects(config),
        drop_absorbed=True,
        check_rank=False,
    )
    try:
        fitted = model.fit(**_covariance_config(config.get("standard_errors", "cluster_entity")))
    except Exception as exc:
        raise ValueError(_panel_fe_failure_message(exc)) from exc

    conf_int = fitted.conf_int()
    regression_table = pd.DataFrame(
        {
            "variable": fitted.params.index.astype(str),
            "coefficient": fitted.params.values,
            "std_error": fitted.std_errors.reindex(fitted.params.index).values,
            "t_value": fitted.tstats.reindex(fitted.params.index).values,
            "p_value": fitted.pvalues.reindex(fitted.params.index).values,
            "conf_int_low": conf_int.iloc[:, 0].reindex(fitted.params.index).values,
            "conf_int_high": conf_int.iloc[:, 1].reindex(fitted.params.index).values,
        }
    )
    regression_table["significance"] = regression_table["p_value"].apply(significance_stars)

    model_summary = {
        "model_type": "panel_fe",
        "dependent_variable": y_col,
        "independent_variables": list(regression_table["variable"].astype(str)),
        "main_independent_variables": _main_x_cols(config),
        "numeric_control_variables": _numeric_control_cols(config),
        "categorical_control_variables": [],
        "encoded_categorical_controls": [],
        "dummy_variables": [],
        "reference_categories": {},
        "entity_id": entity_col,
        "time_id": time_col,
        "entity_effects": _entity_effects(config),
        "time_effects": _time_effects(config),
        "standard_errors": config.get("standard_errors", "cluster_entity"),
        "robust_standard_errors": config.get("standard_errors") in {"robust", "cluster_entity"},
        "n_obs": int(fitted.nobs),
        "entities": panel_structure["entities"],
        "time_periods": panel_structure["time_periods"],
        "r_squared_within": float(fitted.rsquared_within) if fitted.rsquared_within is not None else None,
        "r_squared_between": float(fitted.rsquared_between) if fitted.rsquared_between is not None else None,
        "r_squared_overall": float(fitted.rsquared_overall) if fitted.rsquared_overall is not None else None,
        "f_statistic": float(fitted.f_statistic.stat) if fitted.f_statistic is not None else None,
        "f_pvalue": float(fitted.f_statistic.pval) if fitted.f_statistic is not None else None,
        "panel_structure": panel_structure,
        "within_variation": variation,
    }

    return {
        "fitted_model": fitted,
        "regression_table": regression_table,
        "model_summary": model_summary,
        "cleaned_df": model_df,
        "cleaning_log": cleaning_log,
        "encoding_info": {
            "enabled": False,
            "selected_categorical_controls": [],
            "encoded_categorical_controls": [],
            "reference_categories": {},
            "dummy_variables": [],
            "ignored_categorical_controls": [],
            "high_cardinality_warnings": [],
        },
        "panel_structure": panel_structure,
        "warnings": [],
    }


def diagnose_panel_fe(
    df_original: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    config: dict[str, Any],
    fit_result: dict[str, Any],
) -> dict[str, Any]:
    warnings = list(fit_result.get("warnings", []))
    summary = fit_result["model_summary"]
    structure = summary.get("panel_structure") or check_panel_structure(
        df_cleaned, summary.get("entity_id", ""), summary.get("time_id", "")
    )
    variation = summary.get("within_variation") or {}

    if not structure.get("balanced_panel", False):
        warnings.append("Unbalanced panel detected: some entities are missing observations in some time periods.")
    if structure.get("singleton_entities", 0) > 0:
        warnings.append(f"Singleton entities detected: {structure['singleton_entities']} entity/entities have only one observation.")
    if structure.get("duplicate_entity_time_rows", 0) > 0:
        warnings.append("Duplicate entity-time observations detected. Panel Fixed Effects requires each entity-time combination to be unique.")

    original_rows = len(df_original)
    n_obs = len(df_cleaned)
    if original_rows:
        dropped_pct = (original_rows - n_obs) / original_rows * 100
        if dropped_pct > 30:
            warnings.append(f"Missing values reduced the model sample by {dropped_pct:.1f}%, which may bias results.")

    no_within = variation.get("no_within_entity_variation", [])
    low_within = variation.get("low_within_entity_variation", [])
    time_only = variation.get("time_only_variation", [])
    if summary.get("entity_effects") and no_within:
        warnings.append("Variables may be absorbed by entity fixed effects because they do not vary within entities: " + ", ".join(no_within) + ".")
    if summary.get("entity_effects") and low_within:
        warnings.append("Variables have little within-entity variation and may be imprecisely estimated: " + ", ".join(low_within) + ".")
    if summary.get("time_effects") and time_only:
        warnings.append("Variables may be absorbed by time fixed effects because they vary only by time: " + ", ".join(time_only) + ".")

    x_cols = list(summary.get("independent_variables", []))
    if x_cols and n_obs <= len(x_cols) * 10:
        warnings.append("There are many predictors relative to the panel sample size. Consider a simpler model or more data.")

    return {
        "vif_df": pd.DataFrame(columns=["variable", "VIF"]),
        "warnings": warnings,
        "panel_structure": structure,
        "within_variation": variation,
    }


PANEL_FE_MODEL = ModelDefinition(
    model_id="panel_fe",
    display_name_en="Panel Fixed Effects",
    display_name_zh="面板固定效应",
    description_en=(
        "Use Panel Fixed Effects when the same firms, regions, countries, or individuals are observed over time. "
        "This model controls for unobserved time-invariant differences across entities and, optionally, common time shocks."
    ),
    description_zh=(
        "当同一企业、地区、国家或个体在多个时间点被重复观察时，可使用面板固定效应模型。"
        "该模型可以控制个体层面不随时间变化的不可观测差异，也可选择控制共同时间冲击。"
    ),
    required_roles=["dependent_variable", "independent_variables", "entity_id", "time_id"],
    validate=validate_panel_fe_inputs,
    fit=fit_panel_fe,
    diagnostics=diagnose_panel_fe,
    report_label_en="Panel Fixed Effects",
    report_label_zh="面板固定效应",
    limitations_en=[
        "Requires repeated observations for the same entities over time.",
        "Fixed effects control for time-invariant entity differences but do not automatically prove causality.",
        "Variables with little within-entity variation may be hard to estimate precisely.",
    ],
    limitations_zh=[
        "要求同一个体在多个时间点被重复观察。",
        "固定效应可以控制不随时间变化的个体差异，但并不自动证明因果关系。",
        "个体内部变化很少的变量可能难以被精确估计。",
    ],
)
