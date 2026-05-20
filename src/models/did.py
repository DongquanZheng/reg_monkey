from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.models.base import ModelDefinition
from src.models.diagnostics import DiagnosticCode, DiagnosticSeverity, ModelDiagnostic
from src.utils import significance_stars


DID_MODEL_ID = "did"
DID_INTERACTION_SEPARATOR = ":"
HIGH_CARDINALITY_THRESHOLD = 20
DID_SPARSE_CELL_THRESHOLD = 2


def validate_did_inputs(df: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    from src.models.did_contract import validate_did_spec_contract

    spec = _spec_from_config(config)
    validation = validate_did_spec_contract(df, spec)
    errors = list(validation.errors)
    if errors:
        return errors

    selected = _selected_columns(config)
    working = df[selected].dropna().copy()
    if working.empty:
        return ["No complete observations remain after dropping missing values for the DID specification."]

    for column, label in [(config.get("treatment_variable"), "treatment variable"), (config.get("post_variable"), "post variable")]:
        values = _binary_indicator_values(working[column])
        if values is None:
            errors.append(f"The DID {label} must be a binary-like indicator with two 0/1 numeric values or two categorical levels.")
        elif values.nunique(dropna=True) < 2:
            errors.append(f"The DID {label} has insufficient variation.")

    if errors:
        return errors

    treatment = _binary_indicator_values(working[config["treatment_variable"]])
    post = _binary_indicator_values(working[config["post_variable"]])
    interaction = treatment * post
    predictors = [config["treatment_variable"], config["post_variable"], _did_term(config)] + list(config.get("numeric_control_variables") or [])
    cell_counts = _did_cell_counts_from_values(treatment, post, config["treatment_variable"], config["post_variable"])
    has_missing_cell = bool((cell_counts["count"] <= 0).any()) if not cell_counts.empty else False
    if interaction.nunique(dropna=True) < 2:
        errors.append("The DID treatment-by-post interaction has insufficient variation.")
    if len(working) <= len(predictors) + 1:
        errors.append("Not enough observations to estimate this DID regression.")
    if _has_staggered_or_time_varying_treatment(working, config, treatment):
        errors.append("The minimal DID runner does not support staggered adoption or time-varying treatment assignment within groups.")
    if not has_missing_cell and _has_collinear_did_design(working, config, treatment, post, interaction):
        errors.append("The DID design matrix is collinear; the treatment-by-post interaction may be absorbed by selected variables.")

    cluster_variable = str(config.get("cluster_variable") or "")
    standard_errors = str(config.get("standard_errors") or "").lower()
    if standard_errors in {"cluster", "clustered", "cluster_entity"} or cluster_variable:
        cluster_col = cluster_variable or str(config.get("entity_id") or config.get("group_variable") or "")
        if not cluster_col:
            errors.append("Clustered standard errors require a cluster variable.")
        elif cluster_col not in df.columns:
            errors.append(f"Cluster variable '{cluster_col}' was not found in the dataset.")
        elif df[cluster_col].dropna().nunique() < 2:
            errors.append("Clustered standard errors require at least two clusters.")

    return errors


def prepare_did_dataframe(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, str, list[str], dict[str, Any], dict[str, Any], pd.DataFrame]:
    y_col = config["dependent_variable"]
    treatment_col = config["treatment_variable"]
    post_col = config["post_variable"]
    numeric_controls = list(config.get("numeric_control_variables") or [])
    categorical_controls = list(config.get("categorical_control_variables") or [])
    encode_categoricals = bool(config.get("encode_categorical_controls", False))
    cluster_col = _cluster_column(config, for_standard_errors=True)

    selected = [y_col, treatment_col, post_col, *numeric_controls]
    if cluster_col:
        selected.append(cluster_col)
    if encode_categoricals:
        selected.extend(categorical_controls)
    selected = list(dict.fromkeys([column for column in selected if column]))

    original_rows = len(df)
    working = df[selected].dropna().copy()
    working[treatment_col] = _binary_indicator_values(working[treatment_col])
    working[post_col] = _binary_indicator_values(working[post_col])
    did_term = _did_term(config)
    working[did_term] = working[treatment_col].astype(float) * working[post_col].astype(float)
    cell_counts = _did_cell_counts(working, treatment_col, post_col)

    x_cols = [treatment_col, post_col, did_term, *numeric_controls]
    encoding_info = {
        "enabled": encode_categoricals,
        "selected_categorical_controls": categorical_controls,
        "encoded_categorical_controls": [],
        "reference_categories": {},
        "dummy_variables": [],
        "ignored_categorical_controls": [] if encode_categoricals else categorical_controls,
        "high_cardinality_warnings": [],
    }
    if encode_categoricals and categorical_controls:
        for column in categorical_controls:
            levels = sorted(working[column].dropna().astype(str).unique().tolist())
            if len(levels) > HIGH_CARDINALITY_THRESHOLD:
                encoding_info["high_cardinality_warnings"].append(
                    f"Categorical control '{column}' has {len(levels)} levels; dummy encoding may create a large DID model."
                )
            if len(levels) <= 1:
                continue
            reference = levels[0]
            encoding_info["reference_categories"][column] = reference
            dummies = pd.get_dummies(working[column].astype(str), prefix=column, drop_first=True, dtype=float)
            working = pd.concat([working.drop(columns=[column]), dummies], axis=1)
            dummy_names = dummies.columns.tolist()
            x_cols.extend(dummy_names)
            encoding_info["dummy_variables"].extend(dummy_names)
            encoding_info["encoded_categorical_controls"].append(column)

    cleaning_log = {
        "original_row_count": original_rows,
        "final_row_count": len(working),
        "dropped_row_count": original_rows - len(working),
        "dropped_row_percentage": round(((original_rows - len(working)) / original_rows * 100), 2) if original_rows else 0.0,
    }
    return working, y_col, x_cols, cleaning_log, encoding_info, cell_counts


def fit_did(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    errors = validate_did_inputs(df, config)
    if errors:
        raise ValueError(errors[0])

    model_df, y_col, x_cols, cleaning_log, encoding_info, cell_counts = prepare_did_dataframe(df, config)
    y = model_df[y_col].astype(float)
    x = sm.add_constant(model_df[x_cols].astype(float), has_constant="add")
    fitted = sm.OLS(y, x).fit()

    standard_errors = str(config.get("standard_errors") or "hc3").lower()
    cluster_col = _cluster_column(config, for_standard_errors=True)
    if standard_errors in {"cluster", "clustered", "cluster_entity"} and cluster_col:
        result = fitted.get_robustcov_results(cov_type="cluster", groups=model_df[cluster_col])
        se_label = f"cluster:{cluster_col}"
    elif standard_errors in {"conventional", "none"}:
        result = fitted
        se_label = "conventional"
    else:
        result = fitted.get_robustcov_results(cov_type="HC3")
        se_label = "hc3"

    regression_table = _regression_table(result, x.columns)
    did_term = _did_term(config)
    did_row = regression_table[regression_table["variable"] == did_term].iloc[0]
    model_summary = {
        "model_type": DID_MODEL_ID,
        "dependent_variable": y_col,
        "treatment_variable": config["treatment_variable"],
        "post_variable": config["post_variable"],
        "group_variable": config.get("group_variable", ""),
        "entity_id": config.get("entity_id", ""),
        "time_id": config.get("time_id", ""),
        "cluster_variable": cluster_col,
        "standard_errors": se_label,
        "main_independent_variables": [did_term],
        "numeric_control_variables": list(config.get("numeric_control_variables") or []),
        "categorical_control_variables": list(config.get("categorical_control_variables") or []),
        "encoded_categorical_controls": encoding_info["encoded_categorical_controls"],
        "dummy_variables": encoding_info["dummy_variables"],
        "reference_categories": encoding_info["reference_categories"],
        "independent_variables": list(x_cols),
        "r_squared": float(result.rsquared),
        "adj_r_squared": float(result.rsquared_adj),
        "n_obs": int(result.nobs),
        "num_predictors": len(x_cols),
        "did_term": did_term,
        "did_estimate": float(did_row["coefficient"]),
        "did_p_value": float(did_row["p_value"]),
    }
    return {
        "fitted_model": result,
        "regression_table": regression_table,
        "model_summary": model_summary,
        "cleaned_df": model_df,
        "cleaning_log": cleaning_log,
        "encoding_info": encoding_info,
        "advanced_outputs": {
            "did_summary": {
                "did_estimate": float(did_row["coefficient"]),
                "did_term": did_term,
                "standard_error": float(did_row["std_error"]),
                "p_value": float(did_row["p_value"]),
                "did_standard_error": float(did_row["std_error"]),
                "did_p_value": float(did_row["p_value"]),
                "treatment_variable": config["treatment_variable"],
                "post_variable": config["post_variable"],
                "controls": list(config.get("numeric_control_variables") or []),
                "observations_used": int(result.nobs),
                "cell_counts": cell_counts.to_dict(orient="records"),
            }
        },
    }


def diagnose_did(df_original: pd.DataFrame, df_cleaned: pd.DataFrame, config: dict[str, Any], fit_result: Any) -> dict[str, Any]:
    diagnostics = [
        ModelDiagnostic(
            code=DiagnosticCode.DID_PARALLEL_TRENDS_ASSUMPTION,
            severity=DiagnosticSeverity.CONSTRAINT,
            title="DID parallel trends assumption",
            message="Minimal DID results depend on a parallel trends assumption; this version does not test parallel trends.",
            recommendation="Review pre-treatment trends and study design before interpreting the DID coefficient causally.",
            show_in_ui=True,
            show_in_report=True,
            llm_instruction="Explain that DID estimates require parallel trends and do not automatically establish causality.",
        )
    ]
    warnings: list[str] = []
    treatment_col = config["treatment_variable"]
    post_col = config["post_variable"]
    if treatment_col in df_cleaned.columns and post_col in df_cleaned.columns:
        cell_counts = _did_cell_counts(df_cleaned, treatment_col, post_col)
        missing_cells = cell_counts[cell_counts["count"] <= 0]["cell"].astype(str).tolist()
        sparse_cells = cell_counts[(cell_counts["count"] > 0) & (cell_counts["count"] < DID_SPARSE_CELL_THRESHOLD)]["cell"].astype(str).tolist()
        if missing_cells:
            diagnostics.append(
                ModelDiagnostic(
                    code=DiagnosticCode.DID_CELL_COVERAGE,
                    severity=DiagnosticSeverity.WARNING,
                    title="Incomplete DID treatment/post cells",
                    message="The estimation sample is missing DID treatment/post cell(s): " + ", ".join(missing_cells) + ".",
                    recommendation="Check treatment and post-period coding before interpreting the DID estimate.",
                    affected_variables=[treatment_col, post_col],
                    show_in_ui=True,
                    show_in_report=True,
                    llm_instruction="Mention limited DID cell coverage as a design limitation.",
                )
            )
        if sparse_cells:
            diagnostics.append(
                ModelDiagnostic(
                    code=DiagnosticCode.DID_SPARSE_CELL_SUPPORT,
                    severity=DiagnosticSeverity.WARNING,
                    title="Sparse DID treatment/post cells",
                    message=f"Some DID treatment/post cells have fewer than {DID_SPARSE_CELL_THRESHOLD} observation(s): " + ", ".join(sparse_cells) + ".",
                    recommendation="Use more data or simplify the DID specification before interpreting the DID estimate.",
                    affected_variables=[treatment_col, post_col],
                    show_in_ui=True,
                    show_in_report=True,
                    llm_instruction="Mention sparse DID cell support as an estimation-sample limitation.",
                )
            )
    else:
        cell_counts = pd.DataFrame(columns=[treatment_col, post_col, "count"])
    warnings.extend(fit_result.get("encoding_info", {}).get("high_cardinality_warnings", []))
    return {
        "warnings": warnings,
        "structured_diagnostics": diagnostics,
        "did_cell_counts": cell_counts,
    }


def _did_cell_counts(df: pd.DataFrame, treatment_col: str, post_col: str) -> pd.DataFrame:
    if treatment_col not in df.columns or post_col not in df.columns:
        return pd.DataFrame(columns=[treatment_col, post_col, "count", "cell"])
    working = df[[treatment_col, post_col]].dropna().copy()
    working[treatment_col] = _binary_indicator_values(working[treatment_col])
    working[post_col] = _binary_indicator_values(working[post_col])
    return _did_cell_counts_from_values(working[treatment_col], working[post_col], treatment_col, post_col)


def _did_cell_counts_from_values(treatment: pd.Series, post: pd.Series, treatment_col: str, post_col: str) -> pd.DataFrame:
    working = pd.DataFrame({treatment_col: treatment, post_col: post}).dropna()
    observed = working.groupby([treatment_col, post_col]).size().to_dict()
    rows = []
    for treatment in [0.0, 1.0]:
        for post in [0.0, 1.0]:
            count = int(observed.get((treatment, post), 0))
            rows.append(
                {
                    treatment_col: int(treatment),
                    post_col: int(post),
                    "count": count,
                    "cell": f"treatment={int(treatment)}, post={int(post)}",
                }
            )
    return pd.DataFrame(rows)


def _regression_table(result: Any, columns: pd.Index) -> pd.DataFrame:
    params = pd.Series(result.params, index=columns)
    std_errors = pd.Series(result.bse, index=columns)
    t_values = pd.Series(result.tvalues, index=columns)
    p_values = pd.Series(result.pvalues, index=columns)
    table = pd.DataFrame(
        {
            "variable": columns,
            "coefficient": params.values,
            "std_error": std_errors.values,
            "t_value": t_values.values,
            "p_value": p_values.values,
        }
    )
    conf_int = pd.DataFrame(result.conf_int(), index=columns)
    table["conf_int_low"] = conf_int.iloc[:, 0].values
    table["conf_int_high"] = conf_int.iloc[:, 1].values
    table["significance"] = table["p_value"].apply(significance_stars)
    return table


def _indicator_values(series: pd.Series) -> pd.Series | None:
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
    else:
        non_missing = series.dropna().astype(str)
        levels = sorted(non_missing.unique().tolist())
        if len(levels) > 2:
            return None
        mapping = {level: index for index, level in enumerate(levels)}
        values = series.astype(str).map(mapping)
    return values.astype(float)


def _binary_indicator_values(series: pd.Series) -> pd.Series | None:
    values = _indicator_values(series)
    if values is None:
        return None
    unique = sorted(float(item) for item in values.dropna().unique().tolist())
    if len(unique) > 2:
        return None
    if pd.api.types.is_numeric_dtype(series):
        if not set(unique).issubset({0.0, 1.0}):
            return None
    return values.astype(float)


def _has_staggered_or_time_varying_treatment(
    working: pd.DataFrame,
    config: dict[str, Any],
    treatment: pd.Series,
) -> bool:
    group_col = str(config.get("group_variable") or config.get("entity_id") or "")
    if not group_col or group_col not in working.columns:
        return False
    grouped = pd.DataFrame({"group": working[group_col], "treatment": treatment}).dropna()
    if grouped.empty:
        return False
    variation = grouped.groupby("group")["treatment"].nunique(dropna=True)
    return bool((variation > 1).any())


def _has_collinear_did_design(
    working: pd.DataFrame,
    config: dict[str, Any],
    treatment: pd.Series,
    post: pd.Series,
    interaction: pd.Series,
) -> bool:
    design = pd.DataFrame(
        {
            config["treatment_variable"]: treatment.astype(float),
            config["post_variable"]: post.astype(float),
            _did_term(config): interaction.astype(float),
        }
    )
    for column in list(config.get("numeric_control_variables") or []):
        if column in working.columns:
            design[column] = pd.to_numeric(working[column], errors="coerce")
    design = design.dropna()
    if design.empty:
        return False
    matrix = sm.add_constant(design.astype(float), has_constant="add").to_numpy()
    return bool(np.linalg.matrix_rank(matrix) < matrix.shape[1])


def _selected_columns(config: dict[str, Any]) -> list[str]:
    columns = [
        config.get("dependent_variable"),
        config.get("treatment_variable"),
        config.get("post_variable"),
        config.get("group_variable"),
        config.get("entity_id"),
        config.get("time_id"),
        _cluster_column(config, for_standard_errors=True),
        *list(config.get("numeric_control_variables") or []),
    ]
    if config.get("encode_categorical_controls"):
        columns.extend(list(config.get("categorical_control_variables") or []))
    return [column for column in dict.fromkeys(columns) if column]


def _cluster_column(config: dict[str, Any], for_standard_errors: bool = False) -> str:
    explicit = str(config.get("cluster_variable") or "")
    if explicit:
        return explicit
    if for_standard_errors:
        standard_errors = str(config.get("standard_errors") or "").lower()
        if standard_errors in {"cluster", "clustered", "cluster_entity"}:
            return str(config.get("entity_id") or config.get("group_variable") or "")
    return ""


def _did_term(config: dict[str, Any]) -> str:
    return f"{config['treatment_variable']}{DID_INTERACTION_SEPARATOR}{config['post_variable']}"


def _spec_from_config(config: dict[str, Any]) -> Any:
    from src.models.execution import ModelSpec

    return ModelSpec(
        model_id=DID_MODEL_ID,
        dependent_variable=str(config.get("dependent_variable") or ""),
        treatment_variable=str(config.get("treatment_variable") or ""),
        post_variable=str(config.get("post_variable") or ""),
        group_variable=str(config.get("group_variable") or ""),
        entity_id=str(config.get("entity_id") or ""),
        time_id=str(config.get("time_id") or ""),
        numeric_control_variables=list(config.get("numeric_control_variables") or []),
        categorical_control_variables=list(config.get("categorical_control_variables") or []),
        encode_categorical_controls=bool(config.get("encode_categorical_controls", False)),
        standard_errors=str(config.get("standard_errors") or "hc3"),
        cluster_variable=str(config.get("cluster_variable") or ""),
    )


DID_MODEL = ModelDefinition(
    model_id=DID_MODEL_ID,
    display_name_en="Difference-in-Differences",
    display_name_zh="双重差分",
    description_en="Estimate a minimal DID regression with treatment, post, and treatment-by-post terms.",
    description_zh="估计包含处理组、处理后时期和交互项的最小双重差分回归。",
    required_roles=["dependent_variable", "treatment_variable", "post_variable"],
    validate=validate_did_inputs,
    fit=fit_did,
    diagnostics=diagnose_did,
    report_label_en="Difference-in-Differences",
    report_label_zh="双重差分",
    limitations_en=[
        "This minimal DID model does not test parallel trends.",
        "Causal interpretation depends on the research design and assumptions.",
        "This version does not implement event-study or staggered-adoption DID.",
    ],
    limitations_zh=[
        "当前最小 DID 模型不会检验平行趋势。",
        "因果解释取决于研究设计和识别假设。",
        "当前版本不实现事件研究或错位处理 DID。",
    ],
)
