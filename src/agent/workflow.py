from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from src.agent.schemas import AnalysisPlan
from src.formatting import format_p_value
from src.models.registry import get_model


@dataclass(frozen=True)
class WorkflowRunConfig:
    workflow_id: str
    plan_id: str
    language: str
    workflow_type: str
    baseline_model_ids: list[str]
    main_model_id: str
    alternative_model_ids: list[str]
    dependent_variable: str | None
    main_explanatory_variables: list[str]
    numeric_controls: list[str]
    categorical_controls: list[str]
    entity_id: str | None = None
    time_id: str | None = None
    fixed_effects: dict[str, bool] = field(default_factory=dict)
    standard_errors: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class WorkflowModelResult:
    model_id: str
    model_name: str
    role: str
    status: str
    result: dict[str, Any] | None
    summary_metrics: dict[str, Any]
    key_coefficients: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)
    error_message: str = ""


@dataclass(frozen=True)
class WorkflowComparison:
    compared_models: list[str]
    coefficient_comparison: list[dict[str, Any]]
    observation_comparison: dict[str, Any]
    interpretation: list[str]
    warnings: list[str] = field(default_factory=list)


@dataclass
class GuidedWorkflowResult:
    workflow_id: str
    plan_id: str
    workflow_type: str
    status: str
    baseline_results: list[WorkflowModelResult]
    main_result: WorkflowModelResult | None
    alternative_results: list[WorkflowModelResult]
    comparison: WorkflowComparison | None
    summary: str
    warnings: list[str] = field(default_factory=list)
    user_next_steps: list[str] = field(default_factory=list)


def build_workflow_config(plan: AnalysisPlan, language: str = "en") -> WorkflowRunConfig:
    lang = "zh" if language == "zh" else "en"
    main_model_id = plan.recommended_main_model.model_id if plan.recommended_main_model else ""
    baseline_ids = [model.model_id for model in plan.baseline_models]
    alternative_ids = [model.model_id for model in plan.alternative_models if model.model_id in {"probit"}]

    warnings: list[str] = []
    if not plan.recommended_dependent_variable:
        warnings.append(_txt(lang, "missing_y"))
    if not plan.recommended_main_explanatory_variables:
        warnings.append(_txt(lang, "missing_x"))

    if main_model_id == "panel_fe":
        workflow_type = "panel_baseline_main"
        baseline_ids = ["ols"] if "ols" in baseline_ids else []
        fixed_effects = {"entity": True, "time": True}
        standard_errors = "cluster_entity"
        if not plan.entity_id:
            warnings.append(_txt(lang, "missing_entity"))
        if not plan.time_id:
            warnings.append(_txt(lang, "missing_time"))
    elif main_model_id == "logit":
        workflow_type = "binary_main_model"
        fixed_effects = {}
        standard_errors = "default"
    else:
        workflow_type = "single_main_model"
        fixed_effects = {}
        standard_errors = "hc3"

    return WorkflowRunConfig(
        workflow_id=f"workflow_{plan.plan_id}",
        plan_id=plan.plan_id,
        language=lang,
        workflow_type=workflow_type,
        baseline_model_ids=baseline_ids,
        main_model_id=main_model_id,
        alternative_model_ids=alternative_ids if main_model_id == "logit" else [],
        dependent_variable=plan.recommended_dependent_variable,
        main_explanatory_variables=list(plan.recommended_main_explanatory_variables),
        numeric_controls=list(plan.numeric_controls),
        categorical_controls=list(plan.categorical_controls),
        entity_id=plan.entity_id,
        time_id=plan.time_id,
        fixed_effects=fixed_effects,
        standard_errors=standard_errors,
        warnings=warnings,
    )


def run_guided_workflow(
    df: pd.DataFrame,
    variable_roles: dict[str, str],
    plan: AnalysisPlan,
    language: str = "en",
) -> GuidedWorkflowResult:
    config = build_workflow_config(plan, language)
    warnings = list(config.warnings)
    if warnings:
        return GuidedWorkflowResult(
            workflow_id=config.workflow_id,
            plan_id=config.plan_id,
            workflow_type=config.workflow_type,
            status="failed",
            baseline_results=[],
            main_result=None,
            alternative_results=[],
            comparison=None,
            summary=_txt(config.language, "cannot_run"),
            warnings=warnings,
            user_next_steps=[_txt(config.language, "review_plan")],
        )

    baseline_results = [_run_one_model(df, model_id, "baseline", config) for model_id in config.baseline_model_ids]
    main_result = _run_one_model(df, config.main_model_id, "main", config)

    alternative_results: list[WorkflowModelResult] = []
    if main_result.status == "success":
        alternative_results = [_run_one_model(df, model_id, "alternative", config) for model_id in config.alternative_model_ids]

    comparison = _build_comparison(config, baseline_results, main_result, alternative_results)
    all_results = baseline_results + ([main_result] if main_result else []) + alternative_results
    status = "success" if main_result and main_result.status == "success" else "failed"
    workflow_warnings = warnings + [warning for result in all_results for warning in result.warnings]
    workflow_warnings += [result.error_message for result in all_results if result.status != "success" and result.error_message]

    return GuidedWorkflowResult(
        workflow_id=config.workflow_id,
        plan_id=config.plan_id,
        workflow_type=config.workflow_type,
        status=status,
        baseline_results=baseline_results,
        main_result=main_result,
        alternative_results=alternative_results,
        comparison=comparison,
        summary=_summary_for(config, main_result),
        warnings=workflow_warnings,
        user_next_steps=_next_steps(config.language, status),
    )


def workflow_to_dict(result: GuidedWorkflowResult) -> dict[str, Any]:
    payload = asdict(result)
    for group in ["baseline_results", "alternative_results"]:
        for item in payload[group]:
            item["result"] = _compact_result(item.get("result"))
    if payload.get("main_result"):
        payload["main_result"]["result"] = _compact_result(payload["main_result"].get("result"))
    return payload


def _run_one_model(df: pd.DataFrame, model_id: str, role: str, config: WorkflowRunConfig) -> WorkflowModelResult:
    model = get_model(model_id)
    model_config = _model_config(model_id, config)
    try:
        errors = model.validate(df, model_config)
        if errors:
            raise ValueError(errors[0])
        fit_payload = model.fit(df, model_config)
        diagnostics = model.diagnostics(df, fit_payload["cleaned_df"], model_config, fit_payload) if model.diagnostics else {}
        warnings = list(diagnostics.get("warnings", [])) + list(fit_payload.get("warnings", []))
        return WorkflowModelResult(
            model_id=model_id,
            model_name=model.display_name(config.language),
            role=role,
            status="success",
            result={
                "fit_payload": fit_payload,
                "diagnostics": diagnostics,
                "model_config": model_config,
            },
            summary_metrics=_summary_metrics(fit_payload["model_summary"]),
            key_coefficients=_key_coefficients(fit_payload["regression_table"], config.main_explanatory_variables),
            warnings=warnings,
        )
    except Exception as exc:
        return WorkflowModelResult(
            model_id=model_id,
            model_name=model.display_name(config.language),
            role=role,
            status="failed",
            result=None,
            summary_metrics={},
            key_coefficients=[],
            warnings=[],
            error_message=str(exc),
        )


def _model_config(model_id: str, config: WorkflowRunConfig) -> dict[str, Any]:
    base = {
        "dependent_variable": config.dependent_variable,
        "main_independent_variables": list(config.main_explanatory_variables),
        "numeric_control_variables": list(config.numeric_controls),
        "categorical_control_variables": [],
        "encode_categorical_controls": False,
        "robust_standard_errors": True,
        "variable_roles": {},
    }
    if model_id == "panel_fe":
        base.update(
            {
                "entity_id": config.entity_id,
                "time_id": config.time_id,
                "entity_effects": bool(config.fixed_effects.get("entity", True)),
                "time_effects": bool(config.fixed_effects.get("time", True)),
                "standard_errors": config.standard_errors or "cluster_entity",
            }
        )
    elif model_id == "ols":
        base["robust_standard_errors"] = True
    return base


def _summary_metrics(model_summary: dict[str, Any]) -> dict[str, Any]:
    model_type = str(model_summary.get("model_type", "")).lower()
    metrics = {
        "observations": model_summary.get("n_obs"),
        "standard_errors": model_summary.get("standard_errors") or ("HC3" if model_summary.get("robust_standard_errors") else "conventional"),
    }
    if model_type == "panel_fe":
        metrics.update(
            {
                "entities": model_summary.get("entities"),
                "time_periods": model_summary.get("time_periods"),
                "within_r_squared": model_summary.get("r_squared_within"),
            }
        )
    elif model_type in {"logit", "probit"}:
        metrics.update(
            {
                "pseudo_r_squared": model_summary.get("pseudo_r_squared"),
                "aic": model_summary.get("aic"),
                "bic": model_summary.get("bic"),
            }
        )
    else:
        metrics.update(
            {
                "r_squared": model_summary.get("r_squared"),
                "adj_r_squared": model_summary.get("adj_r_squared"),
            }
        )
    return metrics


def _key_coefficients(regression_table: pd.DataFrame, main_variables: list[str]) -> list[dict[str, Any]]:
    if regression_table is None or regression_table.empty:
        return []
    rows = regression_table[regression_table["variable"].isin(main_variables)]
    result = []
    for _, row in rows.iterrows():
        coefficient = float(row["coefficient"])
        result.append(
            {
                "variable": str(row["variable"]),
                "coefficient": coefficient,
                "std_error": float(row["std_error"]) if "std_error" in row else None,
                "p_value": float(row["p_value"]) if "p_value" in row else None,
                "p_value_formatted": format_p_value(row.get("p_value")),
                "significance": str(row.get("significance", "")),
                "direction": "positive" if coefficient > 0 else "negative" if coefficient < 0 else "zero",
            }
        )
    return result


def _build_comparison(
    config: WorkflowRunConfig,
    baseline_results: list[WorkflowModelResult],
    main_result: WorkflowModelResult,
    alternative_results: list[WorkflowModelResult],
) -> WorkflowComparison:
    compared = [result.model_name for result in baseline_results if result.status == "success"]
    if main_result and main_result.status == "success":
        compared.append(main_result.model_name)
    compared += [result.model_name for result in alternative_results if result.status == "success"]

    coefficient_rows = []
    for variable in config.main_explanatory_variables:
        row: dict[str, Any] = {"variable": variable}
        for result in baseline_results + [main_result] + alternative_results:
            coefficient = next((coef for coef in result.key_coefficients if coef["variable"] == variable), None)
            if coefficient:
                row[f"{result.model_id}_direction"] = coefficient["direction"]
                row[f"{result.model_id}_p_value"] = coefficient["p_value_formatted"]
        coefficient_rows.append(row)

    observation_comparison = {
        result.model_id: result.summary_metrics.get("observations")
        for result in baseline_results + [main_result] + alternative_results
        if result and result.status == "success"
    }

    interpretation = _comparison_interpretation(config)
    return WorkflowComparison(
        compared_models=compared,
        coefficient_comparison=coefficient_rows,
        observation_comparison=observation_comparison,
        interpretation=interpretation,
        warnings=[],
    )


def _comparison_interpretation(config: WorkflowRunConfig) -> list[str]:
    if config.workflow_type == "panel_baseline_main":
        if config.language == "zh":
            return [
                "OLS 估计的是混合样本中的总体相关关系。",
                "面板固定效应估计的是同一个体内部随时间变化的关系。",
                "若 OLS 与 Panel FE 结果差异较大，可能说明个体层面不随时间变化的差异会影响普通 OLS 结果。",
            ]
        return [
            "OLS estimates pooled associations across all observations.",
            "Panel FE estimates within-entity relationships over time.",
            "Differences between OLS and Panel FE may reflect unobserved entity-level heterogeneity.",
        ]
    if config.workflow_type == "binary_main_model":
        if config.language == "zh":
            return ["Logit/Probit 系数大小不能直接比较，重点应看方向和显著性。"]
        return ["Logit/Probit coefficient magnitudes are not directly comparable; focus on direction and significance."]
    return [_txt(config.language, "single_main_interpretation")]


def _summary_for(config: WorkflowRunConfig, main_result: WorkflowModelResult | None) -> str:
    if main_result is None or main_result.status != "success":
        return _txt(config.language, "workflow_failed")
    if config.workflow_type == "panel_baseline_main":
        return _txt(config.language, "panel_summary")
    if config.workflow_type == "binary_main_model":
        return _txt(config.language, "binary_summary")
    return _txt(config.language, "ols_summary")


def _next_steps(language: str, status: str) -> list[str]:
    if status != "success":
        return [_txt(language, "review_plan")]
    return [_txt(language, "review_comparison"), _txt(language, "review_assumptions")]


def _compact_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not result:
        return None
    fit_payload = result.get("fit_payload", {})
    diagnostics = result.get("diagnostics", {})
    return {
        "model_summary": fit_payload.get("model_summary", {}),
        "cleaning_log": fit_payload.get("cleaning_log", {}),
        "warnings": diagnostics.get("warnings", []),
    }


def _txt(language: str, key: str) -> str:
    lang = "zh" if language == "zh" else "en"
    text = {
        "en": {
            "missing_y": "The workflow needs a dependent variable.",
            "missing_x": "The workflow needs at least one main explanatory variable.",
            "missing_entity": "Panel workflow requires an entity ID.",
            "missing_time": "Panel workflow requires a time ID.",
            "cannot_run": "Guided workflow cannot run until the plan is complete.",
            "review_plan": "Review the analysis plan and variable roles.",
            "review_comparison": "Review the baseline/main model comparison.",
            "review_assumptions": "Check assumptions and limitations before reporting.",
            "panel_summary": "Guided analysis ran an OLS baseline and Panel Fixed Effects main model.",
            "binary_summary": "Guided analysis ran the recommended binary-outcome model.",
            "ols_summary": "Guided analysis ran the recommended main OLS model.",
            "workflow_failed": "Guided workflow did not complete successfully.",
            "single_main_interpretation": "The workflow contains one main model, so no baseline comparison is shown.",
        },
        "zh": {
            "missing_y": "引导式流程需要因变量。",
            "missing_x": "引导式流程需要至少一个核心解释变量。",
            "missing_entity": "面板流程需要个体 ID。",
            "missing_time": "面板流程需要时间变量。",
            "cannot_run": "分析计划尚不完整，暂不能运行引导式流程。",
            "review_plan": "请检查分析计划和变量角色。",
            "review_comparison": "请查看基准模型与主模型比较。",
            "review_assumptions": "写入报告前请检查模型假设与局限。",
            "panel_summary": "引导式分析已运行 OLS 基准模型和面板固定效应主模型。",
            "binary_summary": "引导式分析已运行推荐的二元结果模型。",
            "ols_summary": "引导式分析已运行推荐的 OLS 主模型。",
            "workflow_failed": "引导式分析流程未成功完成。",
            "single_main_interpretation": "该流程仅包含一个主模型，因此不显示基准模型比较。",
        },
    }
    return text[lang][key]
