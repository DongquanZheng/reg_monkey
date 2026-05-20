from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.agent.workflow_summary import workflow_summary_markdown
from src.agent.narrative import generate_narrative, narrative_markdown
from src.diagnostic_rendering import diagnostic_dicts, warning_lines_for_display
from src.formatting import format_diagnostic_table, format_number, format_p_narrative, format_p_value, format_regression_table
from src.i18n import get_text, normalize_language, translate_diagnostic_field, translate_warning
from src.variable_roles import (
    ROLE_BINARY,
    ROLE_CATEGORICAL,
    ROLE_CODE,
    ROLE_ENTITY,
    ROLE_EXCLUDE,
    ROLE_NUMERIC,
    ROLE_TIME,
    role_label,
    summarize_role_counts,
)


ROLE_ORDER = [ROLE_NUMERIC, ROLE_BINARY, ROLE_TIME, ROLE_CODE, ROLE_CATEGORICAL, ROLE_ENTITY, ROLE_EXCLUDE]


def _markdown_table(df: pd.DataFrame, language: str = "en") -> str:
    if df is None or df.empty:
        return "_No data available._" if language != "zh" else "_无可用数据。_"
    safe_df = df.copy()
    safe_df = safe_df.fillna("N/A" if language != "zh" else "不适用")
    columns = [str(col) for col in safe_df.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(_markdown_value(value) for value in row) + " |" for row in safe_df.itertuples(index=False, name=None)]
    return "\n".join([header, separator] + rows)


def _markdown_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _diagnostic_markdown_table(df: pd.DataFrame, language: str = "en") -> str:
    return _markdown_table(format_diagnostic_table(df, language, digits=2), language)


def _model_type(model_summary: dict[str, Any]) -> str:
    return str(model_summary.get("model_type") or "ols").lower()


def _compact_equation(model_summary: dict[str, Any]) -> str:
    model_type = _model_type(model_summary)
    if model_type == "psm":
        treatment = model_summary.get("treatment_variable", "Treatment")
        covariates = ", ".join(model_summary.get("matching_covariates") or ["Covariates"])
        return f"Propensity score: P({treatment}=1) = Logit({covariates}); ATT = mean(Y_treated - Y_matched_control)"
    if model_type == "iv_2sls":
        endogenous = model_summary.get("endogenous_variable", "Endogenous")
        iv_term = model_summary.get("iv_term", f"fitted_{endogenous}")
        instruments = ", ".join(model_summary.get("instruments") or ["Instrument"])
        controls = " + Controls" if model_summary.get("exogenous_controls") else ""
        return f"Stage 1: {endogenous} = Instruments({instruments}){controls}; Stage 2: Y = {iv_term}{controls} + ε"
    if model_type == "did":
        rhs = ["β0", "β1 Treatment_i", "β2 Post_t", "β3 Treatment_i x Post_t"]
        if model_summary.get("numeric_control_variables"):
            rhs.append("γ Controls_i")
        if model_summary.get("encoded_categorical_controls"):
            rhs.append("δ CategoricalControls_i")
        return f"Y_it = {' + '.join(rhs)} + ε_it"

    if model_type == "panel_fe":
        rhs = []
        if model_summary.get("main_independent_variables") or model_summary.get("independent_variables"):
            rhs.append("β MainX_it")
        if model_summary.get("numeric_control_variables"):
            rhs.append("γ Controls_it")
        if model_summary.get("entity_effects"):
            rhs.append("α_i")
        if model_summary.get("time_effects"):
            rhs.append("λ_t")
        rhs.append("ε_it")
        return f"Y_it = {' + '.join(rhs)}"

    rhs = ["β0"]
    if model_summary.get("main_independent_variables") or model_summary.get("independent_variables"):
        rhs.append("β1 MainX_i")
    if model_summary.get("numeric_control_variables"):
        rhs.append("γ Controls_i")
    if model_summary.get("encoded_categorical_controls"):
        rhs.append("δ CategoricalControls_i")

    lhs = "Y_i"
    if model_type == "logit":
        lhs = "logit(P(Y_i = 1))"
    elif model_type == "probit":
        lhs = "Φ^-1(P(Y_i = 1))"

    formula = f"{lhs} = {' + '.join(rhs)}"
    if model_type == "ols":
        formula = f"{formula} + ε_i"
    return formula


def _equation_note(model_summary: dict[str, Any], language: str) -> str:
    if _model_type(model_summary) == "psm":
        if language == "zh":
            return "PSM 估计基于观测协变量匹配后的 ATT；该结果不能处理未观测混杂，也不能自动解释为因果关系。"
        return "PSM estimates an ATT from observed-covariate matching; it does not address unobserved confounding or automatically establish causality."
    if _model_type(model_summary) == "iv_2sls":
        if language == "zh":
            return "IV/2SLS 条件估计依赖工具变量相关性和排除限制；当前结果不能自动解释为因果关系。"
        return "The IV/2SLS conditional estimate depends on instrument relevance and the exclusion restriction; it does not automatically establish causality."
    if _model_type(model_summary) == "did":
        if language == "zh":
            return "DID 估计量是处理组变量与处理后时期变量交互项的系数；因果解释依赖研究设计和平行趋势等识别假设。"
        return "The DID estimate is the coefficient on the treatment-by-post interaction; causal interpretation depends on the research design and assumptions such as parallel trends."
    if _model_type(model_summary) == "panel_fe":
        if language == "zh":
            return "alpha_i 表示个体固定效应，lambda_t 表示时间固定效应。"
        return "alpha_i denotes entity fixed effects and lambda_t denotes time fixed effects."

    notes: list[str] = []
    if model_summary.get("numeric_control_variables"):
        notes.append(
            "Controls 表示数值控制变量。"
            if language == "zh"
            else "Controls are numeric controls."
        )
    if model_summary.get("encoded_categorical_controls"):
        notes.append(
            "CategoricalControls 表示已编码的分类控制变量。"
            if language == "zh"
            else "CategoricalControls are encoded categorical controls."
        )
    return " ".join(notes)


def _role_summary(variable_roles: dict[str, str] | None, language: str) -> tuple[str, pd.DataFrame]:
    if not variable_roles:
        text = "Variable roles were not recorded." if language != "zh" else "未记录变量角色。"
        return text, pd.DataFrame()
    counts = summarize_role_counts(variable_roles)
    rows = []
    for role in ROLE_ORDER:
        variables = [column for column, assigned in variable_roles.items() if assigned == role]
        rows.append(
            {
                "role": role_label(role, language),
                "count": counts.get(role, 0),
                "variables": ", ".join(variables),
            }
        )
    text = "Variable roles were confirmed before modeling." if language != "zh" else "建模前已确认变量角色。"
    return text, pd.DataFrame(rows)


def _profile_stats(profile: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric_stats = profile.get("numeric_descriptive_statistics")
    if not isinstance(numeric_stats, pd.DataFrame):
        numeric_stats = profile.get("descriptive_statistics", pd.DataFrame())
    binary_stats = profile.get("binary_descriptive_statistics")
    if not isinstance(binary_stats, pd.DataFrame):
        binary_stats = pd.DataFrame()
    categorical_stats = profile.get("categorical_descriptive_statistics")
    if not isinstance(categorical_stats, pd.DataFrame):
        categorical_stats = pd.DataFrame()
    return numeric_stats, binary_stats, categorical_stats


def _stats_for_role(stats: pd.DataFrame, variable_roles: dict[str, str] | None, role: str) -> pd.DataFrame:
    if stats is None or stats.empty or not variable_roles or "variable" not in stats.columns:
        return pd.DataFrame()
    variables = [name for name, assigned in variable_roles.items() if assigned == role]
    return stats[stats["variable"].isin(variables)].copy()


def _compact_stats_summary(profile: dict[str, Any], variable_roles: dict[str, str] | None, language: str) -> str:
    numeric_stats, _, _ = _profile_stats(profile)
    numeric_measure_stats = _stats_for_role(numeric_stats, variable_roles, ROLE_NUMERIC)
    count = len(numeric_measure_stats) if not numeric_measure_stats.empty else 0
    if language == "zh":
        return f"简明报告仅概述普通数值变量的描述性统计；当前覆盖 {count} 个数值变量。"
    return f"The simple report summarizes ordinary numeric measures only; {count} numeric measure(s) are covered."


def _fit_summary_lines(model_summary: dict[str, Any], language: str) -> list[str]:
    model_type = _model_type(model_summary)
    if model_type == "psm":
        if language == "zh":
            lines = [
                f"- 观测值：{model_summary.get('n_obs', 'N/A')}",
                f"- 处理组观测：{model_summary.get('treated_count', 'N/A')}",
                f"- 对照组观测：{model_summary.get('control_count', 'N/A')}",
                f"- 已匹配处理组观测：{model_summary.get('matched_treated_count', 'N/A')}",
                f"- 未匹配处理组观测：{model_summary.get('unmatched_treated_count', 'N/A')}",
                f"- ATT 估计：{format_number(model_summary.get('att_estimate'), 4, language)}",
                f"- 匹配方法：{model_summary.get('matching_method', 'nearest_neighbor')}",
            ]
            if model_summary.get("replacement_matching"):
                lines.append("- 匹配模式：1:1 最近邻有放回匹配")
            if model_summary.get("caliper") is not None:
                lines.append(f"- 卡尺：{model_summary.get('caliper')}")
            if model_summary.get("max_absolute_smd_after") is not None:
                lines.append(f"- 匹配后最大绝对标准化均值差：{format_number(model_summary.get('max_absolute_smd_after'), 4, language)}")
            return lines
        lines = [
            f"- Observations: {model_summary.get('n_obs', 'N/A')}",
            f"- Treated observations: {model_summary.get('treated_count', 'N/A')}",
            f"- Control observations: {model_summary.get('control_count', 'N/A')}",
            f"- Matched treated observations: {model_summary.get('matched_treated_count', 'N/A')}",
            f"- Unmatched treated observations: {model_summary.get('unmatched_treated_count', 'N/A')}",
            f"- ATT estimate: {format_number(model_summary.get('att_estimate'), 4, language)}",
            f"- Matching method: {model_summary.get('matching_method', 'nearest_neighbor')}",
        ]
        if model_summary.get("replacement_matching"):
            lines.append("- Matching mode: 1:1 nearest-neighbor with replacement")
        if model_summary.get("caliper") is not None:
            lines.append(f"- Caliper: {model_summary.get('caliper')}")
        if model_summary.get("max_absolute_smd_after") is not None:
            lines.append(f"- Maximum post-match absolute SMD: {format_number(model_summary.get('max_absolute_smd_after'), 4, language)}")
        return lines
    if model_type == "iv_2sls":
        if language == "zh":
            return [
                f"- 观测值：{model_summary.get('n_obs', 'N/A')}",
                f"- 第二阶段 R²：{format_number(model_summary.get('r_squared'), 4, language)}",
                f"- 第二阶段调整 R²：{format_number(model_summary.get('adj_r_squared'), 4, language)}",
                f"- IV 项：{model_summary.get('iv_term', 'N/A')}",
                f"- IV 估计：{format_number(model_summary.get('iv_estimate'), 4, language)}",
                f"- IV p 值：{format_p_value(model_summary.get('iv_p_value'), language)}",
                f"- 第一阶段 R²：{format_number(model_summary.get('first_stage_r_squared'), 4, language)}",
                f"- 第一阶段 F 统计量：{format_number(model_summary.get('first_stage_f_statistic'), 4, language)}",
                f"- 工具变量数量：{model_summary.get('instrument_count', len(model_summary.get('instruments', [])))}",
            ]
        return [
            f"- Observations: {model_summary.get('n_obs', 'N/A')}",
            f"- Second-stage R-squared: {format_number(model_summary.get('r_squared'), 4, language)}",
            f"- Second-stage adjusted R-squared: {format_number(model_summary.get('adj_r_squared'), 4, language)}",
            f"- IV term: {model_summary.get('iv_term', 'N/A')}",
            f"- IV estimate: {format_number(model_summary.get('iv_estimate'), 4, language)}",
            f"- IV p-value: {format_p_value(model_summary.get('iv_p_value'), language)}",
            f"- First-stage R-squared: {format_number(model_summary.get('first_stage_r_squared'), 4, language)}",
            f"- First-stage F-statistic: {format_number(model_summary.get('first_stage_f_statistic'), 4, language)}",
            f"- Instrument count: {model_summary.get('instrument_count', len(model_summary.get('instruments', [])))}",
        ]
    if model_type == "did":
        did_term = model_summary.get("did_term", "treatment:post")
        if language == "zh":
            return [
                f"- 观测值：{model_summary.get('n_obs', 'N/A')}",
                f"- 解释变量数量：{model_summary.get('num_predictors', len(model_summary.get('independent_variables', [])))}",
                f"- 回归 R²：{format_number(model_summary.get('r_squared'), 4, language)}",
                f"- 调整 R²：{format_number(model_summary.get('adj_r_squared'), 4, language)}",
                f"- DID 交互项：{did_term}",
                f"- DID 估计：{format_number(model_summary.get('did_estimate'), 4, language)}",
                f"- DID p 值：{format_p_value(model_summary.get('did_p_value'), language)}",
                f"- 标准误：{model_summary.get('standard_errors', 'hc3')}",
            ]
        return [
            f"- Observations: {model_summary.get('n_obs', 'N/A')}",
            f"- Number of explanatory variables: {model_summary.get('num_predictors', len(model_summary.get('independent_variables', [])))}",
            f"- R-squared: {format_number(model_summary.get('r_squared'), 4, language)}",
            f"- Adjusted R-squared: {format_number(model_summary.get('adj_r_squared'), 4, language)}",
            f"- DID term: {did_term}",
            f"- DID estimate: {format_number(model_summary.get('did_estimate'), 4, language)}",
            f"- DID p-value: {format_p_value(model_summary.get('did_p_value'), language)}",
            f"- Standard errors: {model_summary.get('standard_errors', 'hc3')}",
        ]
    if model_type == "panel_fe":
        fe_parts = []
        if model_summary.get("entity_effects"):
            fe_parts.append("个体固定效应" if language == "zh" else "entity fixed effects")
        if model_summary.get("time_effects"):
            fe_parts.append("时间固定效应" if language == "zh" else "time fixed effects")
        se_label = model_summary.get("standard_errors", "cluster_entity")
        if language == "zh":
            se_text = {"cluster_entity": "按个体聚类", "robust": "稳健", "conventional": "常规"}.get(se_label, str(se_label))
            return [
                f"- 观测值：{model_summary.get('n_obs', 'N/A')}",
                f"- 个体数量：{model_summary.get('entities', 'N/A')}",
                f"- 时间期数：{model_summary.get('time_periods', 'N/A')}",
                f"- 固定效应：{' + '.join(fe_parts) if fe_parts else '无'}",
                f"- 标准误：{se_text}",
                f"- 组内 R²：{format_number(model_summary.get('r_squared_within'), 4, language)}",
                f"- 组间 R²：{format_number(model_summary.get('r_squared_between'), 4, language)}",
                f"- 总体 R²：{format_number(model_summary.get('r_squared_overall'), 4, language)}",
            ]
        se_text = {"cluster_entity": "cluster by entity", "robust": "robust", "conventional": "conventional"}.get(se_label, str(se_label))
        return [
            f"- Observations: {model_summary.get('n_obs', 'N/A')}",
            f"- Entities: {model_summary.get('entities', 'N/A')}",
            f"- Time periods: {model_summary.get('time_periods', 'N/A')}",
            f"- Fixed effects: {' + '.join(fe_parts) if fe_parts else 'none'}",
            f"- Standard errors: {se_text}",
            f"- Within R-squared: {format_number(model_summary.get('r_squared_within'), 4, language)}",
            f"- Between R-squared: {format_number(model_summary.get('r_squared_between'), 4, language)}",
            f"- Overall R-squared: {format_number(model_summary.get('r_squared_overall'), 4, language)}",
        ]
    if model_type in {"logit", "probit"}:
        if language == "zh":
            return [
                f"- 观测值：{model_summary.get('n_obs', 'N/A')}",
                f"- 解释变量数量：{model_summary.get('num_predictors', len(model_summary.get('independent_variables', [])))}",
                f"- McFadden 伪 R²：{format_number(model_summary.get('pseudo_r_squared'), 4, language)}",
                f"- 对数似然：{format_number(model_summary.get('log_likelihood'), 4, language)}",
                f"- AIC：{format_number(model_summary.get('aic'), 2, language)}",
                f"- BIC：{format_number(model_summary.get('bic'), 2, language)}",
                "- 提醒：伪 R² 不能直接与 OLS R² 比较；AIC/BIC 主要用于同类模型比较。",
            ]
        return [
            f"- Observations: {model_summary.get('n_obs', 'N/A')}",
            f"- Number of explanatory variables: {model_summary.get('num_predictors', len(model_summary.get('independent_variables', [])))}",
            f"- McFadden pseudo R-squared: {format_number(model_summary.get('pseudo_r_squared'), 4, language)}",
            f"- Log-likelihood: {format_number(model_summary.get('log_likelihood'), 4, language)}",
            f"- AIC: {format_number(model_summary.get('aic'), 2, language)}",
            f"- BIC: {format_number(model_summary.get('bic'), 2, language)}",
            "- Note: pseudo R-squared is not directly comparable to OLS R-squared; AIC/BIC are mainly for comparing models of the same type.",
        ]

    if language == "zh":
        return [
            f"- R²：{format_number(model_summary.get('r_squared'), 4, language)}",
            f"- 调整 R²：{format_number(model_summary.get('adj_r_squared'), 4, language)}",
            f"- 观测值：{model_summary.get('n_obs', 'N/A')}",
            f"- F 统计量：{format_number(model_summary.get('f_statistic'), 4, language)}",
            f"- F 检验 p 值：{format_p_value(model_summary.get('f_pvalue'), language)}",
        ]
    return [
        f"- R-squared: {format_number(model_summary.get('r_squared'), 4, language)}",
        f"- Adjusted R-squared: {format_number(model_summary.get('adj_r_squared'), 4, language)}",
        f"- Observations: {model_summary.get('n_obs', 'N/A')}",
        f"- F-statistic: {format_number(model_summary.get('f_statistic'), 4, language)}",
        f"- F-test p-value: {format_p_value(model_summary.get('f_pvalue'), language)}",
    ]


def _warning_lines(warnings: list[str], language: str) -> str:
    if not warnings:
        return "- No major warnings were generated." if language != "zh" else "- 未生成主要提醒。"
    translated = [translate_warning(language, warning) for warning in warnings]
    return "\n".join([f"- {warning}" for warning in list(dict.fromkeys(translated))])


def _structured_diagnostics(model_results: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not model_results:
        return []
    return diagnostic_dicts(model_results.get("structured_diagnostics") or [])


def _structured_warning_lines(model_results: dict[str, Any] | None, warnings: list[str], language: str) -> str:
    lines = warning_lines_for_display(
        (model_results or {}).get("structured_diagnostics") if model_results else [],
        warnings,
        language,
        report_only=True,
    )
    if not lines:
        return _warning_lines(warnings, language)
    return "\n".join([f"- {message}" for message in lines])


def _structured_diagnostics_markdown(model_results: dict[str, Any] | None, language: str) -> str:
    diagnostics = [item for item in _structured_diagnostics(model_results) if item.get("show_in_report", True)]
    if not diagnostics:
        return "_No structured diagnostics available._" if language != "zh" else "_暂无结构化诊断信息。_"
    rows = []
    for item in diagnostics:
        code = str(item.get("code") or "")
        rows.append(
            {
                "severity": item.get("severity", ""),
                "code": code,
                "title": translate_diagnostic_field(language, code, "title", str(item.get("title") or "")),
                "message": translate_diagnostic_field(language, code, "message", str(item.get("message") or "")),
                "affected_variables": ", ".join(item.get("affected_variables") or []),
                "recommendation": translate_diagnostic_field(language, code, "recommendation", str(item.get("recommendation") or "")),
            }
        )
    return _markdown_table(pd.DataFrame(rows), language)


def _advanced_outputs(model_results: dict[str, Any] | None) -> dict[str, Any]:
    if not model_results:
        return {}
    return dict(model_results.get("advanced_outputs") or {})



def _advanced_outputs_markdown(model_results: dict[str, Any] | None, language: str) -> str:
    outputs = _advanced_outputs(model_results)
    sections: list[str] = []
    psm_summary = outputs.get("psm_summary")
    if isinstance(psm_summary, dict) and psm_summary:
        title = "PSM ATT 摘要" if language == "zh" else "PSM ATT Summary"
        note = (
            "PSM 摘要记录在所选匹配设定下得到的 ATT。当前最小实现使用 1:1 最近邻有放回匹配；该结果只改善观测协变量上的可比性，不能处理未观测混杂。"
            if language == "zh"
            else "The PSM summary records ATT under the selected matching specification. The current minimal implementation uses 1:1 nearest-neighbor matching with replacement; it improves comparability on observed covariates only and does not address unobserved confounding."
        )
        sections.append(f"### {title}\n\n{note}\n\n{_markdown_table(pd.DataFrame([psm_summary]), language)}")
    propensity_summary = outputs.get("propensity_score_summary")
    if isinstance(propensity_summary, dict) and propensity_summary:
        title = "倾向得分摘要" if language == "zh" else "Propensity Score Summary"
        sections.append(f"### {title}\n\n{_markdown_table(pd.DataFrame([propensity_summary]), language)}")
    balance_overview = outputs.get("psm_balance_overview")
    if isinstance(balance_overview, dict) and balance_overview:
        title = "匹配平衡概览" if language == "zh" else "Matching Balance Overview"
        note = (
            "平衡概览基于匹配前后的绝对标准化均值差；解释 ATT 前应检查匹配后不平衡是否仍然较高。"
            if language == "zh"
            else "The balance overview is based on before/after absolute standardized mean differences; residual imbalance should be reviewed before interpreting the ATT."
        )
        sections.append(f"### {title}\n\n{note}\n\n{_markdown_table(pd.DataFrame([balance_overview]), language)}")
    balance_summary = outputs.get("balance_summary")
    if isinstance(balance_summary, list) and balance_summary:
        title = "匹配平衡摘要" if language == "zh" else "Matching Balance Summary"
        sections.append(f"### {title}\n\n{_markdown_table(pd.DataFrame(balance_summary), language)}")
    iv_summary = outputs.get("iv_summary")
    if isinstance(iv_summary, dict) and iv_summary:
        title = "IV/2SLS 估计摘要" if language == "zh" else "IV/2SLS Estimate Summary"
        note = (
            "IV/2SLS 摘要记录第二阶段拟合内生变量的系数；因果解释依赖工具变量相关性和排除限制等识别假设。"
            if language == "zh"
            else "The IV/2SLS summary records the second-stage coefficient on the fitted endogenous variable; causal interpretation depends on instrument relevance and exclusion-restriction assumptions."
        )
        sections.append(f"### {title}\n\n{note}\n\n{_markdown_table(pd.DataFrame([iv_summary]), language)}")
    did_summary = outputs.get("did_summary")
    if isinstance(did_summary, dict) and did_summary:
        title = "DID 估计摘要" if language == "zh" else "DID Estimate Summary"
        note = (
            "DID 估计摘要记录处理组与处理后时期交互项的系数；因果解释依赖平行趋势等研究设计假设。"
            if language == "zh"
            else "The DID summary records the treatment-by-post coefficient; causal interpretation depends on research-design assumptions such as parallel trends."
        )
        summary_fields = {key: value for key, value in did_summary.items() if key != "cell_counts"}
        block = f"### {title}\n\n{note}\n\n{_markdown_table(pd.DataFrame([summary_fields]), language)}"
        cell_counts = did_summary.get("cell_counts")
        if isinstance(cell_counts, list) and cell_counts:
            cell_title = "DID 单元支持" if language == "zh" else "DID Cell Support"
            block += f"\n\n{cell_title}:\n\n{_markdown_table(pd.DataFrame(cell_counts), language)}"
        sections.append(block)
    odds_ratio_table = outputs.get("odds_ratio_table")
    if isinstance(odds_ratio_table, pd.DataFrame) and not odds_ratio_table.empty:
        title = "Logit 胜算比" if language == "zh" else "Logit Odds Ratios"
        note = (
            "胜算比用于胜算尺度解释；大于 1 表示因变量取 1 的胜算更高，小于 1 表示胜算更低。它不是概率百分点变化。"
            if language == "zh"
            else "Odds ratios are odds-scale outputs. Values above 1 indicate higher odds of Y=1 and values below 1 indicate lower odds; they are not percentage-point probability changes."
        )
        sections.append(f"### {title}\n\n{note}\n\n{_markdown_table(format_regression_table(odds_ratio_table, language), language)}")
    marginal_effects_table = outputs.get("marginal_effects_table")
    if isinstance(marginal_effects_table, pd.DataFrame) and not marginal_effects_table.empty:
        title = "平均边际效应" if language == "zh" else "Average Marginal Effects"
        note = (
            "边际效应用于概率尺度的条件相关解释，并不自动代表因果关系。"
            if language == "zh"
            else "Marginal effects provide probability-scale association estimates and do not establish causality."
        )
        sections.append(f"### {title}\n\n{note}\n\n{_markdown_table(format_regression_table(marginal_effects_table, language), language)}")
    heteroskedasticity = (model_results or {}).get("diagnostics", {}).get("heteroskedasticity") if model_results else None
    if isinstance(heteroskedasticity, dict) and heteroskedasticity:
        title = "异方差诊断" if language == "zh" else "Heteroskedasticity Diagnostic"
        note = (
            "Breusch-Pagan 检验用于识别异方差迹象；显著结果不表示其他模型问题已被解决。"
            if language == "zh"
            else "The Breusch-Pagan test checks for evidence of heteroskedasticity; a significant result does not resolve other model issues."
        )
        frame = pd.DataFrame([heteroskedasticity])
        sections.append(f"### {title}\n\n{note}\n\n{_markdown_table(format_diagnostic_table(frame, language, digits=3), language)}")
    if not sections:
        return "_No advanced outputs available._" if language != "zh" else "_暂无高级输出。_"
    return "\n\n".join(sections)


def _categorical_control_summary(model_results: dict[str, Any] | None, regression_table: pd.DataFrame | None, language: str, detailed: bool = False) -> str:
    if not model_results:
        return "_No categorical controls were encoded._" if language != "zh" else "_未编码分类控制变量。_"
    info = model_results.get("encoding_info", {})
    refs = info.get("reference_categories", {})
    dummies = info.get("dummy_variables", [])
    encoded = info.get("encoded_categorical_controls", [])
    ignored = info.get("ignored_categorical_controls", [])
    lines: list[str] = []

    if ignored:
        if language == "zh":
            lines.append("所选分类控制变量未编码且未纳入模型：" + ", ".join(ignored))
        else:
            lines.append("Selected categorical controls were not encoded and were not included: " + ", ".join(ignored))

    if not encoded:
        return "\n".join([f"- {line}" for line in lines]) if lines else ("_No categorical controls were encoded._" if language != "zh" else "_未编码分类控制变量。_")

    significant_dummies: set[str] = set()
    if regression_table is not None and not regression_table.empty and "p_value" in regression_table.columns:
        dummy_rows = regression_table[regression_table["variable"].isin(dummies)]
        significant_dummies = set(dummy_rows[dummy_rows["p_value"] < 0.10]["variable"].astype(str).tolist())

    for column in encoded:
        prefix = f"{column}_"
        generated = [dummy for dummy in dummies if str(dummy).startswith(prefix)]
        sig_count = len([dummy for dummy in generated if dummy in significant_dummies])
        reference = refs.get(column, "N/A")
        if language == "zh":
            lines.append(
                f"{column} 作为分类控制变量纳入模型，参考类别为 {reference}；"
                f"生成 {len(generated)} 个虚拟变量，其中 {sig_count} 个在 10% 水平显著。"
                f"参考：{column}: {reference}。虚拟变量使用 drop_first=True 生成。"
                "虚拟变量系数相对于参考类别，通常不逐一作为核心发现解释。"
            )
        else:
            lines.append(
                f"{column} was included as a categorical control with {reference} as the reference category; "
                f"{len(generated)} dummy variable(s) were generated and {sig_count} are significant at the 10% level. "
                f"Reference: {column}: {reference}. dummy variables were generated with drop_first=True. "
                "Dummy coefficients are relative to the reference category and are usually not interpreted one by one as main findings."
            )
        if detailed and generated:
            label = "生成的虚拟变量" if language == "zh" else "Generated dummy variables"
            lines.append(f"{label}: " + ", ".join(generated))

    return "\n".join([f"- {line}" for line in lines])

def build_categorical_control_summary(
    model_results: dict[str, Any] | None,
    regression_table: pd.DataFrame | None,
    language: str = "en",
    detailed: bool = False,
) -> str:
    return _categorical_control_summary(model_results, regression_table, normalize_language(language), detailed=detailed)



def _main_findings(regression_table: pd.DataFrame, model_summary: dict[str, Any], language: str) -> str:
    if regression_table is None or regression_table.empty:
        return "No model results are available." if language != "zh" else "无可用模型结果。"
    model_type = _model_type(model_summary)
    if model_type == "psm":
        att = format_number(model_summary.get("att_estimate"), 4, language)
        matched = model_summary.get("matched_treated_count", "N/A")
        balance_after = model_summary.get("max_absolute_smd_after")
        balance_sentence = ""
        if balance_after is not None:
            balance_sentence = (
                f" 匹配后最大绝对标准化均值差为 {format_number(balance_after, 4, language)}，解释前应检查协变量平衡。"
                if language == "zh"
                else f" The maximum post-match absolute SMD is {format_number(balance_after, 4, language)}, so covariate balance should be reviewed before interpretation."
            )
        if language == "zh":
            return (
                f"当前最近邻匹配设定下，PSM ATT 估计为 {att}，基于 {matched} 个已匹配处理组观测。"
                "该结果反映观测协变量匹配后的差异，不能自动证明因果关系。"
                f"{balance_sentence}"
            )
        return (
            f"Under the current nearest-neighbor matching specification, the PSM ATT estimate is {att}, based on {matched} matched treated observation(s). "
            "This reflects a matched difference on observed covariates and does not automatically establish causality."
            f"{balance_sentence}"
        )
    main_vars = list(model_summary.get("main_independent_variables") or [])
    numeric_controls = set(model_summary.get("numeric_control_variables") or [])
    table = regression_table[regression_table["variable"].isin(main_vars)].copy()

    lines: list[str] = []
    if table.empty:
        lines.append("No main explanatory variables were found in the coefficient table." if language != "zh" else "系数表中未找到主要解释变量。")
    else:
        for _, row in table.iterrows():
            direction = "positive" if row["coefficient"] > 0 else "negative"
            zh_direction = "正向" if row["coefficient"] > 0 else "负向"
            if model_type == "did":
                if language == "zh":
                    lines.append(f"{row['variable']} 的 DID 估计为 {format_number(row['coefficient'], 4, language)}，方向为{zh_direction}，{format_p_narrative(row['p_value'], language)}。")
                else:
                    lines.append(f"{row['variable']} has a {direction} DID estimate ({format_number(row['coefficient'], 4, language)}), {format_p_narrative(row['p_value'], language)}.")
            elif model_type == "iv_2sls":
                if language == "zh":
                    lines.append(f"{row['variable']} 的 IV/2SLS 条件估计为 {format_number(row['coefficient'], 4, language)}，方向为{zh_direction}，{format_p_narrative(row['p_value'], language)}。")
                else:
                    lines.append(f"{row['variable']} has a {direction} IV/2SLS conditional estimate ({format_number(row['coefficient'], 4, language)}), {format_p_narrative(row['p_value'], language)}.")
            elif model_type == "panel_fe":
                if language == "zh":
                    direction_text = "正相关" if row["coefficient"] > 0 else "负相关"
                    lines.append(f"控制固定效应后，{row['variable']} 与 {model_summary.get('dependent_variable', 'Y')} 的个体内部变化呈{direction_text}，{format_p_narrative(row['p_value'], language)}。")
                else:
                    direction_text = "positively" if row["coefficient"] > 0 else "negatively"
                    lines.append(f"After controlling for fixed effects, {row['variable']} is {direction_text} associated with within-entity changes in {model_summary.get('dependent_variable', 'Y')}, {format_p_narrative(row['p_value'], language)}.")
            elif language == "zh":
                lines.append(f"{row['variable']} 的系数为 {format_number(row['coefficient'], 4, language)}，方向为{zh_direction}，{format_p_narrative(row['p_value'], language)}。")
            else:
                lines.append(f"{row['variable']} has a {direction} coefficient ({format_number(row['coefficient'], 4, language)}), {format_p_narrative(row['p_value'], language)}.")

    control_table = regression_table[regression_table["variable"].isin(numeric_controls)]
    if not control_table.empty and (control_table["p_value"] < 0.10).any():
        lines.append("Some controls are statistically significant." if language != "zh" else "部分控制变量在统计上显著。")

    if model_type == "panel_fe":
        lines.append(
            "Panel Fixed Effects estimates are interpreted as within-entity changes over time, after accounting for selected fixed effects."
            if language != "zh"
            else "面板固定效应估计应解释为控制所选固定效应后的个体内部随时间变化关系。"
        )
    elif model_type == "did":
        lines.append(
            "The DID coefficient is a conditional difference-in-differences estimate. Causal interpretation depends on research design assumptions, especially parallel trends."
            if language != "zh"
            else "DID 系数是条件双重差分估计；因果解释依赖研究设计假设，尤其是平行趋势。"
        )
    elif model_type == "iv_2sls":
        lines.append(
            "The IV/2SLS coefficient is a conditional estimate under the supplied instrument specification. Causal interpretation depends on instrument relevance and the exclusion restriction."
            if language != "zh"
            else "IV/2SLS 系数是在给定工具变量设定下的条件估计；因果解释依赖工具变量相关性和排除限制。"
        )
    elif model_type == "psm":
        lines.append(
            "PSM improves comparability on observed covariates only; it does not address unobserved confounding."
            if language != "zh"
            else "PSM 仅改善观测协变量上的可比性，不能处理未观测混杂。"
        )
    elif model_type == "logit":
        lines.append(
            "In a Logit model, coefficient signs indicate higher or lower likelihood of Y=1; coefficients are log-odds, not percentage-point probability changes."
            if language != "zh"
            else "在 Logit 模型中，系数方向表示因变量取 1 的可能性倾向；系数是对数胜算单位，不能直接解释为概率百分点变化。"
        )
    elif model_type == "probit":
        lines.append(
            "In a Probit model, coefficient signs indicate higher or lower latent propensity for Y=1; coefficients are latent-index units, not direct probability percentage-point changes."
            if language != "zh"
            else "在 Probit 模型中，系数方向表示因变量取 1 的潜在倾向；系数是潜变量指数单位，不是概率百分点变化。"
        )
    else:
        lines.append(
            "OLS coefficients describe conditional associations: the estimated change in Y associated with a one-unit increase in X, holding other variables constant."
            if language != "zh"
            else "OLS 系数描述条件相关关系：在其他变量不变时，X 增加一个单位与 Y 的估计变化相关。"
        )
    return "\n".join([f"- {line}" for line in lines])


def _model_spec_summary(model_summary: dict[str, Any], language: str) -> str:
    main_x = model_summary.get("main_independent_variables") or []
    controls = model_summary.get("numeric_control_variables") or []
    cats = model_summary.get("categorical_control_variables") or []
    model_type = _model_type(model_summary)
    if language == "zh":
        none_text = "无"
        lines = [
            f"- 因变量：{model_summary.get('dependent_variable', 'N/A')}",
            f"- 主要解释变量：{', '.join(main_x) if main_x else none_text}",
            f"- 数值控制变量：{', '.join(controls) if controls else none_text}",
            f"- 分类控制变量：{', '.join(cats) if cats else none_text}",
        ]
        if model_type == "panel_fe":
            fe_parts = []
            if model_summary.get("entity_effects"):
                fe_parts.append("个体固定效应")
            if model_summary.get("time_effects"):
                fe_parts.append("时间固定效应")
            lines.extend([
                f"- 个体 ID：{model_summary.get('entity_id', 'N/A')}",
                f"- 时间变量：{model_summary.get('time_id', 'N/A')}",
                f"- 固定效应：{' + '.join(fe_parts) if fe_parts else none_text}",
                f"- 标准误：{model_summary.get('standard_errors', 'cluster_entity')}",
            ])
        if model_type == "did":
            lines.extend([
                f"- 处理变量：{model_summary.get('treatment_variable', 'N/A')}",
                f"- 处理后时期变量：{model_summary.get('post_variable', 'N/A')}",
                f"- DID 交互项：{model_summary.get('did_term', 'N/A')}",
                f"- 个体/分组变量：{model_summary.get('entity_id') or model_summary.get('group_variable') or none_text}",
                f"- 时间变量：{model_summary.get('time_id', none_text)}",
                f"- 标准误：{model_summary.get('standard_errors', 'hc3')}",
            ])
        if model_type == "iv_2sls":
            lines.extend([
                f"- 内生变量：{model_summary.get('endogenous_variable', 'N/A')}",
                f"- 工具变量：{', '.join(model_summary.get('instruments') or []) or none_text}",
                f"- 外生控制变量：{', '.join(model_summary.get('exogenous_controls') or model_summary.get('numeric_control_variables') or []) or none_text}",
                f"- IV 项：{model_summary.get('iv_term', 'N/A')}",
            ])
        if model_type == "psm":
            lines.extend([
                f"- 处理变量：{model_summary.get('treatment_variable', 'N/A')}",
                f"- 匹配协变量：{', '.join(model_summary.get('matching_covariates') or []) or none_text}",
                f"- 估计量：{model_summary.get('psm_estimand', 'ATT')}",
                f"- 匹配方法：{model_summary.get('matching_method', 'nearest_neighbor')}",
                f"- 有放回匹配：{'是' if model_summary.get('replacement_matching') else '否'}",
                f"- 卡尺：{model_summary.get('caliper') if model_summary.get('caliper') is not None else none_text}",
            ])
        return "\n".join(lines)
    lines = [
        f"- Dependent variable: {model_summary.get('dependent_variable', 'N/A')}",
        f"- Main explanatory variables: {', '.join(main_x) if main_x else 'none'}",
        f"- Numeric controls: {', '.join(controls) if controls else 'none'}",
        f"- Categorical controls: {', '.join(cats) if cats else 'none'}",
    ]
    if model_type == "panel_fe":
        fe_parts = []
        if model_summary.get("entity_effects"):
            fe_parts.append("entity fixed effects")
        if model_summary.get("time_effects"):
            fe_parts.append("time fixed effects")
        lines.extend([
            f"- Entity ID: {model_summary.get('entity_id', 'N/A')}",
            f"- Time variable: {model_summary.get('time_id', 'N/A')}",
            f"- Fixed effects: {' + '.join(fe_parts) if fe_parts else 'none'}",
            f"- Standard errors: {model_summary.get('standard_errors', 'cluster_entity')}",
        ])
    if model_type == "did":
        lines.extend([
            f"- Treatment variable: {model_summary.get('treatment_variable', 'N/A')}",
            f"- Post variable: {model_summary.get('post_variable', 'N/A')}",
            f"- DID term: {model_summary.get('did_term', 'N/A')}",
            f"- Entity/group variable: {model_summary.get('entity_id') or model_summary.get('group_variable') or 'none'}",
            f"- Time variable: {model_summary.get('time_id', 'none')}",
            f"- Standard errors: {model_summary.get('standard_errors', 'hc3')}",
        ])
    if model_type == "iv_2sls":
        lines.extend([
            f"- Endogenous variable: {model_summary.get('endogenous_variable', 'N/A')}",
            f"- Instruments: {', '.join(model_summary.get('instruments') or []) or 'none'}",
            f"- Exogenous controls: {', '.join(model_summary.get('exogenous_controls') or model_summary.get('numeric_control_variables') or []) or 'none'}",
            f"- IV term: {model_summary.get('iv_term', 'N/A')}",
        ])
    if model_type == "psm":
        lines.extend([
            f"- Treatment variable: {model_summary.get('treatment_variable', 'N/A')}",
            f"- Matching covariates: {', '.join(model_summary.get('matching_covariates') or []) or 'none'}",
            f"- Estimand: {model_summary.get('psm_estimand', 'ATT')}",
            f"- Matching method: {model_summary.get('matching_method', 'nearest_neighbor')}",
            f"- Replacement matching: {'yes' if model_summary.get('replacement_matching') else 'no'}",
            f"- Caliper: {model_summary.get('caliper') if model_summary.get('caliper') is not None else 'none'}",
        ])
    return "\n".join(lines)


def _panel_structure_summary(model_summary: dict[str, Any], language: str) -> str:
    if _model_type(model_summary) != "panel_fe":
        return ""
    structure = model_summary.get("panel_structure") or {}
    if not structure:
        return "_No panel structure information available._" if language != "zh" else "_没有可用的面板结构信息。_"
    labels = {
        "observations": "观测值数量" if language == "zh" else "Observations",
        "entities": "个体数量" if language == "zh" else "Entities",
        "time_periods": "时间期数" if language == "zh" else "Time periods",
        "balanced_panel": "是否平衡面板" if language == "zh" else "Balanced panel",
        "duplicate_entity_time_rows": "重复个体-时间观测" if language == "zh" else "Duplicate entity-time rows",
        "missing_entity_ids": "缺失个体 ID" if language == "zh" else "Missing entity IDs",
        "missing_time_ids": "缺失时间 ID" if language == "zh" else "Missing time IDs",
        "singleton_entities": "单期个体数量" if language == "zh" else "Singleton entities",
        "min_observations_per_entity": "每个个体的最少观测期数" if language == "zh" else "Minimum observations per entity",
        "max_observations_per_entity": "每个个体的最多观测期数" if language == "zh" else "Maximum observations per entity",
        "average_observations_per_entity": "每个个体的平均观测期数" if language == "zh" else "Average observations per entity",
    }
    frame = pd.DataFrame(
        {
            "item": list(labels.values()),
            "value": [structure.get(key, "") for key in labels.keys()],
        }
    )
    return _markdown_table(frame, language)

def _data_quality_payloads(profile: dict[str, Any] | None, model_results: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    results = model_results or {}
    quality = dict(results.get("data_quality_profile") or {})
    missingness = dict(results.get("missingness_profile") or {})
    variable_quality = list(results.get("variable_quality_summary") or [])
    pre_model_risk = dict(results.get("pre_model_risk_profile") or {})
    if not missingness and profile:
        missing_counts = dict(profile.get("missing_counts") or {})
        missing_percentages = dict(profile.get("missing_percentages") or {})
        high_missing = [variable for variable, pct in missing_percentages.items() if float(pct or 0) > 20]
        missingness = {
            "complete_case_rows": None,
            "total_missing_percentage": None,
            "high_missing_variables": high_missing,
            "missing_by_variable": [
                {
                    "variable": variable,
                    "missing_count": missing_counts.get(variable, 0),
                    "missing_percentage": missing_percentages.get(variable, 0),
                }
                for variable in missing_counts
            ],
        }
    return quality, missingness, variable_quality, pre_model_risk



def _data_quality_report_markdown(
    profile: dict[str, Any] | None,
    model_results: dict[str, Any] | None,
    language: str,
    simple: bool = False,
) -> str:
    quality, missingness, variable_quality, pre_model_risk = _data_quality_payloads(profile, model_results)
    high_missing = list(missingness.get("high_missing_variables") or [])
    constant = list(quality.get("constant_columns") or [])
    near_constant = list(quality.get("near_constant_columns") or [])
    complete_case_rows = missingness.get("complete_case_rows")
    total_missing_pct = missingness.get("total_missing_percentage")
    risk_items = [item for item in pre_model_risk.get("risk_items", []) if item.get("show_in_report", True)]
    important_risks = [item for item in risk_items if item.get("severity") in {"warning", "error"}]
    if simple:
        lines = []
        if high_missing:
            lines.append(
                "- 高缺失变量：" + ", ".join(high_missing)
                if language == "zh"
                else "- High-missing variables: " + ", ".join(high_missing)
            )
        if important_risks:
            labels = [str(item.get("title") or item.get("code")) for item in important_risks[:3]]
            lines.append(
                "- 运行前风险：" + "; ".join(labels)
                if language == "zh"
                else "- Pre-model risks: " + "; ".join(labels)
            )
        if not lines:
            lines.append(
                "- 未记录主要数据质量提醒。"
                if language == "zh"
                else "- No major data-quality warnings were recorded."
            )
        heading = "重要数据质量提醒" if language == "zh" else "Important data-quality notes"
        return f"**{heading}**\n" + "\n".join(lines)

    overview = pd.DataFrame(
        [
            {"metric": "complete_case_rows", "value": complete_case_rows if complete_case_rows is not None else "N/A"},
            {"metric": "total_missing_percentage", "value": total_missing_pct if total_missing_pct is not None else "N/A"},
            {"metric": "high_missing_variable_count", "value": len(high_missing)},
            {"metric": "constant_variable_count", "value": len(constant)},
            {"metric": "near_constant_variable_count", "value": len(near_constant)},
            {"metric": "pre_model_risk_item_count", "value": len(risk_items)},
        ]
    )
    risk_frame = pd.DataFrame(
        [
            {
                "code": item.get("code"),
                "severity": item.get("severity"),
                "affected_variables": ", ".join(item.get("affected_variables") or []),
            }
            for item in risk_items
        ]
    )
    variable_quality_frame = pd.DataFrame(variable_quality)
    if not variable_quality_frame.empty:
        columns = [
            "variable",
            "dtype",
            "inferred_role_hint",
            "missing_percentage",
            "unique_count",
            "is_constant",
            "is_near_constant",
            "is_high_cardinality",
            "is_text_numeric_like",
        ]
        variable_quality_frame = variable_quality_frame[[column for column in columns if column in variable_quality_frame.columns]]

    if language == "zh":
        return f"""### 数据质量诊断
数据质量诊断仅用于识别缺失、常数变量、高基数类别变量和运行前风险；不会自动修改数据。

{_markdown_table(overview, language)}

高缺失变量：{", ".join(high_missing) if high_missing else "无"}

常数或近似常数变量：{", ".join(constant + near_constant) if constant or near_constant else "无"}

运行前风险：

{_markdown_table(risk_frame, language)}

变量质量摘要：

{_markdown_table(variable_quality_frame, language)}
"""

    return f"""### Data Quality Diagnostics
Data quality diagnostics identify missingness, constant variables, high-cardinality categorical variables, and pre-model risks. They do not modify the data automatically.

{_markdown_table(overview, language)}

High-missing variables: {", ".join(high_missing) if high_missing else "none"}

Constant or near-constant variables: {", ".join(constant + near_constant) if constant or near_constant else "none"}

Pre-model risks:

{_markdown_table(risk_frame, language)}

Variable quality summary:

{_markdown_table(variable_quality_frame, language)}
"""

def _missing_data_handling_payload(model_results: dict[str, Any] | None) -> dict[str, Any]:
    return dict((model_results or {}).get("missing_data_handling_result") or {})


def _missing_data_handling_report_markdown(
    model_results: dict[str, Any] | None,
    language: str,
    simple: bool = False,
) -> str:
    handling = _missing_data_handling_payload(model_results)
    action_results = [dict(item) for item in handling.get("action_results", []) if isinstance(item, dict)]
    if not handling or not action_results:
        return ""

    rows_dropped = sum(int(item.get("rows_dropped") or 0) for item in action_results)
    values_filled = sum(int(item.get("values_filled") or 0) for item in action_results)
    indicator_count = sum(1 for item in action_results if item.get("indicator_variable_created"))
    section = get_text(language, "missing_data_handling_section")
    note = get_text(language, "missing_data_handling_note")
    caution = get_text(language, "missing_data_handling_caution")
    if simple:
        rows_label = get_text(language, "missing_data_handling_rows_dropped")
        filled_label = get_text(language, "missing_data_handling_values_filled")
        indicators_label = get_text(language, "missing_data_handling_indicators_created")
        return (
            f"**{section}**\n"
            f"- {note}\n"
            f"- {rows_label}: {rows_dropped}; {filled_label}: {values_filled}; {indicators_label}: {indicator_count}."
        )

    overview = pd.DataFrame(
        [
            {"metric": get_text(language, "missing_data_handling_actions_applied"), "value": len(action_results)},
            {"metric": get_text(language, "missing_data_handling_rows_dropped"), "value": rows_dropped},
            {"metric": get_text(language, "missing_data_handling_values_filled"), "value": values_filled},
            {"metric": get_text(language, "missing_data_handling_indicators_created"), "value": indicator_count},
        ]
    )
    details = pd.DataFrame(action_results)
    columns = [
        "variable",
        "strategy_id",
        "rows_affected",
        "values_filled",
        "rows_dropped",
        "indicator_variable_created",
        "fill_value",
        "message_code",
    ]
    details = details[[column for column in columns if column in details.columns]]
    return f"""### {section}
{note}

{caution}

{_markdown_table(overview, language)}

{_markdown_table(details, language)}
"""



def _clean_zh_report_text(text: str) -> str:
    return text

def generate_simple_report(
    regression_table: pd.DataFrame,
    model_summary: dict[str, Any],
    warnings: list[str],
    language: str = "en",
    model_metadata: Any | None = None,
    model_results: dict[str, Any] | None = None,
    profile: dict[str, Any] | None = None,
    variable_roles: dict[str, str] | None = None,
    guided_workflow_result: Any | None = None,
) -> str:
    lang = normalize_language(language)
    model_type = _model_type(model_summary)
    model_label = model_metadata.report_label(lang) if model_metadata is not None and hasattr(model_metadata, "report_label") else model_type.upper()
    fit_lines = "\n".join(_fit_summary_lines(model_summary, lang))
    role_text, _ = _role_summary(variable_roles, lang)
    stats_summary = _compact_stats_summary(profile, variable_roles, lang) if profile else ""
    equation = _compact_equation(model_summary)
    equation_note = _equation_note(model_summary, lang)
    panel_structure_summary = _panel_structure_summary(model_summary, lang)
    panel_structure_block = ""
    if model_type == "panel_fe" and panel_structure_summary:
        panel_structure_label = "面板结构：" if lang == "zh" else "Panel structure:"
        panel_structure_block = f"{panel_structure_label}\n{panel_structure_summary}"
    workflow_section = workflow_summary_markdown(guided_workflow_result, lang, detailed=False) if guided_workflow_result else ""
    narrative = generate_narrative(
        regression_table,
        model_summary,
        warnings,
        language=lang,
        workflow_result=guided_workflow_result,
        variable_roles=variable_roles,
        structured_diagnostics=_structured_diagnostics(model_results),
        advanced_outputs=_advanced_outputs(model_results),
    )
    narrative_section = narrative_markdown(narrative, simple=True) if narrative else ""
    causality = (
        "注意：回归结果描述统计关联，并不自动证明因果效应。"
        if lang == "zh"
        else "Note: regression results describe statistical associations and do not automatically establish causal effects."
    )
    if model_type == "panel_fe":
        causality = (
            "固定效应可以控制不随时间变化的个体差异，但并不自动证明因果关系。"
            if lang == "zh"
            else "Fixed effects can control for time-invariant entity differences, but they do not automatically establish causality."
        )
    elif model_type == "did":
        causality = (
            "DID 估计不自动证明因果关系；因果解释依赖平行趋势和研究设计等识别假设。"
            if lang == "zh"
            else "DID estimates do not automatically establish causality; causal interpretation depends on identification assumptions such as parallel trends and the research design."
        )
    elif model_type == "iv_2sls":
        causality = (
            "IV/2SLS 估计不自动证明因果关系；因果解释依赖工具变量相关性、排除限制和研究设计。"
            if lang == "zh"
            else "IV/2SLS estimates do not automatically establish causality; causal interpretation depends on instrument relevance, the exclusion restriction, and the research design."
        )
    elif model_type == "psm":
        causality = (
            "PSM 估计不自动证明因果关系；它只能改善观测协变量上的可比性，不能处理未观测混杂。"
            if lang == "zh"
            else "PSM estimates do not automatically establish causality; they improve comparability on observed covariates only and do not address unobserved confounding."
        )

    data_quality_section = _data_quality_report_markdown(profile, model_results, lang, simple=True)
    missing_data_section = _missing_data_handling_report_markdown(model_results, lang, simple=True)

    if lang == "zh":
        report = f"""# Reg Monkey \u7b80\u660e\u62a5\u544a

## 1. \u6570\u636e\u6982\u89c8
{stats_summary}

{data_quality_section}

{missing_data_section}

## 2. \u53d8\u91cf\u89d2\u8272
{role_text}

## 3. \u6a21\u578b\u8bbe\u5b9a
\u6a21\u578b\uff1a**{model_label}**

{_model_spec_summary(model_summary, lang)}

{panel_structure_block}

\u516c\u5f0f\uff1a`{equation}`

{equation_note}

## 4. \u6838\u5fc3\u7ed3\u8bba
{_main_findings(regression_table, model_summary, lang)}

{narrative_section}

## 5. \u5206\u7c7b\u63a7\u5236\u53d8\u91cf\u6458\u8981
{_categorical_control_summary(model_results, regression_table, lang, detailed=False)}

## 6. \u6a21\u578b\u62df\u5408
{fit_lines}

{workflow_section}

## 7. \u91cd\u8981\u63d0\u9192
{_structured_warning_lines(model_results, warnings, lang)}

## 8. \u5c40\u9650\u6027
- {causality}
- \u7ed3\u679c\u4f9d\u8d56\u53d8\u91cf\u5b9a\u4e49\u3001\u6570\u636e\u8d28\u91cf\u548c\u6a21\u578b\u8bbe\u5b9a\u3002
"""
        return _clean_zh_report_text(report)
    return f"""# Reg Monkey Simple Report

## 1. Data Overview
{stats_summary}

{data_quality_section}

{missing_data_section}

## 2. Variable Roles
{role_text}

## 3. Model Specification
Model: **{model_label}**

{_model_spec_summary(model_summary, lang)}

{panel_structure_block}

Formula: `{equation}`

{equation_note}

## 4. Key Findings
{_main_findings(regression_table, model_summary, lang)}

{narrative_section}

## 5. Categorical Controls Summary
{_categorical_control_summary(model_results, regression_table, lang, detailed=False)}

## 6. Model Fit
{fit_lines}

{workflow_section}

## 7. Important Warnings
{_structured_warning_lines(model_results, warnings, lang)}

## 8. Limitations
- {causality}
- Results depend on variable definitions, data quality, and model specification.
"""


def generate_markdown_report(
    profile: dict[str, Any],
    cleaning_log: dict[str, Any],
    regression_table: pd.DataFrame,
    model_summary: dict[str, Any],
    vif_df: pd.DataFrame,
    warnings: list[str],
    language: str = "en",
    model_metadata: Any | None = None,
    model_results: dict[str, Any] | None = None,
    variable_roles: dict[str, str] | None = None,
    guided_workflow_result: Any | None = None,
) -> str:
    lang = normalize_language(language)
    rows, cols = profile.get("shape", (0, 0))
    dtypes = profile.get("dtypes", {})
    missing_counts = profile.get("missing_counts", {})
    missing_percentages = profile.get("missing_percentages", {})
    duplicate_rows = profile.get("duplicate_rows", 0)
    numeric_stats, profile_binary_stats, categorical_stats = _profile_stats(profile)
    role_text, role_table = _role_summary(variable_roles, lang)
    model_type = _model_type(model_summary)
    model_label = model_metadata.report_label(lang) if model_metadata is not None and hasattr(model_metadata, "report_label") else model_type.upper()
    fit_lines = "\n".join(_fit_summary_lines(model_summary, lang))
    equation = _compact_equation(model_summary)
    equation_note = _equation_note(model_summary, lang)
    panel_structure_summary = _panel_structure_summary(model_summary, lang)
    panel_structure_block = ""
    if model_type == "panel_fe" and panel_structure_summary:
        panel_structure_label = "面板结构详情：" if lang == "zh" else "Panel structure details:"
        panel_structure_block = f"{panel_structure_label}\n{panel_structure_summary}"
    workflow_section = workflow_summary_markdown(guided_workflow_result, lang, detailed=True) if guided_workflow_result else ""
    narrative = generate_narrative(
        regression_table,
        model_summary,
        warnings,
        language=lang,
        workflow_result=guided_workflow_result,
        variable_roles=variable_roles,
        structured_diagnostics=_structured_diagnostics(model_results),
        advanced_outputs=_advanced_outputs(model_results),
    )
    narrative_section = narrative_markdown(narrative, simple=False) if narrative else ""

    missing_df = pd.DataFrame(
        {
            "variable": list(missing_counts.keys()),
            "missing_count": list(missing_counts.values()),
            "missing_percentage": [missing_percentages.get(col, 0) for col in missing_counts.keys()],
        }
    )
    formatted_table = format_regression_table(regression_table, lang)

    numeric_measure_stats = _stats_for_role(numeric_stats, variable_roles, ROLE_NUMERIC)
    binary_stats = _stats_for_role(profile_binary_stats, variable_roles, ROLE_BINARY)
    time_stats = _stats_for_role(numeric_stats, variable_roles, ROLE_TIME)
    code_stats = _stats_for_role(numeric_stats, variable_roles, ROLE_CODE)
    categorical_role_stats = _stats_for_role(categorical_stats, variable_roles, ROLE_CATEGORICAL)
    entity_stats = _stats_for_role(numeric_stats, variable_roles, ROLE_ENTITY)
    data_quality_section = _data_quality_report_markdown(profile, model_results, lang, simple=False)
    missing_data_section = _missing_data_handling_report_markdown(model_results, lang, simple=False)

    if lang == "zh":
        report = f"""# Reg Monkey \u62a5\u544a

## 1. \u6570\u636e\u6982\u89c8
- \u884c\u6570\uff1a{rows}
- \u5217\u6570\uff1a{cols}
- \u91cd\u590d\u884c\uff1a{duplicate_rows}

\u53d8\u91cf\u7c7b\u578b\u53cd\u6620\u6570\u636e\u5728\u8868\u683c\u4e2d\u7684\u5b58\u50a8\u683c\u5f0f\u3002

{_markdown_table(pd.DataFrame({"variable": list(dtypes.keys()), "dtype": list(dtypes.values())}), lang)}

\u7f3a\u5931\u503c\u6c47\u603b\uff1a

{_markdown_table(missing_df, lang)}

{data_quality_section}

{missing_data_section}

## 2. \u53d8\u91cf\u89d2\u8272\u4e0e\u63cf\u8ff0\u6027\u7edf\u8ba1
\u53d8\u91cf\u89d2\u8272\u53cd\u6620\u53d8\u91cf\u5728\u5206\u6790\u4e2d\u7684\u7528\u9014\u3002

{role_text}

{_markdown_table(role_table, lang)}

\u666e\u901a\u6570\u503c\u53d8\u91cf\u63cf\u8ff0\u6027\u7edf\u8ba1\uff1a

{_markdown_table(numeric_measure_stats, lang)}

\u4e8c\u5143\u53d8\u91cf\u7edf\u8ba1\uff1a

{_markdown_table(binary_stats, lang)}

\u65f6\u95f4\u53d8\u91cf\u7edf\u8ba1\uff1a

{_markdown_table(time_stats, lang)}

\u7f16\u7801 / ID \u53d8\u91cf\u7edf\u8ba1\uff1a

{_markdown_table(code_stats, lang)}

\u5206\u7c7b\u53d8\u91cf\u7edf\u8ba1\uff1a

{_markdown_table(categorical_role_stats, lang)}

\u4e2a\u4f53 ID \u7edf\u8ba1\uff1a

{_markdown_table(entity_stats, lang)}

## 3. \u6a21\u578b\u8bbe\u5b9a
\u6a21\u578b\uff1a**{model_label}**

{_model_spec_summary(model_summary, lang)}

\u6a21\u578b\u516c\u5f0f\uff1a`{equation}`

{equation_note}

{panel_structure_block}

## 4. \u5206\u7c7b\u63a7\u5236\u53d8\u91cf\u7f16\u7801\u8be6\u60c5
{_categorical_control_summary(model_results, regression_table, lang, detailed=True)}

## 5. \u56de\u5f52\u7ed3\u679c
{_markdown_table(formatted_table, lang)}

## 6. \u6a21\u578b\u62df\u5408
{fit_lines}

{workflow_section}

## 7. \u9ad8\u7ea7\u8f93\u51fa
{_advanced_outputs_markdown(model_results, lang)}

## 8. \u6a21\u578b\u6837\u672c\u6e05\u7406\u65e5\u5fd7
\u4ee5\u4e0b\u884c\u6570\u53d8\u5316\u6765\u81ea\u6a21\u578b\u6240\u9009\u53d8\u91cf\u4e2d\u7684\u7f3a\u5931\u503c\u5220\u9664\uff1b\u4e0d\u4ee3\u8868\u5df2\u786e\u8ba4\u7684\u6e05\u7406\u6570\u636e\u88ab\u6c38\u4e45\u4fee\u6539\u3002

{_markdown_table(pd.DataFrame([cleaning_log]), lang)}

## 9. \u8bca\u65ad
VIF \u8868\uff1a

{_diagnostic_markdown_table(vif_df, lang)}

\u7ed3\u6784\u5316\u8bca\u65ad\uff1a

{_structured_diagnostics_markdown(model_results, lang)}

\u63d0\u9192\uff1a
{_structured_warning_lines(model_results, warnings, lang)}

## 10. \u7ed3\u679c\u89e3\u91ca
{_main_findings(regression_table, model_summary, lang)}

{narrative_section}

## 11. \u5c40\u9650\u6027
- \u6ce8\u610f\uff1a\u6a21\u578b\u7ed3\u679c\u63cf\u8ff0\u7edf\u8ba1\u5173\u8054\uff0c\u4e0d\u81ea\u52a8\u8bc1\u660e\u56e0\u679c\u6548\u5e94\u3002
- \u7ed3\u679c\u53ef\u80fd\u53d7\u9057\u6f0f\u53d8\u91cf\u3001\u5185\u751f\u6027\u3001\u6d4b\u91cf\u8bef\u5dee\u548c\u6837\u672c\u9009\u62e9\u5f71\u54cd\u3002
- \u7ed3\u679c\u4f9d\u8d56\u6570\u636e\u8d28\u91cf\u3001\u53d8\u91cf\u7f16\u7801\u548c\u6a21\u578b\u8bbe\u5b9a\u3002
"""
        return _clean_zh_report_text(report)
    return f"""# Reg Monkey Report

## 1. Data Overview
- Rows: {rows}
- Columns: {cols}
- Duplicate rows: {duplicate_rows}

Data types describe how variables are stored in the dataset.

{_markdown_table(pd.DataFrame({"variable": list(dtypes.keys()), "dtype": list(dtypes.values())}), lang)}

Missing value summary:

{_markdown_table(missing_df, lang)}

{data_quality_section}

{missing_data_section}


## 2. Variable Roles and Descriptive Statistics
Variable roles describe how variables are used in analysis.

{role_text}

{_markdown_table(role_table, lang)}

Numeric measure descriptive statistics:

{_markdown_table(numeric_measure_stats, lang)}

Binary variable statistics:

{_markdown_table(binary_stats, lang)}

Time variable statistics:

{_markdown_table(time_stats, lang)}

Code / identifier variable statistics:

{_markdown_table(code_stats, lang)}

Categorical variable statistics:

{_markdown_table(categorical_role_stats, lang)}

Entity ID statistics:

{_markdown_table(entity_stats, lang)}

Note: means and standard deviations for time, code, or binary variables should not be interpreted as ordinary continuous measures.

## 3. Model Specification
Model: **{model_label}**

{_model_spec_summary(model_summary, lang)}

Model formula: `{equation}`

{equation_note} The full explanatory-variable list appears in the coefficient table.

{panel_structure_block}

## 4. Categorical Encoding Details
{_categorical_control_summary(model_results, regression_table, lang, detailed=True)}

## 5. Regression Results
{_markdown_table(formatted_table, lang)}

## 6. Model Fit
{fit_lines}

{workflow_section}

## 7. Advanced Outputs
{_advanced_outputs_markdown(model_results, lang)}

## 8. Model Sample Cleaning Log
These row changes come from dropping missing values in selected model variables. The confirmed cleaned dataset is not permanently modified.

{_markdown_table(pd.DataFrame([cleaning_log]), lang)}

## 9. Diagnostics
VIF table:

{_diagnostic_markdown_table(vif_df, lang)}

Structured diagnostics:

{_structured_diagnostics_markdown(model_results, lang)}

Warnings:

{_structured_warning_lines(model_results, warnings, lang)}

## 10. Interpretation
{_main_findings(regression_table, model_summary, lang)}

{narrative_section}

## 11. Limitations
- Note: regression results describe statistical associations and do not automatically establish causal effects. Correlation does not imply causation.
- Results may be affected by omitted variables, endogeneity, measurement error, and sample selection.
- Results depend on data quality, coding choices, and model specification.
"""
