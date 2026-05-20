from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from src.agent.narrative_templates import narrative_text
from src.agent.workflow import GuidedWorkflowResult
from src.diagnostic_rendering import diagnostic_dicts
from src.formatting import format_number, format_p_narrative
from src.i18n import translate_diagnostic_field, translate_warning


@dataclass(frozen=True)
class NarrativeInput:
    model_result: dict[str, Any] | None = None
    workflow_result: GuidedWorkflowResult | None = None
    analysis_plan: Any | None = None
    variable_roles: dict[str, str] | None = None
    language: str = "en"
    user_confirmed_model_setup: bool = True


@dataclass(frozen=True)
class NarrativeResult:
    language: str
    model_type: str
    narrative_type: str
    executive_summary: str
    technical_interpretation: str
    limitations: list[str]
    next_steps: list[str]
    warnings: list[str] = field(default_factory=list)
    confidence_label: str = ""


def generate_narrative(
    regression_table: pd.DataFrame | None = None,
    model_summary: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    language: str = "en",
    workflow_result: GuidedWorkflowResult | None = None,
    analysis_plan: Any | None = None,
    variable_roles: dict[str, str] | None = None,
    structured_diagnostics: list[Any] | None = None,
    advanced_outputs: dict[str, Any] | None = None,
) -> NarrativeResult | None:
    lang = "zh" if language == "zh" else "en"
    if workflow_result is not None:
        return _workflow_narrative(workflow_result, lang)
    if not model_summary:
        return None

    model_type = str(model_summary.get("model_type") or "ols").lower()
    table = regression_table if regression_table is not None else pd.DataFrame()
    diagnostic_warnings = _warnings_from_structured_diagnostics(structured_diagnostics, lang) if structured_diagnostics else list(warnings or [])
    main_rows = _main_rows(table, model_summary.get("main_independent_variables") or model_summary.get("independent_variables") or [])
    significant = [row for row in main_rows if _p_value(row) is not None and _p_value(row) < 0.10]

    if model_type == "did":
        executive = _did_executive(significant, model_summary, lang)
        technical = _did_technical(model_summary, lang)
        limitations = _did_limitations(lang)
    elif model_type == "psm":
        executive = _psm_executive(model_summary, lang)
        technical = _psm_technical(model_summary, lang, advanced_outputs)
        limitations = _psm_limitations(lang)
    elif model_type == "iv_2sls":
        executive = _iv_executive(significant, model_summary, lang)
        technical = _iv_technical(model_summary, lang)
        limitations = _iv_limitations(lang)
    elif model_type == "panel_fe":
        executive = _panel_executive(significant, model_summary, lang)
        technical = _panel_technical(model_summary, lang)
        limitations = _panel_limitations(lang)
    elif model_type in {"logit", "probit"}:
        executive = _binary_executive(significant, model_summary, model_type, lang, advanced_outputs)
        technical = _binary_technical(model_summary, model_type, lang, advanced_outputs)
        limitations = _binary_limitations(lang)
    else:
        executive = _ols_executive(significant, model_summary, lang)
        technical = _ols_technical(model_summary, lang)
        limitations = _ols_limitations(lang)

    concise_warnings = _concise_warnings(diagnostic_warnings, lang)
    if concise_warnings:
        technical += "\n\n" + concise_warnings[0]

    return NarrativeResult(
        language=lang,
        model_type=model_type,
        narrative_type="model",
        executive_summary=executive,
        technical_interpretation=technical,
        limitations=limitations,
        next_steps=[narrative_text(lang, "review_next")],
        warnings=concise_warnings,
        confidence_label=narrative_text(lang, "confidence"),
    )


def narrative_to_dict(result: NarrativeResult) -> dict[str, Any]:
    return asdict(result)


def narrative_markdown(result: NarrativeResult | None, simple: bool = False) -> str:
    if result is None:
        return ""
    if result.language == "zh":
        lines = ["## 研究解释", "### 核心解读", result.executive_summary]
        if not simple:
            lines += ["", "### 技术解释", result.technical_interpretation]
        lines += ["", "### 局限性与下一步"]
    else:
        lines = ["## Research Interpretation", "### Executive Summary", result.executive_summary]
        if not simple:
            lines += ["", "### Technical Interpretation", result.technical_interpretation]
        lines += ["", "### Limitations and Next Steps"]
    lines += [f"- {item}" for item in result.limitations + result.next_steps]
    return "\n".join(lines)


def _workflow_narrative(result: GuidedWorkflowResult, language: str) -> NarrativeResult:
    main_name = result.main_result.model_name if result.main_result else ("无" if language == "zh" else "none")
    baseline_names = [item.model_name for item in result.baseline_results if item.status == "success"]
    baseline_text = ", ".join(baseline_names) if baseline_names else ("无" if language == "zh" else "none")
    if language == "zh":
        executive = f"引导式分析已运行基准模型 {baseline_text} 和主模型 {main_name}。请优先比较核心变量在基准模型与主模型中的方向和显著性。"
        technical = "OLS 基准模型反映混合样本中的总体相关关系；面板固定效应主模型更强调同一个体内部随时间变化的关系。两者差异较大时，可能说明个体层面不随时间变化的差异会影响普通 OLS 结果。"
        limitations = ["引导式流程是规则式组织结果，不代表自动完成因果识别。", narrative_text(language, "causal_limit")]
        next_steps = ["检查主模型诊断结果。", "确认基准模型与主模型是否符合研究设计。"]
    else:
        executive = f"The guided workflow ran baseline model(s) {baseline_text} and main model {main_name}. Focus on whether the key variable has a consistent direction and statistical evidence across models."
        technical = "The OLS baseline estimates pooled associations, while Panel Fixed Effects emphasizes within-entity relationships over time. Large differences may indicate time-invariant entity heterogeneity in the pooled OLS estimate."
        limitations = ["The guided workflow is rule-based and does not automatically identify causal effects.", narrative_text(language, "causal_limit")]
        next_steps = ["Review main-model diagnostics.", "Check whether the baseline and main model match the research design."]

    return NarrativeResult(
        language=language,
        model_type=result.main_result.model_id if result.main_result else "workflow",
        narrative_type="workflow",
        executive_summary=executive,
        technical_interpretation=technical,
        limitations=limitations,
        next_steps=next_steps,
        warnings=list(result.warnings[:1]),
        confidence_label=narrative_text(language, "confidence"),
    )


def _ols_executive(significant: list[pd.Series], summary: dict[str, Any], language: str) -> str:
    fit = _fit_label(summary.get("r_squared"), language)
    if not significant:
        return f"{narrative_text(language, 'no_significant')} {_fit_sentence(fit, language)}"
    row = significant[0]
    direction = _direction(row, language)
    if language == "zh":
        return f"在控制其他变量后，{row['variable']} 与 {summary.get('dependent_variable', '因变量')} 呈{direction}条件相关（{format_p_narrative(row.get('p_value'), language)}）。{_fit_sentence(fit, language)}"
    return f"Holding selected controls constant, {row['variable']} is {direction} associated with {summary.get('dependent_variable', 'the outcome')} ({format_p_narrative(row.get('p_value'), language)}). {_fit_sentence(fit, language)}"


def _ols_technical(summary: dict[str, Any], language: str) -> str:
    se = "HC3 robust" if summary.get("robust_standard_errors") else "conventional"
    r2 = format_number(summary.get("r_squared"), 3, language)
    adj = format_number(summary.get("adj_r_squared"), 3, language)
    if language == "zh":
        return f"{narrative_text(language, 'ols_coef')} 本模型 R² 为 {r2}，调整后 R² 为 {adj}，标准误类型为 {se}。"
    return f"{narrative_text(language, 'ols_coef')} This model has R-squared {r2}, adjusted R-squared {adj}, and {se} standard errors."


def _binary_executive(
    significant: list[pd.Series],
    summary: dict[str, Any],
    model_type: str,
    language: str,
    advanced_outputs: dict[str, Any] | None = None,
) -> str:
    marginal = _main_marginal_effect(summary, advanced_outputs)
    if marginal is not None:
        value = format_number(marginal.get("marginal_effect"), 4, language)
        variable = marginal.get("variable")
        if language == "zh":
            return f"{model_type.title()} 模型的平均边际效应显示，在当前模型设定下，{variable} 每增加 1 个单位，{summary.get('dependent_variable', '因变量')} 取 1 的估计概率变化约为 {value}。该结果表示条件相关关系，不能自动解释为因果关系。"
        return f"The {model_type.title()} average marginal effect suggests that a one-unit increase in {variable} is associated with an estimated {value} change in the probability that {summary.get('dependent_variable', 'the outcome')} equals 1 under the current model specification. This is an association, not automatic causal evidence."
    if not significant:
        return narrative_text(language, "no_significant")
    row = significant[0]
    direction = _direction(row, language)
    if language == "zh":
        return f"{model_type.title()} 模型用于估计 {summary.get('dependent_variable', '因变量')} 取 1 的倾向。{row['variable']} 的系数方向为{direction}（{format_p_narrative(row.get('p_value'), language)}），但不能解释为概率百分点变化。"
    return f"The {model_type.title()} model estimates the tendency for {summary.get('dependent_variable', 'the outcome')} to equal 1. {row['variable']} has a {direction} coefficient ({format_p_narrative(row.get('p_value'), language)}), but it is not a percentage-point probability change."


def _binary_technical(
    summary: dict[str, Any],
    model_type: str,
    language: str,
    advanced_outputs: dict[str, Any] | None = None,
) -> str:
    metric = format_number(summary.get("pseudo_r_squared"), 3, language)
    aic = format_number(summary.get("aic"), 2, language)
    bic = format_number(summary.get("bic"), 2, language)
    template = narrative_text(language, "logit_coef" if model_type == "logit" else "probit_coef")
    marginal = _main_marginal_effect(summary, advanced_outputs)
    marginal_note = ""
    if marginal is not None:
        marginal_note = (
            f" 平均边际效应已用于概率尺度解释，主变量 {marginal.get('variable')} 的边际效应为 {format_number(marginal.get('marginal_effect'), 4, language)}。"
            if language == "zh"
            else f" Average marginal effects are available for probability-scale interpretation; the main variable {marginal.get('variable')} has marginal effect {format_number(marginal.get('marginal_effect'), 4, language)}."
        )
    if language == "zh":
        return f"{template} McFadden 伪 R² 为 {metric}，AIC 为 {aic}，BIC 为 {bic}。{marginal_note}"
    return f"{template} McFadden pseudo R-squared is {metric}, AIC is {aic}, and BIC is {bic}.{marginal_note}"
    if language == "zh":
        return f"{template} McFadden 伪 R² 为 {metric}，AIC 为 {aic}，BIC 为 {bic}。"
    return f"{template} McFadden pseudo R-squared is {metric}, AIC is {aic}, and BIC is {bic}."


def _panel_executive(significant: list[pd.Series], summary: dict[str, Any], language: str) -> str:
    if not significant:
        return narrative_text(language, "no_significant")
    row = significant[0]
    direction = _direction(row, language)
    if language == "zh":
        return f"在控制固定效应后，{row['variable']} 与同一个体内部 {summary.get('dependent_variable', '因变量')} 随时间变化呈{direction}相关（{format_p_narrative(row.get('p_value'), language)}）。"
    return f"After fixed effects, {row['variable']} is {direction} associated with within-entity changes in {summary.get('dependent_variable', 'the outcome')} over time ({format_p_narrative(row.get('p_value'), language)})."


def _panel_technical(summary: dict[str, Any], language: str) -> str:
    within = format_number(summary.get("r_squared_within"), 3, language)
    entities = summary.get("entities", "N/A")
    periods = summary.get("time_periods", "N/A")
    se = summary.get("standard_errors", "N/A")
    if language == "zh":
        return f"{narrative_text(language, 'panel_coef')} 组内 R² 为 {within}，样本包含 {entities} 个个体和 {periods} 个时间期，标准误类型为 {se}。"
    return f"{narrative_text(language, 'panel_coef')} Within R-squared is {within}; the sample includes {entities} entities and {periods} time periods, with {se} standard errors."


def _did_executive(significant: list[pd.Series], summary: dict[str, Any], language: str) -> str:
    if not significant:
        if language == "zh":
            return "DID 交互项未显示明确统计证据。该结果仍需结合平行趋势和研究设计进行判断。"
        return "The DID interaction term does not show clear statistical evidence. Interpretation still depends on parallel trends and the research design."
    row = significant[0]
    direction = _direction(row, language) if language == "zh" else _direction_adjective(row)
    if language == "zh":
        return f"{row['variable']} 的 DID 估计方向为{direction}（{format_p_narrative(row.get('p_value'), language)}）。该估计是条件双重差分结果，不能自动解释为因果关系。"
    return f"{row['variable']} has a {direction} DID estimate ({format_p_narrative(row.get('p_value'), language)}). This is a conditional difference-in-differences estimate and does not automatically establish causality."


def _did_technical(summary: dict[str, Any], language: str) -> str:
    r2 = format_number(summary.get("r_squared"), 3, language)
    estimate = format_number(summary.get("did_estimate"), 4, language)
    p_value = format_p_narrative(summary.get("did_p_value"), language)
    se = summary.get("standard_errors", "hc3")
    did_term = summary.get("did_term", "treatment:post")
    if language == "zh":
        return f"DID 模型估计处理组变量与处理后时期变量的交互项 {did_term}。DID 估计值为 {estimate}（{p_value}），R² 为 {r2}，标准误类型为 {se}。"
    return f"The DID model estimates the treatment-by-post interaction {did_term}. The DID estimate is {estimate} ({p_value}), with R-squared {r2} and {se} standard errors."


def _iv_executive(significant: list[pd.Series], summary: dict[str, Any], language: str) -> str:
    if not significant:
        if language == "zh":
            return "IV/2SLS 条件估计未显示明确统计证据。该结果仍需结合工具变量相关性、排除限制和研究设计判断。"
        return "The IV/2SLS conditional estimate does not show clear statistical evidence. Interpretation still depends on instrument relevance, the exclusion restriction, and the research design."
    row = significant[0]
    direction = _direction(row, language) if language == "zh" else _direction_adjective(row)
    if language == "zh":
        return f"{row['variable']} 的 IV/2SLS 条件估计方向为{direction}（{format_p_narrative(row.get('p_value'), language)}）。该估计不能自动解释为因果关系。"
    return f"{row['variable']} has a {direction} IV/2SLS conditional estimate ({format_p_narrative(row.get('p_value'), language)}). This estimate does not automatically establish causality."


def _iv_technical(summary: dict[str, Any], language: str) -> str:
    r2 = format_number(summary.get("r_squared"), 3, language)
    estimate = format_number(summary.get("iv_estimate"), 4, language)
    p_value = format_p_narrative(summary.get("iv_p_value"), language)
    first_stage_r2 = format_number(summary.get("first_stage_r_squared"), 3, language)
    first_stage_f = format_number(summary.get("first_stage_f_statistic"), 3, language)
    iv_term = summary.get("iv_term", "fitted_endogenous")
    instruments = ", ".join(summary.get("instruments") or [])
    if language == "zh":
        return f"IV/2SLS 模型使用工具变量 {instruments or '未记录'} 构造 {iv_term}。IV 估计值为 {estimate}（{p_value}），第二阶段 R² 为 {r2}，第一阶段 R² 为 {first_stage_r2}，第一阶段 F 统计量为 {first_stage_f}。"
    return f"The IV/2SLS model uses instrument(s) {instruments or 'not recorded'} to construct {iv_term}. The IV estimate is {estimate} ({p_value}), with second-stage R-squared {r2}, first-stage R-squared {first_stage_r2}, and first-stage F-statistic {first_stage_f}."


def _psm_executive(summary: dict[str, Any], language: str) -> str:
    att = format_number(summary.get("att_estimate"), 4, language)
    matched = summary.get("matched_treated_count", "N/A")
    if language == "zh":
        return f"当前 PSM 最近邻匹配设定得到的 ATT 估计值为 {att}，基于 {matched} 个已匹配处理组观测。该结果仅反映观测协变量匹配后的差异，不能自动解释为因果关系。"
    return f"The current nearest-neighbor PSM specification produces an ATT estimate of {att}, based on {matched} matched treated observation(s). This reflects a matched difference on observed covariates and does not automatically establish causality."


def _psm_technical(summary: dict[str, Any], language: str, advanced_outputs: dict[str, Any] | None = None) -> str:
    matched = summary.get("matched_treated_count", "N/A")
    treated = summary.get("treated_count", "N/A")
    controls = summary.get("control_count", "N/A")
    covariates = ", ".join(summary.get("matching_covariates") or [])
    outputs = advanced_outputs or {}
    propensity = outputs.get("propensity_score_summary", {})
    balance_overview = outputs.get("psm_balance_overview", {})
    score_text = ""
    if isinstance(propensity, dict) and propensity:
        score_text = (
            f" 倾向得分范围为 {format_number(propensity.get('min'), 3, language)} 至 {format_number(propensity.get('max'), 3, language)}。"
            if language == "zh"
            else f" Propensity scores range from {format_number(propensity.get('min'), 3, language)} to {format_number(propensity.get('max'), 3, language)}."
        )
    balance_text = ""
    if isinstance(balance_overview, dict) and balance_overview:
        max_after = balance_overview.get("max_absolute_smd_after")
        worsened = balance_overview.get("covariates_worsened_count")
        high_vars = balance_overview.get("high_residual_imbalance_variables") or []
        if language == "zh":
            balance_text = f" 匹配后最大绝对标准化均值差为 {format_number(max_after, 3, language)}。"
            if worsened:
                balance_text += f" {worsened} 个协变量的平衡未改善。"
            if high_vars:
                balance_text += " 匹配后仍需关注较高不平衡变量：" + ", ".join(str(item) for item in high_vars) + "。"
        else:
            balance_text = f" The maximum post-match absolute SMD is {format_number(max_after, 3, language)}."
            if worsened:
                balance_text += f" Balance did not improve for {worsened} covariate(s)."
            if high_vars:
                balance_text += " Residual imbalance remains for: " + ", ".join(str(item) for item in high_vars) + "."
    replacement_text = (
        " 当前最小实现使用有放回匹配。"
        if language == "zh" and summary.get("replacement_matching")
        else " The minimal implementation uses matching with replacement."
        if summary.get("replacement_matching")
        else ""
    )
    if language == "zh":
        return f"PSM 使用 Logit 倾向得分和最近邻匹配估计 ATT。匹配协变量为 {covariates or '未记录'}；处理组样本 {treated} 个、对照组样本 {controls} 个，其中 {matched} 个处理组观测完成匹配。{score_text}{balance_text}{replacement_text}"
    return f"PSM uses Logit propensity scores and nearest-neighbor matching to estimate ATT. Matching covariates are {covariates or 'not recorded'}; the sample includes {treated} treated and {controls} control observation(s), with {matched} treated observation(s) matched.{score_text}{balance_text}{replacement_text}"


def _main_rows(table: pd.DataFrame, variables: list[str]) -> list[pd.Series]:
    if table is None or table.empty or "variable" not in table.columns:
        return []
    return [row for _, row in table[table["variable"].isin(variables)].iterrows()]


def _main_marginal_effect(summary: dict[str, Any], advanced_outputs: dict[str, Any] | None) -> pd.Series | None:
    if not advanced_outputs:
        return None
    table = advanced_outputs.get("marginal_effects_table")
    if not isinstance(table, pd.DataFrame) or table.empty or "variable" not in table.columns:
        return None
    main_vars = list(summary.get("main_independent_variables") or summary.get("independent_variables") or [])
    rows = table[table["variable"].isin(main_vars)]
    if rows.empty:
        return None
    return rows.iloc[0]


def _p_value(row: pd.Series) -> float | None:
    try:
        value = float(row.get("p_value"))
    except (TypeError, ValueError):
        return None
    return value if pd.notna(value) else None


def _direction(row: pd.Series, language: str) -> str:
    coefficient = float(row.get("coefficient", 0))
    if language == "zh":
        return "正向" if coefficient > 0 else "负向" if coefficient < 0 else "不明显"
    return "positively" if coefficient > 0 else "negatively" if coefficient < 0 else "not clearly"


def _direction_adjective(row: pd.Series) -> str:
    coefficient = float(row.get("coefficient", 0))
    return "positive" if coefficient > 0 else "negative" if coefficient < 0 else "unclear"


def _fit_label(value: Any, language: str) -> str:
    try:
        r2 = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if r2 < 0.10:
        return "weak"
    if r2 < 0.30:
        return "modest"
    if r2 < 0.60:
        return "moderate"
    return "strong"


def _fit_sentence(label: str, language: str) -> str:
    if language == "zh":
        labels = {"weak": "解释力较弱", "modest": "解释力有限", "moderate": "解释力中等", "strong": "解释力较强"}
        return f"模型{labels.get(label, '拟合情况需结合诊断判断')}。"
    labels = {"weak": "weak explanatory power", "modest": "modest explanatory power", "moderate": "moderate explanatory power", "strong": "strong explanatory power"}
    return f"The model shows {labels.get(label, 'fit that should be interpreted with diagnostics')}."


def _concise_warnings(warnings: list[str], language: str) -> list[str]:
    if not warnings:
        return []
    return [narrative_text(language, "diagnostic_warning", warning=translate_warning(language, warnings[0]))]


def _warnings_from_structured_diagnostics(diagnostics: list[Any] | None, language: str) -> list[str]:
    if not diagnostics:
        return []
    messages: list[str] = []
    for item in diagnostic_dicts(diagnostics, ui_only=True):
        severity = str(item.get("severity") or "")
        show_in_ui = bool(item.get("show_in_ui", True))
        if severity not in {"error", "warning"} or not show_in_ui:
            continue
        code = str(item.get("code") or "")
        title = translate_diagnostic_field(language, code, "title", str(item.get("title") or ""))
        message = translate_diagnostic_field(language, code, "message", str(item.get("message") or ""))
        if title and message and title.lower() not in message.lower():
            message = f"{title}: {message}"
        if message:
            messages.append(message)
    return list(dict.fromkeys(messages))


def _ols_limitations(language: str) -> list[str]:
    return [narrative_text(language, "causal_limit"), "Linearity, omitted variables, and measurement quality should be reviewed." if language != "zh" else "需要检查线性设定、遗漏变量和变量测量质量。"]


def _binary_limitations(language: str) -> list[str]:
    return [narrative_text(language, "causal_limit"), "No categorical-control diagnostics were generated for the current model." if language != "zh" else "当前模型未生成分类控制变量诊断。"]


def _panel_limitations(language: str) -> list[str]:
    return [narrative_text(language, "causal_limit"), "Time-varying omitted variables and limited within-entity variation may still affect interpretation." if language != "zh" else "随时间变化的遗漏变量和较弱的组内变化仍可能影响解释。"]


def _did_limitations(language: str) -> list[str]:
    if language == "zh":
        return [
            narrative_text(language, "causal_limit"),
            "DID 解释依赖平行趋势等识别假设，当前版本不会自动检验平行趋势。",
            "请检查处理组、处理后时期和控制变量设定是否符合研究设计。",
        ]
    return [
        narrative_text(language, "causal_limit"),
        "DID interpretation depends on identification assumptions such as parallel trends; this version does not test parallel trends automatically.",
        "Review treatment, post-period, and control-variable definitions against the research design.",
    ]


def _iv_limitations(language: str) -> list[str]:
    if language == "zh":
        return [
            narrative_text(language, "causal_limit"),
            "IV/2SLS 解释依赖工具变量相关性和排除限制，当前最小运行器不会自动证明这些识别假设。",
            "请检查第一阶段强度、工具变量定义和控制变量设定是否符合研究设计。",
        ]
    return [
        narrative_text(language, "causal_limit"),
        "IV/2SLS interpretation depends on instrument relevance and the exclusion restriction; this minimal runner does not prove those identification assumptions.",
        "Review first-stage strength, instrument definitions, and control-variable choices against the research design.",
    ]


def _psm_limitations(language: str) -> list[str]:
    if language == "zh":
        return [
            narrative_text(language, "causal_limit"),
            "PSM 只能改善观测协变量上的平衡，不能处理未观测混杂。",
            "请检查匹配前后的协变量平衡、共同支持和样本损失后再解释 ATT。",
        ]
    return [
        narrative_text(language, "causal_limit"),
        "PSM improves balance on observed covariates only and does not address unobserved confounding.",
        "Review before/after balance, common support, and sample loss before interpreting the ATT.",
    ]
