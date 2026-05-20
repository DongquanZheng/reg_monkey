from __future__ import annotations

from typing import Any

import pandas as pd

from src.agent.workflow import GuidedWorkflowResult
from src.formatting import format_number, prepare_display_table


def workflow_summary_markdown(result: GuidedWorkflowResult | None, language: str = "en", detailed: bool = False) -> str:
    if result is None:
        return ""
    lang = "zh" if language == "zh" else "en"
    if lang == "zh":
        lines = [
            "## 引导式分析流程摘要",
            f"- 工作流类型：{result.workflow_type}",
            f"- 状态：{result.status}",
            f"- 基准模型：{_model_names(result.baseline_results) or '无'}",
            f"- 主模型：{result.main_result.model_name if result.main_result else '无'}",
            f"- 摘要：{result.summary}",
        ]
        if result.comparison:
            lines.append("- 结果比较：")
            lines += [f"  - {item}" for item in result.comparison.interpretation]
        if detailed:
            lines += ["", _comparison_table(result, lang)]
        return "\n".join(lines)

    lines = [
        "## Guided Analysis Workflow Summary",
        f"- Workflow type: {result.workflow_type}",
        f"- Status: {result.status}",
        f"- Baseline model(s): {_model_names(result.baseline_results) or 'none'}",
        f"- Main model: {result.main_result.model_name if result.main_result else 'none'}",
        f"- Summary: {result.summary}",
    ]
    if result.comparison:
        lines.append("- Result comparison:")
        lines += [f"  - {item}" for item in result.comparison.interpretation]
    if detailed:
        lines += ["", _comparison_table(result, lang)]
    return "\n".join(lines)


def workflow_result_frame(result: GuidedWorkflowResult | None, language: str = "en") -> pd.DataFrame:
    if result is None:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for model_result in result.baseline_results + ([result.main_result] if result.main_result else []) + result.alternative_results:
        if model_result is None:
            continue
        first_coef = model_result.key_coefficients[0] if model_result.key_coefficients else {}
        rows.append(
            {
                "model": model_result.model_name,
                "role": model_result.role,
                "status": model_result.status,
                "observations": model_result.summary_metrics.get("observations", "N/A"),
                "key_coefficient": format_number(first_coef.get("coefficient"), 4, language) if first_coef else "N/A",
                "p_value": first_coef.get("p_value_formatted", "N/A"),
                "fit_metric": _fit_metric(model_result.summary_metrics, language),
            }
        )
    return prepare_display_table(pd.DataFrame(rows), language)


def _comparison_table(result: GuidedWorkflowResult, language: str) -> str:
    frame = workflow_result_frame(result, language)
    if frame.empty:
        return "_No workflow comparison table available._" if language != "zh" else "_暂无流程比较表。_"
    return frame.to_markdown(index=False)


def _fit_metric(metrics: dict[str, Any], language: str) -> str:
    if metrics.get("within_r_squared") is not None:
        return "Within R²: " + format_number(metrics.get("within_r_squared"), 3, language)
    if metrics.get("r_squared") is not None:
        return "R²: " + format_number(metrics.get("r_squared"), 3, language)
    if metrics.get("pseudo_r_squared") is not None:
        return "Pseudo R²: " + format_number(metrics.get("pseudo_r_squared"), 3, language)
    return "N/A"


def _model_names(results: list) -> str:
    return ", ".join(result.model_name for result in results if result.status == "success")
