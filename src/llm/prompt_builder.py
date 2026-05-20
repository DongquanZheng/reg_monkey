from __future__ import annotations

import json

from src.llm.contracts import LLMExplanationInput


def build_explanation_prompt(payload: LLMExplanationInput) -> str:
    if payload.language == "zh":
        return _zh_prompt(payload)
    return _en_prompt(payload)


def _en_prompt(payload: LLMExplanationInput) -> str:
    return "\n".join(
        [
            "You are explaining a structured Reg Monkey model result for business, economics, or management research users.",
            "Use only the supplied structured result.",
            "Do not change, invent, round beyond recognition, or reinterpret coefficients, p-values, confidence intervals, fit metrics, diagnostics, warnings, or sample size.",
            "Do not infer causality. Use statistical association language unless the structured input explicitly states a causal design.",
            "Mention important diagnostics and limitations when present.",
            "Return concise English text suitable for a research report.",
            "Structured input:",
            json.dumps(payload.to_dict(), ensure_ascii=False, indent=2),
        ]
    )


def _zh_prompt(payload: LLMExplanationInput) -> str:
    return "\n".join(
        [
            "你正在为商业、经济或管理研究用户解释一份 Reg Monkey 结构化模型结果。",
            "只能使用下方提供的结构化结果。",
            "不得修改、编造、替换或随意改写系数、p 值、置信区间、拟合指标、诊断、警告或样本量。",
            "不得推断因果关系。除非结构化输入明确包含因果识别设计，否则只能使用统计相关或条件相关表述。",
            "如存在重要诊断或局限性，必须简要说明。",
            "请输出简洁、正式、适合研究报告的中文解释。",
            "结构化输入：",
            json.dumps(payload.to_dict(), ensure_ascii=False, indent=2),
        ]
    )
