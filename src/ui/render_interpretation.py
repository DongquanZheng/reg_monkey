from __future__ import annotations

import streamlit as st

from src.i18n import get_text, normalize_language
from src.llm.contracts import LLMExplanationMode
from src.llm.explanation_service import generate_mock_llm_explanation


def explanation_mode_options(language: str) -> dict[str, str]:
    return {
        get_text(language, "explanation_mode_rule_based"): LLMExplanationMode.RULE_BASED.value,
        get_text(language, "explanation_mode_llm_mock"): LLMExplanationMode.LLM_ASSISTED.value,
        get_text(language, "explanation_mode_rule_llm_mock"): LLMExplanationMode.RULE_BASED_LLM_POLISHED.value,
    }


def explanation_layer_label(language: str, mode: str) -> str:
    key = {
        LLMExplanationMode.LLM_ASSISTED.value: "explanation_layer_llm_mock",
        LLMExplanationMode.RULE_BASED_LLM_POLISHED.value: "explanation_layer_rule_llm_mock",
    }.get(mode, "explanation_layer_rule_based")
    return f"{get_text(language, 'explanation_layer')}: {get_text(language, key)}"


def render_research_interpretation(narrative: object, language: str) -> None:
    if narrative is None:
        return
    t = lambda key: get_text(language, key)
    st.markdown(narrative.executive_summary)
    with st.expander(t("technical_interpretation"), expanded=False):
        st.markdown(narrative.technical_interpretation)
    with st.expander(t("limitations_next_steps"), expanded=False):
        for item in narrative.limitations + narrative.next_steps:
            st.markdown(f"- {item}")


def render_mock_llm_interpretation(run_result: object, mode: str, narrative: object, language: str) -> None:
    t = lambda key: get_text(language, key)
    if run_result is None:
        render_research_interpretation(narrative, language)
        return
    llm_result = generate_mock_llm_explanation(run_result, normalize_language(language), mode)
    if llm_result.fallback_to_rule_based or llm_result.output is None:
        st.warning(t("llm_guardrail_fallback"))
        render_research_interpretation(narrative, language)
        return
    st.info(t("llm_mock_disclaimer"))
    st.markdown(llm_result.output.explanation_text)
    with st.expander(t("llm_mock_limitations"), expanded=False):
        st.markdown(llm_result.output.limitations_text)
    with st.expander(t("llm_mock_next_steps"), expanded=False):
        st.markdown(llm_result.output.next_steps_text)
