from __future__ import annotations

from typing import Any

import streamlit as st

from src.i18n import get_text
from src.research_design import assess_research_design_answers, build_research_design_action_guidance, build_research_design_questions


ANSWER_VALUES = {
    "yes": True,
    "no": False,
    "unsure": "unsure",
}


def question_preview_items(design_id: str, language: str) -> list[dict[str, Any]]:
    question_set = build_research_design_questions(design_id, language)
    return [
        {
            "design_id": question.design_id,
            "question_id": question.question_id,
            "prompt": question.prompt,
            "description": question.description,
            "required": question.required,
            "answer_type": question.answer_type,
            "options": list(question.options),
        }
        for question in question_set.questions
    ]


def assess_question_preview_answers(design_id: str, answers: dict[str, Any], language: str) -> dict[str, Any]:
    assessment = assess_research_design_answers(design_id, answers, language)
    return assessment.to_dict()


def action_guidance_preview_items(design_id: str, answers: dict[str, Any], language: str) -> dict[str, Any]:
    assessment = assess_research_design_answers(design_id, answers, language)
    return build_research_design_action_guidance(design_id, assessment, language).to_dict()


def render_research_design_questions(design_id: str, language: str) -> None:
    questions = question_preview_items(design_id, language)
    if not questions:
        return

    expanded = _has_recorded_answer(design_id, questions, language)
    with st.expander(get_text(language, "rd_answer_design_questions"), expanded=expanded):
        st.caption(get_text(language, "rd_questions_preview_caption"))
        answers: dict[str, Any] = {}
        labels = {
            "yes": get_text(language, "rd_answer_yes"),
            "no": get_text(language, "rd_answer_no"),
            "unsure": get_text(language, "rd_answer_unsure"),
        }
        reverse_labels = {label: value for value, label in labels.items()}

        for question in questions:
            st.markdown(f"**{question['prompt']}**")
            st.caption(question["description"])
            selection = st.radio(
                get_text(language, "rd_answer_label"),
                options=list(labels.values()),
                index=2,
                key=f"rd_question_answer_{design_id}_{question['question_id']}",
                horizontal=True,
                label_visibility="collapsed",
            )
            answers[question["question_id"]] = ANSWER_VALUES[reverse_labels[selection]]

        assessment = assess_question_preview_answers(design_id, answers, language)
        _render_assessment(language, assessment)
        _render_action_guidance(language, action_guidance_preview_items(design_id, answers, language))


def _render_assessment(language: str, assessment: dict[str, Any]) -> None:
    st.markdown(f"**{get_text(language, 'rd_assessment_status')}**")
    st.caption(get_text(language, f"rd_assessment_status_{assessment['status']}"))

    missing = assessment.get("missing_confirmations") or []
    if missing:
        st.markdown(f"**{get_text(language, 'rd_assessment_missing_confirmations')}**")
        for item in missing:
            st.markdown(f"- {item}")

    cautions = assessment.get("caution_notes") or []
    if cautions:
        st.markdown(f"**{get_text(language, 'rd_assessment_cautions')}**")
        for item in cautions:
            st.markdown(f"- {item}")

    st.caption(get_text(language, "rd_questions_no_auto_action_note"))


def _render_action_guidance(language: str, guidance: dict[str, Any]) -> None:
    st.markdown(f"**{get_text(language, 'rd_action_guidance_heading')}**")
    if guidance.get("summary"):
        st.caption(guidance["summary"])
    for item in guidance.get("items") or []:
        st.markdown(f"- {item['text']}")


def _has_recorded_answer(design_id: str, questions: list[dict[str, Any]], language: str) -> bool:
    unsure_label = get_text(language, "rd_answer_unsure")
    for question in questions:
        value = st.session_state.get(f"rd_question_answer_{design_id}_{question['question_id']}")
        if value and value != unsure_label:
            return True
    return False
