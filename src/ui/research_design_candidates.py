from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import streamlit as st

from src.i18n import get_text
from src.research_design import ResearchDesignCandidateSet, build_research_design_checklist, detect_research_design_candidates
from src.ui.components import render_badge, render_callout, render_section_header
from src.ui.research_design_questions import render_research_design_questions


EXPERIMENTAL_MANUAL_MODEL_IDS = {"did", "iv_2sls", "psm"}


def candidate_status_label(language: str, status: str) -> str:
    return get_text(language, f"rd_status_{status}")


def candidate_preview_items(candidate_set: ResearchDesignCandidateSet, language: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for candidate in candidate_set.candidates:
        if candidate.status == "not_applicable":
            continue
        if candidate.design_id in EXPERIMENTAL_MANUAL_MODEL_IDS and candidate.status == "insufficient_information":
            continue
        items.append(
            {
                "design_id": candidate.design_id,
                "display_name": get_text(language, candidate.display_name_key),
                "status": candidate.status,
                "status_label": candidate_status_label(language, candidate.status),
                "requires_confirmation": candidate.design_id in EXPERIMENTAL_MANUAL_MODEL_IDS,
                "detected_from": list(candidate.detected_from),
                "confirmations": [get_text(language, item.description_key) for item in candidate.required_user_confirmations],
                "cautions": [get_text(language, item.message_key) for item in candidate.cautions],
                "compatible_model_ids": list(candidate.compatible_model_ids),
                "not_auto_recommended_reason": candidate.not_auto_recommended_reason,
                "checklist": checklist_preview_items(candidate.design_id, language),
            }
        )
    return items


def checklist_preview_items(design_id: str, language: str) -> dict[str, list[dict[str, str]]]:
    checklist = build_research_design_checklist(design_id)
    grouped: dict[str, list[dict[str, str]]] = {"required": [], "recommended": [], "caution": []}
    for item in checklist.items:
        grouped[item.severity].append(
            {
                "item_id": item.item_id,
                "title": get_text(language, item.title),
                "description": get_text(language, item.description),
                "severity": item.severity,
                "user_action": get_text(language, item.user_action) if item.user_action else "",
            }
        )
    return grouped


def render_research_design_candidate_preview(
    data: pd.DataFrame,
    variable_roles: dict[str, str],
    language: str,
    on_go_manual: Callable[[str], None] | None = None,
) -> ResearchDesignCandidateSet:
    candidate_set = detect_research_design_candidates(data, variable_roles)
    items = candidate_preview_items(candidate_set, language)

    render_section_header(get_text(language, "research_design_candidates"), get_text(language, "research_design_candidates_caption"))
    render_callout(
        get_text(language, "rd_caution_not_auto_recommended"),
        get_text(language, "research_design_candidates_caution"),
        tone="info",
    )

    if not items:
        st.caption(get_text(language, "rd_no_candidates"))
        return candidate_set

    for item in items:
        with st.container(border=True):
            title = item["display_name"]
            if item["requires_confirmation"]:
                title = f"{title} - {get_text(language, 'rd_requires_confirmation')}"
            st.markdown(f"**{title}**")
            render_badge(item["status_label"], "warning" if item["requires_confirmation"] else "info")
            st.caption(f"{get_text(language, 'rd_status')}: {item['status_label']}")
            with st.expander(get_text(language, "rd_assumptions_and_cautions"), expanded=False):
                _markdown_list(language, "rd_required_confirmations", item["confirmations"])
                _markdown_list(language, "rd_main_cautions", item["cautions"])
            with st.expander(get_text(language, "planner_details"), expanded=False):
                _markdown_list(language, "rd_detected_from", item["detected_from"] or [get_text(language, "not_available")])
                compatible_labels = [get_text(language, f"model_label_{model_id}") for model_id in item["compatible_model_ids"]]
                _markdown_list(language, "rd_compatible_manual_path", compatible_labels or [get_text(language, "not_available")])
            _render_checklist(language, item["checklist"])
            render_research_design_questions(item["design_id"], language)
            if item["design_id"] in EXPERIMENTAL_MANUAL_MODEL_IDS and on_go_manual is not None:
                if st.button(
                    f"{get_text(language, 'rd_go_manual_config')} ↓",
                    key=f"rd_go_manual_config_{item['design_id']}",
                    type="secondary",
                ):
                    on_go_manual(item["design_id"])
                    st.rerun()

    return candidate_set


def _markdown_list(language: str, title_key: str, values: list[str]) -> None:
    st.markdown(f"**{get_text(language, title_key)}**")
    for value in values:
        st.markdown(f"- {value}")


def _render_checklist(language: str, grouped_items: dict[str, list[dict[str, str]]]) -> None:
    with st.expander(get_text(language, "rd_pre_use_checklist"), expanded=False):
        for severity in ["required", "recommended", "caution"]:
            items = grouped_items.get(severity) or []
            if not items:
                continue
            st.markdown(f"**{get_text(language, f'rd_checklist_group_{severity}')}**")
            for item in items:
                st.markdown(f"- **{item['title']}**: {item['description']}")
                if item.get("user_action"):
                    st.caption(item["user_action"])
