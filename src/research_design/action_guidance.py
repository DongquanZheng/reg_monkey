from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from src.i18n import get_text
from src.research_design.questions import ResearchDesignAssessment


ActionItemSeverity = Literal["next_step", "caution"]


@dataclass(frozen=True)
class ResearchDesignActionItem:
    item_id: str
    text: str
    severity: ActionItemSeverity = "next_step"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchDesignActionGuidance:
    design_id: str
    assessment_status: str
    language: str
    summary: str
    items: list[ResearchDesignActionItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_research_design_action_guidance(
    design_id: str,
    assessment: ResearchDesignAssessment,
    language: str = "en",
) -> ResearchDesignActionGuidance:
    """Translate assessment status into cautious next-step guidance without recommending models."""

    question_statuses = {item.question_id: item.status for item in assessment.items}
    item_ids = [_status_item_id(assessment.status), *_design_item_ids(design_id, assessment.status, question_statuses)]
    items = [
        ResearchDesignActionItem(
            item_id=item_id,
            text=get_text(language, f"rd_action_{item_id}"),
            severity="caution" if item_id.endswith("_caution") or "risk" in item_id else "next_step",
        )
        for item_id in _unique_in_order(item_ids)
    ]
    return ResearchDesignActionGuidance(
        design_id=design_id,
        assessment_status=assessment.status,
        language=language,
        summary=get_text(language, f"rd_action_summary_{assessment.status}"),
        items=items,
    )


def _status_item_id(status: str) -> str:
    return {
        "ready_for_manual_configuration": "status_manual_with_cautions",
        "needs_more_information": "status_needs_more_information",
        "high_identification_risk": "status_high_identification_risk",
        "unsupported_by_current_workflow": "status_unsupported_by_current_workflow",
    }.get(status, "status_needs_more_information")


def _design_item_ids(design_id: str, status: str, question_statuses: dict[str, str]) -> list[str]:
    if design_id == "did":
        return _did_item_ids(status, question_statuses)
    if design_id == "iv_2sls":
        return _iv_item_ids(status, question_statuses)
    if design_id == "psm":
        return _psm_item_ids(status, question_statuses)
    if design_id == "panel_fe":
        return _panel_item_ids(status, question_statuses)
    if design_id == "observational_regression":
        return ["obs_association_only_caution"]
    return []


def _did_item_ids(status: str, question_statuses: dict[str, str]) -> list[str]:
    items = ["did_parallel_trends_not_verified_caution"]
    if question_statuses.get("did_parallel_trends_assessable") in {"missing", "risk"}:
        items.append("did_parallel_trends_needs_information")
    if question_statuses.get("did_control_group") in {"missing", "risk"}:
        items.append("did_control_group_needs_information")
    if question_statuses.get("did_simple_timing") == "unsupported" or status == "unsupported_by_current_workflow":
        items.append("did_timing_unsupported_caution")
    return items


def _iv_item_ids(status: str, question_statuses: dict[str, str]) -> list[str]:
    items = ["iv_exclusion_requires_context_caution", "iv_first_stage_relevance_check"]
    if question_statuses.get("iv_exclusion_restriction") in {"missing", "risk"} or status == "high_identification_risk":
        items.insert(0, "iv_exclusion_high_risk")
    return items


def _psm_item_ids(status: str, question_statuses: dict[str, str]) -> list[str]:
    items = ["psm_balance_diagnostics_next_step", "psm_unobserved_confounding_caution"]
    if question_statuses.get("psm_key_confounders_observed") in {"missing", "risk"}:
        items.insert(0, "psm_covariate_coverage_needs_information")
    if question_statuses.get("psm_overlap") in {"missing", "risk"}:
        items.append("psm_overlap_needs_review")
    return items


def _panel_item_ids(status: str, question_statuses: dict[str, str]) -> list[str]:
    items = ["panel_within_variation_next_step", "panel_omitted_variable_caution"]
    if question_statuses.get("panel_entity_id") in {"missing", "risk"} or question_statuses.get("panel_time_variable") in {"missing", "risk"}:
        items.insert(0, "panel_structure_needs_information")
    return items


def _unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique
