from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from src.i18n import get_text


QuestionAnswerType = Literal["yes_no", "text", "choice"]
AssessmentStatus = Literal[
    "ready_for_manual_configuration",
    "needs_more_information",
    "high_identification_risk",
    "unsupported_by_current_workflow",
]
AssessmentItemStatus = Literal["confirmed", "missing", "risk", "unsupported"]


@dataclass(frozen=True)
class ResearchDesignQuestion:
    design_id: str
    question_id: str
    prompt: str
    description: str
    answer_type: QuestionAnswerType
    required: bool = True
    options: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchDesignAnswer:
    question_id: str
    answer: Any
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchDesignQuestionSet:
    design_id: str
    language: str
    schema_version: str = "research_design_questions.v1"
    questions: list[ResearchDesignQuestion] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchDesignAssessmentItem:
    question_id: str
    status: AssessmentItemStatus
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchDesignAssessment:
    design_id: str
    language: str
    status: AssessmentStatus
    items: list[ResearchDesignAssessmentItem] = field(default_factory=list)
    missing_confirmations: list[str] = field(default_factory=list)
    caution_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_research_design_questions(design_id: str, language: str = "en") -> ResearchDesignQuestionSet:
    """Build localized research-design confirmation questions without running or selecting models."""

    questions = [
        ResearchDesignQuestion(
            design_id=design_id,
            question_id=question_id,
            prompt=get_text(language, f"rd_question_{question_id}_prompt"),
            description=get_text(language, f"rd_question_{question_id}_description"),
            answer_type=answer_type,
            required=required,
            options=[get_text(language, f"rd_question_option_{option}") for option in options],
        )
        for question_id, answer_type, required, options in _QUESTION_DEFINITIONS.get(design_id, [])
    ]
    return ResearchDesignQuestionSet(design_id=design_id, language=language, questions=questions)


def assess_research_design_answers(
    design_id: str,
    answers: dict[str, Any] | list[ResearchDesignAnswer],
    language: str = "en",
) -> ResearchDesignAssessment:
    """Assess confirmation answers without producing a model recommendation or causal claim."""

    question_set = build_research_design_questions(design_id, language)
    normalized_answers = _normalize_answers(answers)
    items: list[ResearchDesignAssessmentItem] = []
    missing: list[str] = []
    cautions = [get_text(language, f"rd_assessment_{design_id}_baseline_caution")]
    has_risk = False
    unsupported = False

    for question in question_set.questions:
        value = normalized_answers.get(question.question_id)
        if _is_missing(value):
            items.append(
                ResearchDesignAssessmentItem(
                    question_id=question.question_id,
                    status="missing",
                    message=get_text(language, "rd_assessment_missing_confirmation"),
                )
            )
            if question.required:
                missing.append(question.prompt)
            continue

        if _is_negative(value):
            risk_status = _QUESTION_RISK_OVERRIDES.get(question.question_id, "risk")
            item_status: AssessmentItemStatus = "unsupported" if risk_status == "unsupported" else "risk"
            unsupported = unsupported or item_status == "unsupported"
            has_risk = has_risk or item_status == "risk"
            items.append(
                ResearchDesignAssessmentItem(
                    question_id=question.question_id,
                    status=item_status,
                    message=get_text(language, f"rd_assessment_{question.question_id}_{item_status}"),
                )
            )
            cautions.append(get_text(language, f"rd_assessment_{question.question_id}_{item_status}"))
            continue

        items.append(
            ResearchDesignAssessmentItem(
                question_id=question.question_id,
                status="confirmed",
                message=get_text(language, "rd_assessment_confirmed"),
            )
        )

    if unsupported:
        status: AssessmentStatus = "unsupported_by_current_workflow"
    elif has_risk:
        status = "high_identification_risk"
    elif missing:
        status = "needs_more_information"
    else:
        status = "ready_for_manual_configuration"

    return ResearchDesignAssessment(
        design_id=design_id,
        language=language,
        status=status,
        items=items,
        missing_confirmations=missing,
        caution_notes=_unique_nonempty(cautions),
    )


def _normalize_answers(answers: dict[str, Any] | list[ResearchDesignAnswer]) -> dict[str, Any]:
    if isinstance(answers, dict):
        return dict(answers)
    return {answer.question_id: answer.answer for answer in answers}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        normalized = value.strip().lower()
        return not normalized or normalized in {"unsure", "not_sure", "unknown", "不确定"}
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _is_negative(value: Any) -> bool:
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"no", "false", "0", "not_confirmed", "insufficient", "否"}
    return False


def _unique_nonempty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


_QUESTION_DEFINITIONS: dict[str, list[tuple[str, QuestionAnswerType, bool, list[str]]]] = {
    "did": [
        ("did_treated_group", "yes_no", True, []),
        ("did_post_period", "yes_no", True, []),
        ("did_control_group", "yes_no", True, []),
        ("did_parallel_trends_assessable", "yes_no", True, []),
        ("did_simple_timing", "yes_no", True, []),
    ],
    "iv_2sls": [
        ("iv_endogenous_variable", "text", True, []),
        ("iv_instrument_variable", "text", True, []),
        ("iv_exclusion_restriction", "text", True, []),
        ("iv_relevance", "yes_no", True, []),
        ("iv_instrument_count", "choice", True, ["single", "multiple"]),
    ],
    "psm": [
        ("psm_covariates_pre_treatment", "yes_no", True, []),
        ("psm_key_confounders_observed", "yes_no", True, []),
        ("psm_overlap", "yes_no", True, []),
        ("psm_att_estimand", "yes_no", True, []),
        ("psm_unmatched_units_boundary", "yes_no", True, []),
    ],
    "panel_fe": [
        ("panel_entity_id", "yes_no", True, []),
        ("panel_time_variable", "yes_no", True, []),
        ("panel_within_variation", "yes_no", True, []),
        ("panel_time_effects_needed", "yes_no", False, []),
        ("panel_cluster_entity", "yes_no", False, []),
    ],
    "observational_regression": [
        ("obs_association_goal", "yes_no", True, []),
        ("obs_key_controls", "yes_no", True, []),
        ("obs_categorical_controls", "yes_no", False, []),
        ("obs_multicollinearity_concern", "yes_no", False, []),
        ("obs_conditional_association_only", "yes_no", True, []),
    ],
}


_QUESTION_RISK_OVERRIDES: dict[str, Literal["risk", "unsupported"]] = {
    "did_simple_timing": "unsupported",
}
