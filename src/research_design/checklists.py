from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ChecklistSeverity = Literal["required", "recommended", "caution"]


@dataclass(frozen=True)
class ResearchDesignChecklistItem:
    item_id: str
    title: str
    description: str
    severity: ChecklistSeverity
    user_action: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchDesignChecklist:
    design_id: str
    items: list[ResearchDesignChecklistItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_research_design_checklist(design_id: str) -> ResearchDesignChecklist:
    """Return a read-only checklist contract for a research-design candidate."""

    item_ids = _CHECKLIST_ITEMS.get(design_id, [])
    return ResearchDesignChecklist(
        design_id=design_id,
        items=[
            ResearchDesignChecklistItem(
                item_id=item_id,
                title=f"rd_checklist_{item_id}_title",
                description=f"rd_checklist_{item_id}_description",
                severity=severity,
                user_action=f"rd_checklist_{item_id}_action",
            )
            for item_id, severity in item_ids
        ],
    )


_CHECKLIST_ITEMS: dict[str, list[tuple[str, ChecklistSeverity]]] = {
    "did": [
        ("did_group_definition", "required"),
        ("did_pre_post_definition", "required"),
        ("did_parallel_trends", "required"),
        ("did_no_confounding_shock", "caution"),
        ("did_cell_support", "recommended"),
        ("did_fixed_effects", "recommended"),
    ],
    "iv_2sls": [
        ("iv_relevance", "required"),
        ("iv_exclusion_restriction", "required"),
        ("iv_exogeneity", "required"),
        ("iv_first_stage_strength", "recommended"),
        ("iv_instrument_count", "recommended"),
        ("iv_no_auto_causal", "caution"),
    ],
    "psm": [
        ("psm_binary_treatment", "required"),
        ("psm_pre_treatment_covariates", "required"),
        ("psm_unobserved_confounding", "caution"),
        ("psm_common_support", "recommended"),
        ("psm_balance_diagnostics", "required"),
        ("psm_att_boundary", "caution"),
    ],
    "panel_fe": [
        ("panel_repeated_entities", "required"),
        ("panel_time_variable", "required"),
        ("panel_within_variation", "required"),
        ("panel_absorbed_variables", "caution"),
        ("panel_cluster_se", "recommended"),
    ],
    "observational_regression": [
        ("obs_meaningful_variables", "required"),
        ("obs_theory_controls", "recommended"),
        ("obs_omitted_variable_bias", "caution"),
        ("obs_multicollinearity", "recommended"),
        ("obs_association_not_causation", "caution"),
    ],
}
