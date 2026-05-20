from __future__ import annotations

import streamlit as st

from src.i18n import get_text
from src.reproducibility import build_reproducibility_bundle
from src.ui.components import render_action_card, render_callout, render_section_header


def render_reproducibility_pack_download(
    language: str,
    run_result: object,
    simple_report: str,
    full_report: str,
    preprocessing_summary: dict | None,
    variable_roles: dict | None,
    explanation_mode: str,
    data_quality_profile: dict | None = None,
    missingness_profile: dict | None = None,
    variable_quality_summary: list | None = None,
    resource_warning_profile: dict | None = None,
    pre_model_risk_profile: dict | None = None,
    missing_data_plan: dict | None = None,
    missing_data_handling_result: dict | None = None,
    show_safe_sharing: bool = True,
) -> None:
    t = lambda key: get_text(language, key)
    render_section_header(t("reproducibility_pack"), t("reproducibility_pack_body"))
    render_action_card(
        t("export_reproducibility_bundle_title"),
        t("export_reproducibility_bundle_body"),
        t("download_reproducibility_pack"),
        metadata={
            t("export_bundle_type"): t("export_technical_audit_bundle"),
            t("export_artifact_scope"): t("export_reproducibility_artifact_scope"),
        },
    )
    if show_safe_sharing:
        render_callout(t("export_safe_sharing_title"), t("export_review_before_sharing"), tone="warning")
    bundle_bytes = build_reproducibility_bundle(
        run_result,
        simple_report,
        full_report,
        preprocessing_summary=preprocessing_summary,
        variable_roles=variable_roles,
        explanation_mode=explanation_mode,
        data_quality_profile=data_quality_profile,
        missingness_profile=missingness_profile,
        variable_quality_summary=variable_quality_summary,
        resource_warning_profile=resource_warning_profile,
        pre_model_risk_profile=pre_model_risk_profile,
        missing_data_plan=missing_data_plan,
        missing_data_handling_result=missing_data_handling_result,
    )
    st.download_button(
        t("download_reproducibility_pack"),
        bundle_bytes,
        "reg_monkey_reproducibility_pack.zip",
        "application/zip",
        width="stretch",
    )
