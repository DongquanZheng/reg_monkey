"""Small reusable Streamlit component helpers for the Reg Monkey visual system."""

from __future__ import annotations

import html
from collections.abc import Iterable, Mapping, Sequence

import streamlit as st

from src.ui.theme import DIAGNOSTIC_SEVERITY_STYLES


def _escape(value: object) -> str:
    return html.escape("" if value is None else str(value))


def _render_html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def render_badge(label: str, tone: str = "neutral", *, render: bool = True) -> str:
    """Render or return a compact status badge."""

    tone_classes = {
        "neutral": "rm-badge rm-badge-neutral",
        "info": "rm-badge rm-badge-info",
        "success": "rm-badge rm-badge-success",
        "warning": "rm-badge rm-badge-warning",
        "danger": "rm-badge rm-badge-danger",
        "experimental": "rm-badge rm-badge-experimental",
    }
    css_class = tone_classes.get(tone, tone_classes["neutral"])
    markup = f"<span class='{css_class}'>{_escape(label)}</span>"
    if render:
        _render_html(markup)
    return markup


def render_page_header(
    title: str,
    subtitle: str | None = None,
    badges: Sequence[str | tuple[str, str]] | None = None,
) -> None:
    """Render a page-level heading with optional subtitle and badges."""

    badge_markup = ""
    if badges:
        rendered_badges = []
        for badge in badges:
            if isinstance(badge, tuple):
                rendered_badges.append(render_badge(badge[0], badge[1], render=False))
            else:
                rendered_badges.append(render_badge(str(badge), render=False))
        badge_markup = "<div class='rm-page-badges'>" + "".join(rendered_badges) + "</div>"

    subtitle_markup = f"<p class='rm-page-subtitle'>{_escape(subtitle)}</p>" if subtitle else ""
    _render_html(
        "<div class='rm-page-header'>"
        f"<h1>{_escape(title)}</h1>"
        f"{subtitle_markup}"
        f"{badge_markup}"
        "</div>"
    )


def render_section_header(
    title: str,
    description: str | None = None,
    badge: str | tuple[str, str] | None = None,
) -> None:
    """Render a section heading used within workflow pages."""

    badge_markup = ""
    if badge:
        badge_markup = (
            render_badge(badge[0], badge[1], render=False)
            if isinstance(badge, tuple)
            else render_badge(str(badge), render=False)
        )
    description_markup = (
        f"<p class='rm-section-description'>{_escape(description)}</p>" if description else ""
    )
    _render_html(
        "<div class='rm-section-header'>"
        f"<div class='rm-section-title-row'><h2>{_escape(title)}</h2>{badge_markup}</div>"
        f"{description_markup}"
        "</div>"
    )


def render_status_card(
    title: str,
    body: str,
    status: str = "info",
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Render a concise status card."""

    _render_html(
        f"<div class='rm-card rm-status-card rm-status-{_escape(status)}'>"
        f"<div class='rm-card-title'>{_escape(title)}</div>"
        f"<div class='rm-card-body'>{_escape(body)}</div>"
        f"{_metadata_markup(metadata)}"
        "</div>"
    )


def render_selection_card(
    title: str,
    body: str,
    *,
    selected: bool = False,
    badge: str | tuple[str, str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Render a selectable path/card state without binding click behavior."""

    selected_class = " rm-selection-card-active" if selected else ""
    badge_markup = ""
    if badge:
        badge_markup = (
            render_badge(badge[0], badge[1], render=False)
            if isinstance(badge, tuple)
            else render_badge(str(badge), render=False)
        )
    _render_html(
        f"<div class='rm-card rm-selection-card{selected_class}'>"
        f"<div class='rm-card-title'>{_escape(title)}{badge_markup}</div>"
        f"<div class='rm-card-body'>{_escape(body)}</div>"
        f"{_metadata_markup(metadata)}"
        "</div>"
    )


def render_metric_card(
    label: str,
    value: object,
    help_text: str | None = None,
    tone: str = "neutral",
) -> None:
    """Render a compact metric card."""

    help_markup = f"<div class='rm-caption'>{_escape(help_text)}</div>" if help_text else ""
    _render_html(
        f"<div class='rm-card rm-metric-card rm-metric-{_escape(tone)}'>"
        f"<div class='rm-metric-label'>{_escape(label)}</div>"
        f"<div class='rm-metric-value'>{_escape(value)}</div>"
        f"{help_markup}"
        "</div>"
    )


def render_action_card(
    title: str,
    body: str,
    action_label: str | None = None,
    action_tone: str = "primary",
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Render an action-oriented card without binding any click behavior."""

    action_markup = (
        f"<div class='rm-action-label rm-action-{_escape(action_tone)}'>"
        f"{_escape(action_label)}</div>"
        if action_label
        else ""
    )
    _render_html(
        "<div class='rm-card rm-action-card'>"
        f"<div class='rm-card-title'>{_escape(title)}</div>"
        f"<div class='rm-card-body'>{_escape(body)}</div>"
        f"{_metadata_markup(metadata)}"
        f"{action_markup}"
        "</div>"
    )


def render_callout(
    title: str,
    body: str | None = None,
    tone: str = "info",
) -> None:
    """Render a short callout for notices, cautions, and confirmations."""

    body_markup = f"<div class='rm-callout-body'>{_escape(body)}</div>" if body else ""
    _render_html(
        f"<div class='rm-callout rm-callout-{_escape(tone)}'>"
        f"<div class='rm-callout-title'>{_escape(title)}</div>"
        f"{body_markup}"
        "</div>"
    )


def render_empty_state(
    title: str,
    body: str,
    action_hint: str | None = None,
) -> None:
    """Render a calm empty state."""

    action_markup = f"<div class='rm-empty-action'>{_escape(action_hint)}</div>" if action_hint else ""
    _render_html(
        "<div class='rm-card-soft rm-empty-state'>"
        f"<div class='rm-card-title'>{_escape(title)}</div>"
        f"<div class='rm-card-body'>{_escape(body)}</div>"
        f"{action_markup}"
        "</div>"
    )


def render_diagnostic_card(
    title: str,
    message: str,
    severity: str = "info",
    affected_variables: Iterable[str] | None = None,
    recommendation: str | None = None,
) -> None:
    """Render a diagnostic without interpreting model semantics."""

    style = DIAGNOSTIC_SEVERITY_STYLES.get(
        severity, DIAGNOSTIC_SEVERITY_STYLES["info"]
    )
    variables = ", ".join(_escape(variable) for variable in affected_variables or [])
    variables_markup = (
        f"<div class='rm-diagnostic-vars'><strong>Variables:</strong> {variables}</div>"
        if variables
        else ""
    )
    recommendation_markup = (
        f"<div class='rm-diagnostic-recommendation'>{_escape(recommendation)}</div>"
        if recommendation
        else ""
    )
    _render_html(
        "<div class='rm-card rm-diagnostic-card' "
        f"style='border-color:{_escape(style['border'])};'>"
        f"<div class='rm-diagnostic-severity'>{_escape(style['label'])}</div>"
        f"<div class='rm-card-title'>{_escape(title)}</div>"
        f"<div class='rm-card-body'>{_escape(message)}</div>"
        f"{variables_markup}"
        f"{recommendation_markup}"
        "</div>"
    )


def render_technical_expander(
    label: str,
    content: str | Mapping[str, object] | Sequence[object],
    expanded: bool = False,
) -> None:
    """Render technical details in a Streamlit expander."""

    with st.expander(label, expanded=expanded):
        if isinstance(content, str):
            st.code(content)
        else:
            st.json(content)


def _metadata_markup(metadata: Mapping[str, object] | None) -> str:
    if not metadata:
        return ""
    items = "".join(
        "<div class='rm-metadata-row'>"
        f"<span>{_escape(key)}</span><strong>{_escape(value)}</strong>"
        "</div>"
        for key, value in metadata.items()
    )
    return f"<div class='rm-metadata'>{items}</div>"
