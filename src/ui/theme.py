"""Reg Monkey visual theme tokens and optional Streamlit CSS injection."""

from __future__ import annotations

import streamlit as st


COLOR_TOKENS = {
    "bg": "#FAF7F2",
    "surface": "#FFFFFF",
    "surface_soft": "#F4EFE7",
    "text": "#25211D",
    "text_muted": "#6F655C",
    "border": "#E4DAD0",
    "accent": "#A84A3F",
    "accent_hover": "#8F3E35",
    "accent_soft": "#F4DDD8",
    "secondary": "#4F7B63",
    "secondary_soft": "#DDE9E1",
    "info": "#3D6F8F",
    "info_soft": "#E3EEF4",
    "success": "#4F7B63",
    "success_soft": "#DDE9E1",
    "warning": "#B9852A",
    "warning_soft": "#F3E7CB",
    "danger": "#9C3B3B",
    "danger_soft": "#F0DADA",
    "muted": "#EDE6DD",
}

SPACING_TOKENS = {
    "xs": "4px",
    "sm": "8px",
    "md": "12px",
    "lg": "16px",
    "xl": "24px",
    "xxl": "36px",
    "page_max_width": "1240px",
    "reading_width": "860px",
    "card_padding": "20px",
    "section_gap": "32px",
}

TYPOGRAPHY_TOKENS = {
    "font_family": (
        '"Inter", "Source Sans Pro", "Noto Sans SC", "Microsoft YaHei", '
        '"PingFang SC", "Helvetica Neue", Arial, sans-serif'
    ),
    "mono_family": (
        '"JetBrains Mono", "SFMono-Regular", Consolas, "Liberation Mono", monospace'
    ),
    "page_title": "1.75rem",
    "section_title": "1.2rem",
    "card_title": "1rem",
    "body": "0.95rem",
    "caption": "0.82rem",
    "line_height": "1.55",
}

RADIUS_TOKENS = {
    "sm": "4px",
    "md": "8px",
    "lg": "10px",
    "pill": "999px",
}

BORDER_TOKENS = {
    "default": f"1px solid {COLOR_TOKENS['border']}",
    "strong": "1px solid #D3C5B8",
}

SHADOW_TOKENS = {
    "none": "none",
    "subtle": "0 1px 2px rgba(37, 33, 29, 0.06)",
    "raised": "0 8px 20px rgba(37, 33, 29, 0.08)",
}

DIAGNOSTIC_SEVERITY_STYLES = {
    "info": {
        "color": COLOR_TOKENS["info"],
        "background": COLOR_TOKENS["info_soft"],
        "border": "#BBD1DF",
        "label": "Info",
    },
    "caution": {
        "color": COLOR_TOKENS["warning"],
        "background": COLOR_TOKENS["warning_soft"],
        "border": "#DEC891",
        "label": "Caution",
    },
    "warning": {
        "color": COLOR_TOKENS["warning"],
        "background": COLOR_TOKENS["warning_soft"],
        "border": "#D6B46A",
        "label": "Warning",
    },
    "critical": {
        "color": COLOR_TOKENS["danger"],
        "background": COLOR_TOKENS["danger_soft"],
        "border": "#D7AAAA",
        "label": "Critical",
    },
    "success": {
        "color": COLOR_TOKENS["success"],
        "background": COLOR_TOKENS["success_soft"],
        "border": "#AEC9B8",
        "label": "Success",
    },
}

BUTTON_STYLE_TOKENS = {
    "primary": {
        "background": COLOR_TOKENS["accent"],
        "text": "#FFFFFF",
        "border": COLOR_TOKENS["accent"],
    },
    "secondary": {
        "background": COLOR_TOKENS["surface"],
        "text": COLOR_TOKENS["accent"],
        "border": COLOR_TOKENS["border"],
    },
    "subtle": {
        "background": COLOR_TOKENS["surface_soft"],
        "text": COLOR_TOKENS["text"],
        "border": COLOR_TOKENS["border"],
    },
    "danger": {
        "background": COLOR_TOKENS["danger"],
        "text": "#FFFFFF",
        "border": COLOR_TOKENS["danger"],
    },
    "disabled": {
        "background": COLOR_TOKENS["muted"],
        "text": COLOR_TOKENS["text_muted"],
        "border": COLOR_TOKENS["border"],
    },
}

CARD_STYLE_TOKENS = {
    "background": COLOR_TOKENS["surface"],
    "border": BORDER_TOKENS["default"],
    "radius": RADIUS_TOKENS["md"],
    "padding": SPACING_TOKENS["card_padding"],
    "shadow": SHADOW_TOKENS["subtle"],
}


def css_variables() -> str:
    """Return Reg Monkey CSS variables and low-risk utility classes."""

    color_vars = "\n".join(
        f"  --rm-{name.replace('_', '-')}: {value};" for name, value in COLOR_TOKENS.items()
    )
    spacing_vars = "\n".join(
        f"  --rm-space-{name.replace('_', '-')}: {value};"
        for name, value in SPACING_TOKENS.items()
    )
    radius_vars = "\n".join(
        f"  --rm-radius-{name}: {value};" for name, value in RADIUS_TOKENS.items()
    )

    return f"""
:root {{
{color_vars}
{spacing_vars}
{radius_vars}
  --rm-font-family: {TYPOGRAPHY_TOKENS["font_family"]};
  --rm-mono-family: {TYPOGRAPHY_TOKENS["mono_family"]};
  --rm-line-height: {TYPOGRAPHY_TOKENS["line_height"]};
}}

html,
body,
.stApp,
[data-testid="stHeader"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.block-container {{
  background: var(--rm-bg) !important;
  color: var(--rm-text) !important;
}}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {{
  background: var(--rm-bg) !important;
  color: var(--rm-text) !important;
}}

[data-testid="stHeader"] button,
[data-testid="stToolbar"] button,
[data-testid="stToolbar"] svg,
[data-testid="stStatusWidget"] button,
[data-testid="stStatusWidget"] svg {{
  color: var(--rm-text) !important;
  fill: var(--rm-text) !important;
}}

main,
main p,
main li,
main label,
main span,
main div[data-testid="stMarkdownContainer"],
main div[data-testid="stMarkdownContainer"] p,
main div[data-testid="stMarkdownContainer"] li,
main h1,
main h2,
main h3,
main h4,
main h5,
main h6,
.section-title,
.rm-card-title,
.rm-callout-title,
.rm-summary-value,
.rm-metric-value,
.rm-app-brand,
.rm-section-header h2 {{
  color: var(--rm-text) !important;
}}

main small,
main [data-testid="stCaptionContainer"],
main .stCaptionContainer,
.section-guide,
.rm-caption,
.rm-page-subtitle,
.rm-section-description,
.rm-card-body,
.rm-summary-label,
.rm-chip-row,
.rm-app-subtitle,
.rm-app-tagline,
.workflow-stage-label {{
  color: var(--rm-text-muted) !important;
}}

main [data-testid="stWidgetLabel"],
main [data-testid="stWidgetLabel"] p,
main [data-testid="stWidgetLabel"] label,
main [data-testid="stWidgetLabel"] span {{
  color: var(--rm-text) !important;
}}

section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {{
  background: #FFFCF7 !important;
  color: var(--rm-text) !important;
}}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
  color: var(--rm-text) !important;
}}

section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
  color: var(--rm-text-muted) !important;
}}

div[data-baseweb="select"] > div,
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input,
textarea {{
  background: #FFFFFF !important;
  color: var(--rm-text) !important;
  border-color: var(--rm-border) !important;
}}

div[data-testid="stNumberInput"] button,
div[data-testid="stNumberInput"] button:hover {{
  background: #FFFFFF !important;
  border-color: var(--rm-border) !important;
  color: var(--rm-text) !important;
}}

div[data-testid="stNumberInput"] button *,
div[data-testid="stNumberInput"] button svg {{
  color: var(--rm-text) !important;
  fill: var(--rm-text) !important;
}}

div[data-baseweb="select"] span,
div[data-baseweb="select"] svg {{
  color: var(--rm-text) !important;
}}

div[data-baseweb="popover"],
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] li {{
  background: #FFFFFF !important;
  color: var(--rm-text) !important;
}}

div[data-testid="stButton"] > button,
div[data-testid="stDownloadButton"] > button,
div[data-testid="stFormSubmitButton"] > button {{
  background: #FFFFFF !important;
  border: 1px solid var(--rm-border) !important;
  color: var(--rm-text) !important;
}}

div[data-testid="stButton"] > button:hover,
div[data-testid="stDownloadButton"] > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {{
  background: var(--rm-surface-soft) !important;
  border-color: #D3C5B8 !important;
  color: var(--rm-text) !important;
}}

div[data-testid="stButton"] > button[kind="primary"],
div[data-testid="stDownloadButton"] > button[kind="primary"],
div[data-testid="stFormSubmitButton"] > button[kind="primary"] {{
  background: var(--rm-accent) !important;
  border-color: var(--rm-accent) !important;
  color: #FFFFFF !important;
}}

div[data-testid="stButton"] > button[kind="primary"] *,
div[data-testid="stDownloadButton"] > button[kind="primary"] *,
div[data-testid="stFormSubmitButton"] > button[kind="primary"] * {{
  color: #FFFFFF !important;
}}

div[data-testid="stButton"] > button:disabled,
div[data-testid="stDownloadButton"] > button:disabled,
div[data-testid="stFormSubmitButton"] > button:disabled {{
  background: var(--rm-muted) !important;
  border-color: var(--rm-border) !important;
  color: var(--rm-text-muted) !important;
  opacity: 1 !important;
}}

div[data-testid="stButton"] > button:disabled *,
div[data-testid="stDownloadButton"] > button:disabled *,
div[data-testid="stFormSubmitButton"] > button:disabled * {{
  color: var(--rm-text-muted) !important;
}}

[data-testid="stFileUploader"],
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"] {{
  background: #FFFFFF !important;
  border-color: var(--rm-border) !important;
  color: var(--rm-text) !important;
}}

[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] label {{
  color: var(--rm-text-muted) !important;
}}

[data-testid="stFileUploader"] button {{
  background: #FFFFFF !important;
  border-color: var(--rm-border) !important;
  color: var(--rm-text) !important;
}}

[data-testid="stExpander"],
[data-testid="stExpander"] details,
[data-testid="stExpander"] summary {{
  background: #FFFFFF !important;
  border-color: var(--rm-border) !important;
  color: var(--rm-text) !important;
}}

[data-testid="stExpander"] summary *,
[data-testid="stExpander"] [data-testid="stMarkdownContainer"],
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
[data-testid="stExpander"] p,
[data-testid="stExpander"] span,
[data-testid="stExpander"] label {{
  color: var(--rm-text) !important;
}}

[data-testid="stExpander"] svg {{
  color: var(--rm-text) !important;
  fill: var(--rm-text) !important;
}}

[data-testid="stAlert"] {{
  background: var(--rm-warning-soft) !important;
  border-color: #DEC891 !important;
  color: var(--rm-text) !important;
}}

[data-testid="stAlert"] *,
[data-testid="stAlert"] [data-testid="stMarkdownContainer"],
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {{
  color: var(--rm-text) !important;
}}

[data-testid="stAlert"] svg {{
  color: var(--rm-warning) !important;
  fill: var(--rm-warning) !important;
}}

[data-testid="stCheckbox"] label,
[data-testid="stCheckbox"] span,
[data-testid="stCheckbox"] p {{
  color: var(--rm-text) !important;
}}

.rm-card {{
  background: var(--rm-surface);
  border: {CARD_STYLE_TOKENS["border"]};
  border-radius: {CARD_STYLE_TOKENS["radius"]};
  padding: {CARD_STYLE_TOKENS["padding"]};
  box-shadow: {CARD_STYLE_TOKENS["shadow"]};
}}

.rm-action-card {{
  background: #FFFCF8;
  border: 1px solid #E9DCD2;
  min-height: 112px;
}}

.rm-status-card {{
  min-height: 92px;
}}

.rm-selection-card {{
  min-height: 132px;
  transition: border-color 120ms ease, box-shadow 120ms ease;
}}

.rm-selection-card-active {{
  border-color: var(--rm-accent);
  box-shadow: 0 0 0 2px var(--rm-accent-soft);
}}

.rm-workspace-card {{
  background: #FFFCF8;
  border: 1px solid #E7D8CB;
  border-radius: var(--rm-radius-md);
  padding: var(--rm-space-lg);
}}

.rm-family-summary {{
  background: var(--rm-surface);
  border: 1px solid var(--rm-border);
  border-left: 4px solid var(--rm-secondary);
  border-radius: var(--rm-radius-md);
  margin: var(--rm-space-md) 0;
  padding: var(--rm-space-lg);
}}

.rm-family-summary-experimental {{
  border-left-color: var(--rm-warning);
}}

.rm-card-soft {{
  background: var(--rm-surface-soft);
  border: {CARD_STYLE_TOKENS["border"]};
  border-radius: {CARD_STYLE_TOKENS["radius"]};
  padding: {CARD_STYLE_TOKENS["padding"]};
}}

.rm-caption {{
  color: var(--rm-text-muted);
  font-size: {TYPOGRAPHY_TOKENS["caption"]};
  line-height: var(--rm-line-height);
}}

.rm-technical {{
  font-family: var(--rm-mono-family);
}}

.rm-page-header {{
  max-width: var(--rm-space-page-max-width);
  margin: 0 0 var(--rm-space-xl) 0;
}}

.rm-page-header h1 {{
  color: var(--rm-text);
  font-size: 1.8rem;
  line-height: 1.2;
  margin: 0 0 var(--rm-space-sm) 0;
}}

.rm-page-subtitle,
.rm-section-description,
.rm-card-body {{
  color: var(--rm-text-muted);
  line-height: var(--rm-line-height);
}}

.rm-page-badges,
.rm-section-title-row {{
  align-items: center;
  display: flex;
  flex-wrap: wrap;
  gap: var(--rm-space-sm);
}}

.rm-section-header {{
  margin: var(--rm-space-xl) 0 var(--rm-space-md) 0;
}}

.rm-section-header h2 {{
  color: var(--rm-text);
  font-size: 1.2rem;
  line-height: 1.25;
  margin: 0;
}}

.rm-card-title,
.rm-callout-title {{
  color: var(--rm-text);
  font-weight: 700;
  margin-bottom: var(--rm-space-xs);
}}

.rm-badge {{
  border: 1px solid var(--rm-border);
  border-radius: var(--rm-radius-pill);
  display: inline-block;
  font-size: 0.78rem;
  line-height: 1.2;
  margin: 0 var(--rm-space-xs) var(--rm-space-xs) 0;
  padding: 0.18rem 0.55rem;
}}

.rm-badge-neutral {{ background: var(--rm-surface-soft); color: var(--rm-text-muted); }}
.rm-badge-info {{ background: var(--rm-info-soft); color: var(--rm-info); }}
.rm-badge-success {{ background: var(--rm-success-soft); color: var(--rm-success); }}
.rm-badge-warning,
.rm-badge-experimental {{ background: var(--rm-warning-soft); color: var(--rm-warning); }}
.rm-badge-danger {{ background: var(--rm-danger-soft); color: var(--rm-danger); }}

.rm-callout {{
  background: var(--rm-info-soft);
  border: 1px solid #BBD1DF;
  border-left: 4px solid var(--rm-info);
  border-radius: var(--rm-radius-md);
  color: var(--rm-text);
  margin: var(--rm-space-md) 0;
  padding: var(--rm-space-md) var(--rm-space-lg);
}}

.rm-callout-warning {{
  background: var(--rm-warning-soft);
  border-color: #DEC891;
  border-left-color: var(--rm-warning);
}}

.rm-callout-success {{
  background: var(--rm-success-soft);
  border-color: #AEC9B8;
  border-left-color: var(--rm-success);
}}

.rm-callout-danger {{
  background: var(--rm-danger-soft);
  border-color: #D7AAAA;
  border-left-color: var(--rm-danger);
}}

.rm-metric-label {{
  color: var(--rm-text-muted);
  font-size: 0.8rem;
  margin-bottom: var(--rm-space-xs);
}}

.rm-metric-value {{
  color: var(--rm-text);
  font-size: 1.34rem;
  font-weight: 760;
  overflow-wrap: anywhere;
}}

.rm-action-label {{
  border-top: 1px solid var(--rm-border);
  color: var(--rm-accent);
  font-weight: 700;
  margin-top: var(--rm-space-md);
  padding-top: var(--rm-space-sm);
}}

.rm-empty-state {{
  margin: var(--rm-space-md) 0;
}}

.rm-diagnostic-severity,
.rm-diagnostic-vars,
.rm-diagnostic-recommendation,
.rm-metadata-row {{
  color: var(--rm-text-muted);
  font-size: 0.84rem;
  line-height: var(--rm-line-height);
}}

section[data-testid="stSidebar"] {{
  background: #FFFCF7;
  border-right: 1px solid var(--rm-border);
}}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
  color: var(--rm-text);
}}
""".strip()


def inject_regmonkey_theme() -> None:
    """Inject Reg Monkey theme variables into Streamlit.

    This remains opt-in for v2.9.1 so the foundation can land without
    redesigning existing pages or changing screenshot expectations.
    """

    st.markdown(f"<style>{css_variables()}</style>", unsafe_allow_html=True)
