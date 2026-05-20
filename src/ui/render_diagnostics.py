from __future__ import annotations

import pandas as pd

from src.diagnostic_rendering import diagnostic_dicts, warning_lines_for_display
from src.formatting import prepare_display_table
from src.i18n import get_text, translate_diagnostic_field


def structured_diagnostics_frame(diagnostics: list[object], language: str) -> pd.DataFrame:
    rows = []
    for item in diagnostic_dicts(diagnostics, ui_only=True):
        code = str(item.get("code") or "")
        rows.append(
            {
                get_text(language, "diagnostic_severity"): item.get("severity", ""),
                get_text(language, "diagnostic_code"): code,
                get_text(language, "diagnostic_message"): translate_diagnostic_field(language, code, "message", str(item.get("message") or "")),
                get_text(language, "diagnostic_recommendation"): translate_diagnostic_field(language, code, "recommendation", str(item.get("recommendation") or "")),
            }
        )
    return prepare_display_table(pd.DataFrame(rows), language)


def warning_lines_for_ui(diagnostics: list[object], warnings: list[str], language: str) -> list[str]:
    return warning_lines_for_display(diagnostics, warnings, language, ui_only=True)
