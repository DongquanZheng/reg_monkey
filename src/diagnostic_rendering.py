from __future__ import annotations

import re
from typing import Any

from src.i18n import translate_diagnostic_field, translate_warning

MULTICOLLINEARITY_CODES = {"high_multicollinearity", "serious_multicollinearity"}


def diagnostic_dicts(diagnostics: list[Any] | None, *, ui_only: bool = False, report_only: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for diagnostic in diagnostics or []:
        item = diagnostic.to_dict() if hasattr(diagnostic, "to_dict") else dict(diagnostic)
        if ui_only and not item.get("show_in_ui", True):
            continue
        if report_only and not item.get("show_in_report", True):
            continue
        rows.append(item)
    return dedupe_diagnostic_dicts(rows)


def warning_lines_for_display(
    diagnostics: list[Any] | None,
    warnings: list[str] | None,
    language: str,
    *,
    ui_only: bool = False,
    report_only: bool = False,
) -> list[str]:
    diagnostic_rows = [
        item
        for item in diagnostic_dicts(diagnostics, ui_only=ui_only, report_only=report_only)
        if item.get("severity") in {"error", "warning"}
    ]
    diagnostic_rows = sorted(diagnostic_rows, key=_warning_sort_key)
    lines: list[str] = []
    covered = {_warning_key_from_diagnostic(item) for item in diagnostic_rows}
    covered.discard(None)
    serious_vars = {
        variable
        for key in covered
        if key[0] == "serious"
        for variable in key[1]
    }

    for item in diagnostic_rows:
        code = str(item.get("code") or "")
        message = translate_diagnostic_field(language, code, "message", str(item.get("message") or ""))
        if message:
            lines.append(message)

    for warning in warnings or []:
        key = _warning_key_from_text(str(warning))
        if key and key in covered:
            continue
        warning_text = str(warning)
        if key and key[0] == "high" and serious_vars:
            remaining = [variable for variable in key[1] if variable not in serious_vars]
            if not remaining:
                continue
            adjusted_key = ("high", tuple(sorted(remaining)))
            if adjusted_key in covered:
                continue
            warning_text = "VIF exceeds 5 for: " + ", ".join(remaining) + "."
        translated = translate_warning(language, str(warning))
        if warning_text != str(warning):
            translated = translate_warning(language, warning_text)
        if translated:
            lines.append(translated)

    return _unique(lines)


def dedupe_diagnostic_dicts(diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    diagnostics = [_copy_diagnostic(item) for item in diagnostics]
    serious_vars = {
        variable
        for item in diagnostics
        if str(item.get("code") or "") == "serious_multicollinearity"
        for variable in _affected_variables(item)
    }
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...], str]] = set()
    for item in diagnostics:
        code = str(item.get("code") or "")
        variables = _affected_variables(item)
        if code == "high_multicollinearity" and serious_vars:
            variables = [variable for variable in variables if variable not in serious_vars]
            if not variables:
                continue
            item["affected_variables"] = variables
            item["message"] = "VIF exceeds 5 for: " + ", ".join(variables) + "."
        message_key = "" if code in MULTICOLLINEARITY_CODES else str(item.get("message") or "")
        key = (code, tuple(variables), message_key)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _copy_diagnostic(item: dict[str, Any]) -> dict[str, Any]:
    copied = dict(item)
    if "affected_variables" in copied and copied["affected_variables"] is not None:
        copied["affected_variables"] = list(copied["affected_variables"])
    return copied


def _affected_variables(item: dict[str, Any]) -> list[str]:
    variables = item.get("affected_variables") or []
    if variables:
        return [str(variable) for variable in variables]
    return _variables_from_text(str(item.get("message") or ""))


def _variables_from_text(text: str) -> list[str]:
    match = re.search(r"(?:for:|for)\s+(.+?)(?:\.|$)", text, flags=re.IGNORECASE)
    if not match:
        return []
    tail = match.group(1)
    tail = tail.replace("VIF above 5", "").replace("VIF above 10", "")
    return [part.strip() for part in tail.split(",") if part.strip()]


def _warning_key_from_diagnostic(item: dict[str, Any]) -> tuple[str, tuple[str, ...]] | None:
    code = str(item.get("code") or "")
    if code not in MULTICOLLINEARITY_CODES:
        return None
    level = "serious" if code == "serious_multicollinearity" else "high"
    return (level, tuple(sorted(_affected_variables(item))))


def _warning_sort_key(item: dict[str, Any]) -> int:
    code = str(item.get("code") or "")
    if code == "serious_multicollinearity":
        return 0
    if code == "high_multicollinearity":
        return 1
    return 2


def _warning_key_from_text(text: str) -> tuple[str, tuple[str, ...]] | None:
    lower = text.lower()
    if "serious multicollinearity" in lower or "vif exceeds 10" in lower or "vif above 10" in lower:
        return ("serious", tuple(sorted(_variables_from_text(text))))
    if "high multicollinearity" in lower or "vif exceeds 5" in lower or "vif above 5" in lower:
        return ("high", tuple(sorted(_variables_from_text(text))))
    return None


def _unique(lines: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for line in lines:
        normalized = " ".join(str(line).split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(str(line))
    return output
