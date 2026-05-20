from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd


ValidationFunction = Callable[[pd.DataFrame, dict[str, Any]], list[str]]
FitFunction = Callable[[pd.DataFrame, dict[str, Any]], dict[str, Any]]
DiagnosticsFunction = Callable[[pd.DataFrame, pd.DataFrame, dict[str, Any], Any], dict[str, Any]]


@dataclass(frozen=True)
class ModelDefinition:
    """Registered metadata and callable hooks for an empirical model."""

    model_id: str
    display_name_en: str
    display_name_zh: str
    description_en: str
    description_zh: str
    required_roles: list[str]
    validate: ValidationFunction
    fit: FitFunction
    diagnostics: DiagnosticsFunction | None
    report_label_en: str
    report_label_zh: str
    limitations_en: list[str]
    limitations_zh: list[str]

    def display_name(self, language: str) -> str:
        return self.display_name_zh if language == "zh" else self.display_name_en

    def description(self, language: str) -> str:
        return self.description_zh if language == "zh" else self.description_en

    def report_label(self, language: str) -> str:
        return self.report_label_zh if language == "zh" else self.report_label_en

    def limitations(self, language: str) -> list[str]:
        return self.limitations_zh if language == "zh" else self.limitations_en
