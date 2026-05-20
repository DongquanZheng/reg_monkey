from typing import Any

import pandas as pd

from src.models.panel_fe import PANEL_FE_MODEL
from src.models.runners.base import DefinitionModelRunner


class PanelFEModelRunner(DefinitionModelRunner):
    def __init__(self) -> None:
        super().__init__(PANEL_FE_MODEL)

    def validate_spec(self, spec: Any, data: pd.DataFrame) -> list[str]:
        return PANEL_FE_MODEL.validate(data, spec.to_config())

    def fit(self, spec: Any, data: pd.DataFrame) -> dict[str, Any]:
        return PANEL_FE_MODEL.fit(data, spec.to_config())

    def diagnostics(
        self,
        original_data: pd.DataFrame,
        cleaned_data: pd.DataFrame,
        spec: Any,
        fit_payload: dict[str, Any],
    ) -> dict[str, Any]:
        return PANEL_FE_MODEL.diagnostics(original_data, cleaned_data, spec.to_config(), fit_payload)
