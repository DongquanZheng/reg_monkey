from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from src.models.base import ModelDefinition


class BaseModelRunner(ABC):
    model_id: str

    @property
    @abstractmethod
    def model_definition(self) -> ModelDefinition:
        raise NotImplementedError

    def validate_spec(self, spec: Any, data: pd.DataFrame) -> list[str]:
        return self.model_definition.validate(data, spec.to_config())

    def fit(self, spec: Any, data: pd.DataFrame) -> dict[str, Any]:
        return self.model_definition.fit(data, spec.to_config())

    def diagnostics(
        self,
        original_data: pd.DataFrame,
        cleaned_data: pd.DataFrame,
        spec: Any,
        fit_payload: dict[str, Any],
    ) -> dict[str, Any]:
        diagnostics_fn = self.model_definition.diagnostics
        if diagnostics_fn is None:
            return {}
        return diagnostics_fn(original_data, cleaned_data, spec.to_config(), fit_payload)

    def run(self, spec: Any, data: pd.DataFrame) -> Any:
        from src.models.execution import _run_with_runner

        return _run_with_runner(data, spec, self)


class DefinitionModelRunner(BaseModelRunner):
    def __init__(self, model_definition: ModelDefinition) -> None:
        self._model_definition = model_definition
        self.model_id = model_definition.model_id

    @property
    def model_definition(self) -> ModelDefinition:
        return self._model_definition
