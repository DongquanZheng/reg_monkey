from __future__ import annotations

from src.models.base import ModelDefinition
from src.models.logit import LOGIT_MODEL
from src.models.ols import OLS_MODEL
from src.models.panel_fe import PANEL_FE_MODEL
from src.models.probit import PROBIT_MODEL


MODEL_REGISTRY: dict[str, ModelDefinition] = {}


def register_model(model: ModelDefinition) -> None:
    MODEL_REGISTRY[model.model_id] = model


def get_available_models() -> list[ModelDefinition]:
    return list(MODEL_REGISTRY.values())


def list_models() -> list[ModelDefinition]:
    return get_available_models()


def get_model(model_id: str) -> ModelDefinition:
    try:
        return MODEL_REGISTRY[model_id]
    except KeyError as exc:
        raise ValueError(f"Unknown model id: {model_id}") from exc


register_model(OLS_MODEL)
register_model(LOGIT_MODEL)
register_model(PROBIT_MODEL)
register_model(PANEL_FE_MODEL)
