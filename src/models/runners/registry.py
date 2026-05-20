from __future__ import annotations

from src.models.runners.base import BaseModelRunner
from src.models.runners.did import DIDModelRunner
from src.models.runners.iv_2sls import IV2SLSModelRunner
from src.models.runners.logit import LogitModelRunner
from src.models.runners.ols import OLSModelRunner
from src.models.runners.panel_fe import PanelFEModelRunner
from src.models.runners.probit import ProbitModelRunner
from src.models.runners.psm import PSMModelRunner


MODEL_RUNNER_REGISTRY: dict[str, BaseModelRunner] = {}


def register_model_runner(runner: BaseModelRunner) -> None:
    MODEL_RUNNER_REGISTRY[runner.model_id] = runner


def get_model_runner(model_id: str) -> BaseModelRunner:
    try:
        return MODEL_RUNNER_REGISTRY[model_id]
    except KeyError as exc:
        raise ValueError(f"Unknown model runner id: {model_id}") from exc


def list_model_runners() -> list[BaseModelRunner]:
    return list(MODEL_RUNNER_REGISTRY.values())


class ModelRunnerRegistry:
    def get(self, model_id: str) -> BaseModelRunner:
        return get_model_runner(model_id)

    def list(self) -> list[BaseModelRunner]:
        return list_model_runners()


register_model_runner(OLSModelRunner())
register_model_runner(LogitModelRunner())
register_model_runner(ProbitModelRunner())
register_model_runner(PanelFEModelRunner())
register_model_runner(DIDModelRunner())
register_model_runner(IV2SLSModelRunner())
register_model_runner(PSMModelRunner())
