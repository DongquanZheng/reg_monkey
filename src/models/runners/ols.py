from src.models.ols import OLS_MODEL
from src.models.runners.base import DefinitionModelRunner


class OLSModelRunner(DefinitionModelRunner):
    def __init__(self) -> None:
        super().__init__(OLS_MODEL)
