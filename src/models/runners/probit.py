from src.models.probit import PROBIT_MODEL
from src.models.runners.base import DefinitionModelRunner


class ProbitModelRunner(DefinitionModelRunner):
    def __init__(self) -> None:
        super().__init__(PROBIT_MODEL)
