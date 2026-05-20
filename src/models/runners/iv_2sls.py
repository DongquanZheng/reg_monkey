from src.models.iv import IV_2SLS_MODEL
from src.models.runners.base import DefinitionModelRunner


class IV2SLSModelRunner(DefinitionModelRunner):
    def __init__(self) -> None:
        super().__init__(IV_2SLS_MODEL)
