from src.models.logit import LOGIT_MODEL
from src.models.runners.base import DefinitionModelRunner


class LogitModelRunner(DefinitionModelRunner):
    def __init__(self) -> None:
        super().__init__(LOGIT_MODEL)
