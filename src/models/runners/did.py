from src.models.did import DID_MODEL
from src.models.runners.base import DefinitionModelRunner


class DIDModelRunner(DefinitionModelRunner):
    def __init__(self) -> None:
        super().__init__(DID_MODEL)
