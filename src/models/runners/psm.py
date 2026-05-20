from src.models.psm import PSM_MODEL
from src.models.runners.base import DefinitionModelRunner


class PSMModelRunner(DefinitionModelRunner):
    def __init__(self) -> None:
        super().__init__(PSM_MODEL)
