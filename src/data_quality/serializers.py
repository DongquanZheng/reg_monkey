from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from src.reproducibility.serializers import to_jsonable


def data_quality_to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, list):
        return [data_quality_to_jsonable(item) for item in value]
    return to_jsonable(value)
