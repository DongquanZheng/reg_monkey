from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class VariableQualitySummary:
    variable: str
    dtype: str
    inferred_role_hint: str
    missing_count: int
    missing_percentage: float
    unique_count: int
    unique_percentage: float
    is_binary_like: bool
    is_id_like: bool
    is_constant: bool
    is_near_constant: bool
    is_high_cardinality: bool
    is_text_numeric_like: bool
    is_datetime_like: bool = False
    is_mixed_type: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MissingnessProfile:
    total_missing_cells: int
    total_missing_percentage: float
    complete_case_rows: int
    complete_case_percentage: float
    missing_by_variable: list[dict[str, Any]] = field(default_factory=list)
    high_missing_variables: list[str] = field(default_factory=list)
    all_missing_variables: list[str] = field(default_factory=list)
    rows_with_any_missing: int = 0
    columns_with_any_missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelSampleImpact:
    selected_variables: list[str]
    original_rows: int
    usable_rows_after_dropna: int
    dropped_rows: int
    dropped_percentage: float
    variables_causing_missing_loss: list[str]
    warning_level: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DataQualityProfile:
    row_count: int
    column_count: int
    duplicate_row_count: int
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    binary_columns: list[str] = field(default_factory=list)
    datetime_like_columns: list[str] = field(default_factory=list)
    id_like_columns: list[str] = field(default_factory=list)
    high_cardinality_columns: list[str] = field(default_factory=list)
    constant_columns: list[str] = field(default_factory=list)
    near_constant_columns: list[str] = field(default_factory=list)
    text_numeric_columns: list[str] = field(default_factory=list)
    mixed_type_columns: list[str] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
