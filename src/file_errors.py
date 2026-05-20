from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


FriendlyFileErrorCode = Literal[
    "unsupported_file_type",
    "empty_file",
    "malformed_csv",
    "malformed_excel",
    "encoding_error",
    "duplicate_column_names",
    "blank_column_names",
    "no_usable_rows",
    "no_usable_columns",
    "no_numeric_variables",
    "too_many_columns_warning_only",
    "unknown_file_load_error",
]


@dataclass(frozen=True)
class FriendlyFileError(Exception):
    code: FriendlyFileErrorCode
    severity: str = "error"
    affected_columns: list[str] = field(default_factory=list)
    detail: str = ""

    @property
    def title_key(self) -> str:
        return f"friendly_file_error_{self.code}_title"

    @property
    def message_key(self) -> str:
        return f"friendly_file_error_{self.code}_message"

    @property
    def action_key(self) -> str:
        return f"friendly_file_error_{self.code}_action"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "affected_columns": list(self.affected_columns),
            "detail": self.detail,
            "title_key": self.title_key,
            "message_key": self.message_key,
            "action_key": self.action_key,
        }


def friendly_file_error_to_dict(error: FriendlyFileError) -> dict[str, Any]:
    return error.to_dict()


def validate_loaded_dataframe(df: Any) -> None:
    columns = list(getattr(df, "columns", []))
    if len(columns) == 0:
        raise FriendlyFileError("no_usable_columns")
    if getattr(df, "empty", False):
        raise FriendlyFileError("no_usable_rows")

    column_labels = [str(column) for column in columns]
    duplicate_columns = _duplicate_column_names(column_labels)
    if duplicate_columns:
        raise FriendlyFileError("duplicate_column_names", affected_columns=duplicate_columns)


def build_file_quality_warnings(df: Any, *, too_many_columns_threshold: int = 300) -> list[FriendlyFileError]:
    warnings: list[FriendlyFileError] = []
    columns = list(getattr(df, "columns", []))
    column_labels = [str(column) for column in columns]
    blank_columns = [column for column in column_labels if _is_blank_column_name(column)]
    if blank_columns:
        warnings.append(FriendlyFileError("blank_column_names", severity="warning", affected_columns=blank_columns))
    if len(columns) > too_many_columns_threshold:
        warnings.append(FriendlyFileError("too_many_columns_warning_only", severity="warning"))
    if columns and not _has_numeric_like_column(df):
        warnings.append(FriendlyFileError("no_numeric_variables", severity="warning"))
    return warnings


def _duplicate_column_names(column_labels: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for label in column_labels:
        normalized = _normalize_mangled_duplicate(label)
        if normalized in seen and normalized not in duplicates:
            duplicates.append(normalized)
        seen.add(normalized)
    return duplicates


def _normalize_mangled_duplicate(label: str) -> str:
    head, dot, suffix = label.rpartition(".")
    if dot and suffix.isdigit() and head:
        return head
    return label


def _is_blank_column_name(label: str) -> bool:
    stripped = label.strip()
    return not stripped or stripped.lower().startswith("unnamed:")


def _has_numeric_like_column(df: Any) -> bool:
    for column in getattr(df, "columns", []):
        values = df[column].dropna().astype(str).str.strip()
        values = values[values != ""]
        if values.empty:
            continue
        sample = values.head(50).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
        numeric = sample.str.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")
        if float(numeric.mean()) >= 0.8:
            return True
    return False
