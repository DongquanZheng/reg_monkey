from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd


ROLE_NUMERIC = "Numeric measure"
ROLE_BINARY = "Binary variable"
ROLE_CATEGORICAL = "Categorical variable"
ROLE_CODE = "Code / Identifier"
ROLE_TIME = "Time variable"
ROLE_ENTITY = "Entity ID"
ROLE_EXCLUDE = "Exclude"

ROLE_OPTIONS = [
    ROLE_NUMERIC,
    ROLE_BINARY,
    ROLE_CATEGORICAL,
    ROLE_CODE,
    ROLE_TIME,
    ROLE_ENTITY,
    ROLE_EXCLUDE,
]


ROLE_LABELS = {
    "en": {
        ROLE_NUMERIC: "Numeric measure",
        ROLE_BINARY: "Binary variable",
        ROLE_CATEGORICAL: "Categorical variable",
        ROLE_CODE: "Code / Identifier",
        ROLE_TIME: "Time variable",
        ROLE_ENTITY: "Entity ID",
        ROLE_EXCLUDE: "Exclude",
    },
    "zh": {
        ROLE_NUMERIC: "数值变量",
        ROLE_BINARY: "二元变量",
        ROLE_CATEGORICAL: "分类变量",
        ROLE_CODE: "编码 / ID",
        ROLE_TIME: "时间变量",
        ROLE_ENTITY: "个体 ID",
        ROLE_EXCLUDE: "忽略",
    },
}

BINARY_TRUE = {"1", "true", "yes", "y", "是", "是的"}
BINARY_FALSE = {"0", "false", "no", "n", "否", "不是"}


def role_label(role: str, language: str = "en") -> str:
    lang = "zh" if language == "zh" else "en"
    return ROLE_LABELS[lang].get(role, role)


def role_from_label(label: str, language: str = "en") -> str:
    lang = "zh" if language == "zh" else "en"
    reverse = {value: key for key, value in ROLE_LABELS[lang].items()}
    return reverse.get(label, label)


def role_options(language: str = "en") -> list[str]:
    return [role_label(role, language) for role in ROLE_OPTIONS]


def is_binary_like(series: pd.Series) -> bool:
    values = series.dropna()
    if values.empty:
        return False
    normalized = set(values.astype(str).str.strip().str.lower().unique().tolist())
    if len(normalized) != 2:
        return False
    if normalized.issubset(BINARY_TRUE | BINARY_FALSE):
        return True
    if normalized.issubset({"0.0", "1.0"}):
        return True
    return False


def _name_tokens(column: Any) -> str:
    return str(column).strip().lower()


def _is_time_name(column: Any) -> bool:
    name = _name_tokens(column)
    return any(token in name for token in ["year", "date", "month", "quarter", "年度", "年份", "时间", "日期", "月份", "季度"])


def _is_entity_name(column: Any) -> bool:
    name = _name_tokens(column)
    entity_tokens = ["firm_id", "company_id", "entity_id", "unit_id", "企业id", "公司id", "个体id"]
    if any(token in name for token in entity_tokens):
        return True
    return ("id" in name or "编号" in name) and any(token in name for token in ["firm", "company", "entity", "unit", "企业", "公司", "个体"])


def _is_code_name(column: Any) -> bool:
    name = _name_tokens(column)
    return any(token in name for token in ["code", "area_code", "region_code", "代码", "编码"])


def _has_leading_zero_codes(series: pd.Series) -> bool:
    text = series.dropna().astype(str).str.strip()
    return bool(text.str.fullmatch(r"0\d+").any())


def infer_variable_roles(df: pd.DataFrame, profile: dict[str, Any]) -> dict[str, str]:
    roles: dict[str, str] = {}
    numeric_columns = set(profile.get("numeric_columns", []))
    missing_pct = profile.get("missing_percentages", {})

    for column in profile.get("columns", df.columns.tolist()):
        series = df[column]
        missing = float(missing_pct.get(column, 0) or 0)
        unique_count = int(series.nunique(dropna=True))

        if missing >= 70 or unique_count == 0:
            roles[column] = ROLE_EXCLUDE
        elif _is_entity_name(column):
            roles[column] = ROLE_ENTITY
        elif _is_time_name(column):
            roles[column] = ROLE_TIME
        elif _is_code_name(column) or _has_leading_zero_codes(series):
            roles[column] = ROLE_CODE
        elif is_binary_like(series):
            roles[column] = ROLE_BINARY
        elif column in numeric_columns:
            roles[column] = ROLE_NUMERIC if unique_count > 1 else ROLE_EXCLUDE
        else:
            roles[column] = ROLE_CATEGORICAL

    return roles


def _examples(series: pd.Series, limit: int = 3) -> str:
    values = series.dropna().astype(str).unique().tolist()
    return ", ".join(values[:limit])


def build_variable_role_table(
    df: pd.DataFrame,
    profile: dict[str, Any],
    confirmed_roles: dict[str, str] | None = None,
) -> pd.DataFrame:
    inferred = infer_variable_roles(df, profile)
    roles = confirmed_roles or inferred
    missing_pct = profile.get("missing_percentages", {})
    numeric_columns = set(profile.get("numeric_columns", []))

    rows = []
    for column in profile.get("columns", df.columns.tolist()):
        series = df[column]
        rows.append(
            {
                "variable": column,
                "inferred_type": "numeric" if column in numeric_columns else "categorical/text",
                "missing_percentage": round(float(missing_pct.get(column, 0) or 0), 2),
                "unique_values": int(series.nunique(dropna=True)),
                "examples": _examples(series),
                "recommended_role": inferred[column],
                "confirmed_role": roles.get(column, inferred[column]),
            }
        )
    return pd.DataFrame(rows)


def get_variables_by_role(roles: dict[str, str], role: str) -> list[str]:
    return [column for column, assigned_role in roles.items() if assigned_role == role]


def get_numeric_measure_variables(roles: dict[str, str]) -> list[str]:
    return get_variables_by_role(roles, ROLE_NUMERIC)


def get_binary_variables(roles: dict[str, str]) -> list[str]:
    return get_variables_by_role(roles, ROLE_BINARY)


def get_categorical_variables(roles: dict[str, str]) -> list[str]:
    return get_variables_by_role(roles, ROLE_CATEGORICAL)


def get_time_variables(roles: dict[str, str]) -> list[str]:
    return get_variables_by_role(roles, ROLE_TIME)


def get_entity_id_variables(roles: dict[str, str]) -> list[str]:
    return get_variables_by_role(roles, ROLE_ENTITY)


def get_code_identifier_variables(roles: dict[str, str]) -> list[str]:
    return get_variables_by_role(roles, ROLE_CODE)


def get_correlation_variables(roles: dict[str, str], include_time: bool = False) -> list[str]:
    variables = get_numeric_measure_variables(roles)
    if include_time:
        variables += get_time_variables(roles)
    return variables


def get_scatter_variables(roles: dict[str, str]) -> list[str]:
    return get_numeric_measure_variables(roles)


def get_ols_selector_variables(roles: dict[str, str]) -> dict[str, list[str]]:
    numeric = get_numeric_measure_variables(roles)
    categorical = get_categorical_variables(roles)
    return {
        "dependent_variables": numeric + get_binary_variables(roles),
        "main_independent_variables": numeric,
        "numeric_controls": numeric,
        "categorical_controls": categorical,
    }


def get_binary_dependent_candidates(df: pd.DataFrame, roles: dict[str, str]) -> list[str]:
    binary = [column for column in get_binary_variables(roles) if column in df.columns]
    compatible: list[str] = []
    for column, role in roles.items():
        if column in binary or column not in df.columns:
            continue
        if role not in {ROLE_CATEGORICAL, ROLE_NUMERIC}:
            continue
        if is_binary_like(df[column]):
            compatible.append(column)
    return binary + compatible


def summarize_role_counts(roles: dict[str, str]) -> dict[str, int]:
    counts = Counter(roles.values())
    return {role: int(counts.get(role, 0)) for role in ROLE_OPTIONS}
