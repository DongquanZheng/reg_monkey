from __future__ import annotations

import pandas as pd


def _is_code_like(column: str, series: pd.Series) -> bool:
    name = str(column).lower()
    if any(token in name for token in ["code", "id", "编码", "代码"]):
        return True
    text = series.dropna().astype(str)
    return bool(text.str.fullmatch(r"0\d+").any())


def _examples(series: pd.Series, limit: int = 3) -> str:
    values = series.dropna().astype(str).unique().tolist()
    return ", ".join(values[:limit])


def build_variable_reference(df: pd.DataFrame, profile: dict, search_query: str = "") -> pd.DataFrame:
    query = search_query.strip().lower()
    numeric_columns = set(profile["numeric_columns"])
    missing_pct = profile["missing_percentages"]
    rows = []

    for column in profile["columns"]:
        if query and query not in str(column).lower():
            continue

        series = df[column]
        missing = float(missing_pct.get(column, 0) or 0)
        non_missing_unique = int(series.nunique(dropna=True))

        if missing >= 70:
            inferred_type = "mostly missing"
            direct_numeric_use = "no"
            categorical_control_use = "maybe"
            required_handling = "mostly missing; inspect before modeling"
        elif non_missing_unique <= 1:
            inferred_type = "constant"
            direct_numeric_use = "no"
            categorical_control_use = "no"
            required_handling = "constant; not useful as a predictor"
        elif _is_code_like(str(column), series):
            inferred_type = "code-like"
            direct_numeric_use = "caution" if column in numeric_columns else "no"
            categorical_control_use = "maybe"
            required_handling = "consider whether this is an identifier/code rather than a true numeric measure"
        elif column in numeric_columns:
            inferred_type = "numeric"
            direct_numeric_use = "yes"
            categorical_control_use = "not needed"
            required_handling = "none"
        else:
            inferred_type = "categorical"
            direct_numeric_use = "no"
            categorical_control_use = "yes"
            required_handling = "requires dummy encoding"

        rows.append(
            {
                "Variable": column,
                "Type": inferred_type,
                "Missing %": round(missing, 2),
                "Examples": _examples(series),
                "Direct numeric use": direct_numeric_use,
                "Available as categorical control": categorical_control_use,
                "Required handling / notes": required_handling,
            }
        )

    return pd.DataFrame(rows)


def get_ols_eligible_columns(variable_reference: pd.DataFrame) -> list[str]:
    if variable_reference.empty:
        return []
    eligible = variable_reference[variable_reference["Direct numeric use"].isin(["yes", "caution"])]
    return eligible["Variable"].astype(str).tolist()


def get_excluded_variables(variable_reference: pd.DataFrame) -> pd.DataFrame:
    if variable_reference.empty:
        return variable_reference
    return variable_reference[~variable_reference["Direct numeric use"].isin(["yes", "caution"])][
        ["Variable", "Type", "Required handling / notes"]
    ]
