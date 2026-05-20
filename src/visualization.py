from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


PlotResult = tuple[Optional[plt.Figure], Optional[str]]


def plot_correlation_heatmap(df: pd.DataFrame) -> PlotResult:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return None, "At least two numeric variables are required for a correlation heatmap."

    fig, ax = plt.subplots(figsize=(8, 5))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    return fig, None


def plot_missing_values(df: pd.DataFrame) -> PlotResult:
    """Plot missing value counts by variable."""
    if df.empty:
        return None, "The dataset is empty, so missing values cannot be plotted."

    missing_counts = df.isna().sum().sort_values(ascending=False)
    if missing_counts.sum() == 0:
        return None, "No missing values were found in this dataset."

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=missing_counts.values, y=missing_counts.index, ax=ax, color="#4c78a8")
    ax.set_title("Missing Values by Variable")
    ax.set_xlabel("Missing value count")
    ax.set_ylabel("Variable")
    fig.tight_layout()
    return fig, None


def plot_histogram(df: pd.DataFrame, column: str) -> PlotResult:
    if column not in df.columns:
        return None, f"Column '{column}' was not found."
    if not pd.api.types.is_numeric_dtype(df[column]):
        return None, "A numeric variable is required for a histogram."

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df[column].dropna(), kde=True, ax=ax)
    ax.set_title(f"Histogram of {column}")
    ax.set_xlabel(column)
    fig.tight_layout()
    return fig, None


def plot_boxplot(df: pd.DataFrame, column: str) -> PlotResult:
    if column not in df.columns:
        return None, f"Column '{column}' was not found."
    if not pd.api.types.is_numeric_dtype(df[column]):
        return None, "A numeric variable is required for a boxplot."

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(y=df[column].dropna(), ax=ax)
    ax.set_title(f"Boxplot of {column}")
    ax.set_ylabel(column)
    fig.tight_layout()
    return fig, None


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str) -> PlotResult:
    missing_cols = [col for col in (x_col, y_col) if col not in df.columns]
    if missing_cols:
        return None, f"Column(s) not found: {', '.join(missing_cols)}"
    if not pd.api.types.is_numeric_dtype(df[x_col]) or not pd.api.types.is_numeric_dtype(df[y_col]):
        return None, "Numeric x and y variables are required for a scatter plot."

    plot_df = df[[x_col, y_col]].dropna()
    if plot_df.empty:
        return None, "No complete observations are available for this scatter plot."

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(data=plot_df, x=x_col, y=y_col, ax=ax)
    ax.set_title(f"{y_col} vs. {x_col}")
    fig.tight_layout()
    return fig, None


def plot_pairwise_scatter(df: pd.DataFrame, columns: list[str]) -> PlotResult:
    """Create a small pairwise scatter matrix for selected numeric variables."""
    if len(columns) < 2:
        return None, "Select at least two numeric variables for a pairwise scatter plot."
    if len(columns) > 5:
        return None, "Select five or fewer variables to keep the pairwise plot readable."

    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        return None, f"Column(s) not found: {', '.join(missing_cols)}"

    non_numeric = [col for col in columns if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric:
        return None, "Pairwise scatter plots require numeric variables: " + ", ".join(non_numeric)

    plot_df = df[columns].dropna()
    if len(plot_df) < 2:
        return None, "At least two complete observations are required for a pairwise scatter plot."

    grid = sns.pairplot(plot_df, diag_kind="hist", corner=True)
    grid.fig.suptitle("Pairwise Scatter Plot", y=1.02)
    grid.fig.tight_layout()
    return grid.fig, None


def plot_categorical_bar(df: pd.DataFrame, column: str, top_n: int = 10) -> PlotResult:
    if column not in df.columns:
        return None, f"Column '{column}' was not found."
    counts = df[column].dropna().astype(str).value_counts().head(top_n)
    if counts.empty:
        return None, "No non-missing categories are available for this variable."

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=counts.values, y=counts.index, ax=ax, color="#4c78a8")
    ax.set_title(f"Top categories: {column}")
    ax.set_xlabel("Count")
    ax.set_ylabel(column)
    fig.tight_layout()
    return fig, None


def plot_numeric_by_category(df: pd.DataFrame, numeric_col: str, category_col: str) -> PlotResult:
    missing_cols = [col for col in (numeric_col, category_col) if col not in df.columns]
    if missing_cols:
        return None, f"Column(s) not found: {', '.join(missing_cols)}"
    if not pd.api.types.is_numeric_dtype(df[numeric_col]):
        return None, "A numeric variable is required for category comparison."

    plot_df = df[[numeric_col, category_col]].dropna().copy()
    if plot_df.empty:
        return None, "No complete observations are available for this category comparison."

    top_categories = plot_df[category_col].astype(str).value_counts().head(10).index
    plot_df = plot_df[plot_df[category_col].astype(str).isin(top_categories)]
    summary = plot_df.groupby(category_col, dropna=True)[numeric_col].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=summary.values, y=summary.index.astype(str), ax=ax, color="#59a14f")
    ax.set_title(f"Mean {numeric_col} by {category_col}")
    ax.set_xlabel(f"Mean {numeric_col}")
    ax.set_ylabel(category_col)
    fig.tight_layout()
    return fig, None


def plot_time_trend(df: pd.DataFrame, time_col: str, numeric_col: str, agg: str = "mean") -> PlotResult:
    missing_cols = [col for col in (time_col, numeric_col) if col not in df.columns]
    if missing_cols:
        return None, f"Column(s) not found: {', '.join(missing_cols)}"
    if not pd.api.types.is_numeric_dtype(df[numeric_col]):
        return None, "A numeric variable is required for a time trend."

    plot_df = df[[time_col, numeric_col]].dropna().copy()
    if plot_df.empty:
        return None, "No complete observations are available for this time trend."

    if agg == "sum":
        trend = plot_df.groupby(time_col, dropna=True)[numeric_col].sum()
    else:
        trend = plot_df.groupby(time_col, dropna=True)[numeric_col].mean()
    trend = trend.sort_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(x=trend.index.astype(str), y=trend.values, marker="o", ax=ax)
    ax.set_title(f"{agg.title()} {numeric_col} by {time_col}")
    ax.set_xlabel(time_col)
    ax.set_ylabel(f"{agg.title()} {numeric_col}")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig, None
