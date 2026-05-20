from __future__ import annotations

from typing import Any

import pandas as pd
from statsmodels.discrete.discrete_model import Probit

from src.models.base import ModelDefinition
from src.models.binary_choice import (
    diagnose_binary_choice_model,
    fit_binary_choice_model,
    validate_binary_choice_inputs,
)


def validate_probit_inputs(df: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    return validate_binary_choice_inputs(df, config, "Probit regression")


def fit_probit(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    return fit_binary_choice_model(
        df=df,
        config=config,
        model_label="Probit regression",
        model_type="probit",
        model_factory=Probit,
    )


PROBIT_MODEL = ModelDefinition(
    model_id="probit",
    display_name_en="Probit Regression",
    display_name_zh="Probit 回归",
    description_en="Estimate a binary outcome model using a normal latent-index link.",
    description_zh="用于二元结果变量的正态潜变量指数模型。",
    required_roles=["dependent_variable", "independent_variables"],
    validate=validate_probit_inputs,
    fit=fit_probit,
    diagnostics=diagnose_binary_choice_model,
    report_label_en="Probit Regression",
    report_label_zh="Probit 回归",
    limitations_en=[
        "Requires a binary dependent variable.",
        "Coefficients are latent-index units and are not direct percentage-point effects.",
        "Regression results alone do not prove causality.",
        "Classification threshold diagnostics were not generated for this model result.",
    ],
    limitations_zh=[
        "因变量必须是二元结果。",
        "系数是潜变量指数单位，不能直接解释为概率提高或降低了多少个百分点。",
        "回归结果本身不能自动证明因果关系。",
        "当前模型结果未生成分类阈值诊断。",
    ],
)
