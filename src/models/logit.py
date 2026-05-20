from __future__ import annotations

from typing import Any

import pandas as pd
from statsmodels.discrete.discrete_model import Logit

from src.models.base import ModelDefinition
from src.models.binary_choice import (
    diagnose_binary_choice_model,
    fit_binary_choice_model,
    validate_binary_choice_inputs,
)


def validate_logit_inputs(df: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    return validate_binary_choice_inputs(df, config, "Logit regression")


def fit_logit(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    return fit_binary_choice_model(
        df=df,
        config=config,
        model_label="Logit regression",
        model_type="logit",
        model_factory=Logit,
    )


LOGIT_MODEL = ModelDefinition(
    model_id="logit",
    display_name_en="Logit Regression",
    display_name_zh="Logit 回归",
    description_en="Estimate a binary outcome model using log-odds, for outcomes such as yes/no or 0/1.",
    description_zh="用于二元结果变量的模型，例如是否违约、是否出口、是否采用某政策或 0/1 结果。",
    required_roles=["dependent_variable", "independent_variables"],
    validate=validate_logit_inputs,
    fit=fit_logit,
    diagnostics=diagnose_binary_choice_model,
    report_label_en="Logit Regression",
    report_label_zh="Logit 回归",
    limitations_en=[
        "Requires a binary dependent variable.",
        "Coefficients are in log-odds units and are not direct percentage-point effects.",
        "Regression results alone do not prove causality.",
        "Classification threshold diagnostics were not generated for this model result.",
    ],
    limitations_zh=[
        "因变量必须是二元结果。",
        "系数是对数胜算单位，不能直接解释为概率提高或降低了多少个百分点。",
        "回归结果本身不能自动证明因果关系。",
        "当前模型结果未生成分类阈值诊断。",
    ],
)
