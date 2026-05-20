from __future__ import annotations


TEXT = {
    "en": {
        "confidence": "Rule-based interpretation",
        "no_significant": "No main explanatory variable is statistically significant at the 10% level.",
        "diagnostic_warning": "One diagnostic issue should be reviewed: {warning}",
        "ols_coef": "In OLS, a coefficient is the estimated change in the dependent variable associated with a one-unit increase in the explanatory variable, holding other selected variables constant.",
        "logit_coef": "In Logit, coefficients are in log-odds units and should not be read as percentage-point probability changes.",
        "probit_coef": "In Probit, coefficients are on a latent normal-index scale and should not be read as percentage-point probability changes.",
        "panel_coef": "Panel Fixed Effects focuses on within-entity changes over time after removing selected fixed effects.",
        "causal_limit": "The results describe statistical associations and do not automatically establish causal effects.",
        "review_next": "Review diagnostics, variable definitions, and whether the selected model matches the research design.",
    },
    "zh": {
        "confidence": "规则式解释",
        "no_significant": "核心解释变量在 10% 水平上没有显示出统计显著性。",
        "diagnostic_warning": "需要关注一个诊断提醒：{warning}",
        "ols_coef": "在 OLS 中，系数表示在控制其他已选变量后，解释变量增加 1 个单位时，因变量估计值的平均变化。",
        "logit_coef": "在 Logit 中，系数是对数胜算单位，不能直接解释为概率提高或降低了多少个百分点。",
        "probit_coef": "在 Probit 中，系数属于潜在正态指数尺度，不能直接解释为概率变化的百分点。",
        "panel_coef": "面板固定效应关注同一个体内部随时间变化的关系，并剔除所选固定效应。",
        "causal_limit": "结果反映统计相关关系，不能自动解释为因果关系。",
        "review_next": "建议继续检查诊断结果、变量定义，以及模型是否匹配研究设计。",
    },
}


def narrative_text(language: str, key: str, **kwargs) -> str:
    lang = "zh" if language == "zh" else "en"
    return TEXT[lang][key].format(**kwargs)
