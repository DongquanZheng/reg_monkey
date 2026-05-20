from __future__ import annotations

from dataclasses import asdict, dataclass

from src.i18n import normalize_language


@dataclass(frozen=True)
class DemoWorkflowStep:
    step_id: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class DemoWorkflowCaution:
    caution_id: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class DemoWorkflowGuide:
    sample_id: str
    language: str
    sample_title: str
    suitable_user_goal: str
    expected_model_family: str
    expected_result_focus: list[str]
    steps: list[DemoWorkflowStep]
    interpretation_cautions: list[DemoWorkflowCaution]
    what_not_to_conclude: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "language": self.language,
            "sample_title": self.sample_title,
            "suitable_user_goal": self.suitable_user_goal,
            "expected_model_family": self.expected_model_family,
            "expected_result_focus": list(self.expected_result_focus),
            "steps": [step.to_dict() for step in self.steps],
            "interpretation_cautions": [caution.to_dict() for caution in self.interpretation_cautions],
            "what_not_to_conclude": list(self.what_not_to_conclude),
        }


_SAMPLE_ALIASES = {
    "ols": "sample_ols",
    "logit": "sample_logit",
    "panel_fe": "sample_panel_fe",
    "did": "sample_did",
    "iv": "sample_iv_2sls",
    "iv_2sls": "sample_iv_2sls",
    "psm": "sample_psm",
}


def build_demo_workflow_guide(sample_id: str, language: str = "en") -> DemoWorkflowGuide:
    canonical_id = _SAMPLE_ALIASES.get(sample_id, sample_id)
    lang = normalize_language(language)
    guide_data = _GUIDES.get(lang, {}).get(canonical_id)
    if guide_data is None:
        raise ValueError(f"Unknown demo workflow guide: {sample_id}")
    return DemoWorkflowGuide(
        sample_id=canonical_id,
        language=lang,
        sample_title=guide_data["sample_title"],
        suitable_user_goal=guide_data["suitable_user_goal"],
        expected_model_family=guide_data["expected_model_family"],
        expected_result_focus=list(guide_data["expected_result_focus"]),
        steps=[DemoWorkflowStep(step_id=f"step_{index + 1}", text=text) for index, text in enumerate(guide_data["steps"])],
        interpretation_cautions=[
            DemoWorkflowCaution(caution_id=f"caution_{index + 1}", text=text)
            for index, text in enumerate(guide_data["interpretation_cautions"])
        ],
        what_not_to_conclude=list(guide_data["what_not_to_conclude"]),
    )


_GUIDES: dict[str, dict[str, dict[str, object]]] = {
    "en": {
        "sample_ols": {
            "sample_title": "OLS synthetic business sample",
            "suitable_user_goal": "Explore a continuous business outcome and its conditional association with several predictors.",
            "expected_model_family": "OLS regression",
            "expected_result_focus": [
                "Coefficient direction and size for digital_index and other predictors.",
                "R-squared and model diagnostics.",
                "Whether warnings suggest unusual data or specification risks.",
            ],
            "steps": [
                "Load the OLS sample dataset.",
                "Confirm preprocessing and variable roles.",
                "Use the suggested setup or configure OLS manually.",
                "Run the model and review Interpret Results.",
                "Export a brief or full report after checking diagnostics.",
            ],
            "interpretation_cautions": [
                "Treat coefficients as conditional associations unless the research design justifies a causal claim.",
                "Check diagnostics before relying on the fitted relationship.",
            ],
            "what_not_to_conclude": [
                "Do not conclude that digital_index causes revenue_growth from this sample alone.",
                "Do not treat a higher R-squared as proof that the specification answers the research question.",
            ],
        },
        "sample_logit": {
            "sample_title": "Logit synthetic adoption sample",
            "suitable_user_goal": "Explore a binary outcome such as adoption versus non-adoption.",
            "expected_model_family": "Logit regression",
            "expected_result_focus": [
                "Coefficient or marginal-effect direction for adoption-related predictors.",
                "Pseudo R-squared and binary-outcome diagnostics.",
                "Class balance and warnings about limited events or separation risk.",
            ],
            "steps": [
                "Load the Logit sample dataset.",
                "Confirm preprocessing and mark the binary outcome appropriately.",
                "Use the suggested setup or configure Logit manually.",
                "Run the model and review coefficient or marginal-effect interpretation.",
                "Check diagnostics before exporting results.",
            ],
            "interpretation_cautions": [
                "Logit coefficients are log-odds unless marginal effects are shown.",
                "Probability interpretation depends on the model scale and selected covariates.",
            ],
            "what_not_to_conclude": [
                "Do not read logit coefficients as direct percentage-point changes.",
                "Do not treat the fitted association as causal by default.",
            ],
        },
        "sample_panel_fe": {
            "sample_title": "Panel FE synthetic firm-year sample",
            "suitable_user_goal": "Explore repeated observations for entities over time using within-entity variation.",
            "expected_model_family": "Panel fixed effects",
            "expected_result_focus": [
                "Within-entity coefficient direction.",
                "Within R-squared and panel diagnostics.",
                "Entity and time structure used by the model.",
            ],
            "steps": [
                "Load the Panel FE sample dataset.",
                "Confirm preprocessing and identify entity and time variables.",
                "Use the suggested setup or configure Panel FE manually.",
                "Run the model and review within-entity interpretation.",
                "Check whether warnings mention weak within-entity variation.",
            ],
            "interpretation_cautions": [
                "Panel fixed effects use within-entity variation; time-invariant differences are absorbed.",
                "Fixed effects do not resolve every omitted-variable concern.",
            ],
            "what_not_to_conclude": [
                "Do not interpret time-invariant group differences as separately estimated effects.",
                "Do not assume fixed effects alone establish causality.",
            ],
        },
        "sample_did": {
            "sample_title": "DID synthetic treated/pre-post sample",
            "suitable_user_goal": "Practice a simple treated/control and pre/post research-design workflow.",
            "expected_model_family": "Difference-in-Differences, manual experimental workflow",
            "expected_result_focus": [
                "The treatment:post estimate.",
                "Regression R-squared and DID diagnostics.",
                "Research-design cautions around treated/control comparability.",
            ],
            "steps": [
                "Load the DID sample dataset.",
                "Confirm preprocessing and variable roles.",
                "Open manual configuration and choose DID.",
                "Select outcome, treatment, post indicator, and optional group or clustering fields.",
                "Run DID only after reviewing the experimental-model caution.",
            ],
            "interpretation_cautions": [
                "DID interpretation requires a credible parallel trends argument; the app cannot prove it from table structure alone.",
                "The current manual DID workflow is for simple treatment timing.",
            ],
            "what_not_to_conclude": [
                "Do not conclude that the treatment effect is identified unless parallel trends and related assumptions are justified.",
                "Do not treat the candidate DID path as an automatic recommendation.",
            ],
        },
        "sample_iv_2sls": {
            "sample_title": "IV/2SLS synthetic instrument sample",
            "suitable_user_goal": "Practice specifying an endogenous variable and an instrument in a guarded manual workflow.",
            "expected_model_family": "IV/2SLS, manual experimental workflow",
            "expected_result_focus": [
                "The fitted endogenous-variable estimate in the second stage.",
                "Second-stage R-squared and IV diagnostics.",
                "First-stage relevance diagnostics when shown.",
            ],
            "steps": [
                "Load the IV/2SLS sample dataset.",
                "Confirm preprocessing and variable roles.",
                "Open manual configuration and choose IV/2SLS.",
                "Select outcome, endogenous variable, instrument, and controls.",
                "Run IV/2SLS only after reviewing the instrument assumptions.",
            ],
            "interpretation_cautions": [
                "First-stage relevance can be diagnosed, but the exclusion restriction requires research/background justification.",
                "Instrument exogeneity cannot be proven from the dataset alone.",
            ],
            "what_not_to_conclude": [
                "Do not conclude that the causal effect is identified without defending the exclusion restriction.",
                "Do not treat the IV candidate path as automatic causal evidence.",
            ],
        },
        "sample_psm": {
            "sample_title": "PSM synthetic matching sample",
            "suitable_user_goal": "Practice comparing treated and control units using observed matching covariates.",
            "expected_model_family": "Propensity score matching, manual experimental workflow",
            "expected_result_focus": [
                "ATT estimate and matched sample size.",
                "Balance diagnostics for observed covariates.",
                "Overlap and matching warnings.",
            ],
            "steps": [
                "Load the PSM sample dataset.",
                "Confirm preprocessing and variable roles.",
                "Open manual configuration and choose PSM.",
                "Select outcome, treatment, and matching covariates.",
                "Run PSM only after checking that covariates are appropriate pre-treatment controls.",
            ],
            "interpretation_cautions": [
                "PSM balances observed covariates; unobserved confounding remains possible.",
                "Interpretation should depend on overlap and post-matching balance diagnostics.",
            ],
            "what_not_to_conclude": [
                "Do not conclude that matching removes all selection bias.",
                "Do not treat ATT as a universal treatment effect for all units.",
            ],
        },
    },
    "zh": {
        "sample_ols": {
            "sample_title": "OLS 合成业务示例",
            "suitable_user_goal": "用于查看连续型业务结果与多个解释变量之间的条件关联。",
            "expected_model_family": "OLS 回归",
            "expected_result_focus": [
                "digital_index 等变量的系数方向和大小。",
                "R² 与模型诊断。",
                "是否存在数据或设定风险提示。",
            ],
            "steps": [
                "加载 OLS 示例数据。",
                "确认预处理结果和变量角色。",
                "使用规则式建议设定，或手动配置 OLS。",
                "运行模型并查看解释结果。",
                "检查诊断后再导出简明或完整报告。",
            ],
            "interpretation_cautions": [
                "除非研究设计能够支持因果解释，否则系数应理解为条件关联。",
                "依赖拟合关系前，请先检查诊断结果。",
            ],
            "what_not_to_conclude": [
                "不要仅凭该示例认定 digital_index 导致 revenue_growth 变化。",
                "不要把较高 R² 视为研究问题已经得到回答的证明。",
            ],
        },
        "sample_logit": {
            "sample_title": "Logit 合成采用示例",
            "suitable_user_goal": "用于查看采用/未采用等二元结果变量。",
            "expected_model_family": "Logit 回归",
            "expected_result_focus": [
                "与采用相关变量的系数或边际效应方向。",
                "伪 R² 与二元结果诊断。",
                "类别平衡、事件数量或分离风险提示。",
            ],
            "steps": [
                "加载 Logit 示例数据。",
                "确认预处理，并将二元结果变量标记为合适角色。",
                "使用规则式建议设定，或手动配置 Logit。",
                "运行模型并查看系数或边际效应解释。",
                "导出结果前先检查诊断。",
            ],
            "interpretation_cautions": [
                "除非显示边际效应，否则 Logit 系数表示对数胜算。",
                "概率解释取决于模型尺度和所选协变量。",
            ],
            "what_not_to_conclude": [
                "不要把 Logit 系数直接解读为百分点变化。",
                "不要默认把拟合关联解释为因果关系。",
            ],
        },
        "sample_panel_fe": {
            "sample_title": "面板固定效应合成企业年度示例",
            "suitable_user_goal": "用于查看同一实体随时间重复观测中的实体内变化。",
            "expected_model_family": "面板固定效应",
            "expected_result_focus": [
                "实体内系数方向。",
                "组内 R² 与面板诊断。",
                "模型使用的实体和时间结构。",
            ],
            "steps": [
                "加载面板固定效应示例数据。",
                "确认预处理，并识别实体变量和时间变量。",
                "使用规则式建议设定，或手动配置面板固定效应。",
                "运行模型并查看实体内解释。",
                "检查是否存在实体内变异不足等提示。",
            ],
            "interpretation_cautions": [
                "面板固定效应依赖实体内变化；不随时间变化的差异会被吸收。",
                "固定效应不能解决所有遗漏变量问题。",
            ],
            "what_not_to_conclude": [
                "不要把时间不变的组间差异解释为单独估计出来的效应。",
                "不要认为固定效应本身即可证明因果关系。",
            ],
        },
        "sample_did": {
            "sample_title": "DID 合成处理前后示例",
            "suitable_user_goal": "用于练习简单的处理组/对照组与处理前/处理后研究设计流程。",
            "expected_model_family": "双重差分，手动实验性工作流",
            "expected_result_focus": [
                "treatment:post 交互项估计。",
                "回归 R² 与 DID 诊断。",
                "关于处理组/对照组可比性的研究设计提示。",
            ],
            "steps": [
                "加载 DID 示例数据。",
                "确认预处理结果和变量角色。",
                "进入手动配置并选择 DID。",
                "选择结果变量、处理变量、处理后变量，以及可选分组或聚类字段。",
                "在查看实验性模型提示后再运行 DID。",
            ],
            "interpretation_cautions": [
                "DID 解释需要可信的平行趋势论证；应用无法仅凭表格结构证明这一点。",
                "当前手动 DID 工作流适用于简单处理时点。",
            ],
            "what_not_to_conclude": [
                "除非平行趋势及相关假设得到论证，否则不要认定处理效应已经被识别。",
                "不要把候选 DID 路径视为自动建议。",
            ],
        },
        "sample_iv_2sls": {
            "sample_title": "IV/2SLS 合成工具变量示例",
            "suitable_user_goal": "用于练习在受保护的手动流程中指定内生变量和工具变量。",
            "expected_model_family": "IV/2SLS，手动实验性工作流",
            "expected_result_focus": [
                "第二阶段中拟合内生变量的估计。",
                "第二阶段 R² 与 IV 诊断。",
                "如有显示，查看第一阶段相关性诊断。",
            ],
            "steps": [
                "加载 IV/2SLS 示例数据。",
                "确认预处理结果和变量角色。",
                "进入手动配置并选择 IV/2SLS。",
                "选择结果变量、内生变量、工具变量和控制变量。",
                "在查看工具变量假设后再运行 IV/2SLS。",
            ],
            "interpretation_cautions": [
                "第一阶段相关性可以诊断，但排除限制需要研究背景论证。",
                "工具变量外生性无法仅凭数据集证明。",
            ],
            "what_not_to_conclude": [
                "未论证排除限制前，不要认定因果效应已经被识别。",
                "不要把 IV 候选路径视为自动因果证据。",
            ],
        },
        "sample_psm": {
            "sample_title": "PSM 合成匹配示例",
            "suitable_user_goal": "用于练习基于可观测协变量比较处理组和对照组。",
            "expected_model_family": "倾向得分匹配，手动实验性工作流",
            "expected_result_focus": [
                "ATT 估计和匹配样本量。",
                "可观测协变量的平衡诊断。",
                "重叠和匹配风险提示。",
            ],
            "steps": [
                "加载 PSM 示例数据。",
                "确认预处理结果和变量角色。",
                "进入手动配置并选择 PSM。",
                "选择结果变量、处理变量和匹配协变量。",
                "确认协变量适合作为处理前控制变量后再运行 PSM。",
            ],
            "interpretation_cautions": [
                "PSM 只能平衡可观测协变量；未观测混杂仍然可能存在。",
                "解释应依赖重叠情况和匹配后的平衡诊断。",
            ],
            "what_not_to_conclude": [
                "不要认定匹配已经消除了所有选择偏差。",
                "不要把 ATT 解释为适用于所有个体的总体处理效应。",
            ],
        },
    },
}
