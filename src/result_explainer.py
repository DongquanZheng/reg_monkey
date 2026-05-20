"""Beginner-facing result reading guides.

The guide layer is intentionally UI-oriented and deterministic. It describes how
to read existing structured results; it does not inspect raw data, run models, or
change result objects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class BeginnerResultGuideItem:
    item_id: str
    title: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BeginnerResultCaution:
    caution_id: str
    title: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BeginnerResultGuide:
    model_id: str
    title: str
    summary: str
    items: list[BeginnerResultGuideItem]
    cautions: list[BeginnerResultCaution]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "title": self.title,
            "summary": self.summary,
            "items": [item.to_dict() for item in self.items],
            "cautions": [caution.to_dict() for caution in self.cautions],
        }


def build_beginner_result_guide(
    result: object,
    model_spec: object,
    language: str = "en",
) -> BeginnerResultGuide:
    """Build a short model-family-aware guide for reading an existing result."""

    model_id = _resolve_model_id(result, model_spec)
    language_key = "zh" if str(language).lower().startswith("zh") else "en"
    guide_factory = _ZH_GUIDES if language_key == "zh" else _EN_GUIDES
    return guide_factory.get(model_id, guide_factory["ols"])()


def _resolve_model_id(result: object, model_spec: object) -> str:
    for source in (model_spec, result):
        value = _read_field(source, "model_id") or _read_field(source, "model_type")
        if value:
            return _normalize_model_id(value)

    summary = _read_field(result, "model_summary")
    value = _read_field(summary, "model_type") or _read_field(summary, "model_id")
    return _normalize_model_id(value or "ols")


def _read_field(source: object, field: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(field)
    return getattr(source, field, None)


def _normalize_model_id(value: object) -> str:
    normalized = str(value or "ols").strip().lower()
    aliases = {
        "panel": "panel_fe",
        "fixed_effects": "panel_fe",
        "panel fixed effects": "panel_fe",
        "iv": "iv_2sls",
        "iv/2sls": "iv_2sls",
        "2sls": "iv_2sls",
        "propensity_score_matching": "psm",
    }
    return aliases.get(normalized, normalized)


def _item(item_id: str, title: str, description: str) -> BeginnerResultGuideItem:
    return BeginnerResultGuideItem(item_id=item_id, title=title, description=description)


def _caution(caution_id: str, title: str, description: str) -> BeginnerResultCaution:
    return BeginnerResultCaution(caution_id=caution_id, title=title, description=description)


def _guide(
    model_id: str,
    title: str,
    summary: str,
    items: list[BeginnerResultGuideItem],
    cautions: list[BeginnerResultCaution],
) -> BeginnerResultGuide:
    return BeginnerResultGuide(
        model_id=model_id,
        title=title,
        summary=summary,
        items=items,
        cautions=cautions,
    )


def _en_ols() -> BeginnerResultGuide:
    return _guide(
        "ols",
        "OLS result guide",
        "Read the coefficient, p-value, R-squared, and diagnostics together.",
        [
            _item("look_first", "What to look at first", "Start with the coefficient direction, p-value, R-squared, and diagnostic status."),
            _item("main_number", "What the main number means", "A coefficient estimates the conditional association between a predictor and the outcome, holding selected variables constant."),
            _item("check_first", "What to check before trusting it", "Review diagnostics for sample size, missing data, outliers, and multicollinearity."),
            _item("do_not_conclude", "What not to conclude", "Do not interpret the estimate as causal by default."),
        ],
        [
            _caution("association_only", "Interpretation boundary", "OLS summarizes association unless the research design supports a stronger interpretation."),
        ],
    )


def _en_logit() -> BeginnerResultGuide:
    return _guide(
        "logit",
        "Logit result guide",
        "Read direction, p-value, pseudo R-squared, and marginal effects when available.",
        [
            _item("look_first", "What to look at first", "Start with coefficient direction, p-value, pseudo R-squared, and any marginal effects table."),
            _item("main_number", "What the main number means", "Raw Logit coefficients are on a log-odds scale and are not probability-point changes by default."),
            _item("check_first", "What to check before trusting it", "Check class balance, sample size, convergence, and diagnostics before interpreting the estimate."),
            _item("do_not_conclude", "What not to conclude", "Do not describe a coefficient as a direct probability-point change unless a marginal effect is shown."),
        ],
        [
            _caution("binary_scale", "Coefficient scale", "Use marginal effects or odds ratios for more intuitive probability-related interpretation when available."),
        ],
    )


def _en_probit() -> BeginnerResultGuide:
    return _guide(
        "probit",
        "Probit result guide",
        "Read direction, p-value, pseudo R-squared, and marginal effects when available.",
        [
            _item("look_first", "What to look at first", "Start with coefficient direction, p-value, pseudo R-squared, and any marginal effects table."),
            _item("main_number", "What the main number means", "Raw Probit coefficients are on an index scale and are not probability-point changes by default."),
            _item("check_first", "What to check before trusting it", "Check class balance, sample size, convergence, and diagnostics before interpreting the estimate."),
            _item("do_not_conclude", "What not to conclude", "Do not describe a coefficient as a direct probability-point change unless a marginal effect is shown."),
        ],
        [
            _caution("binary_scale", "Coefficient scale", "Use marginal effects for more intuitive probability-related interpretation when available."),
        ],
    )


def _en_panel_fe() -> BeginnerResultGuide:
    return _guide(
        "panel_fe",
        "Panel Fixed Effects result guide",
        "Read the within-entity relationship, within R-squared, and panel diagnostics.",
        [
            _item("look_first", "What to look at first", "Start with the coefficient direction, p-value, within R-squared, entity count, and time-period count."),
            _item("main_number", "What the main number means", "The coefficient uses within-entity variation and describes changes within the same entity over time."),
            _item("check_first", "What to check before trusting it", "Confirm that the key predictor varies within entities and that diagnostics do not show major risk."),
            _item("do_not_conclude", "What not to conclude", "Time-invariant differences are absorbed, and fixed effects do not remove every omitted-variable concern."),
        ],
        [
            _caution("within_variation", "Within-entity interpretation", "Panel FE depends on meaningful within-entity variation; variables that do not vary within an entity are not separately interpreted."),
        ],
    )


def _en_did() -> BeginnerResultGuide:
    return _guide(
        "did",
        "DID result guide",
        "Read the treatment:post estimate with research-design assumptions in view.",
        [
            _item("look_first", "What to look at first", "Start with the treatment:post estimate, p-value, regression R-squared, and DID diagnostics."),
            _item("main_number", "What the main number means", "The treatment:post estimate summarizes the post-period change for the treated group relative to the comparison group."),
            _item("check_first", "What to check before trusting it", "Confirm treatment/control definitions, pre/post timing, and a credible parallel trends argument."),
            _item("do_not_conclude", "What not to conclude", "Do not interpret the DID estimate causally unless the research design is justified."),
        ],
        [
            _caution("parallel_trends", "Parallel trends", "Parallel trends are required for causal interpretation and are not verified automatically by the app."),
        ],
    )


def _en_iv() -> BeginnerResultGuide:
    return _guide(
        "iv_2sls",
        "IV/2SLS result guide",
        "Read the fitted endogenous estimate together with first-stage diagnostics.",
        [
            _item("look_first", "What to look at first", "Start with the fitted endogenous estimate, p-value, second-stage R-squared, and first-stage diagnostics if available."),
            _item("main_number", "What the main number means", "The fitted endogenous coefficient estimates the relationship under the chosen instrument setup."),
            _item("check_first", "What to check before trusting it", "Check first-stage relevance and explain why the exclusion restriction is credible in the research context."),
            _item("do_not_conclude", "What not to conclude", "Do not treat IV output as causal unless instrument relevance, exogeneity, and exclusion restriction are justified."),
        ],
        [
            _caution("exclusion_restriction", "Exclusion restriction", "The exclusion restriction cannot be proven from data alone and requires research/background justification."),
        ],
    )


def _en_psm() -> BeginnerResultGuide:
    return _guide(
        "psm",
        "PSM result guide",
        "Read ATT, matched sample size, and balance diagnostics together.",
        [
            _item("look_first", "What to look at first", "Start with the ATT estimate, matched treated/control counts, and balance diagnostics."),
            _item("main_number", "What the main number means", "ATT summarizes the estimated difference for treated units after matching on observed covariates."),
            _item("check_first", "What to check before trusting it", "Check whether observed covariates are balanced after matching and whether common support is reasonable."),
            _item("do_not_conclude", "What not to conclude", "Do not assume unobserved confounding has been removed."),
        ],
        [
            _caution("observed_covariates", "Observed covariates only", "PSM balances observed covariates; unobserved confounding may remain."),
        ],
    )


def _zh_ols() -> BeginnerResultGuide:
    return _guide(
        "ols",
        "OLS 结果阅读指引",
        "结合系数、p 值、R 平方和诊断一起阅读。",
        [
            _item("look_first", "先看什么", "先看系数方向、p 值、R 平方和诊断状态。"),
            _item("main_number", "主要数字含义", "系数表示在控制已选变量后，解释变量与结果变量之间的条件关联。"),
            _item("check_first", "信任前检查", "查看样本量、缺失、异常值和多重共线性等诊断。"),
            _item("do_not_conclude", "不要推断", "不要默认把估计结果解释为因果关系。"),
        ],
        [
            _caution("association_only", "解释边界", "OLS 描述关联；只有研究设计支持时才可作更强解释。"),
        ],
    )


def _zh_logit() -> BeginnerResultGuide:
    return _guide(
        "logit",
        "Logit 结果阅读指引",
        "结合方向、p 值、伪 R 平方和可用的边际效应一起阅读。",
        [
            _item("look_first", "先看什么", "先看系数方向、p 值、伪 R 平方，以及是否有边际效应表。"),
            _item("main_number", "主要数字含义", "Logit 原始系数是对数胜算尺度，默认不是概率百分点变化。"),
            _item("check_first", "信任前检查", "解释前检查类别比例、样本量、收敛情况和诊断信息。"),
            _item("do_not_conclude", "不要推断", "除非展示边际效应，不要把系数直接说成概率百分点变化。"),
        ],
        [
            _caution("binary_scale", "系数尺度", "如有边际效应或胜算比，可用于更直观的概率相关解释。"),
        ],
    )


def _zh_probit() -> BeginnerResultGuide:
    return _guide(
        "probit",
        "Probit 结果阅读指引",
        "结合方向、p 值、伪 R 平方和可用的边际效应一起阅读。",
        [
            _item("look_first", "先看什么", "先看系数方向、p 值、伪 R 平方，以及是否有边际效应表。"),
            _item("main_number", "主要数字含义", "Probit 原始系数是指数尺度，默认不是概率百分点变化。"),
            _item("check_first", "信任前检查", "解释前检查类别比例、样本量、收敛情况和诊断信息。"),
            _item("do_not_conclude", "不要推断", "除非展示边际效应，不要把系数直接说成概率百分点变化。"),
        ],
        [
            _caution("binary_scale", "系数尺度", "如有边际效应，可用于更直观的概率相关解释。"),
        ],
    )


def _zh_panel_fe() -> BeginnerResultGuide:
    return _guide(
        "panel_fe",
        "面板固定效应结果阅读指引",
        "重点阅读实体内关系、组内 R 平方和面板诊断。",
        [
            _item("look_first", "先看什么", "先看系数方向、p 值、组内 R 平方、个体数量和时间期数。"),
            _item("main_number", "主要数字含义", "系数依赖实体内变化，描述同一个体随时间变化时的关系。"),
            _item("check_first", "信任前检查", "确认关键解释变量在个体内部有足够变化，并查看诊断风险。"),
            _item("do_not_conclude", "不要推断", "时间不变差异会被吸收，固定效应也不能解决所有遗漏变量问题。"),
        ],
        [
            _caution("within_variation", "实体内解释", "面板固定效应依赖有意义的实体内变化；个体内不变的变量不能被单独解释。"),
        ],
    )


def _zh_did() -> BeginnerResultGuide:
    return _guide(
        "did",
        "DID 结果阅读指引",
        "结合研究设计假设阅读 treatment:post 估计。",
        [
            _item("look_first", "先看什么", "先看 treatment:post 估计、p 值、回归 R 平方和 DID 诊断。"),
            _item("main_number", "主要数字含义", "treatment:post 估计描述处理组在处理后相对对照组的变化。"),
            _item("check_first", "信任前检查", "确认处理组/对照组定义、前后期划分，以及可信的平行趋势论证。"),
            _item("do_not_conclude", "不要推断", "只有研究设计得到充分说明时，才可谨慎进行因果解释。"),
        ],
        [
            _caution("parallel_trends", "平行趋势", "因果解释需要平行趋势假设；应用不会自动验证该假设。"),
        ],
    )


def _zh_iv() -> BeginnerResultGuide:
    return _guide(
        "iv_2sls",
        "IV/2SLS 结果阅读指引",
        "结合拟合内生变量估计和第一阶段诊断阅读。",
        [
            _item("look_first", "先看什么", "先看拟合内生变量估计、p 值、第二阶段 R 平方，以及可用的第一阶段诊断。"),
            _item("main_number", "主要数字含义", "拟合内生变量系数是在所选工具变量设定下得到的关系估计。"),
            _item("check_first", "信任前检查", "检查第一阶段相关性，并说明为什么排除限制在研究背景中可信。"),
            _item("do_not_conclude", "不要推断", "只有工具变量相关性、外生性和排除限制得到论证时，才可谨慎进行因果解释。"),
        ],
        [
            _caution("exclusion_restriction", "排除限制", "排除限制不能仅凭数据证明，需要研究背景或理论论证。"),
        ],
    )


def _zh_psm() -> BeginnerResultGuide:
    return _guide(
        "psm",
        "PSM 结果阅读指引",
        "结合 ATT、匹配样本量和平衡诊断一起阅读。",
        [
            _item("look_first", "先看什么", "先看 ATT 估计、匹配后的处理组/对照组数量和平衡诊断。"),
            _item("main_number", "主要数字含义", "ATT 表示基于可观测协变量匹配后，处理组的估计差异。"),
            _item("check_first", "信任前检查", "检查匹配后可观测协变量是否平衡，以及共同支持是否合理。"),
            _item("do_not_conclude", "不要推断", "不要认为未观测混杂已经被消除。"),
        ],
        [
            _caution("observed_covariates", "仅限可观测协变量", "PSM 平衡的是可观测协变量；未观测混杂仍可能存在。"),
        ],
    )


_EN_GUIDES = {
    "ols": _en_ols,
    "logit": _en_logit,
    "probit": _en_probit,
    "panel_fe": _en_panel_fe,
    "did": _en_did,
    "iv_2sls": _en_iv,
    "psm": _en_psm,
}

_ZH_GUIDES = {
    "ols": _zh_ols,
    "logit": _zh_logit,
    "probit": _zh_probit,
    "panel_fe": _zh_panel_fe,
    "did": _zh_did,
    "iv_2sls": _zh_iv,
    "psm": _zh_psm,
}
