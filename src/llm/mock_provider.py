from __future__ import annotations

from src.llm.contracts import LLMExplanationInput, LLMExplanationMode, LLMExplanationOutput
from src.llm.guardrails import assert_llm_output_safe
from src.llm.providers import LLMProviderMetadata


class MockLLMExplanationProvider:
    """Deterministic test provider; it never calls external services."""

    provider_id = "mock"
    metadata = LLMProviderMetadata(
        provider_id="mock",
        display_name_key="llm_provider_mock",
        supports_network=False,
        is_enabled_by_default=True,
        mode_support=[
            LLMExplanationMode.LLM_ASSISTED.value,
            LLMExplanationMode.RULE_BASED_LLM_POLISHED.value,
        ],
    )

    def generate(self, payload: LLMExplanationInput) -> LLMExplanationOutput:
        if payload.language == "zh":
            output = LLMExplanationOutput(
                explanation_text=f"当前结果来自 {payload.model_name}。解释应基于结构化系数、拟合指标和诊断信息。",
                limitations_text="当前解释仅说明统计相关关系，不能自动解释为因果关系。",
                next_steps_text="建议核查模型设定、关键诊断和报告中的局限性。",
                safety_flags=["mock_provider", "no_external_api", "structured_input_only"],
                language="zh",
                source_mode=payload.mode,
            )
        else:
            output = LLMExplanationOutput(
                explanation_text=f"The current result comes from {payload.model_name}. The explanation should use structured coefficients, fit metrics, and diagnostics.",
                limitations_text="This explanation describes statistical association and does not establish causality.",
                next_steps_text="Review the model setup, key diagnostics, and report limitations.",
                safety_flags=["mock_provider", "no_external_api", "structured_input_only"],
                language="en",
                source_mode=payload.mode,
            )
        assert_llm_output_safe(output, payload)
        return output

    def explain(self, payload: LLMExplanationInput) -> LLMExplanationOutput:
        return self.generate(payload)
