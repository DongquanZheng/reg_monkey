from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.llm.contracts import LLMExplanationMode, LLMExplanationOutput, build_llm_explanation_input
from src.llm.guardrails import LLMGuardrailViolation, validate_llm_output
from src.llm.provider_registry import get_provider
from src.llm.providers import DisabledLLMProviderError
from src.models.execution import ModelRunResult


class ExplanationProvider(Protocol):
    def generate(self, payload): ...


@dataclass(frozen=True)
class LLMExplanationRenderResult:
    output: LLMExplanationOutput | None
    violations: list[LLMGuardrailViolation]
    fallback_to_rule_based: bool
    source_mode: str


def generate_mock_llm_explanation(
    result: ModelRunResult,
    language: str,
    mode: LLMExplanationMode | str,
    report_mode: str = "interpret",
    provider_id: str = "mock",
    provider: ExplanationProvider | None = None,
) -> LLMExplanationRenderResult:
    payload = build_llm_explanation_input(result, language=language, report_mode=report_mode, mode=mode)
    try:
        active_provider = provider or get_provider(provider_id)
    except Exception:
        return LLMExplanationRenderResult(
            output=None,
            violations=[],
            fallback_to_rule_based=True,
            source_mode=str(mode),
        )
    try:
        if hasattr(active_provider, "generate"):
            output = active_provider.generate(payload)
        else:
            output = active_provider.explain(payload)
    except DisabledLLMProviderError:
        return LLMExplanationRenderResult(
            output=None,
            violations=[],
            fallback_to_rule_based=True,
            source_mode=str(mode),
        )
    except Exception:
        return LLMExplanationRenderResult(
            output=None,
            violations=[],
            fallback_to_rule_based=True,
            source_mode=str(mode),
        )
    violations = validate_llm_output(output, payload)
    return LLMExplanationRenderResult(
        output=None if violations else output,
        violations=violations,
        fallback_to_rule_based=bool(violations),
        source_mode=str(mode),
    )
