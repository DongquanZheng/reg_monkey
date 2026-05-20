from __future__ import annotations

from dataclasses import dataclass

from src.llm.contracts import LLMExplanationMode
from src.llm.mock_provider import MockLLMExplanationProvider
from src.llm.providers import DisabledLLMProvider, DisabledLLMProviderError, LLMProvider, LLMProviderMetadata


@dataclass(frozen=True)
class UnknownLLMProviderError(Exception):
    provider_id: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"provider_id": self.provider_id, "message": self.message}


def list_providers(include_disabled: bool = True) -> list[LLMProviderMetadata]:
    providers = _provider_map()
    metadata = [provider.metadata for provider in providers.values()]
    if include_disabled:
        return metadata
    return [item for item in metadata if item.is_enabled_by_default]


def get_provider(provider_id: str = "mock", allow_disabled: bool = False) -> LLMProvider:
    providers = _provider_map()
    if provider_id not in providers:
        raise UnknownLLMProviderError(
            provider_id=provider_id,
            message=f"Unknown LLM provider: {provider_id}",
        )
    provider = providers[provider_id]
    if not provider.metadata.is_enabled_by_default and not allow_disabled:
        raise DisabledLLMProviderError(
            provider_id=provider_id,
            message=f"Provider {provider_id} is disabled.",
        )
    return provider


def _provider_map() -> dict[str, LLMProvider]:
    mode_support = [
        LLMExplanationMode.LLM_ASSISTED.value,
        LLMExplanationMode.RULE_BASED_LLM_POLISHED.value,
    ]
    return {
        "mock": MockLLMExplanationProvider(),
        "openai_placeholder": DisabledLLMProvider(
            "openai_placeholder",
            "llm_provider_openai_placeholder",
            mode_support,
        ),
        "anthropic_placeholder": DisabledLLMProvider(
            "anthropic_placeholder",
            "llm_provider_anthropic_placeholder",
            mode_support,
        ),
    }
