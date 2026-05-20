from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol

from src.llm.contracts import LLMExplanationInput, LLMExplanationOutput


@dataclass(frozen=True)
class LLMProviderMetadata:
    provider_id: str
    display_name_key: str
    supports_network: bool
    is_enabled_by_default: bool
    mode_support: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class LLMProvider(Protocol):
    provider_id: str
    metadata: LLMProviderMetadata

    def generate(self, explanation_input: LLMExplanationInput) -> LLMExplanationOutput:
        ...


@dataclass(frozen=True)
class DisabledLLMProviderError(Exception):
    provider_id: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"provider_id": self.provider_id, "message": self.message}


class DisabledLLMProvider:
    def __init__(self, provider_id: str, display_name_key: str, mode_support: list[str] | None = None) -> None:
        self.provider_id = provider_id
        self.metadata = LLMProviderMetadata(
            provider_id=provider_id,
            display_name_key=display_name_key,
            supports_network=False,
            is_enabled_by_default=False,
            mode_support=list(mode_support or []),
        )

    def generate(self, explanation_input: LLMExplanationInput) -> LLMExplanationOutput:
        raise DisabledLLMProviderError(
            provider_id=self.provider_id,
            message=f"Provider {self.provider_id} is disabled and cannot generate explanations.",
        )
