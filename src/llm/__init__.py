"""Controlled LLM explanation contract layer.

This package defines backend-only contracts for future LLM-assisted explanation.
It does not connect to external providers.
"""

from src.llm.contracts import (
    LLMExplanationInput,
    LLMExplanationMode,
    LLMExplanationOutput,
    build_llm_explanation_input,
    llm_input_to_dict,
    llm_output_to_dict,
)
from src.llm.provider_registry import get_provider, list_providers
from src.llm.providers import DisabledLLMProviderError, LLMProviderMetadata

__all__ = [
    "LLMExplanationInput",
    "LLMExplanationMode",
    "LLMExplanationOutput",
    "build_llm_explanation_input",
    "llm_input_to_dict",
    "llm_output_to_dict",
    "get_provider",
    "list_providers",
    "DisabledLLMProviderError",
    "LLMProviderMetadata",
]
