from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from src.llm.contracts import LLMExplanationInput, LLMExplanationOutput


@dataclass(frozen=True)
class LLMGuardrailViolation:
    code: str
    message: str
    evidence: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


CAUSAL_PATTERNS = [
    r"\bcauses?\b",
    r"\bcaused\b",
    r"\bproves?\b",
    r"\bproof\b",
    r"\bcausal effect\b",
    r"导致",
    r"证明",
    r"因果效应",
]

IGNORE_WARNING_PATTERNS = [
    r"ignore (the )?(warnings|diagnostics)",
    r"diagnostics (can|may) be ignored",
    r"无需关注.*诊断",
    r"可以忽略.*诊断",
    r"可以忽略.*警告",
]


def validate_llm_output(payload: LLMExplanationOutput, source: LLMExplanationInput) -> list[LLMGuardrailViolation]:
    text = _combined_text(payload)
    violations: list[LLMGuardrailViolation] = []
    violations.extend(_causal_violations(text))
    violations.extend(_ignored_warning_violations(text))
    violations.extend(_language_violations(payload, source))
    violations.extend(_invented_number_violations(text, source))
    return violations


def assert_llm_output_safe(payload: LLMExplanationOutput, source: LLMExplanationInput) -> None:
    violations = validate_llm_output(payload, source)
    if violations:
        details = "; ".join(f"{item.code}: {item.evidence}" for item in violations)
        raise ValueError(f"LLM explanation failed guardrails: {details}")


def _combined_text(payload: LLMExplanationOutput) -> str:
    return "\n".join(
        [
            payload.explanation_text,
            payload.limitations_text,
            payload.next_steps_text,
            " ".join(payload.safety_flags),
        ]
    )


def _causal_violations(text: str) -> list[LLMGuardrailViolation]:
    violations = []
    for pattern in CAUSAL_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            violations.append(
                LLMGuardrailViolation(
                    code="causal_overclaim",
                    message="The explanation uses unsupported causal language.",
                    evidence=match.group(0),
                )
            )
    return violations


def _ignored_warning_violations(text: str) -> list[LLMGuardrailViolation]:
    violations = []
    for pattern in IGNORE_WARNING_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            violations.append(
                LLMGuardrailViolation(
                    code="diagnostic_dismissal",
                    message="The explanation suggests diagnostics or warnings can be ignored.",
                    evidence=match.group(0),
                )
            )
    return violations


def _language_violations(payload: LLMExplanationOutput, source: LLMExplanationInput) -> list[LLMGuardrailViolation]:
    if payload.language != source.language:
        return [
            LLMGuardrailViolation(
                code="language_mismatch",
                message="The output language does not match the requested language.",
                evidence=f"{payload.language} != {source.language}",
            )
        ]
    return []


def _invented_number_violations(text: str, source: LLMExplanationInput) -> list[LLMGuardrailViolation]:
    allowed = _allowed_number_tokens(source.to_dict())
    violations = []
    for token in _number_tokens(text):
        normalized = _normalize_number_token(token)
        if normalized in _COMMON_ALLOWED_NUMBERS:
            continue
        if normalized not in allowed:
            violations.append(
                LLMGuardrailViolation(
                    code="invented_statistical_value",
                    message="The explanation contains a numerical value not present in the structured input.",
                    evidence=token,
                )
            )
    return violations


def _allowed_number_tokens(value: Any) -> set[str]:
    tokens: set[str] = set()
    if isinstance(value, dict):
        for item in value.values():
            tokens.update(_allowed_number_tokens(item))
    elif isinstance(value, list):
        for item in value:
            tokens.update(_allowed_number_tokens(item))
    elif isinstance(value, int | float):
        tokens.update(_numeric_variants(value))
    elif isinstance(value, str):
        for token in _number_tokens(value):
            tokens.add(_normalize_number_token(token))
    return tokens


def _numeric_variants(value: int | float) -> set[str]:
    numeric = float(value)
    variants = {
        _normalize_number_token(str(value)),
        _normalize_number_token(f"{numeric:.0f}"),
        _normalize_number_token(f"{numeric:.1f}"),
        _normalize_number_token(f"{numeric:.2f}"),
        _normalize_number_token(f"{numeric:.3f}"),
        _normalize_number_token(f"{numeric:.4f}"),
        _normalize_number_token(f"{numeric:.6f}"),
    }
    return {item for item in variants if item}


def _number_tokens(text: str) -> list[str]:
    return re.findall(r"(?<![A-Za-z_])-?\d+(?:\.\d+)?(?:e[+-]?\d+)?%?", text, flags=re.IGNORECASE)


def _normalize_number_token(token: str) -> str:
    token = token.strip().rstrip("%")
    try:
        numeric = float(token)
    except ValueError:
        return token
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.6f}".rstrip("0").rstrip(".")


_COMMON_ALLOWED_NUMBERS = {"0", "1", "5", "10", "0.1", "0.05", "0.01", "0.001"}
