"""Substring expectations over the response's full JSON: presence and absence."""

from __future__ import annotations

import re
from typing import Any

from .base import EvaluationContext, Expectation, ExpectationResult, _term_check


def _text_terms(raw: Any) -> list[str]:
    """Normalize the plain-list and ``{text: [...]}`` authoring shapes to a list."""
    if isinstance(raw, dict):
        return list(raw.get("text", []))
    return list(raw or [])


class MustInclude(Expectation):
    """Each phrase must appear (whole-word) in the response's full JSON."""

    key = "must_include"

    def __init__(self, phrases: list[str]) -> None:
        self.phrases = phrases

    @classmethod
    def parse(cls, raw: Any) -> "MustInclude":
        return cls(_text_terms(raw))

    def to_raw(self) -> Any:
        return list(self.phrases)

    def term_labels(self) -> list[str]:
        return list(self.phrases)

    def resolve(self, ctx: EvaluationContext) -> ExpectationResult:
        checks = [
            _term_check(
                phrase,
                bool(re.search(rf"\b{re.escape(phrase)}\b", ctx.haystack)),
                f"missing required phrase: {phrase!r}",
            )
            for phrase in self.phrases
        ]
        return ExpectationResult(self.key, checks)


class MustNotInclude(Expectation):
    """Each phrase must be absent (whole-word) from the response's full JSON."""

    key = "must_not_include"

    def __init__(self, phrases: list[str]) -> None:
        self.phrases = phrases

    @classmethod
    def parse(cls, raw: Any) -> "MustNotInclude":
        return cls(_text_terms(raw))

    def to_raw(self) -> Any:
        return list(self.phrases)

    def term_labels(self) -> list[str]:
        return list(self.phrases)

    def resolve(self, ctx: EvaluationContext) -> ExpectationResult:
        checks = [
            _term_check(
                phrase,
                not re.search(rf"\b{re.escape(phrase)}\b", ctx.haystack),
                f"found forbidden phrase: {phrase!r}",
            )
            for phrase in self.phrases
        ]
        return ExpectationResult(self.key, checks)
