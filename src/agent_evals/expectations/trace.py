"""Trace expectations: assert on the OpenInference trace exported for a context."""

from __future__ import annotations

from typing import Any

from .base import (
    CheckResult,
    EvaluationContext,
    Expectation,
    ExpectationResult,
    _term_check,
)

_SPAN_KIND_ATTR = "openinference.span.kind"


def _extract_span_kinds(trace: dict[str, Any] | None) -> set[str]:
    """Collect every ``openinference.span.kind`` value across all spans."""
    if not trace:
        return set()
    kinds: set[str] = set()
    for item in trace.get("traces", []):
        for span in item.get("spans", []):
            attrs = span.get("attributes") or {}
            kind = attrs.get(_SPAN_KIND_ATTR)
            if isinstance(kind, str):
                kinds.add(kind)
    return kinds


class TraceSpanKinds(Expectation):
    """Each listed OpenInference span kind must appear in the trace.

    Example::

        expectations:
          trace_span_kinds:
            - CHAIN
            - LLM
    """

    key = "trace_span_kinds"

    def __init__(self, kinds: list[str]) -> None:
        self.kinds = kinds

    @classmethod
    def parse(cls, raw: Any) -> "TraceSpanKinds":
        return cls(list(raw or []))

    def to_raw(self) -> Any:
        return list(self.kinds)

    def term_labels(self) -> list[str]:
        return list(self.kinds)

    def resolve(self, ctx: EvaluationContext) -> ExpectationResult:
        present = _extract_span_kinds(ctx.trace)
        if not present and ctx.trace is None:
            return ExpectationResult(
                self.key,
                [
                    _term_check(kind, False, "no trace available for this step")
                    for kind in self.kinds
                ],
            )
        checks = [
            _term_check(
                kind,
                kind in present,
                f"span kind {kind!r} not found in trace (present: {sorted(present) or 'none'})",
            )
            for kind in self.kinds
        ]
        return ExpectationResult(self.key, checks)


def _count_span_kinds(trace: dict[str, Any] | None) -> dict[str, int]:
    """Count spans per ``openinference.span.kind`` value."""
    if not trace:
        return {}
    counts: dict[str, int] = {}
    for item in trace.get("traces", []):
        for span in item.get("spans", []):
            attrs = span.get("attributes") or {}
            kind = attrs.get(_SPAN_KIND_ATTR)
            if isinstance(kind, str):
                counts[kind] = counts.get(kind, 0) + 1
    return counts


class TraceSpanCounts(Expectation):
    """The trace must contain exactly the specified count of each span kind.

    Example::

        expectations:
          trace_span_counts:
            LLM: 1
            TOOL: 1
    """

    key = "trace_span_counts"

    def __init__(self, counts: dict[str, int]) -> None:
        self.counts = counts

    @classmethod
    def parse(cls, raw: Any) -> "TraceSpanCounts":
        return cls({k: int(v) for k, v in (raw or {}).items()})

    def to_raw(self) -> Any:
        return dict(self.counts)

    def term_labels(self) -> list[str]:
        return list(self.counts)

    def resolve(self, ctx: EvaluationContext) -> ExpectationResult:
        actual = _count_span_kinds(ctx.trace)
        if not actual and ctx.trace is None:
            return ExpectationResult(
                self.key,
                [
                    _term_check(kind, False, "no trace available for this step")
                    for kind in self.counts
                ],
            )
        checks = [
            _term_check(
                kind,
                actual.get(kind, 0) == expected,
                f"expected {expected} {kind!r} span(s), found {actual.get(kind, 0)} "
                f"(actual: {dict(sorted(actual.items())) or 'none'})",
            )
            for kind, expected in self.counts.items()
        ]
        return ExpectationResult(self.key, checks)


def _spans_of_kind(trace: dict[str, Any] | None, kind: str) -> list[dict[str, Any]]:
    """Return every span whose ``openinference.span.kind`` equals *kind*."""
    if not trace:
        return []
    result: list[dict[str, Any]] = []
    for item in trace.get("traces", []):
        for span in item.get("spans", []):
            attrs = span.get("attributes") or {}
            if attrs.get(_SPAN_KIND_ATTR) == kind:
                result.append(span)
    return result


class TraceSpanAttributes(Expectation):
    """Assert that at least one span of each given kind has matching attributes.

    Each entry is a dict with a ``kind`` key to select spans, plus
    attribute key-value pairs to check. The pseudo-key ``name`` checks the
    span's top-level ``name`` field rather than an attribute.

    Example::

        expectations:
          trace_span_attributes:
            - kind: LLM
              llm.finish_reason: tool_calls
            - kind: TOOL
              name: complete_tool
    """

    key = "trace_span_attributes"

    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self.entries = entries

    @classmethod
    def parse(cls, raw: Any) -> "TraceSpanAttributes":
        return cls(list(raw or []))

    def to_raw(self) -> Any:
        return list(self.entries)

    def term_labels(self) -> list[str]:
        labels: list[str] = []
        for entry in self.entries:
            kind = entry.get("kind", "?")
            for key in entry:
                if key == "kind":
                    continue
                labels.append(f"{kind}.{key}")
        return labels

    def resolve(self, ctx: EvaluationContext) -> ExpectationResult:
        if ctx.trace is None:
            return ExpectationResult(
                self.key,
                [
                    _term_check(label, False, "no trace available for this step")
                    for label in self.term_labels()
                ],
            )
        checks: list[CheckResult] = []
        for entry in self.entries:
            kind = entry.get("kind", "")
            spans = _spans_of_kind(ctx.trace, kind)
            attr_checks = {k: v for k, v in entry.items() if k != "kind"}
            for attr_key, expected_val in attr_checks.items():
                label = f"{kind}.{attr_key}"
                found = False
                for span in spans:
                    if attr_key == "name":
                        actual = span.get("name")
                    else:
                        actual = (span.get("attributes") or {}).get(attr_key)
                    if actual == expected_val:
                        found = True
                        break
                detail = (
                    "passed"
                    if found
                    else f"no {kind!r} span with {attr_key}={expected_val!r} "
                    f"(checked {len(spans)} span(s))"
                )
                checks.append(_term_check(label, found, detail))
        return ExpectationResult(self.key, checks)
