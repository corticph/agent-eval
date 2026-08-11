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


def _iter_spans(trace: dict[str, Any] | None):
    """Yield every span across all traces in the response."""
    if not trace:
        return
    for item in trace.get("traces", []):
        for span in item.get("spans", []):
            yield span


def _build_span_index(
    trace: dict[str, Any] | None
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return (spans, id_to_span) from the trace."""
    spans: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for span in _iter_spans(trace):
        spans.append(span)
        sid = span.get("span_id")
        if sid:
            by_id[sid] = span
    return spans, by_id


def _span_matches(span: dict[str, Any], matcher: dict[str, Any]) -> bool:
    """Check whether *span* satisfies all fields of *matcher*."""
    kind = matcher.get("kind")
    if kind is not None:
        attrs = span.get("attributes") or {}
        if attrs.get(_SPAN_KIND_ATTR) != kind:
            return False
    name = matcher.get("name")
    if name is not None and span.get("name") != name:
        return False
    attributes = matcher.get("attributes")
    if attributes:
        attrs = span.get("attributes") or {}
        for key, expected in attributes.items():
            if attrs.get(key) != expected:
                return False
    return True


def _resolve_parent_chain(
    span: dict[str, Any],
    parent: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
) -> bool:
    """Check whether *span*'s parent satisfies the *parent* matcher."""
    parent_id = span.get("parent_span_id")
    if not parent_id:
        return False
    parent_span = by_id.get(parent_id)
    if parent_span is None:
        return False
    if not _span_matches(parent_span, parent):
        return False
    grandparent = parent.get("parent")
    if grandparent:
        if not _resolve_parent_chain(parent_span, grandparent, by_id):
            return False
    return True


class Trace(Expectation):
    """Assert that the OpenInference trace for the case's context matches.

    Each entry is a span matcher: a dict that selects spans by ``kind``,
    ``name``, and/or ``attributes``, with an optional ``exact`` count for the
    expected number of matching spans (default 1), and an optional ``parent``
    matcher that a matched span's parent must also satisfy.

    Matching runs against the trace of the WHOLE context, fetched per step:
    in a sequential (multi-step) case sharing one context, span counts
    accumulate across earlier steps. An ``exact`` count therefore describes
    the conversation so far, not just the current step — a step-2 entry
    ``kind: LLM`` fails when both steps produced one LLM span each. Assert
    cumulative totals, or scope matchers with ``attributes``/``parent`` so
    prior steps cannot match them.

    Example::

        expectations:
          trace:
            - kind: CHAIN
              attributes:
                opik.metadata.final_state: TASK_STATE_COMPLETED
              exact: 2
            - kind: LLM
              attributes:
                llm.finish_reason: tool_calls
            - kind: TOOL
              name: complete_tool
            - kind: LLM
              parent:
                kind: CHAIN
    """

    key = "trace"

    def __init__(self, matchers: list[dict[str, Any]]) -> None:
        self.matchers = matchers

    @classmethod
    def parse(cls, raw: Any) -> "Trace":
        return cls(list(raw or []))

    def to_raw(self) -> Any:
        return list(self.matchers)

    def needs_trace(self) -> bool:
        return True

    def term_labels(self) -> list[str]:
        labels: list[str] = []
        for i, m in enumerate(self.matchers):
            kind = m.get("kind", "?")
            name = m.get("name")
            label = kind if not name else f"{kind}:{name}"
            if "exact" in m and m["exact"] != 1:
                label = f"{label}(exact={m['exact']})"
            if "parent" in m:
                pkind = m["parent"].get("kind", "?")
                label = f"{label}<-{pkind}"
            labels.append(label)
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
        spans, by_id = _build_span_index(ctx.trace)
        checks: list[CheckResult] = []
        for i, matcher in enumerate(self.matchers):
            expected_count = matcher.get("exact", 1)
            parent = matcher.get("parent")
            matched = 0
            for span in spans:
                if not _span_matches(span, matcher):
                    continue
                if parent and not _resolve_parent_chain(span, parent, by_id):
                    continue
                matched += 1
            label = self.term_labels()[i]
            passed = matched == expected_count
            if passed:
                detail = f"found {matched} matching span(s)"
            else:
                kind = matcher.get("kind", "any")
                detail = (
                    f"expected exactly {expected_count} {kind!r} span(s) matching; "
                    f"found {matched}"
                )
                if ctx.trace_url:
                    detail += f" — {ctx.trace_url}"
            checks.append(_term_check(label, passed, detail))
        return ExpectationResult(self.key, checks)
