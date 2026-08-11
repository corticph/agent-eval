"""Tests for the standalone ``expectations`` module.

These exercise the registry seam directly — parsing an ``expectations:`` block
into a homogeneous list of ``Expectation`` objects and resolving each against a
response into per-term ``ExpectationResult``s. They assert external behaviour
(keys, per-term ``label``/``passed``/``detail``, ``show_on_pass``), never
registry internals or class structure.
"""

from __future__ import annotations

import pytest

from agent_evals import expectations as exp
from agent_evals.expectations import (
    EvaluationContext,
    ExpectedState,
    Judge,
    parse_expectations,
    resolve_all,
)


def _ctx(
    response: dict | None = None,
    *,
    duration_seconds: float | None = None,
    trace: dict | None = None,
    trace_url: str | None = None,
) -> EvaluationContext:
    return EvaluationContext.from_response(
        response,
        duration_seconds=duration_seconds,
        trace=trace,
        trace_url=trace_url,
    )


def _resolve(block: dict | None, response: dict | None = None, **ctx_kw):
    """Parse *block*, resolve against *response*, return {key: ExpectationResult}."""
    parsed = parse_expectations(block)
    results = resolve_all(parsed, _ctx(response, **ctx_kw))
    return {r.key: r for r in results}


def _text_response(text: str) -> dict:
    return {"task": {"status": {"message": {"parts": [{"text": text}]}}}}


def _data_response(data) -> dict:
    return {"task": {"artifacts": [{"parts": [{"data": data}]}]}}


def _state_response(state: str, text: str = "ok") -> dict:
    return {
        "task": {
            "id": "task-1",
            "status": {"state": state, "message": {"parts": [{"text": text}]}},
        }
    }


# --- Registry: every declared type is discoverable in one place -------------


def test_registry_lists_all_eleven_types():
    assert set(exp.registry()) == {
        "must_include",
        "must_not_include",
        "must_match",
        "jsonpath",
        "must_include_json",
        "max_duration_seconds",
        "expected_output_text",
        "cited_pubmed_ids",
        "faithful_to_source",
        "expected_state",
        "trace",
    }


# --- Strict parse -----------------------------------------------------------


def test_unknown_key_raises_listing_known_keys():
    with pytest.raises(ValueError) as excinfo:
        parse_expectations({"must_inlcude": ["hi"]})
    message = str(excinfo.value)
    assert "must_inlcude" in message
    # Every known key is offered so the author can spot their typo.
    assert "must_include" in message
    assert "expected_state" in message


def test_default_expected_state_injected_when_none_declared():
    parsed = parse_expectations({"must_include": ["hi"]})
    keys = [e.key for e in parsed]
    assert "expected_state" in keys


def test_empty_block_still_injects_the_terminal_state_guard():
    parsed = parse_expectations(None)
    assert [e.key for e in parsed] == ["expected_state"]
    assert isinstance(parsed[0], ExpectedState)


def test_declared_expected_state_not_duplicated():
    parsed = parse_expectations({"expected_state": "input-required"})
    assert [e.key for e in parsed] == ["expected_state"]


def test_result_list_follows_canonical_not_authoring_order():
    # Authored judge-first, must_include-second; canonical order is the reverse.
    block = {"expected_output_text": "ref", "must_include": ["hi"]}
    parsed = parse_expectations(block)
    keys = [e.key for e in parsed]
    assert keys.index("must_include") < keys.index("expected_output_text")


# --- must_include / must_not_include: list + {text} normalization -----------


def test_must_include_plain_list_one_check_per_term():
    results = _resolve(
        {"must_include": ["alpha", "omega"]}, _text_response("alpha only")
    )
    checks = results["must_include"].checks
    assert [(c.label, c.passed) for c in checks] == [("alpha", True), ("omega", False)]
    assert not results["must_include"].passed


def test_must_include_text_dict_shape_normalizes():
    results = _resolve(
        {"must_include": {"text": ["alpha"]}}, _text_response("alpha here")
    )
    checks = results["must_include"].checks
    assert [(c.label, c.passed) for c in checks] == [("alpha", True)]


def test_must_not_include_passes_and_fails_per_term():
    results = _resolve(
        {"must_not_include": ["forbidden", "clean"]},
        _text_response("this is forbidden"),
    )
    checks = results["must_not_include"].checks
    assert [(c.label, c.passed) for c in checks] == [
        ("forbidden", False),
        ("clean", True),
    ]


def test_must_not_include_text_dict_shape_normalizes():
    results = _resolve(
        {"must_not_include": {"text": ["nope"]}}, _text_response("all good")
    )
    assert results["must_not_include"].passed


# --- must_match -------------------------------------------------------------


def test_must_match_regex_alternation():
    results = _resolve(
        {"must_match": ["life[ -]?threatening|fatal", "nowhere"]},
        _text_response("this is life-threatening"),
    )
    checks = results["must_match"].checks
    assert checks[0].passed
    assert not checks[1].passed


# --- jsonpath / must_include_json -------------------------------------------


def test_jsonpath_pass_and_fail():
    response = _data_response({"count": 3})
    passing = _resolve({"jsonpath": [{"path": "$.count", "equals": 3}]}, response)
    assert passing["jsonpath"].passed
    assert passing["jsonpath"].checks[0].label == "$.count"

    failing = _resolve({"jsonpath": [{"path": "$.count", "equals": 4}]}, response)
    assert not failing["jsonpath"].passed


def test_jsonpath_no_data_parts_fails():
    results = _resolve(
        {"jsonpath": [{"path": "$.x", "exists": True}]}, _text_response("no data")
    )
    assert not results["jsonpath"].passed


def test_jsonpath_unknown_comparator_fails_loudly():
    results = _resolve(
        {"jsonpath": [{"path": "$.count", "bogus": 1}]}, _data_response({"count": 3})
    )
    assert not results["jsonpath"].passed


def test_must_include_json_partial_match():
    response = _data_response({"items": [{"type": "TEMP", "value": 38.5, "unit": "C"}]})
    passing = _resolve(
        {"must_include_json": [{"items": [{"type": "TEMP", "value": 38.5}]}]}, response
    )
    assert passing["must_include_json"].passed

    failing = _resolve(
        {"must_include_json": [{"items": [{"type": "TEMP", "value": 99}]}]}, response
    )
    assert not failing["must_include_json"].passed


# --- max_duration_seconds (scalar) ------------------------------------------


def test_max_duration_scalar_one_check():
    within = _resolve(
        {"max_duration_seconds": 10}, _text_response("x"), duration_seconds=5.0
    )
    assert len(within["max_duration_seconds"].checks) == 1
    assert within["max_duration_seconds"].checks[0].label == ""
    assert within["max_duration_seconds"].passed

    over = _resolve(
        {"max_duration_seconds": 10}, _text_response("x"), duration_seconds=20.0
    )
    assert not over["max_duration_seconds"].passed


def test_max_duration_without_recorded_duration_passes():
    results = _resolve({"max_duration_seconds": 10}, _text_response("x"))
    assert results["max_duration_seconds"].passed


# --- judge: show_on_pass ----------------------------------------------------


def test_judge_show_on_pass_carries_explanation(monkeypatch):
    monkeypatch.setattr(
        Judge, "_judge", staticmethod(lambda prompt: (True, "looks right"))
    )
    results = _resolve(
        {"expected_output_text": "the reference"}, _text_response("some answer")
    )
    result = results["expected_output_text"]
    assert result.show_on_pass is True
    assert result.passed
    assert result.checks[0].detail == "looks right"


def test_judge_failure_detail(monkeypatch):
    monkeypatch.setattr(
        Judge, "_judge", staticmethod(lambda prompt: (False, "missing facts"))
    )
    results = _resolve(
        {"expected_output_text": "the reference"}, _text_response("wrong")
    )
    result = results["expected_output_text"]
    assert not result.passed
    assert result.checks[0].detail == "missing facts"


def test_judge_reference_list_joins_lines(monkeypatch):
    captured = {}

    def _fake(prompt):
        captured["prompt"] = prompt
        return True, "ok"

    monkeypatch.setattr(Judge, "_judge", staticmethod(_fake))
    _resolve({"expected_output_text": ["line one", "line two"]}, _text_response("x"))
    assert "line one\nline two" in captured["prompt"]


def test_only_judge_result_shows_on_pass():
    parsed = parse_expectations({"must_include": ["hi"]})
    ctx = _ctx(_text_response("hi"))
    for result in resolve_all(parsed, ctx):
        if result.key != "expected_output_text":
            assert result.show_on_pass is False


# --- expected_state: assertion, guard, threading ----------------------------


def test_expected_state_declared_match():
    results = _resolve(
        {"expected_state": "input-required"},
        _state_response("TASK_STATE_INPUT_REQUIRED"),
    )
    assert results["expected_state"].passed


def test_expected_state_declared_mismatch():
    results = _resolve(
        {"expected_state": "completed"}, _state_response("TASK_STATE_INPUT_REQUIRED")
    )
    assert not results["expected_state"].passed


def test_injected_guard_fails_on_undeclared_terminal_failure():
    results = _resolve(None, _state_response("TASK_STATE_FAILED"))
    assert not results["expected_state"].passed


def test_injected_guard_passes_on_completed():
    results = _resolve(None, _state_response("TASK_STATE_COMPLETED"))
    assert results["expected_state"].passed


def test_continues_task_true_only_for_input_required():
    assert ExpectedState.parse("input-required").continues_task() is True
    assert ExpectedState.parse("completed").continues_task() is False
    # The injected default never threads.
    assert ExpectedState.parse(None).continues_task() is False


def test_non_state_expectations_never_continue_task():
    assert Judge.parse("ref").continues_task() is False


# --- trace: span matchers ----------------------------------------------------


def _span(kind: str, name: str = "", span_id: str = "", parent_id: str | None = None, **attrs):
    """Build a minimal span dict."""
    attributes = {"openinference.span.kind": kind, **attrs}
    span: dict = {"attributes": attributes, "span_id": span_id, "name": name}
    if parent_id is not None:
        span["parent_span_id"] = parent_id
    return span


def _trace_of(*spans) -> dict:
    """Build a trace response from spans."""
    return {"traces": [{"spans": list(spans)}]}


def test_trace_kind_match():
    results = _resolve(
        {"trace": [{"kind": "LLM"}]},
        _text_response("ok"),
        trace=_trace_of(_span("LLM", span_id="s1")),
    )
    assert results["trace"].passed


def test_trace_kind_no_match():
    results = _resolve(
        {"trace": [{"kind": "CHAIN"}]},
        _text_response("ok"),
        trace=_trace_of(_span("LLM", span_id="s1")),
    )
    assert not results["trace"].passed


def test_trace_name_match():
    results = _resolve(
        {"trace": [{"kind": "TOOL", "name": "complete_tool"}]},
        _text_response("ok"),
        trace=_trace_of(_span("TOOL", name="complete_tool", span_id="s1")),
    )
    assert results["trace"].passed


def test_trace_name_wrong_fails():
    results = _resolve(
        {"trace": [{"kind": "TOOL", "name": "complete_tool"}]},
        _text_response("ok"),
        trace=_trace_of(_span("TOOL", name="other_tool", span_id="s1")),
    )
    assert not results["trace"].passed


def test_trace_attributes_match():
    results = _resolve(
        {"trace": [{"kind": "LLM", "attributes": {"llm.finish_reason": "tool_calls"}}]},
        _text_response("ok"),
        trace=_trace_of(_span("LLM", span_id="s1", **{"llm.finish_reason": "tool_calls"})),
    )
    assert results["trace"].passed


def test_trace_attributes_wrong_value_fails():
    results = _resolve(
        {"trace": [{"kind": "LLM", "attributes": {"llm.finish_reason": "tool_calls"}}]},
        _text_response("ok"),
        trace=_trace_of(_span("LLM", span_id="s1", **{"llm.finish_reason": "length"})),
    )
    assert not results["trace"].passed


def test_trace_multiple_matchers_all_pass():
    trace = _trace_of(
        _span("CHAIN", span_id="root", **{"opik.metadata.final_state": "TASK_STATE_COMPLETED"}),
        _span("CHAIN", span_id="run", parent_id="root"),
        _span("LLM", span_id="llm", parent_id="run", **{"llm.finish_reason": "tool_calls"}),
        _span("TOOL", name="complete_tool", span_id="tool", parent_id="run"),
    )
    results = _resolve(
        {
            "trace": [
                {"kind": "CHAIN", "attributes": {"opik.metadata.final_state": "TASK_STATE_COMPLETED"}},
                {"kind": "LLM", "attributes": {"llm.finish_reason": "tool_calls"}},
                {"kind": "TOOL", "name": "complete_tool"},
            ]
        },
        _text_response("ok"),
        trace=trace,
    )
    assert results["trace"].passed
    assert len(results["trace"].checks) == 3


def test_trace_multiple_matchers_partial_fail():
    trace = _trace_of(_span("LLM", span_id="s1"))
    results = _resolve(
        {
            "trace": [
                {"kind": "LLM"},
                {"kind": "CHAIN"},
            ]
        },
        _text_response("ok"),
        trace=trace,
    )
    assert not results["trace"].passed
    checks = results["trace"].checks
    assert [(c.label, c.passed) for c in checks] == [("LLM", True), ("CHAIN", False)]


def test_trace_no_trace_fails_all():
    results = _resolve(
        {"trace": [{"kind": "LLM"}]},
        _text_response("ok"),
        trace=None,
    )
    assert not results["trace"].passed
    assert results["trace"].checks[0].detail == "no trace available for this step"


def test_trace_exact_count_match():
    trace = _trace_of(
        _span("CHAIN", span_id="root"),
        _span("CHAIN", span_id="run", parent_id="root"),
    )
    results = _resolve(
        {"trace": [{"kind": "CHAIN", "exact": 2}]},
        _text_response("ok"),
        trace=trace,
    )
    assert results["trace"].passed


def test_trace_exact_count_too_many_fails():
    trace = _trace_of(
        _span("CHAIN", span_id="root"),
        _span("CHAIN", span_id="run", parent_id="root"),
    )
    results = _resolve(
        {"trace": [{"kind": "CHAIN", "exact": 1}]},
        _text_response("ok"),
        trace=trace,
    )
    assert not results["trace"].passed


def test_trace_exact_count_not_met_fails():
    trace = _trace_of(_span("CHAIN", span_id="root"))
    results = _resolve(
        {"trace": [{"kind": "CHAIN", "exact": 2}]},
        _text_response("ok"),
        trace=trace,
    )
    assert not results["trace"].passed


def test_trace_parent_match():
    trace = _trace_of(
        _span("CHAIN", span_id="root"),
        _span("LLM", span_id="llm", parent_id="root"),
    )
    results = _resolve(
        {"trace": [{"kind": "LLM", "parent": {"kind": "CHAIN"}}]},
        _text_response("ok"),
        trace=trace,
    )
    assert results["trace"].passed


def test_trace_parent_no_match():
    trace = _trace_of(
        _span("TOOL", span_id="tool"),
        _span("LLM", span_id="llm", parent_id="tool"),
    )
    results = _resolve(
        {"trace": [{"kind": "LLM", "parent": {"kind": "CHAIN"}}]},
        _text_response("ok"),
        trace=trace,
    )
    assert not results["trace"].passed


def test_trace_parent_no_parent_fails():
    trace = _trace_of(_span("LLM", span_id="llm"))
    results = _resolve(
        {"trace": [{"kind": "LLM", "parent": {"kind": "CHAIN"}}]},
        _text_response("ok"),
        trace=trace,
    )
    assert not results["trace"].passed


def test_trace_nested_parent_chain():
    trace = _trace_of(
        _span("CHAIN", span_id="root"),
        _span("CHAIN", span_id="run", parent_id="root"),
        _span("LLM", span_id="llm", parent_id="run"),
    )
    results = _resolve(
        {"trace": [{"kind": "LLM", "parent": {"kind": "CHAIN", "parent": {"kind": "CHAIN"}}}]},
        _text_response("ok"),
        trace=trace,
    )
    assert results["trace"].passed


def test_trace_trace_url_in_failure_detail():
    results = _resolve(
        {"trace": [{"kind": "CHAIN"}]},
        _text_response("ok"),
        trace=_trace_of(_span("LLM", span_id="s1")),
        trace_url="https://opik.dev-weu.corti.app/default/projects/abc/logs?thread=ctx-1",
    )
    assert not results["trace"].passed
    assert "https://opik.dev-weu.corti.app" in results["trace"].checks[0].detail


def test_trace_combined_kind_name_attributes():
    trace = _trace_of(
        _span("LLM", name="gpt-5.4", span_id="s1", **{"llm.finish_reason": "tool_calls", "llm.model_name": "gpt-5.4"}),
    )
    results = _resolve(
        {
            "trace": [
                {
                    "kind": "LLM",
                    "name": "gpt-5.4",
                    "attributes": {
                        "llm.finish_reason": "tool_calls",
                        "llm.model_name": "gpt-5.4",
                    },
                }
            ]
        },
        _text_response("ok"),
        trace=trace,
    )
    assert results["trace"].passed
