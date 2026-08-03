"""Sequential cases through the Opik sink.

Covers the two seams the sink spans: ``suite_to_dataset_items`` (a Case becomes
one dataset item whose ordered Steps ride a single ``steps`` column) and
``ExpectationMetric.score`` (a pure renderer that aggregates the stored step
outcomes into one column per expectation type — it never re-evaluates). The
step loop that once lived here is the runner's now; see
``test_execute_case_threading.py``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent_evals.loader import EvaluationCase, EvaluationSuite
from agent_evals.reporting.opik import ExpectationMetric, suite_to_dataset_items


def _sequential_case(**overrides) -> EvaluationCase:
    data = {
        "name": "seq_case",
        "agent": {"name": "SeqAgent"},
        "steps": [
            {
                "name": "intake",
                "message": {"message": {"parts": [{"text": "start intake"}]}},
                "expectations": {
                    "must_include": ["received"],
                    "expected_state": "input-required",
                },
            },
            {
                "name": "follow_up",
                "message": {"message": {"parts": [{"text": "here is more info"}]}},
                "expectations": {"must_not_include": ["error"]},
                "delay_before_seconds": 1.5,
            },
        ],
    }
    data.update(overrides)
    return EvaluationCase.from_dict(data)


def _single_case() -> EvaluationCase:
    return EvaluationCase.from_dict(
        {
            "name": "single_case",
            "agent": {"name": "SingleAgent"},
            "message": {"message": {"parts": [{"text": "hi"}]}},
            "expectations": {"must_include": ["hello"]},
        }
    )


def test_mixed_suite_converts_every_case_to_steps() -> None:
    # Under the unified model every Case is steps-based, so a single-message
    # Case converts into a one-Step item — its message and expectations ride
    # that lone (unnamed) Step, never a top-level column.
    suite = EvaluationSuite(name="mixed", cases=[_single_case(), _sequential_case()])

    items = suite_to_dataset_items(suite)

    assert [item["name"] for item in items] == ["single_case", "seq_case"]
    single = items[0]
    assert "expectations" not in single
    assert "message" not in single
    (only_step,) = single["steps"]
    assert only_step["name"] is None
    assert only_step["expectations"] == {"must_include": ["hello"]}


def test_sequential_item_carries_steps_as_a_structured_list() -> None:
    items = suite_to_dataset_items(
        EvaluationSuite(name="s", cases=[_sequential_case()])
    )

    (item,) = items
    # A real list of step dicts, not a double-encoded JSON string —
    # the experiment input panel renders it as structured YAML.
    steps = item["steps"]
    assert isinstance(steps, list)
    assert json.dumps(item)  # JSON-safe as inserted into the dataset
    assert [step["name"] for step in steps] == ["intake", "follow_up"]
    assert steps[0]["expectations"] == {
        "must_include": ["received"],
        "expected_state": "input-required",
    }
    assert steps[0]["message"]["message"]["parts"] == [
        {"kind": "text", "text": "start intake"}
    ]
    assert steps[1]["delay_before_seconds"] == 1.5
    assert "delay_before_seconds" not in steps[0]
    # No top-level expectations column for a sequential case.
    assert "expectations" not in item


# --- the metric: aggregate and render only ----------------------------------


def _result(
    key: str,
    checks: list[tuple[str, bool, str]],
    *,
    show_on_pass: bool = False,
) -> dict[str, Any]:
    """A serialized ExpectationResult; checks are ``(label, passed, detail)``."""
    return {
        "key": key,
        "checks": [
            {"label": label, "passed": p, "detail": d} for label, p, d in checks
        ],
        "show_on_pass": show_on_pass,
    }


def _include(term: str, passed: bool) -> tuple[str, bool, str]:
    detail = "passed" if passed else f"missing required phrase: {term!r}"
    return (term, passed, detail)


def _step_result(
    name: str,
    *,
    success: bool = True,
    results: list[dict[str, Any]] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """A trail entry as the task serializes it — per-term results, no flat errors."""
    entry: dict[str, Any] = {
        "name": name,
        "success": success,
        "expectation_results": results or [],
        "duration_seconds": 0.1,
    }
    if error is not None:
        entry["error"] = error
    return entry


def by_name(results: list[Any]) -> dict[str, Any]:
    return {r.name: r for r in results}


def test_metric_merges_step_terms_into_per_type_columns() -> None:
    scores = by_name(
        ExpectationMetric().score(
            steps=[
                {"name": "intake", "expectations": {"must_include": ["received"]}},
                {
                    "name": "follow_up",
                    "expectations": {
                        "must_include": ["done", "thanks"],
                        "must_not_include": ["error"],
                    },
                },
            ],
            step_results=[
                _step_result(
                    "intake",
                    results=[_result("must_include", [_include("received", True)])],
                ),
                _step_result(
                    "follow_up",
                    success=False,
                    results=[
                        _result(
                            "must_include",
                            [_include("done", True), _include("thanks", False)],
                        ),
                        _result("must_not_include", [("error", True, "passed")]),
                    ],
                ),
            ],
        )
    )

    # One column per expectation *type*, averaged across ALL steps' terms.
    assert scores["must_include"].value == pytest.approx(2 / 3)
    detail = json.loads(scores["must_include"].reason)
    # Every term is keyed by its step name in the reason JSON.
    assert detail["intake"] == {"received": "passed"}
    assert detail["follow_up"]["done"] == "passed"
    assert "thanks" in detail["follow_up"]["thanks"]
    assert scores["must_not_include"].value == 1.0
    # overall is the block-weighted mean of the merged columns.
    assert scores["overall"].value == pytest.approx((2 / 3 + 1.0) / 2)
    assert "thanks" in scores["overall"].reason


def test_passing_default_state_guard_adds_no_column() -> None:
    # The task stores the always-on guard's passing result on every step; an
    # ordinary step must not sprout a spurious expected_state column from it.
    scores = by_name(
        ExpectationMetric().score(
            steps=[{"name": "one", "expectations": {"must_include": ["hi"]}}],
            step_results=[
                _step_result(
                    "one",
                    results=[
                        _result("must_include", [_include("hi", True)]),
                        _result("expected_state", [("", True, "passed")]),
                    ],
                )
            ],
        )
    )
    assert set(scores) == {"overall", "must_include"}
    assert scores["overall"].value == 1.0


def test_metric_module_has_no_evaluation_path() -> None:
    # The sink replays; it never re-evaluates. The metric is a pure renderer, so
    # the module must not even carry ``evaluate_response`` — there is no seam
    # through which a stored result could be recomputed.
    import agent_evals.reporting.opik as opik_sink

    assert not hasattr(opik_sink, "evaluate_response")


def test_metric_renders_stored_verdicts_without_recomputing() -> None:
    # The metric renders exactly the stored per-term result; the column value is
    # the stored verdict, never a fresh computation.
    scores = by_name(
        ExpectationMetric().score(
            steps=[{"name": "one", "expectations": {"must_include": ["hi"]}}],
            step_results=[
                _step_result(
                    "one", results=[_result("must_include", [_include("hi", True)])]
                )
            ],
        )
    )
    assert scores["must_include"].value == 1.0
    assert scores["overall"].value == 1.0


def test_unexecuted_steps_score_zero_with_not_executed_reason() -> None:
    # A four-step case that dies at step two must not read as a near-pass:
    # steps three and four never ran, so their terms count as failed.
    steps = [
        {"name": f"s{i}", "expectations": {"must_include": [f"term{i}"]}}
        for i in range(1, 5)
    ]
    scores = by_name(
        ExpectationMetric().score(
            steps=steps,
            step_results=[
                _step_result(
                    "s1", results=[_result("must_include", [_include("term1", True)])]
                ),
                _step_result(
                    "s2",
                    success=False,
                    results=[_result("must_include", [_include("term2", False)])],
                ),
            ],
        )
    )

    assert scores["must_include"].value == pytest.approx(1 / 4)
    detail = json.loads(scores["must_include"].reason)
    assert detail["s1"] == {"term1": "passed"}
    assert detail["s3"] == {"term3": "not executed: step s2 failed"}
    assert detail["s4"] == {"term4": "not executed: step s2 failed"}
    assert scores["overall"].value == pytest.approx(1 / 4)


def test_task_level_failure_fails_every_step_term() -> None:
    scores = by_name(
        ExpectationMetric().score(
            steps=[
                {"name": "one", "expectations": {"must_include": ["a"]}},
                {"name": "two", "expectations": {"must_not_include": ["b"]}},
            ],
            step_results=[],
            error="agent creation failed",
        )
    )

    # Matching single-turn semantics: the task error is every term's reason.
    assert scores["must_include"].value == 0.0
    assert scores["must_not_include"].value == 0.0
    assert scores["overall"].value == 0.0
    assert "task failed: agent creation failed" in scores["overall"].reason
    detail = json.loads(scores["must_include"].reason)
    assert detail["one"]["a"] == "task failed: agent creation failed"


def test_task_error_prefixes_the_overall_reason() -> None:
    # A Harness Failure is the case-level error, never a trail row: the aborted
    # step leaves no entry, so the overall reason leads with the task-failed
    # prefix and the executed step's genuine pass still counts.
    scores = by_name(
        ExpectationMetric().score(
            steps=[
                {"name": "intake", "expectations": {"must_include": ["hi"]}},
                {"name": "follow_up", "expectations": {"must_include": ["ok"]}},
            ],
            # Only intake executed; the transport failure at follow_up lifted to
            # the case-level error, so it never became a trail entry.
            step_results=[
                _step_result(
                    "intake",
                    results=[_result("must_include", [_include("hi", True)])],
                ),
            ],
            error="boom",
        )
    )

    assert scores["overall"].reason.startswith("task failed: boom")
    # intake's pass survives; only the unexecuted follow_up is placeholdered.
    assert scores["overall"].value == pytest.approx(0.5)


def test_scoring_ignores_the_trail_response_payloads() -> None:
    # The metric reads only name/success/expectation_results/error from each
    # entry, so the step trail's raw response payloads never leak into columns.
    steps = [
        {"name": "intake", "expectations": {"must_include": ["received"]}},
        {
            "name": "follow_up",
            "expectations": {
                "must_include": ["done", "thanks"],
                "expected_state": "completed",
            },
        },
    ]
    bare = [
        _step_result(
            "intake", results=[_result("must_include", [_include("received", True)])]
        ),
        _step_result(
            "follow_up",
            success=False,
            results=[
                _result(
                    "must_include",
                    [_include("done", True), _include("thanks", False)],
                ),
                _result("expected_state", [("", True, "passed")]),
            ],
        ),
    ]
    with_trail = [
        {**entry, "response": {"task": {"status": {"message": entry["name"]}}}}
        for entry in bare
    ]

    def rendered(step_results: list[dict[str, Any]]) -> list[tuple[str, float, str]]:
        return [
            (s.name, s.value, s.reason)
            for s in ExpectationMetric().score(steps=steps, step_results=step_results)
        ]

    assert rendered(with_trail) == rendered(bare)


# --- the metric: scalar checks merge across steps ---------------------------


def test_expected_state_merges_across_declaring_steps() -> None:
    # Two steps declare expected_state and one mismatches: the column reads 0.5
    # with both steps' outcomes keyed by step name. The middle step declares no
    # state and stays out of the denominator.
    scores = by_name(
        ExpectationMetric().score(
            steps=[
                {
                    "name": "intake",
                    "expectations": {"expected_state": "input-required"},
                },
                {"name": "chat", "expectations": {"must_include": ["ok"]}},
                {"name": "wrap_up", "expectations": {"expected_state": "completed"}},
            ],
            step_results=[
                _step_result(
                    "intake",
                    results=[_result("expected_state", [("", True, "passed")])],
                ),
                _step_result(
                    "chat", results=[_result("must_include", [_include("ok", True)])]
                ),
                _step_result(
                    "wrap_up",
                    success=False,
                    results=[
                        _result(
                            "expected_state",
                            [
                                (
                                    "",
                                    False,
                                    "expected status 'completed' but received 'failed'",
                                )
                            ],
                        )
                    ],
                ),
            ],
        )
    )

    assert scores["expected_state"].value == pytest.approx(0.5)
    detail = json.loads(scores["expected_state"].reason)
    assert set(detail) == {"intake", "wrap_up"}
    assert detail["intake"] == "passed"
    assert "expected status" in detail["wrap_up"]


def test_tripped_terminal_guard_surfaces_on_undeclared_state() -> None:
    # No step declares expected_state, but the terminal-state guard tripped —
    # the column still appears and lowers the score.
    scores = by_name(
        ExpectationMetric().score(
            steps=[{"name": "only", "expectations": {"must_include": ["hi"]}}],
            step_results=[
                _step_result(
                    "only",
                    success=False,
                    results=[
                        _result("must_include", [_include("hi", True)]),
                        _result(
                            "expected_state",
                            [
                                (
                                    "",
                                    False,
                                    "task ended in terminal state 'failed'; declare expected_state: failed if this outcome is intended",
                                )
                            ],
                        ),
                    ],
                )
            ],
        )
    )

    assert scores["expected_state"].value == 0.0
    detail = json.loads(scores["expected_state"].reason)
    assert "terminal state" in detail["only"]
    # The guard trip does not touch the deterministic block.
    assert scores["must_include"].value == 1.0


def test_judge_column_aggregates_declaring_steps_with_keyed_explanations() -> None:
    # The explanation arrives via the serialized step outcomes; the metric only
    # renders the stored judge verdict — it never re-runs the judge.
    scores = by_name(
        ExpectationMetric().score(
            steps=[
                {
                    "name": "one",
                    "expectations": {"expected_output_text": "greets the user"},
                },
                {
                    "name": "two",
                    "expectations": {"expected_output_text": "gives a total"},
                },
                {"name": "three", "expectations": {"must_include": ["bye"]}},
            ],
            step_results=[
                _step_result(
                    "one",
                    results=[
                        _result(
                            "expected_output_text",
                            [("", True, "The agent greeted the user warmly.")],
                            show_on_pass=True,
                        )
                    ],
                ),
                _step_result(
                    "two",
                    success=False,
                    results=[
                        _result(
                            "expected_output_text",
                            [("", False, "No total appears in the reply.")],
                            show_on_pass=True,
                        )
                    ],
                ),
            ],
        )
    )

    # Only the two declaring steps count; per-step values stay strict 0/1.
    assert scores["expected_output_text"].value == pytest.approx(0.5)
    detail = json.loads(scores["expected_output_text"].reason)
    assert detail == {
        "one": "The agent greeted the user warmly.",
        "two": "No total appears in the reply.",
    }


def test_judge_reason_falls_back_to_verdict_word_when_explanation_blank() -> None:
    scores = by_name(
        ExpectationMetric().score(
            steps=[
                {"name": "one", "expectations": {"expected_output_text": "a"}},
                {"name": "two", "expectations": {"expected_output_text": "b"}},
            ],
            step_results=[
                _step_result(
                    "one",
                    results=[
                        _result(
                            "expected_output_text", [("", True, "")], show_on_pass=True
                        )
                    ],
                ),
                _step_result(
                    "two",
                    success=False,
                    results=[
                        _result(
                            "expected_output_text",
                            [("", False, "  ")],
                            show_on_pass=True,
                        )
                    ],
                ),
            ],
        )
    )

    assert json.loads(scores["expected_output_text"].reason) == {
        "one": "PASS",
        "two": "FAIL",
    }


def test_max_duration_checked_against_each_declaring_steps_own_duration() -> None:
    scores = by_name(
        ExpectationMetric().score(
            steps=[
                {"name": "fast", "expectations": {"max_duration_seconds": 2.0}},
                {"name": "slow", "expectations": {"max_duration_seconds": 2.0}},
                {"name": "unbounded", "expectations": {"must_include": ["x"]}},
            ],
            step_results=[
                _step_result(
                    "fast",
                    results=[_result("max_duration_seconds", [("", True, "passed")])],
                ),
                _step_result(
                    "slow",
                    success=False,
                    results=[
                        _result(
                            "max_duration_seconds",
                            [
                                (
                                    "",
                                    False,
                                    "duration 3.000s exceeded max_duration_seconds=2.000s",
                                )
                            ],
                        )
                    ],
                ),
            ],
        )
    )

    assert scores["max_duration_seconds"].value == pytest.approx(0.5)
    detail = json.loads(scores["max_duration_seconds"].reason)
    assert detail["fast"] == "passed"
    assert "exceeded" in detail["slow"]
    assert set(detail) == {"fast", "slow"}


def test_unexecuted_declaring_steps_fail_their_scalar_columns() -> None:
    scores = by_name(
        ExpectationMetric().score(
            steps=[
                {"name": "s1", "expectations": {"must_include": ["term1"]}},
                {
                    "name": "s2",
                    "expectations": {
                        "expected_state": "completed",
                        "max_duration_seconds": 2.0,
                        "expected_output_text": "wraps up",
                    },
                },
            ],
            step_results=[
                _step_result(
                    "s1",
                    success=False,
                    results=[_result("must_include", [_include("term1", False)])],
                ),
            ],
        )
    )

    for name in ("expected_state", "max_duration_seconds", "expected_output_text"):
        assert scores[name].value == 0.0
        assert json.loads(scores[name].reason) == {"s2": "not executed: step s1 failed"}


def test_task_failure_fails_scalar_columns_with_task_error() -> None:
    scores = by_name(
        ExpectationMetric().score(
            steps=[
                {
                    "name": "one",
                    "expectations": {
                        "expected_state": "completed",
                        "max_duration_seconds": 2.0,
                        "expected_output_text": "answers",
                    },
                },
            ],
            step_results=[],
            error="agent creation failed",
        )
    )

    for name in ("expected_state", "max_duration_seconds", "expected_output_text"):
        assert scores[name].value == 0.0
        assert json.loads(scores[name].reason) == {
            "one": "task failed: agent creation failed"
        }
    assert scores["overall"].value == 0.0


def test_overall_weighs_scalar_columns_alongside_list_blocks() -> None:
    # One step: must_include passes, expected_state and judge fail — overall is
    # the block-weighted mean over all three columns, same formula as single-turn.
    scores = by_name(
        ExpectationMetric().score(
            steps=[
                {
                    "name": "only",
                    "expectations": {
                        "must_include": ["hi"],
                        "expected_state": "completed",
                        "expected_output_text": "greets",
                    },
                },
            ],
            step_results=[
                _step_result(
                    "only",
                    success=False,
                    results=[
                        _result("must_include", [_include("hi", True)]),
                        _result(
                            "expected_state",
                            [
                                (
                                    "",
                                    False,
                                    "expected status 'completed' but received 'failed'",
                                )
                            ],
                        ),
                        _result(
                            "expected_output_text",
                            [("", False, "Wrong outcome.")],
                            show_on_pass=True,
                        ),
                    ],
                )
            ],
        )
    )

    assert scores["must_include"].value == 1.0
    assert scores["expected_state"].value == 0.0
    assert scores["expected_output_text"].value == 0.0
    assert scores["overall"].value == pytest.approx(1 / 3)


def test_all_list_block_types_merge_across_steps() -> None:
    # Every list-valued block type merges into its own column with step-keyed
    # reasons, in canonical order.
    scores = by_name(
        ExpectationMetric().score(
            steps=[
                {
                    "name": "one",
                    "expectations": {
                        "must_include": ["a"],
                        "must_match": ["ab+c"],
                        "jsonpath": [{"path": "$.foo", "equals": 1}],
                    },
                },
                {
                    "name": "two",
                    "expectations": {
                        "must_not_include": ["bad"],
                        "must_include_json": [{"foo": 1}],
                    },
                },
            ],
            step_results=[
                _step_result(
                    "one",
                    results=[
                        _result("must_include", [_include("a", True)]),
                        _result("must_match", [("ab+c", True, "passed")]),
                        _result("jsonpath", [("$.foo", True, "passed")]),
                    ],
                ),
                _step_result(
                    "two",
                    results=[
                        _result("must_not_include", [("bad", True, "passed")]),
                        _result("must_include_json", [('{"foo": 1}', True, "passed")]),
                    ],
                ),
            ],
        )
    )

    assert set(scores) == {
        "overall",
        "must_include",
        "must_not_include",
        "must_match",
        "must_include_json",
        "jsonpath",
    }
    assert all(s.value == 1.0 for s in scores.values())
    assert json.loads(scores["jsonpath"].reason) == {"one": {"$.foo": "passed"}}
    assert json.loads(scores["must_include_json"].reason) == {
        "two": {'{"foo": 1}': "passed"}
    }


# --- the metric fix: an executed step's real verdicts survive a later error --


def test_executed_steps_stored_verdicts_survive_a_task_error() -> None:
    # A Case that passed its first step and then died on transport must keep the
    # first step's real PASS visible — the task error placeholders only the
    # steps that never executed, not the ones that did (the old bug
    # placeholdered every step, masking a genuine pass).
    scores = by_name(
        ExpectationMetric().score(
            steps=[
                {"name": "intake", "expectations": {"must_include": ["hi"]}},
                {"name": "wrap_up", "expectations": {"must_include": ["bye"]}},
            ],
            # Only the intake step executed; the runner lifted the transport
            # failure to the case-level error and never recorded a wrap_up entry.
            step_results=[
                _step_result(
                    "intake",
                    results=[_result("must_include", [_include("hi", True)])],
                ),
            ],
            error="boom",
        )
    )

    detail = json.loads(scores["must_include"].reason)
    # The executed step keeps its stored pass; only the unexecuted remainder is
    # a "task failed" placeholder.
    assert detail["intake"] == {"hi": "passed"}
    assert detail["wrap_up"] == {"bye": "task failed: boom"}
    # One of two terms genuinely passed → 0.5, not a full 0.0 wipe.
    assert scores["must_include"].value == pytest.approx(0.5)


# --- list-block partial credit and bounding ---------------------------------


def test_duplicate_terms_keep_the_column_within_bounds() -> None:
    # Duplicate terms collapse to one reason key, but the denominator stays the
    # declared term count — the value must never exceed 1.0.
    scores = by_name(
        ExpectationMetric().score(
            steps=[{"name": "one", "expectations": {"must_include": ["ok", "ok"]}}],
            step_results=[
                _step_result(
                    "one",
                    results=[
                        _result(
                            "must_include",
                            [_include("ok", True), _include("ok", True)],
                        )
                    ],
                )
            ],
        )
    )
    assert scores["must_include"].value == 1.0
    assert json.loads(scores["must_include"].reason) == {"one": {"ok": "passed"}}


def test_must_include_json_partial_credit() -> None:
    # One of two JSON blocks matches → 0.5, with per-block detail in the reason.
    scores = by_name(
        ExpectationMetric().score(
            steps=[
                {
                    "name": "one",
                    "expectations": {"must_include_json": [{"foo": 1}, {"bar": 2}]},
                }
            ],
            step_results=[
                _step_result(
                    "one",
                    success=False,
                    results=[
                        _result(
                            "must_include_json",
                            [
                                ('{"foo": 1}', True, "passed"),
                                ('{"bar": 2}', False, "no JSON block matched"),
                            ],
                        )
                    ],
                )
            ],
        )
    )
    assert scores["must_include_json"].value == pytest.approx(0.5)
    detail = json.loads(scores["must_include_json"].reason)["one"]
    assert detail['{"foo": 1}'] == "passed"
    assert detail['{"bar": 2}'] != "passed"


def test_jsonpath_column_fails_as_zero() -> None:
    scores = by_name(
        ExpectationMetric().score(
            steps=[
                {
                    "name": "one",
                    "expectations": {"jsonpath": [{"path": "$.foo", "equals": 1}]},
                }
            ],
            step_results=[
                _step_result(
                    "one",
                    success=False,
                    results=[
                        _result(
                            "jsonpath",
                            [("$.foo", False, "no data part carried $.foo")],
                        )
                    ],
                )
            ],
        )
    )
    assert scores["jsonpath"].value == 0.0
    assert scores["overall"].value == 0.0
