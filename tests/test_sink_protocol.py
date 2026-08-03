"""The Sink protocol at the ``run_suite`` seam.

A recording fake proves the runner drives every Sink through the four hooks
— ``on_start(suite)``, one ``write(case, result)`` per executed Case in the
order results are collected, ``close()``, and ``aggregate_runs()`` after every
run in a multi-run loop — and that every started Sink is closed no matter how
the run ends, so a crashed or stopped run still flushes the partial results
written so far.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_evals.expectations import ExpectedState
from agent_evals.loader import (
    EvaluationCase,
    EvaluationSuite,
    Step,
    SuiteOptions,
)
from agent_evals.results import EvaluationResult
from agent_evals.runner import run_suite, run_suite_multiple
from agent_evals.schemas.agent import Agent
from agent_evals.schemas.message import MessagePayload


class _RecordingSink:
    """Records every hook invocation as an event tuple, in order."""

    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []

    def on_start(self, suite: EvaluationSuite) -> None:
        self.events.append(("on_start", suite.name))

    def write(self, case: EvaluationCase, result: EvaluationResult) -> None:
        self.events.append(("write", case.name, result.success))

    def close(self) -> None:
        self.events.append(("close",))

    def aggregate_runs(self) -> None:
        self.events.append(("aggregate_runs",))


class _FakeClient:
    """Stand-in for AgentClient keyed on the message text.

    ``"fail"`` produces a task in state ``failed``; ``"boom"`` raises
    ``KeyboardInterrupt`` — the one crash that escapes ``execute_case``'s
    defensive catch, standing in for a user stopping the run.
    """

    def create_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"id": "agent-1"}

    def send_message(self, agent_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        text = payload["message"]["parts"][0]["text"]
        if text == "boom":
            raise KeyboardInterrupt
        state = "failed" if text == "fail" else "completed"
        return {"task": {"status": {"state": state}}}

    def clone(self) -> "_FakeClient":
        return self

    def close(self) -> None:
        pass


def _case(name: str) -> EvaluationCase:
    return EvaluationCase(
        name=name,
        agent=Agent(name="SinkAgent"),
        steps=[
            Step(
                message=MessagePayload.from_dict(
                    {"message": {"parts": [{"kind": "text", "text": name}]}}
                ),
                expectations=[ExpectedState.parse(None)],
            )
        ],
    )


def _suite(cases: list[EvaluationCase], **options: Any) -> EvaluationSuite:
    return EvaluationSuite(
        name="sink_suite",
        cases=cases,
        options=SuiteOptions(concurrency=1, **options),
    )


def test_hooks_fire_in_order() -> None:
    sink = _RecordingSink()
    suite = _suite([_case("first"), _case("second")])

    results = run_suite(suite, _FakeClient(), sinks=[sink])

    assert [r.success for r in results] == [True, True]
    assert sink.events == [
        ("on_start", "sink_suite"),
        ("write", "first", True),
        ("write", "second", True),
        ("close",),
    ]


def test_every_sink_is_driven() -> None:
    sinks = [_RecordingSink(), _RecordingSink()]
    run_suite(_suite([_case("only")]), _FakeClient(), sinks=sinks)

    for sink in sinks:
        assert sink.events == [
            ("on_start", "sink_suite"),
            ("write", "only", True),
            ("close",),
        ]


def test_no_sinks_still_returns_results() -> None:
    results = run_suite(_suite([_case("bare")]), _FakeClient())
    assert [r.success for r in results] == [True]


def test_close_runs_when_a_case_crashes() -> None:
    sink = _RecordingSink()
    suite = _suite([_case("first"), _case("boom")])

    with pytest.raises(KeyboardInterrupt):
        run_suite(suite, _FakeClient(), sinks=[sink])

    assert sink.events == [
        ("on_start", "sink_suite"),
        ("write", "first", True),
        ("close",),
    ]


def test_provisioning_failure_closes_started_sinks() -> None:
    class _BrokenProvisionClient(_FakeClient):
        def create_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("agent spec rejected")

    sink = _RecordingSink()
    with pytest.raises(RuntimeError):
        run_suite(_suite([_case("never-runs")]), _BrokenProvisionClient(), sinks=[sink])

    assert sink.events == [("on_start", "sink_suite"), ("close",)]


def test_failing_on_start_closes_only_already_started_sinks() -> None:
    class _BrokenSink(_RecordingSink):
        def on_start(self, suite: EvaluationSuite) -> None:
            raise RuntimeError("cannot start")

    healthy, broken = _RecordingSink(), _BrokenSink()
    with pytest.raises(RuntimeError):
        run_suite(_suite([_case("never-runs")]), _FakeClient(), sinks=[healthy, broken])

    assert healthy.events == [("on_start", "sink_suite"), ("close",)]
    assert broken.events == []


def test_stop_on_failure_flushes_partial_results() -> None:
    sink = _RecordingSink()
    suite = _suite(
        [_case("first"), _case("fail"), _case("never-runs")],
        stop_on_failure=True,
    )

    results = run_suite(suite, _FakeClient(), sinks=[sink])

    assert [r.name for r in results] == ["first", "fail"]
    assert sink.events == [
        ("on_start", "sink_suite"),
        ("write", "first", True),
        ("write", "fail", False),
        ("close",),
    ]


def test_concurrent_run_drives_all_hooks() -> None:
    sink = _RecordingSink()
    suite = EvaluationSuite(
        name="sink_suite",
        cases=[_case("a"), _case("b"), _case("c")],
        options=SuiteOptions(concurrency=3),
    )

    run_suite(suite, _FakeClient(), sinks=[sink])

    assert sink.events[0] == ("on_start", "sink_suite")
    assert sink.events[-1] == ("close",)
    assert sorted(sink.events[1:-1]) == [
        ("write", "a", True),
        ("write", "b", True),
        ("write", "c", True),
    ]


# ---------------------------------------------------------------------------
# Multi-run: run_suite_multiple drives aggregate_runs on the last run's sinks
# ---------------------------------------------------------------------------


def test_run_suite_multiple_calls_aggregate_runs_on_last_sinks() -> None:
    sink = _RecordingSink()
    suite = _suite([_case("only")])

    all_results = run_suite_multiple(
        suite,
        _FakeClient(),
        runs=2,
        sinks_factory=lambda _: [sink],
    )

    assert sink.events.count(("aggregate_runs",)) == 1
    assert sink.events[-1] == ("aggregate_runs",)
    assert len(all_results) == 2
    assert len(all_results[0]) == 1
    assert len(all_results[1]) == 1


def test_run_suite_multiple_returns_per_run_lists() -> None:
    sink = _RecordingSink()
    suite = _suite([_case("a"), _case("b")])

    all_results = run_suite_multiple(
        suite,
        _FakeClient(),
        runs=3,
        sinks_factory=lambda _: [sink],
    )

    assert len(all_results) == 3
    for run_results in all_results:
        assert [r.name for r in run_results] == ["a", "b"]


def test_run_suite_multiple_drives_hooks_per_run() -> None:
    sink = _RecordingSink()
    suite = _suite([_case("only")])

    run_suite_multiple(
        suite,
        _FakeClient(),
        runs=2,
        sinks_factory=lambda _: [sink],
    )

    on_start_count = sink.events.count(("on_start", "sink_suite"))
    close_count = sink.events.count(("close",))
    aggregate_count = sink.events.count(("aggregate_runs",))
    assert on_start_count == 2
    assert close_count == 2
    assert aggregate_count == 1
