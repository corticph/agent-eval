"""Tests for the per-eval wall-clock timeout (timeout_seconds)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from agent_evals.client import MAX_HTTP_TIMEOUT_SECONDS
from agent_evals.errors import ErrorCode
from agent_evals.loader import (
    EvaluationCase,
    EvaluationSuite,
    Step,
    SuiteOptions,
    load_suite,
)
from agent_evals.runner import _effective_timeout, run_suite
from agent_evals.schemas.agent import Agent
from agent_evals.schemas.message import MessagePayload


def _write(tmp_path: Path, rel: str, content: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


SUITE_HEADER = """\
name: timeout_suite
globals:
  agent:
    name: TimeoutAgent
    description: agent for timeout tests
"""


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def test_per_eval_timeout_parsed_on_single_case(tmp_path: Path) -> None:
    suite_path = _write(
        tmp_path,
        "suite.yaml",
        SUITE_HEADER
        + """\
evals:
  - name: case1
    timeout_seconds: 300
    message:
      message:
        parts:
          - kind: text
            text: hello
""",
    )
    suite = load_suite(suite_path)
    assert suite.cases[0].timeout_seconds == 300.0


def test_per_eval_timeout_parsed_on_sequential_case(tmp_path: Path) -> None:
    suite_path = _write(
        tmp_path,
        "suite.yaml",
        SUITE_HEADER
        + """\
evals:
  - name: case1
    type: sequential
    timeout_seconds: 152.5
    steps:
      - name: step1
        message:
          message:
            parts:
              - kind: text
                text: hello
""",
    )
    suite = load_suite(suite_path)
    assert suite.cases[0].timeout_seconds == 152.5


def test_globals_timeout_used_as_fallback(tmp_path: Path) -> None:
    suite_path = _write(
        tmp_path,
        "suite.yaml",
        SUITE_HEADER
        + """\
  timeout_seconds: 600
evals:
  - name: uses_global
    message:
      message:
        parts:
          - kind: text
            text: hello
  - name: overrides_global
    timeout_seconds: 300
    message:
      message:
        parts:
          - kind: text
            text: hello
""",
    )
    suite = load_suite(suite_path)
    assert suite.options.timeout_seconds == 600.0
    assert suite.cases[0].timeout_seconds is None
    assert _effective_timeout(suite.cases[0], suite.options) == 600.0
    assert _effective_timeout(suite.cases[1], suite.options) == 300.0


def test_folder_defaults_timeout_picked_up(tmp_path: Path) -> None:
    _write(tmp_path, "pyproject.toml", "")  # mark tmp_path as the project root
    _write(
        tmp_path,
        "_defaults.yaml",
        "globals:\n  timeout_seconds: 450\n",
    )
    suite_path = _write(
        tmp_path,
        "problem/suite.yaml",
        SUITE_HEADER
        + """\
evals:
  - name: case1
    message:
      message:
        parts:
          - kind: text
            text: hello
""",
    )
    suite = load_suite(suite_path)
    assert suite.options.timeout_seconds == 450.0


def test_per_eval_timeout_must_exceed_http_timeout(tmp_path: Path) -> None:
    suite_path = _write(
        tmp_path,
        "suite.yaml",
        SUITE_HEADER
        + f"""\
evals:
  - name: case1
    timeout_seconds: {MAX_HTTP_TIMEOUT_SECONDS}
    message:
      message:
        parts:
          - kind: text
            text: hello
""",
    )
    with pytest.raises(
        ValueError,
        match="timeout_seconds .* must be greater than the HTTP request timeout",
    ):
        load_suite(suite_path)


def test_sequential_timeout_must_exceed_http_timeout(tmp_path: Path) -> None:
    suite_path = _write(
        tmp_path,
        "suite.yaml",
        SUITE_HEADER
        + """\
evals:
  - name: case1
    type: sequential
    timeout_seconds: 30
    steps:
      - name: step1
        message:
          message:
            parts:
              - kind: text
                text: hello
""",
    )
    with pytest.raises(
        ValueError,
        match="timeout_seconds .* must be greater than the HTTP request timeout",
    ):
        load_suite(suite_path)


def test_globals_timeout_must_exceed_http_timeout(tmp_path: Path) -> None:
    suite_path = _write(
        tmp_path,
        "suite.yaml",
        SUITE_HEADER
        + """\
  timeout_seconds: 60
evals:
  - name: case1
    message:
      message:
        parts:
          - kind: text
            text: hello
""",
    )
    with pytest.raises(
        ValueError,
        match="timeout_seconds .* must be greater than the HTTP request timeout",
    ):
        load_suite(suite_path)


# ---------------------------------------------------------------------------
# Runtime enforcement
# ---------------------------------------------------------------------------


class _FakeClient:
    """Stand-in for AgentClient; messages whose text is "slow" sleep for `delay`."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay

    def create_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"id": "agent-1"}

    def send_message(self, agent_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        text = payload["message"]["parts"][0]["text"]
        if text == "slow":
            time.sleep(self.delay)
        return {"task": {"status": {"state": "completed"}}}

    def clone(self) -> "_FakeClient":
        return _FakeClient(delay=self.delay)

    def close(self) -> None:
        pass


def _case(name: str, timeout_seconds: float | None = None) -> EvaluationCase:
    return EvaluationCase(
        name=name,
        agent=Agent(name="TimeoutAgent"),
        steps=[
            Step(
                message=MessagePayload.from_dict(
                    {"message": {"parts": [{"kind": "text", "text": name}]}}
                )
            )
        ],
        timeout_seconds=timeout_seconds,
    )


def _suite(cases: list[EvaluationCase], concurrency: int) -> EvaluationSuite:
    return EvaluationSuite(
        name="timeout_suite",
        cases=cases,
        options=SuiteOptions(concurrency=concurrency),
    )


def _assert_timeout_then_success(results: list[Any]) -> None:
    assert len(results) == 2
    assert results[0].success is False
    # The timeout has no Step of its own: it rides a synthesized Step Trail
    # entry — a Harness Failure beside empty verdicts, never mixed into them.
    (timeout_step,) = results[0].step_results
    assert timeout_step.harness_error is not None
    assert timeout_step.harness_error.code is ErrorCode.EVAL_TIMEOUT
    assert timeout_step.results == []
    assert results[0].duration_seconds == 0.05
    assert results[1].success is True


def test_timeout_marks_failure_and_suite_continues_sequential() -> None:
    slow = _case("slow", timeout_seconds=0.05)
    fast = _case("fast")
    results = run_suite(_suite([slow, fast], concurrency=1), _FakeClient(delay=0.5))
    _assert_timeout_then_success(results)


def test_timeout_marks_failure_and_suite_continues_concurrent() -> None:
    slow = _case("slow", timeout_seconds=0.05)
    fast = _case("fast")
    results = run_suite(_suite([slow, fast], concurrency=2), _FakeClient(delay=0.5))
    _assert_timeout_then_success(results)


def test_timeout_not_hit_leaves_result_untouched() -> None:
    case = _case("quick", timeout_seconds=5)
    results = run_suite(_suite([case], concurrency=1), _FakeClient(delay=0))
    assert len(results) == 1
    assert results[0].success is True
    assert results[0].step_results[0].harness_error is None
