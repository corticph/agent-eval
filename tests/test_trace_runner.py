"""Trace expectations at the runner seam: fetch gating, threaded ids, semantics.

The runner's contract around ``trace:`` lives here rather than in expectation
unit tests: the fetch only runs for steps that declare a trace expectation,
it keys on the threaded context id rather than the per-response candidate, and
counts are evaluated against the whole-context trace, which accumulates across
steps of a sequential case.
"""

from __future__ import annotations

from typing import Any

from agent_evals.expectations import parse_expectations
from agent_evals.loader import EvaluationCase, Step
from agent_evals.provisioning import AgentPool
from agent_evals.runner import execute_case
from agent_evals.schemas.agent import Agent
from agent_evals.schemas.message import MessagePayload


class _Env:
    trace_base_url = None


class _TraceScriptedClient:
    """Canned responses plus a controllable, observable trace endpoint.

    ``traces_per_step`` holds the cumulative per-context trace after each step;
    ``trace_calls`` records the context ids the fetch keyed on. No
    ``environment``-free non-trace step ever touches either: the gate is what
    makes that absence safe.
    """

    def __init__(
        self,
        responses: list[dict[str, Any]],
        traces_per_step: list[dict[str, Any]] | None = None,
    ) -> None:
        self._responses = responses
        self._traces = traces_per_step or []
        self.sent: list[dict[str, Any]] = []
        self.trace_calls: list[str] = []
        self.environment = _Env()

    def create_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"id": "agent-1"}

    def send_message(self, agent_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.sent.append(payload)
        return self._responses[len(self.sent) - 1]

    def get_trace(self, context_id: str) -> dict[str, Any]:
        self.trace_calls.append(context_id)
        # Per-step, not per-call: the two stabilization fetches of one step
        # must see the same trace, keyed by which message went out before it.
        step_index = len(self.sent) - 1
        return self._traces[min(step_index, len(self._traces) - 1)]

    def clone(self) -> "_TraceScriptedClient":
        return self

    def close(self) -> None:
        pass


class _NoTraceClient:
    """No environment, no get_trace — an AttributeError waiting to happen if
    the runner touches trace machinery for a step that never asked for it."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = responses
        self.sent: list[dict[str, Any]] = []

    def create_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"id": "agent-1"}

    def send_message(self, agent_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.sent.append(payload)
        return self._responses[len(self.sent) - 1]

    def clone(self) -> "_NoTraceClient":
        return self

    def close(self) -> None:
        pass


def _llm_span(span_id: str) -> dict[str, Any]:
    return {"span_id": span_id, "name": "llm", "attributes": {"openinference.span.kind": "LLM"}}


def _step(name: str, expectations: dict[str, Any] | None = None) -> Step:
    return Step(
        message=MessagePayload.from_dict(
            {"message": {"parts": [{"kind": "text", "text": name}]}}
        ),
        expectations=parse_expectations(expectations),
        name=name,
    )


def _run(steps: list[Step], client: Any) -> Any:
    case = EvaluationCase(name="trace-flow", agent=Agent(name="A"), steps=steps)
    pool = AgentPool()
    pool.provision([case], client)
    return execute_case(case, client, pool)


def test_no_trace_expectation_never_fetches() -> None:
    client = _NoTraceClient([{"task": {"status": {"state": "completed"}}}])
    result = _run([_step("plain")], client)
    assert result.success


def test_counts_accumulate_across_steps_of_a_shared_context() -> None:
    # F6 contract pin: matchers see the whole-context trace AFTER each step,
    # so step 2's ``kind: LLM`` with default ``exact: 1`` fails once both
    # steps have contributed one LLM span each.
    client = _TraceScriptedClient(
        responses=[
            {"contextId": "ctx-1", "task": {"id": "t-1",
                                            "status": {"state": "TASK_STATE_INPUT_REQUIRED"}}},
            {"task": {"status": {"state": "completed"}}},
        ],
        traces_per_step=[
            {"traces": [{"spans": [_llm_span("s1")]}]},
            {"traces": [{"spans": [_llm_span("s1"), _llm_span("s2")]}]},
        ],
    )
    result = _run(
        [
            _step("ask", {"expected_state": "input-required", "trace": [{"kind": "LLM"}]}),
            _step("answer", {"trace": [{"kind": "LLM"}]}),
        ],
        client,
    )

    step_results = result.step_results
    assert step_results[0].success
    assert not step_results[1].success
    trace_result = [r for r in step_results[1].results if r.key == "trace"][0]
    assert trace_result.checks[0].detail.endswith("found 2")


def test_fetch_uses_threaded_context_when_response_omits_id() -> None:
    # F2 regression: step 2's response carries no contextId, but its request
    # went out under ctx-1 — the trace fetch must key on ctx-1, not on None.
    client = _TraceScriptedClient(
        responses=[
            {"contextId": "ctx-1", "task": {"id": "t-1",
                                            "status": {"state": "TASK_STATE_INPUT_REQUIRED"}}},
            {"task": {"status": {"state": "completed"}}},
        ],
        traces_per_step=[
            {"traces": [{"spans": [_llm_span("s1")]}]},
            {"traces": [{"spans": [_llm_span("s1"), _llm_span("s2")]}]},
        ],
    )
    result = _run(
        [
            _step("ask", {"expected_state": "input-required"}),
            _step("answer", {"trace": [{"kind": "LLM", "exact": 2}]}),
        ],
        client,
    )

    assert result.success
    assert set(client.trace_calls) == {"ctx-1"}
