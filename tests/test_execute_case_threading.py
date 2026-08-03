"""Threading through the execute loop: continuation and the missing-task guard.

``expected_state: input-required`` both threads the task into the next step and,
when the continuation arrives without a task id, folds a failure into that step's
``expected_state`` result. Asserted at the runner seam via ``execute_case`` with
a scripted client — threading is a property of the loop, not of a single
expectation, so it is tested where the loop runs.
"""

from __future__ import annotations

from typing import Any

from agent_evals.errors import ErrorCode, NetworkError
from agent_evals.expectations import parse_expectations
from agent_evals.loader import EvaluationCase, Step
from agent_evals.provisioning import AgentPool
from agent_evals.runner import execute_case
from agent_evals.schemas.agent import Agent
from agent_evals.schemas.message import MessagePayload


class _ScriptedClient:
    """Returns a canned response per step and records every request it sent."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = responses
        self.sent: list[dict[str, Any]] = []

    def create_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"id": "agent-1"}

    def send_message(self, agent_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.sent.append(payload)
        return self._responses[len(self.sent) - 1]

    def clone(self) -> "_ScriptedClient":
        return self

    def close(self) -> None:
        pass


def _step(name: str, expectations: dict[str, Any] | None = None) -> Step:
    return Step(
        message=MessagePayload.from_dict(
            {"message": {"parts": [{"kind": "text", "text": name}]}}
        ),
        expectations=parse_expectations(expectations),
        name=name,
    )


def _run(
    steps: list[Step], responses: list[dict[str, Any]]
) -> tuple[Any, _ScriptedClient]:
    case = EvaluationCase(name="flow", agent=Agent(name="A"), steps=steps)
    client = _ScriptedClient(responses)
    pool = AgentPool()
    pool.provision([case], client)
    return execute_case(case, client, pool), client


def test_input_required_threads_task_id_into_next_step() -> None:
    result, client = _run(
        steps=[_step("ask", {"expected_state": "input-required"}), _step("answer")],
        responses=[
            {
                "task": {
                    "id": "task-xyz",
                    "status": {"state": "TASK_STATE_INPUT_REQUIRED"},
                }
            },
            {"task": {"status": {"state": "completed"}}},
        ],
    )

    assert result.success
    # The continuation carries the first step's task id forward.
    assert client.sent[1]["message"]["taskId"] == "task-xyz"


def test_non_continuing_step_does_not_thread_a_task_id() -> None:
    # A completed (non-continuing) first step resets threading — no isinstance,
    # just ``continues_task()`` returning False — so the next send is fresh.
    _, client = _run(
        steps=[_step("first"), _step("second")],
        responses=[
            {"task": {"id": "task-abc", "status": {"state": "completed"}}},
            {"task": {"status": {"state": "completed"}}},
        ],
    )
    assert "taskId" not in client.sent[1]["message"]


def test_missing_task_id_folds_into_expected_state_result() -> None:
    result, _ = _run(
        steps=[_step("ask", {"expected_state": "input-required"}), _step("answer")],
        responses=[
            # input-required matches, but no task id comes back to continue with.
            {"task": {"status": {"state": "TASK_STATE_INPUT_REQUIRED"}}},
            {"task": {"status": {"state": "completed"}}},
        ],
    )

    # The loop stops at the failed step; the second step never runs.
    assert not result.success
    assert len(result.step_results) == 1
    ask = result.step_results[0]
    assert not ask.success

    # The miss is visible in the per-term view, folded onto expected_state — the
    # declared state check still passed, an extra check records the missing id.
    state_result = next(r for r in ask.results if r.key == "expected_state")
    assert not state_result.passed
    assert [c.passed for c in state_result.checks] == [True, False]
    assert "task id" in state_result.checks[-1].detail

    # The folded check is the miss's only record — no Harness Failure rides along.
    assert ask.harness_error is None


def test_check_failure_is_only_recorded_as_its_failed_check() -> None:
    result, _ = _run(
        steps=[_step("greet", {"must_include": ["apology"]}), _step("later")],
        responses=[
            {"task": {"status": {"state": "completed"}}},
            {"task": {"status": {"state": "completed"}}},
        ],
    )

    # The trail stops at the failing step; the second step never runs.
    assert not result.success
    assert len(result.step_results) == 1
    greet = result.step_results[0]
    assert not greet.success
    # The failed check is the miss's only record — no derived error object.
    assert greet.harness_error is None

    failed = [c for r in greet.results for c in r.checks if not c.passed]
    assert [c.label for c in failed] == ["apology"]


class _RaisingClient:
    """Fails every send with a typed transport exception."""

    def create_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"id": "agent-1"}

    def send_message(self, agent_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise NetworkError("connection reset by peer")

    def clone(self) -> "_RaisingClient":
        return self

    def close(self) -> None:
        pass


def test_transport_exception_aborts_with_typed_harness_failure() -> None:
    case = EvaluationCase(name="flow", agent=Agent(name="A"), steps=[_step("greet")])
    client = _RaisingClient()
    pool = AgentPool()
    pool.provision([case], client)

    result = execute_case(case, client, pool)

    assert not result.success
    (aborted,) = result.step_results
    # The aborted entry is unnamed and request-less: the failure struck the
    # send, so there is nothing evaluated to record — no verdicts, no response.
    # Only a Harness Failure like this one rides the typed slot.
    assert aborted.name is None
    assert aborted.request is None
    assert aborted.response is None
    assert aborted.results == []
    assert aborted.harness_error is not None
    assert aborted.harness_error.code is ErrorCode.NETWORK_ERROR
