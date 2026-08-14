"""Eager agent pre-provisioning through the ``run_suite`` seam.

Every behaviour is asserted from the outside: how many times the client's
create-agent operation runs, in what order relative to message sends, and which
agent ids messages land on. Nothing here reaches into the ``AgentPool``'s
internal map or key format (prior art: ``tests/test_eval_timeout.py`` and the
raising ``create_agent`` in ``tests/test_opik_sequential.py``).
"""

from __future__ import annotations

import threading
from copy import deepcopy
from typing import Any

import pytest

from agent_evals.loader import EvaluationCase, EvaluationSuite, Step, SuiteOptions
from agent_evals.runner import run_suite
from agent_evals.schemas.agent import Agent
from agent_evals.schemas.message import MessagePayload


class _RecordingClient:
    """Records create-agent and send-message calls in one ordered event log.

    Clones share the same recorder, so a suite that fans out onto per-case
    client clones still records every call in one place. ``create_agent`` hands
    back a fresh incrementing id so distinct specs get distinguishable agents.
    """

    def __init__(self, recorder: "_Recorder | None" = None) -> None:
        self._recorder = recorder or _Recorder()

    def create_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        agent_id = self._recorder.record_create(payload)
        response: dict[str, Any] = {"id": agent_id}
        connectors = payload.get("connectors") or []
        if connectors:
            response["connectors"] = [
                {**c, "id": self._recorder.next_connector_id()} for c in connectors
            ]
        return response

    def send_message(self, agent_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._recorder.record_send(agent_id)
        return {"task": {"status": {"state": "completed"}}}

    def clone(self) -> "_RecordingClient":
        return _RecordingClient(self._recorder)

    def close(self) -> None:
        pass

    # --- observation helpers (read the shared recorder) --------------------
    @property
    def create_count(self) -> int:
        return self._recorder.create_count

    @property
    def sent_to(self) -> list[str]:
        return self._recorder.sent_to

    @property
    def all_creates_precede_all_sends(self) -> bool:
        return self._recorder.all_creates_precede_all_sends


class _Recorder:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[tuple[str, str]] = []  # ("create", id) | ("send", agent_id)
        self._counter = 0
        self._con_counter = 0
        self.created_payloads: list[dict[str, Any]] = []

    def record_create(self, payload: dict[str, Any]) -> str:
        with self._lock:
            self._counter += 1
            agent_id = f"agent-{self._counter}"
            self._events.append(("create", agent_id))
            # a real client would serialize; the copy catches later mutation
            self.created_payloads.append(deepcopy(payload))
            return agent_id

    def next_connector_id(self) -> str:
        with self._lock:
            self._con_counter += 1
            return f"con-{self._con_counter}"

    def record_send(self, agent_id: str) -> None:
        with self._lock:
            self._events.append(("send", agent_id))

    @property
    def create_count(self) -> int:
        return sum(1 for kind, _ in self._events if kind == "create")

    @property
    def sent_to(self) -> list[str]:
        return [ident for kind, ident in self._events if kind == "send"]

    @property
    def all_creates_precede_all_sends(self) -> bool:
        last_create = max(
            (i for i, (kind, _) in enumerate(self._events) if kind == "create"),
            default=-1,
        )
        first_send = next(
            (i for i, (kind, _) in enumerate(self._events) if kind == "send"),
            len(self._events),
        )
        return last_create < first_send


class _RaisingClient(_RecordingClient):
    """Fails the second create-agent call — a broken spec mid-provisioning."""

    def __init__(self, recorder: "_Recorder | None" = None, fail_on: int = 2) -> None:
        super().__init__(recorder)
        self._fail_on = fail_on

    def create_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._recorder.create_count + 1 == self._fail_on:
            raise RuntimeError("agent creation failed")
        return super().create_agent(payload)


def _agent(name: str = "Agent", **extra: Any) -> Agent:
    return Agent(name=name, **extra)


def _message(text: str) -> MessagePayload:
    return MessagePayload.from_dict(
        {"message": {"parts": [{"kind": "text", "text": text}]}}
    )


def _case(name: str, agent: Agent | None = None, **overrides: Any) -> EvaluationCase:
    return EvaluationCase(
        name=name,
        agent=agent if agent is not None else _agent(),
        steps=[Step(message=_message(name))],
        **overrides,
    )


def _suite(cases: list[Any], concurrency: int = 1) -> EvaluationSuite:
    return EvaluationSuite(
        name="provisioning_suite",
        cases=cases,
        options=SuiteOptions(concurrency=concurrency),
    )


# ---------------------------------------------------------------------------
# Dedup by spec
# ---------------------------------------------------------------------------


def test_shared_spec_creates_one_agent_and_every_case_messages_it() -> None:
    client = _RecordingClient()
    cases = [_case(f"c{i}") for i in range(4)]

    results = run_suite(_suite(cases), client)

    assert client.create_count == 1
    assert client.sent_to == ["agent-1"] * 4
    assert all(r.agent_id == "agent-1" for r in results)


def test_two_distinct_specs_each_get_their_own_agent() -> None:
    client = _RecordingClient()
    cases = [
        _case("a", _agent("Alpha")),
        _case("b", _agent("Beta")),
    ]

    run_suite(_suite(cases), client)

    assert client.create_count == 2
    assert sorted(client.sent_to) == ["agent-1", "agent-2"]


def test_different_targeted_connectors_split_but_same_connector_collapses() -> None:
    spec = _agent(
        "Orchestrator",
        connectors=[
            {"type": "registry", "name": "research"},
            {"type": "registry", "name": "writing"},
        ],
    )
    client = _RecordingClient()
    cases = [
        _case("research_1", deepcopy(spec), use_connector_name="research"),
        _case("research_2", deepcopy(spec), use_connector_name="research"),
        _case("writing_1", deepcopy(spec), use_connector_name="writing"),
    ]

    run_suite(_suite(cases), client)

    # research (shared once) + writing = two agents, not three.
    assert client.create_count == 2
    assert client.sent_to[0] == client.sent_to[1]  # both research cases
    assert client.sent_to[2] != client.sent_to[0]  # writing is its own agent


def test_orchestrator_and_isolated_connector_from_one_spec_are_distinct() -> None:
    spec = _agent("Orchestrator", connectors=[{"type": "registry", "name": "research"}])
    client = _RecordingClient()
    cases = [
        _case("full", deepcopy(spec)),
        _case("isolated", deepcopy(spec), use_connector_name="research"),
    ]

    run_suite(_suite(cases), client)

    assert client.create_count == 2
    assert client.sent_to[0] != client.sent_to[1]


def test_inline_agent_connector_is_created_first_and_referenced_by_id() -> None:
    spec = _agent(
        "Orchestrator",
        connectors=[
            {"type": "registry", "name": "research"},
            {
                "type": "agent",
                "name": "BMIExpert",
                "systemPrompt": "You compute BMI.",
                "connectors": [
                    {"type": "mcp", "name": "calc", "url": "http://calc/mcp"}
                ],
            },
        ],
    )
    client = _RecordingClient()

    results = run_suite(_suite([_case("inline", spec)]), client)

    # The sub-agent is created from its inline definition, then the
    # orchestrator's payload references it by the fresh id — nothing inline
    # survives on the wire.
    assert client.create_count == 2
    sub_payload, orchestrator_payload = client._recorder.created_payloads
    assert sub_payload["name"] == "BMIExpert"
    assert sub_payload["connectors"] == [
        {"type": "mcp", "name": "calc", "url": "http://calc/mcp"}
    ]
    assert orchestrator_payload["connectors"] == [
        {"type": "registry", "name": "research"},
        {"type": "agent", "agentId": "agent-1"},
    ]
    assert results[0].agent_id == "agent-2"
    assert client.sent_to == ["agent-2"]


# ---------------------------------------------------------------------------
# Overrides and ordering
# ---------------------------------------------------------------------------


def test_override_cases_create_nothing_and_message_the_override_id() -> None:
    client = _RecordingClient()
    cases = [
        _case("byo_a", agent_id_override="pinned-1"),
        _case("byo_b", agent_id_override="pinned-2"),
    ]

    results = run_suite(_suite(cases), client)

    assert client.create_count == 0
    assert client.sent_to == ["pinned-1", "pinned-2"]
    assert [r.agent_id for r in results] == ["pinned-1", "pinned-2"]


def test_all_creations_happen_before_any_message_is_sent() -> None:
    client = _RecordingClient()
    cases = [
        _case("a", _agent("Alpha")),
        _case("b", _agent("Beta")),
        _case("c", _agent("Gamma")),
    ]

    run_suite(_suite(cases), client)

    assert client.create_count == 3
    assert client.all_creates_precede_all_sends


# ---------------------------------------------------------------------------
# Fail-fast
# ---------------------------------------------------------------------------


def test_provisioning_failure_aborts_the_run_before_any_message() -> None:
    client = _RaisingClient(fail_on=2)
    cases = [
        _case("a", _agent("Alpha")),
        _case("b", _agent("Beta")),  # its create raises
        _case("c", _agent("Gamma")),
    ]

    with pytest.raises(RuntimeError, match="agent creation failed"):
        run_suite(_suite(cases), client)

    assert client.sent_to == []


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_dedup_holds_under_concurrency() -> None:
    client = _RecordingClient()
    cases = [_case(f"c{i}") for i in range(8)]

    run_suite(_suite(cases, concurrency=4), client)

    assert client.create_count == 1
    assert client.sent_to == ["agent-1"] * 8
    assert client.all_creates_precede_all_sends


# ---------------------------------------------------------------------------
# Multi-turn reuse
# ---------------------------------------------------------------------------


def test_sequential_case_reuses_one_agent_across_all_turns() -> None:
    client = _RecordingClient()
    case = EvaluationCase(
        name="multi",
        agent=_agent("SeqAgent"),
        steps=[
            Step(name="one", message=_message("first")),
            Step(name="two", message=_message("second")),
            Step(name="three", message=_message("third")),
        ],
    )

    results = run_suite(_suite([case]), client)

    assert client.create_count == 1
    assert client.sent_to == ["agent-1", "agent-1", "agent-1"]
    assert results[0].agent_id == "agent-1"
