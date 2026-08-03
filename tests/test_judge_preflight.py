"""Judge preflight through the ``run_suite`` seam.

A suite that will need judge verdicts but has no ``JUDGE_API_KEY`` must abort
at run start — before any agent is provisioned or message sent — instead of
burning a full run to discover the misconfiguration. Everything is asserted
from the outside: what ``run_suite`` raises and what the fake client saw
(prior art: ``tests/test_agent_provisioning.py``).
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_evals.expectations import parse_expectations
from agent_evals.expectations.llm.base import Judge
from agent_evals.loader import EvaluationCase, EvaluationSuite, Step, SuiteOptions
from agent_evals.runner import run_suite
from agent_evals.schemas.agent import Agent
from agent_evals.schemas.message import MessagePayload


class _CountingClient:
    """Counts create-agent and send-message calls; every response completes."""

    def __init__(self) -> None:
        self.create_calls = 0
        self.send_calls = 0

    def create_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.create_calls += 1
        return {"id": f"agent-{self.create_calls}"}

    def send_message(self, agent_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.send_calls += 1
        return {"task": {"status": {"state": "completed"}}}

    def clone(self) -> "_CountingClient":
        return self

    def close(self) -> None:
        pass


def _suite(expectations_block: dict[str, Any] | None) -> EvaluationSuite:
    message = MessagePayload.from_dict(
        {"message": {"parts": [{"kind": "text", "text": "hi"}]}}
    )
    case = EvaluationCase(
        name="case",
        agent=Agent(name="Agent"),
        steps=[
            Step(message=message, expectations=parse_expectations(expectations_block))
        ],
    )
    return EvaluationSuite(name="preflight_suite", cases=[case], options=SuiteOptions())


@pytest.fixture
def no_judge_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear every credential the judge can authenticate with."""
    for var in (
        "JUDGE_API_KEY",
        "CORTI_TENANT_NAME",
        "CORTI_CLIENT_ID",
        "CORTI_CLIENT_SECRET",
    ):
        monkeypatch.delenv(var, raising=False)


def test_judge_expectation_without_key_aborts_before_any_provision_or_send(
    no_judge_key: None,
) -> None:
    client = _CountingClient()
    suite = _suite({"expected_output_text": "the answer is 4"})

    with pytest.raises(RuntimeError, match="JUDGE_API_KEY"):
        run_suite(suite, client)

    assert client.create_calls == 0
    assert client.send_calls == 0


def test_empty_key_counts_as_missing(
    no_judge_key: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JUDGE_API_KEY", "")
    client = _CountingClient()
    suite = _suite({"expected_output_text": "the answer is 4"})

    with pytest.raises(RuntimeError, match="JUDGE_API_KEY"):
        run_suite(suite, client)

    assert client.create_calls == 0
    assert client.send_calls == 0


def test_corti_console_credentials_satisfy_the_preflight(
    no_judge_key: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Console's CORTI_* pair alone is a valid judge credential — the
    harness assembles the bearer, so the run starts without JUDGE_API_KEY."""
    monkeypatch.setenv("CORTI_CLIENT_ID", "corti-acme-default_client")
    monkeypatch.setenv("CORTI_CLIENT_SECRET", "s3cret")
    # Preflight is the subject; stub the verdict call so no request leaves.
    monkeypatch.setattr(Judge, "_judge", staticmethod(lambda prompt: (True, "")))
    client = _CountingClient()
    suite = _suite({"expected_output_text": "the answer is 4"})

    results = run_suite(suite, client)

    assert client.send_calls == 1
    assert len(results) == 1


def test_error_names_the_variables_and_why_they_are_required(
    no_judge_key: None,
) -> None:
    suite = _suite({"expected_output_text": "the answer is 4"})

    with pytest.raises(RuntimeError) as excinfo:
        run_suite(suite, _CountingClient())

    message = str(excinfo.value)
    assert "expected_output_text" in message  # why: the suite declares judge checks
    # What to set: either credential route must be named.
    assert "CORTI_CLIENT_ID" in message
    assert "CORTI_CLIENT_SECRET" in message
    assert "JUDGE_API_KEY" in message


def test_suite_without_judge_expectations_runs_without_the_key(
    no_judge_key: None,
) -> None:
    client = _CountingClient()
    suite = _suite({"must_include": ["hi"]})

    results = run_suite(suite, client)

    assert client.send_calls == 1
    assert len(results) == 1


def test_judge_expectation_with_empty_reference_needs_no_key(
    no_judge_key: None,
) -> None:
    client = _CountingClient()
    suite = _suite({"expected_output_text": ""})

    results = run_suite(suite, client)

    assert client.send_calls == 1
    assert results[0].success


# --- faithful_to_source inherits the same preflight --------------------------


def test_faithful_to_source_without_key_aborts_before_any_provision_or_send(
    no_judge_key: None,
) -> None:
    client = _CountingClient()
    suite = _suite({"faithful_to_source": True})

    with pytest.raises(RuntimeError, match="JUDGE_API_KEY"):
        run_suite(suite, client)

    assert client.create_calls == 0
    assert client.send_calls == 0


def test_faithful_to_source_error_names_the_expectation(
    no_judge_key: None,
) -> None:
    suite = _suite({"faithful_to_source": True})

    with pytest.raises(RuntimeError) as excinfo:
        run_suite(suite, _CountingClient())

    message = str(excinfo.value)
    assert "faithful_to_source" in message
    assert "CORTI_CLIENT_ID" in message
    assert "CORTI_CLIENT_SECRET" in message
    assert "JUDGE_API_KEY" in message
