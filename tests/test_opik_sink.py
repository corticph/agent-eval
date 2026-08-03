"""Tests for the Opik sink at the mocked ``evaluate()`` boundary.

The sink never sends a message — the runner does. These isolate the sink's
contract with the Opik SDK (as ``test_client.py`` isolates the HTTP session):
``on_start`` verifies connectivity and the dataset shell before any send;
``write`` accumulates executed Cases; ``close`` rebuilds the dataset from
*exactly* those Cases and replays their stored results through ``evaluate()``.
The captured replay task, fed a dataset item, returns the stored output — a
no-network lookup, never a re-execution.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_evals.environment import Environment
from agent_evals.loader import EvaluationCase, EvaluationSuite
from agent_evals.reporting.opik import OpikSink
from agent_evals.results import EvaluationResult, StepResult, UsageMetrics

from conftest import OPIK_HOSTS

_RESOLVED_URL = "http://localhost:18080"


@pytest.fixture(autouse=True)
def _pristine_opik_env():
    """Isolate the env vars ``on_start`` reads *and writes*.

    ``monkeypatch`` alone cannot restore a variable the code under test sets
    after the fixture ran, so snapshot/restore by hand.
    """
    names = ("OPIK_URL_OVERRIDE", "OPIK_API_KEY", "OPIK_PROJECT_NAME")
    saved = {name: os.environ.pop(name) for name in names if name in os.environ}
    yield
    for name in names:
        os.environ.pop(name, None)
    os.environ.update(saved)


def _case(name: str) -> EvaluationCase:
    return EvaluationCase.from_dict(
        {
            "name": name,
            "agent": {"name": "Agent"},
            "message": {"message": {"parts": [{"text": "hi"}]}},
            "expectations": {"must_include": ["ok"]},
        }
    )


def _passing_result(name: str) -> EvaluationResult:
    """A single-step success carrying usage and a conversation id."""
    step = StepResult(
        name="only",
        success=True,
        request=None,
        response=None,
        duration_seconds=1.0,
        context_id="ctx-1",
        usage=UsageMetrics(input_tokens=100, output_tokens=25, credits=0.1),
        results=[],
    )
    return EvaluationResult(name=name, success=True, agent_id="a1", step_results=[step])


class _FakeDataset:
    def __init__(self, project_name: str = "Agents") -> None:
        self.project_name = project_name
        self.inserted: list[dict[str, Any]] = []
        self.cleared = 0

    def clear(self) -> None:
        self.cleared += 1

    def insert(self, items: list[dict[str, Any]]) -> None:
        self.inserted.extend(items)


def _mock_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict,
    *,
    dataset: _FakeDataset | None = None,
) -> MagicMock:
    """Substitute everything below the sink except the assertion target."""
    monkeypatch.setattr(
        "agent_evals.reporting.opik.resolve_opik_url", lambda: _RESOLVED_URL
    )

    the_dataset = dataset or _FakeDataset()
    fake_opik = MagicMock()
    fake_opik.get_or_create_dataset.side_effect = lambda name, project_name=None: (
        the_dataset
    )
    captured["opik_client"] = fake_opik
    captured["dataset"] = the_dataset

    def fake_opik_ctor(**kwargs):
        captured["opik_kwargs"] = kwargs
        return fake_opik

    monkeypatch.setattr("agent_evals.reporting.opik.Opik", fake_opik_ctor)

    def fake_evaluate(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("agent_evals.reporting.opik.evaluate", fake_evaluate)
    return fake_opik


def _drive(
    monkeypatch: pytest.MonkeyPatch,
    *,
    suite_name: str = "mysuite",
    entries: list[tuple[EvaluationCase, EvaluationResult]] | None = None,
    environment: str = "eu",
    dataset: _FakeDataset | None = None,
    **sink_kwargs: Any,
) -> dict:
    """Run a full on_start → write* → close cycle and return the captured calls."""
    captured: dict = {}
    _mock_boundaries(monkeypatch, captured, dataset=dataset)
    entries = (
        entries
        if entries is not None
        else [(_case("case-1"), _passing_result("case-1"))]
    )
    suite = EvaluationSuite(name=suite_name, cases=[case for case, _ in entries])

    sink = OpikSink(Environment(environment), **sink_kwargs)
    sink.on_start(suite)
    for case, result in entries:
        sink.write(case, result)
    sink.close()
    return captured


# --- on_start: fail fast before any send -------------------------------------


def test_on_start_propagates_url_resolution_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A run whose experiment cannot even be reached must abort in on_start —
    # the runner starts every sink before the first message is sent, so the
    # raised error aborts the run before any Case executes.
    captured: dict = {}
    _mock_boundaries(monkeypatch, captured)
    monkeypatch.setattr(
        "agent_evals.reporting.opik.resolve_opik_url",
        MagicMock(side_effect=RuntimeError("tunnel down")),
    )

    sink = OpikSink(Environment("dev-weu"))
    with pytest.raises(RuntimeError, match="tunnel down"):
        sink.on_start(EvaluationSuite(name="s", cases=[]))

    # Nothing was created and no replay was attempted.
    assert "evaluate" not in captured
    assert "opik_kwargs" not in captured


def test_close_without_on_start_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    # If an earlier sink's on_start aborted the run, this sink's close (from the
    # runner's finally) must not touch Opik.
    captured: dict = {}
    _mock_boundaries(monkeypatch, captured)

    OpikSink(Environment("eu")).close()

    assert "evaluate" not in captured


# --- on_start: client + dataset provisioning ---------------------------------


def test_client_targets_resolved_url_with_default_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _drive(monkeypatch)

    # Explicit host + workspace beat ``~/.opik.config`` resolution, which points
    # every developer at their personal (usually local) Opik.
    assert captured["opik_kwargs"]["host"] == _RESOLVED_URL
    assert captured["opik_kwargs"]["workspace"] == "default"


def test_resolved_url_is_exported_for_the_tracing_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _drive(monkeypatch)

    # ``evaluate()`` routes its per-case wrapper trace through the SDK's global
    # client, which reads env config, not our instance; the env export keeps
    # those wrappers and the experiment writes in the same Opik.
    assert os.environ["OPIK_URL_OVERRIDE"] == _RESOLVED_URL


def test_pre_existing_override_is_not_clobbered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    os.environ["OPIK_URL_OVERRIDE"] = "http://pre-set:5173/api"
    _drive(monkeypatch)
    assert os.environ["OPIK_URL_OVERRIDE"] == "http://pre-set:5173/api"


def test_dataset_defaults_to_the_agents_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _drive(monkeypatch)

    # The dataset's project drives where evaluate() puts everything; it must be
    # the project agent-api traces to, not the SDK's "Default Project".
    kwargs = captured["opik_client"].get_or_create_dataset.call_args.kwargs
    assert kwargs["project_name"] == "Agents"
    assert os.environ["OPIK_PROJECT_NAME"] == "Agents"
    # A dataset already in the right project is used as-is.
    captured["opik_client"].delete_dataset.assert_not_called()


def test_explicit_project_name_wins_over_default_and_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    os.environ["OPIK_PROJECT_NAME"] = "pre-set"
    captured = _drive(monkeypatch, project_name="my-sandbox")

    kwargs = captured["opik_client"].get_or_create_dataset.call_args.kwargs
    assert kwargs["project_name"] == "my-sandbox"
    assert os.environ["OPIK_PROJECT_NAME"] == "my-sandbox"


def test_project_name_env_var_overrides_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    os.environ["OPIK_PROJECT_NAME"] = "pre-set"
    captured = _drive(monkeypatch)

    kwargs = captured["opik_client"].get_or_create_dataset.call_args.kwargs
    assert kwargs["project_name"] == "pre-set"
    assert os.environ["OPIK_PROJECT_NAME"] == "pre-set"


def test_dataset_pinned_to_the_wrong_project_is_recreated_there(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}
    client = _mock_boundaries(monkeypatch, captured)
    # A dataset from before the Agents default: the SDK resolves its *stored*
    # project and discards the requested one, so the run would land in the wrong
    # project unless recreated.
    stale = _FakeDataset(project_name="Default Project")
    fresh = _FakeDataset(project_name="Agents")
    client.get_or_create_dataset.side_effect = lambda name, project_name=None: stale
    client.create_dataset.return_value = fresh

    sink = OpikSink(Environment("dev-weu"))
    sink.on_start(EvaluationSuite(name="mysuite", cases=[]))
    sink.write(_case("case-1"), _passing_result("case-1"))
    sink.close()

    client.delete_dataset.assert_called_once_with(name="mysuite")
    client.create_dataset.assert_called_once_with(name="mysuite", project_name="Agents")
    # The recreated dataset is the one populated and evaluated — not the stale one.
    assert fresh.cleared == 1
    assert fresh.inserted
    assert stale.cleared == 0
    assert captured["dataset"] is fresh


def test_agent_auth_token_is_not_reused_as_opik_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _drive(monkeypatch)
    # The environment's bearer token never granted access — the raw backend
    # behind the tunnel takes no auth at all.
    assert captured["opik_kwargs"].get("api_key") is None


def test_explicit_opik_api_key_env_passes_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    os.environ["OPIK_API_KEY"] = "user-supplied-key"
    captured = _drive(monkeypatch)
    assert captured["opik_kwargs"]["api_key"] == "user-supplied-key"


# --- experiment identity -----------------------------------------------------


def test_environment_reaches_experiment_config(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _drive(monkeypatch, environment="eu")
    # The run's target env lands on its experiment, keyed alongside the suite
    # name — the sole discriminator between same-suite runs.
    assert captured["experiment_config"] == {"suite": "mysuite", "environment": "eu"}


def test_experiment_and_dataset_default_to_the_suite_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _drive(monkeypatch, suite_name="mysuite")
    assert captured["experiment_name"] == "mysuite"
    kwargs = captured["opik_client"].get_or_create_dataset.call_args.kwargs
    assert kwargs["name"] == "mysuite"


def test_name_overrides_are_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _drive(
        monkeypatch,
        suite_name="mysuite",
        dataset_name="my-dataset",
        experiment_name="my-experiment",
    )
    assert captured["experiment_name"] == "my-experiment"
    kwargs = captured["opik_client"].get_or_create_dataset.call_args.kwargs
    assert kwargs["name"] == "my-dataset"


def test_experiment_tags_default_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _drive(monkeypatch)
    assert captured["experiment_tags"] is None


def test_experiment_tags_are_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _drive(monkeypatch, experiment_tags=["baseline", "gpt-4o"])
    assert captured["experiment_tags"] == ["baseline", "gpt-4o"]


# --- close: the dataset mirrors exactly the executed Cases -------------------


def test_close_inserts_exactly_the_executed_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A stopped or aborted run writes only the Cases that executed; the dataset
    # must mirror those alone, never render an unexecuted Case as a failure.
    executed = [
        (_case("case-1"), _passing_result("case-1")),
        (_case("case-3"), _passing_result("case-3")),
    ]
    captured = _drive(monkeypatch, entries=executed)

    dataset = captured["dataset"]
    assert dataset.cleared == 1  # rebuilt, not upserted
    assert [item["name"] for item in dataset.inserted] == ["case-1", "case-3"]


def test_close_with_no_executed_cases_clears_but_does_not_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A run that executed nothing still clears stale items (rebuild every run),
    # but must not replay: ``evaluate()`` on an empty dataset only errors, and
    # no experiment should record a run with zero rows.
    captured = _drive(monkeypatch, entries=[])
    dataset = captured["dataset"]
    assert dataset.cleared == 1
    assert dataset.inserted == []
    assert "task" not in captured  # evaluate() never called


# --- the replay task: a no-network lookup of the stored result ----------------


def _replay(captured: dict, item_name: str) -> dict[str, Any]:
    """Feed the captured replay task a dataset item, as ``evaluate()`` would."""
    return captured["task"]({"name": item_name})


def test_replay_task_returns_the_stored_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _drive(monkeypatch, environment="eu")
    output = _replay(captured, "case-1")

    # The executed Step rides the trail; usage sums across steps; the trace URL
    # derives from the last executed Step's context id; the environment is the
    # run's target — the same derivations the file Sink uses.
    assert [s["name"] for s in output["step_results"]] == ["only"]
    assert output["usage"] == {"input_tokens": 100, "output_tokens": 25, "credits": 0.1}
    assert output["environment"] == "eu"
    assert output["trace_url"].endswith("ctx-1")
    assert output["trace_url"].startswith(f"{OPIK_HOSTS['eu']}/")
    assert "error" not in output


def test_replay_trace_url_ignores_the_internally_exported_tunnel_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # on_start exports the resolved tunnel URL for the SDK's global client, but
    # that localhost address is not somewhere a browser can see traces — the
    # replay's trace links must still point at the environment's real Opik.
    captured = _drive(monkeypatch, environment="eu")
    output = _replay(captured, "case-1")

    assert os.environ["OPIK_URL_OVERRIDE"] == _RESOLVED_URL  # tunnel exported
    assert output["trace_url"].startswith(f"{OPIK_HOSTS['eu']}/")


def test_replay_lifts_a_harness_failure_to_the_case_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A timeout/transport failure is a verdict-less synthetic Step: it must lift
    # into the case-level ``error`` field, never a name-less trail row.
    from agent_evals.errors import ErrorCode, EvalError

    executed_step = StepResult(
        name="intake",
        success=True,
        request=None,
        response=None,
        duration_seconds=1.0,
        context_id="ctx-1",
        usage=None,
        results=[],
    )
    harness_step = StepResult(
        name=None,
        success=False,
        request=None,
        response=None,
        harness_error=EvalError(
            ErrorCode.EVAL_TIMEOUT, "eval exceeded wall-clock timeout"
        ),
    )
    result = EvaluationResult(
        name="case-1",
        success=False,
        agent_id="a1",
        step_results=[executed_step, harness_step],
    )
    captured = _drive(monkeypatch, entries=[(_case("case-1"), result)])
    output = _replay(captured, "case-1")

    # The trail carries the executed Step only; the harness failure is the error.
    assert [s["name"] for s in output["step_results"]] == ["intake"]
    assert "timeout" in output["error"]
    # The thread the executed Step opened still links.
    assert output["trace_url"].endswith("ctx-1")


# --- the sink logs no traces of its own --------------------------------------


def test_module_does_no_span_tracking() -> None:
    # Regression guard for the no-own-traces rule: the tracing API must not even
    # be imported — agent-api owns the traces for every interaction, and the
    # only wrapper traces are the ones ``evaluate()`` creates itself.
    import agent_evals.reporting.opik as opik_sink

    assert not hasattr(opik_sink, "track")
    assert not hasattr(opik_sink, "opik_context")


def test_reporting_package_does_not_eagerly_import_opik() -> None:
    # ADR 0009: a bare ``run`` must not import the optional ``opik`` package.
    # Importing the reporting package (as ``__main__`` does for FileSink) must
    # pull in the file sink and the trace helper only — the Opik sink is served
    # lazily. A fresh interpreter, so sys.modules is not polluted by this suite.
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import agent_evals.reporting, sys; "
            "assert 'opik' not in sys.modules, 'opik imported eagerly'",
        ],
        check=True,
    )
