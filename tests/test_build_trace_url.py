"""Tests for the trace-link seam: the shared ``build_trace_url`` helper and
the rendering rules in results output (remote link shape, local omission)."""

from __future__ import annotations

import json

from agent_evals.loader import EvaluationCase, Step
from agent_evals.reporting import FileSink, build_trace_url
from agent_evals.results import EvaluationResult, StepResult
from agent_evals.schemas.agent import Agent
from agent_evals.schemas.message import MessagePayload
from agent_evals.schemas.response import Response

BASE = "https://opik.example/projects/proj-123/logs?thread="


def test_context_id_present_returns_exact_concatenation() -> None:
    assert build_trace_url(BASE, "ctx-abc") == f"{BASE}ctx-abc"


def test_context_id_present_composes_full_thread_url() -> None:
    assert build_trace_url(BASE, "ctx-abc") == (
        "https://opik.example/projects/proj-123/logs?thread=ctx-abc"
    )


def test_context_id_none_returns_none() -> None:
    assert build_trace_url(BASE, None) is None


def test_context_id_empty_string_returns_none() -> None:
    assert build_trace_url(BASE, "") is None


def test_context_id_whitespace_returns_none() -> None:
    assert build_trace_url(BASE, "   ") is None


def test_no_base_url_returns_none() -> None:
    # A None base means the environment has no trace destination (local);
    # results must omit links rather than render broken ones.
    assert build_trace_url(None, "ctx-abc") is None


def _write_results(tmp_path, trace_base_url):
    """Write one single-turn case and one sequential case through the writer."""
    single_case = EvaluationCase(
        name="single",
        agent=Agent(name="a"),
        steps=[Step(message=MessagePayload())],
    )
    # A single-message case is one synthesized (unnamed) Step; its thread is the
    # case-level trace, so the case row links to it directly. The context id is
    # always stamped by execution — the writer never re-extracts it.
    single_result = EvaluationResult(
        name="single",
        success=True,
        agent_id="agent-1",
        step_results=[
            StepResult(
                name=None,
                success=True,
                request=MessagePayload(),
                response=Response.from_dict({"contextId": "ctx-case"}),
                context_id="ctx-case",
            )
        ],
    )
    seq_case = EvaluationCase(
        name="sequential",
        agent=Agent(name="a"),
        steps=[Step(message=MessagePayload())],
    )
    step = StepResult(
        name="step-1",
        success=True,
        request=MessagePayload(),
        response=None,
        context_id="ctx-step",
    )
    seq_result = EvaluationResult(
        name="sequential", success=True, agent_id="agent-2", step_results=[step]
    )

    out_path = tmp_path / "out.json"
    writer = FileSink(out_path, trace_base_url=trace_base_url)
    writer.write(single_case, single_result)
    writer.write(seq_case, seq_result)
    writer.close()

    entries = {
        entry["name"]: entry
        for entry in json.loads(out_path.read_text(encoding="utf-8"))
    }
    markdown = (tmp_path / "out.md").read_text(encoding="utf-8")
    return entries, markdown


def test_remote_run_links_case_and_step_traces(tmp_path) -> None:
    entries, markdown = _write_results(tmp_path, BASE)

    assert entries["single"]["trace_url"] == f"{BASE}ctx-case"
    assert entries["sequential:step-1"]["trace_url"] == f"{BASE}ctx-step"
    assert f"- **Trace:** [ctx-case]({BASE}ctx-case)" in markdown
    assert f"- **Trace:** [ctx-step]({BASE}ctx-step)" in markdown


def test_no_trace_id_key_survives_anywhere_in_output(tmp_path) -> None:
    # There is no trace id: the identifier's key is context_id, everywhere.
    def keys_of(payload):
        if isinstance(payload, dict):
            for key, value in payload.items():
                yield key
                yield from keys_of(value)
        elif isinstance(payload, list):
            for item in payload:
                yield from keys_of(item)

    entries, _ = _write_results(tmp_path, BASE)

    assert "trace_id" not in set(keys_of(list(entries.values())))
    assert "context_id" in set(keys_of(list(entries.values())))


def test_local_run_omits_trace_lines_entirely(tmp_path) -> None:
    # A None base is how the local environment arrives at the writer: no
    # dead links in the markdown and no trace_url keys in the JSON.
    entries, markdown = _write_results(tmp_path, None)

    assert all("trace_url" not in entry for entry in entries.values())
    assert "**Trace:**" not in markdown


def test_override_redirects_result_file_links(tmp_path, monkeypatch) -> None:
    # End-to-end across the seam: a base derived under OPIK_URL_OVERRIDE
    # lands in the results file, so links point where traces actually went.
    from agent_evals.environment import Environment

    monkeypatch.setenv("OPIK_URL_OVERRIDE", "http://localhost:5173")
    entries, markdown = _write_results(tmp_path, Environment("dev-weu").trace_base_url)

    assert entries["single"]["trace_url"].startswith(
        "http://localhost:5173/default/projects/"
    )
    assert "- **Trace:** [ctx-case](http://localhost:5173/" in markdown


def test_external_environment_without_project_id_omits_links(
    tmp_path, monkeypatch
) -> None:
    # An external checkout sets no OPIK_PROJECT_ID_*: the environment yields a
    # None trace base, so result files carry no trace links — the same path
    # local takes, no external-only branch.
    from agent_evals.environment import Environment

    monkeypatch.delenv("OPIK_PROJECT_ID_EU", raising=False)
    entries, markdown = _write_results(tmp_path, Environment("eu").trace_base_url)

    assert all("trace_url" not in entry for entry in entries.values())
    assert "**Trace:**" not in markdown
