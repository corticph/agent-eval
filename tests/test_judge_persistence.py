"""Tests for persisting LLM-judge verdicts in results output.

The judge is not a distinct type: it is one ``show_on_pass`` ``ExpectationResult``
whose lone check detail is the explanation. These tests pin that it rides a
Step's per-term results (not a standalone slot) and renders on a pass as well as
a fail.
"""

from __future__ import annotations

import json

from agent_evals.expectations import (
    CheckResult,
    ExpectationResult,
    evaluate_response,
    parse_expectations,
)
from agent_evals.expectations.llm.base import Judge
from agent_evals.loader import EvaluationCase, Step
from agent_evals.reporting import FileSink
from agent_evals.results import EvaluationResult, StepResult
from agent_evals.schemas.agent import Agent
from agent_evals.schemas.message import MessagePayload


def _judge_result(passed: bool, explanation: str) -> ExpectationResult:
    """The judge as it now rides a Step: one ``show_on_pass`` expectation result."""
    return ExpectationResult(
        "expected_output_text",
        [CheckResult("", passed, explanation)],
        show_on_pass=True,
    )


_RESPONSE = {
    "task": {
        "status": {
            "message": {
                "parts": [
                    {
                        "kind": "text",
                        "text": "The tallest building is the Burj Khalifa.",
                    }
                ]
            }
        }
    }
}

_JUDGE_BLOCK = {"expected_output_text": "Burj Khalifa is the tallest building."}


def _judge(results: list[ExpectationResult]) -> ExpectationResult:
    return next(r for r in results if r.key == "expected_output_text")


def test_judge_pass_verdict_is_captured(monkeypatch):
    monkeypatch.setattr(
        Judge,
        "_judge",
        staticmethod(lambda prompt: (True, "Matches the reference answer.")),
    )
    results = evaluate_response(parse_expectations(_JUDGE_BLOCK), _RESPONSE)
    assert all(r.passed for r in results)
    judge = _judge(results)
    # The explanation is kept even on a pass (show_on_pass), so an author sees
    # *why* the output was accepted.
    assert judge.show_on_pass
    assert judge.checks[0].detail == "Matches the reference answer."


def test_judge_fail_verdict_is_captured(monkeypatch):
    monkeypatch.setattr(
        Judge,
        "_judge",
        staticmethod(lambda prompt: (False, "Missing key facts about the building.")),
    )
    results = evaluate_response(parse_expectations(_JUDGE_BLOCK), _RESPONSE)
    judge = _judge(results)
    assert not judge.passed
    assert judge.checks[0].detail == "Missing key facts about the building."


def test_no_judge_means_no_judge_result():
    results = evaluate_response(
        parse_expectations({"must_include": ["Burj"]}), _RESPONSE
    )
    assert all(r.passed for r in results)
    assert not any(r.key == "expected_output_text" for r in results)


def test_as_dict_carries_per_term_results_not_a_judge_slot():
    step_with = StepResult(
        name="s",
        success=True,
        request=MessagePayload(),
        response=None,
        results=[_judge_result(True, "Matches the reference answer.")],
    )
    as_dict = step_with.as_dict()
    assert as_dict["expectation_results"] == [
        {
            "key": "expected_output_text",
            "checks": [
                {"label": "", "passed": True, "detail": "Matches the reference answer."}
            ],
            "show_on_pass": True,
        }
    ]
    # The standalone judge slot is gone — the judge is one of the per-term results.
    assert "judge_assessment" not in as_dict

    # A Step that resolved no expectations serializes an empty per-term list.
    step_without = StepResult(
        name="s", success=True, request=MessagePayload(), response=None
    )
    assert step_without.as_dict()["expectation_results"] == []

    # The case-level result carries no duplicated judge column — the verdict is
    # read only from its Steps.
    result = EvaluationResult(
        name="e", success=True, agent_id=None, step_results=[step_with]
    )
    assert "judge_assessment" not in result.as_dict()


def test_result_writer_persists_per_term_verdicts_in_json_and_markdown(tmp_path):
    single_case = EvaluationCase(
        name="single",
        agent=Agent(name="a"),
        steps=[Step(message=MessagePayload())],
    )
    # A single-message case's verdicts ride its lone synthesized (unnamed) Step.
    single_result = EvaluationResult(
        name="single",
        success=True,
        agent_id="agent-1",
        step_results=[
            StepResult(
                name=None,
                success=True,
                request=MessagePayload(),
                response=None,
                results=[_judge_result(True, "Matches the reference answer.")],
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
        success=False,
        request=MessagePayload(),
        response=None,
        results=[_judge_result(False, "Wrong answer provided.")],
    )
    seq_result = EvaluationResult(
        name="sequential", success=False, agent_id="agent-2", step_results=[step]
    )

    out_path = tmp_path / "out.json"
    writer = FileSink(out_path)
    writer.write(single_case, single_result)
    writer.write(seq_case, seq_result)
    writer.close()

    entries = {
        entry["name"]: entry
        for entry in json.loads(out_path.read_text(encoding="utf-8"))
    }
    # The single case's per-term results are nested in its Step Trail, not a
    # case-level column.
    assert entries["single"]["step_results"][0]["expectation_results"] == [
        {
            "key": "expected_output_text",
            "checks": [
                {"label": "", "passed": True, "detail": "Matches the reference answer."}
            ],
            "show_on_pass": True,
        }
    ]
    assert "judge_assessment" not in entries["single"]
    assert (
        entries["sequential:step-1"]["expectation_results"][0]["checks"][0]["detail"]
        == "Wrong answer provided."
    )
    assert "judge_assessment" not in entries["sequential:step-1"]

    markdown = (tmp_path / "out.md").read_text(encoding="utf-8")
    # The judge block renders on a pass (with its explanation) as well as a fail.
    assert "- **Judge Verdict:** ✅ PASS" in markdown
    assert "> Matches the reference answer." in markdown
    assert "- **Judge Verdict:** ❌ FAIL" in markdown
    assert "> Wrong answer provided." in markdown
