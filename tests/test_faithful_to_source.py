"""Tests for the faithful_to_source expectation.

The faithfulness judge is mocked to a fixed verdict — we assert the expectation
extracts the sources (at source_path) and the summary, wires them into the
prompt, and that a FAIL surfaces as a failed check. The judge call itself
(Judge._judge) is tested in test_judge.py.
"""

from __future__ import annotations

from agent_evals.expectations import parse_expectations
from agent_evals.expectations.base import EvaluationContext
from agent_evals.expectations.llm.faithful import FaithfulToSource


def _response(summary: str, source_payload) -> dict:
    return {
        "task": {
            "artifacts": [
                {
                    "parts": [
                        {"text": summary},
                        {"data": {"response": source_payload}},
                    ]
                }
            ]
        }
    }


def _eval(expectations_dict: dict, response: dict):
    expectations = parse_expectations(expectations_dict)
    ctx = EvaluationContext.from_response(response)
    return [e.resolve(ctx) for e in expectations]


def _install_judge(monkeypatch, verdict: tuple[bool, str]):
    captured: dict = {}

    def _fake(prompt):
        captured["prompt"] = prompt
        return verdict

    monkeypatch.setattr(FaithfulToSource, "_judge", staticmethod(_fake))
    return captured


def test_pass_verdict_passes_and_wires_summary_and_sources(monkeypatch):
    captured = _install_judge(monkeypatch, (True, "PASS, grounded"))
    results = _eval(
        {"faithful_to_source": True},
        _response("Metformin is first-line.", "sources about metformin"),
    )
    assert all(r.passed for r in results), results
    assert "Metformin is first-line." in captured["prompt"]
    assert "sources about metformin" in captured["prompt"]


def test_fail_verdict_surfaces_as_failed_check(monkeypatch):
    _install_judge(monkeypatch, (False, "FAIL, unsupported claim"))
    results = _eval(
        {"faithful_to_source": True},
        _response("It cures everything.", "modest evidence"),
    )
    faithful_result = [r for r in results if r.key == "faithful_to_source"]
    assert faithful_result
    assert not faithful_result[0].passed
    assert any("unsupported claim" in c.detail for c in faithful_result[0].checks)


def test_custom_source_path_selects_that_field(monkeypatch):
    captured = _install_judge(monkeypatch, (True, "PASS"))
    response = {
        "task": {
            "artifacts": [
                {
                    "parts": [
                        {"text": "summary"},
                        {
                            "data": {
                                "abstract": "the real abstract",
                                "response": "other",
                            }
                        },
                    ]
                }
            ]
        }
    }
    _eval({"faithful_to_source": {"source_path": "$.abstract"}}, response)
    assert "the real abstract" in captured["prompt"]
    assert "other" not in captured["prompt"]


def test_empty_sources_still_calls_judge(monkeypatch):
    captured = _install_judge(monkeypatch, (False, "FAIL, no support"))
    results = _eval(
        {"faithful_to_source": True},
        _response("I found strong evidence.", ""),
    )
    faithful_result = [r for r in results if r.key == "faithful_to_source"]
    assert faithful_result
    assert not faithful_result[0].passed
    assert "(no sources were retrieved)" in captured["prompt"]


def test_disabled_when_absent(monkeypatch):
    called = {"n": 0}

    def _fake(prompt):
        called["n"] += 1
        return (True, "PASS")

    monkeypatch.setattr(FaithfulToSource, "_judge", staticmethod(_fake))
    _eval({"must_include": ["x"]}, _response("x here", "s"))
    assert called["n"] == 0
