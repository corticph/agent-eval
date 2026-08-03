"""Tests for the ``expected_state`` expectation and the terminal-state guard."""

from __future__ import annotations

from agent_evals.expectations import evaluate_response, parse_expectations
from agent_evals.expectations.base import extract_plain_text
from agent_evals.expectations.state import normalize_task_state


def _task_response(state: str, text: str = "the answer") -> dict:
    return {
        "task": {
            "id": "task-1",
            "status": {
                "state": state,
                "message": {"parts": [{"kind": "text", "text": text}]},
            },
        }
    }


def _v2_task_response(state: str, text: str = "the answer") -> dict:
    # Shape returned by the live Agent API v2 server: TASK_STATE_* enum and
    # message parts with no ``kind`` discriminator.
    return {
        "task": {
            "id": "019f474a-3b61-78ad-9c08-30c928d3a0fd",
            "contextId": "019f474a-3b61-794b-8dea-33255ed1db5e",
            "status": {
                "state": state,
                "message": {"role": "ROLE_AGENT", "parts": [{"text": text}]},
            },
        }
    }


def _results(block: dict | None, response: dict) -> dict:
    """Resolve *block* against *response* into a ``{key: ExpectationResult}`` map."""
    return {r.key: r for r in evaluate_response(parse_expectations(block), response)}


def _passed(block: dict | None, response: dict) -> bool:
    return all(r.passed for r in evaluate_response(parse_expectations(block), response))


class TestNormalizeTaskState:
    def test_wire_enum_short_form_and_snake_case_collapse(self) -> None:
        assert normalize_task_state("TASK_STATE_INPUT_REQUIRED") == "INPUT_REQUIRED"
        assert normalize_task_state("input-required") == "INPUT_REQUIRED"
        assert normalize_task_state("input_required") == "INPUT_REQUIRED"
        assert normalize_task_state("  Input-Required  ") == "INPUT_REQUIRED"

    def test_empty_and_non_string_return_none(self) -> None:
        assert normalize_task_state("") is None
        assert normalize_task_state("   ") is None
        assert normalize_task_state(None) is None


class TestV2WireShapes:
    def test_v2_completed_passes(self) -> None:
        assert _passed(
            {"must_include": ["answer"]}, _v2_task_response("TASK_STATE_COMPLETED")
        )

    def test_v2_failed_fails_by_default(self) -> None:
        results = _results({}, _v2_task_response("TASK_STATE_FAILED"))
        assert not results["expected_state"].passed

    def test_v2_canceled_fails_by_default(self) -> None:
        assert not _passed({}, _v2_task_response("TASK_STATE_CANCELED"))

    def test_short_form_matches_wire_enum(self) -> None:
        # Author writes the readable short form; server returns the TASK_STATE_* enum.
        assert _passed(
            {"expected_state": "input-required"},
            _v2_task_response("TASK_STATE_INPUT_REQUIRED"),
        )

    def test_v2_parts_without_kind_extract_text(self) -> None:
        assert (
            extract_plain_text(_v2_task_response("TASK_STATE_COMPLETED", "hello v2"))
            == "hello v2"
        )


class TestTerminalStateGuard:
    def test_completed_state_passes(self) -> None:
        assert _passed({"must_include": ["answer"]}, _task_response("completed"))

    def test_input_required_state_passes(self) -> None:
        assert _passed({}, _task_response("input-required"))

    def test_failed_state_fails_even_when_content_matches(self) -> None:
        results = _results({"must_include": ["answer"]}, _task_response("failed"))
        # The content matched, but the terminal-failure guard still fails the step.
        assert results["must_include"].passed
        assert not results["expected_state"].passed

    def test_rejected_state_fails_by_default(self) -> None:
        results = _results({}, _task_response("rejected"))
        assert not results["expected_state"].passed


class TestExpectedState:
    def test_allows_declared_terminal_failure(self) -> None:
        assert _passed({"expected_state": "failed"}, _task_response("failed"))

    def test_mismatch_fails(self) -> None:
        results = _results({"expected_state": "failed"}, _task_response("completed"))
        assert not results["expected_state"].passed

    def test_parsed_from_dict(self) -> None:
        (expected_state,) = [
            e
            for e in parse_expectations({"expected_state": "rejected"})
            if e.key == "expected_state"
        ]
        assert expected_state.expected == "rejected"
        assert _passed({"expected_state": "rejected"}, _task_response("rejected"))


class TestExpectedOutputTextCoercion:
    def test_list_joined_to_single_string(self) -> None:
        (judge,) = [
            e
            for e in parse_expectations(
                {"expected_output_text": ["line one", "line two"]}
            )
            if e.key == "expected_output_text"
        ]
        assert judge.reference == "line one\nline two"

    def test_plain_string_unchanged(self) -> None:
        (judge,) = [
            e
            for e in parse_expectations({"expected_output_text": "reference"})
            if e.key == "expected_output_text"
        ]
        assert judge.reference == "reference"
