"""Per-step results survive a round-trip through plain dicts.

The Opik sink's replay task serializes each step's per-term results into its
task output, and the metric deserializes and renders them. These tests pin that
channel: an ``ExpectationResult`` reconstructed from its ``to_dict()`` output
equals the original — checks, labels, verdicts, and ``show_on_pass`` intact.
"""

from __future__ import annotations

from agent_evals.errors import ErrorCode, EvalError
from agent_evals.expectations import CheckResult, ExpectationResult
from agent_evals.results import StepResult


def _roundtrip(result: ExpectationResult) -> ExpectationResult:
    return ExpectationResult.from_dict(result.to_dict())


class TestExpectationResultRoundTrip:
    def test_list_block_checks_survive(self) -> None:
        result = ExpectationResult(
            "must_include",
            [
                CheckResult("refund", True, "passed"),
                CheckResult("apology", False, "missing required phrase: 'apology'"),
            ],
        )
        assert _roundtrip(result) == result

    def test_scalar_check_survives(self) -> None:
        result = ExpectationResult(
            "max_duration_seconds",
            [
                CheckResult(
                    "", False, "duration 3.000s exceeded max_duration_seconds=2.000s"
                )
            ],
        )
        assert _roundtrip(result) == result

    def test_judge_show_on_pass_flag_survives(self) -> None:
        # The judge is one show_on_pass result; the flag must ride the channel so
        # the sink keeps rendering its explanation on a pass.
        result = ExpectationResult(
            "expected_output_text",
            [CheckResult("", True, "same core facts")],
            show_on_pass=True,
        )
        restored = _roundtrip(result)
        assert restored == result
        assert restored.show_on_pass is True

    def test_missing_show_on_pass_defaults_false(self) -> None:
        # A payload written before ``show_on_pass`` existed restores as a plain
        # (non-judge) result rather than raising.
        restored = ExpectationResult.from_dict(
            {
                "key": "must_match",
                "checks": [
                    {
                        "label": "life[ -]?threatening",
                        "passed": True,
                        "detail": "passed",
                    }
                ],
            }
        )
        assert restored.show_on_pass is False
        assert restored.passed


class TestStepResultHarnessErrorSerialization:
    def test_present_harness_error_serializes_as_typed_object(self) -> None:
        step = StepResult(
            name=None,
            success=False,
            request=None,
            response=None,
            harness_error=EvalError(ErrorCode.TIMEOUT, "read timed out"),
        )
        payload = step.as_dict()
        assert payload["harness_error"] == {
            "code": "timeout",
            "message": "read timed out",
            "context": {},
        }
        assert "errors" not in payload

    def test_absent_harness_error_serializes_as_null(self) -> None:
        payload = StepResult(
            name="s", success=True, request=None, response=None
        ).as_dict()
        assert payload["harness_error"] is None
        assert "errors" not in payload
