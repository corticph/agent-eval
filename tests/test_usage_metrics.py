"""Tests for usage/credits accounting extracted from task metadata."""

from __future__ import annotations

from agent_evals.results import EvaluationResult, StepResult, UsageMetrics
from agent_evals.schemas.message import MessagePayload


def _task_response(metadata: dict | None) -> dict:
    task: dict = {"id": "task-1", "status": {"state": "completed"}}
    if metadata is not None:
        task["metadata"] = metadata
    return {"task": task}


class TestFromResponse:
    def test_v2_usage_shape(self) -> None:
        # Agent API v2: metadata.$usage with camelCase fields and nested credits.
        response = _task_response(
            {
                "$usage": {
                    "model": "corti-default",
                    "inputTokens": 26514,
                    "outputTokens": 3319,
                    "totalTokens": 29833,
                    "credits": 0.15916,
                }
            }
        )
        usage = UsageMetrics.from_response(response)
        assert usage == UsageMetrics(
            input_tokens=26514, output_tokens=3319, credits=0.15916
        )

    def test_full_metadata(self) -> None:
        # v1 back-compat shape: _usage snake_case with top-level credits.
        response = _task_response(
            {
                "_usage": {"input_tokens": 26514, "output_tokens": 3319},
                "credits": 0.15916,
            }
        )
        usage = UsageMetrics.from_response(response)
        assert usage == UsageMetrics(
            input_tokens=26514, output_tokens=3319, credits=0.15916
        )

    def test_credits_only(self) -> None:
        usage = UsageMetrics.from_response(_task_response({"credits": 0.03582}))
        assert usage == UsageMetrics(
            input_tokens=None, output_tokens=None, credits=0.03582
        )

    def test_usage_only(self) -> None:
        usage = UsageMetrics.from_response(
            _task_response({"_usage": {"input_tokens": 10, "output_tokens": 2}})
        )
        assert usage == UsageMetrics(input_tokens=10, output_tokens=2, credits=None)

    def test_absent_metadata_returns_none(self) -> None:
        assert UsageMetrics.from_response(_task_response(None)) is None
        assert UsageMetrics.from_response(_task_response({})) is None
        assert UsageMetrics.from_response({}) is None
        assert UsageMetrics.from_response(None) is None

    def test_malformed_values_ignored(self) -> None:
        usage = UsageMetrics.from_response(
            _task_response({"_usage": "not-a-dict", "credits": "not-a-number"})
        )
        assert usage is None
        usage = UsageMetrics.from_response(
            _task_response({"_usage": {"input_tokens": "12"}, "credits": True})
        )
        assert usage is None


class TestAggregate:
    def test_sums_across_steps(self) -> None:
        total = UsageMetrics.aggregate(
            [
                UsageMetrics(input_tokens=100, output_tokens=10, credits=0.1),
                UsageMetrics(input_tokens=200, output_tokens=20, credits=0.2),
            ]
        )
        assert total == UsageMetrics(
            input_tokens=300, output_tokens=30, credits=0.30000000000000004
        )

    def test_partial_fields_sum_what_exists(self) -> None:
        total = UsageMetrics.aggregate(
            [
                UsageMetrics(input_tokens=100, output_tokens=10, credits=None),
                UsageMetrics(input_tokens=None, output_tokens=None, credits=0.5),
                None,
            ]
        )
        assert total == UsageMetrics(input_tokens=100, output_tokens=10, credits=0.5)

    def test_all_empty_returns_none(self) -> None:
        assert UsageMetrics.aggregate([]) is None
        assert UsageMetrics.aggregate([None, None]) is None


class TestResultSerialization:
    def test_evaluation_result_sums_usage_across_steps(self) -> None:
        # Case-level usage is derived from the Steps, not stored: two Steps'
        # accounting sums field-wise. No duplicated top-level ``usage`` column.
        result = EvaluationResult(
            name="case",
            success=True,
            agent_id="a-1",
            step_results=[
                StepResult(
                    name="one",
                    success=True,
                    request=MessagePayload.from_dict({}),
                    response=None,
                    usage=UsageMetrics(input_tokens=5, output_tokens=1, credits=0.01),
                ),
                StepResult(
                    name="two",
                    success=True,
                    request=MessagePayload.from_dict({}),
                    response=None,
                    usage=UsageMetrics(input_tokens=3, output_tokens=2),
                ),
            ],
        )
        assert result.aggregate_usage() == UsageMetrics(
            input_tokens=8, output_tokens=3, credits=0.01
        )
        assert "usage" not in result.as_dict()

    def test_step_result_emits_usage_and_null_when_absent(self) -> None:
        step = StepResult(
            name="step",
            success=True,
            request=MessagePayload.from_dict({}),
            response=None,
            usage=UsageMetrics(input_tokens=5, output_tokens=1),
        )
        assert step.as_dict()["usage"] == {
            "input_tokens": 5,
            "output_tokens": 1,
            "credits": None,
        }
        bare = StepResult(
            name="step",
            success=True,
            request=MessagePayload.from_dict({}),
            response=None,
        )
        assert bare.as_dict()["usage"] is None
