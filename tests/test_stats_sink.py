"""Tests for the StatsSink: cross-run accumulation via the Sink Protocol.

StatsSink is driven through the same four hooks as every other Sink
(``on_start``, ``write``, ``close``, ``aggregate_runs``), but the same
instance is reused across runs so it can accumulate.  ``aggregate_runs``
writes the CSV, JSON, and Markdown artifacts after the multi-run loop.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from agent_evals.loader import EvaluationCase, EvaluationSuite, Step, SuiteOptions
from agent_evals.reporting.stats_sink import StatsSink
from agent_evals.results import EvaluationResult, StepResult, UsageMetrics
from agent_evals.schemas.agent import Agent
from agent_evals.schemas.message import MessagePayload


def _result(name: str, success: bool = True) -> EvaluationResult:
    return EvaluationResult(
        name=name,
        success=success,
        agent_id="agent-1",
        duration_seconds=1.0,
        step_results=[
            StepResult(
                name="step1",
                success=success,
                request=None,
                response=None,
                duration_seconds=1.0,
                usage=UsageMetrics(input_tokens=10, output_tokens=5),
            )
        ],
    )


def _suite(name: str = "test_suite") -> EvaluationSuite:
    return EvaluationSuite(
        name=name,
        cases=[
            EvaluationCase(
                name="eval_a",
                agent=Agent(name="TestAgent"),
                steps=[
                    Step(
                        message=MessagePayload.from_dict(
                            {"message": {"parts": [{"kind": "text", "text": "hi"}]}}
                        ),
                        expectations=[],
                    )
                ],
            )
        ],
        options=SuiteOptions(concurrency=1),
    )


class TestStatsSinkAccumulation:
    """The same instance accumulates across runs via on_start/write/close."""

    def test_single_run_accumulates(self) -> None:
        sink = StatsSink(Path("out.json"), suite_path="evals/test.yaml")
        suite = _suite()

        sink.on_start(suite)
        sink.write(suite.cases[0], _result("eval_a", True))
        sink.close()

        assert len(sink._all_results) == 1
        assert len(sink._all_results[0]) == 1
        assert sink._all_results[0][0].name == "eval_a"

    def test_multiple_runs_accumulate_separately(self) -> None:
        sink = StatsSink(Path("out.json"), suite_path="evals/test.yaml")
        suite = _suite()

        for _ in range(3):
            sink.on_start(suite)
            sink.write(suite.cases[0], _result("eval_a", True))
            sink.close()

        assert len(sink._all_results) == 3
        for run_results in sink._all_results:
            assert len(run_results) == 1

    def test_on_start_resets_current_run(self) -> None:
        sink = StatsSink(Path("out.json"), suite_path="evals/test.yaml")
        suite = _suite()

        sink.on_start(suite)
        sink.write(suite.cases[0], _result("first", True))
        sink.close()

        sink.on_start(suite)
        assert sink._current_run == []

    def test_close_without_on_start_is_empty_run(self) -> None:
        sink = StatsSink(Path("out.json"), suite_path="evals/test.yaml")

        sink.close()
        assert len(sink._all_results) == 1
        assert sink._all_results[0] == []


class TestStatsSinkAggregateRuns:
    """aggregate_runs writes CSV, JSON, and Markdown artifacts."""

    def test_writes_three_artifacts(self, tmp_path: Path) -> None:
        output_path = tmp_path / "results.json"
        sink = StatsSink(output_path, suite_path="evals/test.yaml", env="eu")
        suite = _suite()

        sink.on_start(suite)
        sink.write(suite.cases[0], _result("eval_a", True))
        sink.close()

        sink.aggregate_runs()

        assert (tmp_path / "results_stats.csv").exists()
        assert (tmp_path / "results_combined.json").exists()
        assert (tmp_path / "results_stats.md").exists()

    def test_csv_has_env_and_suite(self, tmp_path: Path) -> None:
        output_path = tmp_path / "results.json"
        sink = StatsSink(output_path, suite_path="evals/test.yaml", env="eu")
        suite = _suite()

        sink.on_start(suite)
        sink.write(suite.cases[0], _result("eval_a", True))
        sink.close()
        sink.aggregate_runs()

        csv_path = tmp_path / "results_stats.csv"
        with csv_path.open() as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["environment"] == "eu"
        assert rows[0]["suite"] == "evals/test.yaml"
        assert rows[0]["eval_name"] == "eval_a"

    def test_json_includes_run_and_suite(self, tmp_path: Path) -> None:
        output_path = tmp_path / "results.json"
        sink = StatsSink(output_path, suite_path="evals/test.yaml", env="dev")
        suite = _suite()

        for run_idx in range(2):
            sink.on_start(suite)
            sink.write(suite.cases[0], _result("eval_a", True))
            sink.close()

        sink.aggregate_runs()

        json_path = tmp_path / "results_combined.json"
        with json_path.open() as fh:
            data = json.load(fh)
        result_entries = [e for e in data if not e.get("is_step_result")]
        assert len(result_entries) == 2
        assert result_entries[0]["run"] == 1
        assert result_entries[1]["run"] == 2
        assert all(e["suite"] == "evals/test.yaml" for e in result_entries)

    def test_markdown_has_summary(self, tmp_path: Path) -> None:
        output_path = tmp_path / "results.json"
        sink = StatsSink(output_path, suite_path="evals/test.yaml", env="eu")
        suite = _suite()

        sink.on_start(suite)
        sink.write(suite.cases[0], _result("eval_a", True))
        sink.close()
        sink.aggregate_runs()

        md_path = tmp_path / "results_stats.md"
        md = md_path.read_text()
        assert "Multi-Run Summary" in md
        assert "eval_a" in md
        assert "Environment: eu" in md

    def test_aggregate_runs_with_no_results(self, tmp_path: Path) -> None:
        output_path = tmp_path / "results.json"
        sink = StatsSink(output_path, suite_path="evals/test.yaml")

        sink.aggregate_runs()

        md_path = tmp_path / "results_stats.md"
        assert md_path.exists()
        assert "No results" in md_path.read_text()


class TestStatsSinkIsASink:
    """StatsSink satisfies the Sink Protocol structurally."""

    def test_implements_all_four_hooks(self) -> None:
        sink = StatsSink(Path("out.json"), suite_path="evals/test.yaml")
        assert callable(sink.on_start)
        assert callable(sink.write)
        assert callable(sink.close)
        assert callable(sink.aggregate_runs)
