"""Tests for the stats aggregation and CSV/JSON writing module."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from agent_evals.results import EvaluationResult, StepResult, SuiteResults, UsageMetrics
from agent_evals import stats
from agent_evals.reporting.multi_run import (
    format_stats_markdown,
    write_combined_json,
    write_stats_csv,
)


def _result(
    name: str,
    success: bool,
    agent_id: str,
    duration: float,
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    credits: float | None = None,
) -> EvaluationResult:
    """Build an EvaluationResult with one step carrying the given usage."""
    usage = None
    if input_tokens is not None or output_tokens is not None or credits is not None:
        usage = UsageMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            credits=credits,
        )
    return EvaluationResult(
        name=name,
        success=success,
        agent_id=agent_id,
        duration_seconds=duration,
        step_results=[
            StepResult(
                name="step1",
                success=success,
                request=None,
                response=None,
                duration_seconds=duration,
                usage=usage,
            )
        ],
    )


@pytest.fixture()
def sample_results() -> list[list[EvaluationResult]]:
    """Three runs, two evals each."""
    return [
        [
            _result(
                "eval_a",
                True,
                "a1",
                3.0,
                input_tokens=100,
                output_tokens=50,
                credits=0.01,
            ),
            _result(
                "eval_b",
                False,
                "b1",
                5.0,
                input_tokens=200,
                output_tokens=75,
                credits=0.02,
            ),
        ],
        [
            _result(
                "eval_a",
                True,
                "a2",
                4.0,
                input_tokens=110,
                output_tokens=55,
                credits=0.011,
            ),
            _result(
                "eval_b",
                True,
                "b2",
                6.0,
                input_tokens=210,
                output_tokens=80,
                credits=0.022,
            ),
        ],
        [
            _result(
                "eval_a",
                True,
                "a3",
                3.5,
                input_tokens=105,
                output_tokens=52,
                credits=0.012,
            ),
            _result(
                "eval_b",
                True,
                "b3",
                5.5,
                input_tokens=205,
                output_tokens=77,
                credits=0.021,
            ),
        ],
    ]


@pytest.fixture()
def sample_suite_runs(
    sample_results: list[list[EvaluationResult]],
) -> list[SuiteResults]:
    """Two suites, each with the same 3-run results."""
    return [
        SuiteResults(suite_path="evals/foo/bar.yaml", all_results=sample_results),
        SuiteResults(suite_path="evals/baz/qux.yaml", all_results=sample_results),
    ]


class TestComputeStats:
    def test_success_rate(self, sample_suite_runs: list[SuiteResults]) -> None:
        stats_list = stats.compute_stats(sample_suite_runs)
        by_key = {(e["suite"], e["eval_name"]): e for e in stats_list}
        assert by_key[("evals/foo/bar.yaml", "eval_a")]["success_rate"] == 1.0
        assert by_key[("evals/foo/bar.yaml", "eval_b")][
            "success_rate"
        ] == pytest.approx(2 / 3)

    def test_mean_duration(self, sample_suite_runs: list[SuiteResults]) -> None:
        stats_list = stats.compute_stats(sample_suite_runs)
        by_key = {(e["suite"], e["eval_name"]): e for e in stats_list}
        assert by_key[("evals/foo/bar.yaml", "eval_a")][
            "mean_duration_seconds"
        ] == pytest.approx(3.5)
        assert by_key[("evals/baz/qux.yaml", "eval_b")][
            "mean_duration_seconds"
        ] == pytest.approx(5.5)

    def test_mean_tokens(self, sample_suite_runs: list[SuiteResults]) -> None:
        stats_list = stats.compute_stats(sample_suite_runs)
        by_key = {(e["suite"], e["eval_name"]): e for e in stats_list}
        assert by_key[("evals/foo/bar.yaml", "eval_a")][
            "mean_input_tokens"
        ] == pytest.approx(105.0)
        assert by_key[("evals/foo/bar.yaml", "eval_a")][
            "mean_output_tokens"
        ] == pytest.approx(52.3333, rel=1e-3)

    def test_std_dev(self, sample_suite_runs: list[SuiteResults]) -> None:
        stats_list = stats.compute_stats(sample_suite_runs)
        by_key = {(e["suite"], e["eval_name"]): e for e in stats_list}
        assert by_key[("evals/foo/bar.yaml", "eval_a")][
            "std_duration_seconds"
        ] == pytest.approx(0.4082, rel=1e-3)

    def test_suite_column_present(self, sample_suite_runs: list[SuiteResults]) -> None:
        stats_list = stats.compute_stats(sample_suite_runs)
        suites = {e["suite"] for e in stats_list}
        assert suites == {"evals/foo/bar.yaml", "evals/baz/qux.yaml"}

    def test_missing_values_reported_as_none(self) -> None:
        suite_runs = [
            SuiteResults(
                suite_path="evals/x.yaml",
                all_results=[
                    [
                        _result("eval_x", True, "x", 1.0),
                    ]
                ],
            )
        ]
        stats_list = stats.compute_stats(suite_runs)
        entry = stats_list[0]
        assert entry["mean_input_tokens"] is None
        assert entry["mean_output_tokens"] is None
        assert entry["mean_credits"] is None

    def test_single_run_std_is_zero(self) -> None:
        suite_runs = [
            SuiteResults(
                suite_path="evals/x.yaml",
                all_results=[
                    [
                        _result(
                            "eval_x", True, "x", 1.0, input_tokens=10, output_tokens=5
                        ),
                    ]
                ],
            )
        ]
        stats_list = stats.compute_stats(suite_runs)
        assert stats_list[0]["std_duration_seconds"] == 0.0


class TestWriteStatsCsv:
    def test_stats_csv_has_one_row_per_suite_eval(
        self, sample_suite_runs: list[SuiteResults], tmp_path: Path
    ) -> None:
        path = tmp_path / "stats.csv"
        write_stats_csv(path, sample_suite_runs)
        with path.open() as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        # 2 suites * 2 evals = 4 rows
        assert len(rows) == 4
        keys = {(r["suite"], r["eval_name"]) for r in rows}
        assert keys == {
            ("evals/foo/bar.yaml", "eval_a"),
            ("evals/foo/bar.yaml", "eval_b"),
            ("evals/baz/qux.yaml", "eval_a"),
            ("evals/baz/qux.yaml", "eval_b"),
        }

    def test_stats_csv_contains_mean_columns(
        self, sample_suite_runs: list[SuiteResults], tmp_path: Path
    ) -> None:
        path = tmp_path / "stats.csv"
        write_stats_csv(path, sample_suite_runs)
        with path.open() as fh:
            reader = csv.DictReader(fh)
            assert "mean_duration_seconds" in reader.fieldnames
            assert "mean_input_tokens" in reader.fieldnames
            assert "mean_output_tokens" in reader.fieldnames
            assert "mean_credits" in reader.fieldnames
            assert "success_rate" in reader.fieldnames
            assert "suite" in reader.fieldnames

    def test_stats_csv_includes_env(
        self, sample_suite_runs: list[SuiteResults], tmp_path: Path
    ) -> None:
        path = tmp_path / "stats.csv"
        write_stats_csv(path, sample_suite_runs, env="eu")
        with path.open() as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert all(r["environment"] == "eu" for r in rows)


class TestWriteCombinedJson:
    def test_json_has_run_and_suite_fields(
        self, sample_suite_runs: list[SuiteResults], tmp_path: Path
    ) -> None:
        path = tmp_path / "results.json"
        write_combined_json(path, sample_suite_runs)
        with path.open() as fh:
            data = json.load(fh)
        # 2 suites * 3 runs * 2 evals = 12 result entries + 12 step entries = 24
        result_entries = [e for e in data if not e.get("is_step_result")]
        step_entries = [e for e in data if e.get("is_step_result")]
        assert len(result_entries) == 12
        assert len(step_entries) == 12
        for entry in result_entries:
            assert "run" in entry
            assert "suite" in entry
        assert result_entries[0]["run"] == 1
        assert result_entries[0]["suite"] == "evals/foo/bar.yaml"

    def test_json_includes_env(
        self, sample_suite_runs: list[SuiteResults], tmp_path: Path
    ) -> None:
        path = tmp_path / "results.json"
        write_combined_json(path, sample_suite_runs, env="dev-weu")
        with path.open() as fh:
            data = json.load(fh)
        assert all(e.get("environment") == "dev-weu" for e in data)


class TestFormatStatsMarkdown:
    def test_markdown_has_summary_table(
        self, sample_suite_runs: list[SuiteResults]
    ) -> None:
        md = format_stats_markdown(sample_suite_runs)
        assert "Multi-Run Summary" in md
        assert "eval_a" in md
        assert "eval_b" in md
        assert "Mean In Tok" in md

    def test_markdown_includes_env(self, sample_suite_runs: list[SuiteResults]) -> None:
        md = format_stats_markdown(sample_suite_runs, env="staging-eu")
        assert "Environment: staging-eu" in md

    def test_markdown_groups_by_suite(
        self, sample_suite_runs: list[SuiteResults]
    ) -> None:
        md = format_stats_markdown(sample_suite_runs)
        assert "evals/foo/bar.yaml" in md
        assert "evals/baz/qux.yaml" in md

    def test_markdown_empty_results(self) -> None:
        md = format_stats_markdown([])
        assert "No results" in md
