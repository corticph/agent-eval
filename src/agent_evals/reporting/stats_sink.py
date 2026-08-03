"""The stats Sink: cross-run aggregation via the Sink Protocol.

StatsSink is a regular Sink passed to every ``run_suite`` call (the same
instance every run). It accumulates per-run results via ``write``, no-ops on
``close`` (per-run flush is not its job), and does all its work in
``aggregate_runs`` — called once after the multi-run loop by
``run_suite_multiple``.

The writer functions it calls live in :mod:`agent_evals.reporting.multi_run`;
the stats computation lives in :mod:`agent_evals.stats`. This module only
glues the Sink Protocol to the writers.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..loader import EvaluationCase, EvaluationSuite
from ..results import EvaluationResult, SuiteResults
from .multi_run import format_stats_markdown, write_combined_json, write_stats_csv

_LOGGER = logging.getLogger(__name__)


class StatsSink:
    """A Sink that accumulates results across runs and writes stats artifacts.

    The same instance is reused for every run in a multi-run loop. Run
    boundaries are tracked by counting ``on_start`` calls — each one starts
    a fresh batch, and ``close`` seals it into ``_all_results``. No run count
    is needed upfront.
    """

    def __init__(
        self,
        output_path: Path,
        *,
        suite_path: str,
        env: str | None = None,
    ) -> None:
        self._output_path = output_path
        self._suite_path = suite_path
        self._env = env
        self._all_results: list[list[EvaluationResult]] = []
        self._current_run: list[EvaluationResult] = []

    def on_start(self, suite: EvaluationSuite) -> None:
        self._current_run = []

    def write(self, case: EvaluationCase, result: EvaluationResult) -> None:
        self._current_run.append(result)

    def close(self) -> None:
        self._all_results.append(self._current_run)

    def aggregate_runs(self) -> None:
        """Write ``<stem>_stats.csv``, ``<stem>_combined.json``, and ``<stem>_stats.md``.

        Called once after the multi-run loop. Builds a :class:`SuiteResults`
        from the accumulated per-run result lists and hands it to the three
        writer functions in :mod:`agent_evals.reporting.multi_run`.
        """
        suite_results = SuiteResults(
            suite_path=self._suite_path,
            all_results=self._all_results,
        )

        stats_csv_path = self._output_path.with_name(
            f"{self._output_path.stem}_stats.csv"
        )
        write_stats_csv(stats_csv_path, [suite_results], env=self._env)
        _LOGGER.info("Stats CSV: %s", stats_csv_path)

        combined_json_path = self._output_path.with_name(
            f"{self._output_path.stem}_combined.json"
        )
        write_combined_json(combined_json_path, [suite_results], env=self._env)
        _LOGGER.info("Combined JSON: %s", combined_json_path)

        md_path = self._output_path.with_name(f"{self._output_path.stem}_stats.md")
        md_path.write_text(
            format_stats_markdown([suite_results], env=self._env), encoding="utf-8"
        )
        _LOGGER.info("Stats Markdown: %s", md_path)
