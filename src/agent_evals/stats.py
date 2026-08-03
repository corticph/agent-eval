"""Aggregation helpers for multi-run evaluation results."""

from __future__ import annotations

import statistics
from typing import Any

from .results import EvaluationResult, SuiteResults


METRICS: tuple[str, ...] = (
    "duration_seconds",
    "input_tokens",
    "output_tokens",
    "credits",
)


def compute_stats(
    suite_runs: list[SuiteResults],
) -> list[dict[str, Any]]:
    """Compute per-(suite, eval) aggregated statistics across multiple runs.

    Returns a list of dicts (one per suite/eval combination) with keys:
    ``suite``, ``eval_name``, ``runs``, ``success_rate``, and for each
    metric: ``mean``, ``min``, ``max``, ``std``.
    """
    by_key: dict[tuple[str, str], list[EvaluationResult]] = {}
    for suite_run in suite_runs:
        for results in suite_run.all_results:
            for result in results:
                key = (suite_run.suite_path, result.name)
                by_key.setdefault(key, []).append(result)

    stats_list: list[dict[str, Any]] = []
    for (suite_path, eval_name), results in by_key.items():
        n = len(results)
        success_count = sum(1 for r in results if r.success)
        entry: dict[str, Any] = {
            "suite": suite_path,
            "eval_name": eval_name,
            "runs": n,
            "success_rate": success_count / n if n else 0.0,
        }
        for metric in METRICS:
            if metric == "duration_seconds":
                values = [
                    r.duration_seconds
                    for r in results
                    if r.duration_seconds is not None
                ]
            else:
                values = [
                    getattr(usage, metric)
                    for r in results
                    if (usage := r.aggregate_usage()) is not None
                ]
                values = [v for v in values if v is not None]
            if values:
                entry[f"mean_{metric}"] = statistics.fmean(values)
                entry[f"min_{metric}"] = min(values)
                entry[f"max_{metric}"] = max(values)
                entry[f"std_{metric}"] = (
                    statistics.pstdev(values) if len(values) > 1 else 0.0
                )
            else:
                entry[f"mean_{metric}"] = None
                entry[f"min_{metric}"] = None
                entry[f"max_{metric}"] = None
                entry[f"std_{metric}"] = None
        stats_list.append(entry)
    return stats_list
