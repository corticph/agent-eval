"""Multi-run aggregate serialization: CSV, JSON, and Markdown writers.

These operate on ``list[SuiteResults]`` (cross-run aggregates), not on a
single ``run_suite``'s case-by-case stream — that is the per-run
:class:`~agent_evals.reporting.file.FileSink`'s job.  The actual statistics
computation lives in :mod:`agent_evals.stats`; this module only serializes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ..results import SuiteResults
from ..stats import METRICS, compute_stats


def write_stats_csv(
    path: Path,
    suite_runs: list[SuiteResults],
    *,
    env: str | None = None,
) -> None:
    """Write an aggregated stats CSV with one row per (suite, eval) combination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stats_list = compute_stats(suite_runs)
    header: list[str] = ["environment", "suite", "eval_name", "runs", "success_rate"]
    for metric in METRICS:
        for stat in ("mean", "min", "max", "std"):
            header.append(f"{stat}_{metric}")
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for entry in stats_list:
            row: list[str] = [
                env or "",
                entry["suite"],
                entry["eval_name"],
                str(entry["runs"]),
                f"{entry['success_rate']:.4f}",
            ]
            for metric in METRICS:
                for stat in ("mean", "min", "max", "std"):
                    value = entry.get(f"{stat}_{metric}")
                    row.append(_fmt(value))
            writer.writerow(row)


def write_combined_json(
    path: Path,
    suite_runs: list[SuiteResults],
    *,
    env: str | None = None,
) -> None:
    """Write a combined JSON array with all results from every run.

    Each entry gets ``run``, ``suite``, and ``environment`` fields.
    Step results are expanded into their own entries with ``is_step_result: true``.
    """
    entries: list[dict[str, Any]] = []
    for suite_run in suite_runs:
        for run_index, results in enumerate(suite_run.all_results, 1):
            for result in results:
                entry = result.as_dict()
                entry["run"] = run_index
                entry["suite"] = suite_run.suite_path
                if env:
                    entry["environment"] = env
                entries.append(entry)
                if result.step_results:
                    for step in result.step_results:
                        step_entry = step.as_dict()
                        step_entry["run"] = run_index
                        step_entry["suite"] = suite_run.suite_path
                        if env:
                            step_entry["environment"] = env
                        step_entry["parent_name"] = result.name
                        step_entry["is_step_result"] = True
                        entries.append(step_entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(entries, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def format_stats_markdown(
    suite_runs: list[SuiteResults],
    *,
    env: str | None = None,
) -> str:
    """Return a Markdown summary with aggregated stats, grouped by suite."""
    stats_list = compute_stats(suite_runs)
    if not stats_list:
        return "# Evaluation Results\n\nNo results to summarise.\n"

    total_runs = max(
        (len(sr.all_results) for sr in suite_runs),
        default=0,
    )
    env_label = f"Environment: {env}" if env else "Environment: (default)"
    lines: list[str] = [
        "# Evaluation Results (Multi-Run Summary)",
        "",
        f"Runs: {total_runs}",
        env_label,
        "",
    ]

    # Group by suite
    by_suite: dict[str, list[dict[str, Any]]] = {}
    for entry in stats_list:
        by_suite.setdefault(entry["suite"], []).append(entry)

    for suite_path, entries in by_suite.items():
        lines.append(f"## {suite_path}")
        lines.append("")
        lines.append(
            "| Eval | Runs | Pass% | Mean Dur (s) | Mean In Tok | Mean Out Tok | Mean Credits |"
        )
        lines.append(
            "|-----|------|-------|-------------|-------------|-------------|-------------|"
        )
        for entry in entries:
            lines.append(
                "| {name} | {runs} | {rate:.1%} | {dur} | {in_tok} | {out_tok} | {credits} |".format(
                    name=entry["eval_name"],
                    runs=entry["runs"],
                    rate=entry["success_rate"],
                    dur=_fmt(entry.get("mean_duration_seconds")),
                    in_tok=_fmt(entry.get("mean_input_tokens")),
                    out_tok=_fmt(entry.get("mean_output_tokens")),
                    credits=_fmt(entry.get("mean_credits")),
                )
            )
        lines.append("")

    # Detailed stats per metric
    for metric in METRICS:
        lines.append(f"## {metric}")
        lines.append("")
        lines.append("| Suite | Eval | Mean | Min | Max | Std |")
        lines.append("|-------|------|------|-----|-----|-----|")
        for entry in stats_list:
            lines.append(
                "| {suite} | {name} | {mean} | {min} | {max} | {std} |".format(
                    suite=entry["suite"],
                    name=entry["eval_name"],
                    mean=_fmt(entry.get(f"mean_{metric}")),
                    min=_fmt(entry.get(f"min_{metric}")),
                    max=_fmt(entry.get(f"max_{metric}")),
                    std=_fmt(entry.get(f"std_{metric}")),
                )
            )
        lines.append("")

    return "\n".join(lines)


def _fmt(value: Any) -> str:
    """Format a numeric value for CSV/Markdown output."""
    if value is None:
        return ""
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        return f"{value:.4f}"
    return str(value)
