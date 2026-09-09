"""Read local eval results from ``results/*.json`` as if they were Opik experiments.

Provides the same data shape the Opik scripts consume (``feedback_scores``,
``dataset_item_data``, ``evaluation_task_output``) so ``compare_experiments``,
``inspect_eval``, and ``generate_report`` can work without Opik.

A local "experiment" is one JSON file written by ``FileSink``.  Its file stem
encodes the suite name (mirroring ``_auto_output_path``).  Tags and environment
are stored in a ``_metadata`` entry at the start of the JSON array, so
``has_tag`` can match experiments by sweep tag the same way Opik does.

Usage::

    from agent_evals.scripts.local_store import list_experiments, get_items

    exps = list_experiments(Path("results"))
    items = get_items(exps["my_suite"])
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _score_name(er: dict[str, Any]) -> str:
    return er.get("key", "?")


def _compute_feedback_scores(step_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute ``feedback_scores`` from ``step_results[].expectation_results``.

    Mirrors ``ExpectationMetric._score_sequential``: one score per expectation
    type (average of passed/total across steps), plus an ``overall`` that is
    the mean of the per-type scores.
    """
    merged: dict[str, dict[str, Any]] = {}
    for sr in step_results:
        step_name = sr.get("name") or "null"
        for er in sr.get("expectation_results") or []:
            key = _score_name(er)
            checks = er.get("checks") or []
            if not checks:
                continue
            passed = sum(1 for c in checks if c.get("passed"))
            total = len(checks)
            slot = merged.setdefault(key, {"passed": 0, "total": 0, "reason": {}})
            slot["passed"] += passed
            slot["total"] += total
            failed_details = [c["detail"] for c in checks if not c.get("passed")]
            if failed_details:
                slot["reason"][step_name] = "; ".join(failed_details)

    columns: list[dict[str, Any]] = []
    for key, slot in merged.items():
        value = slot["passed"] / slot["total"] if slot["total"] else 1.0
        reason = json.dumps(slot["reason"]) if slot["reason"] else "passed"
        columns.append({"name": key, "value": value, "reason": reason})

    if columns:
        overall = sum(c["value"] for c in columns) / len(columns)
    else:
        overall = 1.0
    overall_reason = "; ".join(
        f"{sr.get('name') or 'null'}: {c['detail']}"
        for sr in step_results
        for er in sr.get("expectation_results") or []
        for c in er.get("checks") or []
        if not c.get("passed")
    ) or "passed"

    return [{"name": "overall", "value": overall, "reason": overall_reason}, *columns]


def _build_item(entry: dict[str, Any]) -> SimpleNamespace:
    """Build a pseudo-Opik item from one JSON entry in a results file.

    The shape mirrors what the Opik scripts read: ``dataset_item_data.name``,
    ``evaluation_task_output`` (step_results, trace_url, usage, environment),
    and ``feedback_scores``.
    """
    name = entry.get("name", "(unnamed)")

    # Step results live directly in the entry; step trail entries (with
    # is_step_result) are separate rows that we skip — the parent entry
    # already carries the full step_results list.
    step_results = entry.get("step_results") or []

    feedback_scores = _compute_feedback_scores(step_results)

    task_output: dict[str, Any] = {
        "step_results": step_results,
        "trace_url": entry.get("trace_url"),
    }
    # Aggregate usage from step results
    usage_parts = [sr.get("usage") for sr in step_results if sr.get("usage")]
    if usage_parts:
        merged_usage: dict[str, Any] = {}
        for u in usage_parts:
            for k, v in u.items():
                if v is not None:
                    merged_usage[k] = merged_usage.get(k, 0) + v
        task_output["usage"] = merged_usage

    return SimpleNamespace(
        dataset_item_data={"name": name},
        evaluation_task_output=task_output,
        feedback_scores=feedback_scores,
    )


def _load_results_file(path: Path) -> list[SimpleNamespace]:
    """Load a results JSON file and return pseudo-Opik items."""
    data = json.loads(path.read_text(encoding="utf-8"))
    items: list[SimpleNamespace] = []
    for entry in data:
        if entry.get("_metadata") or entry.get("is_step_result"):
            continue
        items.append(_build_item(entry))
    return items


def _read_metadata(path: Path) -> dict[str, Any]:
    """Read the _metadata entry from a results JSON file (if present)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list) and data and isinstance(data[0], dict) and data[0].get("_metadata"):
            return data[0]
    except (json.JSONDecodeError, OSError, TypeError):
        pass
    return {}


def _suite_name_from_path(path: Path) -> str:
    """Extract the suite name from a results file path.

    FileSink writes ``results/<topic>/<name>_<timestamp>[_<env>].json``.
    The stem is ``<name>_<timestamp>[_<env>]``; we strip the timestamp and
    env suffix to recover the suite name.
    """
    stem = path.stem
    # Strip _<env> suffix if present (e.g. _eu, _staging-eu)
    stem = re.sub(r"_([a-z][a-z0-9-]*)$", "", stem, count=1)
    # Strip _<timestamp> suffix (YYYYMMDD_HHMMSS)
    stem = re.sub(r"_\d{8}_\d{6}$", "", stem)
    return stem


def list_experiments(
    results_dir: Path,
    *,
    env: str | None = None,
    name: str | None = None,
    tag: str | None = None,
) -> dict[str, SimpleNamespace]:
    """Discover local result files and return ``{suite_name: exp_row}``.

    Each result file is one experiment.  When multiple files map to the same
    suite name (e.g. multiple runs), the newest (by mtime) wins.

    If *tag* is provided, only files matching the tag are considered, so the
    dedup picks the newest among the tagged set rather than the global newest
    (which may carry a different tag and hide the one we care about).
    """
    if not results_dir.is_dir():
        raise SystemExit(f"Results directory not found: {results_dir}")

    pattern = "*.json"
    if env:
        pattern = f"*_{env}.json"

    by_name: dict[str, list[SimpleNamespace]] = {}
    for path in results_dir.rglob(pattern):
        if path.name.startswith("_"):
            continue
        meta = _read_metadata(path)
        # Prefer the suite name from metadata (matches Opik's experiment name);
        # fall back to filename-derived name for older results without metadata.
        suite_name = meta.get("suite_name") or _suite_name_from_path(path)
        if name and name not in suite_name:
            continue
        exp = SimpleNamespace(
            id=str(path),
            name=suite_name,
            tags=meta.get("tags", []),
            created_at=str(path.stat().st_mtime),
            metadata=meta,
            _path=path,
        )
        # Filter by tag BEFORE grouping so dedup only picks from tagged files
        if tag is not None:
            if not has_tag(exp, tag):
                continue
        by_name.setdefault(suite_name, []).append(exp)

    chosen: dict[str, SimpleNamespace] = {}
    for suite_name, lst in by_name.items():
        lst.sort(key=lambda e: e.created_at or "", reverse=True)
        if len(lst) > 1:
            print(f"Warning: {len(lst)} result files for suite {suite_name!r}; using newest.")
        chosen[suite_name] = lst[0]
    return chosen


def get_items(exp: SimpleNamespace) -> list[SimpleNamespace]:
    """Load items for a local experiment (results file)."""
    path = Path(exp.id)
    if not path.exists():
        return []
    return _load_results_file(path)


def has_tag(exp: SimpleNamespace, tag: str) -> bool:
    """Check if a local experiment matches a tag.

    Tags are read from the ``_metadata`` entry in the results JSON.  Falls
    back to substring match on the file path **only** for files without
    metadata (written before metadata support was added).
    """
    if tag in (exp.tags or []):
        return True
    # Only fall back to substring match for files without metadata —
    # otherwise "smoke" matches the directory name, "eu" matches the
    # env suffix, etc.
    if not exp.tags and not (exp.metadata or {}).get("_metadata"):
        return tag in exp.id
    return False
