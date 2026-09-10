"""Inspect a single experiment item: input, output, and expectations.

Designed as a follow-up to ``compare_experiments``: once you've found a case
with a score delta, use this to see the full input (what was sent to the
agent), the output (what the agent responded), and the per-expectation
verdicts (why each check passed or failed).

Works with both Opik (default) and local JSON results (``--source local``).

Usage:
    # By experiment id + case name (from the compare report):
    uv run python -m agent_evals.scripts.inspect_eval \\
        --exp <experiment-id> --case mixed_narrative_1

    # With local source, --exp is a file path:
    uv run python -m agent_evals.scripts.inspect_eval \\
        --source local --exp results/smoke/hello.json --case my_case

    # List all case names in an experiment:
    uv run python -m agent_evals.scripts.inspect_eval \\
        --exp <experiment-id> --list

    # Show only the expectations + verdicts (skip the full response text):
    uv run python -m agent_evals.scripts.inspect_eval \\
        --exp <experiment-id> --case mixed_narrative_1 --expectations-only
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import dotenv

from .data_source import DataSource, make_source

_REPO_ROOT = Path(__file__).resolve().parents[3]
dotenv.load_dotenv(_REPO_ROOT / ".env")


# --- fetching -----------------------------------------------------------------


def _find_item(items: list, case_name: str):
    """Find an item by case name (exact match)."""
    for item in items:
        data = item.dataset_item_data or {}
        name = data.get("name") or "(unnamed)"
        if name == case_name:
            return item
    return None


# --- rendering ----------------------------------------------------------------


def _section(title: str) -> None:
    print(f"\n{'─' * 3} {title} {'─' * (75 - len(title))}")


def _print_json(data, indent: int = 2, max_len: int | None = None) -> None:
    """Pretty-print JSON, optionally truncated."""
    text = json.dumps(data, indent=indent, ensure_ascii=False)
    if max_len and len(text) > max_len:
        text = text[:max_len] + "\n... (truncated)"
    print(text)


def _print_input(item, show_agent: bool = False) -> None:
    """Print the dataset item (the input sent to the agent).

    The agent config can be very large, so it's hidden by default —
    pass ``show_agent=True`` to include it. Messages and expectations are
    always shown (messages are truncated, expectations are compact).
    """
    data = item.dataset_item_data or {}
    _section("INPUT — dataset item")
    print(f"  case name: {data.get('name', '(unnamed)')}")

    if show_agent:
        agent = data.get("agent")
        if agent:
            _section("agent")
            _print_json(agent)
    elif data.get("agent"):
        agent_name = (data.get("agent") or {}).get("name", "?")
        print(f"  agent: {agent_name}  (use --show-agent for full config)")

    steps = data.get("steps") or []
    for i, step in enumerate(steps):
        step_name = step.get("name") or f"step {i + 1}"
        _section(f"step: {step_name}")

        msg = step.get("message") or {}
        _section("  message")
        _print_json(msg, max_len=2000)

        expectations = step.get("expectations") or {}
        if expectations:
            _section("  expectations declared")
            _print_json(expectations)

        if step.get("delay_before_seconds") is not None:
            print(f"  delay_before_seconds: {step['delay_before_seconds']}")


def _print_output(item) -> None:
    """Print the evaluation task output (the agent's responses + verdicts)."""
    out = item.evaluation_task_output or {}
    _section("OUTPUT — task result")

    if out.get("error"):
        print(f"  CASE ERROR: {out['error']}")

    if out.get("trace_url"):
        print(f"  trace: {out['trace_url']}")

    if out.get("environment"):
        print(f"  environment: {out['environment']}")

    usage = out.get("usage")
    if usage:
        print(f"  usage: {json.dumps(usage)}")

    step_results = out.get("step_results") or []
    for i, sr in enumerate(step_results):
        step_name = sr.get("name") or f"step {i + 1}"
        _section(f"step result: {step_name}")
        success = sr.get("success")
        print(f"  success: {success}")
        state = sr.get("response_state")
        if state:
            print(f"  response_state: {state}")
        duration = sr.get("duration_seconds")
        if duration is not None:
            print(f"  duration: {duration:.3f}s")

        response_text = sr.get("response_text")
        if response_text:
            _section("  response text")
            wrapped = textwrap.indent(
                textwrap.fill(response_text, width=100), "  "
            )
            print(wrapped)

        exp_results = sr.get("expectation_results") or []
        if exp_results:
            _section("  expectation verdicts")
            for er in exp_results:
                key = er.get("key", "?")
                checks = er.get("checks") or []
                all_pass = all(c.get("passed") for c in checks)
                icon = "PASS" if all_pass else "FAIL"
                print(f"    [{icon}] {key}")
                for check in checks:
                    label = check.get("label", "")
                    passed = check.get("passed", False)
                    detail = check.get("detail", "")
                    mark = "+" if passed else "x"
                    if label:
                        print(f"      {mark} {label}: {detail}")
                    else:
                        print(f"      {mark} {detail}")


def _print_scores(item) -> None:
    """Print the feedback scores (the overall + per-expectation-type scores)."""
    scores = item.feedback_scores or []
    if not scores:
        return
    _section("FEEDBACK SCORES")
    for fs in scores:
        name = fs.get("name", "?")
        value = fs.get("value", 0)
        reason = fs.get("reason") or ""
        print(f"  {name:<30} {value:>8.4f}")
        if reason:
            wrapped = textwrap.indent(
                textwrap.fill(reason, width=100), "    "
            )
            print(wrapped)


# --- main --------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a single experiment item: input, output, and expectations."
    )
    parser.add_argument("--exp", required=True, help="Experiment ID (Opik) or file path (local).")
    parser.add_argument("--source", default="opik", choices=("opik", "local"), help="Data source (default: opik).")
    parser.add_argument("--results-dir", action="append", default=[], help="Path to results directory for local source (repeatable, default: results/).")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--case", default=None, help="Case name to inspect.")
    grp.add_argument("--list", action="store_true", help="List all case names and exit.")
    parser.add_argument(
        "--expectations-only",
        action="store_true",
        help="Show only expectations + verdicts (skip full input/output).",
    )
    parser.add_argument(
        "--show-agent",
        action="store_true",
        help="Include the full agent config in the input section (very large).",
    )
    args = parser.parse_args()

    source = make_source(args.source, results_dir=args.results_dir or None)
    exp = type("Exp", (), {"id": args.exp})()
    items = source.get_items(exp)

    if not items:
        print("No items found in this experiment.")
        return

    if args.list:
        print(f"{'case name':<50} {'overall':>8}")
        print("-" * 60)
        for item in items:
            name = source.case_name(item)
            sm = source.score_map(item)
            overall = sm.get("overall")
            v_str = f"{overall[0]:.3f}" if overall else "-"
            print(f"{name[:50]:<50} {v_str:>8}")
        return

    if not args.case:
        parser.error("Provide --case <name> or --list.")

    item = _find_item(items, args.case)
    if item is None:
        print(f"Case {args.case!r} not found. Available:")
        for it in items:
            print(f"  {source.case_name(it)}")
        return

    if not args.expectations_only:
        _print_input(item, show_agent=args.show_agent)
        _print_output(item)
    else:
        # In expectations-only mode, still show step names + verdicts
        out = item.evaluation_task_output or {}
        step_results = out.get("step_results") or []
        for i, sr in enumerate(step_results):
            step_name = sr.get("name") or f"step {i + 1}"
            _section(f"step: {step_name}")
            exp_results = sr.get("expectation_results") or []
            for er in exp_results:
                key = er.get("key", "?")
                checks = er.get("checks") or []
                all_pass = all(c.get("passed") for c in checks)
                icon = "PASS" if all_pass else "FAIL"
                print(f"  [{icon}] {key}")
                for check in checks:
                    label = check.get("label", "")
                    passed = check.get("passed", False)
                    detail = check.get("detail", "")
                    mark = "+" if passed else "x"
                    if label:
                        print(f"    {mark} {label}: {detail}")
                    else:
                        print(f"    {mark} {detail}")

    _print_scores(item)


if __name__ == "__main__":
    main()
