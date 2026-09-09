"""Compare experiments and find the evals with the biggest score differences.

Designed for comparing runs of ``run_all_evals.sh`` across environments or
tags. Fetches per-item feedback scores from experiments, matches items by
case name, and shows the items with the largest score delta — defaulting to
the ``overall`` feedback score but supporting any per-item score (``must_include``,
``expected_state``, ``judge``, etc.).

Works with both Opik (default) and local JSON results (``--source local``).

Two modes:

1. **Direct IDs** — pass two experiment IDs::

        uv run python -m agent_evals.scripts.compare_experiments \\
            --exp1 <id1> --exp2 <id2>

    With local source, IDs are file paths (e.g. ``results/smoke/hello.json``).

2. **Discover by name + environment** — finds all experiments matching a name
   substring for each environment, pairs them by experiment name, and compares
   every matched pair::

        uv run python -m agent_evals.scripts.compare_experiments \\
            --name <suite-prefix> --env1 staging-eu --env2 local

    Or compare by tag::

        uv run python -m agent_evals.scripts.compare_experiments \\
            --name <suite-prefix> --tag1 release-v1.1 --tag2 release-v1.2

    Use ``--list`` to discover available experiments without comparing::

        uv run python -m agent_evals.scripts.compare_experiments --list --name <suite-prefix>

Options:
    --score       Feedback score to compare (default: overall)
    --n           Number of items to show (default: 20)
    --sort        delta (default) | regression | improvement
    --show-reason Show the score reason (failure details) for each side
    --show-trace  Show trace URLs
    --source      opik (default) or local (results/*.json)
    --results-dir Path to results directory (default: results/)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import dotenv

from .data_source import DataSource, make_source

_REPO_ROOT = Path(__file__).resolve().parents[3]
dotenv.load_dotenv(_REPO_ROOT / ".env")

DEFAULT_SCORE = "overall"
DEFAULT_N = 20


def _find_by_selector(
    source: DataSource,
    name: str | None,
    *,
    env: str | None = None,
    tag: str | None = None,
    label: str,
    limit: int,
) -> dict[str, SimpleNamespace]:
    """Find all experiments matching name + (env and/or tag), grouped by name.

    At least one of ``env`` or ``tag`` must be provided. When both are given,
    experiments must match both. When multiple experiments share the same name
    (e.g. a re-run), the newest is kept (with a warning).
    Returns ``{experiment_name: exp_row}``.
    """
    exps = source.list_experiments(name=name, env=env, tag=tag, limit=limit)
    if not exps:
        parts = [f"name={name!r}"]
        if env:
            parts.append(f"env={env!r}")
        if tag:
            parts.append(f"tag={tag!r}")
        raise SystemExit(f"{label}: no experiments found for {' '.join(parts)}.")
    return dict(exps)


# --- per-item helpers (moved to DataSource) ----------------------------------

# --- comparison --------------------------------------------------------------


def _build_rows(
    source: DataSource,
    exps1: dict[str, SimpleNamespace],
    exps2: dict[str, SimpleNamespace],
    score: str,
) -> tuple[list[SimpleNamespace], set[str], set[str]]:
    """Fetch items for every matched experiment pair and build comparison rows.

    Each row carries the experiment (suite) name and the case name so the user
    can see which suite a delta comes from. Returns ``(rows, only1_names,
    only2_names)`` where the ``only`` sets are experiment names present on only
    one side.
    """
    matched_names = sorted(set(exps1) & set(exps2))
    rows: list[SimpleNamespace] = []

    for exp_name in matched_names:
        e1 = exps1[exp_name]
        e2 = exps2[exp_name]
        items1 = source.get_items(e1)
        items2 = source.get_items(e2)
        by_name1 = {source.case_name(it): it for it in items1}
        by_name2 = {source.case_name(it): it for it in items2}

        for case_name in sorted(set(by_name1) | set(by_name2)):
            it1 = by_name1.get(case_name)
            it2 = by_name2.get(case_name)
            sm1 = source.score_map(it1) if it1 else {}
            sm2 = source.score_map(it2) if it2 else {}

            if score not in sm1 and score not in sm2:
                continue

            v1, r1 = sm1.get(score, (None, ""))
            v2, r2 = sm2.get(score, (None, ""))
            delta = (v2 - v1) if v1 is not None and v2 is not None else None

            rows.append(
                SimpleNamespace(
                    exp_name=exp_name,
                    exp_id1=e1.id,
                    exp_id2=e2.id,
                    case_name=case_name,
                    v1=v1,
                    v2=v2,
                    delta=delta,
                    reason1=r1,
                    reason2=r2,
                    trace1=source.trace_url(it1) if it1 else None,
                    trace2=source.trace_url(it2) if it2 else None,
                )
            )

    return rows, set(exps1) - set(exps2), set(exps2) - set(exps1)


def _sort_rows(rows: list[SimpleNamespace], sort_mode: str) -> None:
    """Sort rows in-place by the chosen mode."""
    if sort_mode == "regression":
        rows.sort(key=lambda r: r.delta if r.delta is not None else 1)
    elif sort_mode == "improvement":
        rows.sort(key=lambda r: -(r.delta if r.delta is not None else -1))
    else:
        rows.sort(key=lambda r: -(abs(r.delta) if r.delta is not None else 0))


def _print_summary(
    rows: list[SimpleNamespace],
    score: str,
    sort_mode: str,
    only1: set[str],
    only2: set[str],
    label1: str,
    label2: str,
) -> None:
    """Print the summary header: mean scores, improved/regressed/unchanged, and
    experiments present on only one side."""
    scored = [r for r in rows if r.delta is not None]
    exp_count = len({r.exp_name for r in rows})
    print(f"\n{'=' * 80}")
    print(
        f"Comparison: score={score!r}  sort={sort_mode}  "
        f"experiments={exp_count}  items={len(rows)}"
    )
    if scored:
        mean1 = sum(r.v1 for r in scored) / len(scored)
        mean2 = sum(r.v2 for r in scored) / len(scored)
        regressed = sum(1 for r in scored if r.delta < 0)
        improved = sum(1 for r in scored if r.delta > 0)
        unchanged = sum(1 for r in scored if r.delta == 0)
        print(
            f"  mean {score}: {mean1:.3f} -> {mean2:.3f}  "
            f"({len(scored)} matched: {improved} improved, {regressed} regressed, "
            f"{unchanged} unchanged)"
        )
    if only1:
        print(
            f"  experiments only in {label1} ({len(only1)}): "
            f"{', '.join(sorted(only1)[:10])}"
        )
    if only2:
        print(
            f"  experiments only in {label2} ({len(only2)}): "
            f"{', '.join(sorted(only2)[:10])}"
        )
    print(f"{'=' * 80}\n")


def _print_detail(
    rows: list[SimpleNamespace],
    n: int,
    show_reason: bool,
    show_trace: bool,
) -> None:
    """Print the top-N rows with the biggest deltas."""
    shown = rows[:n]
    for r in shown:
        v1_str = f"{r.v1:.3f}" if r.v1 is not None else "  -  "
        v2_str = f"{r.v2:.3f}" if r.v2 is not None else "  -  "
        delta_str = f"{r.delta:+.3f}" if r.delta is not None else "  -  "
        print(f"{r.exp_name} / {r.case_name}")
        print(f"  {v1_str}  ->  {v2_str}   delta={delta_str}")
        print(f"  exp1={r.exp_id1}  exp2={r.exp_id2}  case={r.case_name}")
        if show_reason:
            if r.reason1:
                print(f"  reason1: {_truncate_reason(r.reason1)}")
            if r.reason2:
                print(f"  reason2: {_truncate_reason(r.reason2)}")
        if show_trace:
            if r.trace1:
                print(f"  trace1:  {r.trace1}")
            if r.trace2:
                print(f"  trace2:  {r.trace2}")
        print()

    if not rows:
        print("No matching items found.")


def _truncate_reason(reason: str, max_len: int = 500) -> str:
    """Truncate a score reason to a readable length."""
    if len(reason) <= max_len:
        return reason
    return reason[:max_len] + "..."


# --- single-pair comparison (direct IDs) -------------------------------------


def _compare_single(
    source: DataSource,
    exp_id_1: str,
    exp_id_2: str,
    score: str,
    n: int,
    sort_mode: str,
    show_reason: bool,
    show_trace: bool,
    label1: str,
    label2: str,
) -> None:
    """Compare a single pair of experiment IDs."""
    exp1 = SimpleNamespace(id=exp_id_1, name=label1)
    exp2 = SimpleNamespace(id=exp_id_2, name=label2)
    rows, only1, only2 = _build_rows(source, {label1: exp1}, {label2: exp2}, score)
    _sort_rows(rows, sort_mode)
    _print_summary(rows, score, sort_mode, only1, only2, label1, label2)
    _print_detail(rows, n, show_reason, show_trace)

    if not rows:
        items = source.get_items(exp1)
        if items:
            print(
                f"No items with score {score!r}; available scores: "
                f"{source.available_scores(items[0])}"
            )


# --- listing -----------------------------------------------------------------


def _print_list(source: DataSource, name: str | None, limit: int) -> None:
    exps = source.list_experiments(name=name, limit=limit)
    if not exps:
        print("No experiments found.")
        return
    print(f"{'name':<55} {'tags':<25} {'id':<50} {'created':<20}")
    print("-" * 155)
    for e_name, e in exps.items():
        tags = ",".join(e.tags or [])
        print(
            f"{(e.name or '?')[:55]:<55} {tags[:25]:<25} {(e.id or '?')[:50]:<50} "
            f"{(e.created_at or '')[:19]:<20}"
        )


# --- main --------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Opik experiments and find evals with the biggest score differences."
    )

    parser.add_argument(
        "--exp1",
        default=None,
        help="Experiment 1 ID (direct mode).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Name substring for discovery mode (matches all experiments with that prefix).",
    )

    parser.add_argument("--exp2", default=None, help="Experiment 2 ID (direct mode).")
    parser.add_argument("--env1", default=None, help="Environment for side 1 (discovery mode).")
    parser.add_argument("--env2", default=None, help="Environment for side 2 (discovery mode).")
    parser.add_argument("--tag1", default=None, help="Tag for side 1 (discovery mode, alternative to --env1).")
    parser.add_argument("--tag2", default=None, help="Tag for side 2 (discovery mode, alternative to --env2).")
    parser.add_argument(
        "--score",
        default=DEFAULT_SCORE,
        help=f"Feedback score to compare (default: {DEFAULT_SCORE}).",
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N, help=f"Items to show (default: {DEFAULT_N}).")
    parser.add_argument(
        "--sort",
        choices=("delta", "regression", "improvement"),
        default="delta",
        help="Sort order: delta=biggest absolute change (default), regression=worst first, improvement=best first.",
    )
    parser.add_argument("--show-reason", action="store_true", help="Show the score reason (failure details).")
    parser.add_argument("--no-trace", dest="show_trace", action="store_false", help="Hide trace URLs (shown by default).")
    parser.set_defaults(show_trace=True)
    parser.add_argument("--list", action="store_true", help="List experiments and exit (use --name to filter).")
    parser.add_argument("--limit", type=int, default=500, help="Discovery cap (default: 500).")
    parser.add_argument("--source", default="opik", choices=("opik", "local"), help="Data source: opik (default) or local (results/*.json).")
    parser.add_argument("--results-dir", action="append", default=[], help="Path to results directory for local source (repeatable, default: results/).")
    args = parser.parse_args()

    source = make_source(args.source, results_dir=args.results_dir or None)

    if args.list:
        _print_list(source, args.name, args.limit)
        return

    # Direct mode: two experiment IDs
    if args.exp1 and args.exp2:
        label1, label2 = args.exp1, args.exp2
        print(f"exp1 = {label1}")
        print(f"exp2 = {label2}")
        _compare_single(
            source,
            args.exp1,
            args.exp2,
            score=args.score,
            n=args.n,
            sort_mode=args.sort,
            show_reason=args.show_reason,
            show_trace=args.show_trace,
            label1=label1,
            label2=label2,
        )
        return

    # Discovery mode: name substring + two selectors (env and/or tag)
    if args.name is not None and args.name:
        if not (args.env1 or args.tag1) or not (args.env2 or args.tag2):
            raise SystemExit(
                "Discovery mode requires --env1/--tag1 and --env2/--tag2 "
                "(or use --exp1/--exp2 for direct mode)."
            )
        sel1_parts, sel2_parts = [], []
        if args.env1:
            sel1_parts.append(f"env={args.env1!r}")
        if args.tag1:
            sel1_parts.append(f"tag={args.tag1!r}")
        if args.env2:
            sel2_parts.append(f"env={args.env2!r}")
        if args.tag2:
            sel2_parts.append(f"tag={args.tag2!r}")
        label1 = f"{args.name!r} / {' '.join(sel1_parts)}"
        label2 = f"{args.name!r} / {' '.join(sel2_parts)}"
        exps1 = _find_by_selector(
            source, args.name, env=args.env1, tag=args.tag1,
            label="side1", limit=args.limit,
        )
        exps2 = _find_by_selector(
            source, args.name, env=args.env2, tag=args.tag2,
            label="side2", limit=args.limit,
        )
        matched = sorted(set(exps1) & set(exps2))
        print(
            f"\nside1 = {label1}  ({len(exps1)} experiments)"
            f"    side2 = {label2}  ({len(exps2)} experiments)"
            f"    matched: {len(matched)}"
        )
        rows, only1, only2 = _build_rows(source, exps1, exps2, args.score)
        _sort_rows(rows, args.sort)
        _print_summary(rows, args.score, args.sort, only1, only2, label1, label2)
        _print_detail(rows, args.n, args.show_reason, args.show_trace)

        if not rows:
            # Grab a sample item from the first matched experiment to show
            # available scores
            if matched:
                sample = source.get_items(exps1[matched[0]])
                if sample:
                    print(
                        f"No items with score {args.score!r}; available scores: "
                        f"{source.available_scores(sample[0])}"
                    )
        return

    # Tag-only mode: --tag1/--tag2 without --name (matches all experiments)
    if (args.tag1 or args.tag2) and args.name is None:
        if not (args.tag1) or not (args.tag2):
            raise SystemExit(
                "Tag-only mode requires both --tag1 and --tag2 "
                "(or use --name for discovery mode, or --exp1/--exp2 for direct mode)."
            )
        exps1 = _find_by_selector(
            source, name=None, tag=args.tag1,
            label="side1", limit=args.limit,
        )
        exps2 = _find_by_selector(
            source, name=None, tag=args.tag2,
            label="side2", limit=args.limit,
        )
        matched = sorted(set(exps1) & set(exps2))
        label1 = f"tag={args.tag1!r}"
        label2 = f"tag={args.tag2!r}"
        print(
            f"\nside1 = {label1}  ({len(exps1)} experiments)"
            f"    side2 = {label2}  ({len(exps2)} experiments)"
            f"    matched: {len(matched)}"
        )
        rows, only1, only2 = _build_rows(source, exps1, exps2, args.score)
        _sort_rows(rows, args.sort)
        _print_summary(rows, args.score, args.sort, only1, only2, label1, label2)
        _print_detail(rows, args.n, args.show_reason, args.show_trace)

        if not rows:
            if matched:
                sample = source.get_items(exps1[matched[0]])
                if sample:
                    print(
                        f"No items with score {args.score!r}; available scores: "
                        f"{source.available_scores(sample[0])}"
                    )
        return

    parser.error(
        "Provide one of: --exp1 and --exp2 (direct mode), --name with --tag1/--tag2 (discovery), "
        "or --tag1/--tag2 alone (tag-only mode)."
    )


if __name__ == "__main__":
    main()
