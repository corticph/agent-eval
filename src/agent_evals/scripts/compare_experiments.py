"""Compare Opik experiments and find the evals with the biggest score differences.

Designed for comparing runs of ``run_all_evals.sh`` across environments. Fetches
per-item feedback scores from experiments, matches items by case name, and
shows the items with the largest score delta — defaulting to the ``overall``
feedback score but supporting any per-item score (``must_include``,
``expected_state``, ``judge``, etc.).

Two modes:

1. **Direct IDs** — pass two experiment IDs you found in the Opik UI::

       uv run python -m agent_evals.scripts.compare_experiments \\
           --exp1 <id1> --exp2 <id2>

2. **Discover by name + environment** — finds all experiments matching a name
   substring for each environment, pairs them by experiment name, and compares
   every matched pair. This is the mode for comparing ``run_all_evals.sh``
   sweeps::

        uv run python -m agent_evals.scripts.compare_experiments \\
            --name <suite-prefix> --env1 staging-eu --env2 local

    Or compare by tag (using ``--tag`` with ``run_all_evals.sh``)::

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
"""

from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import dotenv
import opik

from ..environment import OPIK_URL_OVERRIDE_VAR
from ..reporting.opik_target import resolve_opik_url

_REPO_ROOT = Path(__file__).resolve().parents[3]
dotenv.load_dotenv(_REPO_ROOT / ".env")

DEFAULT_SCORE = "overall"
DEFAULT_N = 20
DEFAULT_PROJECT = "Agents"

_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0


def _make_client() -> opik.Opik:
    """Create an Opik client using the same URL resolution as the eval harness.

    ``resolve_opik_url`` honours ``OPIK_URL_OVERRIDE`` (pinging first) or starts
    a kubectl port-forward to dev-weu — identical to what the Opik sink does at
    run time. Exporting the resolved URL to the environment ensures the SDK's
    global client (used by ``get_experiment_by_id``) routes to the same Opik.
    """
    url = resolve_opik_url()
    os.environ.setdefault(OPIK_URL_OVERRIDE_VAR, url)
    project = os.environ.get("OPIK_PROJECT_NAME") or DEFAULT_PROJECT
    os.environ["OPIK_PROJECT_NAME"] = project
    return opik.Opik(
        host=url,
        workspace="default",
        api_key=os.environ.get("OPIK_API_KEY"),
    )


# --- retry helper (tunnel is flaky under load) -------------------------------


def _retry(fn, *, what: str):
    """Retry a callable up to _MAX_RETRIES times with exponential backoff.

    The kubectl port-forward tunnel drops connections under concurrent load;
    a brief wait + retry is enough to recover.
    """
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == _MAX_RETRIES:
                raise
            wait = _RETRY_BACKOFF * attempt
            print(f"  retry {attempt}/{_MAX_RETRIES} ({what}): {exc} — waiting {wait:.0f}s")
            time.sleep(wait)


# --- experiment discovery ----------------------------------------------------


def _list_experiments(
    client: opik.Opik,
    name: str | None = None,
    limit: int = 500,
) -> list[SimpleNamespace]:
    """List experiments from Opik, optionally filtered by name substring.

    Returns lightweight rows with ``id``, ``name``, ``dataset_name``, ``tags``,
    ``created_at``, and ``metadata`` (which carries ``suite`` and
    ``environment`` from the experiment config).
    """
    http = client._rest_client._client_wrapper.httpx_client

    rows: list[SimpleNamespace] = []
    page, size = 1, 100
    while len(rows) < limit:
        resp = _retry(
            lambda: http.request(
                "v1/private/experiments",
                method="GET",
                params={
                    "page": page,
                    "size": size,
                    "name": name,
                    "dataset_deleted": False,
                },
            ),
            what=f"list experiments page {page}",
        )
        resp.raise_for_status()
        content = (resp.json() or {}).get("content") or []
        if not content:
            break
        for d in content:
            rows.append(
                SimpleNamespace(
                    id=d.get("id"),
                    name=d.get("name"),
                    dataset_name=d.get("dataset_name"),
                    tags=d.get("tags") or [],
                    created_at=d.get("created_at"),
                    metadata=d.get("metadata") or {},
                )
            )
            if len(rows) >= limit:
                break
        if len(content) < size:
            break
        page += 1

    rows.sort(key=lambda e: e.created_at or "", reverse=True)
    return rows


def _env_of(exp: SimpleNamespace) -> str | None:
    """Extract the ``environment`` from an experiment's config metadata."""
    meta = exp.metadata or {}
    if isinstance(meta, dict):
        return meta.get("environment")
    return None


def _has_tag(exp: SimpleNamespace, tag: str) -> bool:
    """Check if an experiment carries a given tag."""
    return tag in (exp.tags or [])


def _find_by_selector(
    client: opik.Opik,
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
    exps = _list_experiments(client, name=name, limit=limit)
    if env is not None:
        exps = [e for e in exps if _env_of(e) == env]
    if tag is not None:
        exps = [e for e in exps if _has_tag(e, tag)]
    if not exps:
        parts = [f"name={name!r}"]
        if env:
            parts.append(f"env={env!r}")
        if tag:
            parts.append(f"tag={tag!r}")
        raise SystemExit(f"{label}: no experiments found for {' '.join(parts)}.")

    by_name: dict[str, list[SimpleNamespace]] = defaultdict(list)
    for e in exps:
        by_name[e.name].append(e)

    chosen: dict[str, SimpleNamespace] = {}
    selector_desc = []
    if env:
        selector_desc.append(f"env={env!r}")
    if tag:
        selector_desc.append(f"tag={tag!r}")
    sel_str = " ".join(selector_desc)

    for exp_name, lst in by_name.items():
        lst.sort(key=lambda e: e.created_at or "", reverse=True)
        if len(lst) > 1:
            print(
                f"Warning: {label}: {len(lst)} experiments named "
                f"{exp_name!r} ({sel_str}); using newest (id={lst[0].id})."
            )
        chosen[exp_name] = lst[0]
    return chosen


# --- per-item helpers --------------------------------------------------------


def _get_items(client: opik.Opik, experiment_id: str) -> list:
    """Fetch all experiment items (with feedback scores) by experiment id."""
    return _retry(
        lambda: client.get_experiment_by_id(experiment_id).get_items(),
        what=f"get items {experiment_id[:8]}",
    )


def _score_map(item) -> dict[str, tuple[float, str]]:
    """Build ``{name: (value, reason)}`` from an item's feedback scores."""
    out: dict[str, tuple[float, str]] = {}
    for fs in item.feedback_scores or []:
        out[fs["name"]] = (fs["value"], fs.get("reason") or "")
    return out


def _case_name(item) -> str:
    """Extract the case name from an item's dataset data."""
    data = item.dataset_item_data or {}
    return data.get("name") or "(unnamed)"


def _trace_url(item) -> str | None:
    """Extract the trace URL from an item's evaluation task output."""
    out = item.evaluation_task_output or {}
    return out.get("trace_url")


def _available_scores(item) -> list[str]:
    """List the feedback score names available on an item."""
    return sorted(fs["name"] for fs in item.feedback_scores or [])


# --- comparison --------------------------------------------------------------


def _build_rows(
    client: opik.Opik,
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
        items1 = _get_items(client, e1.id)
        items2 = _get_items(client, e2.id)
        by_name1 = {_case_name(it): it for it in items1}
        by_name2 = {_case_name(it): it for it in items2}

        for case_name in sorted(set(by_name1) | set(by_name2)):
            it1 = by_name1.get(case_name)
            it2 = by_name2.get(case_name)
            sm1 = _score_map(it1) if it1 else {}
            sm2 = _score_map(it2) if it2 else {}

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
                    trace1=_trace_url(it1) if it1 else None,
                    trace2=_trace_url(it2) if it2 else None,
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
    client: opik.Opik,
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
    rows, only1, only2 = _build_rows(client, {label1: exp1}, {label2: exp2}, score)
    _sort_rows(rows, sort_mode)
    _print_summary(rows, score, sort_mode, only1, only2, label1, label2)
    _print_detail(rows, n, show_reason, show_trace)

    if not rows:
        items = _get_items(client, exp_id_1)
        if items:
            print(
                f"No items with score {score!r}; available scores: "
                f"{_available_scores(items[0])}"
            )


# --- listing -----------------------------------------------------------------


def _print_list(client: opik.Opik, name: str | None, limit: int) -> None:
    exps = _list_experiments(client, name=name, limit=limit)
    if not exps:
        print("No experiments found.")
        return
    print(f"{'name':<55} {'env':<12} {'tags':<25} {'id':<40} {'created':<20}")
    print("-" * 155)
    for e in exps:
        env = _env_of(e) or "-"
        tags = ",".join(e.tags or [])
        print(
            f"{(e.name or '?')[:55]:<55} {env[:12]:<12} {tags[:25]:<25} {e.id:<40} "
            f"{(e.created_at or '')[:19]:<20}"
        )


# --- main --------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Opik experiments and find evals with the biggest score differences."
    )

    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--exp1",
        default=None,
        help="Experiment 1 ID (direct mode).",
    )
    grp.add_argument(
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
    args = parser.parse_args()

    client = _make_client()

    if args.list:
        _print_list(client, args.name, args.limit)
        return

    # Direct mode: two experiment IDs
    if args.exp1 and args.exp2:
        label1, label2 = args.exp1, args.exp2
        print(f"exp1 = {label1}")
        print(f"exp2 = {label2}")
        _compare_single(
            client,
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
    if args.name:
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
            client, args.name, env=args.env1, tag=args.tag1,
            label="side1", limit=args.limit,
        )
        exps2 = _find_by_selector(
            client, args.name, env=args.env2, tag=args.tag2,
            label="side2", limit=args.limit,
        )
        matched = sorted(set(exps1) & set(exps2))
        print(
            f"\nside1 = {label1}  ({len(exps1)} experiments)"
            f"    side2 = {label2}  ({len(exps2)} experiments)"
            f"    matched: {len(matched)}"
        )
        rows, only1, only2 = _build_rows(client, exps1, exps2, args.score)
        _sort_rows(rows, args.sort)
        _print_summary(rows, args.score, args.sort, only1, only2, label1, label2)
        _print_detail(rows, args.n, args.show_reason, args.show_trace)

        if not rows:
            # Grab a sample item from the first matched experiment to show
            # available scores
            if matched:
                sample = _get_items(client, exps1[matched[0]].id)
                if sample:
                    print(
                        f"No items with score {args.score!r}; available scores: "
                        f"{_available_scores(sample[0])}"
                    )
        return

    parser.error(
        "Provide --exp1 and --exp2 (direct mode) or --name with --env1/--tag1 and --env2/--tag2 "
        "(discovery mode)."
    )


if __name__ == "__main__":
    main()
