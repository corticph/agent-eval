"""Find suites missing from Opik for a given tag.

Used by ``run_all_evals.sh --resume`` to re-run only the suites that didn't
create an Opik experiment — i.e. suites that failed before ``close()``
(tunnel drops, rate limiting, crashes).  Suites whose evals ran but failed
still have an experiment in Opik, so they are *not* resumed.

Usage::

    uv run python -m agent_evals.scripts.resume_missing \\
        --tag eu-20260907-120000 --env eu --evals-dir ./evals

Prints one suite file path per line to stdout.  Nothing on stdout means
every suite has an experiment in Opik (nothing to resume).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from .compare_experiments import _env_of, _has_tag, _list_experiments, _make_client


def _suite_name(suite_path: Path) -> str:
    """Extract the experiment name from a suite YAML file.

    Mirrors ``load_suite``: the ``name:`` field if present, else the file stem.
    Only reads the top-level ``name`` — no connector / data-file resolution.
    """
    with suite_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return raw.get("name", suite_path.stem)


def _discover_suites(evals_dir: Path, suite_filters: list[str]) -> list[Path]:
    """Discover all suite YAML files, mirroring run_all_evals.sh discovery."""
    all_suites = sorted(
        p
        for p in evals_dir.rglob("*.yaml")
        if not p.name.startswith("_")
        and not p.name.endswith("_local.yaml")
        and p.stat().st_size > 0
    )
    if suite_filters:
        all_suites = [
            p for p in all_suites if any(pat in str(p) for pat in suite_filters)
        ]
    return all_suites


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find suites missing from Opik for a given tag."
    )
    parser.add_argument("--tag", required=True, help="Opik tag to check.")
    parser.add_argument(
        "--evals-dir", required=True, help="Directory of suite YAML files."
    )
    parser.add_argument(
        "--env", default=None, help="Environment filter (optional, for extra safety)."
    )
    parser.add_argument(
        "--suite",
        action="append",
        default=[],
        help="Substring filter for suite paths (repeatable).",
    )
    args = parser.parse_args()

    evals_dir = Path(args.evals_dir)
    if not evals_dir.is_dir():
        print(f"Error: {evals_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    all_suites = _discover_suites(evals_dir, args.suite)
    if not all_suites:
        print("Error: No suite files found.", file=sys.stderr)
        sys.exit(1)

    name_to_path: dict[str, str] = {}
    for p in all_suites:
        name_to_path[_suite_name(p)] = str(p)

    client = _make_client()
    exps = _list_experiments(client, limit=1000)
    exps = [e for e in exps if _has_tag(e, args.tag)]
    if args.env:
        exps = [e for e in exps if _env_of(e) == args.env]

    found_names = {e.name for e in exps}
    missing = [
        (name, path)
        for name, path in sorted(name_to_path.items())
        if name not in found_names
    ]

    if missing:
        for _name, path in missing:
            print(path)
    else:
        print(
            f"All {len(all_suites)} suites have experiments in Opik "
            f"for tag {args.tag!r}.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
