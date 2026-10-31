"""Find suites missing from a given tag.

Used by ``agent-evals sweep --resume`` to re-run only the suites that didn't
create an experiment — i.e. suites that failed before ``close()``
(tunnel drops, rate limiting, crashes).  Suites whose evals ran but failed
still have an experiment, so they are *not* resumed.

Checks local results first (local-cache-first); falls back to the Opik
API when no local results match the tag.

Usage::

    uv run python -m agent_evals.scripts.resume_missing \\
        --tag eu-20260907-120000 --env eu --evals-dir ./evals
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from .data_source import make_source


def _suite_name(suite_path: Path) -> str:
    """Extract the experiment name from a suite YAML file."""
    with suite_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return raw.get("name", suite_path.stem)


def _discover_suites(evals_dir: Path, suite_filters: list[str]) -> list[Path]:
    """Discover all suite YAML files."""
    from .sweep import _discover_suites as _discover

    return _discover(evals_dir, suite_filters)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find suites missing from a given tag."
    )
    parser.add_argument("--tag", required=True, help="Tag to check.")
    parser.add_argument(
        "--evals-dir", required=True, help="Directory of suite YAML files."
    )
    parser.add_argument(
        "--env", default=None, help="Environment filter (optional)."
    )
    parser.add_argument(
        "--suite",
        action="append",
        default=[],
        help="Substring filter for suite paths (repeatable).",
    )
    parser.add_argument(
        "--source", default="opik", choices=("opik", "local"),
        help="Data source (default: opik with local-cache-first).",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Path to results directory for local source.",
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

    source = make_source(args.source, results_dir=args.results_dir)
    found = source.list_experiments(tag=args.tag, env=args.env, limit=5000)

    found_names = set(found.keys())
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
            f"All {len(all_suites)} suites have experiments "
            f"for tag {args.tag!r}.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
