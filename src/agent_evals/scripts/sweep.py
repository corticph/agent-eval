"""Run every suite under an evals directory as a parallel sweep.

Replaces ``run_all_evals.sh``.  Discovers suite YAML files, pre-warms the
Opik tunnel, runs suites concurrently with retries, and supports resuming
failed suites from Opik or a file.

Usage::

    uv run agent-evals sweep --env local --tag "local-$(date +%Y%m%d-%H%M%S)"
    uv run agent-evals sweep --env staging-eu --suite smoke --jobs 1
    uv run agent-evals sweep --env eu --tag eu-20260907-120000 --resume
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import yaml

from ..environment import OPIK_URL_OVERRIDE_VAR, supported_environments
from ..reporting.opik_target import resolve_opik_url

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _discover_suites(evals_dir: Path, suite_filters: list[str]) -> list[Path]:
    """Find all suite YAML files under *evals_dir*, applying substring filters.

    Handles symlinked subdirectories (``rglob`` does not follow directory
    symlinks on some Python versions, so we glob within each child).
    """
    all_suites: list[Path] = []
    for p in evals_dir.rglob("*.yaml"):
        if not p.name.startswith("_") and not p.name.endswith("_local.yaml") and p.stat().st_size > 0:
            all_suites.append(p)
    # Also check immediate subdirectories that are symlinks (rglob skips them).
    for d in evals_dir.iterdir():
        if d.is_dir() and d.is_symlink():
            for p in d.rglob("*.yaml"):
                if not p.name.startswith("_") and not p.name.endswith("_local.yaml") and p.stat().st_size > 0:
                    if p not in all_suites:
                        all_suites.append(p)
    all_suites.sort()
    if suite_filters:
        all_suites = [
            p for p in all_suites if any(pat in str(p) for pat in suite_filters)
        ]
    return all_suites


def _locate_evals_dir(evals_dir_arg: str | None) -> Path:
    """Resolve the evals directory: explicit arg > ./evals > sibling checkout."""
    if evals_dir_arg:
        p = Path(evals_dir_arg)
        if not p.is_dir():
            raise SystemExit(f"--evals-dir '{evals_dir_arg}' does not exist or is not a directory.")
        return p
    local = _REPO_ROOT / "evals"
    if local.is_dir():
        return local
    sibling = _REPO_ROOT.parent / "agent-eval-cases" / "evals"
    if sibling.is_dir():
        return sibling
    raise SystemExit(
        "No evals/ directory found.\n"
        "This repo is the eval harness only. Use --evals-dir <path>, "
        "symlink evals/, or clone the cases repo as a sibling."
    )


def _suite_name(suite_path: Path) -> str:
    """Extract the experiment name from a suite YAML file."""
    with suite_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return raw.get("name", suite_path.stem)


def _resume_from_opik(
    tag: str,
    env: str,
    evals_dir: Path,
    suite_filters: list[str],
) -> list[Path]:
    """Query Opik for suites missing from *tag* and return their file paths."""
    from .resume_missing import _discover_suites as _discover
    from .compare_experiments import _env_of, _has_tag, _list_experiments, _make_client

    all_suites = _discover(evals_dir, suite_filters)
    if not all_suites:
        return []

    name_to_path = {_suite_name(p): p for p in all_suites}
    client = _make_client()
    exps = _list_experiments(client, limit=5000)
    exps = [e for e in exps if _has_tag(e, tag)]
    if env:
        exps = [e for e in exps if _env_of(e) == env]

    found_names = {e.name for e in exps}
    missing = [p for name, p in sorted(name_to_path.items()) if name not in found_names]
    return missing


def _resume_from_file(resume_file: Path) -> list[Path]:
    """Read suite paths from a file (one per line, ``#`` comments)."""
    suites: list[Path] = []
    for line in resume_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = Path(line)
        if not p.exists():
            print(f"Warning: suite file '{line}' no longer exists; skipping.", file=sys.stderr)
            continue
        suites.append(p)
    return suites


def _run_one_suite(
    suite_path: Path,
    env: str,
    extra_args: list[str],
    retries: int,
    *,
    use_opik: bool = True,
) -> tuple[str, int, str]:
    """Run a single suite via ``agent-evals run``, returning (name, exit_code, output).

    Retries on failure up to *retries* times with a 2s backoff.
    """
    cmd = ["uv", "run", "agent-evals", "run", str(suite_path), "--env", env]
    if use_opik:
        cmd.append("--opik")
    cmd += extra_args
    rel = suite_path.name
    max_attempts = retries + 1
    last_output = ""
    for attempt in range(1, max_attempts + 1):
        attempt_label = f" attempt {attempt}/{max_attempts}" if max_attempts > 1 else ""
        prefix = f"[agent-evals] Running {rel}{attempt_label}\n"
        result = subprocess.run(cmd, capture_output=True, text=True)
        last_output = prefix + result.stdout + result.stderr
        if result.returncode == 0:
            return rel, 0, last_output
        if attempt < max_attempts:
            time.sleep(2)
    return rel, result.returncode, last_output


def _prewarm_tunnel(use_opik: bool = True) -> str | None:
    """Pre-warm the Opik tunnel so parallel suites don't race to bind the port.

    Returns the resolved URL, or None if the tunnel couldn't start (suites
    will retry individually).  Skipped entirely when Opik is not in use.
    """
    if not use_opik:
        return None
    if os.environ.get(OPIK_URL_OVERRIDE_VAR):
        return os.environ[OPIK_URL_OVERRIDE_VAR]
    try:
        url = resolve_opik_url()
        print(f"Opik tunnel ready at {url}")
        return url
    except Exception as exc:
        print(f"Warning: Opik tunnel did not become ready: {exc}", file=sys.stderr)
        print("Suites will retry individually.", file=sys.stderr)
        return None


def run_sweep(
    *,
    env: str,
    evals_dir: str | None = None,
    suite_filters: list[str] | None = None,
    tags: list[str] | None = None,
    jobs: int = 10,
    retries: int = 2,
    resume: bool = False,
    resume_file: str | None = None,
    model: str | None = None,
    extra_args: list[str] | None = None,
    verbose: int = 0,
    use_opik: bool = True,
) -> int:
    """Run a sweep — the core entry point called by ``agent-evals sweep``."""
    suite_filters = suite_filters or []
    tags = tags or []
    extra_args = extra_args or []

    supported = supported_environments()
    if env not in supported:
        raise SystemExit(f"Invalid environment '{env}'. Supported: {', '.join(supported)}")

    evals_dir_path = _locate_evals_dir(evals_dir)

    # --- build the suite list (resume or discover) ---
    if resume:
        if resume_file:
            suites = _resume_from_file(Path(resume_file))
            if not suites:
                print("Nothing to resume: file contains no suite paths.")
                return 0
            print(f"Resuming {len(suites)} suite(s) from {resume_file}")
        else:
            if not tags:
                raise SystemExit("--resume requires --tag (to find the Opik experiments).")
            resume_tag = tags[0]
            print(f"Querying Opik for suites missing from tag '{resume_tag}'...")
            suites = _resume_from_opik(resume_tag, env, evals_dir_path, suite_filters)
            if not suites:
                print(f"Nothing to resume: all suites have experiments in Opik for tag '{resume_tag}'.")
                return 0
            print(f"Resuming {len(suites)} missing suite(s) from Opik tag '{resume_tag}'")
    else:
        suites = _discover_suites(evals_dir_path, suite_filters)
        if not suites:
            if suite_filters:
                raise SystemExit(f"No suite files matched the --suite filter(s): {suite_filters}")
            raise SystemExit(f"No suite files found under {evals_dir_path}")

    if verbose:
        print(f"Suites to run ({len(suites)}):")
        for p in suites:
            print(f"  {p}")

    # --- build extra args for agent-evals run ---
    run_extra_args: list[str] = []
    if model:
        run_extra_args += ["--model", model]
    # Tags are stored in local JSON metadata regardless of Opik.
    for tag in tags:
        run_extra_args += ["--tag", tag]
    run_extra_args += extra_args

    # --- pre-warm the Opik tunnel ---
    _prewarm_tunnel(use_opik=use_opik)

    # --- run suites ---
    print(f"Running {len(suites)} suite(s) against {env} (jobs={jobs}, retries={retries})")

    if jobs <= 1:
        # Sequential: capture output per-suite and print when it completes.
        failed = 0
        for suite_path in suites:
            name, rc, output = _run_one_suite(suite_path, env, run_extra_args, retries, use_opik=use_opik)
            print(output, end="")
            if rc != 0:
                failed += 1
        _print_summary(env, jobs, len(suites), failed, tags)
        return 1 if failed else 0

    # Parallel: buffer output per-suite, print when done.
    failed = 0
    done_count = 0
    total = len(suites)

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(_run_one_suite, suite_path, env, run_extra_args, retries, use_opik=use_opik): suite_path
            for suite_path in suites
        }
        for future in as_completed(futures):
            suite_path = futures[future]
            name, rc, output = future.result()
            done_count += 1
            status = "exit 0" if rc == 0 else f"exit {rc}"
            print(f"-- {name} {done_count}/{total} {status} --")
            print(output, end="")
            if rc != 0:
                failed += 1

    _print_summary(env, jobs, total, failed, tags)
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Standalone CLI entry point (``python -m agent_evals.scripts.sweep``)."""
    parser = argparse.ArgumentParser(
        prog="agent-evals sweep",
        description="Run every suite under evals/ against the given environment.",
    )
    parser.add_argument("--env", required=True, help="Environment to run against.")
    parser.add_argument("--evals-dir", default=None, help="Directory of suite YAML files (default: ./evals, then ../agent-eval-cases/evals).")
    parser.add_argument("--suite", action="append", default=[], help="Substring filter for suite paths (repeatable).")
    parser.add_argument("--tag", action="append", default=[], help="Tag for all Opik experiments (repeatable).")
    parser.add_argument("-j", "--jobs", type=int, default=10, help="Max concurrent suites (default 10).")
    parser.add_argument("--retries", type=int, default=2, help="Retry failed suites N times (default 2).")
    parser.add_argument("--resume", action="store_true", help="Re-run only suites missing from Opik for the first --tag.")
    parser.add_argument("--resume-file", default=None, help="Resume from an explicit file (one suite path per line).")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity.")
    parser.add_argument("--no-opik", action="store_true", help="Skip Opik recording and tunnel (write local JSON only).")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args forwarded to agent-evals run (e.g. -v, --runs 3).")
    args = parser.parse_args(argv)

    return run_sweep(
        env=args.env,
        evals_dir=args.evals_dir,
        suite_filters=args.suite,
        tags=args.tag,
        jobs=args.jobs,
        retries=args.retries,
        resume=args.resume,
        resume_file=args.resume_file,
        extra_args=list(args.extra) if args.extra else None,
        verbose=args.verbose,
        use_opik=not args.no_opik,
    )


def _print_summary(env: str, jobs: int, total: int, failed: int, tags: list[str] | None = None) -> None:
    print()
    print("=" * 60)
    print(f"SUMMARY  env={env} jobs={jobs} suites={total}")
    if failed:
        print(f"  {failed} suite(s) FAILED")
        if tags:
            resume_tag = tags[0]
            print(f"  check if any are resumable: agent-evals sweep --env {env} --tag {resume_tag} --resume")
            print(f"  (or re-run with a new tag: agent-evals sweep --env {env} --tag \"{env}-$(date +%Y%m%d-%H%M%S)\"")
        else:
            print(f"  re-run with a tag: agent-evals sweep --env {env} --tag \"{env}-$(date +%Y%m%d-%H%M%S)\"")
    else:
        print("  all suites passed")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
