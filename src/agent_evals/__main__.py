"""Command line interface for the agent evaluation harness."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

from dotenv import load_dotenv

load_dotenv()

from .client import AgentClient
from .environment import (
    Environment,
    configuration_help,
    supported_environments,
)
from .loader import EvaluationSuite, load_suite
from .reporting import FileSink, StatsSink
from .results import Sink
from .runner import run_suite, run_suite_multiple


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    parse_intermixed = getattr(parser, "parse_intermixed_args", None)
    if parse_intermixed:
        try:
            args = parse_intermixed(argv)
        except TypeError:
            args = parser.parse_args(argv)
    else:
        args = parser.parse_args(argv)

    if getattr(args, "show", None) is not None:
        if not args.show:
            parser.error("--show requires a suite path")
        return _handle_show(argparse.Namespace(suite=args.show))

    log_level = (
        logging.WARNING - (10 * min(args.verbose, 2)) + (10 * min(args.quiet, 2))
    )
    log_level = max(logging.DEBUG, min(logging.CRITICAL, log_level))
    logging.basicConfig(level=log_level, format="%(levelname)s %(name)s: %(message)s")

    if not hasattr(args, "handler"):
        parser.print_help()
        return 1

    return args.handler(args)


def _build_parser() -> argparse.ArgumentParser:
    verbosity = argparse.ArgumentParser(add_help=False)
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (repeatable)",
    )
    verbosity.add_argument(
        "-q", "--quiet", action="count", default=0, help="Reduce logging verbosity"
    )

    # The epilog is the configuration inventory, rendered from the knob
    # declarations; raw formatting keeps its line structure intact.
    parser = argparse.ArgumentParser(
        description="Run evaluation suites against the agent service",
        parents=[verbosity],
        add_help=True,
        epilog=configuration_help(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-show",
        "-s",
        "--show",
        nargs="?",
        help="Display eval names in a suite and exit",
    )

    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run",
        parents=[verbosity],
        help="Execute an evaluation suite",
        epilog=configuration_help(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_parser.add_argument(
        "suite", type=str, help="Path to a suite YAML file or a directory of suites"
    )
    run_parser.add_argument(
        "--output",
        type=str,
        help="Override output path (default: results/<topic>/<name>_<timestamp>[_<env>].json)",
    )
    run_parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop after the first failing eval",
    )
    # Resolved at parse time, not import: the reachable roster follows the
    # environment loaded for this run.
    env_help = (
        f"Environment to run against. One of: {', '.join(supported_environments())}"
    )
    run_parser.add_argument("--env", type=str, required=True, help=env_help)
    run_parser.add_argument(
        "--concurrency", type=int, help="Maximum number of evals to run in parallel"
    )
    run_parser.add_argument(
        "--name",
        type=str,
        action="append",
        help="Run only evals matching this name (repeatable)",
    )
    run_parser.add_argument(
        "--opik",
        action="store_true",
        help="Also record the run in Opik (needs a reachable Opik)",
    )
    run_parser.add_argument(
        "--dataset-name",
        type=str,
        help="Override the Opik dataset name (default: suite name); only with --opik",
    )
    run_parser.add_argument(
        "--experiment-name",
        type=str,
        help="Override the Opik experiment name (default: suite name); only with --opik",
    )
    run_parser.add_argument(
        "--tag",
        type=str,
        action="append",
        help="Tag the Opik experiment for grouping runs (repeatable); only with --opik",
    )
    run_parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of times to run the suite (aggregates stats across runs; default 1)",
    )
    run_parser.set_defaults(handler=_handle_run)

    show_parser = subparsers.add_parser(
        "show", parents=[verbosity], help="Display eval names in a suite"
    )
    show_parser.add_argument("suite", type=str, help="Path to the suite YAML file")
    show_parser.set_defaults(handler=_handle_show)

    return parser


def _auto_output_path(suite_path: Path, environment: str | None = None) -> Path:
    """Generate a timestamped output path mirroring the suite's folder structure.

    For example ``evals/interviewing/single.yaml`` with env ``eu`` produces
    ``results/interviewing/single_20260414_120000_eu.json``.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_suffix = f"_{environment}" if environment else ""
    resolved = suite_path.resolve()

    # Walk parents to find the 'evals' directory so we can mirror the relative path.
    for parent in resolved.parents:
        if parent.name == "evals":
            rel = resolved.relative_to(parent)
            results_dir = parent.parent / "results"
            return results_dir / rel.parent / f"{rel.stem}_{timestamp}{env_suffix}.json"

    # Fallback: put results next to the suite file.
    return resolved.parent / "results" / f"{resolved.stem}_{timestamp}{env_suffix}.json"


def _resolve_suite_paths(suite_arg: str) -> list[Path]:
    """Return a list of YAML suite files for the given argument.

    *suite_arg* may be a path to a single ``.yaml`` file **or** a directory,
    in which case every ``.yaml`` file inside it (non-recursive, excluding
    ``*_local.yaml`` and empty placeholder stubs) is returned sorted by name.
    """
    target = Path(suite_arg)
    if target.is_dir():
        paths = sorted(
            p
            for p in target.glob("*.yaml")
            if not p.name.startswith("_")
            and not p.name.endswith("_local.yaml")
            and p.stat().st_size > 0  # skip empty placeholder stubs
        )
        if not paths:
            raise SystemExit(f"No .yaml suite files found in {target}")
        return paths
    return [target]


def resolve_environment(name: str | None) -> Environment:
    """Turn the CLI environment selection into an :class:`Environment`.

    There is deliberately no default: a run must name its target via
    ``--env`` before any network activity.
    """
    if not name:
        raise SystemExit(
            f"No environment selected. Pass --env <name>. "
            f"Supported environments: {', '.join(supported_environments())}"
        )
    try:
        return Environment(name)
    except ValueError as exc:
        raise SystemExit(str(exc))


def _handle_run(args: argparse.Namespace) -> int:
    environment = resolve_environment(args.env)
    # The name overrides only reach the Opik sink, so a bare run silently
    # dropping them would betray the help text — reject them loudly instead.
    if not args.opik and (args.dataset_name or args.experiment_name or args.tag):
        raise SystemExit("--dataset-name, --experiment-name, and --tag require --opik.")
    suite_paths = _resolve_suite_paths(args.suite)
    # Resolved once, before any suite runs: a missing extra aborts here rather
    # than after the first suite's messages have already gone out.
    make_opik_sink = _opik_sink_factory(args, environment) if args.opik else None
    all_failed: list[str] = []

    for suite_path in suite_paths:
        exit_code = _run_single_suite(suite_path, args, environment, make_opik_sink)
        if exit_code != 0:
            all_failed.append(str(suite_path))

    if len(suite_paths) > 1:
        ok = len(suite_paths) - len(all_failed)
        print(f"\n[agent-evals] Overall: {ok}/{len(suite_paths)} suites passed.")
        if all_failed:
            print(f"[agent-evals] Failed suites: {', '.join(all_failed)}")

    return 2 if all_failed else 0


def _opik_sink_factory(
    args: argparse.Namespace, environment: Environment
) -> Callable[[], Sink]:
    """Return a per-suite Opik-sink constructor.

    A directory run records one dataset and one experiment per Suite, so each
    suite gets its own sink instance (each carries its own accumulated results).
    The ``opik`` package ships in the base dependencies but is imported here,
    only when ``--opik`` is passed, so a bare run never pays for it.
    """
    from .reporting import OpikSink as opik_sink

    def make() -> Sink:
        return opik_sink(
            environment,
            dataset_name=args.dataset_name,
            experiment_name=args.experiment_name,
            experiment_tags=args.tag,
        )

    return make


def _run_single_suite(
    suite_path: Path,
    args: argparse.Namespace,
    environment: Environment,
    make_opik_sink: Callable[[], Sink] | None,
) -> int:
    suite = load_suite(str(suite_path))
    _apply_runtime_overrides(suite, args)

    total = len(suite.cases)
    suite_label = suite.name or suite_path.stem
    runs = max(1, getattr(args, "runs", 1))
    run_label = f" x{runs} runs" if runs > 1 else ""
    logging.info(
        "Running suite '%s' (%d case%s)%s...",
        suite_label,
        total,
        "s" if total != 1 else "",
        run_label,
    )

    output_path = (
        Path(args.output)
        if args.output
        else _auto_output_path(suite_path, environment.name)
    )

    stats_sink: StatsSink | None = None
    if runs > 1:
        stats_sink = StatsSink(
            output_path,
            suite_path=str(suite_path),
            env=environment.name,
        )

    def sinks_factory(run_index: int) -> list[Sink]:
        """Build the sink set for run *run_index* (1-based).

        Per-run sinks (FileSink, OpikSink) get a fresh instance each call;
        the stats sink is reused across runs so it can accumulate.
        """
        if runs > 1:
            per_run_path = output_path.with_name(
                f"{output_path.stem}_run{run_index}{output_path.suffix}"
            )
        else:
            per_run_path = output_path
        sinks: list[Sink] = [
            FileSink(per_run_path, trace_base_url=environment.trace_base_url)
        ]
        if make_opik_sink is not None:
            sinks.append(make_opik_sink())
        if stats_sink is not None:
            sinks.append(stats_sink)
        return sinks

    client = AgentClient(environment)
    try:
        if runs > 1:
            all_results = run_suite_multiple(
                suite,
                client,
                runs=runs,
                sinks_factory=sinks_factory,
            )
            final_results = all_results[-1]
        else:
            final_results = run_suite(suite, client, sinks=sinks_factory(1))
    finally:
        client.close()

    failed = [result for result in final_results if not result.success]
    passed = total - len(failed)
    if failed:
        failing_names = ", ".join(result.name for result in failed)
        print(
            f"[agent-evals] Completed with failures: {passed}/{total} passed. Failing evals: {failing_names}."
        )
    else:
        print(f"[agent-evals] Completed successfully: {passed}/{total} passed.")
    if failed:
        for result in failed:
            logging.error(
                "Eval %s failed: %s",
                result.name,
                result.error_summary() or "see response",
            )
        return 2
    return 0


def _handle_show(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite)
    for case in suite.cases:
        print(case.name)
    return 0


def _apply_runtime_overrides(suite: EvaluationSuite, args: argparse.Namespace) -> None:
    if args.stop_on_failure:
        suite.options.stop_on_failure = True
    if getattr(args, "concurrency", None):
        suite.options.concurrency = max(1, args.concurrency)
    names = getattr(args, "name", None)
    if names:
        filtered = [e for e in suite.cases if e.name in names]
        if not filtered:
            raise SystemExit(f"No evals matched --name filter: {names}")
        suite.cases = filtered


if __name__ == "__main__":
    sys.exit(main())
