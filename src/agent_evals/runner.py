"""Evaluation runner: orchestration only — send → resolve → record."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Callable, Sequence

from .client import AgentClient
from .errors import ErrorCode, EvalError
from .expectations import (
    CheckResult,
    ExpectationResult,
    ExpectedState,
    evaluate_response,
)
from .loader import EvaluationCase, EvaluationSuite, SuiteOptions
from .provisioning import AgentPool
from .results import EvaluationResult, Sink, StepResult, UsageMetrics
from .schemas.response import Response

_LOGGER = logging.getLogger(__name__)


def run_suite(
    suite: EvaluationSuite,
    client: AgentClient,
    *,
    sinks: Sequence[Sink] = (),
) -> list[EvaluationResult]:
    """Run every evaluation in the suite and return their results.

    Each Sink is started before any network activity (a Sink that cannot
    start aborts the run here) and handed every result as it is collected.
    Every started Sink is closed no matter how the run ends, so a crashed or
    stopped run flushes the partial results written so far.
    """
    results: list[EvaluationResult] = []
    concurrency = max(1, suite.options.concurrency)
    stop_on_failure = suite.options.stop_on_failure and concurrency == 1
    if concurrency > 1 and suite.options.stop_on_failure:
        _LOGGER.warning(
            "stop_on_failure is not supported with concurrency > 1; proceeding without early exit"
        )
    started: list[Sink] = []
    try:
        for sink in sinks:
            sink.on_start(suite)
            started.append(sink)
        # Fail-fast phase: a misconfiguration or broken spec aborts here,
        # before any message is sent or budget spent; the resulting map is
        # read-only for the rest of the run.
        _preflight(suite.cases)
        pool = AgentPool()
        pool.provision(suite.cases, client)
        if concurrency == 1 or len(suite.cases) == 1:
            for case in suite.cases:
                _LOGGER.info("Running eval: %s", case.name)
                result = _run_case_bounded(
                    case,
                    client,
                    pool,
                    _effective_timeout(case, suite.options),
                    clone_client=False,
                )
                results.append(result)
                for sink in sinks:
                    sink.write(case, result)
                if stop_on_failure and not result.success:
                    _LOGGER.warning("Stopping early because eval %s failed", case.name)
                    break
        else:
            _LOGGER.info("Running up to %s evals in parallel", concurrency)
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(
                        _run_case_bounded,
                        case,
                        client,
                        pool,
                        _effective_timeout(case, suite.options),
                        clone_client=True,
                    )
                    for case in suite.cases
                ]
                for case, future in zip(suite.cases, futures):
                    _LOGGER.info("Awaiting eval: %s", case.name)
                    result = future.result()
                    results.append(result)
                    for sink in sinks:
                        sink.write(case, result)
    finally:
        for sink in started:
            sink.close()
    return results


def _preflight(cases: Sequence[EvaluationCase]) -> None:
    """Abort the run when any expectation reports it cannot start.

    Reduced polymorphically over ``preflight()`` — like ``continues_task()``,
    so the runner names no expectation type. Identical reports (the same
    missing credential declared by many steps) collapse to one.
    """
    problems: list[str] = []
    for case in cases:
        for step in case.steps:
            for expectation in step.expectations:
                problem = expectation.preflight()
                if problem and problem not in problems:
                    problems.append(problem)
    if problems:
        raise RuntimeError("\n".join(problems))


def run_suite_multiple(
    suite: EvaluationSuite,
    client: AgentClient,
    *,
    runs: int,
    sinks_factory: Callable[[int], Sequence[Sink]],
) -> list[list[EvaluationResult]]:
    """Run *suite* ``runs`` times and return the per-run result lists.

    Each run gets a fresh set of sinks from *sinks_factory* (indexed from 1)
    so per-run sinks that accumulate-and-flush on ``close`` never bleed across
    runs.  A single *client* is reused across all runs — per-run agent
    isolation comes from :class:`AgentPool` being rebuilt inside each
    :func:`run_suite` call, not from the HTTP connection.

    After all runs complete, ``aggregate_runs`` is called on each Sink from
    the last run so cross-run artifacts (stats CSV/JSON/Markdown) can be
    emitted.  Per-run Sinks no-op this hook; a stats-oriented Sink does the
    real work.

    Returns one list of :class:`EvaluationResult` per run.
    """
    all_results: list[list[EvaluationResult]] = []
    last_sinks: Sequence[Sink] = ()
    for run_index in range(1, runs + 1):
        if runs > 1:
            _LOGGER.info("=== Run %d/%d ===", run_index, runs)
        sinks = sinks_factory(run_index)
        last_sinks = sinks
        all_results.append(run_suite(suite, client, sinks=sinks))

    for sink in last_sinks:
        sink.aggregate_runs()

    return all_results


def _effective_timeout(case: EvaluationCase, options: SuiteOptions) -> float | None:
    return case.timeout_seconds if case.timeout_seconds else options.timeout_seconds


def _run_case_bounded(
    case: EvaluationCase,
    shared_client: AgentClient,
    pool: AgentPool,
    timeout: float | None,
    *,
    clone_client: bool,
) -> EvaluationResult:
    if timeout is None:
        if clone_client:
            return _run_case_with_new_client(case, shared_client, pool)
        return execute_case(case, shared_client, pool)
    return _run_case_with_timeout(case, shared_client, pool, timeout)


_SHUTDOWN_GRACE_SECONDS: float = 2.0


class _EvalCancelled(Exception):
    """Raised inside a worker thread when cooperative cancellation has been requested."""


def _check_cancelled(stop_event: threading.Event | None) -> None:
    """Raise :class:`_EvalCancelled` when *stop_event* has been set."""
    if stop_event is not None and stop_event.is_set():
        raise _EvalCancelled


def _run_case_with_timeout(
    case: EvaluationCase,
    shared_client: AgentClient,
    pool: AgentPool,
    timeout: float,
) -> EvaluationResult:
    result_box: list[EvaluationResult] = []
    stop_event = threading.Event()

    def target() -> None:
        try:
            result_box.append(
                _run_case_with_new_client(
                    case, shared_client, pool, stop_event=stop_event
                )
            )
        except _EvalCancelled:
            pass
        except (
            Exception
        ) as exc:  # defensive: clone() could raise before execute_case catches
            result_box.append(
                EvaluationResult(
                    name=case.name,
                    success=False,
                    agent_id=None,
                    step_results=[_harness_failure_step(EvalError.from_exception(exc))],
                )
            )

    thread = threading.Thread(target=target, name=f"eval-{case.name}", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        stop_event.set()
        _LOGGER.warning(
            "Eval %s exceeded wall-clock timeout of %.1fs; signalling worker to stop",
            case.name,
            timeout,
        )
        thread.join(timeout=_SHUTDOWN_GRACE_SECONDS)
        if thread.is_alive():
            _LOGGER.warning(
                "Eval %s worker did not stop within %.1fs grace period; abandoning",
                case.name,
                _SHUTDOWN_GRACE_SECONDS,
            )
        return EvaluationResult(
            name=case.name,
            success=False,
            agent_id=None,
            duration_seconds=timeout,
            step_results=[
                _harness_failure_step(
                    EvalError(
                        ErrorCode.EVAL_TIMEOUT,
                        f"eval exceeded wall-clock timeout of {timeout}s",
                        {"timeout_seconds": timeout},
                    )
                )
            ],
        )
    return result_box[0]


def _harness_failure_step(error: EvalError) -> StepResult:
    """A synthetic Step Trail entry for a failure that never reached a send.

    A timeout or a failure before the request was built has no Step of its own,
    so its Harness Failure rides an unnamed, request-less entry with no
    verdicts — the case's sole record of its death.
    """
    return StepResult(
        name=None,
        success=False,
        request=None,
        response=None,
        harness_error=error,
    )


def _fold_missing_task_id(results: list[ExpectationResult], detail: str) -> None:
    """Record a missing continuation task id as a failed ``expected_state`` check.

    A step that declared ``input-required`` but got no task id back has a runtime
    failure with no declared term of its own. Folding it into the always-present
    ``expected_state`` result keeps the per-term structure the single source both
    sinks render — the synthesized failed check is the miss's only record.
    The result carrier is frozen, so it is rebuilt with the extra check rather
    than mutated in place.
    """
    for index, result in enumerate(results):
        if result.key == ExpectedState.key:
            extended = [*result.checks, CheckResult("", False, detail)]
            results[index] = replace(result, checks=extended)
            return


def execute_case(
    case: EvaluationCase,
    client: AgentClient,
    pool: AgentPool,
    *,
    stop_event: threading.Event | None = None,
) -> EvaluationResult:
    """Run every Step of *case* through one loop, threading context/task ids.

    One path for a case of any length: each Step sends, builds its Response,
    evaluates its expectations, and appends a :class:`StepResult`; a failing Step
    stops the loop. The case succeeds only if every executed Step did.
    """
    agent_id: str | None = None
    response: Response | None = None
    step_results: list[StepResult] = []
    start = time.perf_counter()
    current_context_id: str | None = None
    current_task_id: str | None = None
    overall_success = True
    try:
        _check_cancelled(stop_event)
        agent_id = pool.agent_id_for(case)

        for step in case.steps:
            _check_cancelled(stop_event)
            if step.delay_before_seconds is not None and step.delay_before_seconds > 0:
                _LOGGER.debug(
                    "Sleeping %.3f seconds before step %s",
                    step.delay_before_seconds,
                    step.name,
                )
                if stop_event is not None:
                    if stop_event.wait(timeout=step.delay_before_seconds):
                        raise _EvalCancelled
                else:
                    time.sleep(step.delay_before_seconds)
            _check_cancelled(stop_event)
            request = step.message.prepare(current_context_id, current_task_id)
            step_start = time.perf_counter()
            raw_response = client.send_message(agent_id, request.to_dict())
            response = Response.from_dict(raw_response)
            step_duration = time.perf_counter() - step_start
            results = evaluate_response(
                step.expectations, raw_response, duration_seconds=step_duration
            )

            context_candidate = response.resolved_context_id()
            if context_candidate:
                current_context_id = context_candidate

            # Threading is polymorphic: a step carries its taskId forward iff one
            # of its expectations says so (only ``expected_state: input-required``
            # does), reduced over ``continues_task()`` rather than matched by type.
            # A continuation that arrives without a task id folds into the step's
            # ``expected_state`` result as an extra failed check — the per-term
            # view is the miss's one and only record.
            current_task_id = None
            if any(e.continues_task() for e in step.expectations):
                task_candidate = response.resolved_task_id()
                if not task_candidate:
                    _fold_missing_task_id(
                        results, "expected task id in response to continue the flow"
                    )
                current_task_id = task_candidate

            step_successful = all(result.passed for result in results)
            step_results.append(
                StepResult(
                    name=step.name,
                    success=step_successful,
                    request=request,
                    response=response,
                    duration_seconds=step_duration,
                    context_id=context_candidate,
                    usage=UsageMetrics.from_response(raw_response),
                    results=results,
                )
            )
            if not step_successful:
                overall_success = False
                break

        duration = time.perf_counter() - start
        return EvaluationResult(
            name=case.name,
            success=overall_success,
            agent_id=agent_id,
            duration_seconds=duration,
            step_results=step_results,
        )
    except _EvalCancelled:
        raise
    except Exception as exc:  # pragma: no cover - defensive logging
        _LOGGER.exception("Evaluation %s raised an exception", case.name)
        duration = time.perf_counter() - start
        step_results.append(_harness_failure_step(EvalError.from_exception(exc)))
        return EvaluationResult(
            name=case.name,
            success=False,
            agent_id=agent_id,
            duration_seconds=duration,
            step_results=step_results,
        )


def _run_case_with_new_client(
    case: EvaluationCase,
    shared_client: AgentClient,
    pool: AgentPool,
    *,
    stop_event: threading.Event | None = None,
) -> EvaluationResult:
    clone = shared_client.clone()
    try:
        return execute_case(case, clone, pool, stop_event=stop_event)
    finally:
        clone.close()
