"""The Opik sink: replays stored results through the SDK's ``evaluate()``.

The runner is the only sender; this module never sends a message. ``on_start``
verifies the whole Opik side (URL, tunnel, client, dataset shell) before the
first send; ``close`` rebuilds the dataset from exactly the executed Cases and
replays the stored results through ``evaluate()`` — a no-network lookup by
Case name, not a re-execution.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import fields
from typing import Any

from opik import Opik
from opik.evaluation import evaluate
from opik.evaluation.metrics import base_metric, score_result

from ..environment import OPIK_URL_OVERRIDE_VAR, Environment
from ..expectations import (
    Expectation,
    ExpectationResult,
    parse_expectations,
    registry,
)
from ..loader import EvaluationCase, EvaluationSuite, Step
from ..results import EvaluationResult
from ..schemas.agent import Agent
from .opik_target import resolve_opik_url
from .trace import build_trace_url

_LOGGER = logging.getLogger(__name__)

# The project agent-api logs its traces to (the Environment carries its id per
# environment for trace links); keeping evals in the same project keeps the
# experiment rows next to the real traces they link to.
DEFAULT_OPIK_PROJECT_NAME = "Agents"

_UNNAMED_STEP_KEY = "case"


# --- dataset item construction -----------------------------------------------


def _expectations_to_block(expectations: list[Expectation]) -> dict[str, Any]:
    """Serialize a parsed expectation list back to an ``expectations:`` block.

    The inverse of the strict parse: each authored expectation re-emits under its
    key so the dataset item carries a re-parseable block — the metric renders a
    step that never ran from it. The auto-injected default state guard is dropped
    (parsing re-injects it), so it never persists as a phantom key.
    """
    return {e.key: e.to_raw() for e in expectations if not e.is_injected_default()}


def _step_to_dict(step: Step) -> dict[str, Any]:
    """Serialize one Step for the dataset item's ``steps`` column."""
    data: dict[str, Any] = {
        "name": step.name,
        "message": step.message.to_dict(),
        "expectations": _expectations_to_block(step.expectations),
    }
    if step.delay_before_seconds is not None:
        data["delay_before_seconds"] = step.delay_before_seconds
    return data


def _case_to_dataset_item(case: EvaluationCase) -> dict[str, Any]:
    """Build one Opik dataset item from a Case's authored shape.

    The ``steps`` column carries each Step's declared expectations so the metric
    can render placeholders for Steps that never executed. Other envelope fields
    (agent, description) ride along for the Opik UI; the replay task ignores
    them — it looks up by name only.
    """
    item: dict[str, Any] = {}
    for field in fields(case):
        value = getattr(case, field.name)
        if not value:
            continue
        if field.name == "steps":
            item["steps"] = [_step_to_dict(step) for step in value]
        elif isinstance(value, Agent):
            item[field.name] = value.to_dict()
        else:
            item[field.name] = value
    return item


def suite_to_dataset_items(suite: EvaluationSuite) -> list[dict[str, Any]]:
    """Convert an :class:`EvaluationSuite` to a list of Opik dataset item dicts.

    Every Case is steps-based, so each becomes one item whose ordered Steps ride
    a single ``steps`` column. A single-message Case is simply a one-Step item.
    """
    return [_case_to_dataset_item(case) for case in suite.cases]


# --- task output construction (the replay) -----------------------------------


def _build_task_output(
    result: EvaluationResult,
    *,
    trace_base_url: str | None,
    environment: str,
) -> dict[str, Any]:
    """Render a stored :class:`EvaluationResult` as an Opik task output dict.

    The replay task's no-network lookup returns this. A Harness Failure (a
    synthetic verdict-less Step the runner stamps on a timeout or transport
    error) lifts into the case-level ``error`` field — never a name-less trail
    row — so the trail carries executed Steps only and the metric renders the
    unexecuted remainder as ``task failed`` placeholders. Usage sums across
    executed Steps; the trace URL derives from the last executed Step's
    context id — the same derivations the file Sink uses.
    """
    trail: list[dict[str, Any]] = []
    case_error: str | None = None

    for step in result.step_results:
        if step.harness_error is not None:
            # The runner's synthetic Step (name=None, no verdicts) records a
            # death that never reached a send. Lift it to the case level so
            # the trail carries executed Steps only.
            if case_error is None:
                case_error = str(step.harness_error)
            continue
        entry: dict[str, Any] = {
            "name": step.name,
            "success": step.success,
            "expectation_results": [r.to_dict() for r in step.results],
            "duration_seconds": step.duration_seconds,
        }
        trail.append(entry)

    usage = result.aggregate_usage()
    output: dict[str, Any] = {
        "step_results": trail,
        "usage": usage.as_dict() if usage else None,
        "trace_url": build_trace_url(trace_base_url, result.trace_context_id()),
        "environment": environment,
    }
    if case_error is not None:
        output["error"] = case_error
    return output


# --- the metric: a pure renderer over stored per-term results ----------------


def _task_failed(message: str) -> str:
    """Render the reason shown when the agent task itself errored.

    Centralises the ``task failed: ...`` prefix so every column (per-term
    blocks, scalar checks, ``expected_state`` and ``overall``) reports it
    identically.
    """
    return f"task failed: {message}"


def _emits_column(expectation: Expectation, result: ExpectationResult) -> bool:
    """Whether a resolved expectation surfaces as its own score column.

    A column appears for every authored expectation that declared at least one
    term. The always-on default state guard is the sole exception: it runs on
    every step but stays silent unless it actually caught a failure, so an
    ordinary step does not sprout a spurious ``expected_state`` column.
    """
    if not result.checks:
        return False
    if expectation.is_injected_default() and result.passed:
        return False
    return True


def _result_contribution(
    result: ExpectationResult,
) -> tuple[int, int, str | dict[str, str]]:
    """Reduce a result to ``(passed_units, total_units, reason_fragment)``.

    So a column can collapse to one score without ever asking what expectation
    type produced it: a scalar (every check unlabelled) contributes one strict
    pass/fail unit and a plain verdict string; a list expectation contributes one
    unit per term and a label->detail fragment that keeps the per-term detail
    visible in the reason.
    """
    checks = result.checks
    if checks and all(check.label == "" for check in checks):
        detail = next((c.detail for c in checks if not c.passed), checks[-1].detail)
        # Never leave the cell blank: a whitespace-only detail falls back to the
        # verdict word (a passing scalar's detail is already "passed").
        reason = detail if detail.strip() else ("PASS" if result.passed else "FAIL")
        return (1 if result.passed else 0), 1, reason
    passed = sum(1 for c in checks if c.passed)
    return passed, len(checks), {c.label: c.detail for c in checks}


def _reason_text(fragment: str | dict[str, str]) -> str:
    """A scalar fragment is its own reason; a per-term fragment is JSON."""
    return fragment if isinstance(fragment, str) else json.dumps(fragment)


def _overall_score(
    columns: list[score_result.ScoreResult],
    reasons: list[str],
    *,
    error: str | None,
) -> score_result.ScoreResult:
    """The block-weighted mean of the component columns.

    Each expectation type counts once. A case with no columns is a vacuous pass
    unless the agent task itself errored.
    """
    if columns:
        value = sum(c.value for c in columns) / len(columns)
    else:
        value = 0.0 if error else 1.0
    return score_result.ScoreResult(
        name="overall", value=value, reason="; ".join(reasons) or "passed"
    )


def _placeholder_pairs(
    expectations: list[Expectation], reason: str
) -> list[tuple[Expectation, ExpectationResult]]:
    """Declared expectations rendered as all-failed — a step that never ran.

    The injected default guard is dropped: an unexecuted step with no declared
    state must not invent a failing ``expected_state`` column.
    """
    return [
        (e, e.failed_placeholder(reason))
        for e in expectations
        if not e.is_injected_default()
    ]


def _stored_pairs(
    expectations: list[Expectation], entry: dict[str, Any]
) -> list[tuple[Expectation, ExpectationResult]]:
    """Pair each declared expectation with the result the task stored for it."""
    by_key = {
        r["key"]: ExpectationResult.from_dict(r)
        for r in entry.get("expectation_results") or []
    }
    return [(e, by_key[e.key]) for e in expectations if e.key in by_key]


def _step_reason_key(step: dict[str, Any]) -> str:
    """The reason-key for a step — its name, or a fallback for an unnamed one.

    A single-message Case's lone Step has ``name=None``; JSON serialising a
    ``None`` dict key yields ``"null"``, so a stable string keeps the reason
    readable and round-trippable.
    """
    return step.get("name") or _UNNAMED_STEP_KEY


def _sequential_columns(
    steps: list[dict[str, Any]],
    step_results: list[dict[str, Any]],
    *,
    task_error: str | None,
) -> list[score_result.ScoreResult]:
    """Merge per-step outcomes into one score column per expectation type.

    Pure rendering over the per-term results the task already produced: each
    column pools its units across every step (list terms, or one unit per
    declaring step for a scalar), keyed in the reason by step name. Steps that
    executed keep their stored verdicts even when the case has a task-level
    error — a real pass stays visible as one. Steps that never executed —
    because an earlier one stopped the case, or the task itself failed —
    contribute all-failed placeholders, so a case that died early cannot read
    as a near-pass. Columns follow canonical order for a stable table.
    """
    failed_step = next((r["name"] for r in step_results if not r.get("success")), None)
    merged: dict[str, dict[str, Any]] = {}
    for index, step in enumerate(steps):
        expectations = parse_expectations(step.get("expectations") or {})
        if index < len(step_results):
            # An executed Step's stored verdicts always win — even when the
            # case died later, a real pass on this Step stays visible.
            pairs = _stored_pairs(expectations, step_results[index])
        elif task_error is not None:
            pairs = _placeholder_pairs(expectations, _task_failed(task_error))
        else:
            pairs = _placeholder_pairs(
                expectations, f"not executed: step {failed_step} failed"
            )
        for expectation, result in pairs:
            if not _emits_column(expectation, result):
                continue
            passed, total, fragment = _result_contribution(result)
            slot = merged.setdefault(
                result.key, {"passed": 0, "total": 0, "reason": {}}
            )
            slot["passed"] += passed
            slot["total"] += total
            slot["reason"][_step_reason_key(step)] = fragment
    columns: list[score_result.ScoreResult] = []
    for key in registry():
        slot = merged.get(key)
        if slot is None:
            continue
        columns.append(
            score_result.ScoreResult(
                name=key,
                value=slot["passed"] / slot["total"] if slot["total"] else 1.0,
                reason=json.dumps(slot["reason"]),
            )
        )
    return columns


class ExpectationMetric(base_metric.BaseMetric):
    """Opik metric that renders the per-term expectation results as columns.

    One ``ScoreResult`` per expectation *type* plus an ``overall``, so each kind
    shows up as a single column in the Experiments table (per-term detail lives
    in the score reason). List-valued types score the average of their per-term
    verdicts; scalar checks stay strict ``1.0`` / ``0.0``; ``overall`` is the
    block-weighted mean. The metric never branches on expectation type — it
    renders generically from the ``checks`` structure the task already stored,
    never re-evaluating.
    """

    def __init__(self) -> None:
        super().__init__(name="expectations")

    def score(
        self,
        steps: list[dict[str, Any]],
        step_results: list[dict[str, Any]] | None = None,
        error: str | None = None,
        **kwargs: Any,
    ) -> list[score_result.ScoreResult]:
        return self._score_sequential(steps, step_results or [], error)

    def _score_sequential(
        self,
        steps: list[dict[str, Any]],
        step_results: list[dict[str, Any]],
        error: str | None,
    ) -> list[score_result.ScoreResult]:
        columns = _sequential_columns(steps, step_results, task_error=error)

        # A Harness Failure is the case-level ``error``, never a trail row, so a
        # trail entry only ever carries failed *checks* — the per-step reason is
        # built from those alone.
        reasons: list[str] = []
        if error:
            reasons.append(_task_failed(error))
        for entry in step_results:
            for result in entry.get("expectation_results") or []:
                reasons.extend(
                    f"{entry.get('name') or _UNNAMED_STEP_KEY}: {check['detail']}"
                    for check in result["checks"]
                    if not check["passed"]
                )
        return [_overall_score(columns, reasons, error=error), *columns]


# --- the sink ----------------------------------------------------------------


class OpikSink:
    """The Opik sink: replays stored results through the SDK's ``evaluate()``.

    Driven by the runner through the :class:`Sink` protocol. ``on_start``
    verifies the whole Opik side (resolve the URL, spawn the managed tunnel,
    create the client, get-or-create the dataset shell — keeping the
    project-pin recreate) and aborts the run on any failure before the first
    message is sent. ``write`` accumulates each executed Case's result.
    ``close`` rebuilds the dataset from exactly the executed Cases and replays
    the stored results through ``evaluate()`` — a no-network lookup by Case
    name, not a re-execution.
    """

    def __init__(
        self,
        environment: Environment,
        *,
        dataset_name: str | None = None,
        experiment_name: str | None = None,
        project_name: str | None = None,
        experiment_tags: list[str] | None = None,
    ) -> None:
        self._environment = environment
        self._dataset_name_override = dataset_name
        self._experiment_name_override = experiment_name
        self._project_name_override = project_name
        self._experiment_tags = experiment_tags
        self._suite: EvaluationSuite | None = None
        self._entries: list[tuple[EvaluationCase, EvaluationResult]] = []
        self._opik_client: Opik | None = None
        self._dataset: Any = None
        self._trace_base_url: str | None = None

    def on_start(self, suite: EvaluationSuite) -> None:
        """Resolve the Opik side, create the client, and get-or-create the dataset.

        Any failure aborts the run here, before any message is sent — a run
        whose requested experiment cannot exist must not spend twenty minutes
        finding out.
        """
        self._suite = suite
        url = resolve_opik_url()
        project = (
            self._project_name_override
            or os.environ.get("OPIK_PROJECT_NAME")
            or DEFAULT_OPIK_PROJECT_NAME
        )
        # Only a *user-set* OPIK_URL_OVERRIDE may redirect trace links; capture
        # the base now, before the tunnel URL is exported below — that internal
        # export is not somewhere a browser can see traces.
        self._trace_base_url = self._environment.trace_base_url
        # ``evaluate()`` routes its per-case wrapper trace through the SDK's
        # *global* client, which reads env config rather than this instance;
        # exporting both keeps everything in the same Opik and project.
        os.environ.setdefault(OPIK_URL_OVERRIDE_VAR, url)
        # Assignment, not setdefault: resolution above already honored the env
        # var, and an explicit ``project_name`` argument must win on both sides.
        os.environ["OPIK_PROJECT_NAME"] = project
        self._opik_client = Opik(
            host=url,
            workspace="default",
            api_key=os.environ.get("OPIK_API_KEY"),
        )
        name = self._dataset_name_override or suite.name
        dataset = self._opik_client.get_or_create_dataset(
            name=name, project_name=project
        )
        if dataset.project_name != project:
            # Datasets from before the project default stay pinned to the project
            # they were created in: the SDK resolves the *stored* project over the
            # requested one, ``evaluate()`` routes its traces to the dataset's
            # project, and the backend ignores project moves. The rows are rebuilt
            # from the suite YAML below, so recreating loses nothing.
            _LOGGER.warning(
                "Dataset %r lives in project %r; recreating it in %r",
                name,
                dataset.project_name,
                project,
            )
            self._opik_client.delete_dataset(name=name)
            dataset = self._opik_client.create_dataset(name=name, project_name=project)
        self._dataset = dataset

    def write(self, case: EvaluationCase, result: EvaluationResult) -> None:
        self._entries.append((case, result))

    def close(self) -> None:
        """Rebuild the dataset from the executed Cases and replay through ``evaluate()``.

        The dataset mirrors the run: ``clear()`` then insert items built from
        exactly the accumulated ``(case, result)`` pairs, then the replay —
        ``evaluate()`` iterates exactly the items that have results, so an
        aborted run cannot render unexecuted Cases as failures.
        """
        if self._suite is None or self._opik_client is None:
            return  # on_start never ran (a later sink's on_start failed first)
        # Rebuild every run: clear stale/renamed items first, then mirror exactly
        # what executed. A run that executed nothing (aborted before the first
        # Case, or fully filtered out) leaves a cleared dataset and no
        # experiment — ``evaluate()`` on an empty dataset would only error.
        items = [_case_to_dataset_item(case) for case, _ in self._entries]
        self._dataset.clear()
        if not items:
            return
        self._dataset.insert(items)

        by_name = {case.name: result for case, result in self._entries}
        trace_base_url = self._trace_base_url
        environment_name = self._environment.name

        def task(item: dict[str, Any]) -> dict[str, Any]:
            # A no-network lookup: the result the runner already produced,
            # keyed by the Case name the dataset item carries.
            return _build_task_output(
                by_name[item["name"]],
                trace_base_url=trace_base_url,
                environment=environment_name,
            )

        evaluate(
            dataset=self._dataset,
            task=task,
            scoring_metrics=[ExpectationMetric()],
            experiment_name=self._experiment_name_override or self._suite.name,
            # The target environment is run-level metadata read from the single
            # authoritative source (the Environment), so two runs of the same
            # suite against different deployments can be told apart in the
            # Experiments table.
            experiment_config={
                "suite": self._suite.name,
                "environment": environment_name,
            },
            experiment_tags=self._experiment_tags,
        )

    def aggregate_runs(self) -> None:
        """No-op — cross-run aggregation is not needed for Opik experiments."""
