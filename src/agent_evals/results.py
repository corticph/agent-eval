"""The result model and the Sink protocol: the execute→report contract.

The runner produces these results and hands them to Sinks through the
:class:`Sink` hooks defined here — neither side imports the other's concrete
types, so the seam stays one-directional.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, Protocol, Sequence

from .errors import EvalError
from .schemas.message import MessagePayload
from .schemas.response import Response

if TYPE_CHECKING:
    from .expectations import ExpectationResult
    from .loader import EvaluationCase, EvaluationSuite


class Sink(Protocol):
    """A consumer of a run's results, driven by the runner through four hooks.

    The runner calls ``on_start`` once before any Case executes (a Sink that
    cannot start aborts the run here, before any message is sent) and
    ``write`` once per executed Case as its result is collected. Every
    started Sink gets ``close`` no matter how the run ends, so a crashed or
    stopped run flushes partial results. After every run in a multi-run
    loop completes, each Sink from the last run gets ``aggregate_runs`` so
    cross-run artifacts (stats, combined summaries) can be emitted. A Sink
    only renders what execution produced; it never sends messages or
    re-evaluates.
    """

    def on_start(self, suite: "EvaluationSuite") -> None: ...

    def write(self, case: "EvaluationCase", result: "EvaluationResult") -> None: ...

    def close(self) -> None: ...

    def aggregate_runs(self) -> None: ...


def _first_int(source: dict[str, Any], *keys: str) -> int | None:
    """Return the first key's value that is a real int (not bool), else None."""
    for key in keys:
        value = source.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


@dataclass(slots=True)
class UsageMetrics:
    """LLM usage/credits accounting from ``task.metadata`` (``$usage`` / ``credits``).

    Both keys are documented public extensions of the agent service's wire
    contract, emitted per request on terminal events. Either may be absent
    depending on the deployment, so all fields are optional.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    credits: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "credits": self.credits,
        }

    @classmethod
    def from_response(cls, response: dict[str, Any] | None) -> "UsageMetrics | None":
        """Extract usage accounting from a task response, or None if absent."""
        if not isinstance(response, dict):
            return None
        task = response.get("task")
        if not isinstance(task, dict):
            return None
        metadata = task.get("metadata")
        if not isinstance(metadata, dict):
            return None

        # Agent API v2 carries usage under the ``$usage`` key with camelCase
        # fields and ``credits`` nested inside it. v1 used ``_usage`` with
        # snake_case and a top-level ``credits``; accept both.
        usage = metadata.get("$usage")
        if not isinstance(usage, dict):
            usage = metadata.get("_usage")
        usage = usage if isinstance(usage, dict) else {}

        input_tokens = _first_int(usage, "inputTokens", "input_tokens")
        output_tokens = _first_int(usage, "outputTokens", "output_tokens")

        credits: float | None = None
        raw_credits = usage.get("credits")
        if raw_credits is None:
            raw_credits = metadata.get("credits")
        if isinstance(raw_credits, (int, float)) and not isinstance(raw_credits, bool):
            credits = float(raw_credits)

        if input_tokens is None and output_tokens is None and credits is None:
            return None
        return cls(
            input_tokens=input_tokens, output_tokens=output_tokens, credits=credits
        )

    @classmethod
    def aggregate(cls, items: Sequence["UsageMetrics | None"]) -> "UsageMetrics | None":
        """Field-wise sum across steps; a field stays None if no step reported it."""
        input_tokens = [
            u.input_tokens
            for u in items
            if u is not None and u.input_tokens is not None
        ]
        output_tokens = [
            u.output_tokens
            for u in items
            if u is not None and u.output_tokens is not None
        ]
        credits = [u.credits for u in items if u is not None and u.credits is not None]
        if not input_tokens and not output_tokens and not credits:
            return None
        return cls(
            input_tokens=sum(input_tokens) if input_tokens else None,
            output_tokens=sum(output_tokens) if output_tokens else None,
            credits=sum(credits) if credits else None,
        )


@dataclass(slots=True)
class StepResult:
    """Outcome of running one Step of a case.

    ``name`` and ``request`` are nullable so a synthesized single-message Step
    (no name) and a Step that never reached a send — a timeout or a failure
    before the request was built — can still record their Harness Failure here
    rather than on the case.
    """

    name: str | None
    success: bool
    request: MessagePayload | None
    response: Response | None
    # At most one Harness Failure (timeout, exception before/during a send) —
    # beside, never mixed into, the check verdicts. A failed check has no
    # representation here: its failed ``CheckResult`` is its only record.
    harness_error: EvalError | None = None
    duration_seconds: float | None = None
    # The conversation identifier (Opik threads are keyed by it) — stamped once
    # by execution; there is no separate trace id.
    context_id: str | None = None
    usage: UsageMetrics | None = None
    # The per-term ``ExpectationResult`` list — the source of truth both sinks
    # render from, judge included (as a ``show_on_pass`` result). Supersedes the
    # old standalone ``judge_assessment`` slot.
    results: list["ExpectationResult"] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "success": self.success,
            "request": self.request.to_dict() if self.request is not None else None,
            "response": self.response.to_dict() if self.response is not None else None,
            "harness_error": self.harness_error.to_dict()
            if self.harness_error
            else None,
            "duration_seconds": self.duration_seconds,
            "context_id": self.context_id,
            "usage": self.usage.as_dict() if self.usage else None,
            "expectation_results": [result.to_dict() for result in self.results],
        }

    def failure_messages(self) -> list[str]:
        """Every failed check labelled by its expectation key, then the Harness Failure.

        Derived on demand — the failed ``CheckResult``s are the only stored
        record of a miss, so this view can never drift from them.
        """
        messages = [
            f"{result.key}: {check.detail}"
            for result in self.results
            for check in result.checks
            if not check.passed
        ]
        if self.harness_error is not None:
            messages.append(str(self.harness_error))
        return messages


@dataclass(slots=True)
class EvaluationResult:
    """The outcome of running a case: its identity plus the Step Trail.

    Per-Step ``response``, ``harness_error``, ``usage``, and the per-term
    ``expectation results`` live only in ``step_results`` — the case exposes no
    duplicated top-level rollup of them, so no consumer can drift from the
    per-Step truth. Case-level views (aggregate usage, the failure summary) are
    derived on demand from the Steps.
    """

    name: str
    success: bool
    agent_id: str | None
    duration_seconds: float | None = None
    step_results: list[StepResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "success": self.success,
            "agent_id": self.agent_id,
            "duration_seconds": self.duration_seconds,
            "step_results": [step.as_dict() for step in self.step_results],
        }

    def aggregate_usage(self) -> UsageMetrics | None:
        """Field-wise sum of usage across Steps; ``None`` when no Step reported any."""
        return UsageMetrics.aggregate([step.usage for step in self.step_results])

    def trace_context_id(self) -> str | None:
        """The last Step's conversation id — the case-level trace a row links to.

        Both sinks link a case row to the thread its final executed Step opened.
        A Harness Failure Step never reached a send, so it carries none and drops
        out of this fold on its own.
        """
        context_id: str | None = None
        for step in self.step_results:
            if step.context_id:
                context_id = step.context_id
        return context_id

    def iter_failures(self) -> Iterator[tuple[str | None, str]]:
        """Yield each Step's failure messages paired with the Step name.

        Failed checks first (labelled by expectation key), then the Step's
        Harness Failure — all derived from the Step Trail on demand.
        """
        for step in self.step_results:
            for message in step.failure_messages():
                yield step.name, message

    def error_summary(self) -> str:
        """One line of every failure, each prefixed by the Step that produced it."""
        return "; ".join(
            f"{name}: {message}" if name else message
            for name, message in self.iter_failures()
        )

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False)


@dataclass
class SuiteResults:
    """Aggregated results from running one suite across multiple runs.

    ``suite_path`` is the path to the YAML file (relative to the project
    root when possible) so the CSV consumer can tell suites apart.
    ``all_results`` holds one list of :class:`EvaluationResult` per run.
    """

    suite_path: str
    all_results: list[list[EvaluationResult]] = field(default_factory=list)
