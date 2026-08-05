"""The expectation model: the carriers, the context, and the registry.

Each expectation *type* is one self-registering :class:`Expectation` subclass
(defined in a sibling module). This file holds what they all share — the
response views the context is built from, the per-term result carriers, the
per-step context, the ABC, the strict parse that turns an ``expectations:``
block into a homogeneous, canonically ordered list, and the resolution front
door (:func:`evaluate_response`).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from ..schemas.response import Response

_PASSED = "passed"


def _task_without_history(response: dict[str, Any] | None) -> dict[str, Any]:
    """Return the task payload with conversation ``history`` stripped out."""
    if response is None:
        return {}
    task = response.get("task", response)
    if not isinstance(task, dict):
        return {}
    return {k: v for k, v in task.items() if k != "history"}


def extract_textual_response(response: dict[str, Any] | None) -> str:
    """Build a string representation for expectation evaluation.

    Serializes the full task response with history stripped out, so
    must_include/must_not_include checks are not contaminated by previous
    conversation turns.
    """
    if response is None:
        return ""
    return json.dumps(_task_without_history(response), ensure_ascii=False)


def extract_plain_text(response: dict[str, Any] | None) -> str:
    """Extract only the human-readable text from a response.

    Collects text parts from ``task.status.message.parts`` and
    ``task.artifacts[].parts``, returning them joined by newlines.
    This is used for the LLM judge so that JSON structure noise does not
    distort its verdict.
    """
    if response is None:
        return ""
    task = response.get("task", response)
    texts: list[str] = []

    status_msg = task.get("status", {}).get("message", {})
    if isinstance(status_msg, dict):
        for part in status_msg.get("parts", []):
            text = _part_text(part)
            if text is not None:
                texts.append(text)

    for artifact in task.get("artifacts", []):
        if not isinstance(artifact, dict):
            continue
        for part in artifact.get("parts", []):
            text = _part_text(part)
            if text is not None:
                texts.append(text)

    return "\n".join(texts)


def _part_text(part: Any) -> str | None:
    """Return a part's text, or None if it is not a text part.

    Agent API v2 Parts carry no ``kind`` discriminator: a text part is simply
    any part with a string ``text`` field. This also accepts the v1
    ``{kind: "text", ...}`` shape, since the ``text`` field is present either
    way.
    """
    if isinstance(part, dict) and isinstance(part.get("text"), str):
        return part["text"]
    return None


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One term's verdict: a phrase, pattern, or path — and whether it held.

    ``label`` names the term (``""`` for a scalar expectation that has a single,
    unnamed term); ``detail`` is ``"passed"`` or the failure message.
    """

    label: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "passed": self.passed, "detail": self.detail}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckResult":
        return cls(label=data["label"], passed=data["passed"], detail=data["detail"])


@dataclass(frozen=True, slots=True)
class ExpectationResult:
    """The resolved outcome of one expectation — its key plus its checks.

    One concrete, never-subtyped carrier: per-type knowledge ends at ``resolve``
    (which labels the checks), so both sinks render generically over ``checks``
    without branching on type. ``show_on_pass`` is the judge's sole deviation —
    its explanation is worth showing even when it passes.
    """

    key: str
    checks: list[CheckResult]
    show_on_pass: bool = False

    @property
    def passed(self) -> bool:
        """The expectation holds iff every one of its checks held."""
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "checks": [check.to_dict() for check in self.checks],
            "show_on_pass": self.show_on_pass,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExpectationResult":
        """Reconstruct a result serialized by :meth:`to_dict`.

        The Opik sink stores each executed step's results on its task output and
        re-reads them here to render, so the per-term verdict survives the
        round-trip through the dataset item without being re-derived.
        """
        return cls(
            key=data["key"],
            checks=[CheckResult.from_dict(check) for check in data["checks"]],
            show_on_pass=data.get("show_on_pass", False),
        )


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    """The response views every expectation type might need, built once per step.

    Bundling them means each ``resolve`` reads from a uniform surface instead of
    re-deriving text, data parts, or task state on its own.
    """

    haystack: str  # full JSON, history stripped — substring checks
    plain_text: str  # human-readable text only — the judge
    response: dict[str, Any] | None  # parsed response tree — jsonpath / json checks
    task_state: str | None  # raw terminal state string — expected_state
    duration_seconds: float | None
    trace: dict[str, Any] | None = None  # OpenInference trace JSON — trace expectations
    trace_url: str | None = None  # Opik trace URL — included in trace failure details

    @classmethod
    def from_response(
        cls,
        response: dict[str, Any] | None,
        *,
        duration_seconds: float | None = None,
        trace: dict[str, Any] | None = None,
        trace_url: str | None = None,
    ) -> "EvaluationContext":
        return cls(
            haystack=extract_textual_response(response),
            plain_text=extract_plain_text(response),
            response=response,
            task_state=Response.from_dict(response).state,
            duration_seconds=duration_seconds,
            trace=trace,
            trace_url=trace_url,
        )


_REGISTRY: dict[str, type["Expectation"]] = {}


class Expectation(ABC):
    """One expectation type: parses its own params, resolves them to checks.

    Subclasses self-register under their ``key`` at import; a missing or
    duplicate key is a load error, so a broken registration cannot ship
    silently. Definition order is the canonical order the registry preserves.
    """

    key: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        key = cls.__dict__.get("key")
        if not isinstance(key, str) or not key:
            raise TypeError(
                f"Expectation subclass {cls.__name__!r} must define a non-empty 'key'"
            )
        if key in _REGISTRY:
            raise TypeError(
                f"duplicate expectation key {key!r}: {cls.__name__} collides with "
                f"{_REGISTRY[key].__name__}"
            )
        _REGISTRY[key] = cls

    @classmethod
    @abstractmethod
    def parse(cls, raw: Any) -> "Expectation":
        """Build an instance from the raw YAML value under this type's key."""

    @abstractmethod
    def resolve(self, ctx: EvaluationContext) -> ExpectationResult:
        """Evaluate against *ctx* into one result carrying per-term checks."""

    @abstractmethod
    def to_raw(self) -> Any:
        """The YAML value this type was parsed from — the inverse of :meth:`parse`.

        Lets a parsed expectation be re-serialized into an ``expectations:`` block
        that :func:`parse_expectations` reads back identically, so the Opik
        dataset item can carry each step's declared expectations for the sink to
        render a step that never ran.
        """

    def continues_task(self) -> bool:
        """Whether this expectation signals the case should thread into the next
        step. Only ``expected_state: input-required`` does; the runner reduces
        over this hook so threading stays polymorphic (no ``isinstance``)."""
        return False

    def is_injected_default(self) -> bool:
        """True when this instance was auto-injected rather than authored.

        Only the default terminal-state guard (an ``expected_state`` with no
        declared value) is: it runs on every step but is not something an author
        wrote, so its passing result is a non-event a sink should not surface as
        a column, and it is dropped from re-serialization (parsing re-injects it)
        rather than persisted as a phantom key."""
        return False

    def preflight(self) -> str | None:
        """Why a run containing this expectation cannot start, or ``None``.

        The runner reduces over this during its fail-fast phase — before any
        agent is provisioned or message sent — so a misconfiguration (say, a
        credential ``resolve`` will need) aborts the run at start instead of
        failing every check mid-run."""
        return None

    def term_labels(self) -> list[str]:
        """The check labels :meth:`resolve` will emit, in order.

        Lets a sink synthesize a column for a step that never ran — labelling
        each declared term as failed — without evaluating (and so without the
        judge's side effects). A scalar has one unnamed term."""
        return [""]

    def failed_placeholder(self, reason: str) -> ExpectationResult:
        """An all-failed result standing in for a step that never produced one.

        Every declared term is marked failed with *reason* (a task-level failure,
        or an earlier step that stopped the case), so a case that died early
        cannot read as a near-pass. No evaluation happens."""
        return ExpectationResult(
            self.key,
            [CheckResult(label, False, reason) for label in self.term_labels()],
        )


def registry() -> dict[str, type[Expectation]]:
    """The canonical enumeration of every expectation type, in definition order.

    A copy, so a caller enumerating the vocabulary cannot mutate the live
    registry the whole harness resolves against.
    """
    return dict(_REGISTRY)


def _term_check(label: str, passed: bool, failure: str) -> CheckResult:
    """A list-term check: ``"passed"`` on success, else the failure message."""
    return CheckResult(
        label=label, passed=passed, detail=_PASSED if passed else failure
    )


def parse_expectations(block: dict[str, Any] | None) -> list["Expectation"]:
    """Parse an ``expectations:`` block into expectations in canonical order.

    Every declared key is looked up in the registry — an unknown key is a load
    error listing the known keys, so a typo fails loudly instead of passing
    vacuously. A default ``expected_state`` is injected when none is declared so
    the terminal-failure guard runs on every step. The returned list follows
    canonical (class-definition) order, not authoring order, so both sinks emit
    stable columns across cases.
    """
    # Deferred so the base need not import the type module that imports it back.
    from .state import ExpectedState  # noqa: PLC0415

    parsed: dict[str, Expectation] = {}
    for raw_key, value in (block or {}).items():
        expectation_cls = _REGISTRY.get(raw_key)
        if expectation_cls is None:
            known = ", ".join(_REGISTRY)
            raise ValueError(
                f"unknown expectation key {raw_key!r}; known keys: {known}"
            )
        parsed[raw_key] = expectation_cls.parse(value)
    parsed.setdefault(ExpectedState.key, ExpectedState.parse(None))
    return [parsed[key] for key in _REGISTRY if key in parsed]


def resolve_all(
    expectations: list["Expectation"], ctx: EvaluationContext
) -> list[ExpectationResult]:
    """Resolve every expectation against *ctx*, preserving canonical order."""
    return [expectation.resolve(ctx) for expectation in expectations]


def evaluate_response(
    expectations: list["Expectation"],
    response: dict[str, Any] | None,
    *,
    duration_seconds: float | None = None,
    trace: dict[str, Any] | None = None,
    trace_url: str | None = None,
) -> list[ExpectationResult]:
    """Resolve *expectations* against *response* into per-term results.

    The one evaluation seam: it builds the step's :class:`EvaluationContext`
    once and maps each expectation to its :class:`ExpectationResult` in canonical
    order. The returned list is the single per-term truth both sinks render from;
    a miss exists only as its failed check within it.
    """
    ctx = EvaluationContext.from_response(
        response,
        duration_seconds=duration_seconds,
        trace=trace,
        trace_url=trace_url,
    )
    return resolve_all(expectations, ctx)
