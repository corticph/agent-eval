"""The terminal task-state expectation — assertion, guard, and threading signal."""

from __future__ import annotations

from typing import Any

from .base import (
    CheckResult,
    EvaluationContext,
    Expectation,
    ExpectationResult,
    _PASSED,
)

# Agent API v2 reports task state as the A2A ``TASK_STATE_*`` enum
# (e.g. ``TASK_STATE_INPUT_REQUIRED``). We normalise to a bare canonical token
# so eval authors can keep writing the readable short form
# (``input-required``) and it still matches the wire value.
# ``REJECTED`` is retained alongside the v2 ``CANCELED``/``FAILED`` for
# backward compatibility with deployments still emitting the v1 vocabulary.
_TERMINAL_FAILURE_STATES = frozenset({"FAILED", "CANCELED", "REJECTED"})


def normalize_task_state(state: str | None) -> str | None:
    """Reduce a task-state string to a canonical token.

    Accepts the v2 wire enum (``TASK_STATE_INPUT_REQUIRED``), the human short
    form (``input-required``), or snake case; all collapse to the same token
    (``INPUT_REQUIRED``). Returns None for empty/non-string input.
    """
    if not isinstance(state, str):
        return None
    token = state.strip().upper().replace("-", "_")
    if not token:
        return None
    if token.startswith("TASK_STATE_"):
        token = token[len("TASK_STATE_") :]
    return token


class ExpectedState(Expectation):
    """The response's terminal task state — an assertion, a guard, and a signal.

    When a state is declared, the response's normalized state must equal it. When
    none is declared, an always-on guard fails a step that ends in an undeclared
    terminal *failure* (``FAILED``/``CANCELED``/``REJECTED``), forcing the author
    to opt in to such an outcome. Declaring ``input-required`` also threads the
    task into the next step (see :meth:`continues_task`).
    """

    key = "expected_state"

    def __init__(self, expected: str | None) -> None:
        self.expected = expected

    @classmethod
    def parse(cls, raw: Any) -> "ExpectedState":
        return cls(raw or None)

    def to_raw(self) -> Any:
        return self.expected

    def is_injected_default(self) -> bool:
        # The guard with no declared state is the one auto-injected on every step.
        return self.expected is None

    def resolve(self, ctx: EvaluationContext) -> ExpectationResult:
        state = ctx.task_state
        normalized = normalize_task_state(state)
        if self.expected is not None:
            passed = normalized == normalize_task_state(self.expected)
            state_display = repr(state) if state else "no status"
            detail = (
                _PASSED
                if passed
                else f"expected status {self.expected!r} but received {state_display}"
            )
        else:
            passed = normalized not in _TERMINAL_FAILURE_STATES
            detail = (
                _PASSED
                if passed
                else f"task ended in terminal state {state!r}; declare "
                f"expected_state: {state} if this outcome is intended"
            )
        return ExpectationResult(self.key, [CheckResult("", passed, detail)])

    def continues_task(self) -> bool:
        return normalize_task_state(self.expected) == "INPUT_REQUIRED"
