"""The wall-clock duration expectation."""

from __future__ import annotations

from typing import Any

from .base import (
    CheckResult,
    EvaluationContext,
    Expectation,
    ExpectationResult,
    _PASSED,
)


class MaxDuration(Expectation):
    """The step's wall-clock duration must not exceed the bound (scalar)."""

    key = "max_duration_seconds"

    def __init__(self, limit: float) -> None:
        self.limit = limit

    @classmethod
    def parse(cls, raw: Any) -> "MaxDuration":
        return cls(float(raw))

    def to_raw(self) -> Any:
        return self.limit

    def resolve(self, ctx: EvaluationContext) -> ExpectationResult:
        duration = ctx.duration_seconds
        # An unrecorded duration cannot exceed a bound — it never fails the step.
        exceeded = duration is not None and duration > self.limit
        detail = (
            f"duration {duration:.3f}s exceeded max_duration_seconds={self.limit:.3f}s"
            if exceeded
            else _PASSED
        )
        return ExpectationResult(self.key, [CheckResult("", not exceeded, detail)])
