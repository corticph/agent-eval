"""Pattern and structured-data expectations: regex, JSONPath, partial JSON."""

from __future__ import annotations

import json
import re
from typing import Any

from .base import (
    CheckResult,
    EvaluationContext,
    Expectation,
    ExpectationResult,
    _term_check,
)
from .shared.utils import extract_data_parts

# Comparator keys recognised in a jsonpath assertion (besides ``path``).
_JSONPATH_COMPARATORS = frozenset(
    {"equals", "contains", "length", "count", "min", "max", "regex", "exists"}
)


def _partial_json_match(pattern: Any, actual: Any) -> bool:
    """Return True if *pattern* is a partial match of *actual*.

    - Dict: every key in *pattern* must exist in *actual* and match recursively.
    - List: every element in *pattern* must have at least one counterpart in
      *actual* that partially matches (order-independent subset semantics).
    - Scalar: type-sensitive equality (bool kept distinct from int).
    """
    if isinstance(pattern, bool) or isinstance(actual, bool):
        return (
            isinstance(pattern, bool) and isinstance(actual, bool) and pattern == actual
        )
    if isinstance(pattern, dict):
        if not isinstance(actual, dict):
            return False
        return all(
            k in actual and _partial_json_match(v, actual[k])
            for k, v in pattern.items()
        )
    if isinstance(pattern, list):
        if not isinstance(actual, list):
            return False
        return all(any(_partial_json_match(p, a) for a in actual) for p in pattern)
    if isinstance(pattern, (int, float)):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and pattern == actual
        )
    return pattern == actual


def _check_jsonpath_comparators(
    assertion: dict[str, Any], matches: list[Any]
) -> str | None:
    """Return ``None`` if all comparators hold for *matches*, else a detail string."""
    if "exists" in assertion:
        want = bool(assertion["exists"])
        has = len(matches) > 0
        if has != want:
            return f"exists={has}, expected {want}"
    if "length" in assertion:
        # Collection-aware: a single matched list/str reports its own length
        # (so `$.items` length 4 means "the array has 4 elements"); otherwise
        # falls back to the number of matched nodes.
        want_n = int(assertion["length"])
        if len(matches) == 1 and isinstance(matches[0], (list, str)):
            actual_n = len(matches[0])
        else:
            actual_n = len(matches)
        if actual_n != want_n:
            return f"length {actual_n}, expected {want_n}"
    if "count" in assertion:
        # Raw number of matched nodes (use `$.items[*]` to count elements).
        want_n = int(assertion["count"])
        if len(matches) != want_n:
            return f"matched {len(matches)} node(s), expected {want_n}"
    if "equals" in assertion:
        want = assertion["equals"]
        if isinstance(want, list):
            if matches != want:
                return f"matched {matches!r}, expected exactly {want!r}"
        elif not matches or any(not _deep_match(want, m) for m in matches):
            return f"matched {matches!r}, expected every value == {want!r}"
    if "contains" in assertion:
        for needle in assertion["contains"]:
            if not any(_deep_match(needle, m) for m in matches):
                return f"matched {matches!r} is missing {needle!r}"
    if "min" in assertion:
        bound = assertion["min"]
        bad = [
            m
            for m in matches
            if not (
                isinstance(m, (int, float)) and not isinstance(m, bool) and m >= bound
            )
        ]
        if not matches or bad:
            return f"values {bad or matches!r} not all >= {bound}"
    if "max" in assertion:
        bound = assertion["max"]
        bad = [
            m
            for m in matches
            if not (
                isinstance(m, (int, float)) and not isinstance(m, bool) and m <= bound
            )
        ]
        if not matches or bad:
            return f"values {bad or matches!r} not all <= {bound}"
    if "regex" in assertion:
        pattern = re.compile(assertion["regex"])
        bad = [m for m in matches if not (isinstance(m, str) and pattern.search(m))]
        if not matches or bad:
            return f"values {bad or matches!r} do not all match /{assertion['regex']}/"
    return None


def _deep_match(expected: Any, actual: Any) -> bool:
    """Type-sensitive deep match. Dicts are matched as subsets of *actual*."""
    # bool is a subclass of int — keep them distinct so True != 1.
    if isinstance(expected, bool) or isinstance(actual, bool):
        return (
            isinstance(expected, bool)
            and isinstance(actual, bool)
            and expected == actual
        )
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(
            key in actual and _deep_match(val, actual[key])
            for key, val in expected.items()
        )
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return False
        return all(_deep_match(e, a) for e, a in zip(expected, actual))
    if isinstance(expected, (int, float)):
        return isinstance(actual, (int, float)) and expected == actual
    return expected == actual


class MustMatch(Expectation):
    """Each regex must match (``re.search``) the response's full JSON.

    Use when a phrase has acceptable synonyms — an alternation ``must_include``
    (literal AND) cannot express.
    """

    key = "must_match"

    def __init__(self, patterns: list[str]) -> None:
        self.patterns = patterns

    @classmethod
    def parse(cls, raw: Any) -> "MustMatch":
        return cls(list(raw or []))

    def to_raw(self) -> Any:
        return list(self.patterns)

    def term_labels(self) -> list[str]:
        return list(self.patterns)

    def resolve(self, ctx: EvaluationContext) -> ExpectationResult:
        checks = [
            _term_check(
                pattern,
                bool(re.search(pattern, ctx.haystack)),
                f"no match for required pattern: {pattern!r}",
            )
            for pattern in self.patterns
        ]
        return ExpectationResult(self.key, checks)


class JsonPath(Expectation):
    """JSONPath assertions against the response's data part(s), one check per path.

    Each assertion is a mapping with a ``path`` and one or more AND-ed
    comparators; it holds if it holds against at least one data part.
    """

    key = "jsonpath"

    def __init__(self, assertions: list[Any]) -> None:
        self.assertions = assertions

    @classmethod
    def parse(cls, raw: Any) -> "JsonPath":
        return cls(list(raw or []))

    def to_raw(self) -> Any:
        return list(self.assertions)

    def term_labels(self) -> list[str]:
        return [self._label(assertion) for assertion in self.assertions]

    def resolve(self, ctx: EvaluationContext) -> ExpectationResult:
        data_parts = extract_data_parts(ctx.response)
        checks = [self._check(assertion, data_parts) for assertion in self.assertions]
        return ExpectationResult(self.key, checks)

    @staticmethod
    def _label(assertion: Any) -> str:
        label = (
            assertion.get("path", "?")
            if isinstance(assertion, dict)
            else repr(assertion)
        )
        return str(label)

    def _check(self, assertion: Any, data_parts: list[Any]) -> CheckResult:
        label = self._label(assertion)
        failure = self._failure(assertion, data_parts)
        return CheckResult(
            label=label, passed=failure is None, detail=failure or "passed"
        )

    @staticmethod
    def _failure(assertion: Any, data_parts: list[Any]) -> str | None:
        """Return ``None`` if the assertion holds, else its failure message."""
        from jsonpath_ng.ext import parse as parse_jsonpath  # noqa: PLC0415

        if not isinstance(assertion, dict):
            return f"jsonpath assertion must be a mapping, got: {assertion!r}"
        path = assertion.get("path")
        if not path:
            return f"jsonpath assertion missing 'path': {assertion!r}"
        unknown = {
            k for k in assertion if k != "path" and k not in _JSONPATH_COMPARATORS
        }
        if unknown:
            return f"unknown key(s) {sorted(unknown)!r}"
        comparators = {
            k for k in assertion if k != "path" and k in _JSONPATH_COMPARATORS
        }
        if not comparators:
            return f"no comparator specified (known: {sorted(_JSONPATH_COMPARATORS)!r})"
        if not data_parts:
            return "response has no data parts to evaluate"
        try:
            expr = parse_jsonpath(path)
        except Exception as exc:  # malformed path
            return f"invalid path ({exc})"

        last_detail: str | None = None
        for part in data_parts:
            try:
                matches = [m.value for m in expr.find(part)]
                detail = _check_jsonpath_comparators(assertion, matches)
            except Exception as exc:
                last_detail = f"error evaluating comparator ({exc})"
                continue
            if detail is None:
                return None  # held against at least one part
            last_detail = detail
        return last_detail


class MustIncludeJson(Expectation):
    """Each fragment must partially match at least one data part, one check each.

    Dicts match as subsets; list elements match order-independently.
    """

    key = "must_include_json"

    def __init__(self, patterns: list[Any]) -> None:
        self.patterns = patterns

    @classmethod
    def parse(cls, raw: Any) -> "MustIncludeJson":
        return cls(list(raw or []))

    def to_raw(self) -> Any:
        return list(self.patterns)

    def term_labels(self) -> list[str]:
        return [self._label(pattern) for pattern in self.patterns]

    @staticmethod
    def _label(pattern: Any) -> str:
        return json.dumps(pattern, ensure_ascii=False, sort_keys=True)

    def resolve(self, ctx: EvaluationContext) -> ExpectationResult:
        data_parts = extract_data_parts(ctx.response)
        checks: list[CheckResult] = []
        for pattern in self.patterns:
            label = self._label(pattern)
            if not data_parts:
                checks.append(
                    CheckResult(label, False, "response has no data parts to evaluate")
                )
            else:
                passed = any(_partial_json_match(pattern, part) for part in data_parts)
                checks.append(_term_check(label, passed, "no data part matched"))
        return ExpectationResult(self.key, checks)
