"""The expectation vocabulary in one package.

Each expectation *type* an author can declare under an ``expectations:`` block
is one self-registering :class:`Expectation` subclass, each in its own module.
Importing the package imports every type module below — in canonical
(definition) order — so the registry lists the full vocabulary and both parse
and resolve iterate in a stable order.

``resolve`` emits per-term structure with passes included as its primary output:
the sole per-term truth both the local and Opik sinks read, rather than a
failure-only error list each has to reverse-map. Matching semantics (substring
rules, jsonpath comparators, partial-JSON matching, state normalization) are the
existing ones — reused rather than reimplemented, so there is one source of
truth for what each check means.
"""

from __future__ import annotations

from .base import (
    CheckResult,
    EvaluationContext,
    Expectation,
    ExpectationResult,
    evaluate_response,
    parse_expectations,
    registry,
    resolve_all,
)

# Import order is canonical order: it seeds the registry and thus every
# result list and sink column. Keep it aligned with the type enumeration.
from .must_include import MustInclude, MustNotInclude
from .json import JsonPath, MustIncludeJson, MustMatch
from .duration import MaxDuration
from .llm import FaithfulToSource, Judge
from .cited_pubmed_ids import CitedPubmedIds
from .state import ExpectedState
from .trace import Trace

__all__ = [
    "CheckResult",
    "CitedPubmedIds",
    "EvaluationContext",
    "Expectation",
    "ExpectationResult",
    "ExpectedState",
    "FaithfulToSource",
    "Judge",
    "JsonPath",
    "MaxDuration",
    "MustInclude",
    "MustIncludeJson",
    "MustMatch",
    "MustNotInclude",
    "Trace",
    "evaluate_response",
    "parse_expectations",
    "registry",
    "resolve_all",
]
