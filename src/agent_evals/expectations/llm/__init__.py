"""The LLM-expectation sub-package.

``Judge`` is the base LLM expectation (semantic equivalence via Corti
Models). ``FaithfulToSource`` inherits from it, grading grounding against
retrieved sources. The ``JUDGE_*`` constants are re-exported so
:mod:`~agent_evals.environment` and other consumers can import them from a
single place.
"""

from __future__ import annotations

from .base import (
    DEFAULT_JUDGE_BASE_URL,
    DEFAULT_JUDGE_MODEL,
    JUDGE_API_KEY_VAR,
    JUDGE_BASE_URL_VAR,
    JUDGE_MODEL_VAR,
    Judge,
)
from .faithful import FaithfulToSource

__all__ = [
    "DEFAULT_JUDGE_BASE_URL",
    "DEFAULT_JUDGE_MODEL",
    "JUDGE_API_KEY_VAR",
    "JUDGE_BASE_URL_VAR",
    "JUDGE_MODEL_VAR",
    "Judge",
    "FaithfulToSource",
]
