"""The faithfulness-to-source LLM expectation.

``FaithfulToSource`` inherits from :class:`~agent_evals.expectations.llm.base.
Judge` and overrides four members — ``key``, ``_needs_judge``, ``_build_prompt``,
and ``parse``/``to_raw`` — to grade grounding (no claim unsupported by or
contradicting the retrieved sources) rather than paraphrase similarity to a
reference. It rides ``Judge._judge`` (the Corti Models call), the credential
``preflight``, and the ``resolve`` template unchanged.

The agent's summary makes no claim unsupported by, or contradicting, the
retrieved sources. This is universal — not PubMed-specific — it grades
grounding against any retrieved source content.
"""

from __future__ import annotations

from typing import Any

from ..base import EvaluationContext
from ..shared.utils import extract_source_at
from .base import Judge

_FAITHFULNESS_PROMPT = """\
You are an impartial evaluator checking whether an AI agent's summary is \
FAITHFUL to the source documents it retrieved.

Faithful means EVERY claim in the summary is supported by the sources below, and \
NOTHING in it contradicts them. Fabricated citations, invented findings, or \
embellishment beyond what the sources state are UNFAITHFUL. If the sources are \
empty or thin and do not support the summary's assertions, that is UNFAITHFUL — \
UNLESS the summary honestly states that no supporting evidence was found or \
declines to assert findings, which is FAITHFUL.

Retrieved sources:
{sources}

Agent's summary (may be JSON-encoded; extract the relevant text):
{summary}

Respond with JSON in exactly this shape:
{{"result": "PASS", "explanation": "..."}}
or
{{"result": "FAIL", "explanation": "..."}}

PASS = faithful to the sources. FAIL = unfaithful (unsupported, fabricated, or \
contradicting). Keep the explanation to ONE sentence (max 20 words)."""


def _normalize_faithful_config(raw: Any) -> dict[str, Any] | None:
    """Coerce a ``faithful_to_source`` value into a config dict or ``None``.

    ``true`` enables it with defaults; a mapping is used as-is (recognised key:
    ``source_path``); ``None`` / ``false`` disables it. Any other type raises.
    """
    if raw is None or raw is False:
        return None
    if raw is True:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    raise TypeError(
        f"faithful_to_source must be a bool or mapping, got {type(raw).__name__}"
    )


class FaithfulToSource(Judge):
    """The agent's summary must be faithful to its retrieved sources.

    An LLM judge checks that every claim in the summary is supported by the
    sources the agent retrieved (the data parts at ``source_path``).
    Fabricated citations, invented findings, or embellishment beyond what the
    sources state fail. Honest "no evidence found" summaries pass.
    """

    key = "faithful_to_source"

    def __init__(self, config: dict[str, Any] | None) -> None:
        self.config = config

    @classmethod
    def parse(cls, raw: Any) -> "FaithfulToSource":
        return cls(_normalize_faithful_config(raw))

    def to_raw(self) -> Any:
        if self.config is None:
            return False
        if not self.config:
            return True
        return dict(self.config)

    def _needs_judge(self) -> bool:
        return self.config is not None

    def _build_prompt(self, ctx: EvaluationContext) -> str:
        source_path = self.config.get("source_path", "$.response")
        sources = extract_source_at(ctx.response, source_path)
        return _FAITHFULNESS_PROMPT.format(
            sources=sources.strip()
            if sources and sources.strip()
            else "(no sources were retrieved)",
            summary=ctx.plain_text,
        )
