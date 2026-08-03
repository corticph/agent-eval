"""The LLM-expectation base class.

Judge is a concrete, functional base class (``key="expected_output_text"``)
that checks semantic equivalence of an agent response to a reference answer
via Corti Models. It keeps its default prompt and Corti Models call, and
exposes two hooks — ``_needs_judge`` and ``_build_prompt`` — so subclasses
( FaithfulToSource, ticket 02) can specialise behaviour without touching the
Corti call.

Verdicts come from Corti Models with the customer's own credentials — one
stateless chat-completions call per verdict. The bearer is the Corti
Console's ``CORTI_*`` credentials assembled by :func:`~agent_evals.
environment.corti_models_api_key`, or a pre-built ``JUDGE_API_KEY`` which
overrides them; ``JUDGE_BASE_URL`` and ``JUDGE_MODEL`` pick the endpoint and
model. The harness never mints or caches a credential, and a judge failure
is a failed check carrying the error — there is no degraded approximation.
"""

from __future__ import annotations

import json
import os
from typing import Any

import openai

from ...environment import corti_models_api_key
from ..base import (
    CheckResult,
    EvaluationContext,
    Expectation,
    ExpectationResult,
    _PASSED,
)

# The judge's env vars and defaults, declared once here: the configuration
# knob inventory imports them, so --help and behaviour cannot drift.
JUDGE_API_KEY_VAR = "JUDGE_API_KEY"
JUDGE_BASE_URL_VAR = "JUDGE_BASE_URL"
JUDGE_MODEL_VAR = "JUDGE_MODEL"
DEFAULT_JUDGE_BASE_URL = "https://ai.eu.corti.app/v1"
DEFAULT_JUDGE_MODEL = "corti-s1-instant"

_JUDGE_PROMPT = """\
You are an impartial evaluator assessing whether an AI agent's response is \
semantically equivalent to a reference answer.

Reference answer:
{expected}

Agent's actual response (may be JSON-encoded; extract the relevant text):
{actual}

Does the agent's response convey the same key information as the reference answer?
Consider it a PASS if the core facts are correct and present and no dangerous or \
incorrect information is stated that contradicts the reference.

Respond with JSON in exactly this shape:
{{"result": "PASS", "explanation": "..."}}
or
{{"result": "FAIL", "explanation": "..."}}

Keep the explanation to ONE sentence (max 15 words)."""


def _resolve_api_key() -> str | None:
    """The judge's bearer: an explicit key wins, else the CORTI_* composite."""
    return os.environ.get(JUDGE_API_KEY_VAR) or corti_models_api_key()


class Judge(Expectation):
    """The response must be semantically equivalent to a reference answer.

    The base LLM expectation: subclasses inherit the Corti Models call
    (``_judge``), the preflight credential check, and the ``resolve``
    template, overriding ``_needs_judge`` and ``_build_prompt`` to
    specialise.
    """

    key = "expected_output_text"

    def __init__(self, reference: str | None) -> None:
        self.reference = reference

    @classmethod
    def parse(cls, raw: Any) -> "Judge":
        if isinstance(raw, list):
            reference = "\n".join(str(line) for line in raw) or None
        else:
            reference = raw or None
        return cls(reference)

    def to_raw(self) -> Any:
        return self.reference

    def _needs_judge(self) -> bool:
        """Whether a judge call is required (defaults to ``bool(self.reference)``).

        Both ``preflight`` and ``resolve`` call this instead of poking
        ``self.reference`` directly, so a subclass with a different skip
        condition overrides only this hook.
        """
        return bool(self.reference)

    def _build_prompt(self, ctx: EvaluationContext) -> str:
        """Build the judge prompt from the response context.

        Defaults to the semantic-equivalence prompt using ``ctx.plain_text``
        and ``self.reference``.
        """
        return _JUDGE_PROMPT.format(expected=self.reference, actual=ctx.plain_text)

    @staticmethod
    def _judge(prompt: str) -> tuple[bool, str]:
        """Call Corti Models with *prompt*, returning ``(passed, explanation)``.

        Resolves credentials, base URL, and model from the ``JUDGE_*`` env
        vars. Any failure — connection error, API error, unparseable or
        malformed reply — returns ``(False, <error detail>)`` so the check
        fails carrying the error. No degraded approximation is used.
        """
        client = openai.OpenAI(
            base_url=os.environ.get(JUDGE_BASE_URL_VAR) or DEFAULT_JUDGE_BASE_URL,
            # "" rather than unset: the SDK must never fall back to OPENAI_API_KEY
            # or raise at construction — a missing key fails the check at call time.
            api_key=_resolve_api_key() or "",
        )
        model = os.environ.get(JUDGE_MODEL_VAR) or DEFAULT_JUDGE_MODEL
        try:
            completion = client.chat.completions.create(
                model=model,
                # Corti Models' published schema takes max_tokens, not
                # max_completion_tokens.
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            return False, f"judge call failed: {exc}"
        content = (completion.choices[0].message.content or "").strip()
        try:
            data = json.loads(content)
            result = data["result"]
            if result not in ("PASS", "FAIL"):
                raise ValueError(f"unexpected result value: {result!r}")
        except (json.JSONDecodeError, TypeError, KeyError, ValueError) as exc:
            return False, f"judge verdict malformed ({exc}): {content!r}"
        return result == "PASS", data.get("explanation", "")

    def preflight(self) -> str | None:
        if self._needs_judge() and _resolve_api_key() is None:
            return (
                f"No judge credentials: this suite declares {self.key} "
                "expectations, and judge verdicts are Corti Models calls. "
                "Set CORTI_CLIENT_ID and CORTI_CLIENT_SECRET from the Corti "
                f"Console, or a pre-built {JUDGE_API_KEY_VAR} bearer "
                "(see example.env)."
            )
        return None

    def resolve(self, ctx: EvaluationContext) -> ExpectationResult:
        if not self._needs_judge():
            return ExpectationResult(
                self.key, [CheckResult("", True, _PASSED)], show_on_pass=True
            )
        prompt = self._build_prompt(ctx)
        passed, explanation = self._judge(prompt)
        return ExpectationResult(
            self.key, [CheckResult("", passed, explanation)], show_on_pass=True
        )
