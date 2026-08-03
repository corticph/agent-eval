"""The cited-PMID anti-fabrication expectation.

Deterministic (no LLM): every PMID the agent cites in its prose must be a
member of the PMIDs present in the retrieved sources (the data parts). Fails
naming any cited PMID absent from the retrieved set.
"""

from __future__ import annotations

import re
from typing import Any

from .base import CheckResult, EvaluationContext, Expectation, ExpectationResult
from .shared.utils import extract_source_at

_PMID_CITE_PATTERN = re.compile(r"\bPMID:?\s*(\d+)", re.IGNORECASE)
_PMID_URL_PATTERN = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)")


def _extract_pmids(text: str) -> set[str]:
    """Collect PMIDs mentioned as ``PMID <n>`` or inside a pubmed.ncbi URL."""
    ids: set[str] = set()
    for pattern in (_PMID_CITE_PATTERN, _PMID_URL_PATTERN):
        ids.update(match.group(1) for match in pattern.finditer(text or ""))
    return ids


class CitedPubmedIds(Expectation):
    """Every PMID cited in the agent's summary must appear in retrieved sources.

    Deterministic (no LLM): the retrieved set is the PMIDs present in the tool
    output carrier (``$.response``); the cited set is the PMIDs in the agent's
    summary. Any cited PMID not in the retrieved set is a fabricated citation
    and fails, named explicitly. No citations passes; no sources with a
    citation fails (nothing to ground to).
    """

    key = "cited_pubmed_ids"

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    @classmethod
    def parse(cls, raw: Any) -> "CitedPubmedIds":
        return cls(bool(raw))

    def to_raw(self) -> Any:
        return self.enabled

    def resolve(self, ctx: EvaluationContext) -> ExpectationResult:
        if not self.enabled:
            return ExpectationResult(self.key, [])

        cited = _extract_pmids(ctx.plain_text)
        retrieved = _extract_pmids(extract_source_at(ctx.response, "$.response"))
        fabricated = sorted(cited - retrieved, key=int)
        if fabricated:
            detail = (
                f"cited PMID(s) not present in retrieved sources: {fabricated} "
                f"(retrieved: {sorted(retrieved, key=int) or 'none'})"
            )
            return ExpectationResult(
                self.key,
                [
                    CheckResult(
                        label="",
                        passed=False,
                        detail=detail,
                    )
                ],
            )
        return ExpectationResult(self.key, [CheckResult("", True, "passed")])
