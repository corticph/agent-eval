"""Data-driven regression categorization for eval reports.

Each category is a dict with:
- ``key``: stable identifier used in HTML element IDs (``rca-{key}``)
- ``title``: display title in the report
- ``color``: badge color — ``"red"`` (agent bug), ``"amber"`` (mixed/infra), ``"green"`` (improved)
- ``description``: callout box text explaining the pattern
- ``any``: list of substrings — match if **any** appears in the reason (case-insensitive)
- ``all``: list of substrings — match if **all** appear in the reason (case-insensitive)

A category with ``any`` matches if any listed substring is found.
A category with ``all`` matches if every listed substring is found.
Categories are checked in order; the first match wins.

To add a new pattern:
1. Investigate the uncategorized regressions (fetch traces, inspect evals).
2. Identify the common failure reason text.
3. Append a dict to ``CATEGORIES`` with matching ``any`` or ``all`` substrings.
4. Re-run the report generator — the new category appears automatically.

If >10% of regressions fall into "Other", the categorization needs new
patterns — see ``AGENTS.md`` § "Generating eval reports".
"""

from __future__ import annotations

CATEGORIES: list[dict] = [
    {
        "key": "infra-502",
        "title": "Infrastructure: HTTP 502",
        "color": "amber",
        "description": (
            "The agent API returned 502 Bad Gateway during the eval run. "
            "These are transient infrastructure failures, not agent bugs. The cases "
            "scored 0.0 because the agent never responded."
        ),
        "any": ["502", "http 502"],
    },
    {
        "key": "infra-timeout",
        "title": "Infrastructure: Request timeout (connection failure)",
        "color": "amber",
        "description": (
            "The request to the agent API timed out or the connection failed. "
            "These are transient infrastructure failures, not agent bugs."
        ),
        "any": ["connection failed", "could not reach"],
    },
    {
        "key": "infra-timeout",
        "title": "Infrastructure: Request timeout (connection failure)",
        "color": "amber",
        "description": (
            "The request to the agent API timed out or the connection failed. "
            "These are transient infrastructure failures, not agent bugs."
        ),
        "all": ["timed out", "request"],
    },
    {
        "key": "timeout",
        "title": "Performance: Eval timeout (max_duration_seconds exceeded)",
        "color": "amber",
        "description": (
            "The agent is slower, exceeding max_duration_seconds. "
            "The answers are often correct \u2014 only the timeout check fails."
        ),
        "all": ["duration", "exceeded"],
    },
    {
        "key": "premature-completion",
        "title": "Agent bug: Premature task completion",
        "color": "red",
        "description": (
            "The agent completed the task instead of staying in input-required state. "
            "This may indicate a change in the orchestrator's completion signalling."
        ),
        "all": ["input-required", "received", "task_state_completed"],
    },
    {
        "key": "stuck-input-required",
        "title": "Agent bug: Failed to complete task (stuck in input-required)",
        "color": "red",
        "description": (
            "The agent stayed in input-required state when the eval expected it "
            "to complete. The agent may not have gathered enough information to "
            "proceed, or the completion condition was never met."
        ),
        "all": ["expected status", "task_state_input_required"],
    },
    {
        "key": "forbidden-phrase",
        "title": "Agent bug: Forbidden phrase in response",
        "color": "red",
        "description": (
            "The agent's output contains a phrase that the eval explicitly "
            "forbids (e.g., 'CANCEL', internal tool names, or English text "
            "in a German-language context)."
        ),
        "any": ["forbidden phrase"],
    },
    {
        "key": "no-data-parts",
        "title": "Agent bug: Response missing data parts",
        "color": "red",
        "description": (
            "The agent's response has no data parts to evaluate, or no data part "
            "matched the expected structure. The response may be text-only where "
            "structured data parts were expected."
        ),
        "any": ["no data parts", "no data part matched"],
    },
    {
        "key": "rate-limited",
        "title": "External: API rate-limiting (429)",
        "color": "amber",
        "description": (
            "An external API returned HTTP 429 (rate limited). The agent gave up "
            "instead of using the results it already had."
        ),
        "any": ["429"],
    },
    {
        "key": "fabrication",
        "title": "Agent bug: Fabricated citations",
        "color": "red",
        "description": (
            "The agent cites sources not present in the retrieved results."
        ),
        "any": ["fabricated", "not present in retrieved sources"],
    },
    {
        "key": "judge-empty",
        "title": "Eval infra: Judge returned empty content",
        "color": "amber",
        "description": (
            "The LLM judge returned empty content after 3 attempts. This is a judge "
            "infrastructure issue, not an agent bug or eval correctness issue."
        ),
        "any": ["judge returned empty"],
    },
    {
        "key": "wrong-values",
        "title": "Agent bug: Wrong or extra values in response",
        "color": "red",
        "description": (
            "The agent returned wrong or extra values in its response — e.g., "
            "matched ['CLINICAL_PROBLEMS', 'ALERT'] when only 'CLINICAL_PROBLEMS' "
            "was expected, or returned empty results for a required field."
        ),
        "any": ["expected every value ==", "matched [], expected"],
    },
    {
        "key": "clinical-content",
        "title": "Agent bug: Missing expected content",
        "color": "red",
        "description": (
            "The agent's response is missing key terms that the eval expects. "
            "This could be a real content regression or an eval brittleness issue."
        ),
        "any": ["missing required phrase", "no match for required pattern"],
    },
]

OTHER_CATEGORY = {
    "key": "other",
    "title": "Other regressions",
    "color": "amber",
    "description": (
        "Regressions that don't fit the main patterns. Inspect individually."
    ),
}


def categorize(reason: str) -> tuple[str, str, str, str]:
    """Categorize a regression by root cause pattern from the failure reason.

    Returns ``(key, title, color, description)``.
    Categories are checked in order; the first match wins.
    Falls back to ``OTHER_CATEGORY`` when no pattern matches.
    """
    r_lower = (reason or "").lower()
    for cat in CATEGORIES:
        if "any" in cat and any(s in r_lower for s in cat["any"]):
            return (cat["key"], cat["title"], cat["color"], cat["description"])
        if "all" in cat and all(s in r_lower for s in cat["all"]):
            return (cat["key"], cat["title"], cat["color"], cat["description"])
    return (
        OTHER_CATEGORY["key"],
        OTHER_CATEGORY["title"],
        OTHER_CATEGORY["color"],
        OTHER_CATEGORY["description"],
    )
