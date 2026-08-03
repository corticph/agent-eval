"""The trace-link seam: assemble an agent-api Trace URL from its parts.

Shared by both Sinks so the file results and the Opik experiment link a case
row to the same thread through one definition.
"""

from __future__ import annotations


def build_trace_url(base_url: str | None, context_id: str | None) -> str | None:
    """Assemble a trace URL from a resolved base URL and a context id.

    Returns the base with ``context_id`` appended verbatim for a real (non-blank)
    id, and ``None`` when either side is missing: a blank id would make a broken
    URL, and a ``None`` base means the environment has no trace destination
    (local).
    """
    if base_url and context_id and context_id.strip():
        return f"{base_url}{context_id}"
    return None
