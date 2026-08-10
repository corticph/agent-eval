"""Trace fetching for the runner: poll the context trace until it stabilizes.

Lives apart from :mod:`~agent_evals.runner` so run-execution logic stays lean
and the fetch/retry policy is the one thing this module owns. ``build_trace_url``
stays in :mod:`~agent_evals.reporting.trace` — the sink-link seam.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from .client import AgentClient

_LOGGER = logging.getLogger(__name__)

_TRACE_RETRIES = 15
_TRACE_RETRY_DELAY = 2.0
_TRACE_STABILIZE_DELAY = 1.0


def _span_count(trace: dict[str, Any] | None) -> int:
    if not trace:
        return 0
    return sum(len(item.get("spans", [])) for item in trace.get("traces", []))


def fetch_trace(
    client: AgentClient,
    context_id: str | None,
    *,
    stop_event: threading.Event | None = None,
) -> dict[str, Any] | None:
    """Fetch the OpenInference trace for *context_id*, retrying until stable.

    The DB span exporter batches asynchronously, so spans appear in
    successive flushes rather than all at once. We retry until the span
    count stabilizes (two consecutive fetches agree), so a partial trace
    does not cause false expectation failures. Returns ``None`` if no
    context id is available or the trace never appears within the retry
    budget. Sleeps honor *stop_event* so cancellation stays cooperative.
    """
    if not context_id:
        return None
    last_count = -1
    trace: dict[str, Any] | None = None
    for attempt in range(_TRACE_RETRIES):
        if stop_event is not None and stop_event.is_set():
            return None
        try:
            trace = client.get_trace(context_id)
            count = _span_count(trace)
            if count > 0 and count == last_count:
                return trace
            last_count = count
        except Exception:
            _LOGGER.debug(
                "trace fetch attempt %d failed for context %s",
                attempt + 1,
                context_id,
                exc_info=True,
            )
        if attempt < _TRACE_RETRIES - 1:
            delay = _TRACE_STABILIZE_DELAY if last_count > 0 else _TRACE_RETRY_DELAY
            if stop_event is not None:
                if stop_event.wait(timeout=delay):
                    return None
            else:
                time.sleep(delay)
    if trace and _span_count(trace) > 0:
        _LOGGER.warning(
            "trace for context %s did not stabilize after %d attempts; "
            "returning last fetch with %d spans",
            context_id, _TRACE_RETRIES, _span_count(trace),
        )
        return trace
    _LOGGER.warning(
        "no trace available for context %s after %d attempts",
        context_id, _TRACE_RETRIES,
    )
    return None
