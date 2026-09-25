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

_TRACE_RETRIES = 20
_TRACE_RETRY_DELAY = 2.0
_TRACE_STABILIZE_DELAY = 2.0
# Consecutive equal counts required before a trace is considered complete.
# With the delay above this is a ~6s quiet window, comfortably wider than the
# exporter's batch period — one agreement only proves we sampled twice inside
# the same lull between flushes.
_TRACE_STABLE_POLLS = 4

_MAX_PAGE_SIZE = 200
_MAX_PAGES = 100


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
    count holds steady across ``_TRACE_STABLE_POLLS`` consecutive fetches,
    so a partial trace does not cause false expectation failures. The window
    matters: a single agreement is satisfied by two samples taken inside one
    lull between flushes, which is how a trace whose CHAIN spans have landed
    but whose TOOL span has not reads as finished. Returns ``None`` if no
    context id is available or the trace never appears within the retry
    budget. Sleeps honor *stop_event* so cancellation stays cooperative.
    """
    if not context_id:
        return None
    last_count = -1
    repeats = 0
    trace: dict[str, Any] | None = None
    for attempt in range(_TRACE_RETRIES):
        if stop_event is not None and stop_event.is_set():
            return None
        try:
            trace = client.get_trace(context_id)
            count = _span_count(trace)
            if count > 0 and count == last_count:
                repeats += 1
                if repeats >= _TRACE_STABLE_POLLS - 1:
                    return trace
            else:
                repeats = 0
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
            context_id,
            _TRACE_RETRIES,
            _span_count(trace),
        )
        return trace
    _LOGGER.warning(
        "no trace available for context %s after %d attempts",
        context_id,
        _TRACE_RETRIES,
    )
    return None


def fetch_all_trace_pages(
    client: AgentClient,
    context_id: str,
    *,
    page_size: int = _MAX_PAGE_SIZE,
    max_pages: int = _MAX_PAGES,
) -> dict[str, Any] | None:
    """Fetch every page of the OpenInference trace for *context_id*.

    Unlike :func:`fetch_trace` (which polls until the span count stabilizes),
    this walks the endpoint's ``pageToken`` / ``nextPageToken`` cursor to
    completion, concatenating the ``traces`` arrays from each page. Returns a
    single ``{"traces": [...]}`` dict, or ``None`` if the context has no
    traces. Used by the ``fetch_traces`` script for one-shot trace dumps; the
    runner uses :func:`fetch_trace` because it needs the settle policy to
    avoid reading a half-exported trace.
    """
    all_traces: list[dict[str, Any]] = []
    page_token: str | None = None
    for _ in range(max_pages):
        response = client.get_trace(
            context_id, page_size=page_size, page_token=page_token
        )
        all_traces.extend(response.get("traces", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    if not all_traces:
        return None
    return {"traces": all_traces}
