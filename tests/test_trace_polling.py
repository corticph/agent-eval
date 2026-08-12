"""``fetch_trace``'s settle policy: a quiet window, not a single agreement.

The DB span exporter flushes in batches, so a lull between two flushes looks
exactly like a finished trace to anyone who only requires two agreeing polls —
that is how a trace whose CHAIN spans have landed but whose TOOL span has not
gets accepted, failing a ``trace:`` expectation that the agent actually met.
These pin the wider window that keeps a half-exported trace out.
"""

from __future__ import annotations

import threading
from typing import Any

from agent_evals import tracing
from agent_evals.tracing import fetch_trace


def _trace(span_count: int) -> dict[str, Any]:
    return {"traces": [{"spans": [{"span_id": f"s{i}"} for i in range(span_count)]}]}


class _ScriptedClient:
    """Serves one scripted span count per call; the final entry repeats."""

    def __init__(self, counts: list[int]) -> None:
        self._counts = counts
        self.calls = 0

    def get_trace(self, context_id: str) -> dict[str, Any]:
        count = self._counts[min(self.calls, len(self._counts) - 1)]
        self.calls += 1
        return _trace(count)


def test_lull_between_flushes_is_not_mistaken_for_a_finished_trace() -> None:
    """The regression: 3 spans hold across two polls, then 2 more arrive."""
    client = _ScriptedClient([3, 3, 3, 5, 5, 5, 5])
    trace = fetch_trace(client, "ctx-1")
    # A two-agreeing-polls rule returns the partial trace at the second 3.
    assert tracing._span_count(trace) == 5


def test_returns_once_the_count_holds_for_the_whole_window() -> None:
    client = _ScriptedClient([4])
    trace = fetch_trace(client, "ctx-1")
    assert tracing._span_count(trace) == 4
    assert client.calls == tracing._TRACE_STABLE_POLLS


def test_a_late_span_resets_the_window() -> None:
    """Growth partway through the window restarts it rather than shortening it."""
    client = _ScriptedClient([2, 2, 3, 3, 3, 3])
    trace = fetch_trace(client, "ctx-1")
    assert tracing._span_count(trace) == 3
    assert client.calls == 6


def test_returns_none_when_no_span_ever_appears() -> None:
    client = _ScriptedClient([0])
    assert fetch_trace(client, "ctx-1") is None
    assert client.calls == tracing._TRACE_RETRIES


def test_returns_last_fetch_when_the_trace_never_settles() -> None:
    """A forever-growing trace exhausts the budget rather than looping."""

    class _GrowingClient:
        def __init__(self) -> None:
            self.calls = 0

        def get_trace(self, context_id: str) -> dict[str, Any]:
            self.calls += 1
            return _trace(self.calls)

    client = _GrowingClient()
    trace = fetch_trace(client, "ctx-1")
    assert tracing._span_count(trace) == tracing._TRACE_RETRIES
    assert client.calls == tracing._TRACE_RETRIES


def test_missing_context_id_short_circuits() -> None:
    client = _ScriptedClient([3])
    assert fetch_trace(client, None) is None
    assert client.calls == 0


def test_set_stop_event_cancels_before_any_fetch() -> None:
    client = _ScriptedClient([3])
    stop = threading.Event()
    stop.set()
    assert fetch_trace(client, "ctx-1", stop_event=stop) is None
    assert client.calls == 0
