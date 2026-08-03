"""Equivalence tests for the typed ``Response`` accessors.

`Response.state` / `.resolved_context_id()` / `.resolved_task_id()` replaced the
dict crawlers `_extract_status_state` / `_extract_context_id` / `_extract_task_id`
that used to live in ``runner.py``. The expected values below are what those
helpers produced, so these tests pin the behaviour one-for-one.
"""

from __future__ import annotations

import pytest

from agent_evals.schemas.response import Response


# (raw response dict, expected state, expected context id, expected task id)
CASES = [
    pytest.param(
        {"task": {"id": "t1", "contextId": "c1", "status": {"state": "completed"}}},
        "completed",
        "c1",
        "t1",
        id="normal-wrapped",
    ),
    pytest.param(
        # Unwrapped task (response IS the task object).
        {"id": "t2", "contextId": "c2", "status": {"state": "working"}},
        "working",
        "c2",
        "t2",
        id="unwrapped-task",
    ),
    pytest.param(
        # Empty state shadowed by legacy ``status`` key -> falls through to it.
        {"task": {"id": "t3", "status": {"state": "", "status": "input-required"}}},
        "input-required",
        None,
        "t3",
        id="empty-state-falls-through-to-status",
    ),
    pytest.param(
        # contextId only at top level, not on the task.
        {"contextId": "top-ctx", "task": {"id": "t4", "status": {"state": "done"}}},
        "done",
        "top-ctx",
        "t4",
        id="top-level-contextId",
    ),
    pytest.param(
        # taskId buried deep (deep-search fallback), no task.id present.
        {
            "task": {"status": {"state": "done"}},
            "meta": {"nested": {"taskId": "deep-task"}},
        },
        "done",
        None,
        "deep-task",
        id="deep-search-taskId",
    ),
    pytest.param(
        # Missing task entirely.
        {"foo": "bar"},
        None,
        None,
        None,
        id="no-task",
    ),
    pytest.param(
        # Whitespace-only ids are treated as absent.
        {"task": {"id": "   ", "contextId": "  ", "status": {"state": "  "}}},
        None,
        None,
        None,
        id="whitespace-only",
    ),
    pytest.param(
        # None response.
        None,
        None,
        None,
        None,
        id="none-response",
    ),
]


@pytest.mark.parametrize("raw, state, context_id, task_id", CASES)
def test_response_accessors(raw, state, context_id, task_id) -> None:
    response = Response.from_dict(raw)
    assert response.state == state
    assert response.resolved_context_id() == context_id
    assert response.resolved_task_id() == task_id


def test_to_dict_is_verbatim_raw() -> None:
    """``to_dict`` must return the original dict untouched (full fidelity)."""
    raw = {
        "task": {"id": "t", "status": {"state": "completed"}, "kind": "task"},
        "metadata": {"opik_distributed_trace_headers": {"opik_trace_id": "trace-1"}},
    }
    assert Response.from_dict(raw).to_dict() == raw
