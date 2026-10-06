"""Tests for ``AgentClient.get_trace`` pagination params and ``fetch_all_trace_pages``."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import requests

from agent_evals.client import AgentClient
from agent_evals.environment import Environment
from agent_evals.tracing import fetch_all_trace_pages


@pytest.fixture()
def environment(monkeypatch: pytest.MonkeyPatch) -> Environment:
    monkeypatch.setenv("AGENT_API_TOKEN_LOCAL", "tok")
    return Environment("local")


@pytest.fixture()
def mock_session() -> MagicMock:
    session = MagicMock(spec=requests.Session)
    session.headers = {}
    response = MagicMock()
    response.content = b'{"traces": []}'
    response.json.return_value = {"traces": []}
    response.raise_for_status.return_value = None
    session.request.return_value = response
    return session


@pytest.fixture()
def client(environment: Environment, mock_session: MagicMock) -> AgentClient:
    return AgentClient(environment, session=mock_session)


class TestGetTracePagination:
    def test_no_params_sends_no_query_string(
        self, client: AgentClient, mock_session: MagicMock
    ) -> None:
        client.get_trace("ctx-1")
        _, kwargs = mock_session.request.call_args
        assert kwargs["params"] is None

    def test_page_size_forwarded(
        self, client: AgentClient, mock_session: MagicMock
    ) -> None:
        client.get_trace("ctx-1", page_size=200)
        _, kwargs = mock_session.request.call_args
        assert kwargs["params"] == {"pageSize": 200}

    def test_page_token_forwarded(
        self, client: AgentClient, mock_session: MagicMock
    ) -> None:
        client.get_trace("ctx-1", page_token="abc")
        _, kwargs = mock_session.request.call_args
        assert kwargs["params"] == {"pageToken": "abc"}

    def test_both_params_forwarded(
        self, client: AgentClient, mock_session: MagicMock
    ) -> None:
        client.get_trace("ctx-1", page_size=100, page_token="tok-1")
        _, kwargs = mock_session.request.call_args
        assert kwargs["params"] == {"pageSize": 100, "pageToken": "tok-1"}

    def test_hits_correct_path(
        self, client: AgentClient, mock_session: MagicMock
    ) -> None:
        client.get_trace("ctx-1")
        args, _ = mock_session.request.call_args
        assert args == ("GET", "http://localhost:8080/v2/agentic/contexts/ctx-1/trace")


class _PaginatedClient:
    """Serves scripted trace pages; drives ``fetch_all_trace_pages``."""

    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self._pages = pages
        self.calls: list[dict[str, Any]] = []

    def get_trace(
        self,
        context_id: str,
        *,
        page_size: int | None = None,
        page_token: str | None = None,
        timeout: Any = None,
    ) -> dict[str, Any]:
        call = {
            "context_id": context_id,
            "page_size": page_size,
            "page_token": page_token,
        }
        self.calls.append(call)
        idx = len(self.calls) - 1
        return self._pages[min(idx, len(self._pages) - 1)]


class TestFetchAllTracePages:
    def test_single_page_no_token(self) -> None:
        page = {"traces": [{"spans": [{"span_id": "s1"}]}], "nextPageToken": None}
        client = _PaginatedClient([page])
        result = fetch_all_trace_pages(client, "ctx-1")
        assert result is not None
        assert len(result["traces"]) == 1
        assert client.calls[0]["page_token"] is None
        assert len(client.calls) == 1

    def test_walks_multiple_pages(self) -> None:
        p1 = {"traces": [{"spans": [{"span_id": "s1"}]}], "nextPageToken": "tok-2"}
        p2 = {"traces": [{"spans": [{"span_id": "s2"}]}], "nextPageToken": "tok-3"}
        p3 = {"traces": [{"spans": [{"span_id": "s3"}]}], "nextPageToken": None}
        client = _PaginatedClient([p1, p2, p3])
        result = fetch_all_trace_pages(client, "ctx-1")
        assert result is not None
        assert len(result["traces"]) == 3
        assert client.calls[0]["page_token"] is None
        assert client.calls[1]["page_token"] == "tok-2"
        assert client.calls[2]["page_token"] == "tok-3"
        assert len(client.calls) == 3

    def test_returns_none_for_empty_traces(self) -> None:
        page = {"traces": [], "nextPageToken": None}
        client = _PaginatedClient([page])
        assert fetch_all_trace_pages(client, "ctx-1") is None

    def test_uses_max_page_size_by_default(self) -> None:
        page = {"traces": [{"spans": []}], "nextPageToken": None}
        client = _PaginatedClient([page])
        fetch_all_trace_pages(client, "ctx-1")
        assert client.calls[0]["page_size"] == 200

    def test_respects_custom_page_size(self) -> None:
        page = {"traces": [{"spans": []}], "nextPageToken": None}
        client = _PaginatedClient([page])
        fetch_all_trace_pages(client, "ctx-1", page_size=50)
        assert client.calls[0]["page_size"] == 50

    def test_stops_at_max_pages(self) -> None:
        page = {"traces": [{"spans": []}], "nextPageToken": "forever"}
        client = _PaginatedClient([page])
        fetch_all_trace_pages(client, "ctx-1", max_pages=3)
        assert len(client.calls) == 3

    def test_stops_when_next_page_token_is_null(self) -> None:
        page = {"traces": [{"spans": [{"span_id": "s1"}]}], "nextPageToken": None}
        client = _PaginatedClient([page])
        result = fetch_all_trace_pages(client, "ctx-1")
        assert result is not None
        assert len(client.calls) == 1

    def test_stops_when_next_page_token_is_empty_string(self) -> None:
        page = {"traces": [{"spans": [{"span_id": "s1"}]}], "nextPageToken": ""}
        client = _PaginatedClient([page])
        result = fetch_all_trace_pages(client, "ctx-1")
        assert result is not None
        assert len(client.calls) == 1
