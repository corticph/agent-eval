"""Tests for AgentClient HTTP behaviour."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

from agent_evals.client import AgentClient, _prefix_agent_name
from agent_evals.environment import Environment
from agent_evals.errors import (
    ErrorCode,
    HttpStatusError,
    InvalidResponseError,
    NetworkError,
    RequestTimeoutError,
)


@pytest.fixture()
def environment(monkeypatch: pytest.MonkeyPatch) -> Environment:
    # The local row resolves without any network: static token, fixed URL.
    monkeypatch.setenv("AGENT_API_TOKEN_LOCAL", "tok")
    return Environment("local")


@pytest.fixture()
def mock_session() -> MagicMock:
    session = MagicMock(spec=requests.Session)
    session.headers = {}
    response = MagicMock()
    response.content = b'{"id": "abc"}'
    response.json.return_value = {"id": "abc"}
    response.raise_for_status.return_value = None
    session.request.return_value = response
    return session


@pytest.fixture()
def client(environment: Environment, mock_session: MagicMock) -> AgentClient:
    return AgentClient(environment, session=mock_session)


class TestPrefixAgentName:
    def test_adds_prefix(self) -> None:
        assert _prefix_agent_name({"name": "my-agent"}) == {"name": "EVAL_my-agent"}

    def test_idempotent(self) -> None:
        assert _prefix_agent_name({"name": "EVAL_my-agent"}) == {
            "name": "EVAL_my-agent"
        }

    def test_missing_name_unchanged(self) -> None:
        payload = {"foo": "bar"}
        assert _prefix_agent_name(payload) == payload


class TestCreateAgent:
    def test_posts_to_v2_path(
        self, client: AgentClient, mock_session: MagicMock
    ) -> None:
        client.create_agent({"name": "my-agent"})
        args, _ = mock_session.request.call_args
        assert args == ("POST", "http://localhost:8080/v2/agentic/agents")

    def test_defaults_ephemeral_lifecycle_in_body(
        self, client: AgentClient, mock_session: MagicMock
    ) -> None:
        # v1 signalled this via ?ephemeral=true; v2 carries lifecycle in the body.
        client.create_agent({"name": "my-agent"})
        _, kwargs = mock_session.request.call_args
        assert kwargs["json"]["lifecycle"] == "ephemeral"
        assert kwargs["params"] is None

    def test_respects_explicit_lifecycle(
        self, client: AgentClient, mock_session: MagicMock
    ) -> None:
        client.create_agent({"name": "my-agent", "lifecycle": "persistent"})
        _, kwargs = mock_session.request.call_args
        assert kwargs["json"]["lifecycle"] == "persistent"

    def test_prefixes_name(self, client: AgentClient, mock_session: MagicMock) -> None:
        client.create_agent({"name": "my-agent"})
        _, kwargs = mock_session.request.call_args
        assert kwargs["json"]["name"] == "EVAL_my-agent"

    def test_returns_response_json(self, client: AgentClient) -> None:
        result = client.create_agent({"name": "x"})
        assert result == {"id": "abc"}


class TestSendMessage:
    def test_posts_to_correct_path(
        self, client: AgentClient, mock_session: MagicMock
    ) -> None:
        result = client.send_message("agent-1", {"text": "hello"})
        args, kwargs = mock_session.request.call_args
        assert args == (
            "POST",
            "http://localhost:8080/v2/agentic/agents/agent-1/a2a/message:send",
        )
        assert kwargs["json"] == {"text": "hello"}
        assert result == {"id": "abc"}

    def test_sends_a2a_version_header(
        self, client: AgentClient, mock_session: MagicMock
    ) -> None:
        client.send_message("agent-1", {"text": "hello"})
        _, kwargs = mock_session.request.call_args
        headers = kwargs.get("headers") or {}
        assert headers.get("A2A-Version") == "1.0"

    def test_sends_no_opik_trace_headers(
        self, client: AgentClient, mock_session: MagicMock
    ) -> None:
        # Sending opik_trace_id/opik_parent_span_id makes the Go agent service
        # treat the trace as caller-owned and skip writing the trace record to
        # Opik entirely (nothing shows up in the UI). The harness must let the
        # service own its traces.
        client.send_message("agent-1", {"text": "hello"})
        _, kwargs = mock_session.request.call_args
        headers = kwargs.get("headers") or {}
        assert "opik_trace_id" not in headers
        assert "opik_parent_span_id" not in headers


class TestRequestErrorHandling:
    def test_connection_error_raises_network_error(
        self, client: AgentClient, mock_session: MagicMock
    ) -> None:
        mock_session.request.side_effect = requests.ConnectionError("refused")
        with pytest.raises(RuntimeError, match="Connection failed") as excinfo:
            client.create_agent({"name": "x"})
        assert isinstance(excinfo.value, NetworkError)
        assert excinfo.value.code == ErrorCode.NETWORK_ERROR

    def test_timeout_raises_timeout_error(
        self, client: AgentClient, mock_session: MagicMock
    ) -> None:
        mock_session.request.side_effect = requests.Timeout("timed out")
        with pytest.raises(RuntimeError, match="timed out") as excinfo:
            client.create_agent({"name": "x"})
        assert isinstance(excinfo.value, RequestTimeoutError)
        assert excinfo.value.code == ErrorCode.TIMEOUT

    def test_http_status_raises_http_error(
        self, client: AgentClient, mock_session: MagicMock
    ) -> None:
        response = mock_session.request.return_value
        response.status_code = 500
        response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        with pytest.raises(RuntimeError) as excinfo:
            client.create_agent({"name": "x"})
        assert isinstance(excinfo.value, HttpStatusError)
        assert excinfo.value.code == ErrorCode.HTTP_ERROR

    def test_non_json_body_raises_invalid_response(
        self, client: AgentClient, mock_session: MagicMock
    ) -> None:
        import json as _json

        response = mock_session.request.return_value
        response.content = b"not json"
        response.text = "not json"
        response.json.side_effect = _json.JSONDecodeError("boom", "not json", 0)
        with pytest.raises(RuntimeError) as excinfo:
            client.create_agent({"name": "x"})
        assert isinstance(excinfo.value, InvalidResponseError)
        assert excinfo.value.code == ErrorCode.INVALID_RESPONSE

    def test_empty_response_returns_empty_dict(
        self, client: AgentClient, mock_session: MagicMock
    ) -> None:
        mock_session.request.return_value.content = b""
        result = client.create_agent({"name": "x"})
        assert result == {}


class TestParamsForwarding:
    def test_int_and_bool_params_accepted(
        self, client: AgentClient, mock_session: MagicMock
    ) -> None:
        client._request("GET", "/foo", params={"limit": 10, "active": True})
        _, kwargs = mock_session.request.call_args
        assert kwargs["params"] == {"limit": 10, "active": True}


class TestClone:
    def test_clone_shares_the_environment_but_not_the_session(
        self, client: AgentClient
    ) -> None:
        clone = client.clone()
        assert clone.environment is client.environment
        assert clone.session is not client.session
