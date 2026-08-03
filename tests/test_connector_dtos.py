"""Tests for typed connector DTOs.

Pins the shape and round-trip behaviour of each connector type the
agent API accepts, and the ``parse_connector`` discriminator that
dispatches on the ``type`` key.
"""

from __future__ import annotations

import pytest

from agent_evals.schemas.agent import Agent
from agent_evals.schemas.connectors import (
    A2A,
    AgentConnector,
    Mcp,
    Registry,
    SchemaConnector,
    parse_connector,
)

# --- SchemaConnector -----------------------------------------------------


def test_schema_round_trips() -> None:
    data = {
        "type": "schema",
        "name": "make_ehr_data_request",
        "description": "Request EHR data",
        "schema": {"type": "object", "properties": {}},
    }
    conn = SchemaConnector.from_dict(data)
    assert conn.TYPE == "schema"
    assert conn.name == "make_ehr_data_request"
    assert conn.description == "Request EHR data"
    assert conn.schema == {"type": "object", "properties": {}}
    assert conn.to_dict() == data


def test_schema_omits_optional_fields() -> None:
    conn = SchemaConnector(name="tool", schema={"type": "object"})
    assert conn.to_dict() == {
        "type": "schema",
        "name": "tool",
        "schema": {"type": "object"},
    }


def test_schema_includes_transition_when_set() -> None:
    data = {
        "type": "schema",
        "name": "tool",
        "schema": {"type": "object"},
        "transition": "immediate",
    }
    conn = SchemaConnector.from_dict(data)
    assert conn.transition == "immediate"
    assert conn.to_dict() == data


# --- Registry connector ----------------------------------------------------


def test_registry_round_trips() -> None:
    data = {"type": "registry", "name": "pubmed-expert"}
    conn = Registry.from_dict(data)
    assert conn.TYPE == "registry"
    assert conn.name == "pubmed-expert"
    assert conn.to_dict() == data


# --- Mcp connector ---------------------------------------------------------


def test_mcp_round_trips() -> None:
    data = {
        "type": "mcp",
        "name": "extract_observations",
        "url": "https://mcp.example.com/schemas/abc/mcp",
    }
    conn = Mcp.from_dict(data)
    assert conn.TYPE == "mcp"
    assert conn.name == "extract_observations"
    assert conn.url == "https://mcp.example.com/schemas/abc/mcp"
    assert conn.auth is None
    assert conn.to_dict() == data


def test_mcp_with_auth() -> None:
    data = {
        "type": "mcp",
        "name": "tool",
        "url": "https://example.com/mcp",
        "auth": {"type": "bearer"},
    }
    conn = Mcp.from_dict(data)
    assert conn.auth is not None
    assert conn.auth.type == "bearer"
    assert conn.to_dict() == data


def test_mcp_omits_auth_when_none() -> None:
    conn = Mcp(name="tool", url="https://example.com/mcp")
    out = conn.to_dict()
    assert "auth" not in out


# --- AgentConnector --------------------------------------------------------


def test_agent_connector_round_trips() -> None:
    data = {"type": "agent", "agentId": "agent-123"}
    conn = AgentConnector.from_dict(data)
    assert conn.TYPE == "agent"
    assert conn.agent_id == "agent-123"
    assert conn.to_dict() == data


# --- A2A connector ---------------------------------------------------------


def test_a2a_round_trips() -> None:
    data = {
        "type": "a2a",
        "name": "interviewing-expert",
        "url": "https://ml-service:8080/a2a",
    }
    conn = A2A.from_dict(data)
    assert conn.TYPE == "a2a"
    assert conn.name == "interviewing-expert"
    assert conn.url == "https://ml-service:8080/a2a"
    assert conn.to_dict() == data


# --- parse_connector discriminator -----------------------------------------


def test_parse_connector_schema() -> None:
    conn = parse_connector(
        {"type": "schema", "name": "tool", "schema": {"type": "object"}}
    )
    assert isinstance(conn, SchemaConnector)


def test_parse_connector_registry() -> None:
    conn = parse_connector({"type": "registry", "name": "pubmed-expert"})
    assert isinstance(conn, Registry)


def test_parse_connector_mcp() -> None:
    conn = parse_connector(
        {"type": "mcp", "name": "tool", "url": "https://example.com"}
    )
    assert isinstance(conn, Mcp)


def test_parse_connector_agent() -> None:
    conn = parse_connector({"type": "agent", "agentId": "agent-42"})
    assert isinstance(conn, AgentConnector)


def test_parse_connector_a2a() -> None:
    conn = parse_connector(
        {"type": "a2a", "name": "expert", "url": "https://example.com"}
    )
    assert isinstance(conn, A2A)


def test_parse_connector_unknown_type_raises() -> None:
    with pytest.raises(ValueError, match="Unknown connector type"):
        parse_connector({"type": "unknown", "name": "thing"})


def test_parse_connector_missing_type_raises() -> None:
    with pytest.raises(ValueError, match="Unknown connector type"):
        parse_connector({"name": "thing"})


# --- Agent with raw-dict connectors ----------------------------------


def test_agent_from_dict_passes_connectors_as_raw_dicts() -> None:
    data = {
        "name": "Orchestrator",
        "connectors": [
            {"type": "schema", "name": "tool", "schema": {"type": "object"}},
            {"type": "registry", "name": "pubmed-expert"},
        ],
    }
    agent = Agent.from_dict(data)
    assert len(agent.connectors) == 2
    assert agent.connectors[0] == {
        "type": "schema",
        "name": "tool",
        "schema": {"type": "object"},
    }
    assert agent.connectors[1] == {"type": "registry", "name": "pubmed-expert"}


def test_agent_to_dict_serializes_raw_dict_connectors() -> None:
    agent = Agent(
        name="Orch",
        connectors=[
            {"type": "schema", "name": "tool", "schema": {"type": "object"}},
            {"type": "registry", "name": "pubmed-expert"},
        ],
    )
    out = agent.to_dict()
    assert out["connectors"] == [
        {"type": "schema", "name": "tool", "schema": {"type": "object"}},
        {"type": "registry", "name": "pubmed-expert"},
    ]


def test_agent_connectors_default_empty_list() -> None:
    agent = Agent(name="Bare")
    assert agent.connectors == []


def test_agent_to_dict_omits_connectors_when_empty() -> None:
    agent = Agent(name="Bare")
    assert "connectors" not in agent.to_dict()
