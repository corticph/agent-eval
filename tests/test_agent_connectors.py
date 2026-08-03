"""Tests for ``Agent`` ``connectors`` and ``model`` fields."""

from __future__ import annotations

from agent_evals.schemas.agent import Agent

_CONNECTOR = {"type": "schema", "name": "tool", "schema": {"type": "object"}}


# --- Agent defaults ------------------------------------------------


def test_agent_defaults_connectors_empty_and_model_none() -> None:
    agent = Agent(name="Agent")
    assert agent.connectors == []
    assert agent.model is None


# --- to_dict includes connectors and model when set -----------------------


def test_agent_to_dict_includes_connectors() -> None:
    agent = Agent(name="Agent", connectors=[_CONNECTOR])
    out = agent.to_dict()
    assert out["connectors"] == [_CONNECTOR]


def test_agent_to_dict_includes_model() -> None:
    agent = Agent(name="Agent", model="corti-s1")
    out = agent.to_dict()
    assert out["model"] == "corti-s1"


def test_agent_to_dict_omits_connectors_when_empty() -> None:
    agent = Agent(name="Agent")
    out = agent.to_dict()
    assert "connectors" not in out


def test_agent_to_dict_omits_model_when_none() -> None:
    agent = Agent(name="Agent")
    out = agent.to_dict()
    assert "model" not in out


# --- from_dict parses connectors and model ---------------------------------


def test_agent_from_dict_parses_connectors() -> None:
    agent = Agent.from_dict({"name": "Agent", "connectors": [_CONNECTOR]})
    assert len(agent.connectors) == 1
    assert agent.connectors[0] == _CONNECTOR


def test_agent_from_dict_parses_model() -> None:
    agent = Agent.from_dict({"name": "Agent", "model": "corti-s1"})
    assert agent.model == "corti-s1"


def test_agent_from_dict_defaults_when_absent() -> None:
    agent = Agent.from_dict({"name": "Agent"})
    assert agent.connectors == []
    assert agent.model is None


def test_agent_from_dict_parses_connectors_and_model() -> None:
    agent = Agent.from_dict(
        {"name": "Agent", "connectors": [_CONNECTOR], "model": "corti-s1"}
    )
    assert len(agent.connectors) == 1
    assert agent.connectors[0] == _CONNECTOR
    assert agent.model == "corti-s1"


# --- round-trip through load_suite (connectors pass through) --------------


def test_connectors_round_trip_through_load_suite(tmp_path) -> None:
    from agent_evals.loader import load_suite

    suite_yaml = tmp_path / "suite.yaml"
    suite_yaml.write_text(
        "name: test_suite\n"
        "evals:\n"
        "  - name: case1\n"
        "    agent:\n"
        "      name: Agent\n"
        "      connectors:\n"
        "        - type: schema\n"
        "          name: tool\n"
        "          schema:\n"
        "            type: object\n"
        "      model: corti-s1\n"
        "    message:\n"
        "      message:\n"
        "        parts:\n"
        "          - kind: text\n"
        "            text: hello\n",
        encoding="utf-8",
    )
    suite = load_suite(suite_yaml)
    agent = suite.cases[0].agent
    assert len(agent.connectors) == 1
    assert agent.connectors[0] == {
        "type": "schema",
        "name": "tool",
        "schema": {"type": "object"},
    }
    assert agent.model == "corti-s1"
    out = agent.to_dict()
    assert out["connectors"] == [
        {"type": "schema", "name": "tool", "schema": {"type": "object"}}
    ]
    assert out["model"] == "corti-s1"
