"""Tests for ``resolve_connectors`` — loading ``connectors_file`` and resolving ``schema_ref`` entries."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from agent_evals.loader import resolve_connectors


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _schema_file(
    tmp_path: Path,
    name: str = "make_ehr_data_request",
    description: str = "Request EHR data",
    transition: str | None = None,
) -> Path:
    doc: dict[str, Any] = {
        "name": name,
        "description": description,
        "schema": {
            "type": "object",
            "properties": {"category": {"type": "string"}},
            "required": ["category"],
        },
        "type": "schema",
    }
    if transition is not None:
        doc["transition"] = transition
    return _write_json(tmp_path / "tool_schema.json", doc)


def _legacy_schema_file(
    tmp_path: Path,
    name: str = "make_ehr_data_request",
    description: str = "Request EHR data",
) -> Path:
    return _write_json(
        tmp_path / "tool_schema.json",
        {
            "tools": [
                {
                    "name": name,
                    "description": description,
                    "inputSchema": {
                        "type": "object",
                        "properties": {"category": {"type": "string"}},
                        "required": ["category"],
                    },
                }
            ]
        },
    )


# --- schema_ref resolution -------------------------------------------------


def test_schema_ref_resolves_to_full_connector(tmp_path: Path) -> None:
    schema_path = _schema_file(tmp_path)
    config: dict[str, Any] = {
        "connectors": [
            {"type": "schema", "schema_ref": schema_path.name, "transition": "manual"},
        ]
    }
    resolve_connectors(config, tmp_path)
    connector = config["connectors"][0]
    assert connector["schema"] == {
        "type": "object",
        "properties": {"category": {"type": "string"}},
        "required": ["category"],
    }
    assert connector["name"] == "make_ehr_data_request"
    assert connector["description"] == "Request EHR data"
    assert connector["type"] == "schema"
    assert connector["transition"] == "manual"
    assert "schema_ref" not in connector


# --- transition lifting ----------------------------------------------------


def test_schema_ref_transition_from_schema_file(tmp_path: Path) -> None:
    schema_path = _schema_file(tmp_path, transition="complete")
    config: dict[str, Any] = {
        "connectors": [{"type": "schema", "schema_ref": schema_path.name}]
    }
    resolve_connectors(config, tmp_path)
    assert config["connectors"][0]["transition"] == "complete"


def test_entry_transition_overrides_schema_file_transition(tmp_path: Path) -> None:
    schema_path = _schema_file(tmp_path, transition="complete")
    config: dict[str, Any] = {
        "connectors": [
            {
                "type": "schema",
                "schema_ref": schema_path.name,
                "transition": "input_required",
            }
        ]
    }
    resolve_connectors(config, tmp_path)
    assert config["connectors"][0]["transition"] == "input_required"


def test_transition_absent_when_neither_entry_nor_file_sets_it(
    tmp_path: Path,
) -> None:
    # Absent, not None — a null would serialize into the v2 payload as an
    # explicit null rather than being omitted.
    schema_path = _schema_file(tmp_path)
    config: dict[str, Any] = {
        "connectors": [{"type": "schema", "schema_ref": schema_path.name}]
    }
    resolve_connectors(config, tmp_path)
    assert "transition" not in config["connectors"][0]


def test_entry_transition_survives_when_schema_file_has_none(tmp_path: Path) -> None:
    schema_path = _schema_file(tmp_path)
    config: dict[str, Any] = {
        "connectors": [
            {
                "type": "schema",
                "schema_ref": schema_path.name,
                "transition": "complete",
            }
        ]
    }
    resolve_connectors(config, tmp_path)
    assert config["connectors"][0]["transition"] == "complete"


def test_legacy_format_transition_not_read_from_tool(tmp_path: Path) -> None:
    # Only the top level of the schema doc is consulted, so a transition tucked
    # inside the legacy ``tools[0]`` wrapper is deliberately not lifted.
    schema_path = _write_json(
        tmp_path / "tool_schema.json",
        {
            "tools": [
                {
                    "name": "make_ehr_data_request",
                    "description": "Request EHR data",
                    "inputSchema": {"type": "object", "properties": {}},
                    "transition": "complete",
                }
            ]
        },
    )
    config: dict[str, Any] = {
        "connectors": [{"type": "schema", "schema_ref": schema_path.name}]
    }
    resolve_connectors(config, tmp_path)
    assert "transition" not in config["connectors"][0]


def test_entry_name_overrides_file_tool_name(tmp_path: Path) -> None:
    schema_path = _schema_file(tmp_path, name="file_tool_name")
    config: dict[str, Any] = {
        "connectors": [
            {"type": "schema", "schema_ref": schema_path.name, "name": "entry_name"},
        ]
    }
    resolve_connectors(config, tmp_path)
    assert config["connectors"][0]["name"] == "entry_name"


def test_entry_description_overrides_file_description(tmp_path: Path) -> None:
    schema_path = _schema_file(tmp_path, description="file description")
    config: dict[str, Any] = {
        "connectors": [
            {
                "type": "schema",
                "schema_ref": schema_path.name,
                "name": "x",
                "description": "entry desc",
            },
        ]
    }
    resolve_connectors(config, tmp_path)
    assert config["connectors"][0]["description"] == "entry desc"


def test_name_mismatch_logs_warning_and_uses_entry_name(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    schema_path = _schema_file(tmp_path, name="file_name")
    config: dict[str, Any] = {
        "connectors": [
            {"type": "schema", "schema_ref": schema_path.name, "name": "entry_name"},
        ]
    }
    with caplog.at_level(logging.WARNING, logger="agent_evals.loader"):
        resolve_connectors(config, tmp_path)
    assert config["connectors"][0]["name"] == "entry_name"
    assert any(
        "entry_name" in r.message and "file_name" in r.message for r in caplog.records
    )


def test_no_name_anywhere_raises(tmp_path: Path) -> None:
    schema_path = _write_json(
        tmp_path / "tool_schema.json",
        {"tools": [{"description": "no name here", "inputSchema": {"type": "object"}}]},
    )
    config: dict[str, Any] = {
        "connectors": [{"type": "schema", "schema_ref": schema_path.name}],
    }
    with pytest.raises(ValueError, match="no name"):
        resolve_connectors(config, tmp_path)


def test_no_entry_name_uses_file_tool_name(tmp_path: Path) -> None:
    schema_path = _schema_file(tmp_path, name="file_tool_name")
    config: dict[str, Any] = {
        "connectors": [{"type": "schema", "schema_ref": schema_path.name}],
    }
    resolve_connectors(config, tmp_path)
    assert config["connectors"][0]["name"] == "file_tool_name"


# --- backwards compatibility (legacy tools format) -------------------------


def test_legacy_tools_format_still_works(tmp_path: Path) -> None:
    schema_path = _legacy_schema_file(
        tmp_path, name="legacy_tool", description="Legacy"
    )
    config: dict[str, Any] = {
        "connectors": [{"type": "schema", "schema_ref": schema_path.name}],
    }
    resolve_connectors(config, tmp_path)
    connector = config["connectors"][0]
    assert connector["name"] == "legacy_tool"
    assert connector["description"] == "Legacy"
    assert connector["schema"] == {
        "type": "object",
        "properties": {"category": {"type": "string"}},
        "required": ["category"],
    }
    assert connector["type"] == "schema"
    assert "schema_ref" not in connector


def test_flat_format_type_passes_through_from_file(tmp_path: Path) -> None:
    schema_path = _schema_file(tmp_path, name="my_tool")
    config: dict[str, Any] = {
        "connectors": [{"schema_ref": schema_path.name}],
    }
    resolve_connectors(config, tmp_path)
    connector = config["connectors"][0]
    assert connector["type"] == "schema"


# --- fail-fast errors ------------------------------------------------------


def test_schema_ref_file_not_found_raises_with_path(tmp_path: Path) -> None:
    config: dict[str, Any] = {
        "connectors": [{"type": "schema", "schema_ref": "missing.json"}],
    }
    with pytest.raises((FileNotFoundError, ValueError)) as exc_info:
        resolve_connectors(config, tmp_path)
    assert "missing.json" in str(exc_info.value)


def test_zero_tools_raises(tmp_path: Path) -> None:
    schema_path = _write_json(tmp_path / "tool_schema.json", {"tools": []})
    config: dict[str, Any] = {
        "connectors": [{"type": "schema", "schema_ref": schema_path.name, "name": "x"}],
    }
    with pytest.raises(ValueError, match="0 tools"):
        resolve_connectors(config, tmp_path)


def test_more_than_one_tool_raises(tmp_path: Path) -> None:
    schema_path = _write_json(
        tmp_path / "tool_schema.json",
        {"tools": [{"name": "a", "inputSchema": {}}, {"name": "b", "inputSchema": {}}]},
    )
    config: dict[str, Any] = {
        "connectors": [{"type": "schema", "schema_ref": schema_path.name}],
    }
    with pytest.raises(ValueError, match="more than 1 tool"):
        resolve_connectors(config, tmp_path)


def test_tools0_not_dict_raises_clean_value_error(tmp_path: Path) -> None:
    # ``tools[0]`` not a dict must raise a clean ValueError, not AttributeError.
    schema_path = _write_json(
        tmp_path / "tool_schema.json",
        {"tools": ["not_a_dict"]},
    )
    config: dict[str, Any] = {
        "connectors": [{"type": "schema", "schema_ref": schema_path.name, "name": "x"}],
    }
    with pytest.raises(ValueError, match=r"tools\[0\]"):
        resolve_connectors(config, tmp_path)


def test_tools0_missing_input_schema_raises_clean_value_error(tmp_path: Path) -> None:
    # Tool dict missing ``inputSchema`` must raise a clean ValueError, not KeyError.
    schema_path = _write_json(
        tmp_path / "tool_schema.json",
        {"tools": [{"name": "x", "description": "d"}]},
    )
    config: dict[str, Any] = {
        "connectors": [{"type": "schema", "schema_ref": schema_path.name}],
    }
    with pytest.raises(ValueError, match=r"inputSchema"):
        resolve_connectors(config, tmp_path)


def test_flat_format_null_schema_raises(tmp_path: Path) -> None:
    schema_path = _write_json(
        tmp_path / "tool_schema.json",
        {"name": "x", "description": "d", "schema": None, "type": "schema"},
    )
    config: dict[str, Any] = {
        "connectors": [{"type": "schema", "schema_ref": schema_path.name}],
    }
    with pytest.raises(ValueError, match=r"'schema'"):
        resolve_connectors(config, tmp_path)


# --- resolve_connectors does not re-walk resolved schemas ------------------


def _schema_with_hostile_property(tmp_path: Path) -> Path:
    """A flat-format schema file whose ``schema`` has a property named ``connectors_file``.

    Without the no-re-walk fix, the walk re-enters the resolved connector's
    inline ``schema`` and treats this property as a nested connectors-file
    reference, raising ``TypeError: unsupported operand type(s) for /:
    'PosixPath' and 'dict'``.
    """
    return _write_json(
        tmp_path / "tool_schema.json",
        {
            "name": "x",
            "description": "d",
            "schema": {
                "type": "object",
                "properties": {
                    "connectors_file": {"type": "string"},
                },
            },
            "type": "schema",
        },
    )


def test_resolved_connectors_schema_not_rewalked(tmp_path: Path) -> None:
    # After resolving a ``connectors_file``, the walk must not re-enter the
    # resolved connector dicts and traverse their inline ``schema`` JSON.
    schema_path = _schema_with_hostile_property(tmp_path)
    _write_json(
        tmp_path / "connectors.json",
        [{"type": "schema", "schema_ref": schema_path.name}],
    )
    config: dict[str, Any] = {"connectors_file": "connectors.json"}
    # Must not raise.
    resolve_connectors(config, tmp_path)
    connector = config["connectors"][0]
    # The schema's ``connectors_file`` property survives intact — it was
    # not popped/treated as a nested connectors-file reference.
    assert "connectors_file" in connector["schema"]["properties"]


def test_resolved_inline_connectors_schema_not_rewalked(tmp_path: Path) -> None:
    # Same guarantee for the inline ``connectors`` branch (no ``connectors_file``):
    # a resolved connector's schema must not be re-walked.
    schema_path = _schema_with_hostile_property(tmp_path)
    config: dict[str, Any] = {
        "connectors": [{"type": "schema", "schema_ref": schema_path.name}],
    }
    resolve_connectors(config, tmp_path)
    connector = config["connectors"][0]
    assert "connectors_file" in connector["schema"]["properties"]


# --- pass-through entries --------------------------------------------------


def test_entry_without_schema_ref_passes_through(tmp_path: Path) -> None:
    registry_entry = {"type": "registry", "name": "some_expert"}
    mcp_entry = {"type": "mcp", "name": "server", "url": "https://example.com"}
    config: dict[str, Any] = {"connectors": [registry_entry, mcp_entry]}
    resolve_connectors(config, tmp_path)
    assert config["connectors"][0] == registry_entry
    assert config["connectors"][1] == mcp_entry


# --- connectors_file loading -----------------------------------------------


def test_connectors_file_loads_json_array(tmp_path: Path) -> None:
    schema_path = _schema_file(tmp_path)
    _write_json(
        tmp_path / "connectors.json",
        [
            {"type": "schema", "schema_ref": schema_path.name},
            {"type": "registry", "name": "expert_one"},
        ],
    )
    config: dict[str, Any] = {"connectors_file": "connectors.json"}
    resolve_connectors(config, tmp_path)
    assert "connectors_file" not in config
    assert len(config["connectors"]) == 2
    assert config["connectors"][0]["schema"]["type"] == "object"
    assert config["connectors"][1] == {"type": "registry", "name": "expert_one"}


def test_connectors_file_not_found_raises(tmp_path: Path) -> None:
    config: dict[str, Any] = {"connectors_file": "missing.json"}
    with pytest.raises((FileNotFoundError, ValueError)) as exc_info:
        resolve_connectors(config, tmp_path)
    assert "missing.json" in str(exc_info.value)


def test_connectors_file_must_be_array(tmp_path: Path) -> None:
    _write_json(tmp_path / "connectors.json", {"not": "an array"})
    config: dict[str, Any] = {"connectors_file": "connectors.json"}
    with pytest.raises(ValueError, match="array"):
        resolve_connectors(config, tmp_path)


# --- walking nested config -------------------------------------------------


def test_walks_into_globals_agent(tmp_path: Path) -> None:
    schema_path = _schema_file(tmp_path)
    _write_json(
        tmp_path / "connectors.json",
        [{"type": "schema", "schema_ref": schema_path.name}],
    )
    config: dict[str, Any] = {
        "globals": {"agent": {"name": "Agent", "connectors_file": "connectors.json"}}
    }
    resolve_connectors(config, tmp_path)
    agent = config["globals"]["agent"]
    assert "connectors_file" not in agent
    assert len(agent["connectors"]) == 1
    assert agent["connectors"][0]["schema"]["type"] == "object"


def test_walks_into_per_eval_agent(tmp_path: Path) -> None:
    schema_path = _schema_file(tmp_path)
    config: dict[str, Any] = {
        "evals": [
            {
                "name": "case1",
                "agent": {
                    "name": "Agent",
                    "connectors": [{"type": "schema", "schema_ref": schema_path.name}],
                },
            }
        ]
    }
    resolve_connectors(config, tmp_path)
    connector = config["evals"][0]["agent"]["connectors"][0]
    assert connector["schema"]["type"] == "object"
    assert "schema_ref" not in connector


# --- integration: load_suite with connectors_file in _defaults.yaml ---------


_MINIMAL_SUITE = """\
name: test_suite
evals:
  - name: case1
    message:
      message:
        parts:
          - kind: text
            text: hello
"""

_DEFAULTS_WITH_CONNECTORS_FILE = """\
globals:
  agent:
    name: ExampleAgent
    description: from defaults
    connectors_file: connectors.json
    model: corti-s1
  message:
    message:
      messageId: ""
      role: user
      kind: message
"""


def _write(base: Path, rel: str, content: str) -> Path:
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_load_suite_resolves_connectors_file_from_defaults(tmp_path: Path) -> None:
    _write(tmp_path, "pyproject.toml", "")  # mark as project root
    schema_path = _schema_file(
        tmp_path, name="make_ehr_data_request", description="Request EHR data"
    )
    _write_json(
        tmp_path / "connectors.json",
        [
            {"type": "schema", "schema_ref": schema_path.name, "transition": "manual"},
            {"type": "registry", "name": "expert_one"},
        ],
    )
    _write(tmp_path, "_defaults.yaml", _DEFAULTS_WITH_CONNECTORS_FILE)
    suite_path = _write(tmp_path, "suite.yaml", _MINIMAL_SUITE)

    from agent_evals.loader import load_suite

    suite = load_suite(suite_path)
    agent = suite.cases[0].agent
    # connectors_file resolved into inline connector dicts
    assert agent.connectors[0]["type"] == "schema"
    assert agent.connectors[0]["name"] == "make_ehr_data_request"
    assert agent.connectors[0]["description"] == "Request EHR data"
    assert agent.connectors[0]["schema"] == {
        "type": "object",
        "properties": {"category": {"type": "string"}},
        "required": ["category"],
    }
    assert agent.connectors[0]["transition"] == "manual"
    # pass-through entry untouched
    assert agent.connectors[1] == {"type": "registry", "name": "expert_one"}
    # model passes through
    assert agent.model == "corti-s1"
    # to_dict produces a v2-clean payload
    payload = agent.to_dict()
    assert "connectors_file" not in payload


def test_load_suite_per_eval_connectors_file(tmp_path: Path) -> None:
    _write(tmp_path, "pyproject.toml", "")
    schema_path = _schema_file(tmp_path, name="my_tool")
    _write_json(
        tmp_path / "connectors.json",
        [{"type": "schema", "schema_ref": schema_path.name}],
    )
    suite_yaml = (
        "name: test_suite\n"
        "evals:\n"
        "  - name: case1\n"
        "    agent:\n"
        "      name: Agent\n"
        "      connectors_file: connectors.json\n"
        "    message:\n"
        "      message:\n"
        "        parts:\n"
        "          - kind: text\n"
        "            text: hello\n"
    )
    suite_path = _write(tmp_path, "suite.yaml", suite_yaml)

    from agent_evals.loader import load_suite

    suite = load_suite(suite_path)
    agent = suite.cases[0].agent
    assert agent.connectors[0]["name"] == "my_tool"
    assert agent.connectors[0]["schema"]["type"] == "object"
