"""Fail-fast rejection of v1-only fields in agent payloads and eval cases.

The v1 translation machinery is gone, but older suites may still carry
v1 keys (``experts``, ``mcpServers``, ``use_expert_name``). Without a
guard, those suites load silently as bare orchestrators with no
connectors — wrong results instead of a loud failure.

These tests pin the loud failure: every v1-only key must raise
``ValueError`` at load time, with a message pointing at the v2
connector format as the migration path.
"""

from __future__ import annotations

import pytest

from agent_evals.loader import EvaluationCase
from agent_evals.loader import _validate_agent_payload


# --- _validate_agent_payload: agent-level v1 keys -------------------------


def test_validate_agent_payload_rejects_experts() -> None:
    with pytest.raises(ValueError, match=r"experts"):
        _validate_agent_payload(
            {"name": "Agent", "experts": [{"name": "x"}]},
            case_name="case1",
        )


def test_validate_agent_payload_rejects_mcp_servers() -> None:
    with pytest.raises(ValueError, match=r"mcpServers"):
        _validate_agent_payload(
            {"name": "Agent", "mcpServers": [{"name": "x"}]},
            case_name="case1",
        )


def test_validate_agent_payload_error_mentions_retired_key() -> None:
    with pytest.raises(ValueError, match=r"retired.*connectors"):
        _validate_agent_payload(
            {"name": "Agent", "experts": []},
            case_name="case1",
        )


def test_validate_agent_payload_accepts_v2_agent() -> None:
    # A clean v2 agent payload must pass — no false positives.
    _validate_agent_payload(
        {"name": "Agent", "connectors": [{"type": "registry", "name": "x"}]},
        case_name="case1",
    )


# --- EvaluationCase.from_dict: case-level use_expert_name ------------------


def _minimal_case_yaml_payload(v1_extra: dict) -> dict:
    payload: dict = {
        "name": "case1",
        "agent": {"name": "Agent"},
        "message": {"message": {"messageId": "", "role": "user", "kind": "message"}},
    }
    payload.update(v1_extra)
    return payload


def test_evaluation_case_rejects_use_expert_name() -> None:
    with pytest.raises(ValueError, match=r"use_expert_name"):
        EvaluationCase.from_dict(
            _minimal_case_yaml_payload({"use_expert_name": "DrHouse"})
        )


def test_sequential_evaluation_case_rejects_use_expert_name() -> None:
    payload: dict = {
        "name": "seq1",
        "agent": {"name": "Agent"},
        "use_expert_name": "DrHouse",
        "steps": [
            {
                "name": "step1",
                "message": {
                    "message": {"messageId": "", "role": "user", "kind": "message"}
                },
            }
        ],
    }
    with pytest.raises(ValueError, match=r"use_expert_name"):
        EvaluationCase.from_dict(payload)


def test_evaluation_case_rejects_v1_agent_experts() -> None:
    payload = _minimal_case_yaml_payload({})
    payload["agent"] = {"name": "Agent", "experts": [{"name": "DrHouse"}]}
    with pytest.raises(ValueError, match=r"experts"):
        EvaluationCase.from_dict(payload)


def test_evaluation_case_rejects_v1_agent_mcp_servers() -> None:
    payload = _minimal_case_yaml_payload({})
    payload["agent"] = {"name": "Agent", "mcpServers": [{"name": "x"}]}
    with pytest.raises(ValueError, match=r"mcpServers"):
        EvaluationCase.from_dict(payload)


def test_evaluation_case_accepts_clean_v2_payload() -> None:
    # Sanity: a v2-clean case must still load.
    case = EvaluationCase.from_dict(_minimal_case_yaml_payload({}))
    assert case.name == "case1"
    assert case.agent.name == "Agent"
