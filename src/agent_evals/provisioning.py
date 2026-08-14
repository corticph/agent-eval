"""Agent provisioning: the pool and inline agent-connector creation."""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable

from .client import AgentClient
from .loader import EvaluationCase

_LOGGER = logging.getLogger(__name__)


def _extract_agent_id(agent_response: dict[str, Any]) -> str | None:
    """Try to retrieve the agent identifier from the response."""
    for key in ("agent_id", "agentId", "id", "agentID"):
        value = agent_response.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _create_agent(client: AgentClient, payload: dict[str, Any]) -> str:
    """Create an agent and return its id, or raise if the response has none."""
    response = client.create_agent(payload)
    agent_id = _extract_agent_id(response)
    if not agent_id:
        raise RuntimeError(
            "Agent creation response did not contain an agent identifier"
        )
    return agent_id


# Suites author the v2 payload verbatim, with one extension the wire shape
# cannot express: an ``agent`` connector written as an inline definition
# (name/systemPrompt/connectors) instead of an ``agentId``. Its id only exists
# once the sub-agent is created, so provisioning creates it here and splices
# the fresh ``{type: agent, agentId}`` reference into the payload it sends.


def _is_inline_agent(connector: dict[str, Any]) -> bool:
    return connector.get("type") == "agent" and not connector.get("agentId")


def _connector_definition(connector: dict[str, Any]) -> dict[str, Any]:
    """The create-agent payload embedded in an inline ``agent`` connector."""
    return {k: v for k, v in connector.items() if k != "type"}


def _resolve_inline_agents(
    client: AgentClient, agent_payload: dict[str, Any]
) -> dict[str, Any]:
    """Return *agent_payload* with every inline agent connector created by id."""
    connectors = agent_payload.get("connectors") or []
    if not any(_is_inline_agent(c) for c in connectors):
        return agent_payload
    resolved: list[dict[str, Any]] = []
    for connector in connectors:
        if _is_inline_agent(connector):
            sub_id = _create_agent(
                client, _resolve_inline_agents(client, _connector_definition(connector))
            )
            _LOGGER.debug(
                "Created sub-agent %s for connector %s", sub_id, connector.get("name")
            )
            resolved.append({"type": "agent", "agentId": sub_id})
        else:
            resolved.append(connector)
    return {**agent_payload, "connectors": resolved}


def _create_targeted_agent(
    client: AgentClient, agent_payload: dict[str, Any], connector_name: str
) -> str:
    """Create a standalone agent for a single named connector and return its id.

    Backs ``use_connector_name``: the eval targets one connector in isolation
    instead of the orchestrator. An inline ``agent`` connector becomes its own
    agent; a ``registry`` connector is wrapped in a minimal agent whose only
    connector is that registry entry (verified to route straight to it).
    """
    connector = next(
        (
            c
            for c in agent_payload.get("connectors") or []
            if c.get("name") == connector_name
        ),
        None,
    )
    if connector is None:
        raise RuntimeError(f"Connector {connector_name!r} not found in agent spec")
    if _is_inline_agent(connector):
        payload = _resolve_inline_agents(client, _connector_definition(connector))
        return _create_agent(client, payload)
    elif connector.get("type") == "registry":
        payload = {"name": connector_name, "connectors": [connector]}
        response = client.create_agent(payload)
        # Send messages directly to the connector id (e.g. con.xxx), not to the
        # wrapper agent id — the wrapper's system prompt makes it an orchestrator
        # that re-delegates, defeating the purpose of targeting the connector.
        connector_id = next(
            (
                c.get("id")
                for c in response.get("connectors") or []
                if c.get("name") == connector_name
                and isinstance(c.get("id"), str)
                and c.get("id")
            ),
            None,
        )
        if not connector_id:
            raise RuntimeError(
                f"Registry connector {connector_name!r} creation response did not contain a connector id"
            )
        # TODO Bug... The API returns ids with a type prefix (e.g. "con.uuid"); the send
        # endpoint takes only the bare uuid portion after the first dot.
        return connector_id.split(".", 1)[-1]
    else:
        raise ValueError(
            f"Connector {connector_name!r} of type {connector.get('type')!r} "
            "cannot be targeted with use_connector_name"
        )


def provision_agent(
    client: AgentClient,
    agent_payload: dict[str, Any],
    connector_name: str | None = None,
) -> str:
    """Create the agent an authored payload describes and return its id.

    The one front door both pipelines share: with *connector_name*, the named
    connector runs as the agent in isolation; without, the payload is created
    whole once its inline agent connectors exist.
    """
    if connector_name:
        return _create_targeted_agent(client, agent_payload, connector_name)
    return _create_agent(client, _resolve_inline_agents(client, agent_payload))


def _create_agent_for_case(client: AgentClient, case: EvaluationCase) -> str:
    """Create the agent a case messages and return its id.

    A case pinning an explicit id never reaches here — the pool resolves that
    without creating anything.
    """
    agent_id = provision_agent(client, case.agent.to_dict(), case.use_connector_name)
    _LOGGER.debug("Created agent %s for case %s", agent_id, case.name)
    return agent_id


class AgentPool:
    """A run's ``key -> agent_id`` map.

    Provisioning happens once, single-threaded, before execution fans out, so
    the map is read-only during the run and concurrent cases resolve their
    agent by lookup without locking.
    """

    def __init__(self) -> None:
        self._by_key: dict[str, str] = {}

    @staticmethod
    def _key(case: EvaluationCase) -> str:
        # Key on the authored spec plus the targeted connector — never the
        # provisioned payload, which embeds freshly created inline sub-agent
        # ids and so would differ every run and defeat dedup.
        return json.dumps(
            {
                "agent": case.agent.to_dict(),
                "use_connector_name": case.use_connector_name,
            },
            sort_keys=True,
        )

    def provision(self, cases: Iterable[EvaluationCase], client: AgentClient) -> None:
        """Create every unique agent the suite needs, before any case runs.

        Fail-fast: a creation that raises aborts the whole run here, so a broken
        spec surfaces up front rather than as a wall of per-case errors.
        """
        for case in cases:
            if case.agent_id_override:
                continue  # the override *is* the id; nothing to create
            key = self._key(case)
            if key in self._by_key:
                continue
            self._by_key[key] = _create_agent_for_case(client, case)

    def agent_id_for(self, case: EvaluationCase) -> str:
        """Return the provisioned (or overridden) agent id for a case."""
        if case.agent_id_override:
            return case.agent_id_override
        return self._by_key[self._key(case)]
