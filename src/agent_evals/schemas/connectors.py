"""Typed connector DTOs for the agent API v2.

Each connector type is a ``@dataclass`` with ``from_dict`` / ``to_dict``
round-trip. ``parse_connector`` discriminates on the ``type`` key to
produce the right variant.

Connector types:
  * ``SchemaConnector`` — ``type: "schema"``  (inline JSON-schema tool)
  * ``Registry``   — ``type: "registry"`` (pre-registered connector)
  * ``Mcp``        — ``type: "mcp"``      (MCP server)
  * ``AgentConnector`` — ``type: "agent"``   (reference to another agent)
  * ``A2A``        — ``type: "a2a"``      (Agent-to-Agent endpoint)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Union


@dataclass(slots=True)
class Auth:
    """Authentication block for MCP connectors."""

    type: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Auth":
        return cls(type=data["type"])

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type}


@dataclass(slots=True)
class SchemaConnector:
    """``type: "schema"`` — an inline JSON-schema tool connector."""

    TYPE: ClassVar[str] = "schema"

    name: str
    schema: dict[str, Any]
    description: str | None = None
    transition: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SchemaConnector":
        return cls(
            name=data["name"],
            schema=data["schema"],
            description=data.get("description"),
            transition=data.get("transition"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.TYPE,
            "name": self.name,
            "schema": self.schema,
        }
        if self.description:
            result["description"] = self.description
        if self.transition:
            result["transition"] = self.transition
        return result


@dataclass(slots=True)
class Registry:
    """``type: "registry"`` — a pre-registered connector by name."""

    TYPE: ClassVar[str] = "registry"

    name: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Registry":
        return cls(name=data["name"])

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.TYPE, "name": self.name}


@dataclass(slots=True)
class Mcp:
    """``type: "mcp"`` — an MCP server connector."""

    TYPE: ClassVar[str] = "mcp"

    name: str
    url: str
    auth: Auth | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Mcp":
        auth_raw = data.get("auth")
        return cls(
            name=data["name"],
            url=data["url"],
            auth=Auth.from_dict(auth_raw) if auth_raw else None,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.TYPE,
            "name": self.name,
            "url": self.url,
        }
        if self.auth:
            result["auth"] = self.auth.to_dict()
        return result


@dataclass(slots=True)
class AgentConnector:
    """``type: "agent"`` — a reference to another agent by id."""

    TYPE: ClassVar[str] = "agent"

    agent_id: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentConnector":
        return cls(agent_id=data["agentId"])

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.TYPE, "agentId": self.agent_id}


@dataclass(slots=True)
class InlineAgentConnector:
    """``type: "agent"`` without ``agentId`` — an inline agent definition.

    A full create-agent payload embedded as a connector, resolved by the
    provisioning layer into a reference before the create call.
    """

    TYPE: ClassVar[str] = "agent"

    data: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InlineAgentConnector":
        return cls(data=data)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)


@dataclass(slots=True)
class A2A:
    """``type: "a2a"`` — an Agent-to-Agent protocol endpoint."""

    TYPE: ClassVar[str] = "a2a"

    name: str
    url: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "A2A":
        return cls(name=data["name"], url=data["url"])

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.TYPE, "name": self.name, "url": self.url}


Connector = Union[
    SchemaConnector, Registry, Mcp, AgentConnector, InlineAgentConnector, A2A
]

_CONNECTOR_TYPES: dict[str, type] = {
    cls.TYPE: cls  # type: ignore[attr-defined]
    for cls in [SchemaConnector, Registry, Mcp, AgentConnector, A2A]
}


def parse_connector(data: dict[str, Any]) -> Connector:
    """Discriminate on the ``type`` key and build the right connector variant.

    An ``agent`` connector without ``agentId`` is an inline definition — it
    passes through as an :class:`InlineAgentConnector` for the provisioning
    layer to resolve.

    Raises ``ValueError`` for unknown or missing ``type`` values.
    """
    ctype = data.get("type")
    if ctype == "agent" and "agentId" not in data:
        return InlineAgentConnector.from_dict(data)
    cls = _CONNECTOR_TYPES.get(ctype) if isinstance(ctype, str) else None
    if cls is None:
        raise ValueError(f"Unknown connector type: {ctype!r}")
    return cls.from_dict(data)  # type: ignore[return-value]
