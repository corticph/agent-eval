from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass(slots=True)
class Agent:
    """The v2 create-agent payload, exactly as authored in the suite YAML.

    Connectors ride verbatim: the payload is already the wire shapes, so the
    only work left before the create call is provisioning inline ``agent``
    connectors (a definition instead of an ``agentId``) — see provisioning.
    """

    name: str
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    connectors: List[dict[str, Any]] = field(default_factory=list)
    model: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Agent":
        return cls(
            name=data["name"],
            description=data.get("description"),
            system_prompt=data.get("systemPrompt"),
            connectors=list(data.get("connectors") or []),
            model=data.get("model"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"name": self.name}
        if self.description:
            result["description"] = self.description
        if self.system_prompt:
            result["systemPrompt"] = self.system_prompt
        if self.connectors:
            result["connectors"] = list(self.connectors)
        if self.model:
            result["model"] = self.model
        return result
