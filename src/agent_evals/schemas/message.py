from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass(slots=True)
class Part:
    kind: str = "text"
    text: Optional[str] = None
    data: Optional[Any] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Part":
        return cls(
            kind=data.get("kind", "text"),
            text=data.get("text"),
            data=data.get("data"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind}
        if self.text is not None:
            result["text"] = self.text
        if self.data is not None:
            result["data"] = self.data
        return result


@dataclass(slots=True)
class Message:
    parts: List[Part] = field(default_factory=list)
    role: str = "user"
    kind: str = "message"
    message_id: Optional[str] = None  # messageId
    context_id: Optional[str] = None  # contextId
    task_id: Optional[str] = None  # taskId

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        return cls(
            parts=[Part.from_dict(p) for p in data.get("parts") or []],
            role=data.get("role", "user"),
            kind=data.get("kind", "message"),
            message_id=data.get("messageId", data.get("message_id")),
            context_id=data.get("contextId", data.get("context_id")),
            task_id=data.get("taskId", data.get("task_id")),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "role": self.role,
            "kind": self.kind,
            "parts": [p.to_dict() for p in self.parts],
        }
        # messageId is emitted even when empty ("") to mirror the source YAML.
        if self.message_id is not None:
            result["messageId"] = self.message_id
        if self.context_id:
            result["contextId"] = self.context_id
        if self.task_id:
            result["taskId"] = self.task_id
        return result


@dataclass(slots=True)
class MessagePayload:
    message: Message = field(default_factory=Message)
    configuration: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MessagePayload":
        return cls(
            message=Message.from_dict(data.get("message") or {}),
            configuration=data.get("configuration"),
            metadata=data.get("metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"message": self.message.to_dict()}
        if self.configuration:
            result["configuration"] = self.configuration
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    def prepare(
        self, context_id: str | None = None, task_id: str | None = None
    ) -> "MessagePayload":
        msg = self.message
        return MessagePayload(
            message=Message(
                parts=list(msg.parts),
                role=msg.role,
                kind=msg.kind,
                message_id=msg.message_id or str(uuid.uuid4()),
                context_id=msg.context_id or context_id,
                task_id=msg.task_id or task_id,
            ),
            configuration=self.configuration,
            metadata=self.metadata,
        )
