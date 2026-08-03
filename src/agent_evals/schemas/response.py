from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .message import Message, Part


def _get_string(value: Any) -> Optional[str]:
    """Return ``value`` as a non-empty stripped string, else ``None``."""
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _find_string_in_mapping(payload: Any, keys: set[str]) -> Optional[str]:
    """Recursively search *payload* for the first non-empty string under any of *keys*."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in keys:
                candidate = _get_string(value)
                if candidate:
                    return candidate
            candidate = _find_string_in_mapping(value, keys)
            if candidate:
                return candidate
    elif isinstance(payload, list):
        for item in payload:
            candidate = _find_string_in_mapping(item, keys)
            if candidate:
                return candidate
    return None


@dataclass(slots=True)
class TaskStatus:
    state: Optional[str] = None
    message: Optional[Message] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskStatus":
        msg = data.get("message")
        return cls(
            # ``or`` (not default) semantics: an empty ``state`` falls through to
            # ``status`` — mirrors the old ``_extract_status_state`` helper.
            state=data.get("state") or data.get("status"),
            message=Message.from_dict(msg) if isinstance(msg, dict) else None,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.state is not None:
            result["state"] = self.state
        if self.message is not None:
            result["message"] = self.message.to_dict()
        return result


@dataclass(slots=True)
class Artifact:
    parts: List[Part] = field(default_factory=list)
    artifact_id: Optional[str] = None  # artifactId
    name: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Artifact":
        return cls(
            parts=[Part.from_dict(p) for p in data.get("parts") or []],
            artifact_id=data.get("artifactId", data.get("artifact_id")),
            name=data.get("name"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"parts": [p.to_dict() for p in self.parts]}
        if self.artifact_id:
            result["artifactId"] = self.artifact_id
        if self.name:
            result["name"] = self.name
        return result


@dataclass(slots=True)
class Task:
    id: Optional[str] = None
    context_id: Optional[str] = None  # contextId
    status: TaskStatus = field(default_factory=TaskStatus)
    artifacts: List[Artifact] = field(default_factory=list)
    history: List[Message] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        status = data.get("status")
        return cls(
            id=data.get("id"),
            context_id=data.get("contextId") or data.get("context_id"),
            status=TaskStatus.from_dict(status)
            if isinstance(status, dict)
            else TaskStatus(),
            artifacts=[Artifact.from_dict(a) for a in data.get("artifacts") or []],
            history=[Message.from_dict(m) for m in data.get("history") or []],
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"status": self.status.to_dict()}
        if self.id:
            result["id"] = self.id
        if self.context_id:
            result["contextId"] = self.context_id
        if self.artifacts:
            result["artifacts"] = [a.to_dict() for a in self.artifacts]
        if self.history:
            result["history"] = [m.to_dict() for m in self.history]
        return result


@dataclass(slots=True)
class Response:
    task: Task = field(default_factory=Task)
    context_id: Optional[str] = None
    raw: Optional[dict[str, Any]] = field(default=None, repr=False, compare=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Response":
        if not isinstance(data, dict):
            return cls()
        task_data = data.get("task", data)
        return cls(
            task=Task.from_dict(task_data) if isinstance(task_data, dict) else Task(),
            context_id=data.get("contextId") or data.get("context_id"),
            raw=data,
        )

    def to_dict(self) -> dict[str, Any]:
        # Prefer the verbatim original so nothing is dropped on round-trip.
        if self.raw is not None:
            return self.raw
        result: dict[str, Any] = {"task": self.task.to_dict()}
        if self.context_id:
            result["contextId"] = self.context_id
        return result

    @property
    def state(self) -> Optional[str]:
        """The task's status state (replaces ``_extract_status_state``)."""
        return _get_string(self.task.status.state)

    def resolved_context_id(self) -> Optional[str]:
        """Context id: top-level, then task-level, then a deep search of the raw
        response (replaces ``_extract_context_id``)."""
        direct = _get_string(self.context_id) or _get_string(self.task.context_id)
        if direct:
            return direct
        if self.raw is not None:
            return _find_string_in_mapping(self.raw, {"contextId", "context_id"})
        return None

    def resolved_task_id(self) -> Optional[str]:
        """Task id: the task's ``id``, then a deep search of the raw response for
        ``taskId``/``task_id`` (replaces ``_extract_task_id``)."""
        direct = _get_string(self.task.id)
        if direct:
            return direct
        if self.raw is not None:
            return _find_string_in_mapping(self.raw, {"taskId", "task_id"})
        return None
