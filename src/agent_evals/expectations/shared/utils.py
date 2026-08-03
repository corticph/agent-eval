"""Shared response-shape helpers used across expectation types.

These walk the A2A task response structure (``task.status.message.parts``,
``task.artifacts[].parts``) to extract data-part payloads and source content.
They are not expectations themselves — each expectation type lives in its own
module — but are shared by several (``json``, ``faithful_to_source``,
``cited_pubmed_ids``).
"""

from __future__ import annotations

import json
from typing import Any

from ..base import _task_without_history


def extract_data_parts(response: dict[str, Any] | None) -> list[Any]:
    """Collect the ``data`` payloads of every data part.

    Looks in ``task.status.message.parts`` and ``task.artifacts[].parts``,
    returning each part's ``data`` value. This is the structured, machine-
    actionable output a data-part validator should target.
    """
    task = _task_without_history(response)
    if not task:
        return []
    payloads: list[Any] = []

    def _collect(parts: Any) -> None:
        if not isinstance(parts, list):
            return
        for part in parts:
            # A2A v2 Parts carry no ``kind`` discriminator — a data part is any
            # part with a ``data`` payload. (v1 sent ``kind: data`` alongside.)
            if isinstance(part, dict) and "data" in part:
                payloads.append(part["data"])

    status_msg = task.get("status", {})
    if isinstance(status_msg, dict):
        message = status_msg.get("message", {})
        if isinstance(message, dict):
            _collect(message.get("parts"))
    for artifact in task.get("artifacts", []):
        if isinstance(artifact, dict):
            _collect(artifact.get("parts"))
    return payloads


def _serialize_data_parts(response: dict[str, Any] | None) -> str:
    """Serialize every data part to one string (the retrieved-source corpus)."""
    return json.dumps(extract_data_parts(response), ensure_ascii=False)


def extract_source_at(response: dict[str, Any] | None, source_path: str) -> str:
    """Extract the source text at *source_path* from each data part, joined.

    ``$.response`` (the default) selects the stringified tool-output carrier. If
    the path is malformed or matches nothing, falls back to the whole serialized
    data-part corpus so the judge still sees the retrieved evidence.
    """
    from jsonpath_ng.ext import parse as _parse_jsonpath  # noqa: PLC0415

    data_parts = extract_data_parts(response)
    try:
        expr = _parse_jsonpath(source_path)
    except Exception:  # malformed path — fall back to the whole corpus
        return _serialize_data_parts(response)
    chunks: list[str] = []
    for part in data_parts:
        for match in expr.find(part):
            value = match.value
            chunks.append(
                value
                if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False)
            )
    if not chunks:
        return _serialize_data_parts(response)
    return "\n".join(chunks)
