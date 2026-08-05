"""HTTP client for interacting with the agent service."""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

from .environment import Environment
from .errors import (
    HttpStatusError,
    InvalidResponseError,
    NetworkError,
    RequestTimeoutError,
)

_LOGGER = logging.getLogger(__name__)

# (connect timeout, read timeout) in seconds
_DEFAULT_TIMEOUT: tuple[float, float] = (5, 120)

# Longest a single HTTP request may block with the default timeouts. Eval
# wall-clock timeouts (``timeout_seconds``) must exceed this: the timeout
# thread can only pre-empt a worker between requests, so a smaller bound
# could never interrupt an in-flight request.
MAX_HTTP_TIMEOUT_SECONDS: float = float(sum(_DEFAULT_TIMEOUT))

_EVAL_AGENT_PREFIX = "EVAL_"


_A2A_VERSION = "1.0"


def _prefix_agent_name(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the agent-creation payload with ``EVAL_`` prefixed to the name.

    Ensures every agent created during eval runs is identifiable in downstream
    logging/tracing (e.g. Opik). Idempotent: a name already prefixed is left alone.
    """
    name = payload.get("name")
    if not isinstance(name, str) or name.startswith(_EVAL_AGENT_PREFIX):
        return payload
    return {**payload, "name": f"{_EVAL_AGENT_PREFIX}{name}"}


def _ensure_ephemeral(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("lifecycle"):
        return payload
    return {**payload, "lifecycle": "ephemeral"}


class AgentClient:
    """Simple wrapper over the agent HTTP API."""

    def __init__(
        self, environment: Environment, *, session: requests.Session | None = None
    ) -> None:
        self.environment = environment
        self.session = session or requests.Session()
        self.session.headers.update(environment.headers())

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        params: dict[str, str | int | bool] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | tuple[float, float] | None = _DEFAULT_TIMEOUT,
    ) -> dict[str, Any]:
        url = f"{self.environment.base_url}{path}"
        _LOGGER.debug("%s %s", method, url)
        if json_body is not None:
            _LOGGER.debug("Request body: %s", json.dumps(json_body, indent=2))
        try:
            response = self.session.request(
                method,
                url,
                json=json_body,
                params=params,
                headers=headers,
                timeout=timeout,
            )
        except requests.ConnectionError as exc:
            raise NetworkError(
                f"Connection failed: could not reach {self.environment.base_url}\n"
                f"Is the service running?"
            ) from exc
        except requests.Timeout as exc:
            raise RequestTimeoutError(
                f"Request timed out: {method} {url}\n"
                f"The service at {self.environment.base_url} did not respond in time."
            ) from exc
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            _LOGGER.error("Request failed: %s", exc, exc_info=True)
            if response.content:
                _LOGGER.error("Response body: %s", response.text)
            raise HttpStatusError(
                f"HTTP {response.status_code} error for {method} {url}"
            ) from exc
        if not response.content:
            return {}
        try:
            data = response.json()
            _LOGGER.debug("Response body: %s", json.dumps(data, indent=2))
            return data
        except json.JSONDecodeError as exc:
            _LOGGER.error("Response was not JSON: %s", response.text)
            raise InvalidResponseError(
                f"Response was not valid JSON for {method} {url}"
            ) from exc

    def create_agent(
        self,
        payload: dict[str, Any],
        *,
        timeout: float | tuple[float, float] | None = _DEFAULT_TIMEOUT,
    ) -> dict[str, Any]:
        """Create a new agent and return the response JSON."""
        payload = _ensure_ephemeral(_prefix_agent_name(payload))
        return self._request(
            "POST",
            "/v2/agentic/agents",
            json_body=payload,
            timeout=timeout,
        )

    def send_message(
        self,
        agent_id: str,
        payload: dict[str, Any],
        *,
        timeout: float | tuple[float, float] | None = _DEFAULT_TIMEOUT,
    ) -> dict[str, Any]:
        """Send a message to an existing agent (A2A v1.0 HTTP+JSON binding)."""
        path = f"/v2/agentic/agents/{agent_id}/a2a/message:send"
        return self._request(
            "POST",
            path,
            json_body=payload,
            headers={"A2A-Version": _A2A_VERSION},
            timeout=timeout,
        )

    def get_trace(
        self,
        context_id: str,
        *,
        timeout: float | tuple[float, float] | None = _DEFAULT_TIMEOUT,
    ) -> dict[str, Any]:
        """Fetch the OpenInference trace for a context (GET /v2/agentic/contexts/{id}/trace)."""
        path = f"/v2/agentic/contexts/{context_id}/trace"
        return self._request("GET", path, timeout=timeout)

    def close(self) -> None:
        """Release underlying HTTP resources."""
        self.session.close()

    def clone(self) -> "AgentClient":
        """Return a new client sharing the environment but not the HTTP session."""
        return AgentClient(self.environment)
