"""Shared test fixtures.

The internal environments and every environment's Opik host and project id
now hydrate from environment variables rather than source literals, so
nothing internal — no host, no project id — is committed. Tests
therefore stand in the shoes of *an internal operator holding the shared
`.env`*: an autouse fixture populates those variables with placeholder
values so all five environments resolve, reproducing the behavior-preserving
internal path. Values are deliberately fake; a test that needs the external
(unconfigured) world deletes the variables it cares about.
"""

from __future__ import annotations

import pytest

from agent_evals import tracing

# Placeholder hydration for the environments that used to be source literals.
# Non-corti hosts and non-UUID project ids on purpose: these are test doubles,
# not the real internal values, which live only in the shared `.env`.
INTERNAL_ENV_VARS: dict[str, str] = {
    "AGENT_API_URL_DEV_WEU": "https://api.dev-weu.test",
    "AGENT_API_AUTH_URL_DEV_WEU": "https://auth.dev-weu.test",
    "AGENT_API_URL_STAGING_EU": "https://api.staging-eu.test",
    "AGENT_API_AUTH_URL_STAGING_EU": "https://auth.staging-eu.test",
}

# Every environment's Opik host + project, keyed by env name.
OPIK_HOSTS: dict[str, str] = {
    "dev-weu": "https://opik.dev-weu.test",
    "staging-eu": "https://opik.staging-eu.test",
    "eu": "https://opik.eu.test",
    "us": "https://opik.us.test",
}
OPIK_PROJECT_IDS: dict[str, str] = {
    "dev-weu": "proj-dev-weu",
    "staging-eu": "proj-staging-eu",
    "eu": "proj-eu",
    "us": "proj-us",
}


def _suffix(name: str) -> str:
    return name.upper().replace("-", "_")


OPIK_ENV_VARS: dict[str, str] = {
    **{f"OPIK_URL_{_suffix(n)}": host for n, host in OPIK_HOSTS.items()},
    **{f"OPIK_PROJECT_ID_{_suffix(n)}": pid for n, pid in OPIK_PROJECT_IDS.items()},
}


@pytest.fixture(autouse=True)
def hydrate_environments(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test the internal-operator world: all environments resolve."""
    for var, value in {**INTERNAL_ENV_VARS, **OPIK_ENV_VARS}.items():
        monkeypatch.setenv(var, value)
    # A developer's shell may carry the local URL override; tests must see
    # the default local URL unless a test sets it.
    monkeypatch.delenv("AGENT_API_URL_LOCAL", raising=False)


@pytest.fixture(autouse=True)
def fast_trace_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop the trace-poll sleeps so the suite does not wait out real windows.

    Only the delays are patched; the retry budget and the stabilization rule
    they pace stay exactly as shipped, so tests still exercise the real policy.
    """
    monkeypatch.setattr(tracing, "_TRACE_RETRY_DELAY", 0.0)
    monkeypatch.setattr(tracing, "_TRACE_STABILIZE_DELAY", 0.0)
