"""Tests for the Environment module at the environment-resolution seam.

Inputs: an environment name plus process environment variables (monkeypatched).
Outputs: the resolved connection surface — base URL, request headers,
trace base — or a clean, specific error. The single network interaction
(the OAuth token request) is stubbed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

import agent_evals.environment
from agent_evals.environment import Environment, supported_environments

from conftest import OPIK_HOSTS, OPIK_PROJECT_IDS

# The internal environments' resolved URLs, hydrated by the autouse fixture.
_DEV_WEU_API = "https://api.dev-weu.test"
_DEV_WEU_AUTH = "https://auth.dev-weu.test"


@pytest.fixture(autouse=True)
def no_console_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's shell may carry the Console block; tenant and credential
    derivation tests must see it absent unless a test sets it."""
    for var in ("CORTI_TENANT_NAME", "CORTI_CLIENT_ID", "CORTI_CLIENT_SECRET"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture()
def no_oauth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if anything reaches for the network."""

    def _forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("unexpected OAuth token request")

    monkeypatch.setattr(agent_evals.environment.requests, "post", _forbidden)


class TestConstruction:
    @pytest.mark.parametrize("name", ["local", "dev-weu", "staging-eu", "eu", "us"])
    def test_supported_environments_construct(self, name: str) -> None:
        assert Environment(name).name == name

    def test_unknown_environment_rejected_with_supported_list(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            Environment("dev")
        message = str(excinfo.value)
        assert "dev" in message
        for supported in ("local", "dev-weu", "staging-eu", "eu", "us"):
            assert supported in message

    def test_supported_set_reflects_the_configured_environments(self) -> None:
        # With the internal variables present (the fixture's world), every
        # environment resolves — the behavior-preserving internal path.
        assert list(supported_environments()) == [
            "local",
            "dev-weu",
            "staging-eu",
            "eu",
            "us",
        ]

    def test_internal_environments_absent_when_their_vars_are_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An external checkout holds none of the internal URL variables, so
        # the internal environments are not part of the supported set.
        for name in ("DEV_WEU", "STAGING_EU"):
            monkeypatch.delenv(f"AGENT_API_URL_{name}", raising=False)
            monkeypatch.delenv(f"AGENT_API_AUTH_URL_{name}", raising=False)
        assert list(supported_environments()) == ["local", "eu", "us"]

    def test_unconfigured_internal_environment_is_rejected_by_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Selecting an internal environment a checkout cannot reach fails the
        # same way an unknown name does — the name is never offered back.
        monkeypatch.delenv("AGENT_API_URL_DEV_WEU", raising=False)
        monkeypatch.delenv("AGENT_API_AUTH_URL_DEV_WEU", raising=False)
        with pytest.raises(ValueError) as excinfo:
            Environment("dev-weu")
        message = str(excinfo.value)
        assert "dev-weu" in message
        for offered in ("local", "eu", "us"):
            assert offered in message


class TestBaseUrl:
    @pytest.mark.parametrize(
        ("name", "base_url"),
        [
            # Product environments carry their base URL as a source literal;
            # the internal ones hydrate it from env (the fixture's placeholder).
            ("local", "http://localhost:8080"),
            ("dev-weu", "https://api.dev-weu.test"),
            ("staging-eu", "https://api.staging-eu.test"),
            ("eu", "https://api.eu.corti.app"),
            ("us", "https://api.us.corti.app"),
        ],
    )
    def test_base_url_per_environment(self, name: str, base_url: str) -> None:
        assert Environment(name).base_url == base_url

    def test_local_url_override_wins_over_the_literal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Local carries both a literal default and an override variable, for
        # hosts where another process owns port 8080 and the local stack is
        # published elsewhere.
        monkeypatch.setenv("AGENT_API_URL_LOCAL", "http://localhost:8088")
        assert Environment("local").base_url == "http://localhost:8088"


class TestLocalTokenResolution:
    def test_resolves_static_token_without_oauth(
        self, monkeypatch: pytest.MonkeyPatch, no_oauth: None
    ) -> None:
        monkeypatch.setenv("AGENT_API_TOKEN_LOCAL", "static-tok")
        assert Environment("local").resolve_token() == "static-tok"

    def test_missing_token_names_the_variable(
        self, monkeypatch: pytest.MonkeyPatch, no_oauth: None
    ) -> None:
        monkeypatch.delenv("AGENT_API_TOKEN_LOCAL", raising=False)
        with pytest.raises(ValueError, match="AGENT_API_TOKEN_LOCAL"):
            Environment("local").resolve_token()


def _stub_token_endpoint(
    monkeypatch: pytest.MonkeyPatch, token: str = "oauth-tok"
) -> list[tuple[str, dict[str, str]]]:
    """Substitute the OAuth token request; record (auth_url, payload) calls."""
    calls: list[tuple[str, dict[str, str]]] = []

    def _post(url: str, *, data: dict[str, str], timeout: object = None) -> object:
        calls.append((url, data))
        response = MagicMock()
        response.json.return_value = {"access_token": token}
        response.raise_for_status.return_value = None
        return response

    monkeypatch.setattr(agent_evals.environment.requests, "post", _post)
    return calls


class TestRemoteTokenResolution:
    @pytest.fixture(autouse=True)
    def credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_API_CLIENT_ID_DEV_WEU", "cid")
        monkeypatch.setenv("AGENT_API_CLIENT_SECRET_DEV_WEU", "csecret")

    def test_returns_the_oauth_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_token_endpoint(monkeypatch, token="oauth-tok")
        assert Environment("dev-weu").resolve_token() == "oauth-tok"

    def test_posts_client_credentials_to_the_environments_auth_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _stub_token_endpoint(monkeypatch)
        Environment("dev-weu").resolve_token()
        ((auth_url, payload),) = calls
        assert auth_url == f"{_DEV_WEU_AUTH}/realms/base/protocol/openid-connect/token"
        assert payload == {
            "client_id": "cid",
            "client_secret": "csecret",
            "grant_type": "client_credentials",
            "scope": "openid",
        }

    @pytest.mark.parametrize("name", ["dev-weu", "staging-eu", "eu", "us"])
    def test_missing_credentials_name_both_routes(
        self, monkeypatch: pytest.MonkeyPatch, no_oauth: None, name: str
    ) -> None:
        suffix = name.upper().replace("-", "_")
        monkeypatch.delenv(f"AGENT_API_CLIENT_ID_{suffix}", raising=False)
        monkeypatch.delenv(f"AGENT_API_CLIENT_SECRET_{suffix}", raising=False)
        with pytest.raises(ValueError) as excinfo:
            Environment(name).resolve_token()
        message = str(excinfo.value)
        assert f"AGENT_API_CLIENT_ID_{suffix}" in message
        assert f"AGENT_API_CLIENT_SECRET_{suffix}" in message
        # The Console block is the other way in.
        assert "CORTI_CLIENT_ID" in message
        assert "CORTI_CLIENT_SECRET" in message

    def test_console_block_is_the_credential_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An external customer runs on the pasted Console block alone."""
        monkeypatch.delenv("AGENT_API_CLIENT_ID_DEV_WEU", raising=False)
        monkeypatch.delenv("AGENT_API_CLIENT_SECRET_DEV_WEU", raising=False)
        monkeypatch.setenv("CORTI_CLIENT_ID", "console-cid")
        monkeypatch.setenv("CORTI_CLIENT_SECRET", "console-secret")
        calls = _stub_token_endpoint(monkeypatch)

        Environment("dev-weu").resolve_token()

        ((_, payload),) = calls
        assert payload["client_id"] == "console-cid"
        assert payload["client_secret"] == "console-secret"

    def test_environment_pair_wins_over_the_console_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Specific beats general: the env's own pair outranks the block."""
        monkeypatch.setenv("CORTI_CLIENT_ID", "console-cid")
        monkeypatch.setenv("CORTI_CLIENT_SECRET", "console-secret")
        calls = _stub_token_endpoint(monkeypatch)

        Environment("dev-weu").resolve_token()

        ((_, payload),) = calls
        assert payload["client_id"] == "cid"
        assert payload["client_secret"] == "csecret"

    def test_tenant_names_the_auth_realm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CORTI_TENANT_NAME", "acme")
        calls = _stub_token_endpoint(monkeypatch)

        Environment("dev-weu").resolve_token()

        ((auth_url, _),) = calls
        assert auth_url == f"{_DEV_WEU_AUTH}/realms/acme/protocol/openid-connect/token"

    def test_failed_oauth_call_names_the_variables(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _post(url: str, *, data: dict[str, str], timeout: object = None) -> object:
            raise requests.ConnectionError("refused")

        monkeypatch.setattr(agent_evals.environment.requests, "post", _post)
        with pytest.raises(RuntimeError) as excinfo:
            Environment("dev-weu").resolve_token()
        message = str(excinfo.value)
        assert "AGENT_API_CLIENT_ID_DEV_WEU" in message
        assert "AGENT_API_CLIENT_SECRET_DEV_WEU" in message
        assert _DEV_WEU_AUTH in message

    def test_token_is_fetched_once_per_instance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Concurrent cases clone their client around one Environment; a
        # fetch per clone would re-run the OAuth flow for every case.
        calls = _stub_token_endpoint(monkeypatch)
        environment = Environment("dev-weu")
        environment.resolve_token()
        environment.resolve_token()
        assert len(calls) == 1

    def test_token_response_without_access_token_is_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _post(url: str, *, data: dict[str, str], timeout: object = None) -> object:
            response = MagicMock()
            response.json.return_value = {"error": "nope"}
            response.raise_for_status.return_value = None
            return response

        monkeypatch.setattr(agent_evals.environment.requests, "post", _post)
        with pytest.raises(ValueError, match="access_token"):
            Environment("dev-weu").resolve_token()


class TestHeaders:
    def test_local_headers_carry_bearer_and_default_tenant(
        self, monkeypatch: pytest.MonkeyPatch, no_oauth: None
    ) -> None:
        monkeypatch.setenv("AGENT_API_TOKEN_LOCAL", "static-tok")
        assert Environment("local").headers() == {
            "Content-Type": "application/json",
            "Tenant-Name": "base",
            "Authorization": "Bearer static-tok",
        }

    def test_console_block_tenant_reaches_the_header(
        self, monkeypatch: pytest.MonkeyPatch, no_oauth: None
    ) -> None:
        # Tenant is one concept: the same value picks the realm (covered in
        # TestRemoteTokenResolution) and identifies the tenant per request.
        monkeypatch.setenv("AGENT_API_TOKEN_LOCAL", "static-tok")
        monkeypatch.setenv("CORTI_TENANT_NAME", "acme")
        assert Environment("local").headers()["Tenant-Name"] == "acme"

    def test_remote_headers_carry_the_resolved_oauth_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AGENT_API_CLIENT_ID_EU", "cid")
        monkeypatch.setenv("AGENT_API_CLIENT_SECRET_EU", "csecret")
        _stub_token_endpoint(monkeypatch, token="oauth-tok")
        assert Environment("eu").headers() == {
            "Content-Type": "application/json",
            "Tenant-Name": "base",
            "Authorization": "Bearer oauth-tok",
        }


class TestTraceSurface:
    @pytest.fixture(autouse=True)
    def no_ambient_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A developer's shell may carry OPIK_URL_OVERRIDE; derivation tests
        must see the un-overridden environment unless a test sets it."""
        monkeypatch.delenv("OPIK_URL_OVERRIDE", raising=False)

    def test_local_derives_no_trace_links(self) -> None:
        environment = Environment("local")
        assert environment.trace_base_url is None
        assert environment.trace_project_id is None

    def test_remote_trace_base_is_the_environments_opik_thread_link(self) -> None:
        # The agent's contextId is the Opik thread id; a trace link is the
        # project's Logs view filtered to that thread. Opik host and project
        # both hydrate from env (the fixture's placeholders).
        environment = Environment("dev-weu")
        project_id = OPIK_PROJECT_IDS["dev-weu"]
        assert environment.trace_base_url == (
            f"{OPIK_HOSTS['dev-weu']}/default/projects/{project_id}/logs?thread="
        )
        assert environment.trace_project_id == project_id

    @pytest.mark.parametrize("name", ["dev-weu", "staging-eu", "eu", "us"])
    def test_remote_project_ids_hydrate_per_environment(self, name: str) -> None:
        assert Environment(name).trace_project_id == OPIK_PROJECT_IDS[name]

    def test_no_project_id_yields_no_trace_link(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An external checkout sets no OPIK_PROJECT_ID_*; a remote environment
        # with no project id derives no trace link, with no external-only branch.
        monkeypatch.delenv("OPIK_PROJECT_ID_EU", raising=False)
        environment = Environment("eu")
        assert environment.trace_project_id is None
        assert environment.trace_base_url is None

    def test_override_redirects_the_trace_base_host(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # OPIK_URL_OVERRIDE is the single "which Opik" override: sink writes
        # already honor it, so links must too — or they point somewhere the
        # traces never went.
        monkeypatch.setenv("OPIK_URL_OVERRIDE", "http://localhost:5173")
        project_id = OPIK_PROJECT_IDS["dev-weu"]
        assert Environment("dev-weu").trace_base_url == (
            f"http://localhost:5173/default/projects/{project_id}/logs?thread="
        )

    def test_override_with_trailing_slash_builds_a_clean_link(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPIK_URL_OVERRIDE", "http://localhost:5173/")
        project_id = OPIK_PROJECT_IDS["eu"]
        assert Environment("eu").trace_base_url == (
            f"http://localhost:5173/default/projects/{project_id}/logs?thread="
        )

    def test_override_does_not_conjure_links_for_local(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Local has no trace project anywhere; redirecting the host cannot
        # invent one, so the Trace line stays omitted.
        monkeypatch.setenv("OPIK_URL_OVERRIDE", "http://localhost:5173")
        assert Environment("local").trace_base_url is None
