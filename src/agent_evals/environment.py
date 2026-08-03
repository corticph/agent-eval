"""The Environment: the single source of connection configuration.

Per ADR 0001, an Environment resolves everything needed to talk to one
deployment of the agent platform — base URL, auth, tenant header, trace
destination. Environments differ internally (local authenticates with a
static token; remote environments use OAuth), but that mismatch never
leaves this module: the public surface is identical for every row.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass

import requests
from requests import exceptions as request_exceptions

# Tenant is one concept by design: the Keycloak realm that issues the token
# and the Tenant-Name header sent with every request. Named by the Console
# block's CORTI_TENANT_NAME — external customers run as their own tenant;
# 'base' when none is given.
DEFAULT_TENANT = "base"

# The one "which Opik" override, shared between the Opik sink's writes and
# result-file trace links so links always match where traces actually went.
# The name matches agent-api's own variable.
OPIK_URL_OVERRIDE_VAR = "OPIK_URL_OVERRIDE"

# The Corti Console's "Copy all as .env variables" block (Developer
# Quickstart). CORTI_ENVIRONMENT is part of that block; `make setup` uses
# it to select the judge endpoint (ai.eu/ai.us.corti.app). The run's
# environment stays an explicit choice via --env.
CORTI_TENANT_NAME_VAR = "CORTI_TENANT_NAME"
CORTI_CLIENT_ID_VAR = "CORTI_CLIENT_ID"
CORTI_CLIENT_SECRET_VAR = "CORTI_CLIENT_SECRET"

_OAUTH_SCOPE = "openid"


def tenant() -> str:
    """The Tenant this run acts as: the Console block's name, else 'base'."""
    return os.environ.get(CORTI_TENANT_NAME_VAR) or DEFAULT_TENANT


# Seconds before an unanswered token request fails (rather than hanging).
_TOKEN_REQUEST_TIMEOUT = 30.0


@dataclass(frozen=True, slots=True)
class _Spec:
    """How one environment's row is built: its literal facts plus the
    environment variables that hydrate the sensitive ones.

    The public product environments carry their API/auth URLs as literals —
    those are the product, not a secret. Everything internal or secret (the
    internal environments' URLs, and every environment's Opik host and
    project id) is named here only by the variable that supplies it, so no
    internal host or project id lives in source. A spec whose URL-supplying
    variables do not resolve does not materialise into a row, so an
    unconfigured internal environment is simply not offered.
    """

    # API/auth URLs: a literal (public product / local) or the variable that
    # supplies it (internal environments). When both are present (local), the
    # variable overrides the literal — e.g. the API published on a non-default
    # host port. Auth is spelled as a base host; the realm path is appended
    # per request, since the realm is the run's tenant.
    api_url: str | None = None
    api_url_var: str | None = None
    auth_url: str | None = None
    auth_url_var: str | None = None
    # Exactly one of the two auth strategies per row: a static-token variable
    # (local — its OAuth client credentials are not valid), or credential
    # variables for the OAuth client-credentials flow against the auth host.
    token_var: str | None = None
    client_id_var: str | None = None
    client_secret_var: str | None = None
    # The environment's Opik instance and the project its eval traces live in,
    # both hydrated from env. Absent (local, or an external checkout) means no
    # trace links, which falls out of ``trace_base_url`` on its own.
    opik_url_var: str | None = None
    opik_project_id_var: str | None = None


@dataclass(frozen=True, slots=True)
class _Row:
    """One environment's resolved connection facts.

    URLs are carried explicitly rather than derived from the environment
    name: the name ≡ corti.app-subdomain assumption is exactly what local
    violates. Where a value is sensitive it was hydrated from env at
    construction; the source of a field is an internal detail here. The auth
    host carries no realm path — the realm is the run's tenant, resolved at
    request time, not construction time.
    """

    api_url: str
    token_var: str | None = None
    auth_host: str | None = None
    client_id_var: str | None = None
    client_secret_var: str | None = None
    opik_url: str | None = None
    opik_project_id: str | None = None


_ENVIRONMENT_SPECS: dict[str, _Spec] = {
    "local": _Spec(
        api_url="http://localhost:8080",
        api_url_var="AGENT_API_URL_LOCAL",
        token_var="AGENT_API_TOKEN_LOCAL",
    ),
    "dev-weu": _Spec(
        api_url_var="AGENT_API_URL_DEV_WEU",
        auth_url_var="AGENT_API_AUTH_URL_DEV_WEU",
        client_id_var="AGENT_API_CLIENT_ID_DEV_WEU",
        client_secret_var="AGENT_API_CLIENT_SECRET_DEV_WEU",
        opik_url_var="OPIK_URL_DEV_WEU",
        opik_project_id_var="OPIK_PROJECT_ID_DEV_WEU",
    ),
    "staging-eu": _Spec(
        api_url_var="AGENT_API_URL_STAGING_EU",
        auth_url_var="AGENT_API_AUTH_URL_STAGING_EU",
        client_id_var="AGENT_API_CLIENT_ID_STAGING_EU",
        client_secret_var="AGENT_API_CLIENT_SECRET_STAGING_EU",
        opik_url_var="OPIK_URL_STAGING_EU",
        opik_project_id_var="OPIK_PROJECT_ID_STAGING_EU",
    ),
    "eu": _Spec(
        api_url="https://api.eu.corti.app",
        auth_url="https://auth.eu.corti.app",
        client_id_var="AGENT_API_CLIENT_ID_EU",
        client_secret_var="AGENT_API_CLIENT_SECRET_EU",
        opik_url_var="OPIK_URL_EU",
        opik_project_id_var="OPIK_PROJECT_ID_EU",
    ),
    "us": _Spec(
        api_url="https://api.us.corti.app",
        auth_url="https://auth.us.corti.app",
        client_id_var="AGENT_API_CLIENT_ID_US",
        client_secret_var="AGENT_API_CLIENT_SECRET_US",
        opik_url_var="OPIK_URL_US",
        opik_project_id_var="OPIK_PROJECT_ID_US",
    ),
}


def _var(name: str | None) -> str | None:
    """Read an environment variable, treating empty as absent."""
    return os.environ.get(name) or None if name else None


def _materialize(spec: _Spec) -> _Row | None:
    """Build a row from a spec against the current process environment.

    Returns ``None`` when the environment is not configured to be reachable:
    an internal environment whose URL variables are unset is not offered, so
    an external checkout never lists or resolves it. Credentials are *not*
    part of this gate — a URL-configured environment with missing credentials
    is offered and fails at ``resolve_token`` with a message naming what to
    set (a bad paste should be an actionable error, not a hidden target).
    """
    api_url = _var(spec.api_url_var) or spec.api_url
    if api_url is None:
        return None
    auth_host = _var(spec.auth_url_var) or spec.auth_url
    if spec.token_var is None and auth_host is None:
        return None
    return _Row(
        api_url=api_url,
        token_var=spec.token_var,
        auth_host=auth_host,
        client_id_var=spec.client_id_var,
        client_secret_var=spec.client_secret_var,
        opik_url=_var(spec.opik_url_var),
        opik_project_id=_var(spec.opik_project_id_var),
    )


def supported_environments() -> tuple[str, ...]:
    """The environments this checkout can actually reach, in table order.

    Derived from which specs resolve against the current environment rather
    than a fixed tuple, so the CLI never advertises a target it cannot
    resolve and an external checkout exposes only ``local``, ``eu``, ``us``.
    """
    return tuple(
        name for name, spec in _ENVIRONMENT_SPECS.items() if _materialize(spec)
    )


def corti_models_api_key() -> str | None:
    """The Corti Models bearer assembled from the Console's CORTI_* variables.

    Corti Models does not accept OAuth access tokens yet ("supported soon"
    per its quickstart) nor the raw client secret; its bearer is
    ``base64("<tenant>:client_credentials:<client_id>:<secret>")``.
    The format is undocumented on docs.corti.ai — it matches ``buildBearer()``
    in ``@corti/cli`` (src/core/corti-api.ts), which assembles the same
    Console credentials for coding agents. Returns ``None`` when the client
    id or secret is missing; the tenant falls back to ``DEFAULT_TENANT``.
    """
    client_id = os.environ.get(CORTI_CLIENT_ID_VAR)
    client_secret = os.environ.get(CORTI_CLIENT_SECRET_VAR)
    if not client_id or not client_secret:
        return None
    raw = f"{tenant()}:client_credentials:{client_id}:{client_secret}"
    return base64.b64encode(raw.encode()).decode()


@dataclass(frozen=True, slots=True)
class Knob:
    """One user-facing configuration knob: name, env var(s), default, effect.

    These declarations are the configuration inventory — user-facing help
    renders from them, so documentation and behaviour cannot drift.
    """

    name: str
    env_vars: tuple[str, ...]
    effect: str
    default: str | None = None  # None renders as required / no default


def configuration_knobs() -> tuple[Knob, ...]:
    """Every surviving configuration knob, derived from the resolved roster.

    Only environments this checkout can reach contribute credential knobs, so
    an external checkout's help never names credentials for an environment it
    cannot use. The Console block, the Opik override, and the judge's
    variables always apply and follow.
    """
    knobs = []
    for name in supported_environments():
        spec = _ENVIRONMENT_SPECS[name]
        if spec.api_url and spec.api_url_var:
            knobs.append(
                Knob(
                    name=f"{name} URL",
                    env_vars=(spec.api_url_var,),
                    effect=(
                        f"Base URL for '{name}' — set when the API is "
                        "published somewhere other than the default."
                    ),
                    default=spec.api_url,
                )
            )
        if spec.token_var is not None:
            knobs.append(
                Knob(
                    name=f"{name} token",
                    env_vars=(spec.token_var,),
                    effect=f"Static bearer token for '{name}' (it has no OAuth).",
                )
            )
        else:
            if not spec.client_id_var or not spec.client_secret_var:
                raise RuntimeError(
                    f"Environment '{name}' has no token_var but is missing OAuth "
                    "credential vars. This is a bug in the _ENVIRONMENT_SPECS table."
                )
            knobs.append(
                Knob(
                    name=f"{name} credentials",
                    env_vars=(spec.client_id_var, spec.client_secret_var),
                    effect=f"OAuth client credentials for '{name}'.",
                    default="the Console block's client id and secret",
                )
            )
    knobs.append(
        Knob(
            name="Opik override",
            env_vars=(OPIK_URL_OVERRIDE_VAR,),
            effect=(
                "Redirect the Opik sink's writes and result-file trace links "
                "to another Opik instance."
            ),
            default="the environment's own Opik",
        )
    )
    # Names and defaults come from the judge module so this inventory cannot
    # drift from what the judge actually reads. Imported locally: this is a
    # leaf config module and must not drag the expectations package (and the
    # openai SDK) into everything that merely resolves a connection.
    from .expectations.llm.base import (
        DEFAULT_JUDGE_BASE_URL,
        DEFAULT_JUDGE_MODEL,
        JUDGE_API_KEY_VAR,
        JUDGE_BASE_URL_VAR,
        JUDGE_MODEL_VAR,
    )

    knobs.extend(
        (
            Knob(
                name="Console block",
                env_vars=(
                    CORTI_TENANT_NAME_VAR,
                    CORTI_CLIENT_ID_VAR,
                    CORTI_CLIENT_SECRET_VAR,
                ),
                effect=(
                    "The Corti Console's 'Copy all as .env variables' block: "
                    "names the tenant (realm and Tenant-Name header), serves "
                    "as the fallback OAuth client for any environment without "
                    "its own credentials, and is assembled into the Corti "
                    "Models bearer for judge verdicts. The tenant alone falls "
                    f"back to '{DEFAULT_TENANT}'."
                ),
            ),
            Knob(
                name="judge API key",
                env_vars=(JUDGE_API_KEY_VAR,),
                effect=(
                    "Pre-built Corti Models bearer for judge verdicts; "
                    "overrides the CORTI_* credentials when set."
                ),
                default="assembled from the CORTI_* credentials",
            ),
            Knob(
                name="judge endpoint",
                env_vars=(JUDGE_BASE_URL_VAR,),
                effect="The Corti Models endpoint judge verdicts are sent to.",
                default=DEFAULT_JUDGE_BASE_URL,
            ),
            Knob(
                name="judge model",
                env_vars=(JUDGE_MODEL_VAR,),
                effect="The model that delivers judge verdicts.",
                default=DEFAULT_JUDGE_MODEL,
            ),
        )
    )
    return tuple(knobs)


def configuration_help() -> str:
    """Render the knob inventory as a --help epilog."""
    lines = [
        "configuration (environment variables; a .env file in the working "
        "directory is loaded automatically):"
    ]
    for knob in configuration_knobs():
        lines.append("  " + "\n  ".join(knob.env_vars))
        default = f" Default: {knob.default}." if knob.default else ""
        lines.append(f"        {knob.effect}{default}")
    return "\n".join(lines)


class Environment:
    """A named deployment target with a uniform connection surface."""

    def __init__(self, name: str) -> None:
        spec = _ENVIRONMENT_SPECS.get(name)
        row = _materialize(spec) if spec is not None else None
        if row is None:
            # Both a truly unknown name and an internal environment that this
            # checkout is not configured for land here: the caller is told
            # what it *can* reach, never that a hidden target exists.
            raise ValueError(
                f"Unknown environment '{name}'. "
                f"Supported environments: {', '.join(supported_environments())}"
            )
        self.name = name
        self._row = row
        self._token: str | None = None

    @property
    def base_url(self) -> str:
        return self._row.api_url

    @property
    def trace_project_id(self) -> str | None:
        return self._row.opik_project_id

    @property
    def trace_base_url(self) -> str | None:
        """Thread-link prefix for this environment's traces, or ``None``.

        The agent's ``contextId`` is the Opik thread id; appending it yields
        the project's Logs view filtered to that thread. ``OPIK_URL_OVERRIDE``
        — the one "which Opik" override, shared with sink writes — replaces
        the environment's Opik host so links point where traces actually
        went. It cannot conjure links for an environment with no trace
        project (local, or an external checkout).
        """
        row = self._row
        opik_url = os.environ.get(OPIK_URL_OVERRIDE_VAR) or row.opik_url
        if opik_url is None or row.opik_project_id is None:
            return None
        return (
            f"{opik_url.rstrip('/')}/default/projects/"
            f"{row.opik_project_id}/logs?thread="
        )

    def headers(self) -> dict[str, str]:
        """Request headers for this environment.

        The only place a bearer value is constructed; the auth header is
        always ``Authorization`` and the tenant header is the run's tenant.
        """
        return {
            "Content-Type": "application/json",
            "Tenant-Name": tenant(),
            "Authorization": f"Bearer {self.resolve_token()}",
        }

    def resolve_token(self) -> str:
        """Resolve the bearer token, once per Environment instance.

        Concurrent cases clone their client around the same instance; without
        the cache every clone would re-run the OAuth flow.
        """
        if self._token is not None:
            return self._token
        if self._row.token_var is not None:
            token = os.environ.get(self._row.token_var)
            if not token:
                raise ValueError(
                    f"No token for environment '{self.name}': "
                    f"set {self._row.token_var}."
                )
        else:
            token = self._request_oauth_token()
        self._token = token
        return token

    def _request_oauth_token(self) -> str:
        row = self._row
        if not row.auth_host or not row.client_id_var or not row.client_secret_var:
            raise RuntimeError(
                f"Environment '{self.name}' has no token_var but is missing OAuth "
                "fields (auth_host, client_id_var, client_secret_var). "
                "This is a bug in the _ENVIRONMENT_SPECS table."
            )
        # Specific beats general: the environment's own pair wins; the
        # Console block is the fallback so an external customer runs on the
        # pasted block alone.
        client_id = os.environ.get(row.client_id_var) or os.environ.get(
            CORTI_CLIENT_ID_VAR
        )
        client_secret = os.environ.get(row.client_secret_var) or os.environ.get(
            CORTI_CLIENT_SECRET_VAR
        )
        if not client_id or not client_secret:
            raise ValueError(
                f"No credentials for environment '{self.name}': "
                f"set {row.client_id_var} and {row.client_secret_var}, or "
                f"{CORTI_CLIENT_ID_VAR} and {CORTI_CLIENT_SECRET_VAR} from "
                "the Corti Console."
            )
        auth_url = f"{row.auth_host}/realms/{tenant()}/protocol/openid-connect/token"
        payload = {
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "client_credentials",
            "scope": _OAUTH_SCOPE,
        }
        try:
            response = requests.post(
                auth_url, data=payload, timeout=_TOKEN_REQUEST_TIMEOUT
            )
            response.raise_for_status()
        except request_exceptions.RequestException as exc:
            raise RuntimeError(
                f"Failed to fetch a token for environment '{self.name}' from "
                f"{auth_url}. Check network access and that "
                f"{row.client_id_var} and {row.client_secret_var} (or the "
                f"{CORTI_CLIENT_ID_VAR}/{CORTI_CLIENT_SECRET_VAR} fallback) "
                "hold valid credentials."
            ) from exc
        token = response.json().get("access_token")
        if not token:
            raise ValueError(
                f"Token response from {auth_url} did not include "
                f"'access_token'. Check that the configured client id and "
                "secret hold valid credentials."
            )
        return token
