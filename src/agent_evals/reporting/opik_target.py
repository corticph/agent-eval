"""Resolve the Opik backend URL, tunneling to dev-weu when needed."""

from __future__ import annotations

import atexit
import os
import subprocess
import time
import urllib.request

from ..environment import OPIK_URL_OVERRIDE_VAR

DEFAULT_TUNNEL_PORT = 18080  # not 8080: agent-api's compose tunnel owns that
DEFAULT_TUNNEL_CONTEXT = "dev-weu"
TUNNEL_NAMESPACE = "mlservices"
TUNNEL_SERVICE = "svc/opik-backend"
TUNNEL_REMOTE_PORT = 8080
_READY_TIMEOUT_S = 15.0

_tunnel_proc: subprocess.Popen | None = None


def resolve_opik_url() -> str:
    """Return the Opik base URL evals should write to.

    ``OPIK_URL_OVERRIDE`` wins (no tunnel management), but only after a
    readiness ping: a stale override — e.g. a local Opik from an old workflow
    that is no longer running — would otherwise silently swallow the whole
    run. Otherwise ensure a kubectl port-forward to the dev-weu opik-backend
    and return its localhost URL — no ``/api`` suffix, the backend serves
    ``/v1/private/...`` at the root.
    """
    override = os.environ.get(OPIK_URL_OVERRIDE_VAR)
    if override:
        if not _ping_url(override, timeout=3.0):
            raise RuntimeError(
                f"{OPIK_URL_OVERRIDE_VAR} is set to {override!r} but no Opik "
                f"answers there ({_ping_endpoint(override)} did not respond). "
                f"Unset {OPIK_URL_OVERRIDE_VAR} to target the dev-weu Opik, or "
                f"point it at a running Opik."
            )
        return override
    port = int(os.environ.get("OPIK_TUNNEL_PORT", DEFAULT_TUNNEL_PORT))
    ensure_tunnel(port)
    return f"http://localhost:{port}"


def _ping_endpoint(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/is-alive/ping"


def _ping_url(base_url: str, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(_ping_endpoint(base_url), timeout=timeout) as resp:
            return resp.status == 200
    except OSError:
        return False


def _ping(port: int) -> bool:
    return _ping_url(f"http://localhost:{port}")


def _tunnel_command(context: str, port: int) -> list[str]:
    return [
        "kubectl",
        "--context",
        context,
        "port-forward",
        "-n",
        TUNNEL_NAMESPACE,
        TUNNEL_SERVICE,
        f"{port}:{TUNNEL_REMOTE_PORT}",
    ]


def ensure_tunnel(port: int) -> None:
    """Idempotently ensure something Opik-shaped answers on ``port``.

    Reuses an already-running tunnel (manual or from a previous invocation);
    otherwise spawns ``kubectl port-forward`` and waits for readiness.
    """
    if _ping(port):
        return

    context = os.environ.get("OPIK_TUNNEL_CONTEXT", DEFAULT_TUNNEL_CONTEXT)
    global _tunnel_proc
    try:
        _tunnel_proc = subprocess.Popen(
            _tunnel_command(context, port),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(_prereq_message(context, port)) from exc
    atexit.register(_stop_tunnel)

    deadline = time.monotonic() + _READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if _tunnel_proc.poll() is not None:  # kubectl exited: bad creds/context
            stderr = (_tunnel_proc.stderr.read() or b"").decode(errors="replace")
            _stop_tunnel()
            raise RuntimeError(_prereq_message(context, port, detail=stderr))
        if _ping(port):
            return
        time.sleep(0.3)

    _stop_tunnel()
    raise RuntimeError(
        _prereq_message(context, port, detail="timed out waiting for tunnel readiness")
    )


def _stop_tunnel() -> None:
    global _tunnel_proc
    if _tunnel_proc is not None and _tunnel_proc.poll() is None:
        _tunnel_proc.terminate()
    _tunnel_proc = None


def _prereq_message(context: str, port: int, detail: str = "") -> str:
    manual = " ".join(_tunnel_command(context, port))
    return (
        f"Could not reach dev-weu Opik on localhost:{port}. "
        f"Prerequisites: kubectl, `az login`, and a '{context}' kube context. "
        f"Manual command: {manual}. "
        f"To target a different Opik entirely, set {OPIK_URL_OVERRIDE_VAR}."
        + (f" Details: {detail}" if detail else "")
    )
