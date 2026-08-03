"""Tests for the Opik target resolver (``opik_target``).

These isolate the process/network boundaries (``subprocess.Popen``, the
readiness ping, time) to assert the observable contract: the URL a caller
gets back, and whether a tunnel process was launched or torn down.
"""

from __future__ import annotations

import io
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Callable

import pytest

from agent_evals.reporting import opik_target


def _fake_ping(
    monkeypatch: pytest.MonkeyPatch, *, alive: Callable[[], bool]
) -> list[str]:
    """Substitute the readiness probe's HTTP boundary.

    ``alive`` decides whether the probe succeeds; returns the URLs probed.
    """
    probed: list[str] = []

    @contextmanager
    def fake_urlopen(url: str, timeout: float = 1.0):
        probed.append(url)
        if not alive():
            raise OSError("connection refused")
        yield SimpleNamespace(status=200)

    monkeypatch.setattr(opik_target.urllib.request, "urlopen", fake_urlopen)
    return probed


@pytest.fixture(autouse=True)
def _pristine_module(monkeypatch: pytest.MonkeyPatch):
    """Each test starts with no cached tunnel and no ambient env config."""
    monkeypatch.setattr(opik_target, "_tunnel_proc", None)
    for var in ("OPIK_URL_OVERRIDE", "OPIK_TUNNEL_PORT", "OPIK_TUNNEL_CONTEXT"):
        monkeypatch.delenv(var, raising=False)


def _forbid_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(*args, **kwargs):
        raise AssertionError("a tunnel subprocess was spawned")

    monkeypatch.setattr(opik_target.subprocess, "Popen", _fail)


def test_url_override_wins_verbatim_and_spawns_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_spawn(monkeypatch)
    monkeypatch.setenv("OPIK_URL_OVERRIDE", "http://elsewhere:9999/api")
    probed = _fake_ping(monkeypatch, alive=lambda: True)

    assert opik_target.resolve_opik_url() == "http://elsewhere:9999/api"
    assert probed == ["http://elsewhere:9999/api/is-alive/ping"]


def test_dead_override_fails_fast_instead_of_writing_into_the_void(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A stale override — e.g. a local Opik from an old workflow that is no
    # longer running — must abort the run, not swallow every write.
    _forbid_spawn(monkeypatch)
    monkeypatch.setenv("OPIK_URL_OVERRIDE", "http://localhost:5173/api/")
    _fake_ping(monkeypatch, alive=lambda: False)

    with pytest.raises(RuntimeError) as excinfo:
        opik_target.resolve_opik_url()

    message = str(excinfo.value)
    assert "OPIK_URL_OVERRIDE" in message
    assert "http://localhost:5173/api/" in message
    assert "Unset" in message


def test_already_answering_port_is_reused_without_spawning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_spawn(monkeypatch)
    monkeypatch.setenv("OPIK_TUNNEL_PORT", "19999")
    probed = _fake_ping(monkeypatch, alive=lambda: True)

    assert opik_target.resolve_opik_url() == "http://localhost:19999"
    assert probed == ["http://localhost:19999/is-alive/ping"]


class _FakeTunnelProc:
    """Stand-in for the kubectl port-forward process."""

    def __init__(self, *, exits_with: int | None = None, stderr: bytes = b""):
        self._returncode = exits_with
        self.stderr = io.BytesIO(stderr)
        self.terminated = 0

    def poll(self) -> int | None:
        return self._returncode

    def terminate(self) -> None:
        self.terminated += 1
        self._returncode = -15


def _spawn_returning(
    monkeypatch: pytest.MonkeyPatch, proc: _FakeTunnelProc
) -> list[list[str]]:
    """Substitute the process boundary; returns the spawned commands."""
    spawned: list[list[str]] = []

    def fake_popen(cmd, **kwargs):
        spawned.append(cmd)
        return proc

    monkeypatch.setattr(opik_target.subprocess, "Popen", fake_popen)
    return spawned


def test_spawned_tunnel_that_becomes_ready_is_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPIK_TUNNEL_CONTEXT", "custom-ctx")
    spawned = _spawn_returning(monkeypatch, _FakeTunnelProc())
    _fake_ping(monkeypatch, alive=lambda: bool(spawned))

    assert opik_target.resolve_opik_url() == "http://localhost:18080"

    (cmd,) = spawned
    assert cmd == [
        "kubectl",
        "--context",
        "custom-ctx",
        "port-forward",
        "-n",
        "mlservices",
        "svc/opik-backend",
        "18080:8080",
    ]


def _run_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Advance the module's clock one second per query; never really sleep."""
    now = iter(range(10_000))
    monkeypatch.setattr(opik_target.time, "monotonic", lambda: float(next(now)))
    monkeypatch.setattr(opik_target.time, "sleep", lambda s: None)


def test_readiness_timeout_stops_the_tunnel_and_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_clock(monkeypatch)
    proc = _FakeTunnelProc()  # stays alive, but the port never answers
    _spawn_returning(monkeypatch, proc)
    _fake_ping(monkeypatch, alive=lambda: False)

    with pytest.raises(RuntimeError, match="timed out"):
        opik_target.ensure_tunnel(18080)

    assert proc.terminated


def test_spawned_tunnel_is_terminated_at_interpreter_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered: list[Callable[[], None]] = []
    monkeypatch.setattr(opik_target.atexit, "register", registered.append)
    proc = _FakeTunnelProc()
    spawned = _spawn_returning(monkeypatch, proc)
    _fake_ping(monkeypatch, alive=lambda: bool(spawned))

    opik_target.ensure_tunnel(18080)

    (handler,) = registered
    handler()
    handler()  # interpreter exit must survive a double fire
    assert proc.terminated == 1


def test_kubectl_exiting_early_surfaces_its_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_ping(monkeypatch, alive=lambda: False)
    proc = _FakeTunnelProc(
        exits_with=1, stderr=b'error: context "dev-weu" does not exist'
    )
    _spawn_returning(monkeypatch, proc)

    with pytest.raises(RuntimeError, match='context "dev-weu" does not exist'):
        opik_target.ensure_tunnel(18080)


def test_missing_kubectl_names_the_prerequisites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_ping(monkeypatch, alive=lambda: False)

    def no_binary(*args, **kwargs):
        raise FileNotFoundError("kubectl")

    monkeypatch.setattr(opik_target.subprocess, "Popen", no_binary)

    with pytest.raises(RuntimeError) as excinfo:
        opik_target.ensure_tunnel(18080)

    message = str(excinfo.value)
    for needle in (
        "kubectl",
        "az login",
        "dev-weu",
        "port-forward",
        "OPIK_URL_OVERRIDE",
    ):
        assert needle in message
