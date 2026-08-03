"""Tests for the CLI environment-resolution seam.

The Environment is the sole source of connection configuration: the CLI's
only job is to pick one — from ``--env`` — and fail fast when it can't.
The per-knob escape hatches are deleted, so the parser itself must reject
them loudly rather than silently ignoring stale invocations.
"""

from __future__ import annotations

import pytest

from agent_evals.__main__ import _build_parser, resolve_environment
from agent_evals.environment import configuration_knobs

# The per-knob escape hatches deleted by the refactor, for both subcommands.
# The old mode-selection flags (the purged namespace) are rejected the same
# way: argparse refuses every option the parser no longer declares.
_DELETED_FLAGS = [
    ["--base-url", "http://x"],
    ["--token", "tok"],
    ["--tenant-name", "acme"],
]


@pytest.mark.parametrize("command", ["run"])
@pytest.mark.parametrize("flag", _DELETED_FLAGS, ids=lambda f: f[0])
def test_deleted_flags_are_rejected(command: str, flag: list[str]) -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args([command, "suite.yaml", *flag])
    assert excinfo.value.code != 0


@pytest.mark.parametrize("command", ["run"])
def test_env_flag_parses(command: str) -> None:
    args = _build_parser().parse_args([command, "suite.yaml", "--env", "dev-weu"])
    assert args.env == "dev-weu"


class TestConfigurationInventory:
    """The code is the inventory: every surviving knob is declared once in
    the configuration module, and ``--help`` renders from those declarations,
    so help and behaviour cannot drift."""

    def test_inventory_declares_the_surviving_knobs(self) -> None:
        declared = {var for knob in configuration_knobs() for var in knob.env_vars}
        assert {
            "AGENT_API_TOKEN_LOCAL",
            "AGENT_API_CLIENT_ID_DEV_WEU",
            "AGENT_API_CLIENT_SECRET_DEV_WEU",
            "OPIK_URL_OVERRIDE",
            "CORTI_TENANT_NAME",
            "CORTI_CLIENT_ID",
            "CORTI_CLIENT_SECRET",
            "JUDGE_API_KEY",
            "JUDGE_BASE_URL",
            "JUDGE_MODEL",
        } <= declared

    def test_judge_knobs_carry_the_judge_modules_defaults(self) -> None:
        """The CORTI_* credentials render as required (no default); the
        override key states it falls back to them, and the endpoint and model
        knobs state the same defaults the judge actually falls back to."""
        from agent_evals.expectations.llm.base import (
            DEFAULT_JUDGE_BASE_URL,
            DEFAULT_JUDGE_MODEL,
        )

        by_var = {var: knob for knob in configuration_knobs() for var in knob.env_vars}
        assert by_var["CORTI_CLIENT_ID"].default is None
        assert "CORTI_*" in (by_var["JUDGE_API_KEY"].default or "")
        assert by_var["JUDGE_BASE_URL"].default == DEFAULT_JUDGE_BASE_URL
        assert by_var["JUDGE_MODEL"].default == DEFAULT_JUDGE_MODEL

    def test_every_knob_states_its_effect(self) -> None:
        for knob in configuration_knobs():
            assert knob.name
            assert knob.effect

    def test_top_level_help_renders_every_declared_env_var(self) -> None:
        help_text = _build_parser().format_help()
        for knob in configuration_knobs():
            for var in knob.env_vars:
                assert var in help_text

    @pytest.mark.parametrize("command", ["run"])
    def test_subcommand_help_renders_every_declared_env_var(
        self, command: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit):
            _build_parser().parse_args([command, "--help"])
        help_text = capsys.readouterr().out
        for knob in configuration_knobs():
            for var in knob.env_vars:
                assert var in help_text


class TestResolveEnvironment:
    def test_missing_environment_is_an_actionable_error(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            resolve_environment(None)
        assert excinfo.value.code != 0
        message = str(excinfo.value)
        assert "--env" in message

    def test_unknown_environment_lists_the_supported_set(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            resolve_environment("prod")
        assert excinfo.value.code != 0
        message = str(excinfo.value)
        for name in ("local", "dev-weu", "staging-eu", "eu", "us"):
            assert name in message

    def test_supported_set_follows_the_configured_environments(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An external checkout is not configured for the internal environments,
        # so the CLI's offered roster names only what it can reach.
        for name in ("DEV_WEU", "STAGING_EU"):
            monkeypatch.delenv(f"AGENT_API_URL_{name}", raising=False)
            monkeypatch.delenv(f"AGENT_API_AUTH_URL_{name}", raising=False)
        with pytest.raises(SystemExit) as excinfo:
            resolve_environment("dev-weu")
        message = str(excinfo.value)
        assert "dev-weu" in message  # the rejected name
        for offered in ("local", "eu", "us"):
            assert offered in message
        assert "staging-eu" not in message
