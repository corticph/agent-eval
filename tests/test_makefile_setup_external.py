"""End-to-end tests for the external ``make setup`` path's judge endpoint
derivation.

The external stepper writes the Console block to ``.env``, then derives
``JUDGE_BASE_URL`` from ``CORTI_ENVIRONMENT`` so that US clients hit
``ai.us.corti.app`` and EU clients (the default) hit ``ai.eu.corti.app``.

These tests drive the ``_setup-external`` Makefile target over piped stdin
so the shell loop is exercised exactly as a customer would hit it. The
target is called directly (not via ``setup``) so the ``install``
dependency never runs.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _console_block(
    *,
    tenant: str = "base",
    client_id: str = "corti-acme-default_client",
    client_secret: str = "s3cret",
    environment: str | None = None,
) -> str:
    """Render a Console block terminated by the empty line that ends a paste.

    The empty line breaks the paste loop; ``N`` declines the Opik step.
    """
    lines = [
        f"CORTI_TENANT_NAME={tenant}",
        f"CORTI_CLIENT_ID={client_id}",
        f"CORTI_CLIENT_SECRET={client_secret}",
    ]
    if environment is not None:
        lines.append(f"CORTI_ENVIRONMENT={environment}")
    return "\n".join(lines) + "\n\nN\n"


def _run_setup_external(env_file: Path, stdin: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["make", "_setup-external", f"ENV_FILE={env_file}"],
        input=stdin,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def _env_dict(env_file: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, _, value = line.partition("=")
            parsed[key] = value
    return parsed


class TestJudgeEndpointFromEnvironment:
    def test_us_environment_sets_us_judge_url(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        result = _run_setup_external(env_file, _console_block(environment="us"))
        assert result.returncode == 0, result.stdout
        assert "Setup complete" in result.stdout
        written = _env_dict(env_file)
        assert written["JUDGE_BASE_URL"] == "https://ai.us.corti.app/v1"
        assert written["CORTI_ENVIRONMENT"] == "us"

    def test_eu_environment_sets_eu_judge_url(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        result = _run_setup_external(env_file, _console_block(environment="eu"))
        assert result.returncode == 0, result.stdout
        written = _env_dict(env_file)
        assert written["JUDGE_BASE_URL"] == "https://ai.eu.corti.app/v1"
        assert written["CORTI_ENVIRONMENT"] == "eu"

    def test_missing_environment_defaults_to_eu(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        result = _run_setup_external(env_file, _console_block())
        assert result.returncode == 0, result.stdout
        written = _env_dict(env_file)
        assert written["JUDGE_BASE_URL"] == "https://ai.eu.corti.app/v1"

    def test_empty_environment_defaults_to_eu(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        result = _run_setup_external(env_file, _console_block(environment=""))
        assert result.returncode == 0, result.stdout
        written = _env_dict(env_file)
        assert written["JUDGE_BASE_URL"] == "https://ai.eu.corti.app/v1"

    def test_existing_judge_url_is_overwritten(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("JUDGE_BASE_URL=https://stale.example/v1\n")
        result = _run_setup_external(env_file, _console_block(environment="us"))
        assert result.returncode == 0, result.stdout
        written = _env_dict(env_file)
        assert written["JUDGE_BASE_URL"] == "https://ai.us.corti.app/v1"
        assert env_file.read_text().count("JUDGE_BASE_URL=") == 1

    def test_run_hint_uses_environment(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        result = _run_setup_external(env_file, _console_block(environment="us"))
        assert result.returncode == 0, result.stdout
        assert "--env us" in result.stdout

    def test_run_hint_defaults_to_eu(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        result = _run_setup_external(env_file, _console_block())
        assert result.returncode == 0, result.stdout
        assert "--env eu" in result.stdout

    def test_judge_endpoint_message_printed(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        result = _run_setup_external(env_file, _console_block(environment="us"))
        assert result.returncode == 0, result.stdout
        assert "judge endpoint" in result.stdout
        assert "ai.us.corti.app" in result.stdout
        assert "CORTI_ENVIRONMENT=us" in result.stdout


class TestMissingCredentialsFail:
    def test_missing_client_id_is_a_hard_failure(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        block = "CORTI_TENANT_NAME=base\nCORTI_CLIENT_SECRET=s3cret\n\n\nN\n"
        result = _run_setup_external(env_file, block)
        assert result.returncode != 0
        assert "CORTI_CLIENT_ID" in result.stdout

    def test_missing_client_secret_is_a_hard_failure(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        block = "CORTI_TENANT_NAME=base\nCORTI_CLIENT_ID=id\n\n\nN\n"
        result = _run_setup_external(env_file, block)
        assert result.returncode != 0
        assert "CORTI_CLIENT_SECRET" in result.stdout
