"""Tests for the Judge call mechanics (Judge._judge).

The ``openai`` SDK is the system boundary: each test swaps the module object
the judge calls through for a fake that records what reaches Corti Models and
replies with a canned verdict (or raises).
"""

from __future__ import annotations

import base64
import json
import types
from typing import Any

import pytest

from agent_evals.expectations.llm import base as judge_module
from agent_evals.expectations.llm.base import Judge


class _FakeOpenAI:
    """A stand-in ``openai`` module: canned reply, recorded boundary traffic."""

    def __init__(
        self,
        reply: str | None = None,
        *,
        raise_connection_error: bool = False,
        exc: Exception | None = None,
    ) -> None:
        self.reply = reply
        self.raise_connection_error = raise_connection_error
        self.exc = exc
        self.client_kwargs: dict[str, Any] | None = None
        self.create_kwargs: dict[str, Any] | None = None

        class OpenAIError(Exception):
            pass

        class APIConnectionError(OpenAIError):
            pass

        self.OpenAIError = OpenAIError
        self.APIConnectionError = APIConnectionError

        fake = self

        class OpenAI:
            def __init__(self, **kwargs: Any) -> None:
                fake.client_kwargs = kwargs
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(create=self._create)
                )

            def _create(self, **kwargs: Any) -> Any:
                fake.create_kwargs = kwargs
                if fake.raise_connection_error:
                    raise APIConnectionError("connection refused")
                if fake.exc is not None:
                    raise fake.exc
                message = types.SimpleNamespace(content=fake.reply)
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        self.OpenAI = OpenAI


def _install(monkeypatch: pytest.MonkeyPatch, fake: _FakeOpenAI) -> None:
    """Swap in *fake* and pin a clean judge env: key set, knobs at defaults.

    The CORTI_* credentials are cleared so a developer's shell cannot leak a
    composite bearer into tests that assert on the configured key.
    """
    monkeypatch.setattr(judge_module, "openai", fake)
    monkeypatch.setenv("JUDGE_API_KEY", "ck-test")
    monkeypatch.delenv("JUDGE_BASE_URL", raising=False)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    for var in ("CORTI_TENANT_NAME", "CORTI_CLIENT_ID", "CORTI_CLIENT_SECRET"):
        monkeypatch.delenv(var, raising=False)


def _verdict(result: str, explanation: str) -> str:
    return json.dumps({"result": result, "explanation": explanation})


def test_pass_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _FakeOpenAI(_verdict("PASS", "same core facts")))

    passed, explanation = Judge._judge("test prompt")

    assert passed is True
    assert explanation == "same core facts"


def test_fail_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _FakeOpenAI(_verdict("FAIL", "contradicts reference")))

    passed, explanation = Judge._judge("test prompt")

    assert passed is False
    assert explanation == "contradicts reference"


def test_env_vars_configure_the_call(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeOpenAI(_verdict("PASS", "ok"))
    _install(monkeypatch, fake)
    monkeypatch.setenv("JUDGE_API_KEY", "ck-customer-key")
    monkeypatch.setenv("JUDGE_BASE_URL", "https://staging.example/v1")
    monkeypatch.setenv("JUDGE_MODEL", "corti-s1")

    Judge._judge("test prompt")

    assert fake.client_kwargs == {
        "base_url": "https://staging.example/v1",
        "api_key": "ck-customer-key",
    }
    assert fake.create_kwargs is not None
    assert fake.create_kwargs["model"] == "corti-s1"


def test_defaults_are_corti_models_eu_and_s1_instant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeOpenAI(_verdict("PASS", "ok"))
    _install(monkeypatch, fake)

    Judge._judge("test prompt")

    assert fake.client_kwargs == {
        "base_url": "https://ai.eu.corti.app/v1",
        "api_key": "ck-test",
    }
    assert fake.create_kwargs is not None
    assert fake.create_kwargs["model"] == "corti-s1-instant"


def test_output_is_capped_with_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Corti Models' published schema takes max_tokens; the OpenAI-flavored
    max_completion_tokens is not part of it and must not be sent."""
    fake = _FakeOpenAI(_verdict("PASS", "ok"))
    _install(monkeypatch, fake)

    Judge._judge("test prompt")

    assert fake.create_kwargs is not None
    assert fake.create_kwargs["max_tokens"] == 1000
    assert "max_completion_tokens" not in fake.create_kwargs


def _composite(tenant: str, client_id: str, client_secret: str) -> str:
    raw = f"{tenant}:client_credentials:{client_id}:{client_secret}"
    return base64.b64encode(raw.encode()).decode()


def test_corti_credentials_assemble_the_bearer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without JUDGE_API_KEY, the Console block becomes the base64 composite
    bearer Corti Models expects."""
    fake = _FakeOpenAI(_verdict("PASS", "ok"))
    _install(monkeypatch, fake)
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)
    monkeypatch.setenv("CORTI_TENANT_NAME", "acme")
    monkeypatch.setenv("CORTI_CLIENT_ID", "corti-acme-default_client")
    monkeypatch.setenv("CORTI_CLIENT_SECRET", "s3cret")

    Judge._judge("test prompt")

    assert fake.client_kwargs is not None
    assert fake.client_kwargs["api_key"] == _composite(
        "acme", "corti-acme-default_client", "s3cret"
    )


def test_explicit_judge_api_key_wins_over_corti_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeOpenAI(_verdict("PASS", "ok"))
    _install(monkeypatch, fake)
    monkeypatch.setenv("CORTI_CLIENT_ID", "corti-acme-default_client")
    monkeypatch.setenv("CORTI_CLIENT_SECRET", "s3cret")

    Judge._judge("test prompt")

    assert fake.client_kwargs is not None
    assert fake.client_kwargs["api_key"] == "ck-test"


def test_tenant_defaults_to_base_in_the_bearer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeOpenAI(_verdict("PASS", "ok"))
    _install(monkeypatch, fake)
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)
    monkeypatch.setenv("CORTI_CLIENT_ID", "corti-acme-default_client")
    monkeypatch.setenv("CORTI_CLIENT_SECRET", "s3cret")

    Judge._judge("test prompt")

    assert fake.client_kwargs is not None
    assert fake.client_kwargs["api_key"] == _composite(
        "base", "corti-acme-default_client", "s3cret"
    )


def test_connection_error_is_a_failed_verdict_carrying_the_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _FakeOpenAI(raise_connection_error=True))

    passed, explanation = Judge._judge("test prompt")

    assert passed is False
    assert "connection refused" in explanation


def test_non_openai_exception_is_a_failed_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _FakeOpenAI(exc=ValueError("bad model config")))

    passed, explanation = Judge._judge("test prompt")

    assert passed is False
    assert "bad model config" in explanation


def test_unparseable_reply_is_a_failed_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _FakeOpenAI("PASS, I guess?"))

    passed, explanation = Judge._judge("test prompt")

    assert passed is False
    assert "PASS, I guess?" in explanation


def test_malformed_verdict_is_a_failed_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _FakeOpenAI(_verdict("MAYBE", "on the fence")))

    passed, explanation = Judge._judge("test prompt")

    assert passed is False
    assert "MAYBE" in explanation


def test_empty_reference_auto_passes_without_calling_the_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeOpenAI()
    _install(monkeypatch, fake)

    result = judge_module.Judge(None).resolve(
        judge_module.EvaluationContext.from_response(None)
    )

    assert result.passed
    assert fake.client_kwargs is None
