"""Tests for EvalError construction from foreign exceptions."""

from __future__ import annotations

from agent_evals.errors import (
    ErrorCode,
    EvalError,
    HttpStatusError,
)


class _ForeignError(Exception):
    """Mimics e.g. openai.AuthenticationError, which carries its own .code."""

    def __init__(self, message: str, code=None) -> None:
        super().__init__(message)
        self.code = code


class TestErrorTaxonomy:
    def test_enum_holds_only_harness_failure_codes(self) -> None:
        # Every per-expectation code is gone: a failed check's kind is carried
        # by its Expectation Result's key, never by an error code.
        assert {code.value for code in ErrorCode} == {
            "eval_timeout",
            "step_exception",
            "timeout",
            "network_error",
            "http_error",
            "invalid_response",
        }


class TestFromException:
    def test_foreign_exception_with_none_code_falls_back(self) -> None:
        err = EvalError.from_exception(_ForeignError("Jwt is expired", code=None))
        assert err.code is ErrorCode.STEP_EXCEPTION
        assert err.to_dict()["code"] == "step_exception"

    def test_foreign_exception_with_string_code_falls_back(self) -> None:
        err = EvalError.from_exception(_ForeignError("bad", code="invalid_api_key"))
        assert err.code is ErrorCode.STEP_EXCEPTION
        assert err.to_dict()["code"] == "step_exception"

    def test_plain_exception_uses_step_exception(self) -> None:
        err = EvalError.from_exception(ValueError("boom"))
        assert err.code is ErrorCode.STEP_EXCEPTION

    def test_agent_request_error_code_preserved(self) -> None:
        err = EvalError.from_exception(HttpStatusError("HTTP 500"))
        assert err.code is ErrorCode.HTTP_ERROR
        assert err.to_dict()["code"] == "http_error"

    def test_to_dict_never_raises_on_bad_code(self) -> None:
        # Defensive: even a hand-built EvalError with a non-enum code serialises.
        assert EvalError(code="raw", message="m").to_dict()["code"] == "raw"
