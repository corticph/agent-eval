from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    """Harness Failure codes: the failures no expectation produced.

    A failed check never gets a code — its kind is already carried by its
    Expectation Result's key.
    """

    # runner flow control
    STEP_EXCEPTION = "step_exception"
    EVAL_TIMEOUT = "eval_timeout"

    # transport / client
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    HTTP_ERROR = "http_error"
    INVALID_RESPONSE = "invalid_response"


class AgentRequestError(RuntimeError):
    code: ErrorCode = ErrorCode.STEP_EXCEPTION


class RequestTimeoutError(AgentRequestError):
    code = ErrorCode.TIMEOUT


class NetworkError(AgentRequestError):
    code = ErrorCode.NETWORK_ERROR


class HttpStatusError(AgentRequestError):
    code = ErrorCode.HTTP_ERROR


class InvalidResponseError(AgentRequestError):
    code = ErrorCode.INVALID_RESPONSE


@dataclass(frozen=True, slots=True)
class EvalError:
    code: ErrorCode
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_exception(cls, exc: BaseException) -> "EvalError":
        """Build an :class:`EvalError` from a caught exception.

        Reads a ``code`` attribute if the exception carries one (see
        :class:`AgentRequestError`), otherwise falls back to
        :attr:`ErrorCode.STEP_EXCEPTION` for genuinely unexpected failures.

        Only an :class:`ErrorCode` is trusted: foreign exceptions (e.g. an
        ``openai.AuthenticationError``) may carry their own unrelated ``code``
        attribute — often ``None`` — which must not leak into our error model.
        """
        code = getattr(exc, "code", None)
        if not isinstance(code, ErrorCode):
            code = ErrorCode.STEP_EXCEPTION
        return cls(code, str(exc), {"exception_type": type(exc).__name__})

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value if isinstance(self.code, ErrorCode) else self.code,
            "message": self.message,
            "context": self.context,
        }
