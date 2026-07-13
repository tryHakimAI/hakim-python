"""SDK error hierarchy.

Mirrors ``@hakim/voice`` (Node SDK) — every HTTP-level failure raises a
:class:`HakimError` subclass so callers can branch on either the base
class or a specific subtype::

    from hakim import Hakim
    from hakim import RateLimitError, HakimError

    hakim = Hakim(api_key=...)
    try:
        await hakim.audio.speech.create(...)
    except RateLimitError as err:
        await asyncio.sleep(err.retry_after_ms / 1000 if err.retry_after_ms else 1.0)
    except HakimError as err:
        print(err.request_id, err.code, err.message)

Transport-level failures (DNS, connection reset, timeouts) are wrapped
in :class:`ConnectionError`.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

HakimErrorType = Literal[
    "invalid_request_error",
    "authentication_error",
    "permission_error",
    "not_found",
    "rate_limit_error",
    "quota_exceeded",
    "api_error",
    "service_unavailable",
    "idempotency_conflict",
    "connection_error",
]


class HakimApiErrorPayload(TypedDict, total=False):
    type: HakimErrorType
    code: str
    message: str
    param: str
    request_id: str


class HakimError(Exception):
    """Base class for every non-transient SDK failure."""

    type: HakimErrorType
    code: str
    status: int
    request_id: str | None
    param: str | None
    retry_after_ms: int | None

    def __init__(
        self,
        message: str,
        *,
        type: HakimErrorType,
        code: str,
        status: int,
        request_id: str | None = None,
        param: str | None = None,
        retry_after_ms: int | None = None,
    ) -> None:
        super().__init__(message)
        self.type = type
        self.code = code
        self.status = status
        self.request_id = request_id
        self.param = param
        self.retry_after_ms = retry_after_ms

    @property
    def message(self) -> str:
        return str(self)


class InvalidRequestError(HakimError):
    pass


class AuthenticationError(HakimError):
    pass


class PermissionError(HakimError):
    pass


class NotFoundError(HakimError):
    pass


class QuotaExceededError(HakimError):
    pass


class RateLimitError(HakimError):
    pass


class IdempotencyConflictError(HakimError):
    pass


class ServiceUnavailableError(HakimError):
    pass


class ConnectionError(HakimError):
    """Transport-level failure (DNS / connect / timeout / abort)."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "connection_failed",
        request_id: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            message,
            type="connection_error",
            code=code,
            status=0,
            request_id=request_id,
        )
        if cause is not None:
            self.__cause__ = cause


# An `APIError` alias keeps back-compat with the 0.1.0 scaffold.
APIError = HakimError


_ERROR_CLASSES: dict[HakimErrorType, type[HakimError]] = {
    "invalid_request_error": InvalidRequestError,
    "authentication_error": AuthenticationError,
    "permission_error": PermissionError,
    "not_found": NotFoundError,
    "quota_exceeded": QuotaExceededError,
    "rate_limit_error": RateLimitError,
    "idempotency_conflict": IdempotencyConflictError,
    "service_unavailable": ServiceUnavailableError,
    "api_error": HakimError,
    "connection_error": ConnectionError,
}


def error_from_payload(
    payload: HakimApiErrorPayload,
    *,
    status: int,
    request_id: str | None,
    retry_after_ms: int | None = None,
) -> HakimError:
    """Build the right subclass from a server ``ApiError`` body."""

    err_type = payload.get("type", "api_error")
    cls = _ERROR_CLASSES.get(err_type, HakimError)
    init_kwargs: dict[str, Any] = {
        "type": err_type,
        "code": payload.get("code", f"http_{status}"),
        "status": status,
        "request_id": payload.get("request_id") or request_id,
        "retry_after_ms": retry_after_ms,
    }
    if "param" in payload:
        init_kwargs["param"] = payload["param"]
    # ``ConnectionError`` uses a trimmed signature; easier to carve out.
    if cls is ConnectionError:
        return ConnectionError(
            payload.get("message", "connection error"),
            code=init_kwargs["code"],
            request_id=init_kwargs["request_id"],
        )
    return cls(payload.get("message", f"HTTP {status}"), **init_kwargs)
