"""Async HTTP transport built on :mod:`httpx`.

Contract:

* Uses ``httpx.AsyncClient`` — no per-request connection churn.
* Every call attaches ``Authorization`` + ``User-Agent`` + ``X-Request-Id``.
* Mutating calls with a JSON body auto-generate an ``Idempotency-Key``
  unless the caller supplied one.
* Retries on transient failures: 5xx (except 501/505), 429, 408, 425,
  and low-level connection errors. Back-off is exponential with ±25%
  jitter, capped at 10 s; honours ``Retry-After`` when present.
* 4xx (except 408 / 425 / 429) is never retried — caller error.
* JSON / multipart / raw / streaming bodies supported.
* Non-2xx raises a :class:`HakimError` subclass built from the uniform
  ``ApiError`` body. Network-level failures raise :class:`ConnectionError`.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import random as _random
import sys
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from email.utils import parsedate_to_datetime
from typing import Any, cast

import httpx

from ._version import SDK_NAME, SDK_VERSION
from .errors import (
    ConnectionError,
    HakimApiErrorPayload,
    HakimError,
    error_from_payload,
)

_MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

# Retry-eligible statuses (408 RequestTimeout, 425 TooEarly, 429 TooMany,
# 5xx minus 501/505).
_RETRY_STATUSES = frozenset({408, 425, 429})


class AsyncTransport:
    """Low-level async HTTP helper. Shared by every namespace."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout_s: float,
        max_retries: int,
        user_agent_suffix: str | None = None,
        client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        random: Callable[[], float] | None = None,
        generate_idempotency_key: Callable[[], str] | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("hakim SDK: api_key is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._user_agent = _build_user_agent(user_agent_suffix)
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._owns_client = client is None
        self._sleep: Callable[[float], Awaitable[None]] = sleep or asyncio.sleep
        self._random: Callable[[], float] = random or _random.random
        self._generate_idempotency_key: Callable[[], str] = (
            generate_idempotency_key or _default_idempotency_key
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> AsyncTransport:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        files: list[tuple[str, Any]] | None = None,
        data: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        idempotency_key: str | None = None,
        timeout_s: float | None = None,
        stream: bool = False,
        accept: str | None = None,
    ) -> httpx.Response:
        """Execute ``method path`` with retries + idempotency + UA.

        The caller is responsible for closing a streamed response via
        ``async with resp`` or ``await resp.aclose()``; for non-stream
        calls the transport reads and returns the buffered response.
        """

        url = f"{self.base_url}/{path.lstrip('/')}"
        final_headers = self._build_headers(
            method=method,
            extra=dict(headers or {}),
            has_json=json is not None,
            has_multipart=files is not None or data is not None,
            idempotency_key=idempotency_key,
            accept=accept,
        )
        actual_timeout = timeout_s if timeout_s is not None else self._timeout_s

        attempt = 0
        last_exc: BaseException | None = None
        while attempt <= self._max_retries:
            try:
                build_kwargs: dict[str, Any] = {
                    "method": method,
                    "url": url,
                    "headers": final_headers,
                    "params": _clean_params(params),
                    "timeout": actual_timeout,
                }
                if json is not None:
                    build_kwargs["json"] = json
                if files is not None:
                    build_kwargs["files"] = files
                if data is not None:
                    build_kwargs["data"] = dict(data)
                request = self._client.build_request(**build_kwargs)
                response = await self._client.send(request, stream=stream)
            except httpx.TimeoutException as err:
                last_exc = err
                code = "request_timeout"
                if attempt < self._max_retries:
                    await self._sleep(self._backoff_seconds(attempt))
                    attempt += 1
                    continue
                raise ConnectionError(
                    f"hakim SDK: request timed out after {actual_timeout}s",
                    code=code,
                    request_id=final_headers.get("x-request-id"),
                    cause=err,
                ) from err
            except httpx.HTTPError as err:
                last_exc = err
                if attempt < self._max_retries:
                    await self._sleep(self._backoff_seconds(attempt))
                    attempt += 1
                    continue
                raise ConnectionError(
                    f"hakim SDK: network error: {err}",
                    code="network_error",
                    request_id=final_headers.get("x-request-id"),
                    cause=err,
                ) from err

            if response.is_success:
                return response

            if _is_retryable_status(response.status_code) and attempt < self._max_retries:
                retry_after = _read_retry_after_seconds(response.headers)
                wait_s = (
                    retry_after
                    if retry_after is not None
                    else self._backoff_seconds(attempt)
                )
                if stream:
                    await response.aclose()
                else:
                    await response.aread()
                    await response.aclose()
                await self._sleep(wait_s)
                attempt += 1
                continue

            # Non-retryable (or retries exhausted) — raise as HakimError.
            raise await _parse_error_response(response)

        # If we fall out of the loop, retries were exhausted on transient.
        raise ConnectionError(
            f"hakim SDK: request failed after {self._max_retries} retries",
            request_id=final_headers.get("x-request-id"),
            cause=last_exc,
        )

    def _build_headers(
        self,
        *,
        method: str,
        extra: dict[str, str],
        has_json: bool,
        has_multipart: bool,
        idempotency_key: str | None,
        accept: str | None,
    ) -> dict[str, str]:
        del has_multipart  # for symmetry with Node SDK (multipart uses httpx defaults)
        headers: dict[str, str] = {
            "authorization": f"Bearer {self.api_key}",
            "user-agent": self._user_agent,
        }
        for k, v in extra.items():
            headers[k.lower()] = v
        headers.setdefault("x-request-id", f"sdk-{_default_idempotency_key()}")
        headers.setdefault("accept", accept or "application/json")

        if method in _MUTATING_METHODS and has_json:
            headers["idempotency-key"] = (
                idempotency_key or self._generate_idempotency_key()
            )
        elif idempotency_key is not None:
            headers["idempotency-key"] = idempotency_key
        return headers

    def _backoff_seconds(self, attempt: int) -> float:
        """200 ms, 600 ms, 1800 ms, ..., capped at 10 s; ±25% jitter."""
        base_ms: float = float(min(200 * (3**attempt), 10_000))
        jitter: float = base_ms * (self._random() - 0.5) * 0.5
        total_ms: float = max(0.0, base_ms + jitter)
        return total_ms / 1000.0


def _build_user_agent(suffix: str | None) -> str:
    py = f"python/{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    plat = sys.platform
    base = f"{SDK_NAME}/{SDK_VERSION} ({py}; {plat})"
    return f"{base} {suffix}" if suffix else base


def _default_idempotency_key() -> str:
    return str(uuid.uuid4())


def _is_retryable_status(status: int) -> bool:
    if status in _RETRY_STATUSES:
        return True
    if status in (501, 505):
        return False
    return 500 <= status <= 599


def _read_retry_after_seconds(headers: httpx.Headers) -> float | None:
    raw = headers.get("retry-after")
    if not raw:
        return None
    try:
        n = float(raw)
        if n >= 0:
            return n
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(raw)
        if dt is None:
            return None
        delta = dt.timestamp() - time.time()
        return max(0.0, delta)
    except (TypeError, ValueError):
        return None


async def _parse_error_response(response: httpx.Response) -> HakimError:
    request_id = response.headers.get("x-request-id")
    retry_after = _read_retry_after_seconds(response.headers)
    retry_after_ms = int(retry_after * 1000) if retry_after is not None else None

    content_type = response.headers.get("content-type", "")
    body_bytes: bytes
    try:
        if response.is_stream_consumed:
            body_bytes = response.content
        else:
            body_bytes = await response.aread()
    except Exception:
        body_bytes = b""
    finally:
        with contextlib.suppress(Exception):
            await response.aclose()

    payload: HakimApiErrorPayload | None = None
    try:
        text_body = body_bytes.decode("utf-8", errors="replace")
    except Exception:
        text_body = ""
    if "application/json" in content_type and text_body:
        try:
            import json as _json

            parsed = _json.loads(text_body)
            if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
                payload = cast(HakimApiErrorPayload, parsed["error"])
        except Exception:
            payload = None

    if payload is None:
        payload = cast(
            HakimApiErrorPayload,
            {
                "type": "api_error"
                if response.status_code >= 500
                else "invalid_request_error",
                "code": f"http_{response.status_code}",
                "message": text_body or response.reason_phrase
                or f"HTTP {response.status_code}",
            },
        )

    return error_from_payload(
        payload,
        status=response.status_code,
        request_id=request_id,
        retry_after_ms=retry_after_ms,
    )


def _clean_params(params: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not params:
        return None
    out: dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, bool):
            out[key] = "true" if value else "false"
        else:
            out[key] = value
    return out


# Re-export so tests can import a stable public surface. Not part of
# the caller-facing API; prefixed with `_` in __all__ to hide.
__all__ = ["AsyncTransport"]

# Pull env helper out for the client module.
def env(name: str) -> str | None:
    return os.environ.get(name)
