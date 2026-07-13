"""Tests for AsyncTransport: headers, UA, idempotency, retries, error mapping."""

from __future__ import annotations

import httpx
import pytest

from hakim import (
    AsyncHakim,
    AuthenticationError,
    ConnectionError,
    HakimError,
    InvalidRequestError,
    RateLimitError,
)
from hakim._transport import AsyncTransport
from hakim._version import SDK_NAME, SDK_VERSION

pytestmark = pytest.mark.asyncio


async def test_default_headers_include_ua_authorization_requestid() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update({k.lower(): v for k, v in request.headers.items()})
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        t = AsyncTransport(
            api_key="hk_test_abc",
            base_url="https://api.example.com",
            timeout_s=1.0,
            max_retries=0,
            client=http,
        )
        await t.request("GET", "v1/usage/summary")

    assert seen["authorization"] == "Bearer hk_test_abc"
    assert seen["user-agent"].startswith(f"{SDK_NAME}/{SDK_VERSION}")
    assert seen["x-request-id"].startswith("sdk-")
    # Accept defaults to application/json on non-stream calls.
    assert "application/json" in seen["accept"]


async def test_post_with_json_attaches_idempotency_key() -> None:
    seen_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.update({k.lower(): v for k, v in request.headers.items()})
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        t = AsyncTransport(
            api_key="hk",
            base_url="https://api.example.com",
            timeout_s=1.0,
            max_retries=0,
            client=http,
            generate_idempotency_key=lambda: "idem-42",
        )
        await t.request("POST", "v1/audio/speech", json={"model": "x"})

    assert seen_headers["idempotency-key"] == "idem-42"


async def test_caller_idempotency_key_overrides_auto() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("idempotency-key", ""))
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        t = AsyncTransport(
            api_key="hk",
            base_url="https://api.example.com",
            timeout_s=1.0,
            max_retries=0,
            client=http,
            generate_idempotency_key=lambda: "auto-should-not-appear",
        )
        await t.request(
            "POST",
            "v1/audio/speech",
            json={"model": "x"},
            idempotency_key="user-key-9",
        )

    assert seen == ["user-key-9"]


async def test_get_does_not_attach_idempotency_key() -> None:
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("idempotency-key"))
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        t = AsyncTransport(
            api_key="hk",
            base_url="https://api.example.com",
            timeout_s=1.0,
            max_retries=0,
            client=http,
        )
        await t.request("GET", "v1/usage/summary")

    assert seen == [None]


async def test_user_agent_suffix_is_appended() -> None:
    seen_ua: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_ua.append(request.headers["user-agent"])
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        t = AsyncTransport(
            api_key="hk",
            base_url="https://api.example.com",
            timeout_s=1.0,
            max_retries=0,
            user_agent_suffix="myapp/1.2.3",
            client=http,
        )
        await t.request("GET", "v1/usage/summary")

    assert seen_ua[0].endswith("myapp/1.2.3")


async def test_retries_on_503_then_succeeds() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(
                503,
                headers={"content-type": "application/json"},
                json={
                    "error": {
                        "type": "service_unavailable",
                        "code": "temporary",
                        "message": "try later",
                    }
                },
            )
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        t = AsyncTransport(
            api_key="hk",
            base_url="https://api.example.com",
            timeout_s=1.0,
            max_retries=3,
            client=http,
            sleep=_no_sleep,
            random=lambda: 0.5,
        )
        await t.request("GET", "v1/usage/summary")

    assert calls == 3


async def test_retries_honour_retry_after_header() -> None:
    slept: list[float] = []
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429,
                headers={"retry-after": "2", "content-type": "application/json"},
                json={"error": {"type": "rate_limit_error", "code": "rl", "message": "slow down"}},
            )
        return httpx.Response(200, json={})

    async def tracking_sleep(secs: float) -> None:
        slept.append(secs)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        t = AsyncTransport(
            api_key="hk",
            base_url="https://api.example.com",
            timeout_s=1.0,
            max_retries=2,
            client=http,
            sleep=tracking_sleep,
            random=lambda: 0.5,
        )
        await t.request("GET", "v1/usage/summary")

    assert slept == [2.0]


async def test_4xx_not_retried_raises_invalid_request() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            400,
            headers={"content-type": "application/json"},
            json={
                "error": {
                    "type": "invalid_request_error",
                    "code": "bad_voice",
                    "message": "voice not found",
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        t = AsyncTransport(
            api_key="hk",
            base_url="https://api.example.com",
            timeout_s=1.0,
            max_retries=3,
            client=http,
            sleep=_no_sleep,
        )
        with pytest.raises(InvalidRequestError) as excinfo:
            await t.request("POST", "v1/audio/speech", json={})
        assert excinfo.value.code == "bad_voice"
        assert excinfo.value.status == 400
    assert calls == 1


async def test_401_raises_authentication_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            headers={"content-type": "application/json"},
            json={
                "error": {
                    "type": "authentication_error",
                    "code": "invalid_api_key",
                    "message": "bad key",
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        t = AsyncTransport(
            api_key="hk",
            base_url="https://api.example.com",
            timeout_s=1.0,
            max_retries=0,
            client=http,
        )
        with pytest.raises(AuthenticationError):
            await t.request("GET", "v1/usage/summary")


async def test_rate_limit_populates_retry_after_ms() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"retry-after": "7", "content-type": "application/json"},
            json={
                "error": {
                    "type": "rate_limit_error",
                    "code": "too_many",
                    "message": "slow down",
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        t = AsyncTransport(
            api_key="hk",
            base_url="https://api.example.com",
            timeout_s=1.0,
            max_retries=0,
            client=http,
        )
        with pytest.raises(RateLimitError) as exc:
            await t.request("GET", "v1/usage/summary")
        assert exc.value.retry_after_ms == 7000


async def test_connection_error_wraps_httpx_errors() -> None:
    async def bad_handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(bad_handler)
    ) as http:
        t = AsyncTransport(
            api_key="hk",
            base_url="https://api.example.com",
            timeout_s=1.0,
            max_retries=0,
            client=http,
        )
        with pytest.raises(ConnectionError):
            await t.request("GET", "v1/usage/summary")


async def test_async_client_requires_api_key() -> None:
    with pytest.raises(ValueError, match="missing API key"):
        AsyncHakim()


async def _no_sleep(_: float) -> None:
    return None


async def test_hakimerror_is_base_class_for_subclasses() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            headers={"content-type": "application/json"},
            json={"error": {"type": "permission_error", "code": "forbidden", "message": "no"}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        t = AsyncTransport(
            api_key="hk",
            base_url="https://api.example.com",
            timeout_s=1.0,
            max_retries=0,
            client=http,
        )
        with pytest.raises(HakimError) as exc:
            await t.request("GET", "v1/usage/summary")
        assert exc.value.status == 403
