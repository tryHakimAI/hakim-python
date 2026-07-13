"""Tests for `webhooks` namespace + `verify_webhook_signature`."""

from __future__ import annotations

import hashlib
import hmac

import httpx

from hakim import verify_webhook_signature

from ._helpers import json_response

_WH = {
    "id": "wh_1",
    "url": "https://example.com/hook",
    "events": ["job.completed"],
    "active": True,
    "created_at": "1970-01-01T00:00:00Z",
}


def _make_sig_header(secret: str, body: str, ts: int) -> str:
    mac = hmac.new(secret.encode(), f"{ts}.{body}".encode(), hashlib.sha256).hexdigest()
    return f"t={ts},v1={mac}"


# ---------------------------------------------------------------------------
# CRUD + deliveries
# ---------------------------------------------------------------------------


async def test_webhooks_create_returns_secret(build_mock_client):  # type: ignore[no-untyped-def]
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/webhooks"
        return json_response(201, {**_WH, "secret": "whsec_" + "a" * 32})

    async with build_mock_client(handler) as hakim:
        created = await hakim.webhooks.create(
            url="https://example.com/hook", events=["job.completed"]
        )
    assert created["secret"].startswith("whsec_")


async def test_webhooks_list_retrieve_update_delete(build_mock_client):  # type: ignore[no-untyped-def]
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.method == "DELETE":
            return json_response(
                200, {"object": "webhook", "id": "wh_1", "deleted": True}
            )
        if request.method == "PATCH":
            return json_response(200, {**_WH, "active": False})
        is_list = request.url.path == "/v1/webhooks"
        return json_response(
            200, {"object": "list", "data": [_WH]} if is_list else _WH
        )

    async with build_mock_client(handler) as hakim:
        lst = await hakim.webhooks.list()
        got = await hakim.webhooks.retrieve("wh_1")
        upd = await hakim.webhooks.update("wh_1", active=False)
        deleted = await hakim.webhooks.delete("wh_1")

    assert len(lst["data"]) == 1
    assert got["id"] == "wh_1"
    assert upd["active"] is False
    assert deleted.deleted is True
    assert calls == [
        ("GET", "/v1/webhooks"),
        ("GET", "/v1/webhooks/wh_1"),
        ("PATCH", "/v1/webhooks/wh_1"),
        ("DELETE", "/v1/webhooks/wh_1"),
    ]


async def test_webhooks_iter_deliveries_walks_pages(build_mock_client):  # type: ignore[no-untyped-def]
    pages = [
        {
            "object": "list",
            "data": [
                {"id": "d1", "webhook_id": "wh_1"},
                {"id": "d2", "webhook_id": "wh_1"},
            ],
            "has_more": True,
            "next_cursor": "cur_1",
        },
        {
            "object": "list",
            "data": [{"id": "d3", "webhook_id": "wh_1"}],
            "has_more": False,
            "next_cursor": None,
        },
    ]
    seen_cursors: list[str | None] = []
    call = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        seen_cursors.append(request.url.params.get("cursor"))
        body = pages[call[0]]
        call[0] += 1
        return json_response(200, body)

    async with build_mock_client(handler) as hakim:
        collected: list[str] = []
        async for d in hakim.webhooks.iter_deliveries("wh_1"):
            collected.append(d["id"])  # type: ignore[typeddict-item]

    assert collected == ["d1", "d2", "d3"]
    assert seen_cursors == [None, "cur_1"]


# ---------------------------------------------------------------------------
# Signature verification
# ---------------------------------------------------------------------------


def test_verify_accepts_well_formed_signature() -> None:
    secret = "whsec_" + "a" * 40
    body = '{"event":"job.completed","id":"evt_1"}'
    ts = 1_700_000_000
    header = _make_sig_header(secret, body, ts)
    result = verify_webhook_signature(secret=secret, body=body, signature=header, now=ts)
    assert result.valid


def test_verify_rejects_signature_mismatch() -> None:
    secret = "whsec_real"
    body = "payload"
    ts = 1_700_000_000
    header = _make_sig_header("whsec_other", body, ts)
    result = verify_webhook_signature(secret=secret, body=body, signature=header, now=ts)
    assert not result.valid
    assert result.reason == "signature_mismatch"


def test_verify_rejects_timestamp_out_of_tolerance() -> None:
    secret = "whsec_real"
    body = "payload"
    signed_at = 1_700_000_000
    header = _make_sig_header(secret, body, signed_at)
    result = verify_webhook_signature(
        secret=secret, body=body, signature=header, now=signed_at + 600
    )
    assert not result.valid
    assert result.reason == "timestamp_out_of_tolerance"


def test_verify_rejects_malformed_header() -> None:
    result = verify_webhook_signature(
        secret="whsec", body="b", signature="nope", now=1_700_000_000
    )
    assert not result.valid
    assert result.reason == "malformed_header"


def test_verify_disables_replay_when_tolerance_zero() -> None:
    secret = "whsec"
    body = "b"
    header = _make_sig_header(secret, body, 1_000_000)
    result = verify_webhook_signature(
        secret=secret,
        body=body,
        signature=header,
        tolerance_seconds=0,
        now=9_999_999,
    )
    assert result.valid


def test_verify_rejects_short_signature_hex() -> None:
    result = verify_webhook_signature(
        secret="whsec",
        body="b",
        signature="t=1700000000,v1=deadbeef",
        now=1_700_000_000,
    )
    assert not result.valid
    assert result.reason == "malformed_header"
