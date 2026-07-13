"""Tests for `usage` (summary + events)."""

from __future__ import annotations

import httpx

from ._helpers import json_response


async def test_usage_summary(build_mock_client):  # type: ignore[no-untyped-def]
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/usage/summary"
        return json_response(
            200,
            {
                "period_start": "2026-04-01T00:00:00Z",
                "period_end": "2026-04-30T23:59:59Z",
                "tts_seconds": 42,
                "stt_seconds": 10,
                "voice_clone_seconds": 0,
                "batch_stt_seconds": 0,
                "total_requests": 5,
            },
        )

    async with build_mock_client(handler) as hakim:
        summary = await hakim.usage.summary()
    assert summary["tts_seconds"] == 42
    assert summary["total_requests"] == 5


async def test_usage_events_serializes_from_as_from_keyword(build_mock_client):  # type: ignore[no-untyped-def]
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return json_response(
            200,
            {
                "object": "list",
                "data": [],
                "has_more": False,
                "next_cursor": None,
            },
        )

    async with build_mock_client(handler) as hakim:
        await hakim.usage.events(
            kind="tts",
            from_="2026-04-01T00:00:00Z",
            to="2026-04-30T23:59:59Z",
            limit=50,
        )

    assert captured == {
        "kind": "tts",
        "from": "2026-04-01T00:00:00Z",
        "to": "2026-04-30T23:59:59Z",
        "limit": "50",
    }
