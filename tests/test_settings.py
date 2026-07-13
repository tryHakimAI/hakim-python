"""Tests for ``hakim.settings.*`` and ``hakim.notifications.*`` (M2 follow-up).

Uses ``httpx.MockTransport`` so we can assert on every outgoing request
(method / path / body / headers) without touching a real server.

Coverage:
    - GET /v1/settings/profile — URL + method + bearer header.
    - PATCH /v1/settings/profile — JSON body round-trip.
    - GET + PATCH /v1/settings/organization.
    - GET + PATCH /v1/notifications.
"""

from __future__ import annotations

import json

import httpx

from ._helpers import json_response


async def test_settings_get_profile(build_mock_client):  # type: ignore[no-untyped-def]
    body = {
        "id": "user_1",
        "email": "ada@example.com",
        "email_verified": True,
        "name": "Ada Lovelace",
        "locale": "ar",
        "timezone": "Asia/Dubai",
        "avatar_url": None,
        "marketing_opt_in": False,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/settings/profile"
        assert request.headers.get("authorization") == "Bearer hk_test_x"
        return json_response(200, body)

    async with build_mock_client(handler) as hakim:
        profile = await hakim.settings.get_profile()

    assert profile["id"] == "user_1"
    assert profile["locale"] == "ar"


async def test_settings_update_profile_sends_json(build_mock_client):  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PATCH"
        assert request.url.path == "/v1/settings/profile"
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return json_response(
            200,
            {
                "id": "user_1",
                "email": "ada@example.com",
                "email_verified": True,
                "name": "Ada Lovelace",
                "locale": "en",
                "timezone": "Asia/Dubai",
                "avatar_url": None,
                "marketing_opt_in": False,
            },
        )

    async with build_mock_client(handler) as hakim:
        result = await hakim.settings.update_profile({"locale": "en"})

    assert captured["body"] == {"locale": "en"}
    assert result["locale"] == "en"


async def test_settings_get_organization(build_mock_client):  # type: ignore[no-untyped-def]
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/settings/organization"
        return json_response(
            200,
            {
                "id": "org_1",
                "name": "Acme",
                "slug": "acme",
                "billing_email": "bills@acme.example",
                "default_locale": "ar",
                "logo_url": None,
            },
        )

    async with build_mock_client(handler) as hakim:
        org = await hakim.settings.get_organization()

    assert org["slug"] == "acme"


async def test_settings_update_organization_sends_json(build_mock_client):  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return json_response(
            200,
            {
                "id": "org_1",
                "name": "Acme Corp",
                "slug": "acme",
                "billing_email": None,
                "default_locale": "ar",
                "logo_url": None,
            },
        )

    async with build_mock_client(handler) as hakim:
        org = await hakim.settings.update_organization({"name": "Acme Corp"})

    assert captured["body"] == {"name": "Acme Corp"}
    assert org["name"] == "Acme Corp"


async def test_notifications_get(build_mock_client):  # type: ignore[no-untyped-def]
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/notifications"
        return json_response(
            200,
            {
                "job_completions": True,
                "voice_ready": True,
                "billing_alerts": True,
                "product_updates": False,
            },
        )

    async with build_mock_client(handler) as hakim:
        prefs = await hakim.notifications.get()

    assert prefs["product_updates"] is False


async def test_notifications_update_partial_patch(build_mock_client):  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return json_response(
            200,
            {
                "job_completions": True,
                "voice_ready": True,
                "billing_alerts": True,
                "product_updates": True,
            },
        )

    async with build_mock_client(handler) as hakim:
        prefs = await hakim.notifications.update({"product_updates": True})

    assert captured["body"] == {"product_updates": True}
    assert prefs["product_updates"] is True
