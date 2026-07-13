"""Tests for `voices` (list / retrieve / create / delete / iter)."""

from __future__ import annotations

import httpx

from ._helpers import json_response

_SAMPLE_VOICE = {
    "id": "v_1",
    "slug": "omar",
    "name": "Omar",
    "kind": "preset",
    "language": "ar",
    "gender": "male",
    "description": None,
    "preview_url": None,
    "status": "ready",
}


async def test_voices_list_forwards_query_params(build_mock_client):  # type: ignore[no-untyped-def]
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return json_response(
            200,
            {"object": "list", "data": [_SAMPLE_VOICE], "has_more": False, "next_cursor": None},
        )

    async with build_mock_client(handler) as hakim:
        resp = await hakim.voices.list(kind="preset", language="ar", q="om", limit=5)

    assert captured == {"kind": "preset", "language": "ar", "q": "om", "limit": "5"}
    assert resp["data"][0]["id"] == "v_1"


async def test_voices_retrieve_url_encodes_id(build_mock_client):  # type: ignore[no-untyped-def]
    seen_path: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_path.append(request.url.raw_path.decode())
        return json_response(200, _SAMPLE_VOICE)

    async with build_mock_client(handler) as hakim:
        v = await hakim.voices.retrieve("v with slash/?&")

    assert seen_path == ["/v1/audio/voices/v%20with%20slash%2F%3F%26"]
    assert v["id"] == "v_1"


async def test_voices_create_posts_multipart_with_consent(build_mock_client):  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["content_type"] = request.headers["content-type"]
        captured["body"] = request.content
        return json_response(
            202,
            {
                "id": "v_cloned_1",
                "slug": "my-voice",
                "name": "My Voice",
                "kind": "cloned",
                "language": "ar",
                "gender": "neutral",
                "description": None,
                "preview_url": None,
                "status": "processing",
            },
        )

    async with build_mock_client(handler) as hakim:
        voice = await hakim.voices.create(
            sample=b"\x00\x01\x02",
            name="My Voice",
            language="ar",
            consent_confirmed=True,
            filename="sample.wav",
        )

    assert captured["method"] == "POST"
    assert captured["path"] == "/v1/audio/voices"
    assert "multipart/form-data" in str(captured["content_type"])
    body_bytes = captured["body"]
    assert isinstance(body_bytes, bytes)
    assert b"My Voice" in body_bytes
    assert b"consent_confirmed" in body_bytes
    assert b"sample.wav" in body_bytes
    assert voice["status"] == "processing"
    assert voice["kind"] == "cloned"


async def test_voices_delete(build_mock_client):  # type: ignore[no-untyped-def]
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        return json_response(
            200, {"object": "voice", "id": "v_1", "deleted": True}
        )

    async with build_mock_client(handler) as hakim:
        res = await hakim.voices.delete("v_1")

    assert res["deleted"] is True


async def test_voices_iter_walks_every_page(build_mock_client):  # type: ignore[no-untyped-def]
    pages = [
        {
            "object": "list",
            "data": [{**_SAMPLE_VOICE, "id": "v_a"}, {**_SAMPLE_VOICE, "id": "v_b"}],
            "has_more": True,
            "next_cursor": "cur_1",
        },
        {
            "object": "list",
            "data": [{**_SAMPLE_VOICE, "id": "v_c"}],
            "has_more": False,
            "next_cursor": None,
        },
    ]
    call = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        body = pages[call[0]]
        call[0] += 1
        return json_response(200, body)

    async with build_mock_client(handler) as hakim:
        collected: list[str] = []
        async for v in hakim.voices.iter():
            collected.append(v["id"])

    assert collected == ["v_a", "v_b", "v_c"]
