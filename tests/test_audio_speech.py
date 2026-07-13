"""Tests for `audio.speech` — buffered + streaming TTS."""

from __future__ import annotations

import json

import httpx
import pytest

from ._helpers import json_response


async def test_speech_create_buffers_full_audio(build_mock_client) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        captured["idempotency_key"] = request.headers.get("idempotency-key")
        return httpx.Response(
            200,
            headers={"content-type": "audio/mpeg"},
            content=b"\x00\x11\x22\x33",
        )

    async with build_mock_client(handler) as hakim:
        audio = await hakim.audio.speech.create(
            model="hakim-fast-v1",
            input="hello",
            voice="omar",
        )

    assert audio == b"\x00\x11\x22\x33"
    assert captured["method"] == "POST"
    assert captured["path"] == "/v1/audio/speech"
    assert captured["body"] == {"model": "hakim-fast-v1", "input": "hello", "voice": "omar"}
    assert captured["idempotency_key"]  # auto-generated


async def test_speech_create_forwards_optional_fields(build_mock_client) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=b"ok", headers={"content-type": "audio/wav"})

    async with build_mock_client(handler) as hakim:
        await hakim.audio.speech.create(
            model="hakim-pro-v1",
            input="salam",
            voice="leila",
            response_format="wav",
            sample_rate=24000,
            speed=1.1,
            seed=42,
            idempotency_key="my-key",
        )

    assert captured["body"] == {
        "model": "hakim-pro-v1",
        "input": "salam",
        "voice": "leila",
        "response_format": "wav",
        "sample_rate": 24000,
        "speed": 1.1,
        "seed": 42,
    }


async def test_speech_stream_yields_chunks_in_order(build_mock_client) -> None:  # type: ignore[no-untyped-def]
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "audio/mpeg"},
            content=b"AAAABBBBCCCC",
        )

    async with build_mock_client(handler) as hakim:
        iterator = await hakim.audio.speech.stream(
            model="hakim-fast-v1", input="hi", voice="omar", chunk_size=4,
        )
        chunks: list[bytes] = []
        async for c in iterator:
            chunks.append(c)

    # Order preserved; total bytes match.
    assert b"".join(chunks) == b"AAAABBBBCCCC"


async def test_speech_error_bubbles_as_hakimerror(build_mock_client) -> None:  # type: ignore[no-untyped-def]
    from hakim import QuotaExceededError

    def handler(_: httpx.Request) -> httpx.Response:
        return json_response(
            402,
            {
                "error": {
                    "type": "quota_exceeded",
                    "code": "monthly_chars_exceeded",
                    "message": "no",
                }
            },
        )

    async with build_mock_client(handler) as hakim:
        with pytest.raises(QuotaExceededError):
            await hakim.audio.speech.create(
                model="hakim-fast-v1", input="hi", voice="omar",
            )
