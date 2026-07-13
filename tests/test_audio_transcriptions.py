"""Tests for `audio.transcriptions` (sync STT) + frame translator internals."""

from __future__ import annotations

import json as _json

import httpx
import pytest

from hakim.audio.transcriptions import (
    _build_session_update,
    _build_ws_url,
    _translate_frame,
)

from ._helpers import text_response


async def test_stt_create_uploads_multipart_and_returns_text(build_mock_client):  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["content_type"] = request.headers["content-type"]
        captured["body"] = request.content
        return text_response(200, "hello world", content_type="text/plain")

    async with build_mock_client(handler) as hakim:
        result = await hakim.audio.transcriptions.create(
            file=b"\x00\x01\x02\x03",
            model="hakim-arab-v2",
            language="en",
            response_format="text",
            filename="clip.wav",
        )

    assert result == "hello world"
    assert captured["method"] == "POST"
    assert captured["path"] == "/v1/audio/transcriptions"
    assert "multipart/form-data" in str(captured["content_type"])
    body = captured["body"]
    assert isinstance(body, bytes)
    assert b"hakim-arab-v2" in body
    assert b"language" in body
    assert b"clip.wav" in body


async def test_stt_create_json_body_is_decoded(build_mock_client):  # type: ignore[no-untyped-def]
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={"text": "hi", "language": "en", "duration": 1.2},
        )

    async with build_mock_client(handler) as hakim:
        result = await hakim.audio.transcriptions.create(
            file=b"\x00",
            model="hakim-arab-v2",
            response_format="json",
        )

    assert isinstance(result, dict)
    assert result["text"] == "hi"


async def test_stt_create_url_sends_json_and_returns_async_accept(build_mock_client):  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["content_type"] = request.headers.get("content-type", "")
        captured["body"] = request.content
        return httpx.Response(
            202,
            headers={"content-type": "application/json"},
            json={
                "id": "job_1",
                "status": "queued",
                "type": "batch_stt",
                "reason": "remote_url",
                "poll_url": "/v1/jobs/job_1",
            },
        )

    async with build_mock_client(handler) as hakim:
        result = await hakim.audio.transcriptions.create(
            url="https://my-bucket.s3.eu-central-1.amazonaws.com/call.mp3?sig=x",
            model="hakim-arab-v2",
            language="ar",
        )

    assert isinstance(result, dict)
    assert result["reason"] == "remote_url"
    assert "application/json" in str(captured["content_type"])
    body = _json.loads(captured["body"])  # type: ignore[arg-type]
    assert body["url"].startswith("https://my-bucket")
    assert body["model"] == "hakim-arab-v2"
    assert body["language"] == "ar"


async def test_stt_create_requires_exactly_one_of_file_or_url(build_mock_client):  # type: ignore[no-untyped-def]
    def handler(_: httpx.Request) -> httpx.Response:
        return text_response(200, "x", content_type="text/plain")

    async with build_mock_client(handler) as hakim:
        with pytest.raises(ValueError):
            await hakim.audio.transcriptions.create(model="hakim-arab-v2")
        with pytest.raises(ValueError):
            await hakim.audio.transcriptions.create(
                file=b"x",
                url="https://x.s3.amazonaws.com/a",
                model="hakim-arab-v2",
            )


# ---------------------------------------------------------------------------
# Frame translator — pure-function tests (no WebSocket required).
# ---------------------------------------------------------------------------


def test_translate_partial_delta() -> None:
    event = _translate_frame(
        {"type": "transcription.delta", "event_id": 3, "text": "hel", "is_final": False}
    )
    assert event == {"type": "partial", "text": "hel", "seq": 3}


def test_translate_final_delta_converts_ms_to_seconds() -> None:
    event = _translate_frame(
        {
            "type": "transcription.delta",
            "event_id": 4,
            "text": "hello world",
            "is_final": True,
            "start_ms": 1000,
            "end_ms": 2500,
        }
    )
    assert event == {
        "type": "final",
        "text": "hello world",
        "seq": 4,
        "start": 1.0,
        "end": 2.5,
    }


def test_translate_transcription_done() -> None:
    event = _translate_frame(
        {"type": "transcription.done", "event_id": 7, "text": "done", "language": "ar"}
    )
    assert event == {"type": "final", "text": "done", "seq": 7, "language": "ar"}


def test_translate_usage_frame() -> None:
    event = _translate_frame({"type": "usage", "audio_ms": 1500})
    assert event == {"type": "usage", "seconds": 1.5}


def test_translate_error_frame() -> None:
    event = _translate_frame(
        {"type": "error", "code": "quota_exceeded", "message": "nope"}
    )
    assert event == {"type": "error", "code": "quota_exceeded", "message": "nope"}


def test_translate_ignores_unknown_and_invalid_frames() -> None:
    assert _translate_frame({"type": "session.created", "event_id": 1}) is None
    assert _translate_frame({"type": "?"}) is None
    assert _translate_frame("not an object") is None
    assert _translate_frame(None) is None


def test_build_session_update_returns_none_when_empty() -> None:
    assert _build_session_update({}) is None


def test_build_session_update_maps_caller_fields_to_server_schema() -> None:
    frame = _build_session_update(
        {
            "model": "hakim-arab-v2",
            "language": "ar",
            "sample_rate": 24000,
            "audio_format": "pcm16",
        }
    )
    assert frame == {
        "type": "session.update",
        "session": {
            "model": "hakim-arab-v2",
            "language": "ar",
            "input_sample_rate": 24000,
            "input_audio_format": "pcm16",
        },
    }


def test_build_ws_url_handles_http_https_and_prefixed_paths() -> None:
    assert _build_ws_url("https://api.tryhakim.ai") == (
        "wss://api.tryhakim.ai/v1/audio/transcriptions/stream"
    )
    assert _build_ws_url("http://localhost:8787") == (
        "ws://localhost:8787/v1/audio/transcriptions/stream"
    )
    assert _build_ws_url("https://api.example.com/prefix") == (
        "wss://api.example.com/prefix/v1/audio/transcriptions/stream"
    )
