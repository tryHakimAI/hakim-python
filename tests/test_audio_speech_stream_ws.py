"""Tests for ``audio.speech_stream_ws`` (realtime TTS) frame translator.

Mirrors ``test_audio_transcriptions.py``: we keep the unit tests
focused on the pure helpers (frame translation, session-update
shape, URL building). End-to-end round-trips against a live ``ws``
server are covered by integration tests instead.
"""

from __future__ import annotations

from hakim.audio.speech_stream_ws import (
    _build_session_update,
    _build_ws_url,
    _translate_frame,
)


def test_translate_speech_started_frame() -> None:
    event = _translate_frame(
        {
            "type": "speech.started",
            "event_id": 1,
            "request_id": "utt_1",
            "characters": 12,
            "sample_rate": 24000,
            "encoding": "pcm_s16le",
            "channels": 1,
            "model": "hakim-fast-v1",
            "voice": "rashed-ar",
        }
    )
    assert event == {
        "type": "speech.started",
        "request_id": "utt_1",
        "characters": 12,
        "sample_rate": 24000,
        "encoding": "pcm_s16le",
        "channels": 1,
        "model": "hakim-fast-v1",
        "voice": "rashed-ar",
    }


def test_translate_speech_done_frame() -> None:
    event = _translate_frame(
        {
            "type": "speech.done",
            "event_id": 2,
            "request_id": "utt_1",
            "duration_ms": 1250,
            "usage": {
                "request_id": "req_1",
                "kind": "tts",
                "units": 12,
                "unit_type": "characters",
                "credits": 12,
                "cost_usd": "0.00",
                "model": "hakim-fast-v1",
            },
        }
    )
    assert event is not None
    assert event["type"] == "speech.done"
    assert event["request_id"] == "utt_1"
    assert event["duration_ms"] == 1250.0
    assert event["usage"]["units"] == 12


def test_translate_session_usage_frame() -> None:
    event = _translate_frame(
        {
            "type": "session.usage",
            "event_id": 9,
            "session_id": "wst_abc",
            "session_characters": 240,
            "usage": {
                "request_id": "req_1",
                "kind": "tts",
                "units": 240,
                "unit_type": "characters",
                "credits": 240,
                "cost_usd": "0.00",
                "model": "hakim-fast-v1",
            },
        }
    )
    assert event is not None
    assert event["type"] == "session.usage"
    assert event["session_characters"] == 240


def test_translate_error_frame_with_flags() -> None:
    event = _translate_frame(
        {
            "type": "error",
            "event_id": 0,
            "code": "voice_not_found",
            "message": "no such voice",
            "retryable": False,
            "fatal": False,
            "request_id": "utt_x",
        }
    )
    assert event == {
        "type": "error",
        "code": "voice_not_found",
        "message": "no such voice",
        "retryable": False,
        "fatal": False,
        "request_id": "utt_x",
    }


def test_translate_ignores_unknown_or_malformed_frames() -> None:
    assert _translate_frame({"type": "session.created", "event_id": 1}) is None
    assert _translate_frame({"type": "speech.weird"}) is None
    assert _translate_frame({"type": "speech.started"}) is None
    assert _translate_frame("not an object") is None
    assert _translate_frame(None) is None


def test_build_session_update_returns_none_when_empty() -> None:
    assert _build_session_update({}) is None


def test_build_session_update_maps_fields() -> None:
    frame = _build_session_update(
        {
            "model": "hakim-fast-v1",
            "voice": "rashed-ar",
            "cfg": 3,
            "voice_prompt": "whispering Arabic narrator",
        }
    )
    assert frame == {
        "type": "session.update",
        "session": {
            "model": "hakim-fast-v1",
            "voice": "rashed-ar",
            "cfg": 3,
            "voice_prompt": "whispering Arabic narrator",
        },
    }


def test_build_ws_url_handles_http_https_and_prefixed_paths() -> None:
    assert _build_ws_url("https://api.tryhakim.ai") == (
        "wss://api.tryhakim.ai/v1/audio/speech/stream"
    )
    assert _build_ws_url("http://localhost:8787") == (
        "ws://localhost:8787/v1/audio/speech/stream"
    )
    assert _build_ws_url("https://api.example.com/prefix") == (
        "wss://api.example.com/prefix/v1/audio/speech/stream"
    )
