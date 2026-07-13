"""Tests for ``audio.translate_stream_ws`` frame translator.

Mirrors ``test_audio_speech_stream_ws.py`` — focused on the pure
helpers (frame translation, session-update shape, URL builder, audio
ms estimator). End-to-end round-trips against a live WebSocket
server are covered by integration tests instead.
"""

from __future__ import annotations

from hakim.audio.translate_stream_ws import (
    _build_session_update,
    _build_ws_url,
    _estimate_audio_ms,
    _translate_frame,
)


def test_translate_session_created_frame() -> None:
    event = _translate_frame(
        {
            "type": "session.created",
            "event_id": 0,
            "session_id": "wst_01",
            "session": {"target_language": "en"},
            "voice_id": "voice_db_1",
            "voice_slug": "layla-en",
            "model_stt": "hakim-arab-v2",
            "model_llm": "hakim-chat-v1",
            "model_tts": "hakim-fast-v1",
            "limits": {},
            "usage_snapshot": {},
        }
    )
    assert event == {
        "type": "session.created",
        "session_id": "wst_01",
        "voice_id": "voice_db_1",
        "voice_slug": "layla-en",
        "model_stt": "hakim-arab-v2",
        "model_llm": "hakim-chat-v1",
        "model_tts": "hakim-fast-v1",
    }


def test_translate_transcription_delta_frame() -> None:
    event = _translate_frame(
        {
            "type": "transcription.delta",
            "event_id": 1,
            "utterance_id": "utt_1",
            "text": "مرحبا",
            "is_final": False,
        }
    )
    assert event == {
        "type": "transcription.delta",
        "utterance_id": "utt_1",
        "text": "مرحبا",
        "is_final": False,
    }


def test_translate_transcription_done_frame_carries_optional_language() -> None:
    event = _translate_frame(
        {
            "type": "transcription.done",
            "event_id": 2,
            "utterance_id": "utt_1",
            "text": "مرحبا بالعالم",
            "language": "ar",
            "audio_ms": 1240,
            "usage": {"kind": "stt_realtime", "units": 1},
        }
    )
    assert event is not None
    assert event["type"] == "transcription.done"
    assert event["text"] == "مرحبا بالعالم"
    assert event["language"] == "ar"
    assert event["audio_ms"] == 1240


def test_translate_translation_delta_frame() -> None:
    event = _translate_frame(
        {
            "type": "translation.delta",
            "event_id": 3,
            "utterance_id": "utt_1",
            "text": "Hello",
        }
    )
    assert event == {
        "type": "translation.delta",
        "utterance_id": "utt_1",
        "text": "Hello",
    }


def test_translate_translation_done_frame_with_usage() -> None:
    event = _translate_frame(
        {
            "type": "translation.done",
            "event_id": 4,
            "utterance_id": "utt_1",
            "text": "Hello, world.",
            "usage": {"kind": "llm_chat", "units": 24},
        }
    )
    assert event is not None
    assert event["type"] == "translation.done"
    assert event["text"] == "Hello, world."
    assert event["usage"]["units"] == 24


def test_translate_speech_started_frame() -> None:
    event = _translate_frame(
        {
            "type": "speech.started",
            "event_id": 5,
            "utterance_id": "utt_1",
            "characters": 13,
            "sample_rate": 24000,
            "encoding": "pcm_s16le",
            "channels": 1,
            "voice_id": "voice_db_1",
        }
    )
    assert event == {
        "type": "speech.started",
        "utterance_id": "utt_1",
        "characters": 13,
        "sample_rate": 24000,
        "encoding": "pcm_s16le",
        "channels": 1,
        "voice_id": "voice_db_1",
    }


def test_translate_speech_done_frame() -> None:
    event = _translate_frame(
        {
            "type": "speech.done",
            "event_id": 6,
            "utterance_id": "utt_1",
            "duration_ms": 980,
            "usage": {"kind": "tts", "units": 13},
        }
    )
    assert event is not None
    assert event["type"] == "speech.done"
    assert event["duration_ms"] == 980.0


def test_translate_session_usage_rollup() -> None:
    event = _translate_frame(
        {
            "type": "session.usage",
            "event_id": 7,
            "session_id": "wst_01",
            "totals": {
                "stt_audio_ms": 12400,
                "llm_tokens": 248,
                "tts_characters": 130,
                "credits": 165,
                "cost_usd": "0.0330",
            },
        }
    )
    assert event == {
        "type": "session.usage",
        "session_id": "wst_01",
        "totals": {
            "stt_audio_ms": 12400,
            "llm_tokens": 248,
            "tts_characters": 130,
            "credits": 165,
            "cost_usd": "0.0330",
        },
    }


def test_translate_error_frame_with_utterance_id() -> None:
    event = _translate_frame(
        {
            "type": "error",
            "event_id": 8,
            "code": "voice_not_found",
            "message": "no such voice",
            "retryable": False,
            "fatal": True,
            "utterance_id": "utt_1",
        }
    )
    assert event == {
        "type": "error",
        "code": "voice_not_found",
        "message": "no such voice",
        "retryable": False,
        "fatal": True,
        "utterance_id": "utt_1",
    }


def test_translate_ignores_unknown_or_malformed_frames() -> None:
    assert _translate_frame({"type": "translation.weird"}) is None
    assert _translate_frame("not an object") is None
    assert _translate_frame(None) is None
    assert _translate_frame({"type": "transcription.delta"}) is None


def test_build_session_update_returns_none_when_empty() -> None:
    assert _build_session_update({}) is None


def test_build_session_update_maps_fields() -> None:
    frame = _build_session_update(
        {
            "target_language": "en",
            "source_language": "ar",
            "voice": "voice_db_1",
            "gender": "female",
            "model_stt": "hakim-arab-v2",
            "model_llm": "hakim-chat-v1",
            "model_tts": "hakim-fast-v1",
            "cfg": 2.5,
            "input_audio_format": "pcm16",
            "input_sample_rate": 24000,
            "partials": False,
            "system_prompt": "translate to English",
        }
    )
    assert frame == {
        "type": "session.update",
        "session": {
            "target_language": "en",
            "source_language": "ar",
            "voice": "voice_db_1",
            "gender": "female",
            "model_stt": "hakim-arab-v2",
            "model_llm": "hakim-chat-v1",
            "model_tts": "hakim-fast-v1",
            "cfg": 2.5,
            "input_audio_format": "pcm16",
            "input_sample_rate": 24000,
            "partials": False,
            "system_prompt": "translate to English",
        },
    }


def test_build_ws_url_handles_http_https_and_prefixed_paths() -> None:
    assert _build_ws_url("https://api.tryhakim.ai") == (
        "wss://api.tryhakim.ai/v1/audio/translate/stream"
    )
    assert _build_ws_url("http://localhost:8787") == (
        "ws://localhost:8787/v1/audio/translate/stream"
    )
    assert _build_ws_url("https://api.example.com/prefix") == (
        "wss://api.example.com/prefix/v1/audio/translate/stream"
    )


def test_estimate_audio_ms_pcm16() -> None:
    assert _estimate_audio_ms(32000, 16000, "pcm16") == 1000
    assert _estimate_audio_ms(96000, 24000, "pcm16") == 2000


def test_estimate_audio_ms_non_pcm_returns_none() -> None:
    assert _estimate_audio_ms(1000, 16000, "opus") is None
    assert _estimate_audio_ms(1000, 16000, "mulaw") is None
