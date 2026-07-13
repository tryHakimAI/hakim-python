"""``audio.translate_stream_ws`` — realtime speech translation over WS.

Chains STT → LLM → TTS behind a single WebSocket session at
``WSS /v1/audio/translate/stream``. The caller pushes source audio
via :meth:`TranslateStreamHandle.send_audio` and consumes a mixed
stream of lifecycle frames + binary PCM audio for the synthesised
target.

Same dependency posture as the standalone STT / TTS helpers — we
lazy-import :mod:`websockets` so sync-only callers don't pay the
import cost.
"""

from __future__ import annotations

import base64 as _base64
import contextlib
import json as _json
from collections.abc import AsyncIterator
from typing import Any

from .._types import (
    TranslateStreamEvent,
    TranslateStreamOptions,
)


class TranslateStreamHandle:
    """Long-lived realtime translate WebSocket session.

    Iteration yields server events translated to the SDK schema
    (``session.created`` / ``transcription.delta`` /
    ``transcription.done`` / ``translation.delta`` /
    ``translation.done`` / ``speech.started`` / ``speech.audio`` /
    ``speech.done`` / ``session.usage`` / ``error``). Call
    :meth:`send_audio` to push a source-audio chunk, optionally
    :meth:`commit_audio` to force an utterance boundary, and
    :meth:`close` (or use as an async context manager) when finished.

    The :meth:`audio` async-iterator yields only the synthesised PCM
    chunks for callers piping straight into a speaker buffer.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        options: TranslateStreamOptions,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._options: TranslateStreamOptions = dict(options)  # type: ignore[assignment]
        self._ws: Any = None
        self._closed = False
        # Tracks the in-flight utterance_id so binary frames (which
        # carry no envelope) can be tagged correctly. The proxy
        # always emits ``speech.started`` text before any audio chunks.
        self._current_utterance_id: str | None = None

    async def __aenter__(self) -> TranslateStreamHandle:
        await self._connect()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def _connect(self) -> None:
        if self._ws is not None:
            return
        import websockets

        url = _build_ws_url(self._base_url)
        extra_headers = [("authorization", f"Bearer {self._api_key}")]
        self._ws = await websockets.connect(url, additional_headers=extra_headers)
        session_update = _build_session_update(self._options)
        if session_update is not None:
            await self._ws.send(_json.dumps(session_update))

    async def send_audio(self, chunk: bytes | bytearray | memoryview) -> None:
        """Append a chunk of source PCM audio to the input buffer."""

        if self._ws is None:
            await self._connect()
        assert self._ws is not None
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise TypeError("send_audio: chunk must be bytes-like")
        data = bytes(chunk)
        frame: dict[str, Any] = {
            "type": "input_audio_buffer.append",
            "audio": _base64.b64encode(data).decode("ascii"),
        }
        audio_ms = _estimate_audio_ms(
            len(data),
            int(self._options.get("input_sample_rate", 16000)),
            str(self._options.get("input_audio_format", "pcm16")),
        )
        if audio_ms is not None:
            frame["audio_ms"] = audio_ms
        await self._ws.send(_json.dumps(frame))

    async def commit_audio(self) -> None:
        """Force an immediate STT utterance boundary."""

        if self._ws is None:
            await self._connect()
        assert self._ws is not None
        await self._ws.send(_json.dumps({"type": "input_audio_buffer.commit"}))

    async def update_session(self, session: TranslateStreamOptions) -> None:
        """Update session-wide defaults mid-flight."""

        if self._ws is None:
            await self._connect()
        assert self._ws is not None
        for k in (
            "target_language",
            "source_language",
            "voice",
            "gender",
            "model_stt",
            "model_llm",
            "model_tts",
            "cfg",
            "input_audio_format",
            "input_sample_rate",
            "partials",
            "system_prompt",
        ):
            if k in session:
                self._options[k] = session[k]
        frame = _build_session_update(session)
        if frame is not None:
            await self._ws.send(_json.dumps(frame))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.send(_json.dumps({"type": "session.close"}))
            with contextlib.suppress(Exception):
                await self._ws.close()

    def __aiter__(self) -> AsyncIterator[TranslateStreamEvent]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[TranslateStreamEvent]:
        if self._ws is None:
            await self._connect()
        assert self._ws is not None
        try:
            async for raw in self._ws:
                if isinstance(raw, (bytes, bytearray, memoryview)):
                    if self._current_utterance_id is None:
                        continue
                    yield {
                        "type": "speech.audio",
                        "utterance_id": self._current_utterance_id,
                        "chunk": bytes(raw),
                    }
                    continue
                try:
                    frame = _json.loads(raw)
                except ValueError:
                    continue
                event = _translate_frame(frame)
                if event is None:
                    continue
                if event["type"] == "speech.started":
                    self._current_utterance_id = event["utterance_id"]
                yield event
        finally:
            await self.close()

    async def audio(self) -> AsyncIterator[bytes]:
        """Convenience iterator yielding only synthesised PCM chunks."""

        async for event in self:
            if event["type"] == "speech.audio":
                yield event["chunk"]


# ---------------------------------------------------------------------------
# Internal frame translation — unit-tested via `tests/test_translate_stream_ws.py`.
# ---------------------------------------------------------------------------


def _translate_frame(frame: Any) -> TranslateStreamEvent | None:
    if not isinstance(frame, dict):
        return None
    t = frame.get("type")
    if t == "session.created":
        for k in ("session_id", "voice_id", "voice_slug", "model_stt", "model_llm", "model_tts"):
            if not isinstance(frame.get(k), str):
                return None
        return {
            "type": "session.created",
            "session_id": frame["session_id"],
            "voice_id": frame["voice_id"],
            "voice_slug": frame["voice_slug"],
            "model_stt": frame["model_stt"],
            "model_llm": frame["model_llm"],
            "model_tts": frame["model_tts"],
        }
    if t == "transcription.delta":
        if not isinstance(frame.get("utterance_id"), str) or not isinstance(
            frame.get("text"), str
        ):
            return None
        return {
            "type": "transcription.delta",
            "utterance_id": frame["utterance_id"],
            "text": frame["text"],
            "is_final": bool(frame.get("is_final", False)),
        }
    if t == "transcription.done":
        if not isinstance(frame.get("utterance_id"), str) or not isinstance(
            frame.get("text"), str
        ):
            return None
        out: dict[str, Any] = {
            "type": "transcription.done",
            "utterance_id": frame["utterance_id"],
            "text": frame["text"],
            "audio_ms": int(frame.get("audio_ms", 0)),
            "usage": frame.get("usage", {}),
        }
        if isinstance(frame.get("language"), str):
            out["language"] = frame["language"]
        return out
    if t == "translation.delta":
        if not isinstance(frame.get("utterance_id"), str) or not isinstance(
            frame.get("text"), str
        ):
            return None
        return {
            "type": "translation.delta",
            "utterance_id": frame["utterance_id"],
            "text": frame["text"],
        }
    if t == "translation.done":
        if not isinstance(frame.get("utterance_id"), str) or not isinstance(
            frame.get("text"), str
        ):
            return None
        return {
            "type": "translation.done",
            "utterance_id": frame["utterance_id"],
            "text": frame["text"],
            "usage": frame.get("usage", {}),
        }
    if t == "speech.started":
        if not isinstance(frame.get("utterance_id"), str):
            return None
        return {
            "type": "speech.started",
            "utterance_id": frame["utterance_id"],
            "characters": int(frame.get("characters", 0)),
            "sample_rate": int(frame.get("sample_rate", 24000)),
            "encoding": "pcm_s16le",
            "channels": 1,
            "voice_id": str(frame.get("voice_id", "unknown")),
        }
    if t == "speech.done":
        if not isinstance(frame.get("utterance_id"), str):
            return None
        return {
            "type": "speech.done",
            "utterance_id": frame["utterance_id"],
            "duration_ms": float(frame.get("duration_ms", 0)),
            "usage": frame.get("usage", {}),
        }
    if t == "session.usage":
        if not isinstance(frame.get("session_id"), str):
            return None
        totals = frame.get("totals") or {}
        if not isinstance(totals, dict):
            return None
        return {
            "type": "session.usage",
            "session_id": frame["session_id"],
            "totals": {
                "stt_audio_ms": int(totals.get("stt_audio_ms", 0)),
                "llm_tokens": int(totals.get("llm_tokens", 0)),
                "tts_characters": int(totals.get("tts_characters", 0)),
                "credits": int(totals.get("credits", 0)),
                "cost_usd": str(totals.get("cost_usd", "0")),
            },
        }
    if t == "error":
        out_err: dict[str, Any] = {
            "type": "error",
            "code": frame.get("code", "stream_error"),
            "message": frame.get("message", ""),
            "retryable": bool(frame.get("retryable", False)),
            "fatal": bool(frame.get("fatal", False)),
        }
        if isinstance(frame.get("utterance_id"), str):
            out_err["utterance_id"] = frame["utterance_id"]
        return out_err
    return None


def _build_session_update(opts: TranslateStreamOptions) -> dict[str, Any] | None:
    session: dict[str, Any] = {}
    for k in (
        "target_language",
        "source_language",
        "voice",
        "gender",
        "model_stt",
        "model_llm",
        "model_tts",
        "cfg",
        "input_audio_format",
        "input_sample_rate",
        "partials",
        "system_prompt",
    ):
        if k in opts:
            session[k] = opts[k]
    if not session:
        return None
    return {"type": "session.update", "session": session}


def _build_ws_url(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    if trimmed.startswith("https://"):
        scheme = "wss://" + trimmed[len("https://") :]
    elif trimmed.startswith("http://"):
        scheme = "ws://" + trimmed[len("http://") :]
    else:
        scheme = trimmed
    return f"{scheme}/v1/audio/translate/stream"


def _estimate_audio_ms(byte_length: int, sample_rate: int, fmt: str) -> int | None:
    if fmt == "pcm16" and sample_rate > 0:
        return round((byte_length / 2 / sample_rate) * 1000)
    return None


__internals = {
    "translate_frame": _translate_frame,
    "build_session_update": _build_session_update,
    "build_ws_url": _build_ws_url,
    "estimate_audio_ms": _estimate_audio_ms,
}
