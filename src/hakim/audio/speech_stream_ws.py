"""``audio.speech_stream_ws`` — realtime TTS over a WebSocket.

Companion surface to :meth:`SpeechAPI.create` and :meth:`SpeechAPI.stream`:

* :meth:`SpeechAPI.stream_ws` opens a long-lived WS session
  (``WSS /v1/audio/speech/stream``) and returns a
  :class:`SpeechStreamHandle` you can use to push many
  ``speech.create`` requests over a single TCP/TLS connection.
* Each utterance yields ``speech.started``, ``speech.audio`` chunks
  (raw PCM-S16LE @ 24 kHz mono), and ``speech.done`` in order.
* Session-wide defaults pinned via ``stream_ws(model=..., voice=...)``
  are sent once on connect; per-call ``send_speech`` arguments
  override them.

We lazy-import ``websockets`` (same trick the STT helper uses) so
synchronous-only callers don't have to pay the import cost.
"""

from __future__ import annotations

import contextlib
import json as _json
from collections.abc import AsyncIterator
from typing import Any

from .._types import (
    SpeechStreamCreateRequest,
    SpeechStreamEvent,
    SpeechStreamOptions,
)


class SpeechStreamHandle:
    """Long-lived realtime TTS WebSocket session.

    Iteration yields server events translated to the SDK schema
    (``speech.started`` / ``speech.audio`` / ``speech.done`` /
    ``session.usage`` / ``error``). Call :meth:`send_speech` to
    request an utterance and :meth:`close` (or use as an async
    context manager) when finished.

    The ``audio`` async-iterator is a convenience shortcut that
    yields only raw PCM chunks for callers piping straight into a
    speaker buffer.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        options: SpeechStreamOptions,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._options: SpeechStreamOptions = dict(options)  # type: ignore[assignment]
        self._ws: Any = None
        self._closed = False
        self._request_seq = 0
        # Tracks the request_id of the in-flight utterance so binary
        # frames (which carry no envelope) can be tagged with the
        # right correlation id. The server always emits a
        # ``speech.started`` text frame before any audio chunks, so
        # this single-cursor model is safe.
        self._current_request_id: str | None = None

    async def __aenter__(self) -> SpeechStreamHandle:
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

    async def send_speech(self, request: SpeechStreamCreateRequest) -> str:
        """Request one utterance over the open session.

        Returns the ``request_id`` (auto-generated when not supplied)
        so callers can correlate the subsequent ``speech.started`` /
        ``speech.audio`` / ``speech.done`` events.
        """

        if self._ws is None:
            await self._connect()
        assert self._ws is not None
        text = request.get("input", "")
        if not isinstance(text, str) or not text:
            raise ValueError("send_speech: 'input' must be a non-empty string")
        request_id = request.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            request_id = f"wst_local_{self._request_seq:x}"
            self._request_seq += 1
        frame: dict[str, Any] = {
            "type": "speech.create",
            "input": text,
            "request_id": request_id,
        }
        for k in ("voice", "model", "cfg", "voice_prompt"):
            if k in request:
                frame[k] = request[k]
        await self._ws.send(_json.dumps(frame))
        return request_id

    async def update_session(self, session: SpeechStreamOptions) -> None:
        """Update session-wide defaults mid-flight."""

        if self._ws is None:
            await self._connect()
        assert self._ws is not None
        for k in ("model", "voice", "cfg", "voice_prompt"):
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

    def __aiter__(self) -> AsyncIterator[SpeechStreamEvent]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[SpeechStreamEvent]:
        if self._ws is None:
            await self._connect()
        assert self._ws is not None
        try:
            async for raw in self._ws:
                if isinstance(raw, (bytes, bytearray, memoryview)):
                    if self._current_request_id is None:
                        # Out-of-protocol binary frame before
                        # ``speech.started`` — skip rather than
                        # emit an untyped event.
                        continue
                    yield {
                        "type": "speech.audio",
                        "request_id": self._current_request_id,
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
                    self._current_request_id = event["request_id"]
                yield event
        finally:
            await self.close()

    async def audio(self) -> AsyncIterator[bytes]:
        """Convenience iterator that yields only raw PCM chunks.

        Useful for piping straight into a speaker / file. Lifecycle
        events still need to be observed via the main iterator if you
        care about ``speech.done`` / billing.
        """

        async for event in self:
            if event["type"] == "speech.audio":
                yield event["chunk"]


# ---------------------------------------------------------------------------
# Internal frame translation — unit-tested via `tests/test_speech_stream_ws.py`.
# ---------------------------------------------------------------------------


def _translate_frame(frame: Any) -> SpeechStreamEvent | None:
    if not isinstance(frame, dict):
        return None
    t = frame.get("type")
    if t == "speech.started":
        if not isinstance(frame.get("request_id"), str):
            return None
        return {
            "type": "speech.started",
            "request_id": frame["request_id"],
            "characters": int(frame.get("characters", 0)),
            "sample_rate": int(frame.get("sample_rate", 24000)),
            "encoding": "pcm_s16le",
            "channels": 1,
            "model": frame.get("model", "hakim-fast-v1"),
            "voice": frame.get("voice", "unknown"),
        }
    if t == "speech.done":
        if not isinstance(frame.get("request_id"), str):
            return None
        return {
            "type": "speech.done",
            "request_id": frame["request_id"],
            "duration_ms": float(frame.get("duration_ms", 0)),
            "usage": frame.get("usage", {}),
        }
    if t == "session.usage":
        return {
            "type": "session.usage",
            "session_characters": int(frame.get("session_characters", 0)),
            "usage": frame.get("usage", {}),
        }
    if t == "error":
        out: dict[str, Any] = {
            "type": "error",
            "code": frame.get("code", "stream_error"),
            "message": frame.get("message", ""),
            "retryable": bool(frame.get("retryable", False)),
            "fatal": bool(frame.get("fatal", False)),
        }
        if isinstance(frame.get("request_id"), str):
            out["request_id"] = frame["request_id"]
        return out
    return None


def _build_session_update(opts: SpeechStreamOptions) -> dict[str, Any] | None:
    session: dict[str, Any] = {}
    for k in ("model", "voice", "cfg", "voice_prompt"):
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
    return f"{scheme}/v1/audio/speech/stream"


# Exposed to unit tests.
__internals = {
    "translate_frame": _translate_frame,
    "build_session_update": _build_session_update,
    "build_ws_url": _build_ws_url,
}
