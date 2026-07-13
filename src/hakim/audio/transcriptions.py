"""``audio.transcriptions`` — sync STT upload + realtime WSS stream.

* :meth:`TranscriptionsAPI.create` — multipart upload of a file / bytes /
  file-like; returns whatever the server sent (text, JSON, SRT, VTT, or
  a 202 async-job accept body).
* :meth:`TranscriptionsAPI.stream` — open a realtime STT WebSocket and
  return a :class:`TranscriptionStreamHandle` you can iterate and push
  audio chunks into.
"""

from __future__ import annotations

import base64
import contextlib
import json as _json
from collections.abc import AsyncIterator
from typing import Any, BinaryIO, Literal, cast

from .._transport import AsyncTransport
from .._types import (
    STTLanguage,
    STTModel,
    STTResponseFormat,
    STTTimestamps,
    TranscriptionResult,
    TranscriptionStreamEvent,
    TranscriptionStreamOptions,
)

# An audio payload acceptable to `create()`: raw bytes, file-like with a
# `read()` method, or a tuple of `(filename, bytes_or_fileobj, content_type)`.
AudioInput = bytes | bytearray | memoryview | BinaryIO | tuple[str, Any, str]


class TranscriptionsAPI:
    def __init__(self, transport: AsyncTransport) -> None:
        self._t = transport

    async def create(
        self,
        *,
        file: AudioInput | None = None,
        url: str | None = None,
        model: STTModel,
        language: STTLanguage | None = None,
        response_format: STTResponseFormat | None = None,
        timestamps: STTTimestamps | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        diarization: bool | None = None,
        filename: str | None = None,
        idempotency_key: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio via the batch endpoint.

        Provide exactly one of ``file`` (upload the bytes) or ``url`` (a
        publicly fetchable S3/GCS/Azure presigned URL the server fetches
        for you — handy for buckets full of recordings). URL inputs are
        always processed asynchronously: the server returns a 202 accept
        body with ``job_id`` + ``poll_url``.

        Returns the parsed body. For text/SRT/VTT ``response_format``s
        the return is a :class:`str`; for JSON / verbose_json it's a
        ``dict``; for the async path it's the 202 accept body.
        """

        if (file is None) == (url is None):
            raise ValueError("provide exactly one of `file` or `url`")

        if url is not None:
            body: dict[str, Any] = {"model": model, "url": url}
            if language is not None:
                body["language"] = language
            if response_format is not None:
                body["response_format"] = response_format
            if timestamps is not None:
                body["timestamps"] = timestamps
            if diarization is not None:
                body["diarize"] = diarization
            response = await self._t.request(
                "POST",
                "v1/audio/transcriptions",
                json=body,
                idempotency_key=idempotency_key,
            )
        else:
            form_file = _to_httpx_file(cast(AudioInput, file), filename)
            data: dict[str, Any] = {"model": model}
            if language is not None:
                data["language"] = language
            if response_format is not None:
                data["response_format"] = response_format
            if timestamps is not None:
                data["timestamps"] = timestamps
            if prompt is not None:
                data["prompt"] = prompt
            if temperature is not None:
                data["temperature"] = str(temperature)
            if diarization is not None:
                # Wire field is `diarize` (the server's multipart field name).
                data["diarize"] = "true" if diarization else "false"

            response = await self._t.request(
                "POST",
                "v1/audio/transcriptions",
                files=[("file", form_file)],
                data=data,
                idempotency_key=idempotency_key,
            )

        content_type = response.headers.get("content-type", "")
        body_bytes = await response.aread()
        await response.aclose()
        if "application/json" in content_type:
            return cast(TranscriptionResult, _json.loads(body_bytes.decode("utf-8")))
        return body_bytes.decode("utf-8")

    def stream(
        self,
        *,
        model: STTModel | None = None,
        language: STTLanguage | None = None,
        sample_rate: Literal[16000, 22050, 24000, 44100, 48000] | None = None,
        audio_format: Literal["pcm16", "opus", "mulaw"] | None = None,
    ) -> TranscriptionStreamHandle:
        """Open a realtime STT WebSocket.

        Usage::

            async with hakim.audio.transcriptions.stream(model="hakim-arab-v2") as h:
                async def pump():
                    async for chunk in mic_chunks():
                        await h.send_audio(chunk)
                    await h.commit()
                asyncio.create_task(pump())
                async for event in h:
                    print(event)
        """

        options: TranscriptionStreamOptions = {}
        if model is not None:
            options["model"] = model
        if language is not None:
            options["language"] = language
        if sample_rate is not None:
            options["sample_rate"] = sample_rate
        if audio_format is not None:
            options["audio_format"] = audio_format

        return TranscriptionStreamHandle(
            api_key=self._t.api_key,
            base_url=self._t.base_url,
            options=options,
        )


# ---------------------------------------------------------------------------
# Realtime stream handle
# ---------------------------------------------------------------------------


class TranscriptionStreamHandle:
    """Wrapper around a realtime STT WebSocket.

    Iteration yields server events translated to the SDK schema
    (``partial`` / ``final`` / ``usage`` / ``error``). Call
    :meth:`send_audio` to push raw audio frames and :meth:`commit`
    when the utterance is finished.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        options: TranscriptionStreamOptions,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._options = options
        self._ws: Any = None
        self._closed = False

    async def __aenter__(self) -> TranscriptionStreamHandle:
        await self._connect()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def _connect(self) -> None:
        # Imported lazily so callers that only use sync TTS/STT don't
        # have to pay the ``websockets`` import cost.
        import websockets

        url = _build_ws_url(self._base_url)
        extra_headers = [("authorization", f"Bearer {self._api_key}")]
        self._ws = await websockets.connect(
            url,
            additional_headers=extra_headers,
        )
        session_update = _build_session_update(self._options)
        if session_update is not None:
            await self._ws.send(_json.dumps(session_update))

    async def send_audio(self, chunk: bytes) -> None:
        """Send an audio frame. Encoded as ``input_audio_buffer.append``."""

        if self._ws is None:
            await self._connect()
        assert self._ws is not None
        frame = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(chunk).decode("ascii"),
        }
        await self._ws.send(_json.dumps(frame))

    async def commit(self) -> None:
        """Signal end-of-utterance. Server will flush a ``transcription.done``."""

        if self._ws is None:
            return
        await self._ws.send(_json.dumps({"type": "input_audio_buffer.commit"}))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()

    def __aiter__(self) -> AsyncIterator[TranscriptionStreamEvent]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[TranscriptionStreamEvent]:
        if self._ws is None:
            await self._connect()
        assert self._ws is not None
        try:
            async for raw in self._ws:
                if isinstance(raw, (bytes, bytearray)):
                    continue
                try:
                    frame = _json.loads(raw)
                except ValueError:
                    continue
                event = _translate_frame(frame)
                if event is not None:
                    yield event
        finally:
            await self.close()


# ---------------------------------------------------------------------------
# Internal frame translation — unit-tested via `tests/test_audio_stream.py`.
# ---------------------------------------------------------------------------


def _translate_frame(frame: Any) -> TranscriptionStreamEvent | None:
    if not isinstance(frame, dict):
        return None
    t = frame.get("type")
    if t == "transcription.delta":
        text = frame.get("text", "")
        seq = frame.get("event_id", 0)
        if frame.get("is_final") is True:
            event: dict[str, Any] = {"type": "final", "text": text, "seq": seq}
            if "start_ms" in frame:
                event["start"] = float(frame["start_ms"]) / 1000.0
            if "end_ms" in frame:
                event["end"] = float(frame["end_ms"]) / 1000.0
            if "language" in frame:
                event["language"] = frame["language"]
            return event
        return {"type": "partial", "text": text, "seq": seq}
    if t == "transcription.done":
        out: dict[str, Any] = {
            "type": "final",
            "text": frame.get("text", ""),
            "seq": frame.get("event_id", 0),
        }
        if "language" in frame:
            out["language"] = frame["language"]
        return out
    if t == "usage":
        audio_ms = frame.get("audio_ms")
        if isinstance(audio_ms, (int, float)):
            return {"type": "usage", "seconds": audio_ms / 1000.0}
        seconds = frame.get("seconds", 0)
        return {"type": "usage", "seconds": float(seconds)}
    if t == "error":
        return {
            "type": "error",
            "code": frame.get("code", "stream_error"),
            "message": frame.get("message", ""),
        }
    return None


def _build_session_update(opts: TranscriptionStreamOptions) -> dict[str, Any] | None:
    session: dict[str, Any] = {}
    if "model" in opts:
        session["model"] = opts["model"]
    if "language" in opts:
        session["language"] = opts["language"]
    if "sample_rate" in opts:
        session["input_sample_rate"] = opts["sample_rate"]
    if "audio_format" in opts:
        session["input_audio_format"] = opts["audio_format"]
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
    return f"{scheme}/v1/audio/transcriptions/stream"


def _to_httpx_file(source: AudioInput, filename: str | None) -> tuple[str, Any, str]:
    """Coerce the public ``AudioInput`` union to httpx's multipart tuple."""

    if isinstance(source, tuple) and len(source) == 3:
        # Already (filename, payload, content_type)
        return source
    name = filename or "audio.wav"
    content_type = _guess_content_type(name)
    if isinstance(source, (bytes, bytearray, memoryview)):
        return (name, bytes(source), content_type)
    # File-like — let httpx call .read().
    return (name, source, content_type)


def _guess_content_type(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".wav"):
        return "audio/wav"
    if lower.endswith(".mp3"):
        return "audio/mpeg"
    if lower.endswith(".ogg"):
        return "audio/ogg"
    if lower.endswith(".flac"):
        return "audio/flac"
    if lower.endswith(".m4a"):
        return "audio/mp4"
    return "application/octet-stream"


# Exposed to unit tests.
__internals = {
    "translate_frame": _translate_frame,
    "build_session_update": _build_session_update,
    "build_ws_url": _build_ws_url,
}
