"""``audio.speech`` — text-to-speech.

Two entry points:

* :meth:`SpeechAPI.create` — buffered synthesis. Returns ``bytes``
  (audio in the requested format).
* :meth:`SpeechAPI.stream` — yields raw audio chunks as they arrive
  (``AsyncIterator[bytes]``). Use this for low-latency playback.

The sync :class:`hakim.Hakim` wrapper re-exposes both as blocking calls.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

from .._transport import AsyncTransport
from .._types import (
    ResponseFormat,
    SampleRate,
    SpeechRequest,
    SpeechStreamOptions,
    TTSModel,
)
from .speech_stream_ws import SpeechStreamHandle


class SpeechAPI:
    def __init__(self, transport: AsyncTransport) -> None:
        self._t = transport

    async def create(
        self,
        *,
        model: TTSModel,
        input: str,
        voice: str,
        response_format: ResponseFormat | None = None,
        sample_rate: SampleRate | None = None,
        speed: float | None = None,
        seed: int | None = None,
        idempotency_key: str | None = None,
    ) -> bytes:
        """Synthesise speech and return the full audio buffer.

        Args:
            model: TTS model id.
            input: Text to synthesise.
            voice: Voice slug or id.
            response_format: Audio container. Defaults to server-side
                default (``mp3``).
            sample_rate: Optional sample rate override.
            speed: 0.25 - 4.0 multiplier.
            seed: Deterministic seed for reproducible generations.

        Returns:
            Audio bytes. Decode/save as needed.
        """

        body = cast(SpeechRequest, {"model": model, "input": input, "voice": voice})
        if response_format is not None:
            body["response_format"] = response_format
        if sample_rate is not None:
            body["sample_rate"] = sample_rate
        if speed is not None:
            body["speed"] = speed
        if seed is not None:
            body["seed"] = seed

        response = await self._t.request(
            "POST",
            "v1/audio/speech",
            json=body,
            accept="audio/*",
            idempotency_key=idempotency_key,
        )
        return await response.aread()

    async def stream(
        self,
        *,
        model: TTSModel,
        input: str,
        voice: str,
        response_format: ResponseFormat | None = None,
        sample_rate: SampleRate | None = None,
        speed: float | None = None,
        seed: int | None = None,
        idempotency_key: str | None = None,
        chunk_size: int = 4096,
    ) -> AsyncIterator[bytes]:
        """Stream synthesised audio as it arrives."""

        body = cast(SpeechRequest, {"model": model, "input": input, "voice": voice, "stream": True})
        if response_format is not None:
            body["response_format"] = response_format
        if sample_rate is not None:
            body["sample_rate"] = sample_rate
        if speed is not None:
            body["speed"] = speed
        if seed is not None:
            body["seed"] = seed

        response = await self._t.request(
            "POST",
            "v1/audio/speech",
            json=body,
            stream=True,
            accept="audio/*",
            idempotency_key=idempotency_key,
        )

        async def _iterate() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.aiter_bytes(chunk_size):
                    if chunk:
                        yield chunk
            finally:
                await response.aclose()

        return _iterate()

    def stream_ws(
        self,
        *,
        model: TTSModel | None = None,
        voice: str | None = None,
        cfg: float | None = None,
        voice_prompt: str | None = None,
    ) -> SpeechStreamHandle:
        """Open a realtime TTS WebSocket session.

        Use this when you need to synthesise many utterances over a
        single persistent connection (LLM-→-TTS pipelining, embedded
        widgets that make many calls per session). For one-shot
        synthesis the buffered :meth:`create` / chunked :meth:`stream`
        calls are simpler.

        Usage::

            async with hakim.audio.speech.stream_ws(
                model="hakim-fast-v1",
                voice="cmokbc2b1001pvu39wmj61b7h",
            ) as h:
                await h.send_speech({"input": "Hello, world."})
                async for ev in h:
                    if ev["type"] == "speech.audio":
                        speaker.write(ev["chunk"])
                    elif ev["type"] == "speech.done":
                        break

        Each ``speech.create`` is billed by character; the proxy
        flushes usage every 30s and on close.
        """

        options: SpeechStreamOptions = {}
        if model is not None:
            options["model"] = model
        if voice is not None:
            options["voice"] = voice
        if cfg is not None:
            options["cfg"] = cfg
        if voice_prompt is not None:
            options["voice_prompt"] = voice_prompt

        return SpeechStreamHandle(
            api_key=self._t.api_key,
            base_url=self._t.base_url,
            options=options,
        )


# Re-export these for annotation compatibility if consumers import from here.
__all__ = ["ResponseFormat", "SampleRate", "SpeechAPI", "SpeechStreamHandle", "TTSModel"]

_ = (Any,)  # silence "unused import" on older type-checkers.
