"""``audio.translate`` — realtime speech translation (STT → LLM → TTS).

Single entry point :meth:`TranslateAPI.stream_ws` opens a long-lived
WebSocket session at ``WSS /v1/audio/translate/stream`` and returns
a :class:`TranslateStreamHandle` you can use to push source audio and
consume target captions + synthesised audio.

There is no batch counterpart in this namespace: a batch translate
flow is recoverable from existing
:meth:`audio.transcriptions.create` + :meth:`chat.completions.create`
+ :meth:`audio.speech.create` chains, while the realtime pipeline
relies on the proxy's three-upstream orchestration.
"""

from __future__ import annotations

from .._transport import AsyncTransport
from .._types import TranslateStreamOptions
from .translate_stream_ws import TranslateStreamHandle


class TranslateAPI:
    def __init__(self, transport: AsyncTransport) -> None:
        self._t = transport

    def stream_ws(
        self,
        *,
        target_language: str | None = None,
        source_language: str | None = None,
        voice: str | None = None,
        gender: str | None = None,
        model_stt: str | None = None,
        model_llm: str | None = None,
        model_tts: str | None = None,
        cfg: float | None = None,
        input_audio_format: str | None = None,
        input_sample_rate: int | None = None,
        partials: bool | None = None,
        system_prompt: str | None = None,
    ) -> TranslateStreamHandle:
        """Open a realtime translate WebSocket session.

        Usage::

            async with hakim.audio.translate.stream_ws(
                target_language="en",
                gender="female",
            ) as session:
                # Push source audio from a mic capture.
                while chunk := mic.read(2048):
                    await session.send_audio(chunk)
                async for ev in session:
                    if ev["type"] == "translation.done":
                        print("translation:", ev["text"])
                    elif ev["type"] == "speech.audio":
                        speaker.write(ev["chunk"])
                    elif ev["type"] == "error" and ev.get("fatal"):
                        raise RuntimeError(ev["message"])

        Each utterance is billed across three modalities (STT seconds
        + LLM tokens + TTS characters); the proxy flushes per-modality
        ``UsageEvent`` rows and emits a ``session.usage`` rollup
        every 30s and on close.
        """

        options: TranslateStreamOptions = {}
        if target_language is not None:
            options["target_language"] = target_language
        if source_language is not None:
            options["source_language"] = source_language
        if voice is not None:
            options["voice"] = voice
        if gender is not None:
            options["gender"] = gender  # type: ignore[typeddict-item]
        if model_stt is not None:
            options["model_stt"] = model_stt  # type: ignore[typeddict-item]
        if model_llm is not None:
            options["model_llm"] = model_llm  # type: ignore[typeddict-item]
        if model_tts is not None:
            options["model_tts"] = model_tts  # type: ignore[typeddict-item]
        if cfg is not None:
            options["cfg"] = cfg
        if input_audio_format is not None:
            options["input_audio_format"] = input_audio_format  # type: ignore[typeddict-item]
        if input_sample_rate is not None:
            options["input_sample_rate"] = input_sample_rate  # type: ignore[typeddict-item]
        if partials is not None:
            options["partials"] = partials
        if system_prompt is not None:
            options["system_prompt"] = system_prompt

        return TranslateStreamHandle(
            api_key=self._t.api_key,
            base_url=self._t.base_url,
            options=options,
        )


__all__ = ["TranslateAPI", "TranslateStreamHandle"]
