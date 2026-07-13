"""``chat.completions`` — POST ``/v1/chat/completions``.

Two entry points, mirroring ``audio.speech``:

* :meth:`ChatCompletionsAPI.create` — buffered. Forces
  ``stream: false`` and returns a :class:`ChatCompletionResponse`
  dict.
* :meth:`ChatCompletionsAPI.stream` — SSE. Returns an
  ``AsyncIterator`` of :class:`ChatCompletionChunk` dicts. The
  final chunk carries the rolled-up ``usage`` block — same posture
  as OpenAI.

Reasoning policy: off by default · opt-in for
non-stream via ``reasoning={"enabled": True}`` · forbidden for
streaming (the route rejects the combo at the schema layer with a
400). We don't pre-validate here so the server stays the single
source of truth.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, cast

from .._transport import AsyncTransport
from .._types import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatModel,
    ChatReasoningOption,
)
from ..errors import HakimApiErrorPayload, HakimError, error_from_payload


class ChatCompletionsAPI:
    """OpenAI-compatible chat completions for HKM LLM 1."""

    def __init__(self, transport: AsyncTransport) -> None:
        self._t = transport

    async def create(
        self,
        *,
        model: ChatModel,
        messages: list[ChatMessage],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: str | list[str] | None = None,
        user: str | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        seed: int | None = None,
        reasoning: ChatReasoningOption | None = None,
        idempotency_key: str | None = None,
    ) -> ChatCompletionResponse:
        """Non-streaming chat completion.

        Returns the OpenAI-shaped response dict. The Hakim
        observability block is embedded under ``hakim_usage`` and
        also mirrored on the ``x-hakim-*`` response headers.
        """

        body = self._build_body(
            model=model,
            messages=messages,
            stream=False,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            user=user,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            seed=seed,
            reasoning=reasoning,
        )

        response = await self._t.request(
            "POST",
            "v1/chat/completions",
            json=body,
            idempotency_key=idempotency_key,
        )
        return cast(ChatCompletionResponse, response.json())

    async def stream(
        self,
        *,
        model: ChatModel,
        messages: list[ChatMessage],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: str | list[str] | None = None,
        user: str | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        seed: int | None = None,
        idempotency_key: str | None = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Streaming chat completion (SSE).

        Forces ``stream: True`` on the body. Reasoning is **not**
        accepted on this surface — the route rejects the combo
        with a 400; the parameter is intentionally absent from
        this method's signature so a misuse fails at edit time.
        """

        body = self._build_body(
            model=model,
            messages=messages,
            stream=True,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            user=user,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            seed=seed,
            reasoning=None,
        )

        response = await self._t.request(
            "POST",
            "v1/chat/completions",
            json=body,
            stream=True,
            accept="text/event-stream",
            idempotency_key=idempotency_key,
        )

        request_id = response.headers.get("x-request-id")
        return _iterate_sse(response, request_id=request_id)

    @staticmethod
    def _build_body(
        *,
        model: ChatModel,
        messages: list[ChatMessage],
        stream: bool,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
        stop: str | list[str] | None,
        user: str | None,
        presence_penalty: float | None,
        frequency_penalty: float | None,
        seed: int | None,
        reasoning: ChatReasoningOption | None,
    ) -> ChatCompletionRequest:
        body: ChatCompletionRequest = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if stop is not None:
            body["stop"] = stop
        if user is not None:
            body["user"] = user
        if presence_penalty is not None:
            body["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            body["frequency_penalty"] = frequency_penalty
        if seed is not None:
            body["seed"] = seed
        if reasoning is not None:
            body["reasoning"] = reasoning
        return body


# ---------------------------------------------------------------------------
# SSE parser
# ---------------------------------------------------------------------------


async def _iterate_sse(
    response: Any, *, request_id: str | None
) -> AsyncIterator[ChatCompletionChunk]:
    """Yield parsed chunks from an SSE response.

    Tolerates partial frames split across read() boundaries, skips
    heartbeat comments (``: …``), terminates on ``data: [DONE]``,
    and surfaces ``event: error`` envelopes as :class:`HakimError`.
    """

    buffered = ""
    try:
        async for raw in response.aiter_text():
            if not raw:
                continue
            buffered += raw
            while True:
                sep = buffered.find("\n\n")
                if sep == -1:
                    break
                frame = buffered[:sep]
                buffered = buffered[sep + 2 :]
                done, chunk = _parse_sse_frame(frame, request_id=request_id)
                if done:
                    return
                if chunk is not None:
                    yield chunk
        if buffered.strip():
            done, chunk = _parse_sse_frame(buffered, request_id=request_id)
            if done:
                return
            if chunk is not None:
                yield chunk
    finally:
        await response.aclose()


def _parse_sse_frame(
    frame: str, *, request_id: str | None
) -> tuple[bool, ChatCompletionChunk | None]:
    """Parse one SSE frame.

    Returns ``(done, chunk)``:

    * ``done=True``  — frame was ``data: [DONE]``; outer loop should
      stop iterating without yielding.
    * ``chunk=None`` — frame was a heartbeat / empty / unrecognised
      event; outer loop should skip and read the next frame.
    * Otherwise, ``chunk`` is the parsed chunk to yield.

    Raises :class:`HakimError` for ``event: error`` envelopes or
    JSON-unparseable data frames.
    """
    event_type: str | None = None
    data_lines: list[str] = []
    for raw_line in frame.split("\n"):
        line = raw_line.rstrip("\r")
        if not line:
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_type = line[6:].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip(" "))
            continue

    if not data_lines:
        return False, None
    payload = "\n".join(data_lines)
    if payload == "[DONE]":
        return True, None

    if event_type == "error":
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as err:
            raise HakimError(
                "hakim SDK: streaming error envelope was not JSON-parseable "
                f"({payload[:200]}).",
                type="service_unavailable",
                code="upstream_error",
                status=500,
                request_id=request_id,
            ) from err
        body = parsed.get("error") if isinstance(parsed, dict) else None
        if isinstance(body, dict):
            raise error_from_payload(
                cast(HakimApiErrorPayload, body),
                status=500,
                request_id=request_id,
            )
        raise HakimError(
            "hakim SDK: streaming error envelope missing `error` field.",
            type="service_unavailable",
            code="upstream_error",
            status=500,
            request_id=request_id,
        )

    try:
        return False, cast(ChatCompletionChunk, json.loads(payload))
    except json.JSONDecodeError as err:
        raise HakimError(
            f"hakim SDK: failed to parse SSE chunk JSON: {err}",
            type="api_error",
            code="malformed_sse_chunk",
            status=500,
            request_id=request_id,
        ) from err
