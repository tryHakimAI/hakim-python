"""Tests for `chat.completions` — buffered + streaming + reasoning."""

from __future__ import annotations

import json

import httpx
import pytest

from hakim import HakimError, ServiceUnavailableError

from ._helpers import json_response

SAMPLE_BODY = {
    "id": "chatcmpl-abc",
    "object": "chat.completion",
    "created": 1748400000,
    "model": "hakim-chat-v1",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Marhaba!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}


# ---------------------------------------------------------------------------
# create()
# ---------------------------------------------------------------------------


async def test_chat_create_returns_response_body(build_mock_client) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return json_response(200, SAMPLE_BODY)

    async with build_mock_client(handler) as hakim:
        result = await hakim.chat.completions.create(
            model="hakim-chat-v1",
            messages=[{"role": "user", "content": "Hi"}],
        )

    assert captured["method"] == "POST"
    assert captured["path"] == "/v1/chat/completions"
    assert captured["body"] == {
        "model": "hakim-chat-v1",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
    }
    assert result["id"] == "chatcmpl-abc"
    assert result["choices"][0]["message"]["content"] == "Marhaba!"
    assert result["usage"]["total_tokens"] == 8


async def test_chat_create_forwards_reasoning_opt_in(build_mock_client) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return json_response(
            200,
            {
                **SAMPLE_BODY,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Marhaba!",
                            "reasoning": "User greeted me, I greet back politely.",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    async with build_mock_client(handler) as hakim:
        result = await hakim.chat.completions.create(
            model="hakim-chat-v1",
            messages=[{"role": "user", "content": "Hi"}],
            reasoning={"enabled": True},
        )

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["reasoning"] == {"enabled": True}
    assert body["stream"] is False
    assert result["choices"][0]["message"]["reasoning"].startswith("User greeted")
    assert result["choices"][0]["message"]["content"] == "Marhaba!"


async def test_chat_create_forwards_optional_sampling_fields(build_mock_client) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return json_response(200, SAMPLE_BODY)

    async with build_mock_client(handler) as hakim:
        await hakim.chat.completions.create(
            model="hakim-chat-v1",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7,
            top_p=0.9,
            max_tokens=128,
            stop=["\n\n"],
            user="user-42",
            presence_penalty=0.1,
            frequency_penalty=0.2,
            seed=1234,
            idempotency_key="my-key",
        )

    body = captured["body"]
    assert isinstance(body, dict)
    assert body == {
        "model": "hakim-chat-v1",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 128,
        "stop": ["\n\n"],
        "user": "user-42",
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "seed": 1234,
    }


# ---------------------------------------------------------------------------
# stream()
# ---------------------------------------------------------------------------


def _sse_chunk(content: str | None = None, finish: str | None = None) -> str:
    delta: dict[str, object] = {}
    if content is not None:
        delta["content"] = content
    chunk = {
        "id": "chatcmpl-abc",
        "object": "chat.completion.chunk",
        "created": 1748400000,
        "model": "hakim-chat-v1",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


async def test_chat_stream_forces_stream_true_and_yields_chunks(build_mock_client) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}

    body = (
        ": heartbeat\n\n"
        + _sse_chunk("Mar")
        + _sse_chunk("haba")
        + _sse_chunk("!", finish="stop")
        + "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body,
        )

    async with build_mock_client(handler) as hakim:
        iterator = await hakim.chat.completions.stream(
            model="hakim-chat-v1",
            messages=[{"role": "user", "content": "Hi"}],
        )
        collected: list[str] = []
        async for chunk in iterator:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                collected.append(delta["content"])

    req_body = captured["body"]
    assert isinstance(req_body, dict)
    assert req_body["stream"] is True
    assert "reasoning" not in req_body
    assert "".join(collected) == "Marhaba!"


async def test_chat_stream_surfaces_event_error_envelope(build_mock_client) -> None:  # type: ignore[no-untyped-def]
    err_frame = (
        "event: error\n"
        + "data: "
        + json.dumps(
            {
                "error": {
                    "type": "service_unavailable",
                    "code": "upstream_overloaded",
                    "message": "Together returned 503 mid-stream",
                }
            }
        )
        + "\n\n"
    )

    body = _sse_chunk("Hi") + err_frame + "data: [DONE]\n\n"

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body,
        )

    async with build_mock_client(handler) as hakim:
        iterator = await hakim.chat.completions.stream(
            model="hakim-chat-v1",
            messages=[{"role": "user", "content": "Hi"}],
        )
        with pytest.raises(ServiceUnavailableError):
            async for _chunk in iterator:
                pass


async def test_chat_stream_raises_on_malformed_data_frame(build_mock_client) -> None:  # type: ignore[no-untyped-def]
    body = "data: {not-json\n\n"

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body,
        )

    async with build_mock_client(handler) as hakim:
        iterator = await hakim.chat.completions.stream(
            model="hakim-chat-v1",
            messages=[{"role": "user", "content": "Hi"}],
        )
        with pytest.raises(HakimError):
            async for _chunk in iterator:
                pass
