# hakim — official Python SDK

[![PyPI version](https://img.shields.io/pypi/v/hakim.svg)](https://pypi.org/project/hakim/)
[![Python](https://img.shields.io/pypi/pyversions/hakim.svg)](https://pypi.org/project/hakim/)

Official Python SDK for [Hakim](https://tryhakim.ai) — Arabic-first TTS,
STT, voice cloning, webhooks, and batch jobs. Mirrors the surface of the
Node SDK (`@tryhakim/voice`).

## Install

```bash
pip install hakim
```

## Quick start (async)

```python
import asyncio
from hakim import AsyncHakim

async def main() -> None:
    async with AsyncHakim(api_key="hk_...") as hakim:
        audio = await hakim.audio.speech.create(
            model="hakim-fast-v1",
            input="مرحبا بالعالم",
            voice="omar",
        )
        with open("out.mp3", "wb") as f:
            f.write(audio)

asyncio.run(main())
```

## Quick start (sync)

```python
from hakim import Hakim

with Hakim(api_key="hk_...") as hakim:
    audio = hakim.tts_create(
        model="hakim-fast-v1",
        input="مرحبا بالعالم",
        voice="omar",
    )
```

## Chat completions

OpenAI-compatible chat API backed by `hakim-chat-v1` (HKM LLM 1).
Drop-in for any code that targets the OpenAI Chat Completions
reference — swap the base URL and key and you're done.

```python
async with AsyncHakim(api_key="hk_...") as hakim:
    # Non-streaming.
    completion = await hakim.chat.completions.create(
        model="hakim-chat-v1",
        messages=[
            {"role": "system", "content": "أنت مساعد عربي مفيد."},
            {"role": "user", "content": "اكتب قصيدة قصيرة عن البحر."},
        ],
        temperature=0.7,
    )
    print(completion["choices"][0]["message"]["content"])

    # Streaming (SSE). The final chunk carries the rolled-up `usage`.
    stream = await hakim.chat.completions.stream(
        model="hakim-chat-v1",
        messages=[{"role": "user", "content": "مرحبا!"}],
    )
    async for chunk in stream:
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            print(delta["content"], end="", flush=True)
```

### Reasoning / chain-of-thought

`hakim-chat-v1` is a thinking-capable model. CoT is **off by
default** — it adds 10–50× latency and burns completion tokens
that don't reach your UI in time. Opt in on **non-streaming**
requests only:

```python
completion = await hakim.chat.completions.create(
    model="hakim-chat-v1",
    messages=[{"role": "user", "content": "حلّ ٢٤ × ١٧ خطوة بخطوة."}],
    reasoning={"enabled": True},
)
print(completion["choices"][0]["message"]["reasoning"])  # chain-of-thought
print(completion["choices"][0]["message"]["content"])     # final answer
```

Streaming requests with `reasoning={"enabled": True}` are rejected
with a `400 invalid_request_error` — real-time agents cannot afford
the latency cost. The `stream()` method intentionally omits the
`reasoning` parameter so a misuse fails at edit time rather than
on the wire.

## Streaming TTS

```python
async with AsyncHakim(api_key="hk_...") as hakim:
    stream = await hakim.audio.speech.stream(
        model="hakim-fast-v1",
        input="salam",
        voice="omar",
    )
    async for chunk in stream:
        write_to_speaker(chunk)
```

## Speech-to-text (sync upload)

```python
async with AsyncHakim(api_key="hk_...") as hakim:
    result = await hakim.audio.transcriptions.create(
        file=open("clip.wav", "rb"),
        model="hakim-arab-v2",
        response_format="json",
    )
    print(result["text"])
```

## Speech-to-text (realtime)

```python
async with AsyncHakim(api_key="hk_...") as hakim:
    async with hakim.audio.transcriptions.stream(
        model="hakim-arab-v2",
        language="ar",
        sample_rate=16000,
    ) as handle:
        async def pump_audio() -> None:
            async for chunk in mic_iter():
                await handle.send_audio(chunk)
            await handle.commit()

        asyncio.create_task(pump_audio())

        async for event in handle:
            if event["type"] == "partial":
                print("~", event["text"])
            elif event["type"] == "final":
                print("✔", event["text"])
```

## Voice cloning

```python
with open("my_voice.wav", "rb") as f:
    voice = await hakim.voices.create(
        sample=f,
        name="My Voice",
        language="ar",
        consent_confirmed=True,
    )

while voice["status"] == "processing":
    await asyncio.sleep(2.0)
    voice = await hakim.voices.retrieve(voice["id"])
```

## Webhooks

```python
from hakim import verify_webhook_signature

def handle_webhook(request) -> None:
    body = request.body
    result = verify_webhook_signature(
        secret="whsec_...",
        body=body,
        signature=request.headers["hakim-signature"],
    )
    if not result.valid:
        raise ValueError(f"bad signature: {result.reason}")
    event = json.loads(body)
    ...
```

## Errors

Every HTTP-level failure raises a :class:`HakimError` subclass you can
branch on:

```python
from hakim import Hakim, RateLimitError, HakimError

try:
    hakim.tts_create(...)
except RateLimitError as err:
    retry_after = (err.retry_after_ms or 1000) / 1000
    time.sleep(retry_after)
except HakimError as err:
    print(err.status, err.code, err.request_id)
```

## Observability

Every request auto-attaches:

- `Authorization: Bearer <api_key>`
- `User-Agent: hakim-python/<version> (python/<py_version>; <platform>)`
- `X-Request-Id: sdk-<uuid>` (echoed back by the server — attach to your
  logs for end-to-end tracing).
- `Idempotency-Key: <uuid>` on mutating JSON calls (auto-generated;
  override via `idempotency_key=` when you need deterministic keys).

Transient failures (5xx, 429, 408, 425, network errors) are retried
with exponential back-off (±25% jitter) up to `max_retries` (default
2). `Retry-After` is honoured when the server sends it.

## Development

```bash
pip install -e '.[dev]'
ruff check src tests
mypy src
pytest
```
