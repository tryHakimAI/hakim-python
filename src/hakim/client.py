"""Public client classes.

Two entry points:

* :class:`AsyncHakim` — async-first client. Every namespace method is
  an ``async def``. This is the recommended surface for modern Python
  codebases (FastAPI, asyncio servers, etc.).

* :class:`Hakim` — thin sync wrapper. Runs :class:`AsyncHakim` in a
  private thread-owned event loop so :meth:`Hakim.tts_create` etc. can
  be called from synchronous code without juggling ``asyncio.run``.

Both expose the same method surface. Mutating methods accept an
optional ``idempotency_key`` (auto-generated otherwise).
"""

from __future__ import annotations

import asyncio
import atexit
import os
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any, TypeVar

import httpx

from ._transport import AsyncTransport
from ._version import SDK_VERSION
from .audio import SpeechAPI, TranscriptionsAPI, TranslateAPI, VoicesAPI
from .chat import ChatCompletionsAPI
from .jobs import JobsAPI
from .settings import NotificationsAPI, SettingsAPI
from .usage import UsageAPI
from .webhooks import WebhooksAPI

_DEFAULT_BASE_URL = "https://api.tryhakim.ai"
_DEFAULT_TIMEOUT_S = 120.0
_DEFAULT_MAX_RETRIES = 2


class AsyncHakim:
    """Async client. See module docstring for usage."""

    version: str = SDK_VERSION

    audio: _AsyncAudioNamespace
    voices: VoicesAPI
    chat: _AsyncChatNamespace
    webhooks: WebhooksAPI
    jobs: JobsAPI
    usage: UsageAPI
    settings: SettingsAPI
    notifications: NotificationsAPI

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        user_agent_suffix: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("HAKIM_API_KEY") or os.environ.get(
            "HAKIM_API_TOKEN"
        )
        if not resolved_key:
            raise ValueError(
                "hakim SDK: missing API key. Pass `api_key=` or set HAKIM_API_KEY."
            )
        resolved_base = base_url or os.environ.get("HAKIM_BASE_URL") or _DEFAULT_BASE_URL

        self._transport = AsyncTransport(
            api_key=resolved_key,
            base_url=resolved_base,
            timeout_s=timeout_s,
            max_retries=max_retries,
            user_agent_suffix=user_agent_suffix,
            client=http_client,
        )

        voices = VoicesAPI(self._transport)
        self.voices = voices
        self.audio = _AsyncAudioNamespace(
            speech=SpeechAPI(self._transport),
            transcriptions=TranscriptionsAPI(self._transport),
            translate=TranslateAPI(self._transport),
            voices=voices,
        )
        self.chat = _AsyncChatNamespace(
            completions=ChatCompletionsAPI(self._transport),
        )
        self.webhooks = WebhooksAPI(self._transport)
        self.jobs = JobsAPI(self._transport)
        self.usage = UsageAPI(self._transport)
        self.settings = SettingsAPI(self._transport)
        self.notifications = NotificationsAPI(self._transport)

    async def aclose(self) -> None:
        await self._transport.aclose()

    async def __aenter__(self) -> AsyncHakim:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    # -------------------------------------------------------------------
    # Flat helpers (`tts_create`, `stt_create`, etc.).
    # -------------------------------------------------------------------

    async def tts_create(self, **kwargs: Any) -> bytes:
        return await self.audio.speech.create(**kwargs)

    def tts_stream(self, **kwargs: Any) -> Awaitable[AsyncIterator[bytes]]:
        return self.audio.speech.stream(**kwargs)

    async def stt_create(self, **kwargs: Any) -> Any:
        return await self.audio.transcriptions.create(**kwargs)

    async def voices_list(self, **kwargs: Any) -> Any:
        return await self.voices.list(**kwargs)

    async def voices_create(self, **kwargs: Any) -> Any:
        return await self.voices.create(**kwargs)

    async def usage_summary(self) -> Any:
        return await self.usage.summary()


class _AsyncAudioNamespace:
    __slots__ = ("speech", "transcriptions", "translate", "voices")

    def __init__(
        self,
        *,
        speech: SpeechAPI,
        transcriptions: TranscriptionsAPI,
        translate: TranslateAPI,
        voices: VoicesAPI,
    ) -> None:
        self.speech = speech
        self.transcriptions = transcriptions
        self.translate = translate
        self.voices = voices


class _AsyncChatNamespace:
    __slots__ = ("completions",)

    def __init__(self, *, completions: ChatCompletionsAPI) -> None:
        self.completions = completions


# ---------------------------------------------------------------------------
# Sync wrapper
# ---------------------------------------------------------------------------

T = TypeVar("T")


class _BackgroundLoop:
    """A thread-owned asyncio loop we can ``run_coroutine_threadsafe`` onto."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run, name="hakim-sdk-sync-loop", daemon=True
        )
        self._thread.start()
        atexit.register(self._shutdown)

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def call(self, coro: Awaitable[T]) -> T:
        # `run_coroutine_threadsafe` is typed on `Coroutine`, not `Awaitable`;
        # every caller here hands it a real coroutine. The concurrent.futures
        # Future's generic is T (the coroutine's return type).
        future: Any = asyncio.run_coroutine_threadsafe(coro, self._loop)  # type: ignore[arg-type]
        result: T = future.result()
        return result

    def iter_sync(self, async_iter: AsyncIterator[T]) -> Iterator[T]:
        """Drain an async iterator from sync code."""

        async def _next() -> tuple[bool, T | None]:
            try:
                return False, await async_iter.__anext__()
            except StopAsyncIteration:
                return True, None

        while True:
            done, value = self.call(_next())
            if done:
                return
            yield value  # type: ignore[misc]

    def _shutdown(self) -> None:
        if self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=1.0)
        if not self._loop.is_closed():
            self._loop.close()


class Hakim:
    """Synchronous façade over :class:`AsyncHakim`.

    Every call blocks the calling thread until the response arrives.
    Prefer :class:`AsyncHakim` in async codebases.
    """

    version: str = SDK_VERSION

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        user_agent_suffix: str | None = None,
    ) -> None:
        self._bg = _BackgroundLoop()
        self._async = self._bg.call(
            _make_async_client(
                api_key=api_key,
                base_url=base_url,
                timeout_s=timeout_s,
                max_retries=max_retries,
                user_agent_suffix=user_agent_suffix,
            )
        )
        self.audio = _SyncAudioNamespace(self._bg, self._async)
        self.voices = _SyncVoicesAPI(self._bg, self._async.voices)
        self.chat = _SyncChatNamespace(self._bg, self._async.chat.completions)
        self.webhooks = _SyncWebhooksAPI(self._bg, self._async.webhooks)
        self.jobs = _SyncJobsAPI(self._bg, self._async.jobs)
        self.usage = _SyncUsageAPI(self._bg, self._async.usage)
        self.settings = _SyncSettingsAPI(self._bg, self._async.settings)
        self.notifications = _SyncNotificationsAPI(self._bg, self._async.notifications)

    def close(self) -> None:
        self._bg.call(self._async.aclose())

    def __enter__(self) -> Hakim:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # Flat helpers.
    def tts_create(self, **kwargs: Any) -> bytes:
        return self._bg.call(self._async.tts_create(**kwargs))

    def stt_create(self, **kwargs: Any) -> Any:
        return self._bg.call(self._async.stt_create(**kwargs))

    def voices_list(self, **kwargs: Any) -> Any:
        return self._bg.call(self._async.voices_list(**kwargs))

    def voices_create(self, **kwargs: Any) -> Any:
        return self._bg.call(self._async.voices_create(**kwargs))

    def usage_summary(self) -> Any:
        return self._bg.call(self._async.usage_summary())


async def _make_async_client(
    *,
    api_key: str | None,
    base_url: str | None,
    timeout_s: float,
    max_retries: int,
    user_agent_suffix: str | None,
) -> AsyncHakim:
    """Factory coroutine so `Hakim.__init__` can `bg.call(...)` it.

    We build :class:`AsyncHakim` on the background loop so its httpx
    client is anchored to the loop that will drive it.
    """

    return AsyncHakim(
        api_key=api_key,
        base_url=base_url,
        timeout_s=timeout_s,
        max_retries=max_retries,
        user_agent_suffix=user_agent_suffix,
    )


class _SyncVoicesAPI:
    def __init__(self, bg: _BackgroundLoop, voices: VoicesAPI) -> None:
        self._bg = bg
        self._v = voices

    def list(self, **kwargs: Any) -> Any:
        return self._bg.call(self._v.list(**kwargs))

    def retrieve(self, voice_id: str) -> Any:
        return self._bg.call(self._v.retrieve(voice_id))

    def create(self, **kwargs: Any) -> Any:
        return self._bg.call(self._v.create(**kwargs))

    def delete(self, voice_id: str) -> Any:
        return self._bg.call(self._v.delete(voice_id))

    def iter(self, **kwargs: Any) -> Iterator[Any]:
        return self._bg.iter_sync(self._v.iter(**kwargs))


class _SyncSpeechAPI:
    def __init__(self, bg: _BackgroundLoop, speech: SpeechAPI) -> None:
        self._bg = bg
        self._s = speech

    def create(self, **kwargs: Any) -> bytes:
        return self._bg.call(self._s.create(**kwargs))

    def stream(self, **kwargs: Any) -> Iterator[bytes]:
        async_iter = self._bg.call(self._s.stream(**kwargs))
        return self._bg.iter_sync(async_iter)


class _SyncTranscriptionsAPI:
    def __init__(
        self, bg: _BackgroundLoop, transcriptions: TranscriptionsAPI
    ) -> None:
        self._bg = bg
        self._stt = transcriptions

    def create(self, **kwargs: Any) -> Any:
        return self._bg.call(self._stt.create(**kwargs))

    # NOTE: realtime stream stays async-only — you can't meaningfully
    # iterate a WebSocket from sync code without blocking the only
    # thread. Callers who need sync semantics can run their own loop.


class _SyncAudioNamespace:
    def __init__(self, bg: _BackgroundLoop, async_client: AsyncHakim) -> None:
        self.speech = _SyncSpeechAPI(bg, async_client.audio.speech)
        self.transcriptions = _SyncTranscriptionsAPI(bg, async_client.audio.transcriptions)
        # Translate is async-only — `stream_ws` returns a handle that's
        # meaningless without an event loop to drive it. Surfaced
        # through the sync façade verbatim so callers can drop into
        # `asyncio.run(...)` for their own loop if needed.
        self.translate = async_client.audio.translate
        self.voices = _SyncVoicesAPI(bg, async_client.audio.voices)


class _SyncChatCompletionsAPI:
    def __init__(self, bg: _BackgroundLoop, completions: ChatCompletionsAPI) -> None:
        self._bg = bg
        self._c = completions

    def create(self, **kwargs: Any) -> Any:
        return self._bg.call(self._c.create(**kwargs))

    def stream(self, **kwargs: Any) -> Iterator[Any]:
        async_iter = self._bg.call(self._c.stream(**kwargs))
        return self._bg.iter_sync(async_iter)


class _SyncChatNamespace:
    def __init__(self, bg: _BackgroundLoop, completions: ChatCompletionsAPI) -> None:
        self.completions = _SyncChatCompletionsAPI(bg, completions)


class _SyncWebhooksAPI:
    def __init__(self, bg: _BackgroundLoop, webhooks: WebhooksAPI) -> None:
        self._bg = bg
        self._w = webhooks

    def create(self, **kwargs: Any) -> Any:
        return self._bg.call(self._w.create(**kwargs))

    def list(self) -> Any:
        return self._bg.call(self._w.list())

    def retrieve(self, webhook_id: str) -> Any:
        return self._bg.call(self._w.retrieve(webhook_id))

    def update(self, webhook_id: str, **kwargs: Any) -> Any:
        return self._bg.call(self._w.update(webhook_id, **kwargs))

    def delete(self, webhook_id: str) -> Any:
        return self._bg.call(self._w.delete(webhook_id))

    def list_deliveries(self, webhook_id: str, **kwargs: Any) -> Any:
        return self._bg.call(self._w.list_deliveries(webhook_id, **kwargs))

    def iter_deliveries(self, webhook_id: str, **kwargs: Any) -> Iterator[Any]:
        return self._bg.iter_sync(self._w.iter_deliveries(webhook_id, **kwargs))


class _SyncJobsAPI:
    def __init__(self, bg: _BackgroundLoop, jobs: JobsAPI) -> None:
        self._bg = bg
        self._j = jobs

    def list(self, **kwargs: Any) -> Any:
        return self._bg.call(self._j.list(**kwargs))

    def retrieve(self, job_id: str) -> Any:
        return self._bg.call(self._j.retrieve(job_id))

    def iter(self, **kwargs: Any) -> Iterator[Any]:
        return self._bg.iter_sync(self._j.iter(**kwargs))


class _SyncUsageAPI:
    def __init__(self, bg: _BackgroundLoop, usage: UsageAPI) -> None:
        self._bg = bg
        self._u = usage

    def summary(self) -> Any:
        return self._bg.call(self._u.summary())

    def events(self, **kwargs: Any) -> Any:
        return self._bg.call(self._u.events(**kwargs))

    def iter_events(self, **kwargs: Any) -> Iterator[Any]:
        return self._bg.iter_sync(self._u.iter_events(**kwargs))


class _SyncSettingsAPI:
    def __init__(self, bg: _BackgroundLoop, settings: SettingsAPI) -> None:
        self._bg = bg
        self._s = settings

    def get_profile(self) -> Any:
        return self._bg.call(self._s.get_profile())

    def update_profile(self, patch: Any) -> Any:
        return self._bg.call(self._s.update_profile(patch))

    def get_organization(self) -> Any:
        return self._bg.call(self._s.get_organization())

    def update_organization(self, patch: Any) -> Any:
        return self._bg.call(self._s.update_organization(patch))


class _SyncNotificationsAPI:
    def __init__(self, bg: _BackgroundLoop, notifications: NotificationsAPI) -> None:
        self._bg = bg
        self._n = notifications

    def get(self) -> Any:
        return self._bg.call(self._n.get())

    def update(self, patch: Any) -> Any:
        return self._bg.call(self._n.update(patch))


# For advanced callers who want a callable-based injection point.
SyncCall = Callable[..., Any]
