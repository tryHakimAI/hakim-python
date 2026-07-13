"""``voices`` — list presets + cloned voices, create / retrieve / delete clones."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast
from urllib.parse import quote

from .._transport import AsyncTransport
from .._types import (
    Voice,
    VoiceCreateRequest,
    VoiceDeletedResponse,
    VoicesListQuery,
    VoicesListResponse,
)
from .transcriptions import AudioInput, _to_httpx_file


class VoicesAPI:
    def __init__(self, transport: AsyncTransport) -> None:
        self._t = transport

    async def list(
        self,
        *,
        kind: str | None = None,
        language: str | None = None,
        q: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> VoicesListResponse:
        params: VoicesListQuery = {}
        if kind is not None:
            params["kind"] = kind  # type: ignore[typeddict-item]
        if language is not None:
            params["language"] = language  # type: ignore[typeddict-item]
        if q is not None:
            params["q"] = q
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        response = await self._t.request("GET", "v1/audio/voices", params=params)
        body = await response.aread()
        await response.aclose()
        import json as _json

        return cast(VoicesListResponse, _json.loads(body.decode("utf-8")))

    async def iter(
        self,
        *,
        kind: str | None = None,
        language: str | None = None,
        q: str | None = None,
    ) -> AsyncIterator[Voice]:
        """Walk every page via ``next_cursor``."""

        cursor: str | None = None
        while True:
            page = await self.list(kind=kind, language=language, q=q, cursor=cursor)
            for voice in page.get("data", []) or []:
                yield voice
            cursor = page.get("next_cursor")
            if not cursor:
                return

    async def retrieve(self, voice_id: str) -> Voice:
        response = await self._t.request(
            "GET", f"v1/audio/voices/{quote(voice_id, safe='')}"
        )
        body = await response.aread()
        await response.aclose()
        import json as _json

        return cast(Voice, _json.loads(body.decode("utf-8")))

    async def create(
        self,
        *,
        sample: AudioInput,
        name: str,
        language: str,
        consent_confirmed: bool,
        description: str | None = None,
        filename: str | None = None,
        idempotency_key: str | None = None,
    ) -> Voice:
        """Clone a voice.

        Returns immediately with a ``processing`` voice; poll
        :meth:`retrieve` (or subscribe to a webhook) until
        ``status == "ready"``.
        """

        form_file = _to_httpx_file(sample, filename)
        data: dict[str, Any] = {
            "name": name,
            "language": language,
            "consent_confirmed": "true" if consent_confirmed else "false",
        }
        if description is not None:
            data["description"] = description
        body_type = cast(VoiceCreateRequest, {})  # pure type check
        del body_type  # pragma: no cover

        response = await self._t.request(
            "POST",
            "v1/audio/voices",
            files=[("file", form_file)],
            data=data,
            idempotency_key=idempotency_key,
        )
        body = await response.aread()
        await response.aclose()
        import json as _json

        return cast(Voice, _json.loads(body.decode("utf-8")))

    async def delete(self, voice_id: str) -> VoiceDeletedResponse:
        response = await self._t.request(
            "DELETE", f"v1/audio/voices/{quote(voice_id, safe='')}"
        )
        body = await response.aread()
        await response.aclose()
        import json as _json

        return cast(VoiceDeletedResponse, _json.loads(body.decode("utf-8")))
