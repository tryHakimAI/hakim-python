"""``usage`` — current-period summary + raw per-request events."""

from __future__ import annotations

import json as _json
from collections.abc import AsyncIterator
from typing import cast

from ._transport import AsyncTransport
from ._types import (
    LimitsSnapshot,
    UsageEvent,
    UsageEventDetail,
    UsageEventsList,
    UsageKind,
    UsageSummary,
)


class UsageAPI:
    def __init__(self, transport: AsyncTransport) -> None:
        self._t = transport

    async def summary(self) -> UsageSummary:
        response = await self._t.request("GET", "v1/usage/summary")
        return cast(UsageSummary, _json.loads((await response.aread()).decode("utf-8")))

    async def limits(self) -> LimitsSnapshot:
        """Point-in-time snapshot of every limit the org is subject to.

        Maps to ``GET /v1/limits``. Stable enough for a dashboard tile
        but cheap enough to poll every 30 s. Returns a
        :class:`LimitsSnapshot` carrying plan, billing period, credits,
        concurrency, and rate-limit blocks.
        """

        response = await self._t.request("GET", "v1/limits")
        return cast(
            LimitsSnapshot, _json.loads((await response.aread()).decode("utf-8"))
        )

    async def event(self, event_id: str) -> UsageEventDetail:
        """Look up a single usage event by id.

        The typical use is dereferencing the ``x-request-id`` scraped
        from a TTS / STT response into the row's metadata
        (``credits``, ``cost_usd``, ``model``). Maps to
        ``GET /v1/usage/events/:id``.
        """

        response = await self._t.request("GET", f"v1/usage/events/{event_id}")
        return cast(
            UsageEventDetail, _json.loads((await response.aread()).decode("utf-8"))
        )

    async def events(
        self,
        *,
        kind: UsageKind | None = None,
        from_: str | None = None,
        to: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> UsageEventsList:
        """Raw per-request usage events.

        Note: ``from_`` is serialized as ``from`` on the wire
        (Python reserves ``from`` as a keyword).
        """

        params: dict[str, str | int] = {}
        if kind is not None:
            params["kind"] = kind
        if from_ is not None:
            params["from"] = from_
        if to is not None:
            params["to"] = to
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        response = await self._t.request("GET", "v1/usage/events", params=params)
        return cast(
            UsageEventsList, _json.loads((await response.aread()).decode("utf-8"))
        )

    async def iter_events(
        self,
        *,
        kind: UsageKind | None = None,
        from_: str | None = None,
        to: str | None = None,
    ) -> AsyncIterator[UsageEvent]:
        cursor: str | None = None
        while True:
            page = await self.events(kind=kind, from_=from_, to=to, cursor=cursor)
            for evt in page.get("data", []) or []:
                yield evt
            if not page.get("has_more") or not page.get("next_cursor"):
                return
            cursor = page.get("next_cursor")
