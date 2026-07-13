"""``jobs`` — list + retrieve background jobs (batch STT, voice clone)."""

from __future__ import annotations

import json as _json
from collections.abc import AsyncIterator
from typing import cast
from urllib.parse import quote

from ._transport import AsyncTransport
from ._types import Job, JobsListResponse, JobStatus, JobType


class JobsAPI:
    def __init__(self, transport: AsyncTransport) -> None:
        self._t = transport

    async def list(
        self,
        *,
        status: JobStatus | None = None,
        type: JobType | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> JobsListResponse:
        params: dict[str, str | int] = {}
        if status is not None:
            params["status"] = status
        if type is not None:
            params["type"] = type
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        response = await self._t.request("GET", "v1/jobs", params=params)
        return cast(JobsListResponse, _json.loads((await response.aread()).decode("utf-8")))

    async def retrieve(self, job_id: str) -> Job:
        response = await self._t.request(
            "GET", f"v1/jobs/{quote(job_id, safe='')}"
        )
        return cast(Job, _json.loads((await response.aread()).decode("utf-8")))

    async def iter(
        self,
        *,
        status: JobStatus | None = None,
        type: JobType | None = None,
    ) -> AsyncIterator[Job]:
        cursor: str | None = None
        while True:
            page = await self.list(status=status, type=type, cursor=cursor)
            for job in page.get("data", []) or []:
                yield job
            if not page.get("has_more") or not page.get("next_cursor"):
                return
            cursor = page.get("next_cursor")
