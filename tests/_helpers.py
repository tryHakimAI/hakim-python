"""Plain-function helpers shared across tests (not fixtures)."""

from __future__ import annotations

from typing import Any

import httpx


def json_response(status: int, body: Any) -> httpx.Response:
    return httpx.Response(
        status, headers={"content-type": "application/json"}, json=body
    )


def text_response(
    status: int, body: str, *, content_type: str = "text/plain"
) -> httpx.Response:
    return httpx.Response(
        status, headers={"content-type": content_type}, text=body
    )
