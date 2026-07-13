"""``webhooks`` — public ``/v1/webhooks`` CRUD + deliveries + signature helper.

Both SDKs verify the same ``Hakim-Signature`` header format::

    Hakim-Signature: t=<unix_seconds>,v1=<hex(hmac_sha256(secret, t + "." + raw_body))>

:func:`verify_webhook_signature` is a pure import — no :class:`Hakim`
instance needed — so a serverless receiver can ship without pulling in
``httpx``.
"""

from __future__ import annotations

import hashlib
import hmac
import json as _json
import re
import time
from collections.abc import AsyncIterator, Sequence
from typing import Literal, NamedTuple, cast
from urllib.parse import quote

from ._transport import AsyncTransport
from ._types import (
    Webhook,
    WebhookCreated,
    WebhookCreateRequest,
    WebhookDeliveriesListResponse,
    WebhookDelivery,
    WebhookDeliveryStatus,
    WebhookEventKey,
    WebhooksListResponse,
    WebhookUpdateRequest,
)


class WebhookDeleted(NamedTuple):
    object: Literal["webhook"]
    id: str
    deleted: bool


class WebhooksAPI:
    def __init__(self, transport: AsyncTransport) -> None:
        self._t = transport

    async def create(
        self,
        *,
        url: str,
        events: Sequence[WebhookEventKey],
        active: bool | None = None,
        idempotency_key: str | None = None,
    ) -> WebhookCreated:
        body: WebhookCreateRequest = {"url": url, "events": list(events)}
        if active is not None:
            body["active"] = active
        response = await self._t.request(
            "POST",
            "v1/webhooks",
            json=body,
            idempotency_key=idempotency_key,
        )
        return cast(WebhookCreated, _json.loads((await response.aread()).decode("utf-8")))

    async def list(self) -> WebhooksListResponse:
        response = await self._t.request("GET", "v1/webhooks")
        return cast(WebhooksListResponse, _json.loads((await response.aread()).decode("utf-8")))

    async def retrieve(self, webhook_id: str) -> Webhook:
        response = await self._t.request(
            "GET", f"v1/webhooks/{quote(webhook_id, safe='')}"
        )
        return cast(Webhook, _json.loads((await response.aread()).decode("utf-8")))

    async def update(
        self,
        webhook_id: str,
        *,
        url: str | None = None,
        events: Sequence[WebhookEventKey] | None = None,
        active: bool | None = None,
        idempotency_key: str | None = None,
    ) -> Webhook:
        body: WebhookUpdateRequest = {}
        if url is not None:
            body["url"] = url
        if events is not None:
            body["events"] = list(events)
        if active is not None:
            body["active"] = active
        response = await self._t.request(
            "PATCH",
            f"v1/webhooks/{quote(webhook_id, safe='')}",
            json=body,
            idempotency_key=idempotency_key,
        )
        return cast(Webhook, _json.loads((await response.aread()).decode("utf-8")))

    async def delete(self, webhook_id: str) -> WebhookDeleted:
        response = await self._t.request(
            "DELETE", f"v1/webhooks/{quote(webhook_id, safe='')}"
        )
        body = _json.loads((await response.aread()).decode("utf-8"))
        return WebhookDeleted(
            object=body.get("object", "webhook"),
            id=body.get("id", webhook_id),
            deleted=bool(body.get("deleted", False)),
        )

    async def list_deliveries(
        self,
        webhook_id: str,
        *,
        status: WebhookDeliveryStatus | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> WebhookDeliveriesListResponse:
        params: dict[str, str | int] = {}
        if status is not None:
            params["status"] = status
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        response = await self._t.request(
            "GET",
            f"v1/webhooks/{quote(webhook_id, safe='')}/deliveries",
            params=params,
        )
        return cast(
            WebhookDeliveriesListResponse,
            _json.loads((await response.aread()).decode("utf-8")),
        )

    async def iter_deliveries(
        self,
        webhook_id: str,
        *,
        status: WebhookDeliveryStatus | None = None,
    ) -> AsyncIterator[WebhookDelivery]:
        cursor: str | None = None
        while True:
            page = await self.list_deliveries(
                webhook_id, status=status, cursor=cursor
            )
            for delivery in page.get("data", []) or []:
                yield delivery
            if not page.get("has_more") or not page.get("next_cursor"):
                return
            cursor = page.get("next_cursor")


# ---------------------------------------------------------------------------
# verify_webhook_signature — pure helper, no network / no Hakim instance.
# ---------------------------------------------------------------------------

_SIGNATURE_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_DEFAULT_TOLERANCE_SECONDS = 300


VerifyReason = Literal[
    "malformed_header",
    "timestamp_out_of_tolerance",
    "signature_mismatch",
]


class VerifyResult(NamedTuple):
    valid: bool
    reason: VerifyReason | None = None


def verify_webhook_signature(
    *,
    secret: str,
    body: str,
    signature: str,
    tolerance_seconds: int = _DEFAULT_TOLERANCE_SECONDS,
    now: float | None = None,
) -> VerifyResult:
    """Validate a ``Hakim-Signature`` header.

    Returns ``VerifyResult(valid=True)`` on success, otherwise
    ``VerifyResult(valid=False, reason=...)`` with one of:
    ``malformed_header`` / ``timestamp_out_of_tolerance`` /
    ``signature_mismatch``.
    """

    parsed = _parse_signature_header(signature)
    if parsed is None:
        return VerifyResult(False, "malformed_header")
    ts, v1 = parsed

    if tolerance_seconds > 0:
        current = now if now is not None else time.time()
        if abs(current - ts) > tolerance_seconds:
            return VerifyResult(False, "timestamp_out_of_tolerance")

    expected = hmac.new(
        secret.encode("utf-8"),
        f"{ts}.{body}".encode(),
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected, v1.lower()):
        return VerifyResult(False, "signature_mismatch")
    return VerifyResult(True, None)


def _parse_signature_header(raw: str) -> tuple[int, str] | None:
    timestamp: int | None = None
    v1: str | None = None
    for part in (p.strip() for p in raw.split(",")):
        if part.startswith("t="):
            try:
                n = int(part[2:])
                if n > 0:
                    timestamp = n
            except ValueError:
                continue
        elif part.startswith("v1="):
            v1 = part[3:]
    if timestamp is None or v1 is None:
        return None
    if not _SIGNATURE_HEX_RE.match(v1):
        return None
    return timestamp, v1


__all__ = [
    "VerifyReason",
    "VerifyResult",
    "WebhookDeleted",
    "WebhooksAPI",
    "verify_webhook_signature",
]
