"""``settings`` + ``notifications`` — M2 follow-up SDK surface.

Mirrors the Node SDK's ``client.settings.*`` / ``client.notifications.*``
helpers. Both lean on the public ``/v1/settings/*`` and
``/v1/notifications`` endpoints exposed by the API service.

* ``settings.get_profile`` / ``update_profile`` — the API-key creator's
  profile (name, locale, timezone, marketing opt-in).

* ``settings.get_organization`` / ``update_organization`` — public-ish
  org settings (name, slug, billing email, default locale). Mutations
  require the key's creator to still be an owner/admin at call time.

* ``notifications.get`` / ``update`` — transactional-email toggles for
  the API-key creator user.

All responses are ``TypedDict`` shapes — same philosophy as the rest
of the SDK: the server does runtime validation via zod, the client
decodes + casts.
"""

from __future__ import annotations

import json as _json
from typing import cast

from ._transport import AsyncTransport
from ._types import (
    NotificationPreferences,
    NotificationPreferencesUpdateRequest,
    OrganizationSettings,
    OrganizationSettingsUpdateRequest,
    Profile,
    ProfileUpdateRequest,
)


class SettingsAPI:
    def __init__(self, transport: AsyncTransport) -> None:
        self._t = transport

    # -- profile -----------------------------------------------------
    async def get_profile(self) -> Profile:
        response = await self._t.request("GET", "v1/settings/profile")
        return cast(Profile, _json.loads((await response.aread()).decode("utf-8")))

    async def update_profile(self, patch: ProfileUpdateRequest) -> Profile:
        response = await self._t.request(
            "PATCH",
            "v1/settings/profile",
            json=dict(patch),
        )
        return cast(Profile, _json.loads((await response.aread()).decode("utf-8")))

    # -- organization ------------------------------------------------
    async def get_organization(self) -> OrganizationSettings:
        response = await self._t.request("GET", "v1/settings/organization")
        return cast(
            OrganizationSettings,
            _json.loads((await response.aread()).decode("utf-8")),
        )

    async def update_organization(
        self, patch: OrganizationSettingsUpdateRequest
    ) -> OrganizationSettings:
        response = await self._t.request(
            "PATCH",
            "v1/settings/organization",
            json=dict(patch),
        )
        return cast(
            OrganizationSettings,
            _json.loads((await response.aread()).decode("utf-8")),
        )


class NotificationsAPI:
    def __init__(self, transport: AsyncTransport) -> None:
        self._t = transport

    async def get(self) -> NotificationPreferences:
        response = await self._t.request("GET", "v1/notifications")
        return cast(
            NotificationPreferences,
            _json.loads((await response.aread()).decode("utf-8")),
        )

    async def update(
        self, patch: NotificationPreferencesUpdateRequest
    ) -> NotificationPreferences:
        response = await self._t.request(
            "PATCH",
            "v1/notifications",
            json=dict(patch),
        )
        return cast(
            NotificationPreferences,
            _json.loads((await response.aread()).decode("utf-8")),
        )
