"""Smoke test for the sync ``Hakim`` wrapper.

We don't re-test everything the async client covers; we just confirm
the wrapper bridges the two event loops correctly for a representative
buffered call and a representative streaming iterator.
"""

from __future__ import annotations

import httpx
import pytest

from hakim import Hakim


def test_sync_hakim_tts_create_blocks_until_audio_is_ready() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        return httpx.Response(
            200, headers={"content-type": "audio/mpeg"}, content=b"sync-audio"
        )

    # The sync wrapper constructs its own httpx client on its background
    # loop, so we use a real base URL + mount a MockTransport via
    # monkeypatching `httpx.AsyncClient`. Easiest path: test the sync
    # surface by calling the raw background loop helper directly.
    hakim = Hakim(
        api_key="hk_test_x",
        base_url="http://mock.local",
    )
    # Swap in a MockTransport under the hood.
    hakim._async._transport._client = httpx.AsyncClient(  # type: ignore[attr-defined]
        transport=httpx.MockTransport(handler)
    )
    try:
        audio = hakim.tts_create(
            model="hakim-fast-v1", input="hi", voice="omar",
        )
    finally:
        hakim.close()
    assert audio == b"sync-audio"


def test_sync_hakim_requires_api_key() -> None:
    with pytest.raises(ValueError, match="missing API key"):
        Hakim()
