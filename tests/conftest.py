"""Shared test fixtures — httpx MockTransport helpers."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from hakim import AsyncHakim


@pytest.fixture
def build_mock_client() -> Callable[[Callable[[httpx.Request], httpx.Response]], AsyncHakim]:
    """Return a factory that builds an ``AsyncHakim`` backed by a stub."""

    def _make(handler: Callable[[httpx.Request], httpx.Response]) -> AsyncHakim:
        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        return AsyncHakim(api_key="hk_test_x", http_client=client)

    return _make
