"""Smoke tests for the Python SDK.

Assert the package imports cleanly and exposes its documented public
surface.
"""

from __future__ import annotations

import hakim
from hakim import Hakim, HakimError


def test_package_version_is_semver() -> None:
    assert isinstance(hakim.__version__, str)
    parts = hakim.__version__.split(".")
    assert len(parts) >= 2
    assert all(p.isdigit() for p in parts[:2])


def test_public_surface_exposes_client_and_errors() -> None:
    assert Hakim is hakim.Hakim
    assert HakimError is hakim.HakimError
    assert issubclass(HakimError, Exception)
