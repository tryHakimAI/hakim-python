"""Hakim — official Python SDK for the Hakim voice AI API.

Quickstart (async):

    from hakim import AsyncHakim

    async with AsyncHakim(api_key=...) as hakim:
        audio = await hakim.tts_create(
            model="hakim-fast-v1",
            input="مرحبا بالعالم",
            voice="omar",
        )

Quickstart (sync):

    from hakim import Hakim

    with Hakim(api_key=...) as hakim:
        audio = hakim.tts_create(
            model="hakim-fast-v1", input="مرحبا بالعالم", voice="omar",
        )

Every namespace from the Node SDK (`audio.speech`, `audio.transcriptions`,
`voices`, `webhooks`, `jobs`, `usage`) is available on both clients.
"""

from ._version import SDK_NAME, SDK_VERSION
from .client import AsyncHakim, Hakim
from .errors import (
    APIError,
    AuthenticationError,
    ConnectionError,
    HakimError,
    IdempotencyConflictError,
    InvalidRequestError,
    NotFoundError,
    PermissionError,
    QuotaExceededError,
    RateLimitError,
    ServiceUnavailableError,
)
from .webhooks import VerifyResult, verify_webhook_signature

__all__ = [
    "SDK_NAME",
    "SDK_VERSION",
    "APIError",
    "AsyncHakim",
    "AuthenticationError",
    "ConnectionError",
    "Hakim",
    "HakimError",
    "IdempotencyConflictError",
    "InvalidRequestError",
    "NotFoundError",
    "PermissionError",
    "QuotaExceededError",
    "RateLimitError",
    "ServiceUnavailableError",
    "VerifyResult",
    "verify_webhook_signature",
]

__version__ = SDK_VERSION
