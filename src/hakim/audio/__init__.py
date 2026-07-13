"""Audio namespaces: speech (TTS) + transcriptions (STT) + translate + voices."""

from .speech import SpeechAPI
from .speech_stream_ws import SpeechStreamHandle
from .transcriptions import TranscriptionsAPI, TranscriptionStreamHandle
from .translate import TranslateAPI
from .translate_stream_ws import TranslateStreamHandle
from .voices import VoicesAPI

__all__ = [
    "SpeechAPI",
    "SpeechStreamHandle",
    "TranscriptionStreamHandle",
    "TranscriptionsAPI",
    "TranslateAPI",
    "TranslateStreamHandle",
    "VoicesAPI",
]
