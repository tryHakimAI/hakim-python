"""Typed request / response shapes mirroring `@hakim/voice`.

We use :mod:`typing.TypedDict` (not pydantic) to match the Node SDK's
"types are compile-time, the wire is trusted JSON" philosophy. Runtime
validation happens server-side via zod; the SDK just decodes + casts.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

# ---------------------------------------------------------------------------
# Audio — speech (TTS)
# ---------------------------------------------------------------------------

# Public TTS model identifiers.
#
#   - ``hakim-fast-v1``  — current canonical id; recommended for new code.
#   - ``hakim-flash-v1`` — legacy id from the v1.0.0 launch. Still
#     accepted by the API forever, but normalised to ``hakim-fast-v1``
#     in metrics and audit logs.
#
# Use :data:`DEFAULT_TTS_MODEL` for new code.
TTSModel = Literal[
    "hakim-fast-v1",
    "hakim-flash-v1",
]

DEFAULT_TTS_MODEL: TTSModel = "hakim-fast-v1"

ResponseFormat = Literal["mp3", "wav", "pcm16", "flac"]

SampleRate = Literal[16000, 22050, 24000, 44100, 48000]


class SpeechRequest(TypedDict, total=False):
    model: TTSModel
    input: str
    voice: str
    response_format: ResponseFormat
    sample_rate: SampleRate
    speed: float
    seed: int
    stream: bool


# ---------------------------------------------------------------------------
# Audio — transcription (STT)
# ---------------------------------------------------------------------------

# Public STT model identifier.
#
# ``hakim-arab-v2`` is the only accepted id — the Arabic-first
# acoustic profile that backs every transcription path (batch
# ``POST /v1/audio/transcriptions`` and realtime
# ``WSS /v1/audio/transcriptions/stream``).
#
# Use :data:`DEFAULT_STT_MODEL` for new code.
STTModel = Literal["hakim-arab-v2"]

DEFAULT_STT_MODEL: STTModel = "hakim-arab-v2"

BaseLanguageCode = Literal[
    "ar", "en", "fr", "es", "de", "it", "pt", "ja", "zh", "ko", "tr", "ru", "auto"
]

ArabicDialectCode = Literal[
    "ar-EG", "ar-SA", "ar-AE", "ar-MA", "ar-DZ", "ar-TN", "ar-IQ", "ar-LB", "ar-JO", "ar-SY"
]

STTLanguage = str  # union of BaseLanguageCode | ArabicDialectCode — server validates

STTResponseFormat = Literal["json", "verbose_json", "srt", "vtt", "text"]

STTTimestamps = Literal["none", "word", "segment"]


class TranscriptionCommonOptions(TypedDict, total=False):
    model: STTModel
    language: STTLanguage
    response_format: STTResponseFormat
    timestamps: STTTimestamps
    prompt: str
    temperature: float
    diarization: bool


class TranscriptionRequestCommon(TranscriptionCommonOptions, total=False):
    """Options shared between sync transcriptions and the streaming handle."""


class TranscriptionRequest(TranscriptionCommonOptions, total=False):
    # The audio payload. Passed as ``file=`` to the multipart upload.
    filename: str


class TranscriptionJsonResponse(TypedDict, total=False):
    text: str
    language: str
    duration: float
    segments: list[dict[str, Any]]
    words: list[dict[str, Any]]
    # Enterprise usage observability · OpenAI-shaped `usage` block.
    # Present on live keys only · test keys (`hk_test_…`) deliberately
    # bypass the quota pipeline that computes it.
    usage: UsageBlock


class TranscriptionAsyncAccepted(TypedDict):
    job_id: str
    status: Literal["queued"]
    poll_url: str


# TranscriptionResult is whatever the server chose to send — text for the
# text/srt/vtt formats, a mapping for JSON/verbose_json, or an async
# accept body when the server upgrades to a job. Callers typecheck by
# inspecting the shape.
TranscriptionResult = Any


# ---------------------------------------------------------------------------
# Realtime STT (WSS)
# ---------------------------------------------------------------------------


class TranscriptionStreamOptions(TypedDict, total=False):
    model: STTModel
    language: STTLanguage
    sample_rate: SampleRate
    audio_format: Literal["pcm16", "opus", "mulaw"]


class TranscriptionPartialEvent(TypedDict):
    type: Literal["partial"]
    text: str
    seq: int


class TranscriptionFinalEvent(TypedDict, total=False):
    type: Literal["final"]
    text: str
    seq: int
    start: float
    end: float
    language: str


class TranscriptionUsageEvent(TypedDict):
    type: Literal["usage"]
    seconds: float


class TranscriptionErrorEvent(TypedDict, total=False):
    type: Literal["error"]
    code: str
    message: str


# Discriminated union for the caller.
TranscriptionStreamEvent = Any


# ---------------------------------------------------------------------------
# Realtime TTS (WSS /v1/audio/speech/stream)
# ---------------------------------------------------------------------------


class SpeechStreamOptions(TypedDict, total=False):
    """Session-wide defaults forwarded once on connect."""

    model: TTSModel
    voice: str
    cfg: float
    voice_prompt: str


class SpeechStreamCreateRequest(TypedDict, total=False):
    """Per-utterance request passed to :meth:`SpeechStreamHandle.send_speech`."""

    input: str
    voice: str
    model: TTSModel
    cfg: float
    voice_prompt: str
    request_id: str


class SpeechStreamStartedEvent(TypedDict):
    type: Literal["speech.started"]
    request_id: str
    characters: int
    sample_rate: int
    encoding: Literal["pcm_s16le"]
    channels: int
    model: str
    voice: str


class SpeechStreamAudioEvent(TypedDict):
    type: Literal["speech.audio"]
    request_id: str
    chunk: bytes


class SpeechStreamDoneEvent(TypedDict):
    type: Literal["speech.done"]
    request_id: str
    duration_ms: float
    usage: UsageBlock


class SpeechStreamUsageEvent(TypedDict):
    type: Literal["session.usage"]
    session_characters: int
    usage: UsageBlock


class SpeechStreamErrorEvent(TypedDict, total=False):
    type: Literal["error"]
    code: str
    message: str
    retryable: bool
    fatal: bool
    request_id: str


# Discriminated union for the caller.
SpeechStreamEvent = Any


# ---------------------------------------------------------------------------
# Realtime Translate (WSS /v1/audio/translate/stream)
# ---------------------------------------------------------------------------


class TranslateStreamOptions(TypedDict, total=False):
    """Session config forwarded on :meth:`TranslateAPI.stream_ws`.

    The minimal viable session is ``{"target_language": "en"}``; the
    server picks the voice and applies sensible defaults for every
    other field.
    """

    target_language: str
    source_language: str
    voice: str
    gender: VoiceGender
    model_stt: STTModel
    model_llm: ChatModel
    model_tts: TTSModel
    cfg: float
    input_audio_format: Literal["pcm16", "opus", "mulaw"]
    input_sample_rate: Literal[8000, 16000, 22050, 24000, 44100, 48000]
    partials: bool
    system_prompt: str


class TranslateStreamCreatedEvent(TypedDict):
    type: Literal["session.created"]
    session_id: str
    voice_id: str
    voice_slug: str
    model_stt: str
    model_llm: str
    model_tts: str


class TranslateStreamTranscriptionDeltaEvent(TypedDict):
    type: Literal["transcription.delta"]
    utterance_id: str
    text: str
    is_final: bool


class TranslateStreamTranscriptionDoneEvent(TypedDict, total=False):
    type: Literal["transcription.done"]
    utterance_id: str
    text: str
    language: str
    audio_ms: int
    usage: UsageBlock


class TranslateStreamTranslationDeltaEvent(TypedDict):
    type: Literal["translation.delta"]
    utterance_id: str
    text: str


class TranslateStreamTranslationDoneEvent(TypedDict):
    type: Literal["translation.done"]
    utterance_id: str
    text: str
    usage: UsageBlock


class TranslateStreamSpeechStartedEvent(TypedDict):
    type: Literal["speech.started"]
    utterance_id: str
    characters: int
    sample_rate: int
    encoding: Literal["pcm_s16le"]
    channels: int
    voice_id: str


class TranslateStreamSpeechAudioEvent(TypedDict):
    type: Literal["speech.audio"]
    utterance_id: str
    chunk: bytes


class TranslateStreamSpeechDoneEvent(TypedDict):
    type: Literal["speech.done"]
    utterance_id: str
    duration_ms: float
    usage: UsageBlock


class TranslateStreamSessionTotals(TypedDict):
    stt_audio_ms: int
    llm_tokens: int
    tts_characters: int
    credits: int
    cost_usd: str


class TranslateStreamSessionUsageEvent(TypedDict):
    type: Literal["session.usage"]
    session_id: str
    totals: TranslateStreamSessionTotals


class TranslateStreamErrorEvent(TypedDict, total=False):
    type: Literal["error"]
    code: str
    message: str
    retryable: bool
    fatal: bool
    utterance_id: str


TranslateStreamEvent = Any


# ---------------------------------------------------------------------------
# Voices
# ---------------------------------------------------------------------------

VoiceKind = Literal["preset", "cloned"]
VoiceGender = Literal["male", "female", "neutral"]
VoiceLanguage = Literal["ar", "en", "multi"]
VoiceStatus = Literal["ready", "processing", "failed"]


class Voice(TypedDict, total=False):
    id: str
    slug: str
    name: str
    kind: VoiceKind
    language: VoiceLanguage
    gender: VoiceGender
    description: str | None
    preview_url: str | None
    status: VoiceStatus


class VoicesListResponse(TypedDict, total=False):
    object: Literal["list"]
    data: list[Voice]
    has_more: bool
    next_cursor: str | None


class VoicesListQuery(TypedDict, total=False):
    kind: VoiceKind
    language: VoiceLanguage
    q: str
    limit: int
    cursor: str


class VoiceCreateRequest(TypedDict, total=False):
    name: str
    language: VoiceLanguage
    consent_confirmed: bool
    description: str
    filename: str


class VoiceDeletedResponse(TypedDict):
    object: Literal["voice"]
    id: str
    deleted: bool


# ---------------------------------------------------------------------------
# Chat completions
#
# OpenAI-shape on the wire so `openai-python` users can swap base URLs.
# ---------------------------------------------------------------------------

ChatModel = Literal["hakim-chat-v1", "hkm-llm-1"]

ChatRole = Literal["system", "user", "assistant", "tool"]


class ChatTextContentPart(TypedDict):
    type: Literal["text"]
    text: str


ChatMessageContent = Any
"""``str | list[ChatTextContentPart]``. ``Any`` to keep Literal-based
overloads readable on caller code; the server enforces the shape."""


class ChatMessage(TypedDict, total=False):
    role: ChatRole
    content: ChatMessageContent
    name: str
    tool_call_id: str
    tool_calls: list[Any]
    reasoning: str


class ChatReasoningOption(TypedDict):
    enabled: bool


class ChatCompletionRequest(TypedDict, total=False):
    model: ChatModel
    messages: list[ChatMessage]
    stream: bool
    temperature: float
    top_p: float
    max_tokens: int
    n: Literal[1]
    stop: str | list[str]
    user: str
    presence_penalty: float
    frequency_penalty: float
    seed: int
    tools: list[Any]
    tool_choice: Any
    reasoning: ChatReasoningOption


ChatFinishReason = Literal[
    "stop", "length", "content_filter", "tool_calls", "function_call"
]


class ChatCompletionUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(TypedDict, total=False):
    index: int
    message: ChatMessage
    finish_reason: ChatFinishReason | None


class ChatCompletionResponse(TypedDict, total=False):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage
    hakim_usage: UsageBlock


class ChatCompletionChunkDelta(TypedDict, total=False):
    role: ChatRole
    content: str
    reasoning: str


class ChatCompletionChunkChoice(TypedDict, total=False):
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: ChatFinishReason | None


class ChatCompletionChunk(TypedDict, total=False):
    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]
    usage: ChatCompletionUsage


# ---------------------------------------------------------------------------
# Webhooks
# ---------------------------------------------------------------------------

WebhookEventKey = Literal[
    "job.completed",
    "job.failed",
    "voice.clone.completed",
    "voice.clone.failed",
]


class Webhook(TypedDict, total=False):
    id: str
    url: str
    events: list[WebhookEventKey]
    active: bool
    created_at: str


class WebhookCreated(Webhook, total=False):
    secret: str


class WebhookCreateRequest(TypedDict, total=False):
    url: str
    events: list[WebhookEventKey]
    active: bool


class WebhookUpdateRequest(TypedDict, total=False):
    url: str
    events: list[WebhookEventKey]
    active: bool


class WebhooksListResponse(TypedDict, total=False):
    object: Literal["list"]
    data: list[Webhook]
    has_more: bool
    next_cursor: str | None


WebhookDeliveryStatus = Literal["succeeded", "failed", "retrying", "pending"]


class WebhookDelivery(TypedDict, total=False):
    id: str
    webhook_id: str
    event: WebhookEventKey
    status: WebhookDeliveryStatus
    status_code: int | None
    attempts: int
    next_retry_at: str | None
    delivered_at: str | None
    created_at: str


class WebhookDeliveriesListQuery(TypedDict, total=False):
    status: WebhookDeliveryStatus
    limit: int
    cursor: str


class WebhookDeliveriesListResponse(TypedDict, total=False):
    object: Literal["list"]
    data: list[WebhookDelivery]
    has_more: bool
    next_cursor: str | None


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

JobType = Literal["batch_stt", "voice_clone"]
JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class Job(TypedDict, total=False):
    id: str
    type: JobType
    status: JobStatus
    progress_pct: int
    result_url: str | None
    error_message: str | None
    error_code: str | None
    created_at: str
    finished_at: str | None


class JobsListQuery(TypedDict, total=False):
    status: JobStatus
    type: JobType
    limit: int
    cursor: str


class JobsListResponse(TypedDict, total=False):
    object: Literal["list"]
    data: list[Job]
    has_more: bool
    next_cursor: str | None


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

UsageKind = Literal["tts", "stt", "voice_clone", "batch_stt"]


# ---------------------------------------------------------------------------
# Enterprise usage observability.
# ---------------------------------------------------------------------------

# Decimal-string USD amount (treats cleanly as DECIMAL(10,4)).
UsdAmount = str

UsageUnitType = Literal["characters", "seconds", "count", "credits", "tokens"]

OverageMode = Literal["topup", "postpaid", "hard_stop"]

ObservabilityUsageKind = Literal[
    "tts", "stt_batch", "stt_realtime", "voice_clone", "video_studio", "llm_chat"
]


class UsageBlock(TypedDict, total=False):
    """OpenAI-shaped per-request usage block (header + JSON body + WS)."""

    request_id: str
    kind: ObservabilityUsageKind
    units: int
    unit_type: UsageUnitType
    credits: int
    cost_usd: UsdAmount
    model: str | None
    billing_period_start: str
    billing_period_end: str


class CreditsSnapshot(TypedDict, total=False):
    included: int
    used: int
    remaining: int
    effective_limit: int


class ConcurrencySnapshot(TypedDict, total=False):
    limit: int
    current: int


class RateLimitSnapshot(TypedDict, total=False):
    limit_per_minute: int
    remaining: int
    reset_at: str


class PlanSnapshot(TypedDict, total=False):
    id: str
    name: str
    overage_mode: OverageMode


class PeriodSnapshot(TypedDict, total=False):
    start: str
    end: str


class LimitsSnapshot(TypedDict, total=False):
    """Point-in-time limits returned by ``GET /v1/limits``."""

    generated_at: str
    organization_id: str
    plan: PlanSnapshot
    period: PeriodSnapshot
    credits: CreditsSnapshot
    concurrency: ConcurrencySnapshot
    rate_limit: RateLimitSnapshot


class UsageSummary(TypedDict, total=False):
    # Legacy fields (M3 era).
    period_start: str
    period_end: str
    tts_seconds: int
    stt_seconds: int
    voice_clone_seconds: int
    batch_stt_seconds: int
    total_requests: int
    # Enterprise observability additions.
    period: PeriodSnapshot
    tts: dict[str, int]
    stt: dict[str, int]
    estimated_overage_usd: float
    credits: CreditsSnapshot
    plan: PlanSnapshot
    concurrency: ConcurrencySnapshot
    estimated_overage_cost_usd: UsdAmount


class UsageEvent(TypedDict, total=False):
    id: str
    kind: UsageKind
    seconds: int
    units: int
    created_at: str
    request_id: str
    credits: int
    cost_usd: UsdAmount


class UsageEventDetail(UsageEvent, total=False):
    """Single-event detail shape returned by ``GET /v1/usage/events/:id``."""

    model: str | None


class UsageEventsList(TypedDict, total=False):
    object: Literal["list"]
    data: list[UsageEvent]
    has_more: bool
    next_cursor: str | None


class UsageEventsQuery(TypedDict, total=False):
    kind: UsageKind
    from_: str  # `from` is a reserved keyword; serialized as `from`.
    to: str
    limit: int
    cursor: str


# ---------------------------------------------------------------------------
# Settings + notifications (/v1/settings/* + /v1/notifications)
#
# TypedDicts use `total=False` so PATCH bodies can omit keys; GET responses
# always set every field on the wire, and callers can treat the dict as
# fully populated after a `cast`.
# ---------------------------------------------------------------------------

UserLocale = Literal["ar", "en"]


class Profile(TypedDict, total=False):
    id: str
    email: str
    email_verified: bool
    name: str | None
    locale: UserLocale
    timezone: str
    avatar_url: str | None
    marketing_opt_in: bool


class ProfileUpdateRequest(TypedDict, total=False):
    name: str | None
    locale: UserLocale
    timezone: str
    marketing_opt_in: bool


class OrganizationSettings(TypedDict, total=False):
    id: str
    name: str
    slug: str
    billing_email: str | None
    default_locale: UserLocale
    logo_url: str | None


class OrganizationSettingsUpdateRequest(TypedDict, total=False):
    name: str
    slug: str
    billing_email: str | None
    default_locale: UserLocale


class NotificationPreferences(TypedDict, total=False):
    job_completions: bool
    voice_ready: bool
    billing_alerts: bool
    product_updates: bool


class NotificationPreferencesUpdateRequest(TypedDict, total=False):
    job_completions: bool
    voice_ready: bool
    billing_alerts: bool
    product_updates: bool
