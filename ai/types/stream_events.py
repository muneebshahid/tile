"""Provider stream events and replayable assistant block contracts."""

from typing import Literal, TypeAlias

from pydantic import BaseModel, Field

from ai.types.tools import JsonObject

Phase: TypeAlias = Literal["commentary", "final_answer"]
StopReason: TypeAlias = Literal["stop", "length", "tool_use", "error", "aborted"]


class ProviderSource(BaseModel):
    """Provider and model that produced an assistant turn."""

    provider: str
    model: str | None = None


class ProviderMetadata(BaseModel):
    """Opaque provider-owned metadata needed for compatible replay."""

    data: JsonObject = Field(default_factory=dict)


class TextBlock(BaseModel):
    """Replayable assistant text block."""

    type: Literal["text"] = "text"
    text: str
    provider_metadata: ProviderMetadata | None = None

    @property
    def message_id(self) -> str | None:
        """Return the provider message id when replay metadata includes it."""

        return _metadata_string(self.provider_metadata, "message_id")

    @property
    def phase(self) -> Phase | None:
        """Return the provider message phase when replay metadata includes it."""

        value = _metadata_string(self.provider_metadata, "phase")
        if value == "commentary":
            return "commentary"
        if value == "final_answer":
            return "final_answer"
        return None


class ReasoningBlock(BaseModel):
    """Replayable assistant reasoning block."""

    type: Literal["reasoning"] = "reasoning"
    summary_text: str
    provider_metadata: ProviderMetadata | None = None

    @property
    def reasoning_signature(self) -> str | None:
        """Return the provider reasoning signature when present."""

        return _metadata_string(self.provider_metadata, "reasoning_signature")


class ToolCallBlock(BaseModel):
    """Replayable assistant tool-call block."""

    type: Literal["tool_call"] = "tool_call"
    call_id: str
    name: str
    arguments: JsonObject = Field(default_factory=dict)
    provider_metadata: ProviderMetadata | None = None

    @property
    def provider_item_id(self) -> str | None:
        """Return the provider item id when replay metadata includes it."""

        return _metadata_string(self.provider_metadata, "provider_item_id")


AssistantBlock: TypeAlias = TextBlock | ReasoningBlock | ToolCallBlock


class StreamEvent(BaseModel):
    """Base provider stream event."""

    type: str


class BlockStreamEvent(StreamEvent):
    """Provider stream event scoped to one assistant content block."""

    content_index: int


class StreamStartEvent(StreamEvent):
    """Marks creation of a provider response stream."""

    type: Literal["stream_start"] = "stream_start"
    source: ProviderSource
    response_id: str


class ReasoningStartEvent(BlockStreamEvent):
    """Marks the start of a reasoning block."""

    type: Literal["reasoning_start"] = "reasoning_start"


class ReasoningDeltaEvent(BlockStreamEvent):
    """Carries incremental reasoning text for a block."""

    type: Literal["reasoning_delta"] = "reasoning_delta"
    delta: str


class ReasoningEndEvent(BlockStreamEvent):
    """Carries a completed reasoning block."""

    type: Literal["reasoning_end"] = "reasoning_end"
    block: ReasoningBlock


class TextStartEvent(BlockStreamEvent):
    """Marks the start of a text block."""

    type: Literal["text_start"] = "text_start"


class TextDeltaEvent(BlockStreamEvent):
    """Carries incremental text for a block."""

    type: Literal["text_delta"] = "text_delta"
    delta: str


class TextEndEvent(BlockStreamEvent):
    """Carries a completed text block."""

    type: Literal["text_end"] = "text_end"
    block: TextBlock


class ToolCallStartEvent(BlockStreamEvent):
    """Marks the start of a tool-call block."""

    type: Literal["tool_call_start"] = "tool_call_start"
    call_id: str
    name: str


class ToolCallDeltaEvent(BlockStreamEvent):
    """Carries incremental tool-call argument JSON."""

    type: Literal["tool_call_delta"] = "tool_call_delta"
    delta: str


class ToolCallEndEvent(BlockStreamEvent):
    """Carries a completed tool-call block."""

    type: Literal["tool_call_end"] = "tool_call_end"
    block: ToolCallBlock


StreamBlockEvent: TypeAlias = ReasoningEndEvent | TextEndEvent | ToolCallEndEvent

StreamUpdateEvent: TypeAlias = (
    ReasoningStartEvent
    | ReasoningDeltaEvent
    | ReasoningEndEvent
    | TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
)


class StreamDoneEvent(StreamEvent):
    """Marks successful provider stream completion."""

    type: Literal["stream_done"] = "stream_done"
    source: ProviderSource
    response_id: str | None = None
    stop_reason: StopReason
    blocks: list[AssistantBlock] = Field(default_factory=list)


class StreamErrorEvent(StreamEvent):
    """Marks failed provider stream completion."""

    type: Literal["stream_error"] = "stream_error"
    source: ProviderSource
    response_id: str | None = None
    stop_reason: StopReason = "error"
    error_message: str
    blocks: list[AssistantBlock] = Field(default_factory=list)


StreamTerminalEvent: TypeAlias = StreamDoneEvent | StreamErrorEvent
ProviderStreamEvent: TypeAlias = (
    StreamStartEvent | StreamUpdateEvent | StreamDoneEvent | StreamErrorEvent
)


def _metadata_string(
    metadata: ProviderMetadata | None,
    key: str,
) -> str | None:
    """Read a string value from provider metadata."""

    if metadata is None:
        return None
    value = metadata.data.get(key)
    if isinstance(value, str):
        return value
    return None
