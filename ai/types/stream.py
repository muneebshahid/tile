from typing import Literal, TypeAlias

from pydantic import BaseModel, Field

from ai.types.tools import JsonObject

Phase: TypeAlias = Literal["commentary", "final_answer"]
StopReason: TypeAlias = Literal["stop", "length", "tool_use", "error", "aborted"]


class TextBlock(BaseModel):
    """An assistant text block shared by streaming and replay history."""

    type: Literal["text"] = "text"
    text: str
    message_id: str | None = None
    phase: Phase | None = None


class ReasoningBlock(BaseModel):
    """An assistant reasoning block shared by streaming and replay history."""

    type: Literal["reasoning"] = "reasoning"
    summary_text: str
    reasoning_signature: str | None = None


class ToolCallBlock(BaseModel):
    """An assistant tool call block shared by streaming and replay history."""

    type: Literal["tool_call"] = "tool_call"
    call_id: str
    name: str
    arguments: JsonObject = Field(default_factory=dict)
    provider_item_id: str | None = None


AssistantBlock: TypeAlias = TextBlock | ReasoningBlock | ToolCallBlock


class AssistantMessage(BaseModel):
    """The partial or final assistant message assembled during streaming."""

    role: Literal["assistant"] = "assistant"
    content: list[AssistantBlock] = Field(default_factory=list)
    response_id: str | None = None
    stop_reason: StopReason = "stop"
    error_message: str | None = None


class StreamStartEvent(BaseModel):
    """Marks the start of a new assistant stream with an empty partial message."""

    type: Literal["start"]
    partial: AssistantMessage


class ReasoningStartEvent(BaseModel):
    """Marks the start of a reasoning block."""

    type: Literal["reasoning_start"]
    partial: AssistantMessage


class ReasoningDeltaEvent(BaseModel):
    """Carries incremental reasoning text for the current reasoning block."""

    type: Literal["reasoning_delta"]
    delta: str
    partial: AssistantMessage


class ReasoningEndEvent(BaseModel):
    """Marks the end of the current reasoning block."""

    type: Literal["reasoning_end"]
    partial: AssistantMessage


class TextStartEvent(BaseModel):
    """Marks the start of a text block."""

    type: Literal["text_start"]
    partial: AssistantMessage


class TextDeltaEvent(BaseModel):
    """Carries incremental text for the current text block."""

    type: Literal["text_delta"]
    delta: str
    partial: AssistantMessage


class TextEndEvent(BaseModel):
    """Marks the end of the current text block."""

    type: Literal["text_end"]
    partial: AssistantMessage


class ToolCallStartEvent(BaseModel):
    """Marks the start of a tool call block."""

    type: Literal["tool_call_start"]
    partial: AssistantMessage


class ToolCallDeltaEvent(BaseModel):
    """Carries incremental tool-call argument JSON for the current block."""

    type: Literal["tool_call_delta"]
    delta: str
    partial: AssistantMessage


class ToolCallEndEvent(BaseModel):
    """Marks the end of the current tool call block."""

    type: Literal["tool_call_end"]
    partial: AssistantMessage


class StreamDoneEvent(BaseModel):
    """Marks successful stream completion with the final assistant message."""

    type: Literal["done"]
    message: AssistantMessage


class StreamErrorEvent(BaseModel):
    """Marks failed stream completion with the latest partial assistant message."""

    type: Literal["error"]
    error: AssistantMessage


StreamEvent = (
    StreamStartEvent
    | ReasoningStartEvent
    | ReasoningDeltaEvent
    | ReasoningEndEvent
    | TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
    | StreamDoneEvent
    | StreamErrorEvent
)
