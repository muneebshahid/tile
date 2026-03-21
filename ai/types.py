from typing import Literal

from pydantic import BaseModel, Field


class TextBlock(BaseModel):
    """A streamed assistant text block."""

    type: Literal["text"]
    text: str


class ReasoningBlock(BaseModel):
    """A streamed assistant reasoning block."""

    type: Literal["reasoning"]
    reasoning: str
    reasoning_id: str | None = None


class SystemMessage(BaseModel):
    """The system prompt guiding the assistant's behavior."""

    role: Literal["system"] = "system"
    content: str


class UserMessage(BaseModel):
    """A user message containing a prompt or command."""

    role: Literal["user"] = "user"
    content: str


class AssistantMessage(BaseModel):
    """The partial or final assistant message assembled during streaming."""

    role: Literal["assistant"] = "assistant"
    content: list[TextBlock | ReasoningBlock] = Field(default_factory=list)
    response_id: str | None = None


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


class StreamDoneEvent(BaseModel):
    """Marks successful stream completion with the final assistant message."""

    type: Literal["done"]
    message: AssistantMessage


class StreamErrorEvent(BaseModel):
    """Marks failed stream completion with the latest partial assistant message."""

    type: Literal["error"]
    message: str
    partial: AssistantMessage | None = None


StreamEvent = (
    StreamStartEvent
    | ReasoningStartEvent
    | ReasoningDeltaEvent
    | ReasoningEndEvent
    | TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | StreamDoneEvent
    | StreamErrorEvent
)
