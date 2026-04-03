"""Canonical raw OpenAI response-event definitions for the provider pipeline."""

from enum import StrEnum
from typing import Literal, TypeAlias, TypedDict

from ai.types.stream import Phase, StopReason
from ai.types.tools import JsonObject


class ResponseEventType(StrEnum):
    """Shared raw response-event names emitted and consumed across the OpenAI adapter."""

    CREATED = "response.created"
    REASONING_ADDED = "response.reasoning.added"
    REASONING_DELTA = "response.reasoning.delta"
    REASONING_DONE = "response.reasoning.done"
    MESSAGE_ADDED = "response.message.added"
    MESSAGE_TEXT_PART = "response.message.text_part"
    MESSAGE_TEXT_DELTA = "response.message.text.delta"
    MESSAGE_DONE = "response.message.done"
    TOOL_CALL_ADDED = "response.tool_call.added"
    TOOL_CALL_ARGUMENTS_DELTA = "response.tool_call.arguments.delta"
    TOOL_CALL_ARGUMENTS_DONE = "response.tool_call.arguments.done"
    TOOL_CALL_DONE = "response.tool_call.done"
    COMPLETED = "response.completed"
    INCOMPLETE = "response.incomplete"
    FAILED = "response.failed"


TextPartType: TypeAlias = Literal["output_text", "refusal"]


class CreatedResponseEvent(TypedDict):
    type: Literal[ResponseEventType.CREATED]
    response_id: str


class ReasoningAddedResponseEvent(TypedDict):
    type: Literal[ResponseEventType.REASONING_ADDED]
    item_id: str


class ReasoningDeltaResponseEvent(TypedDict):
    type: Literal[ResponseEventType.REASONING_DELTA]
    delta: str


class ReasoningDoneResponseEvent(TypedDict):
    type: Literal[ResponseEventType.REASONING_DONE]
    item_id: str
    summary_text: str
    reasoning_signature: str | None


class MessageAddedResponseEvent(TypedDict):
    type: Literal[ResponseEventType.MESSAGE_ADDED]
    item_id: str
    phase: Phase | None


class MessageTextPartResponseEvent(TypedDict):
    type: Literal[ResponseEventType.MESSAGE_TEXT_PART]
    part_type: TextPartType | None


class MessageTextDeltaResponseEvent(TypedDict):
    type: Literal[ResponseEventType.MESSAGE_TEXT_DELTA]
    part_type: TextPartType
    delta: str


class MessageDoneResponseEvent(TypedDict):
    type: Literal[ResponseEventType.MESSAGE_DONE]
    item_id: str
    text: str
    phase: Phase | None


class ToolCallAddedResponseEvent(TypedDict):
    type: Literal[ResponseEventType.TOOL_CALL_ADDED]
    provider_item_id: str | None
    call_id: str
    name: str
    arguments: JsonObject
    namespace: str | None


class ToolCallArgumentsDeltaResponseEvent(TypedDict):
    type: Literal[ResponseEventType.TOOL_CALL_ARGUMENTS_DELTA]
    delta: str


class ToolCallArgumentsDoneResponseEvent(TypedDict):
    type: Literal[ResponseEventType.TOOL_CALL_ARGUMENTS_DONE]
    arguments: JsonObject


class ToolCallDoneResponseEvent(TypedDict):
    type: Literal[ResponseEventType.TOOL_CALL_DONE]
    provider_item_id: str | None
    call_id: str
    name: str
    arguments: JsonObject
    namespace: str | None


class CompletedResponseEvent(TypedDict):
    type: Literal[ResponseEventType.COMPLETED]
    stop_reason: StopReason


class IncompleteResponseEvent(TypedDict):
    type: Literal[ResponseEventType.INCOMPLETE]
    stop_reason: StopReason
    error_message: str | None


class FailedResponseEvent(TypedDict):
    type: Literal[ResponseEventType.FAILED]
    message: str


ResponseEvent: TypeAlias = (
    CreatedResponseEvent
    | ReasoningAddedResponseEvent
    | ReasoningDeltaResponseEvent
    | ReasoningDoneResponseEvent
    | MessageAddedResponseEvent
    | MessageTextPartResponseEvent
    | MessageTextDeltaResponseEvent
    | MessageDoneResponseEvent
    | ToolCallAddedResponseEvent
    | ToolCallArgumentsDeltaResponseEvent
    | ToolCallArgumentsDoneResponseEvent
    | ToolCallDoneResponseEvent
    | CompletedResponseEvent
    | IncompleteResponseEvent
    | FailedResponseEvent
)
