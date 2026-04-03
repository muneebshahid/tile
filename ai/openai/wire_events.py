"""Canonical OpenAI wire-event definitions for the provider pipeline."""

from enum import StrEnum
from typing import Literal, TypeAlias, TypedDict

from ai.types.stream import Phase, StopReason
from ai.types.tools import JsonObject


class WireEventType(StrEnum):
    """Shared wire-event names emitted and consumed across the OpenAI adapter."""

    RESPONSE_CREATED = "response.created"
    RESPONSE_REASONING_ADDED = "response.reasoning.added"
    RESPONSE_REASONING_DELTA = "response.reasoning.delta"
    RESPONSE_REASONING_DONE = "response.reasoning.done"
    RESPONSE_MESSAGE_ADDED = "response.message.added"
    RESPONSE_MESSAGE_TEXT_PART = "response.message.text_part"
    RESPONSE_MESSAGE_TEXT_DELTA = "response.message.text.delta"
    RESPONSE_MESSAGE_DONE = "response.message.done"
    RESPONSE_TOOL_CALL_ADDED = "response.tool_call.added"
    RESPONSE_TOOL_CALL_ARGUMENTS_DELTA = "response.tool_call.arguments.delta"
    RESPONSE_TOOL_CALL_ARGUMENTS_DONE = "response.tool_call.arguments.done"
    RESPONSE_TOOL_CALL_DONE = "response.tool_call.done"
    RESPONSE_COMPLETED = "response.completed"
    RESPONSE_INCOMPLETE = "response.incomplete"
    RESPONSE_FAILED = "response.failed"


TextPartType: TypeAlias = Literal["output_text", "refusal"]


class ResponseCreatedWireEvent(TypedDict):
    type: Literal[WireEventType.RESPONSE_CREATED]
    response_id: str


class ResponseReasoningAddedWireEvent(TypedDict):
    type: Literal[WireEventType.RESPONSE_REASONING_ADDED]
    item_id: str


class ResponseReasoningDeltaWireEvent(TypedDict):
    type: Literal[WireEventType.RESPONSE_REASONING_DELTA]
    delta: str


class ResponseReasoningDoneWireEvent(TypedDict):
    type: Literal[WireEventType.RESPONSE_REASONING_DONE]
    item_id: str
    summary_text: str
    reasoning_signature: str | None


class ResponseMessageAddedWireEvent(TypedDict):
    type: Literal[WireEventType.RESPONSE_MESSAGE_ADDED]
    item_id: str
    phase: Phase | None


class ResponseMessageTextPartWireEvent(TypedDict):
    type: Literal[WireEventType.RESPONSE_MESSAGE_TEXT_PART]
    part_type: TextPartType | None


class ResponseMessageTextDeltaWireEvent(TypedDict):
    type: Literal[WireEventType.RESPONSE_MESSAGE_TEXT_DELTA]
    part_type: TextPartType
    delta: str


class ResponseMessageDoneWireEvent(TypedDict):
    type: Literal[WireEventType.RESPONSE_MESSAGE_DONE]
    item_id: str
    text: str
    phase: Phase | None


class ResponseToolCallAddedWireEvent(TypedDict):
    type: Literal[WireEventType.RESPONSE_TOOL_CALL_ADDED]
    provider_item_id: str | None
    call_id: str
    name: str
    arguments: JsonObject
    namespace: str | None


class ResponseToolCallArgumentsDeltaWireEvent(TypedDict):
    type: Literal[WireEventType.RESPONSE_TOOL_CALL_ARGUMENTS_DELTA]
    delta: str


class ResponseToolCallArgumentsDoneWireEvent(TypedDict):
    type: Literal[WireEventType.RESPONSE_TOOL_CALL_ARGUMENTS_DONE]
    arguments: JsonObject


class ResponseToolCallDoneWireEvent(TypedDict):
    type: Literal[WireEventType.RESPONSE_TOOL_CALL_DONE]
    provider_item_id: str | None
    call_id: str
    name: str
    arguments: JsonObject
    namespace: str | None


class ResponseCompletedWireEvent(TypedDict):
    type: Literal[WireEventType.RESPONSE_COMPLETED]
    stop_reason: StopReason


class ResponseIncompleteWireEvent(TypedDict):
    type: Literal[WireEventType.RESPONSE_INCOMPLETE]
    stop_reason: StopReason
    error_message: str | None


class ResponseFailedWireEvent(TypedDict):
    type: Literal[WireEventType.RESPONSE_FAILED]
    message: str


WireEvent: TypeAlias = (
    ResponseCreatedWireEvent
    | ResponseReasoningAddedWireEvent
    | ResponseReasoningDeltaWireEvent
    | ResponseReasoningDoneWireEvent
    | ResponseMessageAddedWireEvent
    | ResponseMessageTextPartWireEvent
    | ResponseMessageTextDeltaWireEvent
    | ResponseMessageDoneWireEvent
    | ResponseToolCallAddedWireEvent
    | ResponseToolCallArgumentsDeltaWireEvent
    | ResponseToolCallArgumentsDoneWireEvent
    | ResponseToolCallDoneWireEvent
    | ResponseCompletedWireEvent
    | ResponseIncompleteWireEvent
    | ResponseFailedWireEvent
)
