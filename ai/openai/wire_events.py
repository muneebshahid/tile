from typing import Literal, TypeAlias, TypedDict

from ai.types.stream import Phase, StopReason
from ai.types.tools import JsonObject

TextPartType: TypeAlias = Literal["output_text", "refusal"]
WireEventType: TypeAlias = Literal[
    "response.created",
    "response.reasoning.added",
    "response.reasoning.delta",
    "response.reasoning.done",
    "response.message.added",
    "response.message.text_part",
    "response.message.text.delta",
    "response.message.done",
    "response.tool_call.added",
    "response.tool_call.arguments.delta",
    "response.tool_call.arguments.done",
    "response.tool_call.done",
    "response.completed",
    "response.incomplete",
    "response.failed",
]


class ResponseCreatedWireEvent(TypedDict):
    type: Literal["response.created"]
    response_id: str


class ResponseReasoningAddedWireEvent(TypedDict):
    type: Literal["response.reasoning.added"]
    item_id: str


class ResponseReasoningDeltaWireEvent(TypedDict):
    type: Literal["response.reasoning.delta"]
    delta: str


class ResponseReasoningDoneWireEvent(TypedDict):
    type: Literal["response.reasoning.done"]
    item_id: str
    summary_text: str
    reasoning_signature: str | None


class ResponseMessageAddedWireEvent(TypedDict):
    type: Literal["response.message.added"]
    item_id: str
    phase: Phase | None


class ResponseMessageTextPartWireEvent(TypedDict):
    type: Literal["response.message.text_part"]
    part_type: TextPartType


class ResponseMessageTextDeltaWireEvent(TypedDict):
    type: Literal["response.message.text.delta"]
    part_type: TextPartType
    delta: str


class ResponseMessageDoneWireEvent(TypedDict):
    type: Literal["response.message.done"]
    item_id: str
    text: str
    phase: Phase | None


class ResponseToolCallAddedWireEvent(TypedDict):
    type: Literal["response.tool_call.added"]
    item_id: str
    call_id: str
    name: str
    arguments: JsonObject
    namespace: str | None


class ResponseToolCallArgumentsDeltaWireEvent(TypedDict):
    type: Literal["response.tool_call.arguments.delta"]
    delta: str


class ResponseToolCallArgumentsDoneWireEvent(TypedDict):
    type: Literal["response.tool_call.arguments.done"]
    arguments: JsonObject


class ResponseToolCallDoneWireEvent(TypedDict):
    type: Literal["response.tool_call.done"]
    item_id: str
    call_id: str
    name: str
    arguments: JsonObject
    namespace: str | None


class ResponseCompletedWireEvent(TypedDict):
    type: Literal["response.completed"]
    stop_reason: StopReason


class ResponseIncompleteWireEvent(TypedDict):
    type: Literal["response.incomplete"]
    stop_reason: StopReason
    error_message: str | None


class ResponseFailedWireEvent(TypedDict):
    type: Literal["response.failed"]
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
