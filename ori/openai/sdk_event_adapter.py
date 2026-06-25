import json
from collections.abc import AsyncIterator, Sequence
from typing import cast

from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseCreatedEvent,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseIncompleteEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningSummaryPartDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseStreamEvent,
    ResponseRefusalDeltaEvent,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_output_message import (
    Content as ResponseMessageContent,
    ResponseOutputMessage,
)
from openai.types.responses.response_output_refusal import ResponseOutputRefusal
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_reasoning_item import (
    Summary as ResponseReasoningSummary,
    ResponseReasoningItem,
)

from ori.openai.normalized_events import (
    NormalizedEvent,
    NormalizedEventType,
    TextPartType,
)
from ori.types.stream_events import Phase, StopReason
from ori.types.tools import JsonObject


async def normalize_sdk_events(
    raw_stream: AsyncIterator[ResponseStreamEvent],
) -> AsyncIterator[NormalizedEvent]:
    async for event in raw_stream:
        normalized_event = _normalize_sdk_event(event)
        if normalized_event is not None:
            yield normalized_event


def _normalize_sdk_event(event: ResponseStreamEvent) -> NormalizedEvent | None:
    match event:
        case ResponseCreatedEvent():
            return {
                "type": NormalizedEventType.CREATED,
                "response_id": event.response.id,
            }
        case ResponseOutputItemAddedEvent() if isinstance(
            event.item, ResponseReasoningItem
        ):
            return {
                "type": NormalizedEventType.REASONING_ADDED,
                "item_id": event.item.id,
            }
        case (
            ResponseReasoningSummaryTextDeltaEvent(delta=delta)
            | ResponseReasoningTextDeltaEvent(delta=delta)
        ):
            return {
                "type": NormalizedEventType.REASONING_DELTA,
                "delta": delta,
            }
        case ResponseReasoningSummaryPartDoneEvent():
            return {
                "type": NormalizedEventType.REASONING_DELTA,
                "delta": "\n\n",
            }
        case ResponseOutputItemDoneEvent() if isinstance(
            event.item, ResponseReasoningItem
        ):
            return {
                "type": NormalizedEventType.REASONING_DONE,
                "item_id": event.item.id,
                "summary_text": _join_reasoning_summary_text(event.item.summary),
                "reasoning_signature": _serialize_reasoning_item(event.item),
            }
        case ResponseOutputItemAddedEvent() if isinstance(
            event.item, ResponseOutputMessage
        ):
            return {
                "type": NormalizedEventType.MESSAGE_ADDED,
                "item_id": event.item.id,
                "phase": _extract_message_phase(event.item),
            }
        case ResponseContentPartAddedEvent():
            return {
                "type": NormalizedEventType.MESSAGE_TEXT_PART,
                "part_type": _extract_supported_text_part_type(event),
            }
        case ResponseTextDeltaEvent():
            return {
                "type": NormalizedEventType.MESSAGE_TEXT_DELTA,
                "part_type": "output_text",
                "delta": event.delta,
            }
        case ResponseRefusalDeltaEvent():
            return {
                "type": NormalizedEventType.MESSAGE_TEXT_DELTA,
                "part_type": "refusal",
                "delta": event.delta,
            }
        case ResponseOutputItemDoneEvent() if isinstance(
            event.item, ResponseOutputMessage
        ):
            return {
                "type": NormalizedEventType.MESSAGE_DONE,
                "item_id": event.item.id,
                "text": _join_message_text(event.item.content),
                "phase": _extract_message_phase(event.item),
            }
        case ResponseOutputItemAddedEvent() if isinstance(
            event.item, ResponseFunctionToolCall
        ):
            return {
                "type": NormalizedEventType.TOOL_CALL_ADDED,
                "provider_item_id": event.item.id,
                "call_id": event.item.call_id,
                "name": event.item.name,
                "arguments": _parse_tool_call_arguments(event.item.arguments or ""),
            }
        case ResponseFunctionCallArgumentsDeltaEvent():
            return {
                "type": NormalizedEventType.TOOL_CALL_ARGUMENTS_DELTA,
                "delta": event.delta,
            }
        case ResponseFunctionCallArgumentsDoneEvent():
            return {
                "type": NormalizedEventType.TOOL_CALL_ARGUMENTS_DONE,
                "arguments": _parse_tool_call_arguments(event.arguments),
            }
        case ResponseOutputItemDoneEvent() if isinstance(
            event.item, ResponseFunctionToolCall
        ):
            return {
                "type": NormalizedEventType.TOOL_CALL_DONE,
                "provider_item_id": event.item.id,
                "call_id": event.item.call_id,
                "name": event.item.name,
                "arguments": _parse_tool_call_arguments(event.item.arguments or ""),
            }
        case ResponseCompletedEvent():
            return {
                "type": NormalizedEventType.COMPLETED,
                "stop_reason": _extract_stop_reason(event),
            }
        case ResponseIncompleteEvent():
            return {
                "type": NormalizedEventType.INCOMPLETE,
                "stop_reason": _extract_stop_reason(event),
                "error_message": _extract_incomplete_error_message(event),
            }
        case ResponseErrorEvent():
            return {
                "type": NormalizedEventType.FAILED,
                "message": _extract_stream_error_message(event),
            }
        case ResponseFailedEvent():
            return {
                "type": NormalizedEventType.FAILED,
                "message": _extract_error_message(event),
            }

    return None


def _extract_supported_text_part_type(
    event: ResponseContentPartAddedEvent,
) -> TextPartType | None:
    if event.part.type == "output_text":
        return "output_text"
    if event.part.type == "refusal":
        return "refusal"
    return None


def _parse_tool_call_arguments(arguments: str) -> JsonObject:
    if not arguments.strip():
        return {}

    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {}

    if isinstance(parsed, dict):
        return cast("JsonObject", parsed)
    return {}


def _extract_error_message(event: ResponseFailedEvent) -> str:
    error = getattr(event.response, "error", None)
    if error is None:
        return "OpenAI response failed."

    message = getattr(error, "message", None)
    if isinstance(message, str) and message:
        return message

    return "OpenAI response failed."


def _extract_stream_error_message(event: ResponseErrorEvent) -> str:
    return event.message or "OpenAI stream error."


def _extract_incomplete_error_message(event: ResponseIncompleteEvent) -> str:
    reason = getattr(event.response.incomplete_details, "reason", None)
    if reason == "content_filter":
        return "OpenAI response was truncated by the content filter."
    return "OpenAI response incomplete."


def _extract_stop_reason(
    event: ResponseCompletedEvent | ResponseIncompleteEvent,
) -> StopReason:
    base_reason = _extract_base_stop_reason(event)
    if base_reason == "stop" and any(
        isinstance(item, ResponseFunctionToolCall) for item in event.response.output
    ):
        return "tool_use"
    return base_reason


def _extract_base_stop_reason(
    event: ResponseCompletedEvent | ResponseIncompleteEvent,
) -> StopReason:
    if isinstance(event, ResponseIncompleteEvent):
        reason = getattr(event.response.incomplete_details, "reason", None)
        if reason == "content_filter":
            return "error"
        return "length"
    return "stop"


def _join_reasoning_summary_text(
    summary: Sequence[ResponseReasoningSummary],
) -> str:
    return "\n\n".join(item.text for item in summary if item.text)


def _serialize_reasoning_item(item: ResponseReasoningItem) -> str:
    return json.dumps(item.model_dump(mode="json", exclude_none=True))


def _join_message_text(content: Sequence[ResponseMessageContent]) -> str:
    parts: list[str] = []
    for item in content:
        if isinstance(item, ResponseOutputText):
            parts.append(item.text)
        elif isinstance(item, ResponseOutputRefusal):
            parts.append(item.refusal)
    return "".join(parts)


def _extract_message_phase(
    item: ResponseOutputMessage,
) -> Phase | None:
    phase = getattr(item, "phase", None)
    if phase in {"commentary", "final_answer"}:
        return phase
    return None
