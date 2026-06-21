"""Normalize ChatGPT subscription SSE payloads into normalized provider events."""

import json
from collections.abc import AsyncIterator, Sequence
from typing import TypeAlias, cast

from pydantic import JsonValue

from ai.openai.normalized_events import (
    NormalizedEvent,
    NormalizedEventType,
    TextPartType,
)
from ai.types.stream_events import Phase, StopReason
from ai.types.tools import JsonObject

SubscriptionEventPayload: TypeAlias = JsonObject


async def normalize_subscription_events(
    raw_stream: AsyncIterator[SubscriptionEventPayload],
) -> AsyncIterator[NormalizedEvent]:
    """Map raw subscription SSE payloads into transport-agnostic normalized events."""

    async for event in raw_stream:
        normalized_event = _normalize_subscription_event(event)
        if normalized_event is not None:
            yield normalized_event


def _normalize_subscription_event(
    event: SubscriptionEventPayload,
) -> NormalizedEvent | None:
    """Convert one raw subscription payload into the shared normalized-event union."""

    event_type = _read_string(event, "type")
    match event_type:
        case "response.created":
            return _normalize_created_event(event)
        case "response.output_item.added":
            return _normalize_output_item_added_event(event)
        case "response.reasoning_summary_text.delta" | "response.reasoning_text.delta":
            return _normalize_reasoning_summary_delta_event(event)
        case "response.reasoning_summary_part.done":
            return _normalize_reasoning_summary_part_done_event()
        case "response.content_part.added":
            return _normalize_content_part_added_event(event)
        case "response.output_text.delta":
            return _normalize_text_delta_event(event)
        case "response.refusal.delta":
            return _normalize_refusal_delta_event(event)
        case "response.function_call_arguments.delta":
            return _normalize_tool_call_arguments_delta_event(event)
        case "response.function_call_arguments.done":
            return _normalize_tool_call_arguments_done_event(event)
        case "response.output_item.done":
            return _normalize_output_item_done_event(event)
        case "response.completed":
            return _normalize_completed_event(event)
        case "response.incomplete":
            return _normalize_incomplete_event(event)
        case "response.done":
            return _normalize_done_event(event)
        case "response.failed":
            return _normalize_failed_event(event)
        case "error":
            return _normalize_error_event(event)

    return None


def _normalize_created_event(
    event: SubscriptionEventPayload,
) -> NormalizedEvent | None:
    response = _read_object(event, "response")
    response_id = _read_string(response, "id") if response is not None else None
    if response_id is None:
        return None
    return {
        "type": NormalizedEventType.CREATED,
        "response_id": response_id,
    }


def _normalize_output_item_added_event(
    event: SubscriptionEventPayload,
) -> NormalizedEvent | None:
    item = _read_object(event, "item")
    item_type = _read_string(item, "type") if item is not None else None
    if item is None or item_type is None:
        return None

    match item_type:
        case "reasoning":
            item_id = _read_string(item, "id")
            if item_id is None:
                return None
            return {
                "type": NormalizedEventType.REASONING_ADDED,
                "item_id": item_id,
            }
        case "message":
            item_id = _read_string(item, "id")
            if item_id is None:
                return None
            return {
                "type": NormalizedEventType.MESSAGE_ADDED,
                "item_id": item_id,
                "phase": _extract_message_phase(item),
            }
        case "function_call":
            call_id = _read_string(item, "call_id")
            name = _read_string(item, "name")
            if call_id is None or name is None:
                return None
            return {
                "type": NormalizedEventType.TOOL_CALL_ADDED,
                "provider_item_id": _read_string(item, "id"),
                "call_id": call_id,
                "name": name,
                "arguments": _parse_tool_call_arguments(
                    _read_string(item, "arguments")
                ),
            }

    return None


def _normalize_reasoning_summary_delta_event(
    event: SubscriptionEventPayload,
) -> NormalizedEvent | None:
    delta = _read_string(event, "delta")
    if delta is None:
        return None
    return {
        "type": NormalizedEventType.REASONING_DELTA,
        "delta": delta,
    }


def _normalize_reasoning_summary_part_done_event() -> NormalizedEvent:
    return {
        "type": NormalizedEventType.REASONING_DELTA,
        "delta": "\n\n",
    }


def _normalize_content_part_added_event(
    event: SubscriptionEventPayload,
) -> NormalizedEvent:
    part = _read_object(event, "part")
    return {
        "type": NormalizedEventType.MESSAGE_TEXT_PART,
        "part_type": _extract_supported_text_part_type(part),
    }


def _normalize_text_delta_event(
    event: SubscriptionEventPayload,
) -> NormalizedEvent | None:
    delta = _read_string(event, "delta")
    if delta is None:
        return None
    return {
        "type": NormalizedEventType.MESSAGE_TEXT_DELTA,
        "part_type": "output_text",
        "delta": delta,
    }


def _normalize_refusal_delta_event(
    event: SubscriptionEventPayload,
) -> NormalizedEvent | None:
    delta = _read_string(event, "delta")
    if delta is None:
        return None
    return {
        "type": NormalizedEventType.MESSAGE_TEXT_DELTA,
        "part_type": "refusal",
        "delta": delta,
    }


def _normalize_tool_call_arguments_delta_event(
    event: SubscriptionEventPayload,
) -> NormalizedEvent | None:
    delta = _read_string(event, "delta")
    if delta is None:
        return None
    return {
        "type": NormalizedEventType.TOOL_CALL_ARGUMENTS_DELTA,
        "delta": delta,
    }


def _normalize_tool_call_arguments_done_event(
    event: SubscriptionEventPayload,
) -> NormalizedEvent:
    return {
        "type": NormalizedEventType.TOOL_CALL_ARGUMENTS_DONE,
        "arguments": _parse_tool_call_arguments(_read_string(event, "arguments")),
    }


def _normalize_output_item_done_event(
    event: SubscriptionEventPayload,
) -> NormalizedEvent | None:
    item = _read_object(event, "item")
    item_type = _read_string(item, "type") if item is not None else None
    if item is None or item_type is None:
        return None

    match item_type:
        case "reasoning":
            item_id = _read_string(item, "id")
            if item_id is None:
                return None
            return {
                "type": NormalizedEventType.REASONING_DONE,
                "item_id": item_id,
                "summary_text": _join_reasoning_summary_text(item),
                "reasoning_signature": _serialize_item(item),
            }
        case "message":
            item_id = _read_string(item, "id")
            if item_id is None:
                return None
            return {
                "type": NormalizedEventType.MESSAGE_DONE,
                "item_id": item_id,
                "text": _join_message_text(item),
                "phase": _extract_message_phase(item),
            }
        case "function_call":
            call_id = _read_string(item, "call_id")
            name = _read_string(item, "name")
            if call_id is None or name is None:
                return None
            return {
                "type": NormalizedEventType.TOOL_CALL_DONE,
                "provider_item_id": _read_string(item, "id"),
                "call_id": call_id,
                "name": name,
                "arguments": _parse_tool_call_arguments(
                    _read_string(item, "arguments")
                ),
            }

    return None


def _normalize_completed_event(
    event: SubscriptionEventPayload,
) -> NormalizedEvent:
    return {
        "type": NormalizedEventType.COMPLETED,
        "stop_reason": _extract_stop_reason(event),
    }


def _normalize_incomplete_event(
    event: SubscriptionEventPayload,
) -> NormalizedEvent:
    stop_reason = _extract_incomplete_stop_reason(event)
    return {
        "type": NormalizedEventType.INCOMPLETE,
        "stop_reason": stop_reason,
        "error_message": _extract_incomplete_error_message(stop_reason),
    }


def _normalize_done_event(
    event: SubscriptionEventPayload,
) -> NormalizedEvent:
    response = _read_object(event, "response")
    status = _read_string(response, "status") if response is not None else None
    if status == "incomplete":
        return _normalize_incomplete_event(event)
    return _normalize_completed_event(event)


def _normalize_failed_event(
    event: SubscriptionEventPayload,
) -> NormalizedEvent:
    response = _read_object(event, "response")
    error = _read_object(response, "error") if response is not None else None
    message = _read_string(error, "message") or "OpenAI response failed."
    return {
        "type": NormalizedEventType.FAILED,
        "message": message,
    }


def _normalize_error_event(
    event: SubscriptionEventPayload,
) -> NormalizedEvent:
    return {
        "type": NormalizedEventType.FAILED,
        "message": _read_string(event, "message") or "OpenAI stream error.",
    }


def _extract_supported_text_part_type(
    part: JsonObject | None,
) -> TextPartType | None:
    part_type = _read_string(part, "type") if part is not None else None
    if part_type == "output_text":
        return "output_text"
    if part_type == "refusal":
        return "refusal"
    return None


def _parse_tool_call_arguments(arguments: str | None) -> JsonObject:
    if arguments is None or not arguments.strip():
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict):
        return cast("JsonObject", parsed)
    return {}


def _join_reasoning_summary_text(item: JsonObject) -> str:
    summary_items = _read_list(item, "summary")
    return "\n\n".join(
        text
        for text in (_extract_reasoning_summary_text(entry) for entry in summary_items)
        if text
    )


def _extract_reasoning_summary_text(value: JsonValue) -> str | None:
    if not isinstance(value, dict):
        return None
    entry = cast("JsonObject", value)
    if _read_string(entry, "type") != "summary_text":
        return None
    return _read_string(entry, "text")


def _serialize_item(item: JsonObject) -> str:
    return json.dumps(item)


def _join_message_text(item: JsonObject) -> str:
    content = _read_list(item, "content")
    return "".join(
        text for text in (_extract_content_text(entry) for entry in content) if text
    )


def _extract_content_text(value: JsonValue) -> str | None:
    if not isinstance(value, dict):
        return None
    entry = cast("JsonObject", value)
    entry_type = _read_string(entry, "type")
    if entry_type == "output_text":
        return _read_string(entry, "text")
    if entry_type == "refusal":
        return _read_string(entry, "refusal")
    return None


def _extract_message_phase(item: JsonObject) -> Phase | None:
    phase = _read_string(item, "phase")
    if phase in {"commentary", "final_answer"}:
        return cast("Phase", phase)
    return None


def _extract_stop_reason(event: SubscriptionEventPayload) -> StopReason:
    response = _read_object(event, "response")
    if response is not None and _response_contains_tool_call(response):
        return "tool_use"
    return "stop"


def _response_contains_tool_call(response: JsonObject) -> bool:
    output = _read_list(response, "output")
    return any(_entry_is_function_call(entry) for entry in output)


def _entry_is_function_call(value: JsonValue) -> bool:
    if not isinstance(value, dict):
        return False
    return _read_string(cast("JsonObject", value), "type") == "function_call"


def _extract_incomplete_stop_reason(
    event: SubscriptionEventPayload,
) -> StopReason:
    response = _read_object(event, "response")
    incomplete_details = (
        _read_object(response, "incomplete_details") if response is not None else None
    )
    reason = (
        _read_string(incomplete_details, "reason")
        if incomplete_details is not None
        else None
    )
    if reason == "content_filter":
        return "error"
    return "length"


def _extract_incomplete_error_message(
    stop_reason: StopReason,
) -> str | None:
    if stop_reason == "error":
        return "OpenAI response was truncated by the content filter."
    return "OpenAI response incomplete."


def _read_object(
    payload: JsonObject | None,
    key: str,
) -> JsonObject | None:
    if payload is None:
        return None
    value = payload.get(key)
    if isinstance(value, dict):
        return cast("JsonObject", value)
    return None


def _read_list(
    payload: JsonObject,
    key: str,
) -> Sequence[JsonValue]:
    value = payload.get(key)
    if isinstance(value, list):
        return cast("list[JsonValue]", value)
    return ()


def _read_string(
    payload: JsonObject | None,
    key: str,
) -> str | None:
    if payload is None:
        return None
    value = payload.get(key)
    if isinstance(value, str):
        return value
    return None
