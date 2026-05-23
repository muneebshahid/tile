"""Tests for raw OpenAI SDK event normalization."""

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

import pytest
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseCreatedEvent,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseIncompleteEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningSummaryPartAddedEvent,
    ResponseReasoningSummaryPartDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseRefusalDeltaEvent,
    ResponseTextDeltaEvent,
)

from ai.openai.normalized_events import NormalizedEvent, NormalizedEventType
from ai.openai.sdk_event_adapter import normalize_sdk_events
from ai.types.tools import JsonObject


@dataclass(frozen=True)
class NormalizationCase:
    """Defines one raw SDK event and its expected normalized provider event."""

    name: str
    raw_event: object
    expected_event: NormalizedEvent


def _build_normalization_cases() -> list[NormalizationCase]:
    """Returns the full raw-event normalization matrix for the adapter."""

    return [
        *_build_lifecycle_cases(),
        *_build_reasoning_cases(),
        *_build_message_cases(),
        *_build_tool_call_cases(),
        *_build_failure_cases(),
    ]


def _build_lifecycle_cases() -> list[NormalizationCase]:
    """Builds normalization cases for stream lifecycle events."""

    return [
        NormalizationCase(
            name="response.created",
            raw_event=_created_raw_event(sequence_number=1, response_id="resp_created"),
            expected_event={
                "type": NormalizedEventType.CREATED,
                "response_id": "resp_created",
            },
        ),
        NormalizationCase(
            name="response.completed.stop",
            raw_event=_completed_raw_event(
                sequence_number=2,
                response_id="resp_completed",
            ),
            expected_event={
                "type": NormalizedEventType.COMPLETED,
                "stop_reason": "stop",
            },
        ),
        NormalizationCase(
            name="response.completed.tool_use",
            raw_event=_completed_raw_event(
                sequence_number=3,
                response_id="resp_tool_use",
                output=[
                    {
                        "id": "fc_123",
                        "type": "function_call",
                        "status": "completed",
                        "call_id": "call_123",
                        "name": "get_weather",
                        "arguments": '{"city":"Berlin"}',
                    }
                ],
            ),
            expected_event={
                "type": NormalizedEventType.COMPLETED,
                "stop_reason": "tool_use",
            },
        ),
        NormalizationCase(
            name="response.incomplete.length",
            raw_event=_incomplete_raw_event(
                sequence_number=4,
                response_id="resp_incomplete",
                reason="max_output_tokens",
            ),
            expected_event={
                "type": NormalizedEventType.INCOMPLETE,
                "stop_reason": "length",
                "error_message": "OpenAI response incomplete.",
            },
        ),
        NormalizationCase(
            name="response.incomplete.content_filter",
            raw_event=_incomplete_raw_event(
                sequence_number=5,
                response_id="resp_filtered",
                reason="content_filter",
            ),
            expected_event={
                "type": NormalizedEventType.INCOMPLETE,
                "stop_reason": "error",
                "error_message": "OpenAI response was truncated by the content filter.",
            },
        ),
    ]


def _build_reasoning_cases() -> list[NormalizationCase]:
    """Builds normalization cases for reasoning-related events."""

    return [
        NormalizationCase(
            name="response.output_item.added.reasoning",
            raw_event=_reasoning_added_raw_event(
                sequence_number=6,
                item_id="rs_added",
            ),
            expected_event={
                "type": NormalizedEventType.REASONING_ADDED,
                "item_id": "rs_added",
            },
        ),
        NormalizationCase(
            name="response.reasoning_summary_text.delta",
            raw_event=_reasoning_delta_raw_event(
                sequence_number=7,
                item_id="rs_delta",
                summary_index=0,
                delta="Thinking...",
            ),
            expected_event={
                "type": NormalizedEventType.REASONING_DELTA,
                "delta": "Thinking...",
            },
        ),
        NormalizationCase(
            name="response.reasoning_text.delta",
            raw_event=_reasoning_text_delta_raw_event(
                sequence_number=8,
                item_id="rs_delta",
                content_index=0,
                delta="Thinking with text...",
            ),
            expected_event={
                "type": NormalizedEventType.REASONING_DELTA,
                "delta": "Thinking with text...",
            },
        ),
        NormalizationCase(
            name="response.reasoning_summary_part.done",
            raw_event=_reasoning_part_done_raw_event(
                sequence_number=9,
                item_id="rs_done_part",
                summary_index=0,
                text="A step",
            ),
            expected_event={
                "type": NormalizedEventType.REASONING_DELTA,
                "delta": "\n\n",
            },
        ),
        NormalizationCase(
            name="response.output_item.done.reasoning",
            raw_event=_reasoning_done_raw_event(
                sequence_number=9,
                item_id="rs_done",
                summary_texts=["step one", "step two"],
            ),
            expected_event={
                "type": NormalizedEventType.REASONING_DONE,
                "item_id": "rs_done",
                "summary_text": "step one\n\nstep two",
                "reasoning_signature": json.dumps(
                    {
                        "id": "rs_done",
                        "summary": [
                            {"text": "step one", "type": "summary_text"},
                            {"text": "step two", "type": "summary_text"},
                        ],
                        "type": "reasoning",
                        "status": "completed",
                    }
                ),
            },
        ),
    ]


def _build_message_cases() -> list[NormalizationCase]:
    """Builds normalization cases for assistant message events."""

    return [
        NormalizationCase(
            name="response.output_item.added.message",
            raw_event=_message_added_raw_event(
                sequence_number=10,
                item_id="msg_added",
                phase="commentary",
            ),
            expected_event={
                "type": NormalizedEventType.MESSAGE_ADDED,
                "item_id": "msg_added",
                "phase": "commentary",
            },
        ),
        NormalizationCase(
            name="response.content_part.added.output_text",
            raw_event=_content_part_added_raw_event(
                sequence_number=11,
                item_id="msg_output_text",
                part_type="output_text",
            ),
            expected_event={
                "type": NormalizedEventType.MESSAGE_TEXT_PART,
                "part_type": "output_text",
            },
        ),
        NormalizationCase(
            name="response.content_part.added.refusal",
            raw_event=_content_part_added_raw_event(
                sequence_number=12,
                item_id="msg_refusal",
                part_type="refusal",
            ),
            expected_event={
                "type": NormalizedEventType.MESSAGE_TEXT_PART,
                "part_type": "refusal",
            },
        ),
        NormalizationCase(
            name="response.content_part.added.unsupported",
            raw_event=_content_part_added_raw_event(
                sequence_number=13,
                item_id="msg_unknown",
                part_type="reasoning_text",
            ),
            expected_event={
                "type": NormalizedEventType.MESSAGE_TEXT_PART,
                "part_type": None,
            },
        ),
        NormalizationCase(
            name="response.output_text.delta",
            raw_event=_text_delta_raw_event(
                sequence_number=14,
                item_id="msg_text_delta",
                delta="Hello",
            ),
            expected_event={
                "type": NormalizedEventType.MESSAGE_TEXT_DELTA,
                "part_type": "output_text",
                "delta": "Hello",
            },
        ),
        NormalizationCase(
            name="response.refusal.delta",
            raw_event=_refusal_delta_raw_event(
                sequence_number=15,
                item_id="msg_refusal_delta",
                delta="No",
            ),
            expected_event={
                "type": NormalizedEventType.MESSAGE_TEXT_DELTA,
                "part_type": "refusal",
                "delta": "No",
            },
        ),
        NormalizationCase(
            name="response.output_item.done.message",
            raw_event=_message_done_raw_event(
                sequence_number=16,
                item_id="msg_done",
                phase="final_answer",
                content=[
                    {"type": "output_text", "text": "Hello", "annotations": []},
                    {"type": "refusal", "refusal": " there"},
                ],
            ),
            expected_event={
                "type": NormalizedEventType.MESSAGE_DONE,
                "item_id": "msg_done",
                "text": "Hello there",
                "phase": "final_answer",
            },
        ),
    ]


def _build_tool_call_cases() -> list[NormalizationCase]:
    """Builds normalization cases for tool-call related events."""

    return [
        NormalizationCase(
            name="response.output_item.added.function_call",
            raw_event=_tool_call_added_raw_event(
                sequence_number=17,
                item_id="fc_added",
                call_id="call_added",
                name="get_weather",
                arguments='{"city":"Berlin"}',
            ),
            expected_event={
                "type": NormalizedEventType.TOOL_CALL_ADDED,
                "provider_item_id": "fc_added",
                "call_id": "call_added",
                "name": "get_weather",
                "arguments": {"city": "Berlin"},
            },
        ),
        NormalizationCase(
            name="response.output_item.added.function_call.blank_arguments",
            raw_event=_tool_call_added_raw_event(
                sequence_number=18,
                item_id="fc_added_blank",
                call_id="call_added_blank",
                name="get_weather",
                arguments="",
            ),
            expected_event={
                "type": NormalizedEventType.TOOL_CALL_ADDED,
                "provider_item_id": "fc_added_blank",
                "call_id": "call_added_blank",
                "name": "get_weather",
                "arguments": {},
            },
        ),
        NormalizationCase(
            name="response.function_call_arguments.delta",
            raw_event=_tool_call_arguments_delta_raw_event(
                sequence_number=19,
                item_id="fc_delta",
                delta='{"city"',
            ),
            expected_event={
                "type": NormalizedEventType.TOOL_CALL_ARGUMENTS_DELTA,
                "delta": '{"city"',
            },
        ),
        NormalizationCase(
            name="response.function_call_arguments.done",
            raw_event=_tool_call_arguments_done_raw_event(
                sequence_number=20,
                item_id="fc_args_done",
                name="get_weather",
                arguments='{"city":"Berlin"}',
            ),
            expected_event={
                "type": NormalizedEventType.TOOL_CALL_ARGUMENTS_DONE,
                "arguments": {"city": "Berlin"},
            },
        ),
        NormalizationCase(
            name="response.function_call_arguments.done.malformed_arguments",
            raw_event=_tool_call_arguments_done_raw_event(
                sequence_number=21,
                item_id="fc_args_done_malformed",
                name="get_weather",
                arguments='{"city"',
            ),
            expected_event={
                "type": NormalizedEventType.TOOL_CALL_ARGUMENTS_DONE,
                "arguments": {},
            },
        ),
        NormalizationCase(
            name="response.output_item.done.function_call",
            raw_event=_tool_call_done_raw_event(
                sequence_number=22,
                item_id="fc_done",
                call_id="call_done",
                name="get_weather",
                arguments='{"city":"Berlin"}',
            ),
            expected_event={
                "type": NormalizedEventType.TOOL_CALL_DONE,
                "provider_item_id": "fc_done",
                "call_id": "call_done",
                "name": "get_weather",
                "arguments": {"city": "Berlin"},
            },
        ),
        NormalizationCase(
            name="response.output_item.done.function_call.non_object_arguments",
            raw_event=_tool_call_done_raw_event(
                sequence_number=23,
                item_id="fc_done_non_object",
                call_id="call_done_non_object",
                name="get_weather",
                arguments='["Berlin"]',
            ),
            expected_event={
                "type": NormalizedEventType.TOOL_CALL_DONE,
                "provider_item_id": "fc_done_non_object",
                "call_id": "call_done_non_object",
                "name": "get_weather",
                "arguments": {},
            },
        ),
    ]


def _build_failure_cases() -> list[NormalizationCase]:
    """Builds normalization cases for stream failure events."""

    return [
        NormalizationCase(
            name="error",
            raw_event=_stream_error_raw_event(
                sequence_number=24,
                message="Socket closed",
            ),
            expected_event={
                "type": NormalizedEventType.FAILED,
                "message": "Socket closed",
            },
        ),
        NormalizationCase(
            name="response.failed",
            raw_event=_failed_raw_event(
                sequence_number=25,
                response_id="resp_failed",
                message="Model overloaded",
            ),
            expected_event={
                "type": NormalizedEventType.FAILED,
                "message": "Model overloaded",
            },
        ),
    ]


def _collect_normalized_events(raw_events: Sequence[object]) -> list[NormalizedEvent]:
    """Collects normalized provider events from the public async adapter."""

    async def _collect() -> list[NormalizedEvent]:
        return [event async for event in normalize_sdk_events(_raw_stream(raw_events))]

    return asyncio.run(_collect())


def _raw_stream(raw_events: Sequence[object]) -> AsyncIterator[object]:
    """Yields raw SDK events from a static sequence."""

    async def _iterate() -> AsyncIterator[object]:
        for raw_event in raw_events:
            yield raw_event

    return _iterate()


def _response_payload(
    response_id: str,
    status: str,
    *,
    output: Sequence[JsonObject] | None = None,
    error: dict[str, str] | None = None,
    incomplete_reason: str | None = None,
) -> JsonObject:
    """Builds a minimal OpenAI response payload for event model validation."""

    return {
        "id": response_id,
        "created_at": 0.0,
        "error": error,
        "incomplete_details": (
            {"reason": incomplete_reason} if incomplete_reason is not None else None
        ),
        "model": "gpt-5.4",
        "object": "response",
        "output": list(output or []),
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "status": status,
    }


def _created_raw_event(
    *,
    sequence_number: int,
    response_id: str,
) -> ResponseCreatedEvent:
    """Builds a raw created event."""

    return ResponseCreatedEvent.model_validate(
        {
            "type": "response.created",
            "sequence_number": sequence_number,
            "response": _response_payload(response_id, "in_progress"),
        }
    )


def _completed_raw_event(
    *,
    sequence_number: int,
    response_id: str,
    output: Sequence[JsonObject] | None = None,
) -> ResponseCompletedEvent:
    """Builds a raw completed event."""

    return ResponseCompletedEvent.model_validate(
        {
            "type": "response.completed",
            "sequence_number": sequence_number,
            "response": _response_payload(response_id, "completed", output=output),
        }
    )


def _incomplete_raw_event(
    *,
    sequence_number: int,
    response_id: str,
    reason: str,
) -> ResponseIncompleteEvent:
    """Builds a raw incomplete event."""

    return ResponseIncompleteEvent.model_validate(
        {
            "type": "response.incomplete",
            "sequence_number": sequence_number,
            "response": _response_payload(
                response_id,
                "incomplete",
                incomplete_reason=reason,
            ),
        }
    )


def _reasoning_added_raw_event(
    *,
    sequence_number: int,
    item_id: str,
) -> ResponseOutputItemAddedEvent:
    """Builds a raw reasoning-item added event."""

    return ResponseOutputItemAddedEvent.model_validate(
        {
            "type": "response.output_item.added",
            "sequence_number": sequence_number,
            "output_index": 0,
            "item": {
                "id": item_id,
                "type": "reasoning",
                "summary": [],
                "status": "in_progress",
            },
        }
    )


def _reasoning_delta_raw_event(
    *,
    sequence_number: int,
    item_id: str,
    summary_index: int,
    delta: str,
) -> ResponseReasoningSummaryTextDeltaEvent:
    """Builds a raw reasoning-summary delta event."""

    return ResponseReasoningSummaryTextDeltaEvent.model_validate(
        {
            "type": "response.reasoning_summary_text.delta",
            "sequence_number": sequence_number,
            "item_id": item_id,
            "output_index": 0,
            "summary_index": summary_index,
            "delta": delta,
        }
    )


def _reasoning_text_delta_raw_event(
    *,
    sequence_number: int,
    item_id: str,
    content_index: int,
    delta: str,
) -> ResponseReasoningTextDeltaEvent:
    """Builds a raw reasoning-text delta event."""

    return ResponseReasoningTextDeltaEvent.model_validate(
        {
            "type": "response.reasoning_text.delta",
            "sequence_number": sequence_number,
            "item_id": item_id,
            "output_index": 0,
            "content_index": content_index,
            "delta": delta,
        }
    )


def _reasoning_summary_part_added_raw_event(
    *,
    sequence_number: int,
    item_id: str,
    summary_index: int,
) -> ResponseReasoningSummaryPartAddedEvent:
    """Builds a raw reasoning-summary part added event."""

    return ResponseReasoningSummaryPartAddedEvent.model_validate(
        {
            "type": "response.reasoning_summary_part.added",
            "sequence_number": sequence_number,
            "item_id": item_id,
            "output_index": 0,
            "part": {"type": "summary_text", "text": ""},
            "summary_index": summary_index,
        }
    )


def _reasoning_part_done_raw_event(
    *,
    sequence_number: int,
    item_id: str,
    summary_index: int,
    text: str,
) -> ResponseReasoningSummaryPartDoneEvent:
    """Builds a raw reasoning-summary part done event."""

    return ResponseReasoningSummaryPartDoneEvent.model_validate(
        {
            "type": "response.reasoning_summary_part.done",
            "sequence_number": sequence_number,
            "item_id": item_id,
            "output_index": 0,
            "part": {"type": "summary_text", "text": text},
            "summary_index": summary_index,
        }
    )


def _reasoning_done_raw_event(
    *,
    sequence_number: int,
    item_id: str,
    summary_texts: Sequence[str],
) -> ResponseOutputItemDoneEvent:
    """Builds a raw reasoning-item done event."""

    return ResponseOutputItemDoneEvent.model_validate(
        {
            "type": "response.output_item.done",
            "sequence_number": sequence_number,
            "output_index": 0,
            "item": {
                "id": item_id,
                "type": "reasoning",
                "summary": [
                    {"type": "summary_text", "text": text} for text in summary_texts
                ],
                "status": "completed",
            },
        }
    )


def _message_added_raw_event(
    *,
    sequence_number: int,
    item_id: str,
    phase: str | None = None,
) -> ResponseOutputItemAddedEvent:
    """Builds a raw message-item added event."""

    return ResponseOutputItemAddedEvent.model_validate(
        {
            "type": "response.output_item.added",
            "sequence_number": sequence_number,
            "output_index": 0,
            "item": {
                "id": item_id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
                "phase": phase,
            },
        }
    )


def _content_part_added_raw_event(
    *,
    sequence_number: int,
    item_id: str,
    part_type: str,
) -> ResponseContentPartAddedEvent:
    """Builds a raw content-part added event."""

    part: JsonObject
    if part_type == "output_text":
        part = {"type": "output_text", "text": "", "annotations": []}
    elif part_type == "refusal":
        part = {"type": "refusal", "refusal": ""}
    else:
        part = {"type": part_type, "text": "internal"}

    return ResponseContentPartAddedEvent.model_validate(
        {
            "type": "response.content_part.added",
            "sequence_number": sequence_number,
            "output_index": 0,
            "item_id": item_id,
            "content_index": 0,
            "part": part,
        }
    )


def _text_delta_raw_event(
    *,
    sequence_number: int,
    item_id: str,
    delta: str,
) -> ResponseTextDeltaEvent:
    """Builds a raw output-text delta event."""

    return ResponseTextDeltaEvent.model_validate(
        {
            "type": "response.output_text.delta",
            "sequence_number": sequence_number,
            "output_index": 0,
            "item_id": item_id,
            "content_index": 0,
            "delta": delta,
            "logprobs": [],
        }
    )


def _refusal_delta_raw_event(
    *,
    sequence_number: int,
    item_id: str,
    delta: str,
) -> ResponseRefusalDeltaEvent:
    """Builds a raw refusal delta event."""

    return ResponseRefusalDeltaEvent.model_validate(
        {
            "type": "response.refusal.delta",
            "sequence_number": sequence_number,
            "output_index": 0,
            "item_id": item_id,
            "content_index": 0,
            "delta": delta,
        }
    )


def _message_done_raw_event(
    *,
    sequence_number: int,
    item_id: str,
    content: Sequence[JsonObject],
    phase: str | None = None,
) -> ResponseOutputItemDoneEvent:
    """Builds a raw message-item done event."""

    return ResponseOutputItemDoneEvent.model_validate(
        {
            "type": "response.output_item.done",
            "sequence_number": sequence_number,
            "output_index": 0,
            "item": {
                "id": item_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": list(content),
                "phase": phase,
            },
        }
    )


def _tool_call_added_raw_event(
    *,
    sequence_number: int,
    item_id: str,
    call_id: str,
    name: str,
    arguments: str,
) -> ResponseOutputItemAddedEvent:
    """Builds a raw function-call added event."""

    return ResponseOutputItemAddedEvent.model_validate(
        {
            "type": "response.output_item.added",
            "sequence_number": sequence_number,
            "output_index": 0,
            "item": {
                "id": item_id,
                "type": "function_call",
                "status": "in_progress",
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
            },
        }
    )


def _tool_call_arguments_delta_raw_event(
    *,
    sequence_number: int,
    item_id: str,
    delta: str,
) -> ResponseFunctionCallArgumentsDeltaEvent:
    """Builds a raw function-call arguments delta event."""

    return ResponseFunctionCallArgumentsDeltaEvent.model_validate(
        {
            "type": "response.function_call_arguments.delta",
            "sequence_number": sequence_number,
            "output_index": 0,
            "item_id": item_id,
            "delta": delta,
        }
    )


def _tool_call_arguments_done_raw_event(
    *,
    sequence_number: int,
    item_id: str,
    name: str,
    arguments: str,
) -> ResponseFunctionCallArgumentsDoneEvent:
    """Builds a raw function-call arguments done event."""

    return ResponseFunctionCallArgumentsDoneEvent.model_validate(
        {
            "type": "response.function_call_arguments.done",
            "sequence_number": sequence_number,
            "output_index": 0,
            "item_id": item_id,
            "name": name,
            "arguments": arguments,
        }
    )


def _tool_call_done_raw_event(
    *,
    sequence_number: int,
    item_id: str,
    call_id: str,
    name: str,
    arguments: str,
) -> ResponseOutputItemDoneEvent:
    """Builds a raw function-call done event."""

    return ResponseOutputItemDoneEvent.model_validate(
        {
            "type": "response.output_item.done",
            "sequence_number": sequence_number,
            "output_index": 0,
            "item": {
                "id": item_id,
                "type": "function_call",
                "status": "completed",
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
            },
        }
    )


def _stream_error_raw_event(
    *,
    sequence_number: int,
    message: str,
) -> ResponseErrorEvent:
    """Builds a raw stream error event."""

    return ResponseErrorEvent.model_validate(
        {
            "type": "error",
            "sequence_number": sequence_number,
            "code": "server_error",
            "message": message,
            "param": None,
        }
    )


def _failed_raw_event(
    *,
    sequence_number: int,
    response_id: str,
    message: str,
) -> ResponseFailedEvent:
    """Builds a raw failed normalized event."""

    return ResponseFailedEvent.model_validate(
        {
            "type": "response.failed",
            "sequence_number": sequence_number,
            "response": _response_payload(
                response_id,
                "failed",
                error={"code": "server_error", "message": message},
            ),
        }
    )


@pytest.mark.parametrize(
    "case",
    _build_normalization_cases(),
    ids=lambda case: case.name,
)
def test_normalize_sdk_events_maps_each_supported_raw_event(
    case: NormalizationCase,
) -> None:
    """Normalizes every supported raw SDK event into the expected provider event."""

    assert _collect_normalized_events([case.raw_event]) == [case.expected_event]


def test_normalize_sdk_events_skips_unmapped_raw_events() -> None:
    """Skips raw SDK events that the adapter does not currently map."""

    ignored_event = _reasoning_summary_part_added_raw_event(
        sequence_number=1,
        item_id="rs_ignored",
        summary_index=0,
    )

    assert _collect_normalized_events([ignored_event]) == []
