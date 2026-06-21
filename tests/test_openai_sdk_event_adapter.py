"""Tests for raw OpenAI SDK event normalization."""

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass

import pytest
from openai.types.responses import ResponseStreamEvent

from ai.openai.normalized_events import NormalizedEvent, NormalizedEventType
from ai.openai.sdk_event_adapter import normalize_sdk_events
from tests.support.openai_response_events import (
    content_part_added_event as _content_part_added_raw_event,
    function_tool_call_added_event as _tool_call_added_raw_event,
    function_tool_call_arguments_delta_event as _tool_call_arguments_delta_raw_event,
    function_tool_call_arguments_done_event as _tool_call_arguments_done_raw_event,
    function_tool_call_done_event as _tool_call_done_raw_event,
    message_added_event as _message_added_raw_event,
    message_done_event as _message_done_raw_event,
    raw_response_stream,
    reasoning_added_event as _reasoning_added_raw_event,
    reasoning_done_event as _reasoning_done_raw_event,
    reasoning_summary_delta_event as _reasoning_delta_raw_event,
    reasoning_summary_part_added_event as _reasoning_summary_part_added_raw_event,
    reasoning_summary_part_done_event as _reasoning_part_done_raw_event,
    reasoning_text_delta_event as _reasoning_text_delta_raw_event,
    refusal_delta_event as _refusal_delta_raw_event,
    response_completed_event as _completed_raw_event,
    response_created_event as _created_raw_event,
    response_error_event as _stream_error_raw_event,
    response_failed_event as _failed_raw_event,
    response_incomplete_event as _incomplete_raw_event,
    text_delta_event as _text_delta_raw_event,
)


@dataclass(frozen=True)
class NormalizationCase:
    """Defines one raw SDK event and its expected normalized provider event."""

    name: str
    raw_event: ResponseStreamEvent
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
                output_index=0,
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
                output_index=0,
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
                output_index=0,
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
                output_index=0,
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
                output_index=0,
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
                output_index=0,
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
                output_index=0,
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
                output_index=0,
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
                output_index=0,
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
                output_index=0,
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
                output_index=0,
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
                output_index=0,
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
                output_index=0,
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
                output_index=0,
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


def _collect_normalized_events(
    raw_events: Sequence[ResponseStreamEvent],
) -> list[NormalizedEvent]:
    """Collects normalized provider events from the public async adapter."""

    async def _collect() -> list[NormalizedEvent]:
        return [
            event
            async for event in normalize_sdk_events(raw_response_stream(raw_events))
        ]

    return asyncio.run(_collect())


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
