"""Tests for assembling normalized provider events into stream events.

These tests document the middle of the streaming lifecycle. OpenAI transport
adapters produce normalized events such as ``CREATED``, ``MESSAGE_TEXT_PART``,
``MESSAGE_TEXT_DELTA``, and ``COMPLETED``. The stream assembler consumes those
events, privately accumulates assistant blocks, and emits provider stream events
such as ``text_start``, ``text_delta``, ``text_end``, and ``stream_done``.
"""

import asyncio
from collections.abc import Sequence

from ori.openai.normalized_events import (
    CompletedNormalizedEvent,
    CreatedNormalizedEvent,
    FailedNormalizedEvent,
    IncompleteNormalizedEvent,
    MessageAddedNormalizedEvent,
    MessageDoneNormalizedEvent,
    MessageTextDeltaNormalizedEvent,
    MessageTextPartNormalizedEvent,
    ReasoningAddedNormalizedEvent,
    ReasoningDeltaNormalizedEvent,
    ReasoningDoneNormalizedEvent,
    NormalizedEvent,
    NormalizedEventType,
    TextPartType,
    ToolCallAddedNormalizedEvent,
    ToolCallArgumentsDeltaNormalizedEvent,
    ToolCallArgumentsDoneNormalizedEvent,
    ToolCallDoneNormalizedEvent,
)
from ori.openai.stream_assembler import assemble_stream
from ori.types.stream_events import (
    Phase,
    ProviderSource,
    ProviderStreamEvent,
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    ReasoningStartEvent,
    StreamDoneEvent,
    StreamErrorEvent,
    StreamStartEvent,
    StopReason,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from ori.types.tools import JsonObject
from tests.support.async_streams import async_stream
from tests.support.stream_assertions import (
    expect_metadata_string as _expect_metadata_string,
    expect_reasoning_block as _expect_reasoning_block,
    expect_stream_event as _expect_event_type,
    expect_text_block as _expect_text_block,
    expect_tool_call_block as _expect_tool_call_block,
)


def test_assemble_stream_accumulates_reasoning_and_text_blocks() -> None:
    """Accumulates reasoning and text blocks onto the terminal stream event."""

    normalized_events = [
        _created_event("resp_success"),
        _reasoning_added_event("rs_123"),
        _reasoning_delta_event("Exploring "),
        _reasoning_delta_event("reasoning traces"),
        _reasoning_delta_event("\n\n"),
        _reasoning_delta_event("Formulating "),
        _reasoning_delta_event("reasoning traces"),
        _reasoning_done_event(
            item_id="rs_123",
            summary_text="Exploring reasoning traces\n\nFormulating reasoning traces",
            reasoning_signature='{"id":"rs_123"}',
        ),
        _message_added_event("msg_123"),
        _message_text_part_event("output_text"),
        _message_text_delta_event("output_text", "Hello"),
        _message_text_delta_event("output_text", " world"),
        _message_done_event("msg_123", "Hello world"),
        _completed_event("stop"),
    ]

    events = _collect_stream_events(normalized_events)

    start = _expect_event_type(events[0], StreamStartEvent)
    reasoning_start = _expect_event_type(events[1], ReasoningStartEvent)
    reasoning_delta_one = _expect_event_type(events[2], ReasoningDeltaEvent)
    reasoning_delta_two = _expect_event_type(events[3], ReasoningDeltaEvent)
    reasoning_delta_separator = _expect_event_type(events[4], ReasoningDeltaEvent)
    reasoning_delta_three = _expect_event_type(events[5], ReasoningDeltaEvent)
    reasoning_delta_four = _expect_event_type(events[6], ReasoningDeltaEvent)
    reasoning_end = _expect_event_type(events[7], ReasoningEndEvent)
    text_start = _expect_event_type(events[8], TextStartEvent)
    text_delta_one = _expect_event_type(events[9], TextDeltaEvent)
    text_delta_two = _expect_event_type(events[10], TextDeltaEvent)
    text_end = _expect_event_type(events[11], TextEndEvent)
    done = _expect_event_type(events[12], StreamDoneEvent)
    final_reasoning_block = _expect_reasoning_block(reasoning_end.block)
    final_text_block = _expect_text_block(text_end.block)
    done_reasoning_block = _expect_reasoning_block(done.blocks[0])
    done_text_block = _expect_text_block(done.blocks[1])

    assert [event.type for event in events] == [
        "stream_start",
        "reasoning_start",
        "reasoning_delta",
        "reasoning_delta",
        "reasoning_delta",
        "reasoning_delta",
        "reasoning_delta",
        "reasoning_end",
        "text_start",
        "text_delta",
        "text_delta",
        "text_end",
        "stream_done",
    ]
    assert start.response_id == "resp_success"
    assert start.source == _source()
    assert reasoning_start.content_index == 0
    assert reasoning_delta_one.content_index == 0
    assert reasoning_delta_two.content_index == 0
    assert reasoning_delta_separator.content_index == 0
    assert reasoning_delta_three.content_index == 0
    assert reasoning_delta_four.content_index == 0
    assert reasoning_end.content_index == 0
    assert text_start.content_index == 1
    assert text_delta_one.content_index == 1
    assert text_delta_two.content_index == 1
    assert text_end.content_index == 1
    assert reasoning_delta_one.delta == "Exploring "
    assert reasoning_delta_two.delta == "reasoning traces"
    assert reasoning_delta_separator.delta == "\n\n"
    assert reasoning_delta_three.delta == "Formulating "
    assert reasoning_delta_four.delta == "reasoning traces"
    assert (
        final_reasoning_block.summary_text
        == "Exploring reasoning traces\n\nFormulating reasoning traces"
    )
    assert (
        _expect_metadata_string(
            final_reasoning_block,
            "reasoning_signature",
        )
        == '{"id":"rs_123"}'
    )
    assert text_delta_one.delta == "Hello"
    assert text_delta_two.delta == " world"
    assert final_text_block.text == "Hello world"
    assert done.response_id == "resp_success"
    assert done.source == _source()
    assert (
        done_reasoning_block.summary_text
        == "Exploring reasoning traces\n\nFormulating reasoning traces"
    )
    assert (
        _expect_metadata_string(
            done_reasoning_block,
            "reasoning_signature",
        )
        == '{"id":"rs_123"}'
    )
    assert done_text_block.text == "Hello world"


def test_assemble_stream_preserves_reasoning_deltas_when_done_summary_is_empty() -> (
    None
):
    """Preserves accumulated reasoning deltas when the done event has no summary."""

    events = _collect_stream_events(
        [
            _created_event("resp_reasoning_empty_done"),
            _reasoning_added_event("rs_123"),
            _reasoning_delta_event("Draft summary"),
            _reasoning_done_event(
                item_id="rs_123",
                summary_text="",
                reasoning_signature='{"id":"rs_123"}',
            ),
            _completed_event("stop"),
        ]
    )

    reasoning_end = _expect_event_type(events[3], ReasoningEndEvent)
    done = _expect_event_type(events[4], StreamDoneEvent)
    reasoning_block = _expect_reasoning_block(reasoning_end.block)
    done_reasoning_block = _expect_reasoning_block(done.blocks[0])

    assert [event.type for event in events] == [
        "stream_start",
        "reasoning_start",
        "reasoning_delta",
        "reasoning_end",
        "stream_done",
    ]
    assert reasoning_block.summary_text == "Draft summary"
    assert done_reasoning_block.summary_text == "Draft summary"


def test_assemble_stream_maps_refusal_deltas() -> None:
    """Accumulates refusal deltas onto the active text block."""

    events = _collect_stream_events(
        [
            _created_event("resp_refusal"),
            _message_added_event("msg_refusal"),
            _message_text_part_event("refusal"),
            _message_text_delta_event("refusal", "No"),
            _message_done_event("msg_refusal", "No thanks"),
            _completed_event("stop"),
        ]
    )

    text_start = _expect_event_type(events[1], TextStartEvent)
    text_delta = _expect_event_type(events[2], TextDeltaEvent)
    text_end = _expect_event_type(events[3], TextEndEvent)
    done = _expect_event_type(events[4], StreamDoneEvent)
    text_block = _expect_text_block(text_end.block)
    done_text_block = _expect_text_block(done.blocks[0])

    assert [event.type for event in events] == [
        "stream_start",
        "text_start",
        "text_delta",
        "text_end",
        "stream_done",
    ]
    assert text_start.content_index == 0
    assert text_delta.content_index == 0
    assert text_end.content_index == 0
    assert text_delta.delta == "No"
    assert text_block.text == "No thanks"
    assert done_text_block.text == "No thanks"


def test_assemble_stream_waits_for_supported_text_part_before_deltas() -> None:
    """Starts text blocks from message events but waits for a supported part."""

    events = _collect_stream_events(
        [
            _created_event("resp_unsupported_text_part"),
            _message_added_event("msg_unsupported_text_part"),
            _message_text_part_event(None),
            _message_text_delta_event("output_text", "Ignored"),
            _message_text_part_event("output_text"),
            _message_text_delta_event("output_text", "Shown"),
            _message_done_event("msg_unsupported_text_part", "Shown"),
            _completed_event("stop"),
        ]
    )

    text_start = _expect_event_type(events[1], TextStartEvent)
    text_delta = _expect_event_type(events[2], TextDeltaEvent)
    text_end = _expect_event_type(events[3], TextEndEvent)
    done = _expect_event_type(events[4], StreamDoneEvent)

    assert [event.type for event in events] == [
        "stream_start",
        "text_start",
        "text_delta",
        "text_end",
        "stream_done",
    ]
    assert text_start.content_index == 0
    assert text_delta.content_index == 0
    assert text_delta.delta == "Shown"
    assert _expect_text_block(text_end.block).text == "Shown"
    assert _expect_text_block(done.blocks[0]).text == "Shown"


def test_assemble_stream_maps_tool_call_events() -> None:
    """Accumulates tool-call events onto terminal stream blocks."""

    events = _collect_stream_events(
        [
            _created_event("resp_tool_call"),
            _tool_call_added_event(
                provider_item_id="fc_123",
                call_id="call_123",
                name="get_weather",
                arguments={},
            ),
            _tool_call_arguments_delta_event('{"'),
            _tool_call_arguments_delta_event('city":"Munich"}'),
            _tool_call_arguments_done_event({"city": "Munich"}),
            _tool_call_done_event(
                provider_item_id="fc_123",
                call_id="call_123",
                name="get_weather",
                arguments={"city": "Munich"},
            ),
            _completed_event("tool_use"),
        ]
    )

    tool_call_start = _expect_event_type(events[1], ToolCallStartEvent)
    tool_call_delta_one = _expect_event_type(events[2], ToolCallDeltaEvent)
    tool_call_delta_two = _expect_event_type(events[3], ToolCallDeltaEvent)
    tool_call_end = _expect_event_type(events[4], ToolCallEndEvent)
    done = _expect_event_type(events[5], StreamDoneEvent)
    tool_call_block = _expect_tool_call_block(tool_call_end.block)
    done_tool_call_block = _expect_tool_call_block(done.blocks[0])

    assert [event.type for event in events] == [
        "stream_start",
        "tool_call_start",
        "tool_call_delta",
        "tool_call_delta",
        "tool_call_end",
        "stream_done",
    ]
    assert tool_call_start.content_index == 0
    assert tool_call_delta_one.content_index == 0
    assert tool_call_delta_two.content_index == 0
    assert tool_call_end.content_index == 0
    assert tool_call_delta_one.delta == '{"'
    assert tool_call_delta_two.delta == 'city":"Munich"}'
    assert tool_call_block.call_id == "call_123"
    assert tool_call_block.name == "get_weather"
    assert _expect_metadata_string(tool_call_block, "provider_item_id") == "fc_123"
    assert tool_call_block.arguments == {"city": "Munich"}
    assert done.stop_reason == "tool_use"
    assert done_tool_call_block.arguments == {"city": "Munich"}


def test_assemble_stream_ignores_text_deltas_when_refusal_part_is_active() -> None:
    """Ignores output-text deltas while a refusal part is active."""

    events = _collect_stream_events(
        [
            _created_event("resp_refusal"),
            _message_added_event("msg_refusal"),
            _message_text_part_event("refusal"),
            _message_text_delta_event("output_text", "Wrong"),
            _message_text_delta_event("refusal", "No"),
            _message_text_delta_event("refusal", " thanks"),
            _message_done_event("msg_refusal", "No thanks"),
            _completed_event("stop"),
        ]
    )

    text_delta_one = _expect_event_type(events[2], TextDeltaEvent)
    text_delta_two = _expect_event_type(events[3], TextDeltaEvent)
    done = _expect_event_type(events[5], StreamDoneEvent)

    assert [event.type for event in events] == [
        "stream_start",
        "text_start",
        "text_delta",
        "text_delta",
        "text_end",
        "stream_done",
    ]
    assert text_delta_one.delta == "No"
    assert text_delta_two.delta == " thanks"
    assert _expect_text_block(done.blocks[0]).text == "No thanks"


def test_assemble_stream_clears_active_text_mode_for_unsupported_parts() -> None:
    """Clears text accumulation when the current content part becomes unsupported."""

    events = _collect_stream_events(
        [
            _created_event("resp_unsupported_part"),
            _message_added_event("msg_unsupported_part"),
            _message_text_part_event("output_text"),
            _message_text_part_event(None),
            _message_text_delta_event("output_text", "Should be ignored"),
            _message_done_event("msg_unsupported_part", ""),
            _completed_event("stop"),
        ]
    )

    text_end = _expect_event_type(events[2], TextEndEvent)
    done = _expect_event_type(events[3], StreamDoneEvent)

    assert [event.type for event in events] == [
        "stream_start",
        "text_start",
        "text_end",
        "stream_done",
    ]
    assert _expect_text_block(text_end.block).text == ""
    assert _expect_text_block(done.blocks[0]).text == ""


def test_assemble_stream_maps_failed_response_into_error_event() -> None:
    """Builds an error stream event for failed responses."""

    events = _collect_stream_events(
        [
            _created_event("resp_failed"),
            _failed_event("Model overloaded"),
        ]
    )

    error = _expect_event_type(events[1], StreamErrorEvent)

    assert [event.type for event in events] == ["stream_start", "stream_error"]
    assert error.error_message == "Model overloaded"
    assert error.stop_reason == "error"
    assert error.response_id == "resp_failed"


def test_assemble_stream_maps_incomplete_length_into_done() -> None:
    """Builds a done event for non-error incomplete responses."""

    events = _collect_stream_events(
        [
            _created_event("resp_incomplete"),
            _message_added_event("msg_incomplete"),
            _message_text_part_event("output_text"),
            _message_text_delta_event("output_text", "Partial answer"),
            _message_done_event("msg_incomplete", "Partial answer"),
            _incomplete_event("length", "OpenAI response incomplete."),
        ]
    )

    done = _expect_event_type(events[-1], StreamDoneEvent)

    assert [event.type for event in events] == [
        "stream_start",
        "text_start",
        "text_delta",
        "text_end",
        "stream_done",
    ]
    assert done.stop_reason == "length"
    assert _expect_text_block(done.blocks[0]).text == "Partial answer"


def test_assemble_stream_maps_incomplete_error_into_error_event() -> None:
    """Builds an error event for incomplete responses with an error stop reason."""

    events = _collect_stream_events(
        [
            _created_event("resp_filtered"),
            _incomplete_event(
                "error",
                "OpenAI response was truncated by the content filter.",
            ),
        ]
    )

    error = _expect_event_type(events[1], StreamErrorEvent)

    assert [event.type for event in events] == ["stream_start", "stream_error"]
    assert error.error_message == "OpenAI response was truncated by the content filter."
    assert error.stop_reason == "error"


def test_assemble_stream_stops_consuming_events_after_terminal_event() -> None:
    """Stops assembly once a terminal normalized event has been emitted."""

    events = _collect_stream_events(
        [
            _created_event("resp_done"),
            _completed_event("stop"),
            _message_added_event("msg_after_done"),
            _message_text_part_event("output_text"),
            _message_text_delta_event("output_text", "ignored"),
        ]
    )

    done = _expect_event_type(events[1], StreamDoneEvent)

    assert [event.type for event in events] == ["stream_start", "stream_done"]
    assert done.response_id == "resp_done"
    assert done.blocks == []


def _collect_stream_events(
    normalized_events: Sequence[NormalizedEvent],
) -> list[ProviderStreamEvent]:
    """Collects stream events emitted by the assembler."""

    async def _collect() -> list[ProviderStreamEvent]:
        return [
            event
            async for event in assemble_stream(
                async_stream(normalized_events),
                source=_source(),
            )
        ]

    return asyncio.run(_collect())


def _source() -> ProviderSource:
    """Build a deterministic provider source for assembler tests."""

    return ProviderSource(provider="openai", model="gpt-5.4")


def _created_event(response_id: str) -> CreatedNormalizedEvent:
    """Builds a created normalized event."""

    return {
        "type": NormalizedEventType.CREATED,
        "response_id": response_id,
    }


def _reasoning_added_event(item_id: str) -> ReasoningAddedNormalizedEvent:
    """Builds a reasoning-added normalized event."""

    return {
        "type": NormalizedEventType.REASONING_ADDED,
        "item_id": item_id,
    }


def _reasoning_delta_event(delta: str) -> ReasoningDeltaNormalizedEvent:
    """Builds a reasoning-delta normalized event."""

    return {
        "type": NormalizedEventType.REASONING_DELTA,
        "delta": delta,
    }


def _reasoning_done_event(
    item_id: str,
    summary_text: str,
    reasoning_signature: str | None,
) -> ReasoningDoneNormalizedEvent:
    """Builds a reasoning-done normalized event."""

    return {
        "type": NormalizedEventType.REASONING_DONE,
        "item_id": item_id,
        "summary_text": summary_text,
        "reasoning_signature": reasoning_signature,
    }


def _message_added_event(
    item_id: str,
    phase: Phase | None = None,
) -> MessageAddedNormalizedEvent:
    """Builds a message-added normalized event."""

    return {
        "type": NormalizedEventType.MESSAGE_ADDED,
        "item_id": item_id,
        "phase": phase,
    }


def _message_text_part_event(
    part_type: TextPartType | None,
) -> MessageTextPartNormalizedEvent:
    """Builds a message text-part normalized event."""

    return {
        "type": NormalizedEventType.MESSAGE_TEXT_PART,
        "part_type": part_type,
    }


def _message_text_delta_event(
    part_type: TextPartType,
    delta: str,
) -> MessageTextDeltaNormalizedEvent:
    """Builds a message text-delta normalized event."""

    return {
        "type": NormalizedEventType.MESSAGE_TEXT_DELTA,
        "part_type": part_type,
        "delta": delta,
    }


def _message_done_event(
    item_id: str,
    text: str,
    phase: Phase | None = None,
) -> MessageDoneNormalizedEvent:
    """Builds a message-done normalized event."""

    return {
        "type": NormalizedEventType.MESSAGE_DONE,
        "item_id": item_id,
        "text": text,
        "phase": phase,
    }


def _tool_call_added_event(
    provider_item_id: str | None,
    call_id: str,
    name: str,
    arguments: JsonObject,
) -> ToolCallAddedNormalizedEvent:
    """Builds a tool-call added normalized event."""

    return {
        "type": NormalizedEventType.TOOL_CALL_ADDED,
        "provider_item_id": provider_item_id,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def _tool_call_arguments_delta_event(
    delta: str,
) -> ToolCallArgumentsDeltaNormalizedEvent:
    """Builds a tool-call arguments delta normalized event."""

    return {
        "type": NormalizedEventType.TOOL_CALL_ARGUMENTS_DELTA,
        "delta": delta,
    }


def _tool_call_arguments_done_event(
    arguments: JsonObject,
) -> ToolCallArgumentsDoneNormalizedEvent:
    """Builds a tool-call arguments done normalized event."""

    return {
        "type": NormalizedEventType.TOOL_CALL_ARGUMENTS_DONE,
        "arguments": arguments,
    }


def _tool_call_done_event(
    provider_item_id: str | None,
    call_id: str,
    name: str,
    arguments: JsonObject,
) -> ToolCallDoneNormalizedEvent:
    """Builds a tool-call done normalized event."""

    return {
        "type": NormalizedEventType.TOOL_CALL_DONE,
        "provider_item_id": provider_item_id,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def _completed_event(stop_reason: StopReason) -> CompletedNormalizedEvent:
    """Builds a completed normalized event."""

    return {
        "type": NormalizedEventType.COMPLETED,
        "stop_reason": stop_reason,
    }


def _incomplete_event(
    stop_reason: StopReason,
    error_message: str | None,
) -> IncompleteNormalizedEvent:
    """Builds an incomplete normalized event."""

    return {
        "type": NormalizedEventType.INCOMPLETE,
        "stop_reason": stop_reason,
        "error_message": error_message,
    }


def _failed_event(message: str) -> FailedNormalizedEvent:
    """Builds a failed normalized event."""

    return {
        "type": NormalizedEventType.FAILED,
        "message": message,
    }
