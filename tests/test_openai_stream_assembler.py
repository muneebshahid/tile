"""Tests for assembling normalized provider events into stream events."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import TypeVar, cast

from ai.openai.normalized_events import (
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
from ai.openai.stream_assembler import assemble_stream
from ai.types.stream import (
    Phase,
    ReasoningBlock,
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    ReasoningStartEvent,
    StreamDoneEvent,
    StreamErrorEvent,
    StreamEvent,
    StreamStartEvent,
    StopReason,
    TextBlock,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolCallBlock,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from ai.types.tools import JsonObject

TEvent = TypeVar("TEvent", bound=StreamEvent)


def test_assemble_stream_accumulates_reasoning_and_text_with_shared_message_state() -> (
    None
):
    """Accumulates reasoning and text blocks onto a shared message message."""

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
    shared_message = reasoning_start.message
    final_reasoning_block = _expect_reasoning_block(shared_message.blocks[0])
    final_text_block = _expect_text_block(shared_message.blocks[1])
    done_reasoning_block = _expect_reasoning_block(done.message.blocks[0])
    done_text_block = _expect_text_block(done.message.blocks[1])

    assert [event.type for event in events] == [
        "start",
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
        "done",
    ]
    assert start.message.response_id == "resp_success"
    assert reasoning_start.message.response_id == "resp_success"
    assert start.message is shared_message
    assert reasoning_delta_one.message is shared_message
    assert reasoning_delta_two.message is shared_message
    assert reasoning_delta_separator.message is shared_message
    assert reasoning_delta_three.message is shared_message
    assert reasoning_delta_four.message is shared_message
    assert reasoning_end.message is shared_message
    assert text_start.message is shared_message
    assert text_delta_one.message is shared_message
    assert text_delta_two.message is shared_message
    assert text_end.message is shared_message
    assert done.message is shared_message
    assert reasoning_delta_one.delta == "Exploring "
    assert reasoning_delta_two.delta == "reasoning traces"
    assert reasoning_delta_separator.delta == "\n\n"
    assert reasoning_delta_three.delta == "Formulating "
    assert reasoning_delta_four.delta == "reasoning traces"
    assert (
        final_reasoning_block.summary_text
        == "Exploring reasoning traces\n\nFormulating reasoning traces"
    )
    assert final_reasoning_block.reasoning_signature == '{"id":"rs_123"}'
    assert text_delta_one.delta == "Hello"
    assert text_delta_two.delta == " world"
    assert final_text_block.text == "Hello world"
    assert done.message.response_id == "resp_success"
    assert (
        done_reasoning_block.summary_text
        == "Exploring reasoning traces\n\nFormulating reasoning traces"
    )
    assert done_reasoning_block.reasoning_signature == '{"id":"rs_123"}'
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
    reasoning_block = _expect_reasoning_block(reasoning_end.message.blocks[0])
    done_reasoning_block = _expect_reasoning_block(done.message.blocks[0])

    assert [event.type for event in events] == [
        "start",
        "reasoning_start",
        "reasoning_delta",
        "reasoning_end",
        "done",
    ]
    assert reasoning_block.summary_text == "Draft summary"
    assert done_reasoning_block.summary_text == "Draft summary"


def test_assemble_stream_maps_refusal_deltas_with_shared_message_state() -> None:
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
    shared_message = text_start.message
    text_block = _expect_text_block(shared_message.blocks[0])
    done_text_block = _expect_text_block(done.message.blocks[0])

    assert [event.type for event in events] == [
        "start",
        "text_start",
        "text_delta",
        "text_end",
        "done",
    ]
    assert text_start.message is shared_message
    assert text_delta.message is shared_message
    assert text_end.message is shared_message
    assert text_delta.delta == "No"
    assert text_block.text == "No thanks"
    assert done.message is shared_message
    assert done_text_block.text == "No thanks"


def test_assemble_stream_maps_tool_call_events_with_shared_message_state() -> None:
    """Accumulates tool-call events onto the shared message message."""

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
    shared_message = tool_call_start.message
    tool_call_block = _expect_tool_call_block(shared_message.blocks[0])
    done_tool_call_block = _expect_tool_call_block(done.message.blocks[0])

    assert [event.type for event in events] == [
        "start",
        "tool_call_start",
        "tool_call_delta",
        "tool_call_delta",
        "tool_call_end",
        "done",
    ]
    assert tool_call_start.message is shared_message
    assert tool_call_delta_one.message is shared_message
    assert tool_call_delta_two.message is shared_message
    assert tool_call_end.message is shared_message
    assert tool_call_delta_one.delta == '{"'
    assert tool_call_delta_two.delta == 'city":"Munich"}'
    assert tool_call_block.call_id == "call_123"
    assert tool_call_block.name == "get_weather"
    assert tool_call_block.provider_item_id == "fc_123"
    assert tool_call_block.arguments == {"city": "Munich"}
    assert done.message.stop_reason == "tool_use"
    assert done.message is shared_message
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
        "start",
        "text_start",
        "text_delta",
        "text_delta",
        "text_end",
        "done",
    ]
    assert text_delta_one.delta == "No"
    assert text_delta_two.delta == " thanks"
    assert _expect_text_block(done.message.blocks[0]).text == "No thanks"


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
        "start",
        "text_start",
        "text_end",
        "done",
    ]
    assert _expect_text_block(text_end.message.blocks[0]).text == ""
    assert _expect_text_block(done.message.blocks[0]).text == ""


def test_assemble_stream_maps_failed_response_into_error_event() -> None:
    """Builds an error stream event for failed responses."""

    events = _collect_stream_events(
        [
            _created_event("resp_failed"),
            _failed_event("Model overloaded"),
        ]
    )

    error = _expect_event_type(events[1], StreamErrorEvent)

    assert [event.type for event in events] == ["start", "error"]
    assert error.error.error_message == "Model overloaded"
    assert error.error.stop_reason == "error"
    assert error.error.response_id == "resp_failed"


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
        "start",
        "text_start",
        "text_delta",
        "text_end",
        "done",
    ]
    assert done.message.stop_reason == "length"
    assert _expect_text_block(done.message.blocks[0]).text == "Partial answer"


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

    assert [event.type for event in events] == ["start", "error"]
    assert (
        error.error.error_message
        == "OpenAI response was truncated by the content filter."
    )
    assert error.error.stop_reason == "error"


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

    assert [event.type for event in events] == ["start", "done"]
    assert done.message.response_id == "resp_done"
    assert done.message.blocks == []


def _collect_stream_events(
    normalized_events: Sequence[NormalizedEvent],
) -> list[StreamEvent]:
    """Collects stream events emitted by the assembler."""

    async def _collect() -> list[StreamEvent]:
        return [
            event
            async for event in assemble_stream(_normalized_stream(normalized_events))
        ]

    return asyncio.run(_collect())


def _normalized_stream(
    normalized_events: Sequence[NormalizedEvent],
) -> AsyncIterator[NormalizedEvent]:
    """Yield normalized provider events from a static sequence."""

    async def _iterate() -> AsyncIterator[NormalizedEvent]:
        for normalized_event in normalized_events:
            yield normalized_event

    return _iterate()


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


def _expect_event_type(event: StreamEvent, event_type: type[TEvent]) -> TEvent:
    """Casts a stream event to the expected runtime type."""

    assert isinstance(event, event_type)
    return cast(TEvent, event)


def _expect_reasoning_block(
    block: TextBlock | ReasoningBlock | ToolCallBlock,
) -> ReasoningBlock:
    """Casts an assistant block to a reasoning block."""

    assert isinstance(block, ReasoningBlock)
    return block


def _expect_text_block(
    block: TextBlock | ReasoningBlock | ToolCallBlock,
) -> TextBlock:
    """Casts an assistant block to a text block."""

    assert isinstance(block, TextBlock)
    return block


def _expect_tool_call_block(
    block: TextBlock | ReasoningBlock | ToolCallBlock,
) -> ToolCallBlock:
    """Casts an assistant block to a tool-call block."""

    assert isinstance(block, ToolCallBlock)
    return block
