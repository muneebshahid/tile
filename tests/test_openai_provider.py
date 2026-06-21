"""Tests for OpenAI provider stream integration.

These tests document the first half of the streaming lifecycle:

1. Raw OpenAI SDK events or ChatGPT subscription SSE payloads are created in the
   test itself.
2. The provider passes those raw events through the matching adapter.
3. The adapter emits normalized events, and ``assemble_stream`` turns them into
   app-level ``StreamEvent`` models.

The expected ``StreamEvent`` order in each test is the executable spec for how
raw OpenAI events correspond to application stream events.
"""

import asyncio
import json
from collections.abc import Sequence
from typing import TypeVar
from unittest.mock import patch

import pytest

from ai.openai.provider import stream_api, stream_subscription
from ai.openai.serialization import serialize_history_items
from ai.openai.subscription_event_adapter import SubscriptionEventPayload
from ai.types.conversation import UserMessage
from ai.types.stream_events import (
    AssistantBlock,
    ProviderStreamEvent,
    ReasoningBlock,
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    ReasoningStartEvent,
    StreamDoneEvent,
    StreamErrorEvent,
    StreamEvent,
    StreamStartEvent,
    TextBlock,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolCallBlock,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from ai.types.tools import ToolDefinition, ToolResult
from tests.support.async_streams import async_stream
from tests.support.openai_response_events import (
    FakeOpenAIClient,
    build_fake_openai_client as _build_client,
    content_part_added_event as _content_part_added_event,
    function_tool_call_added_event as _function_tool_call_added_event,
    function_tool_call_arguments_delta_event as _function_tool_call_arguments_delta_event,
    function_tool_call_arguments_done_event as _function_tool_call_arguments_done_event,
    function_tool_call_done_event as _function_tool_call_done_event,
    message_added_event as _message_added_event,
    message_done_event as _message_done_event,
    reasoning_added_event as _reasoning_added_event,
    reasoning_done_event as _reasoning_done_event,
    reasoning_summary_delta_event as _reasoning_summary_delta_event,
    reasoning_summary_part_added_event as _reasoning_summary_part_added_event,
    reasoning_summary_part_done_event as _reasoning_summary_part_done_event,
    refusal_delta_event as _refusal_delta_event,
    response_completed_event as _completed_event,
    response_created_event as _created_event,
    response_error_event as _error_event,
    response_failed_event as _failed_event,
    response_incomplete_event as _incomplete_event,
    text_delta_event as _text_delta_event,
    unsupported_content_part_added_event as _unsupported_content_part_added_event,
)

TEvent = TypeVar("TEvent", bound=StreamEvent)


def _expect_event_type(event: StreamEvent, event_type: type[TEvent]) -> TEvent:
    assert isinstance(event, event_type)
    return event


def _expect_reasoning_block(
    block: TextBlock | ReasoningBlock | ToolCallBlock,
) -> ReasoningBlock:
    assert isinstance(block, ReasoningBlock)
    return block


def _expect_text_block(
    block: TextBlock | ReasoningBlock | ToolCallBlock,
) -> TextBlock:
    assert isinstance(block, TextBlock)
    return block


def _expect_tool_call_block(
    block: TextBlock | ReasoningBlock | ToolCallBlock,
) -> ToolCallBlock:
    assert isinstance(block, ToolCallBlock)
    return block


def _expect_metadata_string(
    block: AssistantBlock,
    key: str,
) -> str:
    value = block.metadata_string(key)
    assert value is not None
    return value


def _collect_events(
    client: FakeOpenAIClient,
    tools: Sequence[ToolDefinition] | None = None,
) -> list[ProviderStreamEvent]:
    async def _collect() -> list[ProviderStreamEvent]:
        event_stream = await stream_api(
            history=[UserMessage(content="hello")],
            model="gpt-5.4",
            reasoning={"effort": "medium"},
            instructions="Follow the repo conventions.",
            tools=tools,
        )
        return [event async for event in event_stream]

    with patch("ai.openai.provider.create_client", return_value=client):
        return asyncio.run(_collect())


def _sample_tools() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="get_weather",
            description="Return a simple weather report for a city.",
            input_schema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to look up.",
                    }
                },
                "required": ["city"],
                "additionalProperties": False,
            },
            fn=_sample_tool_fn,
        )
    ]


async def _sample_tool_fn(city: str) -> ToolResult:
    """Return a deterministic payload for provider-only tool definitions."""

    return ToolResult.text(f"city={city}")


def test_stream_maps_raw_events_into_block_stream() -> None:
    raw_events = [
        _created_event(1, "resp_success"),
        _reasoning_added_event(2, "rs_123", output_index=0),
        _reasoning_summary_part_added_event(3, "rs_123", 0, output_index=0),
        _reasoning_summary_delta_event(
            4,
            "rs_123",
            0,
            "Exploring ",
            output_index=0,
        ),
        _reasoning_summary_delta_event(
            5,
            "rs_123",
            0,
            "reasoning traces",
            output_index=0,
        ),
        _reasoning_summary_part_done_event(
            6,
            "rs_123",
            0,
            "Exploring reasoning traces",
            output_index=0,
        ),
        _reasoning_summary_part_added_event(7, "rs_123", 1, output_index=0),
        _reasoning_summary_delta_event(
            8,
            "rs_123",
            1,
            "Formulating ",
            output_index=0,
        ),
        _reasoning_summary_delta_event(
            9,
            "rs_123",
            1,
            "reasoning traces",
            output_index=0,
        ),
        _reasoning_done_event(
            10,
            "rs_123",
            [
                "Exploring reasoning traces",
                "Formulating reasoning traces",
            ],
            output_index=0,
        ),
        _message_added_event(11, "msg_123", output_index=1),
        _content_part_added_event(
            12,
            "msg_123",
            "output_text",
            output_index=1,
            content_index=0,
        ),
        _text_delta_event(
            13,
            "msg_123",
            "Hello",
            output_index=1,
            content_index=0,
        ),
        _text_delta_event(
            14,
            "msg_123",
            " world",
            output_index=1,
            content_index=0,
        ),
        _message_done_event(
            15,
            "msg_123",
            [
                {
                    "type": "output_text",
                    "text": "Hello world",
                    "annotations": [],
                }
            ],
            output_index=1,
        ),
        _completed_event(16, "resp_success"),
    ]

    client = _build_client(raw_events)
    events = _collect_events(client)

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
    assert start.source.provider == "openai"
    assert start.source.model == "gpt-5.4"
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
    assert text_delta_one.delta == "Hello"
    assert text_delta_two.delta == " world"
    assert final_text_block.text == "Hello world"
    assert done.response_id == "resp_success"
    assert done.source.provider == "openai"
    assert done.source.model == "gpt-5.4"

    assert (
        done_reasoning_block.summary_text
        == "Exploring reasoning traces\n\nFormulating reasoning traces"
    )
    done_reasoning_signature = _expect_metadata_string(
        done_reasoning_block,
        "reasoning_signature",
    )
    final_reasoning_signature = _expect_metadata_string(
        final_reasoning_block,
        "reasoning_signature",
    )
    assert done_reasoning_signature == final_reasoning_signature
    assert json.loads(done_reasoning_signature) == {
        "id": "rs_123",
        "type": "reasoning",
        "summary": [
            {"type": "summary_text", "text": "Exploring reasoning traces"},
            {"type": "summary_text", "text": "Formulating reasoning traces"},
        ],
        "status": "completed",
    }
    assert done_text_block.text == "Hello world"
    client.responses.create.assert_awaited_once_with(
        model="gpt-5.4",
        input=serialize_history_items([UserMessage(content="hello")]),
        instructions="Follow the repo conventions.",
        stream=True,
        reasoning={"effort": "medium"},
    )


def test_stream_preserves_reasoning_deltas_when_done_summary_is_empty() -> None:
    raw_events = [
        _created_event(1, "resp_reasoning_empty_done"),
        _reasoning_added_event(2, "rs_123", output_index=0),
        _reasoning_summary_delta_event(
            3,
            "rs_123",
            0,
            "Draft summary",
            output_index=0,
        ),
        _reasoning_done_event(4, "rs_123", [], output_index=0),
        _completed_event(5, "resp_reasoning_empty_done"),
    ]

    client = _build_client(raw_events)
    events = _collect_events(client)
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


def test_stream_passes_serialized_tools_when_provided() -> None:
    client = _build_client([_completed_event(1, "resp_tools")])

    _collect_events(client, tools=_sample_tools())

    client.responses.create.assert_awaited_once_with(
        model="gpt-5.4",
        input=serialize_history_items([UserMessage(content="hello")]),
        instructions="Follow the repo conventions.",
        stream=True,
        reasoning={"effort": "medium"},
        tools=[
            {
                "type": "function",
                "name": "get_weather",
                "description": "Return a simple weather report for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to look up.",
                        }
                    },
                    "required": ["city"],
                    "additionalProperties": False,
                },
                "strict": True,
                "defer_loading": False,
            }
        ],
    )


def test_stream_maps_refusal_deltas() -> None:
    raw_events = [
        _created_event(1, "resp_refusal"),
        _message_added_event(2, "msg_refusal", output_index=0),
        _content_part_added_event(
            3,
            "msg_refusal",
            "refusal",
            output_index=0,
            content_index=0,
        ),
        _refusal_delta_event(
            4,
            "msg_refusal",
            "No",
            output_index=0,
            content_index=0,
        ),
        _message_done_event(
            5,
            "msg_refusal",
            [{"type": "refusal", "refusal": "No thanks"}],
            output_index=0,
        ),
        _completed_event(6, "resp_refusal"),
    ]

    client = _build_client(raw_events)
    events = _collect_events(client)
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


def test_stream_maps_function_tool_call_events() -> None:
    raw_events = [
        _created_event(1, "resp_tool_call"),
        _function_tool_call_added_event(
            2,
            "fc_123",
            "call_123",
            "get_weather",
            output_index=0,
        ),
        _function_tool_call_arguments_delta_event(
            3,
            "fc_123",
            '{"',
            output_index=0,
        ),
        _function_tool_call_arguments_delta_event(
            4,
            "fc_123",
            'city":"Munich"}',
            output_index=0,
        ),
        _function_tool_call_arguments_done_event(
            5,
            "fc_123",
            '{"city":"Munich"}',
            output_index=0,
        ),
        _function_tool_call_done_event(
            6,
            "fc_123",
            "call_123",
            "get_weather",
            '{"city":"Munich"}',
            output_index=0,
        ),
        _completed_event(
            7,
            "resp_tool_call",
            output=[
                {
                    "id": "fc_123",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": "call_123",
                    "name": "get_weather",
                    "arguments": '{"city":"Munich"}',
                }
            ],
        ),
    ]

    client = _build_client(raw_events)
    events = _collect_events(client)
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


def test_stream_ignores_text_deltas_when_refusal_part_is_active() -> None:
    raw_events = [
        _created_event(1, "resp_refusal"),
        _message_added_event(2, "msg_refusal", output_index=0),
        _content_part_added_event(
            3,
            "msg_refusal",
            "refusal",
            output_index=0,
            content_index=0,
        ),
        _text_delta_event(
            4,
            "msg_refusal",
            "Wrong",
            output_index=0,
            content_index=0,
        ),
        _refusal_delta_event(
            5,
            "msg_refusal",
            "No",
            output_index=0,
            content_index=0,
        ),
        _refusal_delta_event(
            6,
            "msg_refusal",
            " thanks",
            output_index=0,
            content_index=0,
        ),
        _message_done_event(
            7,
            "msg_refusal",
            [{"type": "refusal", "refusal": "No thanks"}],
            output_index=0,
        ),
        _completed_event(8, "resp_refusal"),
    ]

    client = _build_client(raw_events)
    events = _collect_events(client)
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


def test_stream_clears_active_text_mode_for_unsupported_content_parts() -> None:
    raw_events = [
        _created_event(1, "resp_unsupported_part"),
        _message_added_event(2, "msg_unsupported_part", output_index=0),
        _content_part_added_event(
            3,
            "msg_unsupported_part",
            "output_text",
            output_index=0,
            content_index=0,
        ),
        _unsupported_content_part_added_event(
            4,
            "msg_unsupported_part",
            output_index=0,
            content_index=1,
        ),
        _text_delta_event(
            5,
            "msg_unsupported_part",
            "Should be ignored",
            output_index=0,
            content_index=1,
        ),
        _message_done_event(
            6,
            "msg_unsupported_part",
            [{"type": "output_text", "text": "", "annotations": []}],
            output_index=0,
        ),
        _completed_event(7, "resp_unsupported_part"),
    ]

    client = _build_client(raw_events)
    events = _collect_events(client)
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


def test_stream_maps_failed_response_into_error_event() -> None:
    raw_events = [
        _created_event(1, "resp_failed"),
        _failed_event(2, "resp_failed", "Model overloaded"),
    ]

    client = _build_client(raw_events)
    events = _collect_events(client)
    error = _expect_event_type(events[1], StreamErrorEvent)

    assert [event.type for event in events] == ["stream_start", "stream_error"]
    assert error.error_message == "Model overloaded"
    assert error.stop_reason == "error"
    assert error.response_id == "resp_failed"


def test_stream_maps_error_event_into_error_event() -> None:
    raw_events = [
        _created_event(1, "resp_error"),
        _error_event(2, "Socket closed"),
    ]

    client = _build_client(raw_events)
    events = _collect_events(client)
    error = _expect_event_type(events[1], StreamErrorEvent)

    assert [event.type for event in events] == ["stream_start", "stream_error"]
    assert error.error_message == "Socket closed"
    assert error.stop_reason == "error"
    assert error.response_id == "resp_error"


def test_stream_maps_incomplete_max_output_tokens_into_length_done() -> None:
    raw_events = [
        _created_event(1, "resp_incomplete"),
        _message_added_event(2, "msg_incomplete", output_index=0),
        _content_part_added_event(
            3,
            "msg_incomplete",
            "output_text",
            output_index=0,
            content_index=0,
        ),
        _text_delta_event(
            4,
            "msg_incomplete",
            "Partial answer",
            output_index=0,
            content_index=0,
        ),
        _message_done_event(
            5,
            "msg_incomplete",
            [{"type": "output_text", "text": "Partial answer", "annotations": []}],
            output_index=0,
        ),
        _incomplete_event(6, "resp_incomplete", "max_output_tokens"),
    ]

    client = _build_client(raw_events)
    events = _collect_events(client)
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


def test_stream_maps_incomplete_content_filter_into_error_event() -> None:
    raw_events = [
        _created_event(1, "resp_filtered"),
        _incomplete_event(2, "resp_filtered", "content_filter"),
    ]

    client = _build_client(raw_events)
    events = _collect_events(client)
    error = _expect_event_type(events[1], StreamErrorEvent)

    assert [event.type for event in events] == ["stream_start", "stream_error"]
    assert error.error_message == "OpenAI response was truncated by the content filter."
    assert error.stop_reason == "error"


def test_stream_subscription_maps_raw_events_into_stream_events() -> None:
    raw_events: list[SubscriptionEventPayload] = [
        {
            "type": "response.created",
            "response": {"id": "resp_subscription"},
        },
        {
            "type": "response.output_item.added",
            "item": {
                "id": "msg_subscription",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        },
        {
            "type": "response.content_part.added",
            "item_id": "msg_subscription",
            "part": {
                "type": "output_text",
                "text": "",
                "annotations": [],
            },
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_subscription",
            "delta": "Hello from subscription",
        },
        {
            "type": "response.output_item.done",
            "item": {
                "id": "msg_subscription",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Hello from subscription",
                        "annotations": [],
                    }
                ],
            },
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp_subscription",
                "status": "completed",
                "output": [],
            },
        },
    ]

    async def _collect() -> list[ProviderStreamEvent]:
        event_stream = await stream_subscription(
            history=[UserMessage(content="hello")],
            model="gpt-5.4",
            reasoning={"effort": "medium"},
            instructions="Follow the repo conventions.",
            raw_stream=async_stream(raw_events),
        )
        return [event async for event in event_stream]

    events = asyncio.run(_collect())
    start = _expect_event_type(events[0], StreamStartEvent)
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
    assert start.response_id == "resp_subscription"
    assert text_start.content_index == 0
    assert text_delta.delta == "Hello from subscription"
    assert _expect_text_block(text_end.block).text == "Hello from subscription"
    assert done.response_id == "resp_subscription"


def test_stream_subscription_raises_until_transport_is_implemented() -> None:
    async def _collect() -> None:
        await stream_subscription(
            history=[UserMessage(content="hello")],
            model="gpt-5.4",
            instructions="Follow the repo conventions.",
        )

    with pytest.raises(
        NotImplementedError,
        match="Subscription transport is not implemented yet.",
    ):
        asyncio.run(_collect())
