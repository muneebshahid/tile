"""Tests for OpenAI provider stream integration.

These tests document the first half of the streaming lifecycle:

1. Raw OpenAI SDK events or ChatGPT subscription SSE payloads are created in the
   test itself.
2. The provider passes those raw events through the matching adapter.
3. The adapter emits normalized events, and ``assemble_stream`` turns them into
   app-level ``StreamEvent`` models.

The focused adapter and assembler tests own the detailed event matrix. This file
keeps provider coverage at the transport wiring boundary.
"""

import asyncio
from collections.abc import Sequence
from typing import cast

import pytest
from openai import AsyncOpenAI

from ori.providers.openai.provider import create_stream_api, stream_subscription
from ori.providers.openai.serialization import serialize_history_items
from ori.providers.openai.subscription_event_adapter import SubscriptionEventPayload
from ori.types.conversation import UserMessage
from ori.types.stream_events import (
    ProviderStreamEvent,
    StreamDoneEvent,
    StreamStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
)
from ori.types.tools import ToolDefinition, ToolResult
from tests.support.async_streams import async_stream
from tests.support.openai_response_events import (
    FakeOpenAIClient,
    build_fake_openai_client,
    content_part_added_event,
    message_added_event,
    message_done_event,
    response_completed_event,
    response_created_event,
    text_delta_event,
)
from tests.support.stream_assertions import (
    expect_stream_event as _expect_event_type,
    expect_text_block as _expect_text_block,
)


def _collect_events(
    client: FakeOpenAIClient,
    tools: Sequence[ToolDefinition] | None = None,
) -> list[ProviderStreamEvent]:
    async def _collect() -> list[ProviderStreamEvent]:
        stream_api = create_stream_api(cast("AsyncOpenAI", client))
        event_stream = await stream_api(
            history=[UserMessage(content="hello")],
            model="gpt-5.4",
            reasoning={"effort": "medium"},
            instructions="Follow the repo conventions.",
            tools=tools,
        )
        return [event async for event in event_stream]

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


def test_stream_maps_raw_events_into_text_stream() -> None:
    """Pass raw SDK events through the provider stream pipeline."""

    raw_events = [
        response_created_event(1, "resp_success"),
        message_added_event(2, "msg_123", output_index=0),
        content_part_added_event(3, "msg_123", "output_text", output_index=0),
        text_delta_event(4, "msg_123", "Hello", output_index=0),
        message_done_event(
            5,
            "msg_123",
            [{"type": "output_text", "text": "Hello", "annotations": []}],
            output_index=0,
        ),
        response_completed_event(6, "resp_success"),
    ]

    client = build_fake_openai_client(raw_events)
    events = _collect_events(client)

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
    assert start.response_id == "resp_success"
    assert start.source.provider == "openai"
    assert start.source.model == "gpt-5.4"
    assert text_start.content_index == 0
    assert text_delta.content_index == 0
    assert text_delta.delta == "Hello"
    assert text_end.content_index == 0
    assert _expect_text_block(text_end.block).text == "Hello"
    assert done.response_id == "resp_success"
    assert _expect_text_block(done.blocks[0]).text == "Hello"
    client.responses.create.assert_awaited_once_with(
        model="gpt-5.4",
        input=serialize_history_items([UserMessage(content="hello")]),
        instructions="Follow the repo conventions.",
        stream=True,
        reasoning={"effort": "medium"},
    )


def test_stream_passes_serialized_tools_when_provided() -> None:
    client = build_fake_openai_client([response_completed_event(1, "resp_tools")])

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


def test_stream_subscription_maps_raw_events_into_stream_events() -> None:
    events = _collect_subscription_events(_subscription_text_payloads())

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


def _subscription_text_payloads() -> list[SubscriptionEventPayload]:
    """Build the minimal raw subscription payloads for a final text response."""

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
            },
        },
        {
            "type": "response.content_part.added",
            "part": {"type": "output_text"},
        },
        {
            "type": "response.output_text.delta",
            "delta": "Hello from subscription",
        },
        {
            "type": "response.output_item.done",
            "item": {
                "id": "msg_subscription",
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Hello from subscription",
                    }
                ],
            },
        },
        {
            "type": "response.completed",
            "response": {"output": []},
        },
    ]
    return raw_events


def _collect_subscription_events(
    raw_events: Sequence[SubscriptionEventPayload],
) -> list[ProviderStreamEvent]:
    """Collect provider stream events from raw subscription payloads."""

    async def _collect() -> list[ProviderStreamEvent]:
        event_stream = await stream_subscription(
            history=[UserMessage(content="hello")],
            model="gpt-5.4",
            reasoning={"effort": "medium"},
            instructions="Follow the repo conventions.",
            raw_stream=async_stream(raw_events),
        )
        return [event async for event in event_stream]

    return asyncio.run(_collect())


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
