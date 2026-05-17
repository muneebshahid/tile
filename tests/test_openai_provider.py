import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import TypeAlias, TypeVar, cast
from unittest.mock import AsyncMock, patch

import pytest

from ai.types.conversation import UserMessage
from ai.openai.provider import stream_api, stream_subscription
from ai.openai.subscription_event_adapter import SubscriptionEventPayload
from ai.openai.serialization import serialize_history_items
from ai.types.stream import (
    ReasoningDeltaEvent,
    ReasoningBlock,
    ReasoningEndEvent,
    ReasoningStartEvent,
    StreamDoneEvent,
    StreamErrorEvent,
    StreamEvent,
    StreamStartEvent,
    TextDeltaEvent,
    TextBlock,
    TextEndEvent,
    TextStartEvent,
    ToolCallBlock,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from ai.types.tools import JsonObject, ToolDefinition
from openai.types.responses.response_completed_event import ResponseCompletedEvent
from openai.types.responses.response_content_part_added_event import (
    ResponseContentPartAddedEvent,
)
from openai.types.responses.response_created_event import ResponseCreatedEvent
from openai.types.responses.response_error_event import ResponseErrorEvent
from openai.types.responses.response_failed_event import ResponseFailedEvent
from openai.types.responses.response_function_call_arguments_delta_event import (
    ResponseFunctionCallArgumentsDeltaEvent,
)
from openai.types.responses.response_function_call_arguments_done_event import (
    ResponseFunctionCallArgumentsDoneEvent,
)
from openai.types.responses.response_incomplete_event import ResponseIncompleteEvent
from openai.types.responses.response_output_item_added_event import (
    ResponseOutputItemAddedEvent,
)
from openai.types.responses.response_output_item_done_event import (
    ResponseOutputItemDoneEvent,
)
from openai.types.responses.response_reasoning_summary_part_added_event import (
    ResponseReasoningSummaryPartAddedEvent,
)
from openai.types.responses.response_reasoning_summary_part_done_event import (
    ResponseReasoningSummaryPartDoneEvent,
)
from openai.types.responses.response_reasoning_summary_text_delta_event import (
    ResponseReasoningSummaryTextDeltaEvent,
)
from openai.types.responses.response_refusal_delta_event import (
    ResponseRefusalDeltaEvent,
)
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent

RawResponseEvent: TypeAlias = (
    ResponseCompletedEvent
    | ResponseContentPartAddedEvent
    | ResponseCreatedEvent
    | ResponseErrorEvent
    | ResponseFailedEvent
    | ResponseFunctionCallArgumentsDeltaEvent
    | ResponseFunctionCallArgumentsDoneEvent
    | ResponseIncompleteEvent
    | ResponseOutputItemAddedEvent
    | ResponseOutputItemDoneEvent
    | ResponseReasoningSummaryPartAddedEvent
    | ResponseReasoningSummaryPartDoneEvent
    | ResponseReasoningSummaryTextDeltaEvent
    | ResponseRefusalDeltaEvent
    | ResponseTextDeltaEvent
)

TEvent = TypeVar("TEvent", bound=StreamEvent)


@dataclass
class FakeResponsesEndpoint:
    create: AsyncMock


@dataclass
class FakeOpenAIClient:
    responses: FakeResponsesEndpoint


def _raw_stream(
    events: Sequence[RawResponseEvent],
) -> AsyncIterator[RawResponseEvent]:
    async def _iterate() -> AsyncIterator[RawResponseEvent]:
        for event in events:
            yield event

    return _iterate()


def _build_client(events: Sequence[RawResponseEvent]) -> FakeOpenAIClient:
    return FakeOpenAIClient(
        responses=FakeResponsesEndpoint(
            create=AsyncMock(return_value=_raw_stream(events))
        )
    )


def _response_payload(
    response_id: str,
    status: str,
    *,
    output: Sequence[JsonObject] | None = None,
    error: dict[str, str] | None = None,
    incomplete_reason: str | None = None,
) -> JsonObject:
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


def _created_event(sequence_number: int, response_id: str) -> ResponseCreatedEvent:
    return ResponseCreatedEvent.model_validate(
        {
            "type": "response.created",
            "sequence_number": sequence_number,
            "response": _response_payload(response_id, "in_progress"),
        }
    )


def _completed_event(
    sequence_number: int,
    response_id: str,
    *,
    output: Sequence[JsonObject] | None = None,
) -> ResponseCompletedEvent:
    return ResponseCompletedEvent.model_validate(
        {
            "type": "response.completed",
            "sequence_number": sequence_number,
            "response": _response_payload(response_id, "completed", output=output),
        }
    )


def _failed_event(
    sequence_number: int,
    response_id: str,
    message: str,
) -> ResponseFailedEvent:
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


def _error_event(
    sequence_number: int,
    message: str,
) -> ResponseErrorEvent:
    return ResponseErrorEvent.model_validate(
        {
            "type": "error",
            "sequence_number": sequence_number,
            "code": "server_error",
            "message": message,
            "param": None,
        }
    )


def _incomplete_event(
    sequence_number: int,
    response_id: str,
    reason: str,
    *,
    output: Sequence[JsonObject] | None = None,
) -> ResponseIncompleteEvent:
    return ResponseIncompleteEvent.model_validate(
        {
            "type": "response.incomplete",
            "sequence_number": sequence_number,
            "response": _response_payload(
                response_id,
                "incomplete",
                output=output,
                incomplete_reason=reason,
            ),
        }
    )


def _reasoning_added_event(
    sequence_number: int,
    item_id: str,
    *,
    output_index: int = 0,
) -> ResponseOutputItemAddedEvent:
    return ResponseOutputItemAddedEvent.model_validate(
        {
            "type": "response.output_item.added",
            "sequence_number": sequence_number,
            "output_index": output_index,
            "item": {
                "id": item_id,
                "type": "reasoning",
                "summary": [],
                "status": "in_progress",
            },
        }
    )


def _reasoning_summary_part_added_event(
    sequence_number: int,
    item_id: str,
    summary_index: int,
    *,
    output_index: int = 0,
) -> ResponseReasoningSummaryPartAddedEvent:
    return ResponseReasoningSummaryPartAddedEvent.model_validate(
        {
            "type": "response.reasoning_summary_part.added",
            "sequence_number": sequence_number,
            "item_id": item_id,
            "output_index": output_index,
            "part": {"type": "summary_text", "text": ""},
            "summary_index": summary_index,
        }
    )


def _reasoning_summary_delta_event(
    sequence_number: int,
    item_id: str,
    summary_index: int,
    delta: str,
    *,
    output_index: int = 0,
) -> ResponseReasoningSummaryTextDeltaEvent:
    return ResponseReasoningSummaryTextDeltaEvent.model_validate(
        {
            "type": "response.reasoning_summary_text.delta",
            "sequence_number": sequence_number,
            "item_id": item_id,
            "output_index": output_index,
            "summary_index": summary_index,
            "delta": delta,
        }
    )


def _reasoning_summary_part_done_event(
    sequence_number: int,
    item_id: str,
    summary_index: int,
    text: str,
    *,
    output_index: int = 0,
) -> ResponseReasoningSummaryPartDoneEvent:
    return ResponseReasoningSummaryPartDoneEvent.model_validate(
        {
            "type": "response.reasoning_summary_part.done",
            "sequence_number": sequence_number,
            "item_id": item_id,
            "output_index": output_index,
            "part": {"type": "summary_text", "text": text},
            "summary_index": summary_index,
        }
    )


def _reasoning_done_event(
    sequence_number: int,
    item_id: str,
    summary_texts: list[str],
    *,
    output_index: int = 0,
) -> ResponseOutputItemDoneEvent:
    return ResponseOutputItemDoneEvent.model_validate(
        {
            "type": "response.output_item.done",
            "sequence_number": sequence_number,
            "output_index": output_index,
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


def _message_added_event(
    sequence_number: int,
    item_id: str,
    *,
    output_index: int = 1,
) -> ResponseOutputItemAddedEvent:
    return ResponseOutputItemAddedEvent.model_validate(
        {
            "type": "response.output_item.added",
            "sequence_number": sequence_number,
            "output_index": output_index,
            "item": {
                "id": item_id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        }
    )


def _content_part_added_event(
    sequence_number: int,
    item_id: str,
    part_type: str,
    *,
    output_index: int = 1,
    content_index: int = 0,
) -> ResponseContentPartAddedEvent:
    if part_type == "output_text":
        part: JsonObject = {
            "type": "output_text",
            "text": "",
            "annotations": [],
        }
    else:
        part = {"type": "refusal", "refusal": ""}

    return ResponseContentPartAddedEvent.model_validate(
        {
            "type": "response.content_part.added",
            "sequence_number": sequence_number,
            "output_index": output_index,
            "item_id": item_id,
            "content_index": content_index,
            "part": part,
        }
    )


def _unsupported_content_part_added_event(
    sequence_number: int,
    item_id: str,
    *,
    output_index: int = 1,
    content_index: int = 0,
) -> ResponseContentPartAddedEvent:
    return ResponseContentPartAddedEvent.model_validate(
        {
            "type": "response.content_part.added",
            "sequence_number": sequence_number,
            "output_index": output_index,
            "item_id": item_id,
            "content_index": content_index,
            "part": {
                "type": "reasoning_text",
                "text": "internal",
            },
        }
    )


def _text_delta_event(
    sequence_number: int,
    item_id: str,
    delta: str,
    *,
    output_index: int = 1,
    content_index: int = 0,
) -> ResponseTextDeltaEvent:
    return ResponseTextDeltaEvent.model_validate(
        {
            "type": "response.output_text.delta",
            "sequence_number": sequence_number,
            "output_index": output_index,
            "item_id": item_id,
            "content_index": content_index,
            "delta": delta,
            "logprobs": [],
        }
    )


def _refusal_delta_event(
    sequence_number: int,
    item_id: str,
    delta: str,
    *,
    output_index: int = 1,
    content_index: int = 0,
) -> ResponseRefusalDeltaEvent:
    return ResponseRefusalDeltaEvent.model_validate(
        {
            "type": "response.refusal.delta",
            "sequence_number": sequence_number,
            "output_index": output_index,
            "item_id": item_id,
            "content_index": content_index,
            "delta": delta,
        }
    )


def _message_done_event(
    sequence_number: int,
    item_id: str,
    content: Sequence[JsonObject],
    *,
    output_index: int = 1,
) -> ResponseOutputItemDoneEvent:
    return ResponseOutputItemDoneEvent.model_validate(
        {
            "type": "response.output_item.done",
            "sequence_number": sequence_number,
            "output_index": output_index,
            "item": {
                "id": item_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": content,
            },
        }
    )


def _function_tool_call_added_event(
    sequence_number: int,
    item_id: str,
    call_id: str,
    name: str,
    *,
    arguments: str = "",
    output_index: int = 1,
) -> ResponseOutputItemAddedEvent:
    return ResponseOutputItemAddedEvent.model_validate(
        {
            "type": "response.output_item.added",
            "sequence_number": sequence_number,
            "output_index": output_index,
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


def _function_tool_call_arguments_delta_event(
    sequence_number: int,
    item_id: str,
    delta: str,
    *,
    output_index: int = 1,
) -> ResponseFunctionCallArgumentsDeltaEvent:
    return ResponseFunctionCallArgumentsDeltaEvent.model_validate(
        {
            "type": "response.function_call_arguments.delta",
            "sequence_number": sequence_number,
            "output_index": output_index,
            "item_id": item_id,
            "delta": delta,
        }
    )


def _function_tool_call_arguments_done_event(
    sequence_number: int,
    item_id: str,
    arguments: str,
    *,
    name: str = "get_weather",
    output_index: int = 1,
) -> ResponseFunctionCallArgumentsDoneEvent:
    return ResponseFunctionCallArgumentsDoneEvent.model_validate(
        {
            "type": "response.function_call_arguments.done",
            "sequence_number": sequence_number,
            "output_index": output_index,
            "item_id": item_id,
            "name": name,
            "arguments": arguments,
        }
    )


def _function_tool_call_done_event(
    sequence_number: int,
    item_id: str,
    call_id: str,
    name: str,
    arguments: str,
    *,
    output_index: int = 1,
) -> ResponseOutputItemDoneEvent:
    return ResponseOutputItemDoneEvent.model_validate(
        {
            "type": "response.output_item.done",
            "sequence_number": sequence_number,
            "output_index": output_index,
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


def _expect_event_type(event: StreamEvent, event_type: type[TEvent]) -> TEvent:
    assert isinstance(event, event_type)
    return cast(TEvent, event)


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


def _collect_events(
    client: FakeOpenAIClient,
    tools: Sequence[ToolDefinition] | None = None,
) -> list[StreamEvent]:
    async def _collect() -> list[StreamEvent]:
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


def _subscription_raw_stream(
    events: Sequence[SubscriptionEventPayload],
) -> AsyncIterator[SubscriptionEventPayload]:
    async def _iterate() -> AsyncIterator[SubscriptionEventPayload]:
        for event in events:
            yield event

    return _iterate()


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
        )
    ]


def test_stream_maps_raw_events_with_shared_partial_state() -> None:
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
    shared_partial = reasoning_start.partial
    final_reasoning_block = _expect_reasoning_block(shared_partial.content[0])
    final_text_block = _expect_text_block(shared_partial.content[1])
    done_reasoning_block = _expect_reasoning_block(done.message.content[0])
    done_text_block = _expect_text_block(done.message.content[1])

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
    assert start.partial.response_id == "resp_success"
    assert reasoning_start.partial.response_id == "resp_success"

    assert start.partial is shared_partial
    assert reasoning_delta_one.partial is shared_partial
    assert reasoning_delta_two.partial is shared_partial
    assert reasoning_delta_separator.partial is shared_partial
    assert reasoning_delta_three.partial is shared_partial
    assert reasoning_delta_four.partial is shared_partial
    assert reasoning_end.partial is shared_partial
    assert text_start.partial is shared_partial
    assert text_delta_one.partial is shared_partial
    assert text_delta_two.partial is shared_partial
    assert text_end.partial is shared_partial
    assert done.message is shared_partial

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
    assert done.message.response_id == "resp_success"

    assert (
        done_reasoning_block.summary_text
        == "Exploring reasoning traces\n\nFormulating reasoning traces"
    )
    assert done_reasoning_block.reasoning_signature is not None
    assert (
        done_reasoning_block.reasoning_signature
        == final_reasoning_block.reasoning_signature
    )
    assert json.loads(done_reasoning_block.reasoning_signature) == {
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
    reasoning_block = _expect_reasoning_block(reasoning_end.partial.content[0])
    done_reasoning_block = _expect_reasoning_block(done.message.content[0])

    assert [event.type for event in events] == [
        "start",
        "reasoning_start",
        "reasoning_delta",
        "reasoning_end",
        "done",
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


def test_stream_maps_refusal_deltas_with_shared_partial_state() -> None:
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
    shared_partial = text_start.partial
    text_block = _expect_text_block(shared_partial.content[0])
    done_text_block = _expect_text_block(done.message.content[0])

    assert [event.type for event in events] == [
        "start",
        "text_start",
        "text_delta",
        "text_end",
        "done",
    ]
    assert text_start.partial is shared_partial
    assert text_delta.partial is shared_partial
    assert text_end.partial is shared_partial
    assert text_delta.delta == "No"
    assert text_block.text == "No thanks"
    assert done.message is shared_partial
    assert done_text_block.text == "No thanks"


def test_stream_maps_function_tool_call_events_with_shared_partial_state() -> None:
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
    shared_partial = tool_call_start.partial
    tool_call_block = _expect_tool_call_block(shared_partial.content[0])
    done_tool_call_block = _expect_tool_call_block(done.message.content[0])

    assert [event.type for event in events] == [
        "start",
        "tool_call_start",
        "tool_call_delta",
        "tool_call_delta",
        "tool_call_end",
        "done",
    ]
    assert tool_call_start.partial is shared_partial
    assert tool_call_delta_one.partial is shared_partial
    assert tool_call_delta_two.partial is shared_partial
    assert tool_call_end.partial is shared_partial
    assert tool_call_delta_one.delta == '{"'
    assert tool_call_delta_two.delta == 'city":"Munich"}'
    assert tool_call_block.call_id == "call_123"
    assert tool_call_block.name == "get_weather"
    assert tool_call_block.provider_item_id == "fc_123"
    assert tool_call_block.arguments == {"city": "Munich"}
    assert done.message.stop_reason == "tool_use"
    assert done.message is shared_partial
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
        "start",
        "text_start",
        "text_delta",
        "text_delta",
        "text_end",
        "done",
    ]
    assert text_delta_one.delta == "No"
    assert text_delta_two.delta == " thanks"
    assert _expect_text_block(done.message.content[0]).text == "No thanks"


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
        "start",
        "text_start",
        "text_end",
        "done",
    ]
    assert _expect_text_block(text_end.partial.content[0]).text == ""
    assert _expect_text_block(done.message.content[0]).text == ""


def test_stream_maps_failed_response_into_error_event() -> None:
    raw_events = [
        _created_event(1, "resp_failed"),
        _failed_event(2, "resp_failed", "Model overloaded"),
    ]

    client = _build_client(raw_events)
    events = _collect_events(client)
    error = _expect_event_type(events[1], StreamErrorEvent)

    assert [event.type for event in events] == ["start", "error"]
    assert error.error.error_message == "Model overloaded"
    assert error.error.stop_reason == "error"
    assert error.error.response_id == "resp_failed"


def test_stream_maps_error_event_into_error_event() -> None:
    raw_events = [
        _created_event(1, "resp_error"),
        _error_event(2, "Socket closed"),
    ]

    client = _build_client(raw_events)
    events = _collect_events(client)
    error = _expect_event_type(events[1], StreamErrorEvent)

    assert [event.type for event in events] == ["start", "error"]
    assert error.error.error_message == "Socket closed"
    assert error.error.stop_reason == "error"
    assert error.error.response_id == "resp_error"


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
        "start",
        "text_start",
        "text_delta",
        "text_end",
        "done",
    ]
    assert done.message.stop_reason == "length"
    assert _expect_text_block(done.message.content[0]).text == "Partial answer"


def test_stream_maps_incomplete_content_filter_into_error_event() -> None:
    raw_events = [
        _created_event(1, "resp_filtered"),
        _incomplete_event(2, "resp_filtered", "content_filter"),
    ]

    client = _build_client(raw_events)
    events = _collect_events(client)
    error = _expect_event_type(events[1], StreamErrorEvent)

    assert [event.type for event in events] == ["start", "error"]
    assert (
        error.error.error_message
        == "OpenAI response was truncated by the content filter."
    )
    assert error.error.stop_reason == "error"


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

    async def _collect() -> list[StreamEvent]:
        event_stream = await stream_subscription(
            history=[UserMessage(content="hello")],
            model="gpt-5.4",
            reasoning={"effort": "medium"},
            instructions="Follow the repo conventions.",
            raw_stream=_subscription_raw_stream(raw_events),
        )
        return [event async for event in event_stream]

    events = asyncio.run(_collect())
    start = _expect_event_type(events[0], StreamStartEvent)
    text_start = _expect_event_type(events[1], TextStartEvent)
    text_delta = _expect_event_type(events[2], TextDeltaEvent)
    text_end = _expect_event_type(events[3], TextEndEvent)
    done = _expect_event_type(events[4], StreamDoneEvent)

    assert [event.type for event in events] == [
        "start",
        "text_start",
        "text_delta",
        "text_end",
        "done",
    ]
    assert start.partial.response_id == "resp_subscription"
    assert text_start.partial is start.partial
    assert text_delta.delta == "Hello from subscription"
    assert (
        _expect_text_block(text_end.partial.content[0]).text
        == "Hello from subscription"
    )
    assert done.message.response_id == "resp_subscription"


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
