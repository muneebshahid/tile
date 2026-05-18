"""Tests for translating stream events into agent events.

These tests document the final streaming lifecycle layer. The agent receives
app-level ``StreamEvent`` models, emits ``message_start`` and ``message_update``
events while the assistant message is streaming, finalizes history on ``done``
or ``error``, and executes tools before starting a follow-up assistant turn.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import TypeVar, cast
from unittest.mock import AsyncMock

from agent.agent import Agent
from agent.types import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from ai.types.contracts import Reasoning
from ai.types.conversation import (
    AssistantTurn,
    ConversationItem,
    ToolResultTurn,
    UserMessage,
)
from ai.types.stream import (
    AssistantMessage,
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
from ai.types.tools import ToolDefinition

TEvent = TypeVar("TEvent", bound=AgentEvent)
TStreamEvent = TypeVar("TStreamEvent", bound=StreamEvent)


@dataclass
class StreamInvocation:
    history: tuple[ConversationItem, ...]
    model: str
    instructions: str
    reasoning: Reasoning | None
    tools: tuple[ToolDefinition, ...] | None


def _iter_events(events: Sequence[StreamEvent]) -> AsyncIterator[StreamEvent]:
    async def _iterate() -> AsyncIterator[StreamEvent]:
        for event in events:
            yield event

    return _iterate()


def _build_stream_fn(
    streams: Sequence[Sequence[StreamEvent]],
    invocations: list[StreamInvocation],
):
    pending_streams = list(streams)

    async def _stream_fn(
        history: Sequence[ConversationItem],
        model: str,
        *,
        instructions: str,
        reasoning: Reasoning | None,
        tools: Sequence[ToolDefinition] | None,
    ) -> AsyncIterator[StreamEvent]:
        invocations.append(
            StreamInvocation(
                history=tuple(history),
                model=model,
                instructions=instructions,
                reasoning=reasoning,
                tools=tuple(tools) if tools is not None else None,
            )
        )
        return _iter_events(pending_streams.pop(0))

    return _stream_fn


def _collect_run_events(agent: Agent) -> list[AgentEvent]:
    async def _collect() -> list[AgentEvent]:
        return [event async for event in agent.run()]

    return asyncio.run(_collect())


def _expect_event_type(event: AgentEvent, event_type: type[TEvent]) -> TEvent:
    assert isinstance(event, event_type)
    return cast(TEvent, event)


def _expect_stream_event_type(
    event: StreamEvent, event_type: type[TStreamEvent]
) -> TStreamEvent:
    assert isinstance(event, event_type)
    return cast(TStreamEvent, event)


def _expect_user_message(item: ConversationItem) -> UserMessage:
    assert isinstance(item, UserMessage)
    return item


def _expect_assistant_turn(item: ConversationItem) -> AssistantTurn:
    assert isinstance(item, AssistantTurn)
    return item


def _expect_assistant_message(item: object) -> AssistantMessage:
    assert isinstance(item, AssistantMessage)
    return item


def _expect_agent_assistant_turn(item: object) -> AssistantTurn:
    assert isinstance(item, AssistantTurn)
    return item


def _expect_tool_result_turn(item: ConversationItem) -> ToolResultTurn:
    assert isinstance(item, ToolResultTurn)
    return item


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
                    }
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        )
    ]


def test_add_user_message_appends_user_turn_to_history() -> None:
    agent = Agent(stream_fn=AsyncMock(), model="gpt-5.4")

    agent.add_user_message("Hello, piy")

    assert len(agent.history) == 1
    user_message = _expect_user_message(agent.history[0])
    assert user_message.content == "Hello, piy"


def test_agent_run_yields_current_events_for_tool_use_loop() -> None:
    tools = _sample_tools()
    invocations: list[StreamInvocation] = []
    tool_call_start_message = AssistantMessage(
        response_id="resp_tool_call",
        blocks=[
            ReasoningBlock(summary_text="Thinking about weather"),
            ToolCallBlock(
                call_id="call_123",
                name="get_weather",
                arguments={"city": "Munich"},
                provider_item_id="fc_123",
            ),
        ],
    )
    reasoning_message = AssistantMessage(
        response_id="resp_tool_call",
        blocks=[ReasoningBlock(summary_text="Thinking about weather")],
    )
    text_message = AssistantMessage(
        response_id="resp_follow_up",
        blocks=[TextBlock(text="It is sunny in Munich.")],
    )
    stream_fn = _build_stream_fn(
        streams=[
            [
                StreamStartEvent(
                    type="start",
                    message=AssistantMessage(response_id="resp_tool_call"),
                ),
                ReasoningStartEvent(
                    type="reasoning_start",
                    message=reasoning_message,
                ),
                ReasoningDeltaEvent(
                    type="reasoning_delta",
                    delta="Thinking about weather",
                    message=reasoning_message,
                ),
                ReasoningEndEvent(
                    type="reasoning_end",
                    message=reasoning_message,
                ),
                ToolCallStartEvent(
                    type="tool_call_start",
                    message=tool_call_start_message,
                ),
                ToolCallDeltaEvent(
                    type="tool_call_delta",
                    delta='{"city":"Munich"}',
                    message=tool_call_start_message,
                ),
                ToolCallEndEvent(
                    type="tool_call_end",
                    message=tool_call_start_message,
                ),
                StreamDoneEvent(
                    type="done",
                    message=AssistantMessage(
                        response_id="resp_tool_call",
                        stop_reason="tool_use",
                        blocks=[
                            ToolCallBlock(
                                call_id="call_123",
                                name="get_weather",
                                arguments={"city": "Munich"},
                                provider_item_id="fc_123",
                            )
                        ],
                    ),
                ),
            ],
            [
                StreamStartEvent(
                    type="start",
                    message=AssistantMessage(response_id="resp_follow_up"),
                ),
                TextStartEvent(
                    type="text_start",
                    message=text_message,
                ),
                TextDeltaEvent(
                    type="text_delta",
                    delta="It is sunny in Munich.",
                    message=text_message,
                ),
                TextEndEvent(
                    type="text_end",
                    message=text_message,
                ),
                StreamDoneEvent(
                    type="done",
                    message=AssistantMessage(
                        response_id="resp_follow_up",
                        stop_reason="stop",
                        blocks=[TextBlock(text="It is sunny in Munich.")],
                    ),
                ),
            ],
        ],
        invocations=invocations,
    )
    tool = AsyncMock(return_value={"temperature_c": 18, "condition": "sunny"})
    get_tool_mock = AsyncMock(return_value=tool)
    agent = Agent(stream_fn=stream_fn, model="gpt-5.4", tools=tools)
    agent._get_tool = get_tool_mock  # type: ignore[invalid-assignment]
    agent.add_user_message("What is the weather in Munich?")

    events = _collect_run_events(agent)

    assert [event.type for event in events] == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_update",
        "message_update",
        "message_update",
        "message_update",
        "message_update",
        "message_update",
        "message_end",
        "tool_execution_start",
        "tool_execution_end",
        "turn_end",
        "turn_start",
        "message_start",
        "message_update",
        "message_update",
        "message_update",
        "message_end",
        "turn_end",
        "agent_end",
    ]

    first_turn_start = _expect_event_type(events[1], TurnStartEvent)
    first_message_start = _expect_event_type(events[2], MessageStartEvent)
    first_reasoning_start = _expect_event_type(events[3], MessageUpdateEvent)
    first_reasoning_delta = _expect_event_type(events[4], MessageUpdateEvent)
    first_reasoning_end = _expect_event_type(events[5], MessageUpdateEvent)
    first_tool_call_start = _expect_event_type(events[6], MessageUpdateEvent)
    first_tool_call_delta = _expect_event_type(events[7], MessageUpdateEvent)
    first_tool_call_end = _expect_event_type(events[8], MessageUpdateEvent)
    first_message_end = _expect_event_type(events[9], MessageEndEvent)
    tool_execution_start = _expect_event_type(events[10], ToolExecutionStartEvent)
    tool_execution_end = _expect_event_type(events[11], ToolExecutionEndEvent)
    first_turn_end = _expect_event_type(events[12], TurnEndEvent)
    second_turn_start = _expect_event_type(events[13], TurnStartEvent)
    second_message_start = _expect_event_type(events[14], MessageStartEvent)
    second_text_start = _expect_event_type(events[15], MessageUpdateEvent)
    second_text_delta = _expect_event_type(events[16], MessageUpdateEvent)
    second_text_end = _expect_event_type(events[17], MessageUpdateEvent)
    second_message_end = _expect_event_type(events[18], MessageEndEvent)
    second_turn_end = _expect_event_type(events[19], TurnEndEvent)
    agent_end = _expect_event_type(events[20], AgentEndEvent)
    first_stream_message = _expect_assistant_message(first_message_start.message)
    second_stream_message = _expect_assistant_message(second_message_start.message)
    first_final_message = _expect_agent_assistant_turn(first_message_end.message)
    second_final_message = _expect_agent_assistant_turn(second_message_end.message)

    assert isinstance(events[0], AgentStartEvent)
    assert first_turn_start.type == "turn_start"
    assert first_stream_message.response_id == "resp_tool_call"
    assert first_stream_message.blocks == []
    assert first_reasoning_start.stream_event.type == "reasoning_start"
    assert first_reasoning_start.message is reasoning_message
    assert first_reasoning_delta.stream_event.type == "reasoning_delta"
    assert (
        _expect_stream_event_type(
            first_reasoning_delta.stream_event, ReasoningDeltaEvent
        ).delta
        == "Thinking about weather"
    )
    assert first_reasoning_delta.message is reasoning_message
    assert first_reasoning_end.stream_event.type == "reasoning_end"
    assert first_reasoning_end.message is reasoning_message
    assert first_tool_call_start.stream_event.type == "tool_call_start"
    assert first_tool_call_start.message is tool_call_start_message
    assert first_tool_call_delta.stream_event.type == "tool_call_delta"
    assert (
        _expect_stream_event_type(
            first_tool_call_delta.stream_event, ToolCallDeltaEvent
        ).delta
        == '{"city":"Munich"}'
    )
    assert first_tool_call_delta.message is tool_call_start_message
    assert first_tool_call_end.stream_event.type == "tool_call_end"
    assert first_tool_call_end.message is tool_call_start_message
    assert first_final_message.response_id == "resp_tool_call"
    assert first_final_message.stop_reason == "tool_use"
    assert tool_execution_start.call_id == "call_123"
    assert tool_execution_start.tool_name == "get_weather"
    assert tool_execution_start.arguments == {"city": "Munich"}
    assert tool_execution_end.call_id == "call_123"
    assert tool_execution_end.tool_name == "get_weather"
    assert tool_execution_end.result == {
        "temperature_c": 18,
        "condition": "sunny",
    }
    assert tool_execution_end.is_error is False
    assert first_turn_end.message.response_id == "resp_tool_call"
    assert first_turn_end.message.stop_reason == "tool_use"
    assert first_turn_end.message.status == "completed"
    assert first_turn_end.message.blocks == [
        ToolCallBlock(
            call_id="call_123",
            name="get_weather",
            arguments={"city": "Munich"},
            provider_item_id="fc_123",
        )
    ]
    assert first_turn_end.tool_results[0].call_id == "call_123"
    assert first_turn_end.tool_results[0].tool_name == "get_weather"
    assert (
        json.loads(first_turn_end.tool_results[0].content) == tool_execution_end.result
    )
    assert first_turn_end.tool_results[0].is_error is False
    assert second_turn_start.type == "turn_start"
    assert second_stream_message.response_id == "resp_follow_up"
    assert second_stream_message.blocks == []
    assert second_text_start.stream_event.type == "text_start"
    assert second_text_start.message is text_message
    assert second_text_delta.stream_event.type == "text_delta"
    assert _expect_stream_event_type(
        second_text_delta.stream_event, TextDeltaEvent
    ).delta == ("It is sunny in Munich.")
    assert second_text_delta.message is text_message
    assert second_text_end.stream_event.type == "text_end"
    assert second_text_end.message is text_message
    assert second_final_message.response_id == "resp_follow_up"
    assert second_final_message.stop_reason == "stop"
    assert second_turn_end.message.response_id == "resp_follow_up"
    assert second_turn_end.message.stop_reason == "stop"
    assert second_turn_end.message.blocks == [TextBlock(text="It is sunny in Munich.")]
    assert second_turn_end.tool_results == []
    assert len(invocations) == 2
    assert invocations[0].model == "gpt-5.4"
    assert invocations[0].tools == tuple(tools)
    assert invocations[1].tools == tuple(tools)
    first_request_user = _expect_user_message(invocations[0].history[0])
    second_request_assistant = _expect_assistant_turn(invocations[1].history[1])
    second_request_tool_result = _expect_tool_result_turn(invocations[1].history[2])
    assert first_request_user.content == "What is the weather in Munich?"
    assert second_request_assistant.response_id == "resp_tool_call"
    assert second_request_tool_result.tool_name == "get_weather"
    assert len(agent_end.items) == 4
    first_item = _expect_user_message(agent_end.items[0])
    second_item = _expect_assistant_turn(agent_end.items[1])
    third_item = _expect_tool_result_turn(agent_end.items[2])
    fourth_item = _expect_assistant_turn(agent_end.items[3])
    assert first_item.content == "What is the weather in Munich?"
    assert second_item.response_id == "resp_tool_call"
    assert third_item.tool_name == "get_weather"
    assert fourth_item.response_id == "resp_follow_up"

    get_tool_mock.assert_awaited_once_with("get_weather")
    tool.assert_awaited_once_with(city="Munich")


def test_agent_run_yields_error_turn_end_for_stream_error() -> None:
    invocations: list[StreamInvocation] = []
    stream_fn = _build_stream_fn(
        streams=[
            [
                StreamStartEvent(
                    type="start",
                    message=AssistantMessage(response_id="resp_error"),
                ),
                StreamErrorEvent(
                    type="error",
                    error=AssistantMessage(
                        response_id="resp_error",
                        stop_reason="error",
                        error_message="Socket closed",
                    ),
                ),
            ]
        ],
        invocations=invocations,
    )
    agent = Agent(stream_fn=stream_fn, model="gpt-5.4")
    agent.add_user_message("Say hello")

    events = _collect_run_events(agent)

    assert [event.type for event in events] == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "turn_end",
        "agent_end",
    ]

    message_start = _expect_event_type(events[2], MessageStartEvent)
    message_end = _expect_event_type(events[3], MessageEndEvent)
    turn_end = _expect_event_type(events[4], TurnEndEvent)
    agent_end = _expect_event_type(events[5], AgentEndEvent)
    stream_message = _expect_assistant_message(message_start.message)
    final_message = _expect_agent_assistant_turn(message_end.message)

    assert isinstance(events[0], AgentStartEvent)
    assert isinstance(events[1], TurnStartEvent)
    assert stream_message.response_id == "resp_error"
    assert stream_message.blocks == []
    assert final_message.response_id == "resp_error"
    assert final_message.status == "error"
    assert turn_end.message.response_id == "resp_error"
    assert turn_end.message.stop_reason == "error"
    assert turn_end.message.status == "error"
    assert turn_end.message.error_message == "Socket closed"
    assert turn_end.tool_results == []
    assert len(invocations) == 1
    first_request_user = _expect_user_message(invocations[0].history[0])
    assert first_request_user.content == "Say hello"
    assert len(agent_end.items) == 2
    first_item = _expect_user_message(agent_end.items[0])
    second_item = _expect_assistant_turn(agent_end.items[1])
    assert first_item.content == "Say hello"
    assert second_item.response_id == "resp_error"
