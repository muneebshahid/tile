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
    MessageStartEvent,
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
    StreamDoneEvent,
    StreamErrorEvent,
    StreamEvent,
    StreamStartEvent,
    TextBlock,
    ToolCallBlock,
)
from ai.types.tools import ToolDefinition

TEvent = TypeVar("TEvent", bound=AgentEvent)


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


def _collect_run_events(agent: Agent, prompt: str) -> list[AgentEvent]:
    async def _collect() -> list[AgentEvent]:
        return [event async for event in agent.run(prompt)]

    return asyncio.run(_collect())


def _expect_event_type(event: AgentEvent, event_type: type[TEvent]) -> TEvent:
    assert isinstance(event, event_type)
    return cast(TEvent, event)


def _expect_user_message(item: ConversationItem) -> UserMessage:
    assert isinstance(item, UserMessage)
    return item


def _expect_assistant_turn(item: ConversationItem) -> AssistantTurn:
    assert isinstance(item, AssistantTurn)
    return item


def _expect_assistant_message(item: object) -> AssistantMessage:
    assert isinstance(item, AssistantMessage)
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


def test_agent_run_yields_current_events_for_tool_use_loop() -> None:
    tools = _sample_tools()
    invocations: list[StreamInvocation] = []
    stream_fn = _build_stream_fn(
        streams=[
            [
                StreamStartEvent(
                    type="start",
                    partial=AssistantMessage(response_id="resp_tool_call"),
                ),
                StreamDoneEvent(
                    type="done",
                    message=AssistantMessage(
                        response_id="resp_tool_call",
                        stop_reason="tool_use",
                        content=[
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
                    partial=AssistantMessage(response_id="resp_follow_up"),
                ),
                StreamDoneEvent(
                    type="done",
                    message=AssistantMessage(
                        response_id="resp_follow_up",
                        stop_reason="stop",
                        content=[TextBlock(text="It is sunny in Munich.")],
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

    events = _collect_run_events(agent, prompt="What is the weather in Munich?")

    assert [event.type for event in events] == [
        "agent_start",
        "turn_start",
        "message_start",
        "tool_execution_start",
        "tool_execution_end",
        "turn_end",
        "turn_start",
        "message_start",
        "turn_end",
        "agent_end",
    ]

    first_turn_start = _expect_event_type(events[1], TurnStartEvent)
    first_message_start = _expect_event_type(events[2], MessageStartEvent)
    tool_execution_start = _expect_event_type(events[3], ToolExecutionStartEvent)
    tool_execution_end = _expect_event_type(events[4], ToolExecutionEndEvent)
    first_turn_end = _expect_event_type(events[5], TurnEndEvent)
    second_turn_start = _expect_event_type(events[6], TurnStartEvent)
    second_message_start = _expect_event_type(events[7], MessageStartEvent)
    second_turn_end = _expect_event_type(events[8], TurnEndEvent)
    agent_end = _expect_event_type(events[9], AgentEndEvent)
    first_partial_message = _expect_assistant_message(first_message_start.message)
    second_partial_message = _expect_assistant_message(second_message_start.message)

    assert isinstance(events[0], AgentStartEvent)
    assert first_turn_start.type == "turn_start"
    assert first_partial_message.response_id == "resp_tool_call"
    assert first_partial_message.content == []
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
    assert first_turn_end.message.content == [
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
    assert second_partial_message.response_id == "resp_follow_up"
    assert second_partial_message.content == []
    assert second_turn_end.message.response_id == "resp_follow_up"
    assert second_turn_end.message.stop_reason == "stop"
    assert second_turn_end.message.content == [TextBlock(text="It is sunny in Munich.")]
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
                    partial=AssistantMessage(response_id="resp_error"),
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

    events = _collect_run_events(agent, prompt="Say hello")

    assert [event.type for event in events] == [
        "agent_start",
        "turn_start",
        "message_start",
        "turn_end",
        "agent_end",
    ]

    message_start = _expect_event_type(events[2], MessageStartEvent)
    turn_end = _expect_event_type(events[3], TurnEndEvent)
    agent_end = _expect_event_type(events[4], AgentEndEvent)
    partial_message = _expect_assistant_message(message_start.message)

    assert isinstance(events[0], AgentStartEvent)
    assert isinstance(events[1], TurnStartEvent)
    assert partial_message.response_id == "resp_error"
    assert partial_message.content == []
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
