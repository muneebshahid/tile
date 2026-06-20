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
from pathlib import Path
from typing import TypeVar
from agent.agent import run_agent
from agent.types import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    StreamFn,
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
from ai.types.stream_events import (
    AssistantBlock,
    ProviderMetadata,
    ProviderSource,
    ProviderStreamEvent,
    ReasoningBlock,
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    ReasoningStartEvent,
    StreamDoneEvent,
    StreamErrorEvent,
    StreamEvent,
    StreamStartedEvent,
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
from ai.types.tools import JsonObject, ToolDefinition, ToolResult, ToolTextContent

TEvent = TypeVar("TEvent", bound=AgentEvent)
TStreamEvent = TypeVar("TStreamEvent", bound=StreamEvent)


@dataclass
class StreamInvocation:
    """Captured arguments from one provider stream invocation."""

    history: tuple[ConversationItem, ...]
    model: str
    instructions: str
    reasoning: Reasoning | None
    tools: tuple[ToolDefinition, ...] | None


def _iter_events(
    events: Sequence[ProviderStreamEvent],
) -> AsyncIterator[ProviderStreamEvent]:
    """Yield static stream events asynchronously."""

    async def _iterate() -> AsyncIterator[ProviderStreamEvent]:
        """Yield each provided stream event."""

        for event in events:
            yield event

    return _iterate()


def _build_stream_fn(
    streams: Sequence[Sequence[ProviderStreamEvent]],
    invocations: list[StreamInvocation],
) -> StreamFn:
    """Build a provider stream function that records each invocation."""

    pending_streams = list(streams)

    async def _stream_fn(
        history: Sequence[ConversationItem],
        model: str,
        *,
        instructions: str,
        reasoning: Reasoning | None,
        tools: Sequence[ToolDefinition] | None,
    ) -> AsyncIterator[ProviderStreamEvent]:
        """Return the next queued provider event stream."""

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


def _collect_run_events(
    history: Sequence[ConversationItem],
    *,
    stream_fn: StreamFn,
    model: str = "gpt-5.4",
    reasoning: Reasoning | None = None,
    tools: Sequence[ToolDefinition] = (),
    system_prompt: str = "Base prompt.",
    cwd: Path | str | None = None,
) -> list[AgentEvent]:
    """Collect all events emitted by one stateless agent run."""

    async def _collect() -> list[AgentEvent]:
        """Collect run events from the async generator."""

        return [
            event
            async for event in run_agent(
                history,
                stream_fn=stream_fn,
                model=model,
                reasoning=reasoning,
                tools=tools,
                system_prompt=system_prompt,
                cwd=cwd,
            )
        ]

    return asyncio.run(_collect())


def _expect_event_type(event: AgentEvent, event_type: type[TEvent]) -> TEvent:
    """Assert and return an agent event with a precise type."""

    assert isinstance(event, event_type)
    return event


def _expect_stream_event_type(
    event: StreamEvent, event_type: type[TStreamEvent]
) -> TStreamEvent:
    """Assert and return a stream event with a precise type."""

    assert isinstance(event, event_type)
    return event


def _expect_user_message(item: ConversationItem) -> UserMessage:
    """Assert and return a user conversation item."""

    assert isinstance(item, UserMessage)
    return item


def _expect_assistant_turn(item: ConversationItem) -> AssistantTurn:
    """Assert and return an assistant conversation item."""

    assert isinstance(item, AssistantTurn)
    return item


def _expect_agent_assistant_turn(item: AssistantTurn) -> AssistantTurn:
    """Assert and return an agent assistant turn."""

    assert isinstance(item, AssistantTurn)
    return item


def _expect_tool_result_turn(item: ConversationItem) -> ToolResultTurn:
    """Assert and return a tool result conversation item."""

    assert isinstance(item, ToolResultTurn)
    return item


def _tool_text(result: ToolResult) -> str:
    """Return the single text block from a tool result."""

    assert len(result.content) == 1
    content = result.content[0]
    assert isinstance(content, ToolTextContent)
    return content.text


def _sample_tools() -> list[ToolDefinition]:
    """Build the deterministic tool registry used by agent tests."""

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
            fn=_get_weather,
        )
    ]


async def _get_weather(city: str) -> ToolResult:
    """Return a deterministic weather payload for tests."""

    return ToolResult.text(
        json.dumps({"temperature_c": 18, "condition": "sunny", "city": city})
    )


async def _raise_tool_error(city: str) -> ToolResult:
    """Raise a deterministic tool error for tests."""

    _ = city
    raise RuntimeError("boom")


def _source() -> ProviderSource:
    """Build a deterministic provider source for agent tests."""

    return ProviderSource(provider="test", model="gpt-5.4")


def _metadata(**values: str | None) -> ProviderMetadata | None:
    """Build provider metadata from non-empty string values."""

    data: JsonObject = {
        key: value for key, value in values.items() if value is not None
    }
    if not data:
        return None
    return ProviderMetadata(data=data)


def _stream_started(response_id: str) -> StreamStartedEvent:
    """Build a deterministic stream started event."""

    return StreamStartedEvent(source=_source(), response_id=response_id)


def _stream_done(
    response_id: str,
    *,
    stop_reason: StopReason = "stop",
    blocks: Sequence[AssistantBlock] = (),
) -> StreamDoneEvent:
    """Build a deterministic stream done event."""

    return StreamDoneEvent(
        source=_source(),
        response_id=response_id,
        stop_reason=stop_reason,
        blocks=list(blocks),
    )


def _stream_error(response_id: str, error_message: str) -> StreamErrorEvent:
    """Build a deterministic stream error event."""

    return StreamErrorEvent(
        source=_source(),
        response_id=response_id,
        error_message=error_message,
    )


def _tool_call_block(
    *,
    call_id: str,
    name: str,
    arguments: JsonObject,
    provider_item_id: str | None = None,
) -> ToolCallBlock:
    """Build a tool call block with provider replay metadata."""

    return ToolCallBlock(
        call_id=call_id,
        name=name,
        arguments=arguments,
        provider_metadata=_metadata(provider_item_id=provider_item_id),
    )


def _tool_call_stream(
    *,
    response_id: str,
    call_id: str,
    tool_name: str,
    arguments: JsonObject,
) -> list[ProviderStreamEvent]:
    """Build a minimal assistant stream that requests one tool call."""

    return [
        _stream_started(response_id),
        _stream_done(
            response_id,
            stop_reason="tool_use",
            blocks=[
                _tool_call_block(
                    call_id=call_id,
                    name=tool_name,
                    arguments=arguments,
                    provider_item_id=f"fc_{call_id}",
                )
            ],
        ),
    ]


def _final_text_stream(*, response_id: str, text: str) -> list[ProviderStreamEvent]:
    """Build a minimal assistant stream that returns final text."""

    return [
        _stream_started(response_id),
        _stream_done(response_id, blocks=[TextBlock(text=text)]),
    ]


def test_run_agent_does_not_mutate_supplied_history() -> None:
    """Keep caller-owned history unchanged and return only run-local items."""

    invocations: list[StreamInvocation] = []
    stream_fn = _build_stream_fn(
        streams=[
            [
                _stream_started("resp_done"),
                _stream_done("resp_done"),
            ]
        ],
        invocations=invocations,
    )
    history: list[ConversationItem] = [UserMessage(content="Hello, piy")]

    events = _collect_run_events(history, stream_fn=stream_fn)

    agent_end = _expect_event_type(events[-1], AgentEndEvent)
    assert history == [UserMessage(content="Hello, piy")]
    assert len(agent_end.new_items) == 1
    assert _expect_assistant_turn(agent_end.new_items[0]).response_id == "resp_done"


def test_agent_run_yields_current_events_for_tool_use_loop() -> None:
    """Emit stream and tool events while carrying history into follow-up turns."""

    tools = _sample_tools()
    invocations: list[StreamInvocation] = []
    reasoning_block = ReasoningBlock(summary_text="Thinking about weather")
    tool_call_block = _tool_call_block(
        call_id="call_123",
        name="get_weather",
        arguments={"city": "Munich"},
        provider_item_id="fc_123",
    )
    text_block = TextBlock(text="It is sunny in Munich.")
    stream_fn = _build_stream_fn(
        streams=[
            [
                _stream_started("resp_tool_call"),
                ReasoningStartEvent(content_index=0),
                ReasoningDeltaEvent(
                    content_index=0,
                    delta="Thinking about weather",
                ),
                ReasoningEndEvent(
                    content_index=0,
                    block=reasoning_block,
                ),
                ToolCallStartEvent(
                    content_index=1,
                    call_id="call_123",
                    name="get_weather",
                ),
                ToolCallDeltaEvent(
                    content_index=1,
                    delta='{"city":"Munich"}',
                ),
                ToolCallEndEvent(
                    content_index=1,
                    block=tool_call_block,
                ),
                _stream_done(
                    "resp_tool_call",
                    stop_reason="tool_use",
                    blocks=[tool_call_block],
                ),
            ],
            [
                _stream_started("resp_follow_up"),
                TextStartEvent(
                    content_index=0,
                ),
                TextDeltaEvent(
                    content_index=0,
                    delta="It is sunny in Munich.",
                ),
                TextEndEvent(
                    content_index=0,
                    block=text_block,
                ),
                _stream_done("resp_follow_up", blocks=[text_block]),
            ],
        ],
        invocations=invocations,
    )
    history: list[ConversationItem] = [
        UserMessage(content="What is the weather in Munich?")
    ]

    events = _collect_run_events(history, stream_fn=stream_fn, tools=tools)

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
    first_final_message = _expect_agent_assistant_turn(first_message_end.message)
    second_final_message = _expect_agent_assistant_turn(second_message_end.message)

    assert isinstance(events[0], AgentStartEvent)
    assert first_turn_start.type == "turn_start"
    assert first_message_start.response_id == "resp_tool_call"
    assert first_reasoning_start.stream_event.type == "reasoning_start"
    assert first_reasoning_start.stream_event.content_index == 0
    assert first_reasoning_delta.stream_event.type == "reasoning_delta"
    assert (
        _expect_stream_event_type(
            first_reasoning_delta.stream_event, ReasoningDeltaEvent
        ).delta
        == "Thinking about weather"
    )
    assert first_reasoning_end.stream_event.type == "reasoning_end"
    assert (
        _expect_stream_event_type(
            first_reasoning_end.stream_event, ReasoningEndEvent
        ).block
        == reasoning_block
    )
    assert first_tool_call_start.stream_event.type == "tool_call_start"
    assert first_tool_call_start.stream_event.content_index == 1
    assert first_tool_call_delta.stream_event.type == "tool_call_delta"
    assert (
        _expect_stream_event_type(
            first_tool_call_delta.stream_event, ToolCallDeltaEvent
        ).delta
        == '{"city":"Munich"}'
    )
    assert first_tool_call_end.stream_event.type == "tool_call_end"
    assert (
        _expect_stream_event_type(
            first_tool_call_end.stream_event, ToolCallEndEvent
        ).block
        == tool_call_block
    )
    assert first_final_message.response_id == "resp_tool_call"
    assert first_final_message.stop_reason == "tool_use"
    assert tool_execution_start.call_id == "call_123"
    assert tool_execution_start.tool_name == "get_weather"
    assert tool_execution_start.arguments == {"city": "Munich"}
    assert tool_execution_end.call_id == "call_123"
    assert tool_execution_end.tool_name == "get_weather"
    assert _tool_text(tool_execution_end.result) == (
        '{"temperature_c": 18, "condition": "sunny", "city": "Munich"}'
    )
    assert tool_execution_end.is_error is False
    assert first_turn_end.message.response_id == "resp_tool_call"
    assert first_turn_end.message.stop_reason == "tool_use"
    assert first_turn_end.message.status == "completed"
    assert first_turn_end.message.blocks == [tool_call_block]
    assert first_turn_end.tool_results[0].call_id == "call_123"
    assert first_turn_end.tool_results[0].tool_name == "get_weather"
    assert first_turn_end.tool_results[0].content == tool_execution_end.result.content
    assert first_turn_end.tool_results[0].is_error is False
    assert second_turn_start.type == "turn_start"
    assert second_message_start.response_id == "resp_follow_up"
    assert second_text_start.stream_event.type == "text_start"
    assert second_text_start.stream_event.content_index == 0
    assert second_text_delta.stream_event.type == "text_delta"
    assert _expect_stream_event_type(
        second_text_delta.stream_event, TextDeltaEvent
    ).delta == ("It is sunny in Munich.")
    assert second_text_end.stream_event.type == "text_end"
    assert (
        _expect_stream_event_type(second_text_end.stream_event, TextEndEvent).block
        == text_block
    )
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
    assert len(agent_end.new_items) == 3
    first_item = _expect_assistant_turn(agent_end.new_items[0])
    second_item = _expect_tool_result_turn(agent_end.new_items[1])
    third_item = _expect_assistant_turn(agent_end.new_items[2])
    assert first_item.response_id == "resp_tool_call"
    assert second_item.tool_name == "get_weather"
    assert third_item.response_id == "resp_follow_up"


def test_agent_run_executes_registered_tool_definition() -> None:
    """Execute a registered tool and expose its result through agent events."""

    tools = _sample_tools()
    invocations: list[StreamInvocation] = []
    stream_fn = _build_stream_fn(
        streams=[
            [
                _stream_started("resp_tool_call"),
                _stream_done(
                    "resp_tool_call",
                    stop_reason="tool_use",
                    blocks=[
                        _tool_call_block(
                            call_id="call_123",
                            name="get_weather",
                            arguments={"city": "Munich"},
                            provider_item_id="fc_123",
                        )
                    ],
                ),
            ],
            [
                _stream_started("resp_follow_up"),
                _stream_done("resp_follow_up"),
            ],
        ],
        invocations=invocations,
    )
    history = [UserMessage(content="What is the weather in Munich?")]

    events = _collect_run_events(history, stream_fn=stream_fn, tools=tools)

    tool_execution_end = _expect_event_type(events[5], ToolExecutionEndEvent)
    assert _tool_text(tool_execution_end.result) == (
        '{"temperature_c": 18, "condition": "sunny", "city": "Munich"}'
    )
    assert tool_execution_end.is_error is False


def test_agent_run_continues_after_tool_execution_error() -> None:
    """Return error tool results to the model and continue the run."""

    failing_tool = ToolDefinition(
        name="fail_weather",
        description="Raise a deterministic weather failure.",
        input_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
        fn=_raise_tool_error,
    )
    invocations: list[StreamInvocation] = []
    stream_fn = _build_stream_fn(
        streams=[
            _tool_call_stream(
                response_id="resp_tool_call",
                call_id="call_123",
                tool_name="fail_weather",
                arguments={"city": "Munich"},
            ),
            _final_text_stream(
                response_id="resp_follow_up",
                text="The tool failed.",
            ),
        ],
        invocations=invocations,
    )
    history = [UserMessage(content="What is the weather in Munich?")]

    events = _collect_run_events(history, stream_fn=stream_fn, tools=[failing_tool])

    tool_execution_end = _expect_event_type(events[5], ToolExecutionEndEvent)
    second_request_tool_result = _expect_tool_result_turn(invocations[1].history[2])
    agent_end = _expect_event_type(events[-1], AgentEndEvent)
    assert tool_execution_end.is_error is True
    assert _tool_text(tool_execution_end.result) == "boom"
    assert second_request_tool_result.is_error is True
    assert second_request_tool_result.tool_name == "fail_weather"
    assert len(agent_end.new_items) == 3
    assert _expect_tool_result_turn(agent_end.new_items[1]).is_error is True


def test_agent_run_handles_multiple_tool_use_turns() -> None:
    """Carry cumulative run-local history through repeated tool turns."""

    tools = _sample_tools()
    invocations: list[StreamInvocation] = []
    stream_fn = _build_stream_fn(
        streams=[
            _tool_call_stream(
                response_id="resp_tool_call_1",
                call_id="call_1",
                tool_name="get_weather",
                arguments={"city": "Munich"},
            ),
            _tool_call_stream(
                response_id="resp_tool_call_2",
                call_id="call_2",
                tool_name="get_weather",
                arguments={"city": "Berlin"},
            ),
            _final_text_stream(response_id="resp_final", text="Both cities are sunny."),
        ],
        invocations=invocations,
    )
    history = [UserMessage(content="Compare Munich and Berlin weather.")]

    events = _collect_run_events(history, stream_fn=stream_fn, tools=tools)

    agent_end = _expect_event_type(events[-1], AgentEndEvent)
    assert len(invocations) == 3
    assert len(invocations[1].history) == 3
    assert len(invocations[2].history) == 5
    assert _expect_assistant_turn(invocations[2].history[1]).response_id == (
        "resp_tool_call_1"
    )
    assert _expect_tool_result_turn(invocations[2].history[2]).call_id == "call_1"
    assert _expect_assistant_turn(invocations[2].history[3]).response_id == (
        "resp_tool_call_2"
    )
    assert _expect_tool_result_turn(invocations[2].history[4]).call_id == "call_2"
    assert len(agent_end.new_items) == 5
    assert _expect_assistant_turn(agent_end.new_items[0]).response_id == (
        "resp_tool_call_1"
    )
    assert _expect_tool_result_turn(agent_end.new_items[1]).call_id == "call_1"
    assert _expect_assistant_turn(agent_end.new_items[2]).response_id == (
        "resp_tool_call_2"
    )
    assert _expect_tool_result_turn(agent_end.new_items[3]).call_id == "call_2"
    assert _expect_assistant_turn(agent_end.new_items[4]).response_id == "resp_final"


def test_agent_includes_cwd_in_stream_instructions(tmp_path: Path) -> None:
    """Add the agent working directory to model instructions."""

    invocations: list[StreamInvocation] = []
    stream_fn = _build_stream_fn(
        streams=[
            [
                _stream_started("resp_done"),
                _stream_done("resp_done"),
            ]
        ],
        invocations=invocations,
    )
    history = [UserMessage(content="Hello")]

    _collect_run_events(
        history,
        stream_fn=stream_fn,
        system_prompt="Base prompt.",
        cwd=tmp_path,
    )

    assert invocations[0].instructions == "Base prompt."


def test_agent_formats_cwd_prompt_variable(tmp_path: Path) -> None:
    """Apply cwd prompt variables before sending model instructions."""

    invocations: list[StreamInvocation] = []
    stream_fn = _build_stream_fn(
        streams=[
            [
                _stream_started("resp_done"),
                _stream_done("resp_done"),
            ]
        ],
        invocations=invocations,
    )
    history = [UserMessage(content="Hello")]

    _collect_run_events(
        history,
        stream_fn=stream_fn,
        system_prompt="Current working directory: {cwd}",
        cwd=tmp_path,
    )

    assert invocations[0].instructions == f"Current working directory: {tmp_path}"


def test_agent_run_yields_error_turn_end_for_stream_error() -> None:
    """Finalize an errored assistant stream as an error turn."""

    invocations: list[StreamInvocation] = []
    stream_fn = _build_stream_fn(
        streams=[
            [
                _stream_started("resp_error"),
                _stream_error("resp_error", "Socket closed"),
            ]
        ],
        invocations=invocations,
    )
    history = [UserMessage(content="Say hello")]

    events = _collect_run_events(history, stream_fn=stream_fn)

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
    final_message = _expect_agent_assistant_turn(message_end.message)

    assert isinstance(events[0], AgentStartEvent)
    assert isinstance(events[1], TurnStartEvent)
    assert message_start.response_id == "resp_error"
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
    assert len(agent_end.new_items) == 1
    first_item = _expect_assistant_turn(agent_end.new_items[0])
    assert first_item.response_id == "resp_error"
