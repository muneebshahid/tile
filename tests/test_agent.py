"""Tests for translating stream events into agent events.

These tests document the final streaming lifecycle layer. The agent receives
app-level ``StreamEvent`` models, emits ``message_start`` and ``message_update``
events while the assistant message is streaming, finalizes history on ``done``
or ``error``, and executes tools before starting a follow-up assistant turn.
"""

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from agent.agent import run_agent
from agent.tool_executor import ToolExecutor
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
    ConversationItem,
    ToolResultTurn,
    UserMessage,
)
from ai.types.stream_events import (
    ReasoningBlock,
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    ReasoningStartEvent,
    ProviderStreamEvent,
    TextBlock,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolCallBlock,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from ai.types.tools import (
    ReadDetails,
    ToolDefinition,
    ToolOutputDetails,
    ToolResult,
    ToolTextContent,
)
from tests.support.agent_streams import (
    StreamInvocation,
    build_stream_fn,
    final_text_stream,
    stream_done,
    stream_error,
    stream_start,
    tool_call_block,
    tool_call_stream,
)
from tests.support.conversation_assertions import (
    expect_assistant_turn,
    expect_tool_result_turn,
    expect_user_message,
)
from tests.support.stream_assertions import expect_stream_event

TEvent = TypeVar("TEvent", bound=AgentEvent)


@dataclass(frozen=True)
class ToolUseLoopRun:
    """Captured events and expected blocks for the weather tool-loop scenario."""

    events: list[AgentEvent]
    invocations: list[StreamInvocation]
    tools: list[ToolDefinition]
    reasoning_block: ReasoningBlock
    tool_call_block: ToolCallBlock
    text_block: TextBlock


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
                tool_executor=ToolExecutor(tools),
                reasoning=reasoning,
                system_prompt=system_prompt,
                cwd=cwd,
            )
        ]

    return asyncio.run(_collect())


def _expect_event_type(event: AgentEvent, event_type: type[TEvent]) -> TEvent:
    """Assert and return an agent event with a precise type."""

    assert isinstance(event, event_type)
    return event


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


async def _read_file() -> ToolResult:
    """Return a deterministic file read result with runtime metadata."""

    return ToolResult.text(
        "file contents",
        details=ReadDetails(output=_tool_output_details()),
    )


def _tool_output_details() -> ToolOutputDetails:
    """Build deterministic bounded-output metadata for tool tests."""

    return ToolOutputDetails(
        truncated=False,
        truncated_by=None,
        keep="head",
        total_lines=1,
        total_bytes=13,
        output_lines=1,
        output_bytes=13,
        edge_line_exceeds_limit=False,
        max_lines=100,
        max_bytes=1024,
    )


def _collect_weather_tool_loop_run() -> ToolUseLoopRun:
    """Run a two-turn weather tool loop and capture its observable state."""

    tools = _sample_tools()
    invocations: list[StreamInvocation] = []
    reasoning_block = ReasoningBlock(summary_text="Thinking about weather")
    weather_tool_call_block = tool_call_block(
        call_id="call_123",
        name="get_weather",
        arguments={"city": "Munich"},
        provider_item_id="fc_123",
    )
    text_block = TextBlock(text="It is sunny in Munich.")
    stream_fn = build_stream_fn(
        streams=[
            _weather_tool_call_stream(reasoning_block, weather_tool_call_block),
            _weather_follow_up_stream(text_block),
        ],
        invocations=invocations,
    )
    history: list[ConversationItem] = [
        UserMessage(content="What is the weather in Munich?")
    ]

    events = _collect_run_events(history, stream_fn=stream_fn, tools=tools)
    return ToolUseLoopRun(
        events=events,
        invocations=invocations,
        tools=tools,
        reasoning_block=reasoning_block,
        tool_call_block=weather_tool_call_block,
        text_block=text_block,
    )


def _weather_tool_call_stream(
    reasoning_block: ReasoningBlock,
    weather_tool_call_block: ToolCallBlock,
) -> list[ProviderStreamEvent]:
    """Build a provider stream that requests the weather tool with deltas."""

    return [
        stream_start("resp_tool_call"),
        ReasoningStartEvent(content_index=0),
        ReasoningDeltaEvent(content_index=0, delta="Thinking about weather"),
        ReasoningEndEvent(content_index=0, block=reasoning_block),
        ToolCallStartEvent(
            content_index=1,
            call_id="call_123",
            name="get_weather",
        ),
        ToolCallDeltaEvent(content_index=1, delta='{"city":"Munich"}'),
        ToolCallEndEvent(content_index=1, block=weather_tool_call_block),
        stream_done(
            "resp_tool_call",
            stop_reason="tool_use",
            blocks=[weather_tool_call_block],
        ),
    ]


def _weather_follow_up_stream(
    text_block: TextBlock,
) -> list[ProviderStreamEvent]:
    """Build a provider stream that answers after tool execution."""

    return [
        stream_start("resp_follow_up"),
        TextStartEvent(content_index=0),
        TextDeltaEvent(content_index=0, delta="It is sunny in Munich."),
        TextEndEvent(content_index=0, block=text_block),
        stream_done("resp_follow_up", blocks=[text_block]),
    ]


def test_run_agent_does_not_mutate_supplied_history() -> None:
    """Keep caller-owned history unchanged while emitting stable events."""

    invocations: list[StreamInvocation] = []
    stream_fn = build_stream_fn(
        streams=[
            [
                stream_start("resp_done"),
                stream_done("resp_done"),
            ]
        ],
        invocations=invocations,
    )
    history: list[ConversationItem] = [UserMessage(content="Hello, piy")]

    events = _collect_run_events(history, stream_fn=stream_fn)

    message_end = _expect_event_type(events[3], MessageEndEvent)
    _expect_event_type(events[-1], AgentEndEvent)
    assert history == [UserMessage(content="Hello, piy")]
    assert message_end.assistant_turn.response_id == "resp_done"


def test_agent_run_yields_expected_event_sequence_for_tool_use_loop() -> None:
    """Emit the expected event sequence for a tool-use loop."""

    run = _collect_weather_tool_loop_run()

    assert [event.type for event in run.events] == [
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


def test_agent_run_yields_current_tool_use_stream_events() -> None:
    """Forward first-turn reasoning and tool-call stream events."""

    run = _collect_weather_tool_loop_run()
    events = run.events

    first_message_start = _expect_event_type(events[2], MessageStartEvent)
    first_reasoning_start = _expect_event_type(events[3], MessageUpdateEvent)
    first_reasoning_delta = _expect_event_type(events[4], MessageUpdateEvent)
    first_reasoning_end = _expect_event_type(events[5], MessageUpdateEvent)
    first_tool_call_start = _expect_event_type(events[6], MessageUpdateEvent)
    first_tool_call_delta = _expect_event_type(events[7], MessageUpdateEvent)
    first_tool_call_end = _expect_event_type(events[8], MessageUpdateEvent)
    assert isinstance(events[0], AgentStartEvent)
    assert isinstance(events[1], TurnStartEvent)
    assert first_message_start.response_id == "resp_tool_call"
    assert first_reasoning_start.stream_event.type == "reasoning_start"
    assert first_reasoning_start.stream_event.content_index == 0
    assert first_reasoning_delta.stream_event.type == "reasoning_delta"
    assert (
        expect_stream_event(
            first_reasoning_delta.stream_event, ReasoningDeltaEvent
        ).delta
        == "Thinking about weather"
    )
    assert first_reasoning_end.stream_event.type == "reasoning_end"
    assert (
        expect_stream_event(first_reasoning_end.stream_event, ReasoningEndEvent).block
        == run.reasoning_block
    )
    assert first_tool_call_start.stream_event.type == "tool_call_start"
    assert first_tool_call_start.stream_event.content_index == 1
    assert first_tool_call_delta.stream_event.type == "tool_call_delta"
    assert (
        expect_stream_event(
            first_tool_call_delta.stream_event, ToolCallDeltaEvent
        ).delta
        == '{"city":"Munich"}'
    )
    assert first_tool_call_end.stream_event.type == "tool_call_end"
    assert (
        expect_stream_event(first_tool_call_end.stream_event, ToolCallEndEvent).block
        == run.tool_call_block
    )


def test_agent_run_yields_current_follow_up_stream_events() -> None:
    """Forward second-turn text stream events after tool execution."""

    run = _collect_weather_tool_loop_run()
    events = run.events

    second_message_start = _expect_event_type(events[14], MessageStartEvent)
    second_text_start = _expect_event_type(events[15], MessageUpdateEvent)
    second_text_delta = _expect_event_type(events[16], MessageUpdateEvent)
    second_text_end = _expect_event_type(events[17], MessageUpdateEvent)
    _expect_event_type(events[20], AgentEndEvent)

    assert isinstance(events[13], TurnStartEvent)
    assert second_message_start.response_id == "resp_follow_up"
    assert second_text_start.stream_event.type == "text_start"
    assert second_text_start.stream_event.content_index == 0
    assert second_text_delta.stream_event.type == "text_delta"
    assert expect_stream_event(
        second_text_delta.stream_event, TextDeltaEvent
    ).delta == ("It is sunny in Munich.")
    assert second_text_end.stream_event.type == "text_end"
    assert expect_stream_event(second_text_end.stream_event, TextEndEvent).block == (
        run.text_block
    )


def test_agent_run_emits_tool_execution_outcome_for_tool_use_loop() -> None:
    """Emit tool execution details and attach them to the completed turn."""

    run = _collect_weather_tool_loop_run()
    events = run.events

    first_message_end = _expect_event_type(events[9], MessageEndEvent)
    tool_execution_start = _expect_event_type(events[10], ToolExecutionStartEvent)
    tool_execution_end = _expect_event_type(events[11], ToolExecutionEndEvent)
    first_turn_end = _expect_event_type(events[12], TurnEndEvent)
    first_final_message = first_message_end.assistant_turn

    assert first_final_message.response_id == "resp_tool_call"
    assert first_final_message.stop_reason == "tool_use"
    assert tool_execution_start.call_id == "call_123"
    assert tool_execution_start.tool_name == "get_weather"
    assert tool_execution_start.arguments == {"city": "Munich"}
    tool_execution_outcome = tool_execution_end.outcome
    tool_result_turn = tool_execution_outcome.tool_result_turn
    assert tool_result_turn.call_id == "call_123"
    assert tool_result_turn.tool_name == "get_weather"
    assert _tool_text(tool_execution_outcome.result) == (
        '{"temperature_c": 18, "condition": "sunny", "city": "Munich"}'
    )
    assert tool_result_turn.content == tool_execution_outcome.result.content
    assert tool_result_turn.is_error is False
    assert first_turn_end.assistant_turn.response_id == "resp_tool_call"
    assert first_turn_end.assistant_turn.stop_reason == "tool_use"
    assert first_turn_end.assistant_turn.status == "completed"
    assert first_turn_end.assistant_turn.blocks == [run.tool_call_block]
    first_turn_outcome = first_turn_end.tool_executions[0]
    assert first_turn_outcome.tool_result_turn.call_id == "call_123"
    assert first_turn_outcome.tool_result_turn.tool_name == "get_weather"
    assert first_turn_outcome.result.content == tool_execution_outcome.result.content
    assert first_turn_outcome.tool_result_turn.is_error is False


def test_agent_run_sends_tool_result_history_to_follow_up_turn() -> None:
    """Send assistant and tool result turns into the follow-up model request."""

    run = _collect_weather_tool_loop_run()
    events = run.events

    second_message_end = _expect_event_type(events[18], MessageEndEvent)
    second_turn_end = _expect_event_type(events[19], TurnEndEvent)
    second_final_message = second_message_end.assistant_turn

    assert second_final_message.response_id == "resp_follow_up"
    assert second_final_message.stop_reason == "stop"
    assert second_turn_end.assistant_turn.response_id == "resp_follow_up"
    assert second_turn_end.assistant_turn.stop_reason == "stop"
    assert second_turn_end.assistant_turn.blocks == [
        TextBlock(text="It is sunny in Munich.")
    ]
    assert second_turn_end.tool_executions == []
    assert len(run.invocations) == 2
    assert run.invocations[0].model == "gpt-5.4"
    assert run.invocations[0].tools == tuple(run.tools)
    assert run.invocations[1].tools == tuple(run.tools)
    first_request_user = expect_user_message(run.invocations[0].history[0])
    second_request_assistant = expect_assistant_turn(run.invocations[1].history[1])
    second_request_tool_result = expect_tool_result_turn(run.invocations[1].history[2])
    assert first_request_user.content == "What is the weather in Munich?"
    assert second_request_assistant.response_id == "resp_tool_call"
    assert second_request_tool_result.tool_name == "get_weather"


def test_agent_run_executes_registered_tool_definition() -> None:
    """Execute a registered tool and expose its result through agent events."""

    tools = _sample_tools()
    invocations: list[StreamInvocation] = []
    stream_fn = build_stream_fn(
        streams=[
            [
                stream_start("resp_tool_call"),
                stream_done(
                    "resp_tool_call",
                    stop_reason="tool_use",
                    blocks=[
                        tool_call_block(
                            call_id="call_123",
                            name="get_weather",
                            arguments={"city": "Munich"},
                            provider_item_id="fc_123",
                        )
                    ],
                ),
            ],
            [
                stream_start("resp_follow_up"),
                stream_done("resp_follow_up"),
            ],
        ],
        invocations=invocations,
    )
    history = [UserMessage(content="What is the weather in Munich?")]

    events = _collect_run_events(history, stream_fn=stream_fn, tools=tools)

    tool_execution_end = _expect_event_type(events[5], ToolExecutionEndEvent)
    assert _tool_text(tool_execution_end.outcome.result) == (
        '{"temperature_c": 18, "condition": "sunny", "city": "Munich"}'
    )
    assert tool_execution_end.outcome.tool_result_turn.is_error is False


def test_agent_run_exposes_tool_details_outside_replay_turn() -> None:
    """Expose non-replay tool details through execution outcomes."""

    tools = [
        ToolDefinition(
            name="read_file",
            description="Read a deterministic file.",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            fn=_read_file,
        )
    ]
    invocations: list[StreamInvocation] = []
    stream_fn = build_stream_fn(
        streams=[
            tool_call_stream(
                response_id="resp_tool_call",
                call_id="call_read",
                tool_name="read_file",
                arguments={},
                provider_item_id="fc_call_read",
            ),
            final_text_stream(
                response_id="resp_follow_up",
                text="I read the file.",
            ),
        ],
        invocations=invocations,
    )
    history = [UserMessage(content="Read a file")]

    events = _collect_run_events(history, stream_fn=stream_fn, tools=tools)

    tool_execution_end = _expect_event_type(events[5], ToolExecutionEndEvent)
    turn_end = _expect_event_type(events[6], TurnEndEvent)
    tool_result_turn = expect_tool_result_turn(invocations[1].history[2])
    outcome = tool_execution_end.outcome
    turn_outcome = turn_end.tool_executions[0]
    assert outcome.details == ReadDetails(output=_tool_output_details())
    assert outcome.result.details == outcome.details
    assert turn_outcome.details == outcome.details
    assert tool_result_turn == outcome.tool_result_turn
    assert "details" not in ToolResultTurn.model_fields


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
    stream_fn = build_stream_fn(
        streams=[
            tool_call_stream(
                response_id="resp_tool_call",
                call_id="call_123",
                tool_name="fail_weather",
                arguments={"city": "Munich"},
                provider_item_id="fc_call_123",
            ),
            final_text_stream(
                response_id="resp_follow_up",
                text="The tool failed.",
            ),
        ],
        invocations=invocations,
    )
    history = [UserMessage(content="What is the weather in Munich?")]

    events = _collect_run_events(history, stream_fn=stream_fn, tools=[failing_tool])

    tool_execution_end = _expect_event_type(events[5], ToolExecutionEndEvent)
    second_request_tool_result = expect_tool_result_turn(invocations[1].history[2])
    _expect_event_type(events[-1], AgentEndEvent)
    assert tool_execution_end.outcome.tool_result_turn.is_error is True
    assert _tool_text(tool_execution_end.outcome.result) == "boom"
    assert tool_execution_end.outcome.tool_result_turn.tool_name == "fail_weather"
    assert second_request_tool_result.is_error is True
    assert second_request_tool_result.tool_name == "fail_weather"


def test_agent_run_handles_multiple_tool_use_turns() -> None:
    """Carry cumulative run-local history through repeated tool turns."""

    tools = _sample_tools()
    invocations: list[StreamInvocation] = []
    stream_fn = build_stream_fn(
        streams=[
            tool_call_stream(
                response_id="resp_tool_call_1",
                call_id="call_1",
                tool_name="get_weather",
                arguments={"city": "Munich"},
                provider_item_id="fc_call_1",
            ),
            tool_call_stream(
                response_id="resp_tool_call_2",
                call_id="call_2",
                tool_name="get_weather",
                arguments={"city": "Berlin"},
                provider_item_id="fc_call_2",
            ),
            final_text_stream(response_id="resp_final", text="Both cities are sunny."),
        ],
        invocations=invocations,
    )
    history = [UserMessage(content="Compare Munich and Berlin weather.")]

    events = _collect_run_events(history, stream_fn=stream_fn, tools=tools)

    _expect_event_type(events[-1], AgentEndEvent)
    assert len(invocations) == 3
    assert len(invocations[1].history) == 3
    assert len(invocations[2].history) == 5
    assert expect_assistant_turn(invocations[2].history[1]).response_id == (
        "resp_tool_call_1"
    )
    assert expect_tool_result_turn(invocations[2].history[2]).call_id == "call_1"
    assert expect_assistant_turn(invocations[2].history[3]).response_id == (
        "resp_tool_call_2"
    )
    assert expect_tool_result_turn(invocations[2].history[4]).call_id == "call_2"


def test_agent_leaves_instructions_unchanged_without_cwd_variable(
    tmp_path: Path,
) -> None:
    """Do not append the working directory unless the prompt requests it."""

    invocations: list[StreamInvocation] = []
    stream_fn = build_stream_fn(
        streams=[
            [
                stream_start("resp_done"),
                stream_done("resp_done"),
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
    stream_fn = build_stream_fn(
        streams=[
            [
                stream_start("resp_done"),
                stream_done("resp_done"),
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
    stream_fn = build_stream_fn(
        streams=[
            [
                stream_start("resp_error"),
                stream_error("resp_error", "Socket closed"),
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
    _expect_event_type(events[5], AgentEndEvent)
    final_message = message_end.assistant_turn

    assert isinstance(events[0], AgentStartEvent)
    assert isinstance(events[1], TurnStartEvent)
    assert message_start.response_id == "resp_error"
    assert final_message.response_id == "resp_error"
    assert final_message.status == "error"
    assert turn_end.assistant_turn.response_id == "resp_error"
    assert turn_end.assistant_turn.stop_reason == "error"
    assert turn_end.assistant_turn.status == "error"
    assert turn_end.assistant_turn.error_message == "Socket closed"
    assert turn_end.tool_executions == []
    assert len(invocations) == 1
    first_request_user = expect_user_message(invocations[0].history[0])
    assert first_request_user.content == "Say hello"
