"""Stateless agent run loop for provider streams and tool execution."""

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from ori.types.conversation import AssistantTurn, ConversationItem
from ori.types.stream_events import (
    AssistantBlock,
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    ReasoningStartEvent,
    ProviderStreamEvent,
    StreamDoneEvent,
    StreamErrorEvent,
    StreamStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolCallBlock,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from ori.types.tools import JsonObject
from ori.prompt import PROMPT, build_system_prompt
from ori.tool_executor import ToolExecutor
from ori.events import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    StreamFn,
    ToolExecutionEndEvent,
    ToolExecutionOutcome,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)

ASSISTANT_MESSAGE_UPDATE_EVENT_TYPES = (
    ReasoningStartEvent,
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    TextStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    ToolCallStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
)


async def run_agent(
    history: Sequence[ConversationItem],
    *,
    stream_fn: StreamFn,
    model: str,
    tool_executor: ToolExecutor,
    system_prompt: str = PROMPT,
    cwd: Path | str | None = None,
) -> AsyncIterator[AgentEvent]:
    """Run one stateless agent turn from supplied model-visible history."""

    run_history = list(history)
    instructions = build_system_prompt(system_prompt, _resolve_cwd(cwd))

    yield AgentStartEvent()
    async for event in _run_agent_loop(
        run_history=run_history,
        stream_fn=stream_fn,
        model=model,
        instructions=instructions,
        tool_executor=tool_executor,
    ):
        yield event
    yield AgentEndEvent()


async def _run_agent_loop(
    *,
    run_history: list[ConversationItem],
    stream_fn: StreamFn,
    model: str,
    instructions: str,
    tool_executor: ToolExecutor,
) -> AsyncIterator[AgentEvent]:
    """Call the provider until the assistant stops requesting tools."""

    while True:
        has_tool_executions = False
        stream = await stream_fn(
            tuple(run_history),
            model,
            instructions=instructions,
            tools=tool_executor.tools,
        )

        async for event in stream:
            async for agent_event in _handle_stream_event(
                event,
                run_history=run_history,
                tool_executor=tool_executor,
            ):
                if (
                    isinstance(agent_event, TurnEndEvent)
                    and agent_event.tool_executions
                ):
                    has_tool_executions = True
                yield agent_event

        if not has_tool_executions:
            break


async def _handle_stream_event(
    event: ProviderStreamEvent,
    *,
    run_history: list[ConversationItem],
    tool_executor: ToolExecutor,
) -> AsyncIterator[AgentEvent]:
    """Route one provider stream event into agent-level events."""

    match event:
        case StreamStartEvent():
            yield TurnStartEvent()
            yield MessageStartEvent(response_id=event.response_id)
        case StreamDoneEvent():
            async for agent_event in _handle_stream_done_event(
                event,
                run_history=run_history,
                tool_executor=tool_executor,
            ):
                yield agent_event
        case StreamErrorEvent():
            async for agent_event in _handle_stream_error_event(
                event,
                run_history=run_history,
            ):
                yield agent_event
        case _ if isinstance(event, ASSISTANT_MESSAGE_UPDATE_EVENT_TYPES):
            yield MessageUpdateEvent(stream_event=event)


async def _handle_stream_done_event(
    event: StreamDoneEvent,
    *,
    run_history: list[ConversationItem],
    tool_executor: ToolExecutor,
) -> AsyncIterator[AgentEvent]:
    """Finalize an assistant message and execute requested tools."""

    turn = AssistantTurn.from_stream_done(event)
    run_history.append(turn)
    yield MessageEndEvent(assistant_turn=turn)
    tool_executions: list[ToolExecutionOutcome] = []

    for tool_call in _collect_tool_calls(turn.blocks):
        async for agent_event in _execute_tool(
            call_id=tool_call.call_id,
            tool_name=tool_call.name,
            arguments=tool_call.arguments,
            tool_executor=tool_executor,
        ):
            if isinstance(agent_event, ToolExecutionEndEvent):
                outcome = agent_event.outcome
                run_history.append(outcome.tool_result_turn)
                tool_executions.append(outcome)
            yield agent_event

    yield TurnEndEvent(assistant_turn=turn, tool_executions=tool_executions)


async def _handle_stream_error_event(
    event: StreamErrorEvent,
    *,
    run_history: list[ConversationItem],
) -> AsyncIterator[AgentEvent]:
    """Finalize a failed assistant message."""

    turn = AssistantTurn.from_stream_error(event)
    run_history.append(turn)
    yield MessageEndEvent(assistant_turn=turn)
    yield TurnEndEvent(assistant_turn=turn, tool_executions=[])


async def _execute_tool(
    *,
    call_id: str,
    tool_name: str,
    arguments: JsonObject,
    tool_executor: ToolExecutor,
) -> AsyncIterator[AgentEvent]:
    """Emit the full lifecycle for one tool execution."""

    yield ToolExecutionStartEvent(
        call_id=call_id,
        tool_name=tool_name,
        arguments=arguments,
    )

    outcome = await tool_executor.execute(
        call_id=call_id,
        tool_name=tool_name,
        arguments=arguments,
    )
    yield ToolExecutionEndEvent(outcome=outcome)


def _resolve_cwd(cwd: Path | str | None) -> Path:
    """Resolve the agent working directory."""

    if cwd is None:
        return Path.cwd().resolve()
    return Path(cwd).expanduser().resolve()


def _collect_tool_calls(blocks: Sequence[AssistantBlock]) -> list[ToolCallBlock]:
    """Collect tool calls from finalized assistant blocks."""

    return [
        block.model_copy(deep=True)
        for block in blocks
        if isinstance(block, ToolCallBlock)
    ]
