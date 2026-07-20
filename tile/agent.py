"""Stateless agent run loop for provider streams and tool execution."""

from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import aclosing

from tile.types.conversation import AssistantTurn, ConversationItem
from tile.types.stream_events import (
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
from tile.types.tool_execution import ToolExecutionOutcome
from tile.types.tools import JsonObject
from tile.tool_executor import ToolExecutor
from tile.events import (
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
    instructions: str,
) -> AsyncGenerator[AgentEvent, None]:
    """Run one stateless agent turn from supplied model-visible history.

    A successful tool result with ``terminate=True`` ends the loop after the
    current tool batch without another provider call. ``instructions`` is the
    complete system prompt, sent to the provider verbatim; the caller owns
    its composition.
    """

    run_history = list(history)
    loop_events = _run_agent_loop(
        run_history=run_history,
        stream_fn=stream_fn,
        model=model,
        instructions=instructions,
        tool_executor=tool_executor,
    )

    yield AgentStartEvent()
    async with aclosing(loop_events):
        async for event in loop_events:
            yield event
    yield AgentEndEvent()


async def _run_agent_loop(
    *,
    run_history: list[ConversationItem],
    stream_fn: StreamFn,
    model: str,
    instructions: str,
    tool_executor: ToolExecutor,
) -> AsyncGenerator[AgentEvent, None]:
    """Call the provider until a turn errors, terminates, or stops using tools.

    Each provider stream is closed on every exit — closure does not
    cascade through generator chains on its own, so every layer forwards
    it down to the transport.
    """

    while True:
        has_tool_executions = False
        should_terminate = False
        turn_errored = False
        stream = await stream_fn(
            tuple(run_history),
            model,
            instructions=instructions,
            tools=tool_executor.tools,
        )

        async with aclosing(stream):
            async for event in stream:
                async for agent_event in _handle_stream_event(
                    event,
                    run_history=run_history,
                    tool_executor=tool_executor,
                ):
                    if (
                        isinstance(agent_event, ToolExecutionEndEvent)
                        and agent_event.outcome.terminate
                    ):
                        should_terminate = True
                    if isinstance(agent_event, TurnEndEvent):
                        if agent_event.tool_executions:
                            has_tool_executions = True
                        if agent_event.assistant_turn.status != "completed":
                            turn_errored = True
                    yield agent_event

        if should_terminate or turn_errored or not has_tool_executions:
            return


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
    yield MessageEndEvent(assistant_turn=turn, token_usage=event.usage)
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
    yield MessageEndEvent(assistant_turn=turn, token_usage=event.usage)
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


def _collect_tool_calls(blocks: Sequence[AssistantBlock]) -> list[ToolCallBlock]:
    """Collect tool calls from finalized assistant blocks."""

    return [
        block.model_copy(deep=True)
        for block in blocks
        if isinstance(block, ToolCallBlock)
    ]
