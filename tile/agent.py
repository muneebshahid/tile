"""Stateless agent run loop for provider streams and tool execution."""

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from pydantic import BaseModel

from tile.result import (
    MAX_RESULT_FOLLOW_UPS,
    NO_RESULT_REASON,
    RESULT_CONTRACT,
    RESULT_FOLLOW_UP,
    Completed,
    Failed,
    ResultRecorder,
    RunOutcome,
)
from tile.types.conversation import AssistantTurn, ConversationItem, UserMessage
from tile.types.stream_events import (
    AssistantBlock,
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    ReasoningStartEvent,
    ProviderStreamEvent,
    StreamDoneEvent,
    StreamErrorEvent,
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
from tile.types.tools import JsonObject
from tile.prompt import DEFAULT_INSTRUCTIONS, build_system_prompt
from tile.tool_executor import ToolExecutor
from tile.events import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ResultFollowUpEvent,
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
    instructions: str = DEFAULT_INSTRUCTIONS,
    auto_mode: bool = True,
    result: type[BaseModel] | None = None,
    cwd: Path | str | None = None,
) -> AsyncIterator[AgentEvent]:
    """Run one stateless agent turn from supplied model-visible history."""

    run_history = list(history)
    recorder: ResultRecorder | None = None
    if result is not None:
        recorder = ResultRecorder(result)
        tool_executor = ToolExecutor(
            (*tool_executor.tools, *recorder.tool_definitions())
        )
        instructions = f"{instructions}\n\n{RESULT_CONTRACT}"
    system_prompt = build_system_prompt(
        instructions,
        _resolve_cwd(cwd),
        auto_mode=auto_mode,
    )

    last_turn: AssistantTurn | None = None

    yield AgentStartEvent()
    async for event in _run_agent_loop(
        run_history=run_history,
        stream_fn=stream_fn,
        model=model,
        instructions=system_prompt,
        tool_executor=tool_executor,
        recorder=recorder,
    ):
        if isinstance(event, MessageEndEvent):
            last_turn = event.assistant_turn
        yield event
    yield AgentEndEvent(outcome=_build_outcome(recorder, last_turn))


async def _run_agent_loop(
    *,
    run_history: list[ConversationItem],
    stream_fn: StreamFn,
    model: str,
    instructions: str,
    tool_executor: ToolExecutor,
    recorder: ResultRecorder | None,
) -> AsyncIterator[AgentEvent]:
    """Call the provider until the run reaches a terminal turn or result."""

    follow_ups = 0
    while True:
        has_tool_executions = False
        turn_errored = False
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
                if isinstance(agent_event, TurnEndEvent):
                    if agent_event.tool_executions:
                        has_tool_executions = True
                    if agent_event.assistant_turn.status != "completed":
                        turn_errored = True
                yield agent_event

        if recorder is not None and recorder.has_outcome:
            return
        if turn_errored:
            return
        if has_tool_executions:
            continue
        if recorder is None:
            return
        if follow_ups >= MAX_RESULT_FOLLOW_UPS:
            return
        follow_ups += 1
        follow_up = UserMessage(content=RESULT_FOLLOW_UP)
        run_history.append(follow_up)
        yield ResultFollowUpEvent(message=follow_up)


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


def _build_outcome(
    recorder: ResultRecorder | None,
    last_turn: AssistantTurn | None,
) -> RunOutcome | None:
    """Derive the terminal run outcome from the recorder and the ending turn."""

    if last_turn is None or last_turn.status != "completed":
        return None
    output_text = _turn_text(last_turn)
    if recorder is None:
        return Completed(output_text=output_text)
    if recorder.value is not None:
        return Completed(value=recorder.value, output_text=output_text)
    reason = recorder.reason if recorder.reason is not None else NO_RESULT_REASON
    return Failed(reason=reason, output_text=output_text)


def _turn_text(turn: AssistantTurn) -> str:
    """Join the text blocks of one assistant turn with blank lines."""

    return "\n\n".join(
        block.text for block in turn.blocks if isinstance(block, TextBlock)
    )


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
