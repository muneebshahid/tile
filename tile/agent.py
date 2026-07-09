"""Stateless agent run loop for provider streams and tool execution."""

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from tile.result import (
    COMPLETE_TOOL_NAME,
    FAIL_TOOL_NAME,
    MAX_RESULT_FOLLOW_UPS,
    NO_RESULT_REASON,
    RESULT_ALREADY_RECORDED,
    RESULT_FOLLOW_UP,
    Completed,
    Failed,
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
from tile.types.tools import JsonObject, ToolResult
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
    enforce_output_contract: bool = False,
    cwd: Path | str | None = None,
) -> AsyncIterator[AgentEvent]:
    """Run one stateless agent turn from supplied model-visible history."""

    run_history = list(history)
    if enforce_output_contract:
        _require_result_tools(tool_executor)
    system_prompt = build_system_prompt(
        instructions,
        _resolve_cwd(cwd),
        auto_mode=auto_mode,
    )

    last_turn: AssistantTurn | None = None
    terminal_result: RunOutcome | None = None

    yield AgentStartEvent()
    async for event in _run_agent_loop(
        run_history=run_history,
        stream_fn=stream_fn,
        model=model,
        instructions=system_prompt,
        tool_executor=tool_executor,
        enforce_output_contract=enforce_output_contract,
    ):
        if isinstance(event, MessageEndEvent):
            last_turn = event.assistant_turn
        if (
            enforce_output_contract
            and terminal_result is None
            and isinstance(event, TurnEndEvent)
        ):
            terminal_result = _terminal_result(event)
        yield event
    yield AgentEndEvent(
        outcome=_build_outcome(
            enforce_output_contract=enforce_output_contract,
            terminal_result=terminal_result,
            last_turn=last_turn,
        )
    )


async def _run_agent_loop(
    *,
    run_history: list[ConversationItem],
    stream_fn: StreamFn,
    model: str,
    instructions: str,
    tool_executor: ToolExecutor,
    enforce_output_contract: bool,
) -> AsyncIterator[AgentEvent]:
    """Call the provider until the run reaches a terminal turn or result."""

    follow_ups = 0
    while True:
        has_tool_executions = False
        has_result = False
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
                enforce_output_contract=enforce_output_contract,
            ):
                if isinstance(agent_event, TurnEndEvent):
                    if agent_event.tool_executions:
                        has_tool_executions = True
                    if agent_event.assistant_turn.status != "completed":
                        turn_errored = True
                    if (
                        enforce_output_contract
                        and _terminal_result(agent_event) is not None
                    ):
                        has_result = True
                yield agent_event

        if has_result or turn_errored:
            return
        if has_tool_executions:
            continue
        if not enforce_output_contract:
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
    enforce_output_contract: bool,
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
                enforce_output_contract=enforce_output_contract,
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
    enforce_output_contract: bool,
) -> AsyncIterator[AgentEvent]:
    """Finalize an assistant message and execute requested tools."""

    turn = AssistantTurn.from_stream_done(event)
    run_history.append(turn)
    yield MessageEndEvent(assistant_turn=turn)
    tool_executions: list[ToolExecutionOutcome] = []
    result_recorded = False

    for tool_call in _collect_tool_calls(turn.blocks):
        lifecycle = (
            _skip_tool(
                call_id=tool_call.call_id,
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
            )
            if result_recorded
            else _execute_tool(
                call_id=tool_call.call_id,
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                tool_executor=tool_executor,
            )
        )
        async for agent_event in lifecycle:
            if isinstance(agent_event, ToolExecutionEndEvent):
                outcome = agent_event.outcome
                run_history.append(outcome.tool_result_turn)
                tool_executions.append(outcome)
                if enforce_output_contract and _is_result_execution(outcome):
                    result_recorded = True
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


async def _skip_tool(
    *,
    call_id: str,
    tool_name: str,
    arguments: JsonObject,
) -> AsyncIterator[AgentEvent]:
    """Answer a post-result tool call with an error without executing it."""

    yield ToolExecutionStartEvent(
        call_id=call_id,
        tool_name=tool_name,
        arguments=arguments,
    )
    yield ToolExecutionEndEvent(
        outcome=ToolExecutionOutcome.from_result(
            call_id=call_id,
            tool_name=tool_name,
            result=ToolResult.text(RESULT_ALREADY_RECORDED),
            is_error=True,
        )
    )


def _is_result_execution(outcome: ToolExecutionOutcome) -> bool:
    """Return whether an execution successfully recorded a run result."""

    result_turn = outcome.tool_result_turn
    return not result_turn.is_error and result_turn.tool_name in (
        COMPLETE_TOOL_NAME,
        FAIL_TOOL_NAME,
    )


def _build_outcome(
    *,
    enforce_output_contract: bool,
    terminal_result: RunOutcome | None,
    last_turn: AssistantTurn | None,
) -> RunOutcome | None:
    """Derive the terminal run outcome for the agent end event."""

    if last_turn is None or last_turn.status != "completed":
        return None
    if terminal_result is not None:
        return terminal_result
    output_text = _turn_text(last_turn)
    if not enforce_output_contract:
        return Completed(output_text=output_text)
    return Failed(reason=NO_RESULT_REASON, output_text=output_text)


def _terminal_result(turn_end: TurnEndEvent) -> RunOutcome | None:
    """Return the outcome recorded by a turn's first successful result call."""

    arguments_by_call_id = {
        block.call_id: block.arguments
        for block in _collect_tool_calls(turn_end.assistant_turn.blocks)
    }
    output_text = _turn_text(turn_end.assistant_turn)
    for execution in turn_end.tool_executions:
        result_turn = execution.tool_result_turn
        if result_turn.is_error:
            continue
        arguments = arguments_by_call_id.get(result_turn.call_id, {})
        if result_turn.tool_name == COMPLETE_TOOL_NAME:
            return Completed(value=arguments, output_text=output_text)
        if result_turn.tool_name == FAIL_TOOL_NAME:
            return Failed(
                reason=str(arguments.get("reason", "")),
                output_text=output_text,
            )
    return None


def _require_result_tools(tool_executor: ToolExecutor) -> None:
    """Reject contract enforcement without both result tools registered."""

    names = {tool.name.lower() for tool in tool_executor.tools}
    missing = {COMPLETE_TOOL_NAME, FAIL_TOOL_NAME} - names
    if missing:
        raise ValueError(
            "enforce_output_contract requires result tools named: "
            + ", ".join(sorted(missing))
        )


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
