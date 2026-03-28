import json
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Literal

from pydantic import JsonValue

from ai.types.contracts import Reasoning
from ai.types.conversation import (
    AssistantTurn,
    ConversationItem,
    ToolResultTurn,
    UserMessage,
)
from ai.types.stream import (
    AssistantMessage,
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    ReasoningStartEvent,
    StreamDoneEvent,
    StreamErrorEvent,
    StreamEvent,
    StreamStartEvent,
    ToolCallBlock,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
)
from ai.types.tools import JsonObject, ToolDefinition
from agent.prompt import PROMPT
from agent.types import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    MessageStartEvent,
    StreamFn,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)

IGNORED_STREAM_EVENT_TYPES = (
    ReasoningStartEvent,
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    TextStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
)


class Agent:
    def __init__(
        self,
        stream_fn: StreamFn,
        model: str,
        reasoning: Reasoning | None = None,
        tools: Sequence[ToolDefinition] | None = None,
        history: Sequence[ConversationItem] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._stream_fn = stream_fn
        self._model = model
        self._reasoning = reasoning
        self._tools = tuple(tools or ())
        self._history = list(history or [])
        self._system_prompt = system_prompt or PROMPT

    def update_model(self, model: str) -> None:
        self._model = model

    def update_reasoning(self, reasoning: Reasoning | None) -> None:
        self._reasoning = reasoning

    @property
    def history(self) -> Sequence[ConversationItem]:
        return tuple(self._history)

    def replace_history(self, history: Sequence[ConversationItem]) -> None:
        self._history = list(history)

    def add_item(self, item: ConversationItem) -> None:
        self._history.append(item)

    async def run(self, prompt: str) -> AsyncIterator[AgentEvent]:
        yield AgentStartEvent()
        self._history.append(UserMessage(content=prompt))
        async for event in self._run():
            yield event
        yield AgentEndEvent(items=self._history)

    async def _run(self) -> AsyncIterator[AgentEvent]:
        while True:
            has_tool_results = False
            stream = await self._stream_fn(
                self._history,
                self._model,
                instructions=self._system_prompt,
                reasoning=self._reasoning,
                tools=self._tools,
            )

            async for event in stream:
                async for agent_event in self._handle_stream_event(event):
                    if (
                        isinstance(agent_event, TurnEndEvent)
                        and agent_event.tool_results
                    ):
                        has_tool_results = True

                    yield agent_event

            if not has_tool_results:
                break

    async def _handle_stream_event(
        self, event: StreamEvent
    ) -> AsyncIterator[AgentEvent]:
        if handler := self._select_event_handler(event):
            async for agent_event in handler:
                yield agent_event

    def _select_event_handler(
        self,
        event: StreamEvent,
    ) -> AsyncIterator[AgentEvent] | None:
        match event:
            case StreamStartEvent():
                return self._handle_stream_start_event(event)
            case StreamDoneEvent():
                return self._handle_stream_done_event(event)
            case StreamErrorEvent():
                return self._handle_stream_error_event(event)
            case _ if isinstance(event, IGNORED_STREAM_EVENT_TYPES):
                return None

        return None

    async def _handle_stream_start_event(
        self,
        event: StreamStartEvent,
    ) -> AsyncIterator[AgentEvent]:
        yield TurnStartEvent()
        yield MessageStartEvent(message=event.partial)

    async def _handle_stream_done_event(
        self,
        event: StreamDoneEvent,
    ) -> AsyncIterator[AgentEvent]:
        message = _build_assistant_turn(event.message)
        self._history.append(message)
        tool_results: list[ToolResultTurn] = []

        for tool_call in _collect_tool_calls(event.message):
            yield ToolExecutionStartEvent(
                call_id=tool_call.call_id,
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
            )
            async for agent_event in self._execute_tool(
                call_id=tool_call.call_id,
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
            ):
                if isinstance(agent_event, ToolExecutionEndEvent):
                    tool_result = _build_tool_result_turn(agent_event)
                    self._history.append(tool_result)
                    tool_results.append(tool_result)
                yield agent_event

        yield TurnEndEvent(message=message, tool_results=tool_results)

    async def _handle_stream_error_event(
        self,
        event: StreamErrorEvent,
    ) -> AsyncIterator[AgentEvent]:
        message = _build_assistant_turn(event.error)
        self._history.append(message)
        yield TurnEndEvent(message=message, tool_results=[])

    async def _execute_tool(
        self,
        call_id: str,
        tool_name: str,
        arguments: JsonObject,
    ) -> AsyncIterator[AgentEvent]:
        result: JsonValue = None
        try:
            tool = await self._get_tool(tool_name)
            if tool is None:
                result = {"error": f"Tool '{tool_name}' not found"}
            else:
                result = await tool(**arguments)
        except Exception as e:
            result = {"error": str(e)}

        yield ToolExecutionEndEvent(
            call_id=call_id,
            tool_name=tool_name,
            result=result,
            is_error=isinstance(result, dict) and "error" in result,
        )

    async def _get_tool(
        self,
        tool_name: str,
    ) -> Callable[..., Awaitable[JsonValue]] | None:
        # In a real implementation, this method would look up the tool by name and return a callable that executes it.
        # Here, we just return a dummy callable that simulates tool execution.
        async def dummy_tool(**kwargs: JsonValue) -> JsonObject:
            return {"result": f"Executed {tool_name} with arguments {kwargs}"}

        return dummy_tool


def _build_assistant_turn(message: AssistantMessage) -> AssistantTurn:
    status: Literal["completed", "aborted", "error"] = "completed"
    if message.stop_reason == "aborted":
        status = "aborted"
    elif message.stop_reason == "error":
        status = "error"

    return AssistantTurn(
        content=[block.model_copy(deep=True) for block in message.content],
        response_id=message.response_id,
        stop_reason=message.stop_reason,
        status=status,
        error_message=message.error_message,
    )


def _collect_tool_calls(message: AssistantMessage) -> list[ToolCallBlock]:
    return [
        block.model_copy(deep=True)
        for block in message.content
        if isinstance(block, ToolCallBlock)
    ]


def _build_tool_result_turn(event: ToolExecutionEndEvent) -> ToolResultTurn:
    return ToolResultTurn(
        call_id=event.call_id,
        tool_name=event.tool_name,
        content=json.dumps(event.result),
        is_error=event.is_error,
    )
