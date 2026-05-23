import json
from collections.abc import AsyncIterator, Sequence
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
    ToolCallStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
)
from ai.types.tools import JsonObject, ToolDefinition, ToolFunction
from agent.prompt import PROMPT
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

    def add_user_message(self, text: str) -> UserMessage:
        message = UserMessage(content=text)
        self._history.append(message)
        return message

    def replace_history(self, history: Sequence[ConversationItem]) -> None:
        self._history = list(history)

    def add_item(self, item: ConversationItem) -> None:
        self._history.append(item)

    async def run(self) -> AsyncIterator[AgentEvent]:
        yield AgentStartEvent()
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
            case _ if isinstance(event, ASSISTANT_MESSAGE_UPDATE_EVENT_TYPES):
                return self._handle_message_update_event(event)

        return None

    async def _handle_stream_start_event(
        self,
        event: StreamStartEvent,
    ) -> AsyncIterator[AgentEvent]:
        yield TurnStartEvent()
        yield MessageStartEvent(message=event.message)

    async def _handle_stream_done_event(
        self,
        event: StreamDoneEvent,
    ) -> AsyncIterator[AgentEvent]:
        message = _build_assistant_turn(event.message)
        self._history.append(message)
        yield MessageEndEvent(message=message)
        tool_results: list[ToolResultTurn] = []

        for tool_call in _collect_tool_calls(event.message):
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
        yield MessageEndEvent(message=message)
        yield TurnEndEvent(message=message, tool_results=[])

    async def _handle_message_update_event(
        self,
        event: (
            ReasoningStartEvent
            | ReasoningDeltaEvent
            | ReasoningEndEvent
            | TextStartEvent
            | TextDeltaEvent
            | TextEndEvent
            | ToolCallStartEvent
            | ToolCallDeltaEvent
            | ToolCallEndEvent
        ),
    ) -> AsyncIterator[AgentEvent]:
        yield MessageUpdateEvent(message=event.message, stream_event=event)

    async def _execute_tool(
        self,
        call_id: str,
        tool_name: str,
        arguments: JsonObject,
    ) -> AsyncIterator[AgentEvent]:
        """Emit the full lifecycle for one tool execution."""

        yield ToolExecutionStartEvent(
            call_id=call_id,
            tool_name=tool_name,
            arguments=arguments,
        )

        result = await self._call_tool(tool_name, arguments)
        yield ToolExecutionEndEvent(
            call_id=call_id,
            tool_name=tool_name,
            result=result,
            is_error=isinstance(result, dict) and "error" in result,
        )

    async def _call_tool(
        self,
        tool_name: str,
        arguments: JsonObject,
    ) -> JsonValue:
        """Resolve and call a tool while normalizing tool failures."""

        try:
            tool = await self._get_tool(tool_name)
            if tool is None:
                return {"error": f"Tool '{tool_name}' not found"}
            return await tool(**arguments)
        except Exception as error:
            return {"error": str(error)}

    async def _get_tool(
        self,
        tool_name: str,
    ) -> ToolFunction | None:
        """Find a registered tool implementation by name."""
        tool_name = tool_name.lower().strip()

        for tool in self._tools:
            if tool.name == tool_name:
                return tool.fn

        return None


def _build_assistant_turn(message: AssistantMessage) -> AssistantTurn:
    status: Literal["completed", "aborted", "error"] = "completed"
    if message.stop_reason == "aborted":
        status = "aborted"
    elif message.stop_reason == "error":
        status = "error"

    return AssistantTurn(
        blocks=[block.model_copy(deep=True) for block in message.blocks],
        response_id=message.response_id,
        stop_reason=message.stop_reason,
        status=status,
        error_message=message.error_message,
    )


def _collect_tool_calls(message: AssistantMessage) -> list[ToolCallBlock]:
    return [
        block.model_copy(deep=True)
        for block in message.blocks
        if isinstance(block, ToolCallBlock)
    ]


def _build_tool_result_turn(event: ToolExecutionEndEvent) -> ToolResultTurn:
    return ToolResultTurn(
        call_id=event.call_id,
        tool_name=event.tool_name,
        content=json.dumps(event.result),
        is_error=event.is_error,
    )
