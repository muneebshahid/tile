"""Agent run events and provider stream callable contracts."""

from collections.abc import Awaitable, Sequence
from typing import Literal, Protocol, TypeAlias

from pydantic import BaseModel

from ai.types.contracts import AsyncEventStream, Reasoning
from ai.types.conversation import AssistantTurn, ConversationItem, ToolResultTurn
from ai.types.stream_events import StreamUpdateEvent
from ai.types.tools import JsonObject, ToolDefinition, ToolResult


class StreamFn(Protocol):
    """Callable that starts a provider stream from model-visible history."""

    def __call__(
        self,
        history: Sequence[ConversationItem],
        model: str,
        *,
        instructions: str,
        reasoning: Reasoning | None,
        tools: Sequence[ToolDefinition] | None,
    ) -> Awaitable[AsyncEventStream]: ...


class AgentEvent(BaseModel):
    """Base event emitted by the stateless agent runner."""

    type: str


class AgentStartEvent(AgentEvent):
    """Marks the start of an agent run."""

    type: Literal["agent_start"] = "agent_start"


class AgentEndEvent(AgentEvent):
    """Marks the end of an agent run and returns new conversation items."""

    type: Literal["agent_end"] = "agent_end"
    new_items: list[ConversationItem]


class TurnStartEvent(AgentEvent):
    """Marks the start of a single assistant turn."""

    type: Literal["turn_start"] = "turn_start"


class TurnEndEvent(AgentEvent):
    """Marks the end of a single assistant turn."""

    type: Literal["turn_end"] = "turn_end"
    message: AssistantTurn
    tool_results: list[ToolResultTurn]


class MessageStartEvent(AgentEvent):
    """Marks the start of a message lifecycle event."""

    type: Literal["message_start"] = "message_start"
    response_id: str | None = None


class MessageUpdateEvent(AgentEvent):
    """Carries assistant streaming updates during a message."""

    type: Literal["message_update"] = "message_update"
    stream_event: StreamUpdateEvent


class MessageEndEvent(AgentEvent):
    """Marks the end of a message lifecycle event."""

    type: Literal["message_end"] = "message_end"
    message: AssistantTurn


class ToolExecutionStartEvent(AgentEvent):
    """Marks the start of a tool execution."""

    type: Literal["tool_execution_start"] = "tool_execution_start"
    call_id: str
    tool_name: str
    arguments: JsonObject


class ToolExecutionEndEvent(AgentEvent):
    """Marks the end of a tool execution."""

    type: Literal["tool_execution_end"] = "tool_execution_end"
    call_id: str
    tool_name: str
    result: ToolResult
    is_error: bool


AgentRunEvent: TypeAlias = (
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionEndEvent
)

AgentEventType: TypeAlias = AgentRunEvent
