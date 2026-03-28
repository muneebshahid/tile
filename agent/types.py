from collections.abc import Awaitable, Sequence
from typing import Literal, Protocol, TypeAlias

from pydantic import BaseModel, JsonValue

from ai.types.contracts import AsyncEventStream, Reasoning
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
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from ai.types.tools import JsonObject, ToolDefinition

AgentMessage: TypeAlias = (
    UserMessage | AssistantMessage | AssistantTurn | ToolResultTurn
)
AssistantMessageUpdateEvent: TypeAlias = (
    ReasoningStartEvent
    | ReasoningDeltaEvent
    | ReasoningEndEvent
    | TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
)


class StreamFn(Protocol):
    def __call__(
        self,
        history: Sequence[ConversationItem],
        model: str,
        *,
        instructions: str,
        reasoning: Reasoning | None,
        tools: Sequence[ToolDefinition] | None,
    ) -> Awaitable[AsyncEventStream]: ...


class AgentStartEvent(BaseModel):
    """Marks the start of an agent run."""

    type: Literal["agent_start"] = "agent_start"


class AgentEndEvent(BaseModel):
    """Marks the end of an agent run and returns new conversation items."""

    type: Literal["agent_end"] = "agent_end"
    items: list[ConversationItem]


class TurnStartEvent(BaseModel):
    """Marks the start of a single assistant turn."""

    type: Literal["turn_start"] = "turn_start"


class TurnEndEvent(BaseModel):
    """Marks the end of a single assistant turn."""

    type: Literal["turn_end"] = "turn_end"
    message: AssistantTurn
    tool_results: list[ToolResultTurn]


class MessageStartEvent(BaseModel):
    """Marks the start of a message lifecycle event."""

    type: Literal["message_start"] = "message_start"
    message: AgentMessage


class MessageUpdateEvent(BaseModel):
    """Carries assistant streaming updates during a message."""

    type: Literal["message_update"] = "message_update"
    message: AssistantMessage
    stream_event: AssistantMessageUpdateEvent


class MessageEndEvent(BaseModel):
    """Marks the end of a message lifecycle event."""

    type: Literal["message_end"] = "message_end"
    message: AgentMessage


class ToolExecutionStartEvent(BaseModel):
    """Marks the start of a tool execution."""

    type: Literal["tool_execution_start"] = "tool_execution_start"
    call_id: str
    tool_name: str
    arguments: JsonObject


class ToolExecutionEndEvent(BaseModel):
    """Marks the end of a tool execution."""

    type: Literal["tool_execution_end"] = "tool_execution_end"
    call_id: str
    tool_name: str
    result: JsonValue
    is_error: bool


AgentEvent: TypeAlias = (
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
