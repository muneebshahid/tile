"""Agent run events and provider stream callable contracts."""

from collections.abc import Awaitable, Sequence
from typing import Literal, Protocol, TypeAlias

from pydantic import BaseModel, SerializeAsAny

from tile.types.contracts import AsyncEventStream
from tile.types.conversation import AssistantTurn, ConversationItem, ToolResultTurn
from tile.types.stream_events import StreamUpdateEvent
from tile.types.tools import (
    JsonObject,
    ToolDefinition,
    ToolDetails,
    ToolResult,
)


class StreamFn(Protocol):
    """Callable that starts a provider stream from model-visible history."""

    def __call__(
        self,
        history: Sequence[ConversationItem],
        model: str,
        *,
        instructions: str,
        tools: Sequence[ToolDefinition] | None,
    ) -> Awaitable[AsyncEventStream]: ...


class AgentEvent(BaseModel):
    """Base event emitted by the stateless agent runner."""

    type: str


class AgentStartEvent(AgentEvent):
    """Marks the start of an agent run."""

    type: Literal["agent_start"] = "agent_start"


class AgentEndEvent(AgentEvent):
    """Marks the end of an agent run."""

    type: Literal["agent_end"] = "agent_end"


class TurnStartEvent(AgentEvent):
    """Marks the start of a single assistant turn."""

    type: Literal["turn_start"] = "turn_start"


class ToolExecutionOutcome(BaseModel):
    """Full runtime outcome for a tool execution."""

    tool_result_turn: ToolResultTurn
    details: SerializeAsAny[ToolDetails] | None = None

    @property
    def result(self) -> ToolResult:
        """Return the full tool result including non-replay metadata."""

        return ToolResult(
            content=self.tool_result_turn.content,
            details=self.details,
        )

    @classmethod
    def from_result(
        cls,
        *,
        call_id: str,
        tool_name: str,
        result: ToolResult,
        is_error: bool,
    ) -> "ToolExecutionOutcome":
        """Build an execution outcome from a raw tool result."""

        return cls(
            tool_result_turn=ToolResultTurn(
                call_id=call_id,
                tool_name=tool_name,
                content=result.content,
                is_error=is_error,
            ),
            details=result.details,
        )


class TurnEndEvent(AgentEvent):
    """Marks the end of a single assistant turn."""

    type: Literal["turn_end"] = "turn_end"
    assistant_turn: AssistantTurn
    tool_executions: list[ToolExecutionOutcome]

    @property
    def tool_result_turns(self) -> list[ToolResultTurn]:
        """Return replayable tool result projections for this turn."""

        return [execution.tool_result_turn for execution in self.tool_executions]


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
    assistant_turn: AssistantTurn


class ToolExecutionStartEvent(AgentEvent):
    """Marks the start of a tool execution."""

    type: Literal["tool_execution_start"] = "tool_execution_start"
    call_id: str
    tool_name: str
    arguments: JsonObject


class ToolExecutionEndEvent(AgentEvent):
    """Marks the end of a tool execution."""

    type: Literal["tool_execution_end"] = "tool_execution_end"
    outcome: ToolExecutionOutcome


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
