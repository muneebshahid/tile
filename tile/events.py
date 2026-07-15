"""Agent run events and provider stream callable contracts."""

from collections.abc import Awaitable, Sequence
from typing import Literal, Protocol, TypeAlias

from pydantic import BaseModel

from tile.result import RunOutcome
from tile.types.contracts import AsyncEventStream
from tile.types.conversation import (
    AssistantTurn,
    ConversationItem,
    ToolResultTurn,
    UserMessage,
)
from tile.types.stream_events import StreamUpdateEvent
from tile.types.tool_execution import ToolExecutionOutcome
from tile.types.tools import (
    JsonObject,
    ToolDefinition,
)


class StreamFn(Protocol):
    """Callable that starts a provider stream from model-visible history.

    ``provider`` names the provider identity this callable streams through,
    declared once where the callable is constructed so run records can carry
    provider identity before the first message finalizes.
    """

    provider: str

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
    """Marks the end of an agent run.

    The stateless agent leaves ``outcome`` unset. Layers that compose runs
    into prompts attach the prompt-level outcome.
    """

    type: Literal["agent_end"] = "agent_end"
    outcome: RunOutcome | None = None


class ResultFollowUpEvent(AgentEvent):
    """Marks an injected reminder that the run must end with a result call."""

    type: Literal["result_follow_up"] = "result_follow_up"
    message: UserMessage


class TurnStartEvent(AgentEvent):
    """Marks the start of a single assistant turn."""

    type: Literal["turn_start"] = "turn_start"


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
    | ResultFollowUpEvent
)
