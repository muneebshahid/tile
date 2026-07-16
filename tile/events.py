"""Agent run events and provider stream callable contracts.

Lifecycle pairing contract: every published start event is followed by
exactly one end event or interrupted event, for every in-process
termination path. An end event is the producer finishing its scope and
carries the scope's payload; an interrupted event is the runtime closing a
scope that a failure or abort tore down, and carries no cause of its own —
the run's ``RunEndEvent`` outcome names, exactly once, why anything was
interrupted. Scopes nest strictly:
``run ⊃ agent attempt ⊃ turn ⊃ message / tool executions``, interruptions
close innermost-first, and ``RunEndEvent`` is the final event of every
run. Provider stream fragments (text, reasoning, and tool-call
start/delta/end updates carried by ``MessageUpdateEvent``) are message
content, not lifecycle scopes: their outstanding state is terminated by
the containing message's end or interruption. Hard process death is
outside this contract — no process can publish an event after it has
stopped.
"""

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


class RunStartEvent(AgentEvent):
    """Marks the start of one prompt run.

    Published by the run itself before its event source starts, so every
    run log begins with a run start on every path.
    """

    type: Literal["run_start"] = "run_start"


class RunEndEvent(AgentEvent):
    """Marks the end of one prompt run and commits its terminal outcome.

    Exactly one run end closes every run, as its final event. The outcome
    is the same discriminated value recorded on the durable run summary;
    its variant implies how execution terminated and names the cause of
    any interruptions in the log.
    """

    type: Literal["run_end"] = "run_end"
    outcome: RunOutcome


class AgentStartEvent(AgentEvent):
    """Marks the start of one stateless agent attempt."""

    type: Literal["agent_start"] = "agent_start"
    attempt: int = 0


class AgentEndEvent(AgentEvent):
    """Marks the end of one stateless agent attempt.

    An attempt whose turn errored in-band still ends normally; the
    run-level verdict lives on ``RunEndEvent``.
    """

    type: Literal["agent_end"] = "agent_end"
    attempt: int = 0


class AgentInterruptedEvent(AgentEvent):
    """Closes an agent attempt torn down by a failure or abort."""

    type: Literal["agent_interrupted"] = "agent_interrupted"
    attempt: int = 0


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


class TurnInterruptedEvent(AgentEvent):
    """Closes a turn torn down by a failure or abort."""

    type: Literal["turn_interrupted"] = "turn_interrupted"


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


class MessageInterruptedEvent(AgentEvent):
    """Closes a message torn down before the provider stream finalized it."""

    type: Literal["message_interrupted"] = "message_interrupted"


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


class ToolExecutionInterruptedEvent(AgentEvent):
    """Closes a tool execution torn down before the tool produced a result."""

    type: Literal["tool_execution_interrupted"] = "tool_execution_interrupted"
    call_id: str


AgentRunEvent: TypeAlias = (
    RunStartEvent
    | RunEndEvent
    | AgentStartEvent
    | AgentEndEvent
    | AgentInterruptedEvent
    | TurnStartEvent
    | TurnEndEvent
    | TurnInterruptedEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | MessageInterruptedEvent
    | ToolExecutionStartEvent
    | ToolExecutionEndEvent
    | ToolExecutionInterruptedEvent
    | ResultFollowUpEvent
)
