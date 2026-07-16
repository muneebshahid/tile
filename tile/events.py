"""Agent run events and provider stream callable contracts.

Lifecycle pairing contract: every published start event has exactly one
matching end event, for every in-process termination path. End events carry
a ``LifecycleTermination`` describing how their scope closed; a scope torn
down by an exception or cancellation is closed with a synthesized end whose
payload fields are ``None``. Scopes nest strictly:
``run ⊃ agent attempt ⊃ turn ⊃ message / tool executions``, and end events
close innermost-first, so ``RunEndEvent`` is always the final event of a
run. Provider stream fragments (text, reasoning, and tool-call start/delta/
end updates carried by ``MessageUpdateEvent``) are message content, not
lifecycle scopes: their outstanding state is terminated by the containing
``MessageEndEvent``. Hard process death is outside this contract — no
process can publish an event after it has stopped.
"""

from collections.abc import Awaitable, Sequence
from typing import Literal, Protocol, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, model_validator

from tile.result import ExecutionFailure, RunOutcome
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


class LifecycleCompleted(BaseModel):
    """A lifecycle scope closed normally by its own producer."""

    model_config = ConfigDict(frozen=True)

    type: Literal["lifecycle_completed"] = "lifecycle_completed"


class LifecycleFailed(BaseModel):
    """A lifecycle scope was torn down by an execution failure."""

    model_config = ConfigDict(frozen=True)

    type: Literal["lifecycle_failed"] = "lifecycle_failed"
    cause: ExecutionFailure


class LifecycleAborted(BaseModel):
    """A lifecycle scope was torn down by cancellation."""

    model_config = ConfigDict(frozen=True)

    type: Literal["lifecycle_aborted"] = "lifecycle_aborted"


LifecycleTermination: TypeAlias = (
    LifecycleCompleted | LifecycleFailed | LifecycleAborted
)


class AgentEvent(BaseModel):
    """Base event emitted by the stateless agent runner."""

    type: str


class RunStartEvent(AgentEvent):
    """Marks the start of one prompt run."""

    type: Literal["run_start"] = "run_start"


class RunEndEvent(AgentEvent):
    """Marks the end of one prompt run and commits its terminal outcome.

    Exactly one run end closes every run, as its final event. The outcome
    is the same discriminated value recorded on the durable run summary;
    its variant implies how execution terminated, so there is no separate
    termination field to deviate from it.
    """

    type: Literal["run_end"] = "run_end"
    outcome: RunOutcome


class AgentStartEvent(AgentEvent):
    """Marks the start of one stateless agent attempt."""

    type: Literal["agent_start"] = "agent_start"
    attempt: int = 0


class AgentEndEvent(AgentEvent):
    """Marks the end of one stateless agent attempt.

    ``termination`` describes how the attempt scope closed, not whether the
    task succeeded: an attempt whose turn errored in-band still closes
    ``completed``, and the run-level verdict lives on ``RunEndEvent``.
    """

    type: Literal["agent_end"] = "agent_end"
    attempt: int = 0
    termination: LifecycleTermination = LifecycleCompleted()


class ResultFollowUpEvent(AgentEvent):
    """Marks an injected reminder that the run must end with a result call."""

    type: Literal["result_follow_up"] = "result_follow_up"
    message: UserMessage


class TurnStartEvent(AgentEvent):
    """Marks the start of a single assistant turn."""

    type: Literal["turn_start"] = "turn_start"


class TurnEndEvent(AgentEvent):
    """Marks the end of a single assistant turn.

    ``assistant_turn`` is None only on a synthesized closure, when the turn
    was torn down before its message finalized.
    """

    type: Literal["turn_end"] = "turn_end"
    assistant_turn: AssistantTurn | None = None
    tool_executions: list[ToolExecutionOutcome] = []
    termination: LifecycleTermination = LifecycleCompleted()

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
    """Marks the end of a message lifecycle event.

    ``assistant_turn`` is None only on a synthesized closure, when the
    message was torn down before the provider stream finalized it.
    """

    type: Literal["message_end"] = "message_end"
    assistant_turn: AssistantTurn | None = None
    termination: LifecycleTermination = LifecycleCompleted()


class ToolExecutionStartEvent(AgentEvent):
    """Marks the start of a tool execution."""

    type: Literal["tool_execution_start"] = "tool_execution_start"
    call_id: str
    tool_name: str
    arguments: JsonObject


class ToolExecutionEndEvent(AgentEvent):
    """Marks the end of a tool execution.

    ``outcome`` is None only on a synthesized closure, when the execution
    was torn down before the tool produced a result.
    """

    type: Literal["tool_execution_end"] = "tool_execution_end"
    call_id: str
    outcome: ToolExecutionOutcome | None = None
    termination: LifecycleTermination = LifecycleCompleted()

    @model_validator(mode="after")
    def _validate_call_identity(self) -> Self:
        """Reject an outcome recorded against a different tool call."""

        if (
            self.outcome is not None
            and self.outcome.tool_result_turn.call_id != self.call_id
        ):
            raise ValueError("Tool execution outcome belongs to a different call.")
        return self


AgentRunEvent: TypeAlias = (
    RunStartEvent
    | RunEndEvent
    | AgentStartEvent
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
