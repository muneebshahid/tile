"""Open-scope tracking for the run's lifecycle event log."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from tile.events import (
    AgentEndEvent,
    AgentEvent,
    AgentInterruptedEvent,
    AgentStartEvent,
    MessageEndEvent,
    MessageInterruptedEvent,
    MessageStartEvent,
    RunEndEvent,
    ToolExecutionEndEvent,
    ToolExecutionInterruptedEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnInterruptedEvent,
    TurnStartEvent,
)
from tile.result import RunOutcome

_ScopeKind: TypeAlias = Literal["agent", "turn", "message", "tool"]


@dataclass(frozen=True)
class _OpenScope:
    """One start event awaiting its end or interruption."""

    kind: _ScopeKind
    attempt: int = 0
    call_id: str = ""


class OpenScopeTracker:
    """Tracks open lifecycle scopes so finalization can close what remains.

    Starts push a scope, ends pop their innermost match, and a run end
    records the committed outcome. The tracker validates nothing: producers
    are runtime-internal and pinned by tests, so an unmatched end is
    ignored, and any scope still open when the run concludes is closed by
    ``close``. Tool scopes pop by call id, so tracking stays correct if
    tool executions ever interleave.
    """

    def __init__(self) -> None:
        """Start with no open scopes and no committed run end."""

        self._stack: list[_OpenScope] = []
        self._committed_outcome: RunOutcome | None = None

    @property
    def committed_outcome(self) -> RunOutcome | None:
        """Return the outcome committed by a published run end event."""

        return self._committed_outcome

    def observe(self, event: AgentEvent) -> None:
        """Track one published event's effect on the open scopes."""

        match event:
            case AgentStartEvent():
                self._stack.append(_OpenScope(kind="agent", attempt=event.attempt))
            case TurnStartEvent():
                self._stack.append(_OpenScope(kind="turn"))
            case MessageStartEvent():
                self._stack.append(_OpenScope(kind="message"))
            case ToolExecutionStartEvent():
                self._stack.append(_OpenScope(kind="tool", call_id=event.call_id))
            case AgentEndEvent():
                self._pop("agent")
            case TurnEndEvent():
                self._pop("turn")
            case MessageEndEvent():
                self._pop("message")
            case ToolExecutionEndEvent():
                self._pop("tool", call_id=event.outcome.tool_result_turn.call_id)
            case RunEndEvent():
                self._committed_outcome = event.outcome
            case _:
                pass

    def close(self, outcome: RunOutcome) -> tuple[AgentEvent, ...]:
        """Return events closing every open scope, innermost first.

        The returned sequence ends with a run end committing ``outcome``
        unless a producer already committed one. This transition must not
        raise; it runs during finalization, where a terminal state is the
        only channel that cannot be lost.
        """

        closing_events = [_interruption(scope) for scope in reversed(self._stack)]
        self._stack.clear()
        if self._committed_outcome is None:
            self._committed_outcome = outcome
            closing_events.append(RunEndEvent(outcome=outcome))
        return tuple(closing_events)

    def _pop(self, kind: _ScopeKind, *, call_id: str | None = None) -> None:
        """Remove the innermost open scope matching one end event, if any."""

        for index in range(len(self._stack) - 1, -1, -1):
            scope = self._stack[index]
            if scope.kind != kind:
                continue
            if call_id is not None and scope.call_id != call_id:
                continue
            del self._stack[index]
            return


def _interruption(scope: _OpenScope) -> AgentEvent:
    """Return the interrupted event closing one open scope."""

    match scope.kind:
        case "agent":
            return AgentInterruptedEvent(attempt=scope.attempt)
        case "turn":
            return TurnInterruptedEvent()
        case "message":
            return MessageInterruptedEvent()
        case "tool":
            return ToolExecutionInterruptedEvent(call_id=scope.call_id)
