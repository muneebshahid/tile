"""Lifecycle pairing ledger for the run's authoritative event log."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from tile.events import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    LifecycleAborted,
    LifecycleFailed,
    LifecycleTermination,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ResultFollowUpEvent,
    RunEndEvent,
    RunStartEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from tile.result import Aborted, ExecutionFailure, Failed, RunOutcome

_ScopeKind: TypeAlias = Literal["run", "agent", "turn", "message", "tool"]


class LifecycleProtocolError(RuntimeError):
    """Raised when a published event violates the lifecycle pairing contract."""


@dataclass(frozen=True)
class _OpenScope:
    """One start event awaiting its matching end."""

    kind: _ScopeKind
    attempt: int = 0
    call_id: str = ""


class LifecycleLedger:
    """Validates event nesting and closes scopes a producer left open.

    The ledger observes every event before it enters the run log, holding
    the open-scope stack. Producers close scopes in-band; ``close`` is the
    finalization fallback that synthesizes the missing ends, innermost
    first, when execution failure, cancellation, or a producer bug tore
    scopes down before their ends were published.
    """

    def __init__(self) -> None:
        """Start with no open scopes and no committed run end."""

        self._stack: list[_OpenScope] = []
        self._run_started = False
        self._committed_outcome: RunOutcome | None = None

    @property
    def committed_outcome(self) -> RunOutcome | None:
        """Return the outcome committed by a published run end event."""

        return self._committed_outcome

    def observe(self, event: AgentEvent) -> None:
        """Validate one event against the pairing contract before publication."""

        if self._committed_outcome is not None:
            raise LifecycleProtocolError(
                f"Event published after the run end was committed: {event.type}."
            )
        match event:
            case RunStartEvent():
                self._open_run()
            case AgentStartEvent():
                self._push("agent", event, attempt=event.attempt)
            case TurnStartEvent():
                self._push("turn", event)
            case MessageStartEvent():
                self._push("message", event)
            case ToolExecutionStartEvent():
                self._push("tool", event, call_id=event.call_id)
            case MessageUpdateEvent():
                self._require_top("message", event)
            case ResultFollowUpEvent():
                self._require_top("run", event)
            case MessageEndEvent():
                self._pop("message", event)
            case ToolExecutionEndEvent():
                scope = self._pop("tool", event)
                if scope.call_id != event.call_id:
                    raise LifecycleProtocolError(
                        f"Tool execution end for call {event.call_id!r} does not "
                        f"match the open call {scope.call_id!r}."
                    )
            case TurnEndEvent():
                self._pop("turn", event)
            case AgentEndEvent():
                scope = self._pop("agent", event)
                if scope.attempt != event.attempt:
                    raise LifecycleProtocolError(
                        f"Agent end for attempt {event.attempt} does not match "
                        f"the open attempt {scope.attempt}."
                    )
            case RunEndEvent():
                self._commit_run_end(event)
            case _:
                raise LifecycleProtocolError(
                    f"Event outside the lifecycle contract: {event.type}."
                )

    def close(self, outcome: RunOutcome) -> tuple[AgentEvent, ...]:
        """Return end events closing every open scope, innermost first.

        The returned sequence ends with a run end committing ``outcome``
        unless a producer already committed one. This transition must not
        raise; it runs during finalization, where a terminal state is the
        only channel that cannot be lost.
        """

        termination = _termination_for(outcome)
        closing_events: list[AgentEvent] = []
        while self._stack:
            scope = self._stack.pop()
            if scope.kind == "run":
                continue
            closing_events.append(_closing_event(scope, termination))
        if self._committed_outcome is None:
            self._committed_outcome = outcome
            closing_events.append(RunEndEvent(outcome=outcome))
        return tuple(closing_events)

    def _open_run(self) -> None:
        """Open the run scope exactly once, before any other scope."""

        if self._run_started:
            raise LifecycleProtocolError("Run start published twice.")
        if self._stack:
            raise LifecycleProtocolError("Run start inside an open scope.")
        self._run_started = True
        self._stack.append(_OpenScope(kind="run"))

    def _commit_run_end(self, event: RunEndEvent) -> None:
        """Commit the run end once every inner scope has closed."""

        if [scope.kind for scope in self._stack] != ["run"]:
            raise LifecycleProtocolError(
                "Run end published while inner scopes are still open."
            )
        self._stack.pop()
        self._committed_outcome = event.outcome

    def _push(
        self,
        kind: _ScopeKind,
        event: AgentEvent,
        *,
        attempt: int = 0,
        call_id: str = "",
    ) -> None:
        """Open one nested scope under its required parent."""

        self._require_top(_PARENT_KIND[kind], event)
        self._stack.append(_OpenScope(kind=kind, attempt=attempt, call_id=call_id))

    def _pop(self, kind: _ScopeKind, event: AgentEvent) -> _OpenScope:
        """Close the innermost scope, which must be of the expected kind."""

        self._require_top(kind, event)
        return self._stack.pop()

    def _require_top(self, kind: _ScopeKind, event: AgentEvent) -> None:
        """Reject an event whose innermost open scope is not ``kind``."""

        if not self._stack or self._stack[-1].kind != kind:
            open_kind = self._stack[-1].kind if self._stack else "none"
            raise LifecycleProtocolError(
                f"Event {event.type} requires an open {kind} scope; "
                f"the innermost open scope is {open_kind}."
            )


_PARENT_KIND: dict[_ScopeKind, _ScopeKind] = {
    "agent": "run",
    "turn": "agent",
    "message": "turn",
    "tool": "turn",
}


def _termination_for(outcome: RunOutcome) -> LifecycleTermination:
    """Map a terminal run outcome onto the closure for scopes it tore down."""

    if isinstance(outcome, Aborted):
        return LifecycleAborted()
    if isinstance(outcome, Failed) and isinstance(outcome.cause, ExecutionFailure):
        return LifecycleFailed(cause=outcome.cause)
    return LifecycleFailed(
        cause=ExecutionFailure(
            origin="execution",
            exception_type="LifecycleProtocolError",
            message="The run concluded while lifecycle scopes were still open.",
        )
    )


def _closing_event(scope: _OpenScope, termination: LifecycleTermination) -> AgentEvent:
    """Synthesize the end event for one open scope."""

    match scope.kind:
        case "agent":
            return AgentEndEvent(attempt=scope.attempt, termination=termination)
        case "turn":
            return TurnEndEvent(termination=termination)
        case "message":
            return MessageEndEvent(termination=termination)
        case "tool":
            return ToolExecutionEndEvent(
                call_id=scope.call_id,
                termination=termination,
            )
        case _:
            raise AssertionError(f"Unclosable scope kind: {scope.kind}")
