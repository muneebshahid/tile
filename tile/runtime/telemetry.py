"""Private lifecycle tracking and pure folding for finalized run telemetry."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from tile.events import (
    AgentEvent,
    LifecycleEventMetadata,
    MessageEndEvent,
    MessageStartEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from tile.result import AgentFailure, ExecutionFailure, Failed, RunOutcome
from tile.runs import RunRecord, TerminalRunStatus
from tile.telemetry.models import (
    LifecycleScopeRecord,
    LifecycleScopeStatus,
    LifecycleScopeType,
    RunTelemetryError,
    RunTelemetryRecord,
    ToolAggregate,
)
from tile.telemetry.sink import CapturedRunException
from tile.types.usage import TokenUsage

MonotonicClock = Callable[[], int]
ScopeIdFactory = Callable[[], str]

_START_SCOPE_TYPES: dict[str, LifecycleScopeType] = {
    "run_start": "run",
    "agent_start": "agent",
    "turn_start": "turn",
    "message_start": "message",
    "tool_execution_start": "tool_execution",
}
_END_SCOPE_TYPES: dict[
    str,
    tuple[LifecycleScopeType, LifecycleScopeStatus],
] = {
    "run_end": ("run", "completed"),
    "agent_end": ("agent", "completed"),
    "agent_interrupted": ("agent", "interrupted"),
    "turn_end": ("turn", "completed"),
    "turn_interrupted": ("turn", "interrupted"),
    "message_end": ("message", "completed"),
    "message_interrupted": ("message", "interrupted"),
    "tool_execution_end": ("tool_execution", "completed"),
    "tool_execution_interrupted": ("tool_execution", "interrupted"),
}
_UNSCOPED_EVENT_TYPES = frozenset(
    {
        "message_update",
        "result_follow_up",
    }
)
_PARENT_SCOPE_TYPES: dict[
    LifecycleScopeType,
    LifecycleScopeType | None,
] = {
    "run": None,
    "agent": "run",
    "turn": "agent",
    "message": "turn",
    "tool_execution": "turn",
}


@dataclass(frozen=True)
class _OpenLifecycleScope:
    """Identity retained from a lifecycle start until its matching end."""

    scope_id: str
    parent_scope_id: str | None
    scope_type: LifecycleScopeType


class _LifecycleScopeError(RuntimeError):
    """Raised when runtime events violate lifecycle scope invariants."""


def build_run_telemetry(
    events: Sequence[AgentEvent],
    record: RunRecord,
    *,
    exceptions: Sequence[CapturedRunException] = (),
    context_receipt: str | None = None,
) -> RunTelemetryRecord:
    """Build one canonical telemetry record from finalized run facts."""

    status, outcome = _require_terminal_record(record)
    folded = _fold_lifecycle_scopes(events)
    scopes = tuple(scope.to_record() for scope in folded)
    root = scopes[0]
    return RunTelemetryRecord(
        run_id=record.run_id,
        session_id=record.session_id,
        status=status,
        outcome=outcome,
        started_at=record.started_at,
        started_monotonic_ns=root.started_monotonic_ns,
        ended_monotonic_ns=root.ended_monotonic_ns,
        duration_ns=root.ended_monotonic_ns - root.started_monotonic_ns,
        provider=record.provider,
        model=record.model,
        turn_count=sum(1 for scope in scopes if scope.scope_type == "turn"),
        token_usage=root.token_usage,
        tools=_aggregate_tools(folded),
        scopes=scopes,
        errors=_build_errors(outcome, exceptions),
        context_receipt=context_receipt,
    )


class _LifecycleScopeTracker:
    """Stamp stable lifecycle scope identity at the run publication boundary."""

    def __init__(
        self,
        *,
        clock: MonotonicClock,
        scope_id_factory: ScopeIdFactory,
    ) -> None:
        """Create a tracker around deterministic time and identity providers."""

        self._clock = clock
        self._scope_id_factory = scope_id_factory
        self._open_scopes: list[_OpenLifecycleScope] = []

    def stamp(self, event: AgentEvent) -> AgentEvent:
        """Return an event carrying lifecycle metadata when applicable."""

        if event.lifecycle is not None:
            raise _LifecycleScopeError(
                f"Lifecycle event is already stamped: {event.type}"
            )
        if event.type in _UNSCOPED_EVENT_TYPES:
            return event

        scope = self._transition_scope(event.type)
        metadata = LifecycleEventMetadata(
            scope_id=scope.scope_id,
            parent_scope_id=scope.parent_scope_id,
            monotonic_ns=self._clock(),
        )
        return event.model_copy(update={"lifecycle": metadata})

    def _transition_scope(self, event_type: str) -> _OpenLifecycleScope:
        """Open or close the scope represented by one lifecycle event type."""

        if scope_type := _START_SCOPE_TYPES.get(event_type):
            return self._open_scope(scope_type)
        if end_rule := _END_SCOPE_TYPES.get(event_type):
            return self._close_scope(end_rule[0], event_type=event_type)
        raise _LifecycleScopeError(f"Unclassified runtime event type: {event_type}")

    def _open_scope(
        self,
        scope_type: LifecycleScopeType,
    ) -> _OpenLifecycleScope:
        """Create and retain one scope under its currently open parent."""

        if self._find_open_scope(scope_type) is not None:
            raise _LifecycleScopeError(f"Lifecycle scope is already open: {scope_type}")

        parent_type = _PARENT_SCOPE_TYPES[scope_type]
        parent = self._require_open_parent(scope_type, parent_type=parent_type)
        scope = _OpenLifecycleScope(
            scope_id=self._scope_id_factory(),
            parent_scope_id=parent.scope_id if parent is not None else None,
            scope_type=scope_type,
        )
        self._open_scopes.append(scope)
        return scope

    def _close_scope(
        self,
        scope_type: LifecycleScopeType,
        *,
        event_type: str,
    ) -> _OpenLifecycleScope:
        """Close a matching scope and sweep anything still open inside it."""

        for index in range(len(self._open_scopes) - 1, -1, -1):
            scope = self._open_scopes[index]
            if scope.scope_type == scope_type:
                del self._open_scopes[index:]
                return scope
        raise _LifecycleScopeError(f"Lifecycle end has no matching start: {event_type}")

    def _require_open_parent(
        self,
        scope_type: LifecycleScopeType,
        *,
        parent_type: LifecycleScopeType | None,
    ) -> _OpenLifecycleScope | None:
        """Return the immediate parent required by a new scope."""

        if parent_type is None:
            if self._open_scopes:
                raise _LifecycleScopeError(
                    "Run scope cannot start inside another lifecycle scope."
                )
            return None

        if not self._open_scopes:
            raise _missing_parent_error(scope_type, parent_type)
        parent = self._open_scopes[-1]
        if parent.scope_type != parent_type:
            raise _missing_parent_error(scope_type, parent_type)
        return parent

    def _find_open_scope(
        self,
        scope_type: LifecycleScopeType | None,
    ) -> _OpenLifecycleScope | None:
        """Return the innermost open scope of one type."""

        if scope_type is None:
            return None
        for scope in reversed(self._open_scopes):
            if scope.scope_type == scope_type:
                return scope
        return None


@dataclass
class _ScopeAccumulator:
    """Mutable state for one lifecycle scope while folding the event log."""

    scope_id: str
    parent_scope_id: str | None
    scope_type: LifecycleScopeType
    started_monotonic_ns: int
    operation_name: str | None
    ended_monotonic_ns: int | None = None
    status: LifecycleScopeStatus | None = None
    token_usage: TokenUsage | None = None
    tool_errored: bool | None = None

    def close(
        self,
        *,
        ended_monotonic_ns: int,
        status: LifecycleScopeStatus,
    ) -> None:
        """Close this scope at an observed lifecycle boundary."""

        if self.ended_monotonic_ns is not None:
            raise _LifecycleScopeError(
                f"Lifecycle scope is already closed: {self.scope_id}"
            )
        self.ended_monotonic_ns = ended_monotonic_ns
        self.status = status

    def duration_ns(self) -> int:
        """Return the monotonic duration of this closed scope."""

        if self.ended_monotonic_ns is None:
            raise _LifecycleScopeError(f"Lifecycle scope remains open: {self.scope_id}")
        return self.ended_monotonic_ns - self.started_monotonic_ns

    def to_record(self) -> LifecycleScopeRecord:
        """Freeze a fully closed accumulator into its public record."""

        if self.ended_monotonic_ns is None or self.status is None:
            raise _LifecycleScopeError(f"Lifecycle scope remains open: {self.scope_id}")
        return LifecycleScopeRecord(
            scope_id=self.scope_id,
            parent_scope_id=self.parent_scope_id,
            scope_type=self.scope_type,
            started_monotonic_ns=self.started_monotonic_ns,
            ended_monotonic_ns=self.ended_monotonic_ns,
            status=self.status,
            operation_name=self.operation_name,
            token_usage=self.token_usage,
        )


def _require_terminal_record(
    record: RunRecord,
) -> tuple[TerminalRunStatus, RunOutcome]:
    """Return terminal record facts required by telemetry construction."""

    if record.status == "running" or record.outcome is None:
        raise ValueError("Run telemetry requires a finalized run record.")
    return record.status, record.outcome


def _fold_lifecycle_scopes(
    events: Sequence[AgentEvent],
) -> tuple[_ScopeAccumulator, ...]:
    """Fold stamped lifecycle events into closed scope accumulators."""

    scopes: list[_ScopeAccumulator] = []
    open_scopes: list[_ScopeAccumulator] = []
    seen_scope_ids: set[str] = set()
    for event in events:
        if event.type in _UNSCOPED_EVENT_TYPES:
            _validate_unscoped_event(event)
        elif start_type := _START_SCOPE_TYPES.get(event.type):
            _start_folded_scope(
                event,
                scope_type=start_type,
                scopes=scopes,
                open_scopes=open_scopes,
                seen_scope_ids=seen_scope_ids,
            )
        elif end_rule := _END_SCOPE_TYPES.get(event.type):
            _end_folded_scope(
                event,
                scope_type=end_rule[0],
                status=end_rule[1],
                open_scopes=open_scopes,
            )
        else:
            raise _LifecycleScopeError(f"Unclassified runtime event type: {event.type}")

    if open_scopes:
        raise _LifecycleScopeError("Lifecycle event log has open scopes.")
    _attribute_token_usage(scopes)
    return tuple(scopes)


def _start_folded_scope(
    event: AgentEvent,
    *,
    scope_type: LifecycleScopeType,
    scopes: list[_ScopeAccumulator],
    open_scopes: list[_ScopeAccumulator],
    seen_scope_ids: set[str],
) -> None:
    """Open one scope from a stamped lifecycle start event."""

    metadata = _require_lifecycle_metadata(event)
    if metadata.scope_id in seen_scope_ids:
        raise _LifecycleScopeError(
            f"Lifecycle scope identity is duplicated: {metadata.scope_id}"
        )
    _validate_folded_parent(
        scope_type,
        metadata=metadata,
        open_scopes=open_scopes,
    )
    scope = _ScopeAccumulator(
        scope_id=metadata.scope_id,
        parent_scope_id=metadata.parent_scope_id,
        scope_type=scope_type,
        started_monotonic_ns=metadata.monotonic_ns,
        operation_name=_operation_name(event),
    )
    scopes.append(scope)
    open_scopes.append(scope)
    seen_scope_ids.add(scope.scope_id)


def _end_folded_scope(
    event: AgentEvent,
    *,
    scope_type: LifecycleScopeType,
    status: LifecycleScopeStatus,
    open_scopes: list[_ScopeAccumulator],
) -> None:
    """Close a matching scope and interrupt any still-open descendants."""

    metadata = _require_lifecycle_metadata(event)
    index = _find_scope_index(open_scopes, metadata.scope_id)
    scope = open_scopes[index]
    if scope.scope_type != scope_type:
        raise _LifecycleScopeError(
            f"Lifecycle end type does not match scope: {event.type}"
        )
    if scope.parent_scope_id != metadata.parent_scope_id:
        raise _LifecycleScopeError(
            f"Lifecycle parent changed before scope end: {metadata.scope_id}"
        )

    for descendant in reversed(open_scopes[index + 1 :]):
        descendant.close(
            ended_monotonic_ns=metadata.monotonic_ns,
            status="interrupted",
        )
    scope.close(ended_monotonic_ns=metadata.monotonic_ns, status=status)
    if isinstance(event, MessageEndEvent):
        scope.token_usage = event.token_usage
    if isinstance(event, ToolExecutionEndEvent):
        scope.tool_errored = event.outcome.tool_result_turn.is_error
    del open_scopes[index:]


def _validate_folded_parent(
    scope_type: LifecycleScopeType,
    *,
    metadata: LifecycleEventMetadata,
    open_scopes: Sequence[_ScopeAccumulator],
) -> None:
    """Require a start event to identify its immediate open parent."""

    parent_type = _PARENT_SCOPE_TYPES[scope_type]
    if parent_type is None:
        if open_scopes or metadata.parent_scope_id is not None:
            raise _LifecycleScopeError(
                "Run scope must start without a lifecycle parent."
            )
        return
    if not open_scopes:
        raise _missing_parent_error(scope_type, parent_type)
    parent = open_scopes[-1]
    if parent.scope_type != parent_type or metadata.parent_scope_id != parent.scope_id:
        raise _missing_parent_error(scope_type, parent_type)


def _require_lifecycle_metadata(event: AgentEvent) -> LifecycleEventMetadata:
    """Return required metadata from one scoped lifecycle event."""

    if event.lifecycle is None:
        raise _LifecycleScopeError(
            f"Scoped event has no lifecycle metadata: {event.type}"
        )
    return event.lifecycle


def _validate_unscoped_event(event: AgentEvent) -> None:
    """Reject lifecycle metadata on content-only events."""

    if event.lifecycle is not None:
        raise _LifecycleScopeError(
            f"Unscoped event carries lifecycle metadata: {event.type}"
        )


def _find_scope_index(
    open_scopes: Sequence[_ScopeAccumulator],
    scope_id: str,
) -> int:
    """Return the open-list index for a lifecycle scope identity."""

    for index in range(len(open_scopes) - 1, -1, -1):
        if open_scopes[index].scope_id == scope_id:
            return index
    raise _LifecycleScopeError(f"Lifecycle end has no matching start: {scope_id}")


def _operation_name(event: AgentEvent) -> str | None:
    """Return bounded operation identity from a lifecycle start."""

    if isinstance(event, MessageStartEvent):
        return event.response_id
    if isinstance(event, ToolExecutionStartEvent):
        return event.tool_name
    return None


def _attribute_token_usage(scopes: Sequence[_ScopeAccumulator]) -> None:
    """Attach summed message usage only to the root run scope."""

    usage = _sum_token_usage(
        scope.token_usage for scope in scopes if scope.scope_type == "message"
    )
    roots = [scope for scope in scopes if scope.scope_type == "run"]
    if len(roots) != 1:
        raise _LifecycleScopeError(
            "Lifecycle event log must contain exactly one run scope."
        )
    roots[0].token_usage = usage


def _sum_token_usage(usages: Iterable[TokenUsage | None]) -> TokenUsage | None:
    """Sum provider-reported counters without inventing missing usage."""

    reported = [usage for usage in usages if usage is not None]
    if not reported:
        return None
    return TokenUsage(
        input_tokens=sum(usage.input_tokens for usage in reported),
        output_tokens=sum(usage.output_tokens for usage in reported),
        total_tokens=sum(usage.total_tokens for usage in reported),
        cached_input_tokens=sum(usage.cached_input_tokens for usage in reported),
        reasoning_output_tokens=sum(
            usage.reasoning_output_tokens for usage in reported
        ),
    )


def _aggregate_tools(
    scopes: Sequence[_ScopeAccumulator],
) -> tuple[ToolAggregate, ...]:
    """Aggregate bounded timing and outcome facts by stable tool name."""

    groups: dict[str, list[_ScopeAccumulator]] = {}
    for scope in scopes:
        if scope.scope_type != "tool_execution":
            continue
        if scope.operation_name is None:
            raise _LifecycleScopeError(f"Tool scope has no tool name: {scope.scope_id}")
        groups.setdefault(scope.operation_name, []).append(scope)
    return tuple(_tool_aggregate(name, group) for name, group in groups.items())


def _tool_aggregate(
    tool_name: str,
    scopes: Sequence[_ScopeAccumulator],
) -> ToolAggregate:
    """Fold one tool's closed scopes into its run-level totals."""

    errors = [
        _require_tool_outcome(scope) for scope in scopes if scope.status == "completed"
    ]
    return ToolAggregate(
        tool_name=tool_name,
        call_count=len(scopes),
        completed_count=len(errors),
        error_count=errors.count(True),
        total_duration_ns=sum(scope.duration_ns() for scope in scopes),
    )


def _require_tool_outcome(scope: _ScopeAccumulator) -> bool:
    """Return the handled-error flag every completed tool scope must carry."""

    if scope.tool_errored is None:
        raise _LifecycleScopeError(
            f"Completed tool scope has no concrete outcome: {scope.scope_id}"
        )
    return scope.tool_errored


def _build_errors(
    outcome: RunOutcome,
    exceptions: Sequence[CapturedRunException],
) -> tuple[RunTelemetryError, ...]:
    """Build serialized errors in verdict-first observation order."""

    primary = _primary_error(outcome)
    errors = [primary] if primary is not None else []
    for captured in exceptions:
        if captured.role == "primary" and primary is not None:
            continue
        errors.append(
            RunTelemetryError(
                role=captured.role,
                stage=captured.stage,
                kind="exception",
                exception_type=type(captured.error).__name__,
                message=str(captured.error),
            )
        )
    return tuple(errors)


def _primary_error(outcome: RunOutcome) -> RunTelemetryError | None:
    """Derive the authoritative primary error from the terminal outcome."""

    if not isinstance(outcome, Failed):
        return None
    cause = outcome.cause
    if isinstance(cause, AgentFailure):
        return RunTelemetryError(
            role="primary",
            stage="execution",
            kind="agent_failure",
            message=cause.reason,
        )
    if isinstance(cause, ExecutionFailure):
        return RunTelemetryError(
            role="primary",
            stage=cause.origin,
            kind="exception",
            exception_type=cause.exception_type,
            message=cause.message,
        )
    raise TypeError(f"Unsupported run failure cause: {type(cause).__name__}")


def _missing_parent_error(
    scope_type: LifecycleScopeType,
    parent_type: LifecycleScopeType,
) -> _LifecycleScopeError:
    """Build the invariant error for a scope missing its immediate parent."""

    return _LifecycleScopeError(
        f"Lifecycle scope requires an open parent: {scope_type} requires {parent_type}"
    )
