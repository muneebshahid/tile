"""Private lifecycle tracking used to prepare run telemetry."""

from collections.abc import Callable
from dataclasses import dataclass

from tile.events import AgentEvent, LifecycleEventMetadata
from tile.telemetry.models import LifecycleScopeType

MonotonicClock = Callable[[], int]
ScopeIdFactory = Callable[[], str]

_START_SCOPE_TYPES: dict[str, LifecycleScopeType] = {
    "run_start": "run",
    "agent_start": "agent",
    "turn_start": "turn",
    "message_start": "message",
    "tool_execution_start": "tool_execution",
}
_END_SCOPE_TYPES: dict[str, LifecycleScopeType] = {
    "run_end": "run",
    "agent_end": "agent",
    "agent_interrupted": "agent",
    "turn_end": "turn",
    "turn_interrupted": "turn",
    "message_end": "message",
    "message_interrupted": "message",
    "tool_execution_end": "tool_execution",
    "tool_execution_interrupted": "tool_execution",
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


class _LifecycleScopeTrackingError(RuntimeError):
    """Raised when runtime events violate lifecycle scope invariants."""


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
            raise _LifecycleScopeTrackingError(
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
        if scope_type := _END_SCOPE_TYPES.get(event_type):
            return self._close_scope(scope_type, event_type=event_type)
        raise _LifecycleScopeTrackingError(
            f"Unclassified runtime event type: {event_type}"
        )

    def _open_scope(
        self,
        scope_type: LifecycleScopeType,
    ) -> _OpenLifecycleScope:
        """Create and retain one scope under its currently open parent."""

        if self._find_open_scope(scope_type) is not None:
            raise _LifecycleScopeTrackingError(
                f"Lifecycle scope is already open: {scope_type}"
            )

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
        raise _LifecycleScopeTrackingError(
            f"Lifecycle end has no matching start: {event_type}"
        )

    def _require_open_parent(
        self,
        scope_type: LifecycleScopeType,
        *,
        parent_type: LifecycleScopeType | None,
    ) -> _OpenLifecycleScope | None:
        """Return the immediate parent required by a new scope."""

        if parent_type is None:
            if self._open_scopes:
                raise _LifecycleScopeTrackingError(
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


def _missing_parent_error(
    scope_type: LifecycleScopeType,
    parent_type: LifecycleScopeType,
) -> _LifecycleScopeTrackingError:
    """Build the invariant error for a scope missing its immediate parent."""

    return _LifecycleScopeTrackingError(
        f"Lifecycle scope requires an open parent: {scope_type} requires {parent_type}"
    )
