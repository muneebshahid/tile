"""In-memory history store implementation."""

from collections.abc import Sequence
from dataclasses import dataclass, field

from ai.types.conversation import ConversationItem
from agent.history.base import SessionNotFoundError, SessionRecord


@dataclass
class InMemoryHistoryStore:
    """In-memory session records and conversation history."""

    _sessions: dict[str, SessionRecord] = field(default_factory=dict)
    _items_by_session: dict[str, list[ConversationItem]] = field(default_factory=dict)

    def ensure_session(
        self,
        *,
        session_id: str,
        name: str | None = None,
    ) -> SessionRecord:
        """Create a session record when it does not already exist."""

        existing = self._sessions.get(session_id)
        if existing is not None:
            return existing

        record = SessionRecord(session_id=session_id, name=name)
        self._sessions[session_id] = record
        self._items_by_session[session_id] = []
        return record

    def get_session(self, session_id: str) -> SessionRecord:
        """Return metadata for an existing session."""

        self._require_session(session_id)
        return self._sessions[session_id]

    def list_sessions(self) -> Sequence[SessionRecord]:
        """Return known session records."""

        return tuple(self._sessions.values())

    def get_history(self, session_id: str) -> Sequence[ConversationItem]:
        """Return completed conversation history for a session."""

        self._require_session(session_id)
        return tuple(self._items_by_session[session_id])

    def append_history(
        self,
        session_id: str,
        items: Sequence[ConversationItem],
    ) -> None:
        """Append completed conversation items to a session."""

        self._require_session(session_id)
        self._items_by_session[session_id].extend(items)

    def _require_session(self, session_id: str) -> None:
        """Raise a clear error when a session id is unknown."""

        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Unknown session: {session_id}")
