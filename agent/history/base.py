"""Session metadata and history storage contracts."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ai.types.conversation import ConversationItem


@dataclass(frozen=True)
class SessionRecord:
    """Metadata for one known runtime session."""

    session_id: str
    name: str | None = None


class SessionNotFoundError(KeyError):
    """Raised when a history operation references an unknown session."""


class HistoryStore(Protocol):
    """Stores session records and completed conversation history."""

    def ensure_session(
        self,
        *,
        session_id: str,
        name: str | None = None,
    ) -> SessionRecord:
        """Create a session record when it does not already exist."""
        ...

    def get_session(self, session_id: str) -> SessionRecord:
        """Return metadata for an existing session."""
        ...

    def list_sessions(self) -> Sequence[SessionRecord]:
        """Return known session records."""
        ...

    def get_history(self, session_id: str) -> Sequence[ConversationItem]:
        """Return completed conversation history for a session."""
        ...

    def append_history(
        self,
        session_id: str,
        items: Sequence[ConversationItem],
    ) -> None:
        """Append completed conversation items to a session."""
        ...
