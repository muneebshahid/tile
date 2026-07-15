"""SQLite-backed session metadata and conversation history store."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from tile._sqlite import (
    immediate_transaction,
    initialize_schema,
    resolve_connection_target,
)
from tile.history.base import (
    SessionAlreadyExistsError,
    SessionNotFoundError,
    SessionRecord,
)
from tile.history.serialization import dump_conversation_item, load_conversation_item
from tile.types.conversation import ConversationItem

SQLITE_HISTORY_SCHEMA_VERSION = "1"
_SCHEMA_VERSION_KEY = "schema_version"


class SQLiteHistoryStoreSchemaError(RuntimeError):
    """Raised when a SQLite history database uses an unsupported schema."""


class SQLiteHistoryStore:
    """SQLite implementation of session records and conversation history."""

    def __init__(
        self,
        database_path: Path | str | None = None,
        *,
        in_memory: bool = False,
    ) -> None:
        """Open a SQLite history database and initialize its schema."""

        self._connection_target = resolve_connection_target(
            database_path=database_path,
            in_memory=in_memory,
        )
        self._connection = sqlite3.connect(self._connection_target)
        try:
            self._connection.execute("PRAGMA foreign_keys = ON")
            initialize_schema(
                self._connection,
                version_key=_SCHEMA_VERSION_KEY,
                expected_version=SQLITE_HISTORY_SCHEMA_VERSION,
                store_label="history",
                schema_error=SQLiteHistoryStoreSchemaError,
                create_schema=self._create_current_schema,
            )
        except BaseException:
            self._connection.close()
            raise

    def ensure_session(
        self,
        *,
        session_id: str,
        name: str | None = None,
    ) -> SessionRecord:
        """Create a session record when it does not already exist."""

        with immediate_transaction(self._connection):
            self._connection.execute(
                """
                INSERT OR IGNORE INTO sessions (session_id, name)
                VALUES (?, ?)
                """,
                (session_id, name),
            )
            return self._fetch_session(session_id)

    def get_session(self, session_id: str) -> SessionRecord:
        """Return metadata for an existing session."""

        return self._fetch_session(session_id)

    def list_sessions(self) -> Sequence[SessionRecord]:
        """Return known session records."""

        rows = self._connection.execute(
            """
            SELECT session_id, name
            FROM sessions
            ORDER BY rowid
            """
        ).fetchall()
        session_rows = cast("Sequence[tuple[str, str | None]]", rows)
        return tuple(
            SessionRecord(session_id=session_id, name=name)
            for session_id, name in session_rows
        )

    def get_history(self, session_id: str) -> Sequence[ConversationItem]:
        """Return completed conversation history for a session."""

        self._require_session(session_id)
        rows = self._connection.execute(
            """
            SELECT payload_json
            FROM conversation_items
            WHERE session_id = ?
            ORDER BY position
            """,
            (session_id,),
        ).fetchall()
        history_rows = cast("Sequence[tuple[str]]", rows)
        return tuple(
            load_conversation_item(payload_json) for (payload_json,) in history_rows
        )

    def append_history(
        self,
        session_id: str,
        items: Sequence[ConversationItem],
    ) -> None:
        """Append completed conversation items to a session."""

        with immediate_transaction(self._connection):
            self._require_session(session_id)
            if items:
                next_position = self._next_history_position(session_id)
                self._insert_history_items(session_id, items, next_position)

    def copy_history(
        self,
        *,
        source_session_id: str,
        target_session_id: str,
        target_name: str | None = None,
    ) -> SessionRecord:
        """Create a target session with copied source history."""

        with immediate_transaction(self._connection):
            self._require_session(source_session_id)
            self._reject_existing_session(target_session_id)
            record = self._insert_session(target_session_id, target_name)
            items = self.get_history(source_session_id)
            self._insert_history_items(target_session_id, items, 0)
            return record

    def close(self) -> None:
        """Close the underlying SQLite connection."""

        self._connection.close()

    def _create_current_schema(self) -> None:
        """Create current history schema tables."""

        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                name TEXT
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_items (
                session_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                role TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (session_id, position),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
            """
        )

    def _fetch_session(self, session_id: str) -> SessionRecord:
        """Return a session record or raise a domain lookup error."""

        row = self._connection.execute(
            """
            SELECT session_id, name
            FROM sessions
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        session_row = cast("tuple[str, str | None] | None", row)
        if session_row is None:
            raise SessionNotFoundError(f"Unknown session: {session_id}")
        return SessionRecord(session_id=session_row[0], name=session_row[1])

    def _require_session(self, session_id: str) -> None:
        """Raise a clear error when a session id is unknown."""

        self._fetch_session(session_id)

    def _reject_existing_session(self, session_id: str) -> None:
        """Raise a clear error when a session id already exists."""

        row = self._connection.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is not None:
            raise SessionAlreadyExistsError(f"Session already exists: {session_id}")

    def _insert_session(self, session_id: str, name: str | None) -> SessionRecord:
        """Insert and return one new session record."""

        self._connection.execute(
            """
            INSERT INTO sessions (session_id, name)
            VALUES (?, ?)
            """,
            (session_id, name),
        )
        return SessionRecord(session_id=session_id, name=name)

    def _next_history_position(self, session_id: str) -> int:
        """Return the next append position for a session."""

        row = self._connection.execute(
            """
            SELECT COALESCE(MAX(position), -1) + 1
            FROM conversation_items
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        position_row = cast("tuple[int]", row)
        return position_row[0]

    def _insert_history_items(
        self,
        session_id: str,
        items: Sequence[ConversationItem],
        first_position: int,
    ) -> None:
        """Insert conversation items starting at a session-local position."""

        rows = [
            (
                session_id,
                first_position + offset,
                item.role,
                dump_conversation_item(item),
            )
            for offset, item in enumerate(items)
        ]
        self._connection.executemany(
            """
            INSERT INTO conversation_items (
                session_id,
                position,
                role,
                payload_json
            )
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
