"""Contract tests for history-store implementations."""

import asyncio
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from ori import AgentRuntime, Session
from ori.history import (
    HistoryStore,
    InMemoryHistoryStore,
    SQLiteHistoryStore,
    SQLiteHistoryStoreSchemaError,
    SessionAlreadyExistsError,
    SessionNotFoundError,
)
from ori.types.conversation import AssistantTurn, UserMessage
from tests.support.agent_streams import ProviderStreamMock, final_text_stream
from tests.support.conversation_assertions import (
    expect_assistant_turn,
    expect_user_message,
)

HistoryStoreFactory = Callable[[Path], HistoryStore]


def _in_memory_store(tmp_path: Path) -> HistoryStore:
    """Build an in-memory history store for contract tests."""

    _ = tmp_path
    return InMemoryHistoryStore()


def _sqlite_store(tmp_path: Path) -> HistoryStore:
    """Build a SQLite history store for contract tests."""

    return SQLiteHistoryStore(tmp_path / "history.db")


@pytest.mark.parametrize("store_factory", [_in_memory_store, _sqlite_store])
def test_history_store_creates_and_lists_sessions(
    tmp_path: Path,
    store_factory: HistoryStoreFactory,
) -> None:
    """Create sessions with stable metadata and insertion ordering."""

    store = store_factory(tmp_path)

    first = store.ensure_session(session_id="first", name="First")
    repeated = store.ensure_session(session_id="first", name="Changed")
    second = store.ensure_session(session_id="second")

    assert first.name == "First"
    assert repeated.name == "First"
    assert [session.session_id for session in store.list_sessions()] == [
        first.session_id,
        second.session_id,
    ]


@pytest.mark.parametrize("store_factory", [_in_memory_store, _sqlite_store])
def test_history_store_rejects_unknown_session_writes(
    tmp_path: Path,
    store_factory: HistoryStoreFactory,
) -> None:
    """Reject appends to sessions that have not been created."""

    store = store_factory(tmp_path)

    with pytest.raises(SessionNotFoundError, match="Unknown session: missing"):
        store.append_history("missing", [UserMessage(content="hello")])


@pytest.mark.parametrize("store_factory", [_in_memory_store, _sqlite_store])
def test_history_store_preserves_history_order(
    tmp_path: Path,
    store_factory: HistoryStoreFactory,
) -> None:
    """Append and load conversation items in session-local order."""

    store = store_factory(tmp_path)
    store.ensure_session(session_id="ordered")

    store.append_history("ordered", [UserMessage(content="first")])
    store.append_history("ordered", [AssistantTurn(response_id="resp_first")])
    store.append_history("ordered", [UserMessage(content="second")])

    history = store.get_history("ordered")
    assert expect_user_message(history[0]).content == "first"
    assert expect_assistant_turn(history[1]).response_id == "resp_first"
    assert expect_user_message(history[2]).content == "second"


@pytest.mark.parametrize("store_factory", [_in_memory_store, _sqlite_store])
def test_history_store_returns_defensive_history_snapshots(
    tmp_path: Path,
    store_factory: HistoryStoreFactory,
) -> None:
    """Return immutable containers without leaking mutable stored item instances."""

    store = store_factory(tmp_path)
    store.ensure_session(session_id="snapshot")
    user_message = UserMessage(content="hello")

    store.append_history("snapshot", [user_message])
    user_message.content = "mutated original"
    history = store.get_history("snapshot")
    expect_user_message(history[0]).content = "mutated snapshot"

    stored_history = store.get_history("snapshot")
    assert isinstance(history, tuple)
    assert expect_user_message(stored_history[0]).content == "hello"


@pytest.mark.parametrize("store_factory", [_in_memory_store, _sqlite_store])
def test_history_store_copies_history_to_new_session(
    tmp_path: Path,
    store_factory: HistoryStoreFactory,
) -> None:
    """Fork a session by copying completed conversation history."""

    store = store_factory(tmp_path)
    store.ensure_session(session_id="source", name="Source")
    store.append_history("source", [UserMessage(content="hello")])

    fork = store.copy_history(
        source_session_id="source",
        target_session_id="fork",
        target_name="Fork",
    )

    assert fork.name == "Fork"
    assert store.get_history("fork") == store.get_history("source")


@pytest.mark.parametrize("store_factory", [_in_memory_store, _sqlite_store])
def test_history_store_rejects_duplicate_copy_target(
    tmp_path: Path,
    store_factory: HistoryStoreFactory,
) -> None:
    """Reject copied histories that would overwrite an existing session."""

    store = store_factory(tmp_path)
    store.ensure_session(session_id="source")
    store.ensure_session(session_id="existing")

    with pytest.raises(SessionAlreadyExistsError, match="existing"):
        store.copy_history(
            source_session_id="source",
            target_session_id="existing",
        )


def test_sqlite_history_store_survives_runtime_restart(tmp_path: Path) -> None:
    """Continue a runtime session from completed SQLite history after restart."""

    database_path = tmp_path / "ori.db"
    provider = ProviderStreamMock(
        [
            final_text_stream("resp_first", "first answer"),
            final_text_stream("resp_second", "second answer"),
        ]
    )

    first_store = SQLiteHistoryStore(database_path)
    first_runtime = AgentRuntime(
        stream_fn=provider.fn,
        model="gpt-5.4",
        history_store=first_store,
    )
    first_session = first_runtime.session(session_id="restart")
    _collect_prompt_events(first_session, "first")
    first_store.close()

    second_store = SQLiteHistoryStore(database_path)
    second_runtime = AgentRuntime(
        stream_fn=provider.fn,
        model="gpt-5.4",
        history_store=second_store,
    )
    restarted_session = second_runtime.get_session("restart")
    _collect_prompt_events(restarted_session, "second")

    second_request_history = provider.history(1)
    assert len(second_request_history) == 3
    assert expect_user_message(second_request_history[0]).content == "first"
    assert expect_assistant_turn(second_request_history[1]).response_id == "resp_first"
    assert expect_user_message(second_request_history[2]).content == "second"
    second_store.close()


def test_sqlite_history_store_records_schema_version(tmp_path: Path) -> None:
    """Initialize new SQLite history databases with a schema version marker."""

    database_path = tmp_path / "ori.db"
    store = SQLiteHistoryStore(database_path)
    store.close()

    connection = sqlite3.connect(database_path)
    version = connection.execute(
        "SELECT value FROM ori_meta WHERE key = 'schema_version'"
    ).fetchone()
    connection.close()

    assert version == ("1",)


def test_sqlite_history_store_rejects_unknown_schema_version(tmp_path: Path) -> None:
    """Fail loudly before opening databases written by unsupported schema versions."""

    database_path = tmp_path / "future.db"
    connection = sqlite3.connect(database_path)
    connection.execute(
        "CREATE TABLE ori_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO ori_meta (key, value) VALUES ('schema_version', '999')"
    )
    connection.commit()
    connection.close()

    with pytest.raises(SQLiteHistoryStoreSchemaError, match="999"):
        SQLiteHistoryStore(database_path)


def _collect_prompt_events(session: Session, content: str) -> None:
    """Collect all runtime events from a session prompt."""

    async def _collect() -> None:
        """Drain one prompt stream."""

        async for _ in session.prompt(content):
            pass

    asyncio.run(_collect())
