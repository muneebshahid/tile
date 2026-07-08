"""Contract tests for history-store implementations."""

import asyncio
import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

import pytest

from tile import AgentRuntime, Session
from tile.history import (
    HistoryStore,
    InMemoryHistoryStore,
    SQLiteHistoryStore,
    SQLiteHistoryStoreSchemaError,
    SessionAlreadyExistsError,
    SessionNotFoundError,
)
from tile.types.conversation import (
    AssistantTurn,
    ConversationItem,
    ToolResultTurn,
    UserMessage,
)
from tile.types.stream_events import (
    ProviderMetadata,
    ProviderSource,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
)
from tile.types.tools import ToolImageContent, ToolTextContent
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


@pytest.fixture(params=[_in_memory_store, _sqlite_store])
def store_factory(request: pytest.FixtureRequest) -> Iterator[HistoryStoreFactory]:
    """Return tracked history-store factories and close SQLite stores on teardown."""

    stores: list[HistoryStore] = []
    build_store = cast(HistoryStoreFactory, request.param)

    def _tracked_store_factory(tmp_path: Path) -> HistoryStore:
        """Build and track a history store for one contract test."""

        store = build_store(tmp_path)
        stores.append(store)
        return store

    try:
        yield _tracked_store_factory
    finally:
        for store in stores:
            if isinstance(store, SQLiteHistoryStore):
                store.close()


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


def test_history_store_rejects_unknown_session_writes(
    tmp_path: Path,
    store_factory: HistoryStoreFactory,
) -> None:
    """Reject appends to sessions that have not been created."""

    store = store_factory(tmp_path)

    with pytest.raises(SessionNotFoundError, match="Unknown session: missing"):
        store.append_history("missing", [UserMessage(content="hello")])


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


def test_history_store_round_trips_all_conversation_item_variants(
    tmp_path: Path,
    store_factory: HistoryStoreFactory,
) -> None:
    """Persist and reload every conversation item variant without loss."""

    store = store_factory(tmp_path)
    store.ensure_session(session_id="variants")
    items = _conversation_item_variants()

    store.append_history("variants", items)

    assert list(store.get_history("variants")) == items


def _conversation_item_variants() -> list[ConversationItem]:
    """Build one fully populated example of every conversation item variant."""

    completed_turn = AssistantTurn(
        source=ProviderSource(provider="openai", model="gpt-5.4"),
        blocks=[
            ReasoningBlock(summary_text="Consider the weather."),
            TextBlock(
                text="Checking the weather.",
                provider_metadata=ProviderMetadata(data={"phase": "final_answer"}),
            ),
            ToolCallBlock(
                call_id="call_1",
                name="get_weather",
                arguments={"city": "Munich"},
                provider_metadata=ProviderMetadata(data={"provider_item_id": "fc_1"}),
            ),
        ],
        response_id="resp_1",
        stop_reason="tool_use",
    )
    failed_turn = AssistantTurn(
        stop_reason="error",
        status="error",
        error_message="provider exploded",
    )
    tool_result_turn = ToolResultTurn(
        call_id="call_1",
        tool_name="get_weather",
        content=[
            ToolTextContent(text="sunny"),
            ToolImageContent(data="aGVsbG8=", mime_type="image/png"),
        ],
        is_error=True,
    )
    return [
        UserMessage(content="What is the weather in Munich?"),
        completed_turn,
        failed_turn,
        tool_result_turn,
    ]


def test_history_store_rejects_copy_from_unknown_source(
    tmp_path: Path,
    store_factory: HistoryStoreFactory,
) -> None:
    """Reject history copies from unknown sources without creating the target."""

    store = store_factory(tmp_path)

    with pytest.raises(SessionNotFoundError, match="Unknown session: missing"):
        store.copy_history(source_session_id="missing", target_session_id="fork")

    with pytest.raises(SessionNotFoundError, match="Unknown session: fork"):
        store.get_session("fork")


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

    database_path = tmp_path / "tile.db"
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

    database_path = tmp_path / "tile.db"
    store = SQLiteHistoryStore(database_path)
    store.close()

    connection = sqlite3.connect(database_path)
    version = connection.execute(
        "SELECT value FROM tile_meta WHERE key = 'schema_version'"
    ).fetchone()
    connection.close()

    assert version == ("1",)


def test_sqlite_history_store_supports_explicit_in_memory_mode() -> None:
    """Create an in-memory SQLite history store without a filesystem path."""

    store = SQLiteHistoryStore(in_memory=True)

    store.ensure_session(session_id="memory")
    store.append_history("memory", [UserMessage(content="hello")])

    assert expect_user_message(store.get_history("memory")[0]).content == "hello"
    store.close()


def test_sqlite_history_store_requires_path_for_file_backed_mode() -> None:
    """Require a database path unless the caller opts into in-memory storage."""

    with pytest.raises(ValueError, match="database_path is required"):
        SQLiteHistoryStore()


def test_sqlite_history_store_rejects_unknown_schema_version(tmp_path: Path) -> None:
    """Fail loudly before opening databases written by unsupported schema versions."""

    database_path = tmp_path / "future.db"
    connection = sqlite3.connect(database_path)
    connection.execute(
        "CREATE TABLE tile_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO tile_meta (key, value) VALUES ('schema_version', '999')"
    )
    connection.commit()
    connection.close()

    with pytest.raises(SQLiteHistoryStoreSchemaError, match="999"):
        SQLiteHistoryStore(database_path)


def _collect_prompt_events(session: Session, content: str) -> None:
    """Run one session prompt to completion."""

    async def _collect() -> None:
        """Submit the prompt and wait for its terminal status."""

        run = await session.prompt(content)
        await run.wait()

    asyncio.run(_collect())
