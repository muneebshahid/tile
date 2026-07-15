"""Contract tests for durable run-summary repositories."""

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from tile.history import SQLiteHistoryStore
from tile.result import Completed
from tile.runs import (
    InMemoryRunStore,
    RunAlreadyExistsError,
    RunFailure,
    RunNotFoundError,
    RunRecord,
    SQLiteRunStore,
    SQLiteRunStoreSchemaError,
)
from tile.types.conversation import UserMessage

STARTED_AT = datetime(2026, 7, 15, 9, 30, tzinfo=UTC)
ENDED_AT = STARTED_AT + timedelta(seconds=2)
FAILURE = RunFailure(
    origin="execution",
    exception_type="ConnectionError",
    message="connection failed",
)


def test_run_record_transitions_from_running_to_completed() -> None:
    """Finish a running record while preserving its stable identity."""

    running = _running_record()

    completed = running.finish(
        status="completed",
        ended_at=ENDED_AT,
        provider="test",
        outcome=Completed(value="done"),
    )

    assert completed == RunRecord(
        run_id="run-1",
        session_id="session-1",
        status="completed",
        started_at=STARTED_AT,
        ended_at=ENDED_AT,
        model="gpt-5.4",
        provider="test",
        outcome=Completed(value="done"),
    )


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(
            {"status": "running", "ended_at": ENDED_AT},
            id="running_with_end_time",
        ),
        pytest.param(
            {"status": "completed", "ended_at": None},
            id="terminal_without_end_time",
        ),
        pytest.param(
            {"status": "failed", "ended_at": ENDED_AT},
            id="failed_without_diagnostics",
        ),
        pytest.param(
            {
                "status": "completed",
                "ended_at": STARTED_AT - timedelta(seconds=1),
            },
            id="end_before_start",
        ),
        pytest.param(
            {
                "status": "running",
                "started_at": datetime(2026, 7, 15, 9, 30),
            },
            id="naive_timestamp",
        ),
        pytest.param(
            {
                "status": "aborted",
                "ended_at": ENDED_AT,
                "failure": FAILURE,
            },
            id="aborted_with_failure",
        ),
    ],
)
def test_run_record_rejects_invalid_lifecycle_combinations(
    values: dict[str, str | datetime | RunFailure | None],
) -> None:
    """Keep persisted status and terminal fields internally consistent."""

    record_values = {
        "run_id": "run-1",
        "session_id": "session-1",
        "started_at": STARTED_AT,
        "model": "gpt-5.4",
        **values,
    }
    with pytest.raises(ValidationError):
        RunRecord.model_validate(record_values)


def test_in_memory_run_store_creates_updates_and_lists_records() -> None:
    """Store session-local records in submission order through completion."""

    store = InMemoryRunStore()
    first = _running_record(run_id="run-1")
    second = _running_record(run_id="run-2")
    other = _running_record(run_id="run-3", session_id="session-2")

    store.create_run(first)
    store.create_run(second)
    store.create_run(other)
    completed = first.finish(status="completed", ended_at=ENDED_AT)
    store.update_run(completed)

    assert store.get_run("run-1") == completed
    assert store.list_runs("session-1") == (completed, second)
    assert store.list_runs("missing") == ()


def test_in_memory_run_store_returns_defensive_snapshots() -> None:
    """Prevent callers from mutating nested data held by the repository."""

    store = InMemoryRunStore()
    completed = _running_record().finish(
        status="completed",
        ended_at=ENDED_AT,
        outcome=Completed(value={"answer": "original"}),
    )
    store.create_run(completed)

    fetched = store.get_run(completed.run_id)
    assert isinstance(fetched.outcome, Completed)
    assert isinstance(fetched.outcome.value, dict)
    fetched.outcome.value["answer"] = "mutated"

    assert store.get_run(completed.run_id) == completed


def test_in_memory_run_store_rejects_duplicate_and_unknown_records() -> None:
    """Raise domain errors for conflicting creates and missing lookups."""

    store = InMemoryRunStore()
    record = _running_record()
    store.create_run(record)

    with pytest.raises(RunAlreadyExistsError, match="run-1"):
        store.create_run(record)
    with pytest.raises(RunNotFoundError, match="missing"):
        store.get_run("missing")
    with pytest.raises(RunNotFoundError, match="missing"):
        store.update_run(_running_record(run_id="missing"))


def test_sqlite_run_store_round_trips_terminal_records_after_restart(
    tmp_path: Path,
) -> None:
    """Persist typed outcomes and structured failures across store restarts."""

    database_path = tmp_path / "runs.db"
    store = SQLiteRunStore(database_path)
    completed = _running_record(run_id="completed")
    failed = _running_record(run_id="failed")
    store.create_run(completed)
    store.create_run(failed)
    store.update_run(
        completed.finish(
            status="completed",
            ended_at=ENDED_AT,
            provider="test",
            outcome=Completed(value={"answer": "done"}),
        )
    )
    store.update_run(
        failed.finish(
            status="failed",
            ended_at=ENDED_AT,
            provider="test",
            failure=FAILURE,
        )
    )
    expected = store.list_runs("session-1")
    store.close()

    reopened = SQLiteRunStore(database_path)
    try:
        assert reopened.list_runs("session-1") == expected
        assert reopened.get_run("completed") == expected[0]
        assert reopened.get_run("failed") == expected[1]
    finally:
        reopened.close()


def test_sqlite_run_and_history_stores_share_one_database_file(
    tmp_path: Path,
) -> None:
    """Keep separate store contracts over compatible tables in one database."""

    database_path = tmp_path / "tile.db"
    history_store = SQLiteHistoryStore(database_path)
    run_store = SQLiteRunStore(database_path)
    history_store.ensure_session(session_id="shared")
    history_store.append_history("shared", [UserMessage(content="hello")])
    record = _running_record(session_id="shared")
    run_store.create_run(record)
    run_store.update_run(record.finish(status="aborted", ended_at=ENDED_AT))
    history_store.close()
    run_store.close()

    reopened_history = SQLiteHistoryStore(database_path)
    reopened_runs = SQLiteRunStore(database_path)
    try:
        assert reopened_history.get_history("shared") == (UserMessage(content="hello"),)
        assert reopened_runs.get_run("run-1").status == "aborted"
    finally:
        reopened_history.close()
        reopened_runs.close()


def test_sqlite_run_store_records_and_validates_its_schema_version(
    tmp_path: Path,
) -> None:
    """Use a run-specific schema marker beside the history schema marker."""

    database_path = tmp_path / "tile.db"
    history_store = SQLiteHistoryStore(database_path)
    history_store.close()
    run_store = SQLiteRunStore(database_path)
    run_store.close()

    connection = sqlite3.connect(database_path)
    versions = dict(
        connection.execute("SELECT key, value FROM tile_meta ORDER BY key").fetchall()
    )
    connection.close()

    assert versions == {"run_schema_version": "1", "schema_version": "1"}


def test_sqlite_run_store_rejects_unknown_schema_version(tmp_path: Path) -> None:
    """Fail before reading run records written by an unsupported schema."""

    database_path = tmp_path / "future.db"
    connection = sqlite3.connect(database_path)
    connection.execute(
        "CREATE TABLE tile_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO tile_meta (key, value) VALUES ('run_schema_version', '999')"
    )
    connection.commit()
    connection.close()

    with pytest.raises(SQLiteRunStoreSchemaError, match="999"):
        SQLiteRunStore(database_path)


def test_sqlite_run_store_supports_explicit_in_memory_mode() -> None:
    """Create a process-local SQLite repository without a filesystem path."""

    store = SQLiteRunStore(in_memory=True)
    try:
        store.create_run(_running_record())
        assert store.get_run("run-1").status == "running"
    finally:
        store.close()


def test_sqlite_run_store_requires_path_for_file_backed_mode() -> None:
    """Require a path unless explicit SQLite in-memory mode is selected."""

    with pytest.raises(ValueError, match="database_path is required"):
        SQLiteRunStore()


def _running_record(
    *,
    run_id: str = "run-1",
    session_id: str = "session-1",
) -> RunRecord:
    """Build one deterministic running record."""

    return RunRecord(
        run_id=run_id,
        session_id=session_id,
        status="running",
        started_at=STARTED_AT,
        model="gpt-5.4",
    )
