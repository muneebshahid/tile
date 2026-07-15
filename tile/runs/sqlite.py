"""SQLite-backed durable run-summary repository."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TypeAlias, cast

from pydantic import TypeAdapter

from tile._sqlite import (
    immediate_transaction,
    initialize_schema,
    resolve_connection_target,
)
from tile.result import RunOutcome
from tile.runs.base import (
    RunAlreadyExistsError,
    RunFailure,
    RunNotFoundError,
    RunRecord,
    RunStatus,
)

SQLITE_RUN_SCHEMA_VERSION = "1"
_RUN_SCHEMA_VERSION_KEY = "run_schema_version"

_RUN_COLUMNS = (
    "run_id",
    "session_id",
    "status",
    "started_at",
    "ended_at",
    "model",
    "provider",
    "outcome_json",
    "failure_json",
)
_INSERT_RUN_SQL = f"""
    INSERT INTO run_records ({", ".join(_RUN_COLUMNS)})
    VALUES ({", ".join("?" for _ in _RUN_COLUMNS)})
"""
_UPDATE_RUN_SQL = f"""
    UPDATE run_records
    SET {", ".join(f"{column} = ?" for column in _RUN_COLUMNS[1:])}
    WHERE run_id = ?
"""
_SELECT_RUNS_SQL = f"SELECT {', '.join(_RUN_COLUMNS)} FROM run_records"

_RunRow: TypeAlias = tuple[
    str,
    str,
    str,
    str,
    str | None,
    str,
    str | None,
    str | None,
    str | None,
]

_OUTCOME_ADAPTER: TypeAdapter[RunOutcome] = TypeAdapter(RunOutcome)


class SQLiteRunStoreSchemaError(RuntimeError):
    """Raised when a SQLite run database uses an unsupported schema."""


class SQLiteRunStore:
    """SQLite implementation of the durable run-summary repository."""

    def __init__(
        self,
        database_path: Path | str | None = None,
        *,
        in_memory: bool = False,
    ) -> None:
        """Open a SQLite run database and initialize its schema."""

        self._connection_target = resolve_connection_target(
            database_path=database_path,
            in_memory=in_memory,
        )
        self._connection = sqlite3.connect(self._connection_target)
        try:
            initialize_schema(
                self._connection,
                version_key=_RUN_SCHEMA_VERSION_KEY,
                expected_version=SQLITE_RUN_SCHEMA_VERSION,
                store_label="run",
                schema_error=SQLiteRunStoreSchemaError,
                create_schema=self._create_current_schema,
            )
        except BaseException:
            self._connection.close()
            raise

    def create_run(self, record: RunRecord) -> None:
        """Persist a newly submitted running record."""

        try:
            with immediate_transaction(self._connection):
                self._connection.execute(_INSERT_RUN_SQL, _record_values(record))
        except sqlite3.IntegrityError as error:
            raise RunAlreadyExistsError(
                f"Run already exists: {record.run_id}"
            ) from error

    def update_run(self, record: RunRecord) -> None:
        """Replace an existing run record with its latest state."""

        with immediate_transaction(self._connection):
            cursor = self._connection.execute(
                _UPDATE_RUN_SQL,
                (*_record_values(record)[1:], record.run_id),
            )
            if cursor.rowcount == 0:
                raise RunNotFoundError(f"Unknown run: {record.run_id}")

    def get_run(self, run_id: str) -> RunRecord:
        """Return a run record by its stable id."""

        row = self._connection.execute(
            _SELECT_RUNS_SQL + " WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        run_row = cast("_RunRow | None", row)
        if run_row is None:
            raise RunNotFoundError(f"Unknown run: {run_id}")
        return _record_from_row(run_row)

    def list_runs(self, session_id: str) -> tuple[RunRecord, ...]:
        """Return run records for one session in submission order."""

        rows = self._connection.execute(
            _SELECT_RUNS_SQL + " WHERE session_id = ? ORDER BY rowid",
            (session_id,),
        ).fetchall()
        run_rows = cast("Sequence[_RunRow]", rows)
        return tuple(_record_from_row(row) for row in run_rows)

    def close(self) -> None:
        """Close the underlying SQLite connection."""

        self._connection.close()

    def _create_current_schema(self) -> None:
        """Create the current run-record table and lookup index."""

        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS run_records (
                run_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                model TEXT NOT NULL,
                provider TEXT,
                outcome_json TEXT,
                failure_json TEXT
            )
            """
        )
        self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS run_records_session_id
            ON run_records (session_id)
            """
        )


def _record_values(record: RunRecord) -> _RunRow:
    """Serialize one run record into its SQLite column values."""

    return (
        record.run_id,
        record.session_id,
        record.status,
        record.started_at.isoformat(),
        record.ended_at.isoformat() if record.ended_at is not None else None,
        record.model,
        record.provider,
        _dump_outcome(record.outcome),
        record.failure.model_dump_json() if record.failure is not None else None,
    )


def _record_from_row(row: _RunRow) -> RunRecord:
    """Deserialize one SQLite row into a validated run record."""

    (
        run_id,
        session_id,
        status,
        started_at,
        ended_at,
        model,
        provider,
        outcome_json,
        failure_json,
    ) = row
    return RunRecord(
        run_id=run_id,
        session_id=session_id,
        status=cast("RunStatus", status),
        started_at=datetime.fromisoformat(started_at),
        ended_at=datetime.fromisoformat(ended_at) if ended_at is not None else None,
        model=model,
        provider=provider,
        outcome=_load_outcome(outcome_json),
        failure=(
            RunFailure.model_validate_json(failure_json)
            if failure_json is not None
            else None
        ),
    )


def _dump_outcome(outcome: RunOutcome | None) -> str | None:
    """Serialize a typed run outcome when one is present."""

    if outcome is None:
        return None
    return outcome.model_dump_json()


def _load_outcome(payload_json: str | None) -> RunOutcome | None:
    """Deserialize a typed run outcome when one is present."""

    if payload_json is None:
        return None
    return _OUTCOME_ADAPTER.validate_json(payload_json)
