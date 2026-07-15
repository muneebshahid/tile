"""Shared SQLite connection and schema scaffolding for Tile's durable stores."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

_IN_MEMORY_DATABASE = ":memory:"


def resolve_connection_target(
    *,
    database_path: Path | str | None,
    in_memory: bool,
) -> str:
    """Return the SQLite connection target for file or in-memory mode."""

    if in_memory:
        return _IN_MEMORY_DATABASE
    if database_path is None:
        raise ValueError("database_path is required unless in_memory=True.")
    return str(Path(database_path).expanduser())


@contextmanager
def immediate_transaction(connection: sqlite3.Connection) -> Iterator[None]:
    """Run a block inside an immediate SQLite write transaction."""

    connection.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        connection.rollback()
        raise
    else:
        connection.commit()


def initialize_schema(
    connection: sqlite3.Connection,
    *,
    version_key: str,
    expected_version: str,
    store_label: str,
    schema_error: type[RuntimeError],
    create_schema: Callable[[], None],
) -> None:
    """Create current schema objects or reject unsupported schema versions.

    Each store owns one version key in the shared ``tile_meta`` table, so
    multiple stores can coexist in a single database file.
    """

    with immediate_transaction(connection):
        _create_meta_table(connection)
        stored_version = _read_schema_version(connection, version_key)
        if stored_version not in (None, expected_version):
            raise schema_error(
                f"Unsupported SQLite {store_label} schema version: "
                f"{stored_version}. Expected {expected_version}."
            )
        create_schema()
        if stored_version is None:
            _write_schema_version(connection, version_key, expected_version)


def _create_meta_table(connection: sqlite3.Connection) -> None:
    """Create the database-wide metadata table when absent."""

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS tile_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )


def _read_schema_version(
    connection: sqlite3.Connection,
    version_key: str,
) -> str | None:
    """Return the schema version stored under one store's version key."""

    row = connection.execute(
        "SELECT value FROM tile_meta WHERE key = ?",
        (version_key,),
    ).fetchone()
    version_row = cast("tuple[str] | None", row)
    if version_row is None:
        return None
    return version_row[0]


def _write_schema_version(
    connection: sqlite3.Connection,
    version_key: str,
    version: str,
) -> None:
    """Record a store's schema version in shared database metadata."""

    connection.execute(
        """
        INSERT INTO tile_meta (key, value)
        VALUES (?, ?)
        """,
        (version_key, version),
    )
