"""Shared command execution fakes for tool tests."""

from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock


def executable_lookup(
    command_name: str, resolved_path: str
) -> Callable[[str], str | None]:
    """Return a lookup fake that resolves one command name."""

    def _lookup(command: str) -> str | None:
        """Return the configured path when the command name matches."""

        if command == command_name:
            return resolved_path
        return None

    return _lookup


def no_executable(command: str) -> None:
    """Return no command path for availability checks."""

    _ = command
    return None


def captured_args(execution: AsyncMock) -> list[str]:
    """Return command args captured by a fake execution call."""

    execution.assert_awaited_once()
    await_args = execution.await_args
    assert await_args is not None
    args = await_args.args
    assert isinstance(args[1], list)
    return args[1]


def captured_cwd(execution: AsyncMock) -> Path | None:
    """Return the cwd captured by a fake execution call."""

    execution.assert_awaited_once()
    await_args = execution.await_args
    assert await_args is not None
    cwd = await_args.kwargs.get("cwd")
    assert isinstance(cwd, Path) or cwd is None
    return cwd
