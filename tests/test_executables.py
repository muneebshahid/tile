"""Tests for external executable availability helpers."""

import sys
from pathlib import Path

import pytest

import ori.tools.support.executables as executables
from tests.support.command_mocks import executable_lookup, no_executable


def test_require_executable_returns_resolved_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return the executable path when the command is available."""

    monkeypatch.setattr(
        executables.shutil,
        "which",
        executable_lookup("rg", "/usr/bin/rg"),
    )

    assert executables.require_executable("rg", "ripgrep (rg)") == "/usr/bin/rg"


def test_require_executable_raises_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise a clear error when the command is unavailable."""

    monkeypatch.setattr(executables.shutil, "which", no_executable)

    with pytest.raises(RuntimeError, match="ripgrep"):
        executables.require_executable("rg", "ripgrep (rg)")


@pytest.mark.asyncio
async def test_execute_returns_process_stdout() -> None:
    """Return captured stdout from a successful process."""

    result = await executables.execute(
        sys.executable,
        ["-c", "print('out')"],
        cwd=Path.cwd(),
    )

    assert result == "out\n"


@pytest.mark.asyncio
async def test_execute_runs_process_from_supplied_cwd(tmp_path: Path) -> None:
    """Run a process from the supplied working directory."""

    result = await executables.execute(
        sys.executable,
        ["-c", "from pathlib import Path; print(Path.cwd())"],
        cwd=tmp_path,
    )

    assert result == f"{tmp_path}\n"


@pytest.mark.asyncio
async def test_execute_raises_on_disallowed_exit_code() -> None:
    """Raise stderr output when a process exits with a disallowed code."""

    with pytest.raises(RuntimeError, match="boom"):
        await executables.execute(
            sys.executable,
            ["-c", "import sys; print('boom', file=sys.stderr); sys.exit(2)"],
            cwd=Path.cwd(),
        )


@pytest.mark.asyncio
async def test_execute_allows_configured_exit_code() -> None:
    """Return stdout when a process exits with an explicitly allowed code."""

    result = await executables.execute(
        sys.executable,
        ["-c", "import sys; print('empty'); sys.exit(1)"],
        allowed_exit_codes=(0, 1),
        cwd=Path.cwd(),
    )

    assert result == "empty\n"
