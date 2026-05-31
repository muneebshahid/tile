"""Tests for the default shell command tool scaffold."""

from pathlib import Path

import pytest

import agent.tools.bash as bash
from ai.types.tools import ToolResult, ToolTextContent


def test_bash_schema_requires_only_command() -> None:
    """Require only the command so callers can omit optional timeout."""

    assert bash.tool.input_schema["required"] == ["command"]


def test_bash_schema_exposes_command_controls() -> None:
    """Expose shell command inputs without execution-injected fields."""

    properties = bash.tool.input_schema["properties"]

    assert bash.tool.name == "bash"
    assert isinstance(properties, dict)
    assert set(properties) == {"command", "timeout"}


@pytest.mark.asyncio
async def test_fn_runs_command_from_supplied_cwd(tmp_path: Path) -> None:
    """Run shell commands from the injected working directory."""

    (tmp_path / "sample.txt").write_text("content", encoding="utf-8")

    result = _text(await bash.fn(command="pwd && ls", cwd=tmp_path))

    assert result.splitlines() == [str(tmp_path.resolve()), "sample.txt"]


@pytest.mark.asyncio
async def test_fn_combines_stderr_with_stdout(tmp_path: Path) -> None:
    """Return stdout and stderr in one text result."""

    result = _text(
        await bash.fn(command="printf 'out'; printf 'err' >&2", cwd=tmp_path)
    )

    assert result == "outerr"


@pytest.mark.asyncio
async def test_fn_returns_no_output_marker(tmp_path: Path) -> None:
    """Return an explicit marker when a successful command emits nothing."""

    result = _text(await bash.fn(command="true", cwd=tmp_path))

    assert result == "(no output)"


@pytest.mark.asyncio
async def test_fn_raises_with_output_on_nonzero_exit(tmp_path: Path) -> None:
    """Raise command output and exit code when the command fails."""

    with pytest.raises(RuntimeError) as error:
        await bash.fn(command="printf 'boom'; exit 7", cwd=tmp_path)

    assert str(error.value) == "boom\n\nCommand exited with code 7"


@pytest.mark.asyncio
async def test_fn_raises_status_only_when_failed_command_has_no_output(
    tmp_path: Path,
) -> None:
    """Avoid adding a no-output marker to failed commands."""

    with pytest.raises(RuntimeError) as error:
        await bash.fn(command="exit 7", cwd=tmp_path)

    assert str(error.value) == "Command exited with code 7"


@pytest.mark.asyncio
async def test_fn_raises_with_timeout_status(tmp_path: Path) -> None:
    """Terminate commands that exceed the supplied timeout."""

    with pytest.raises(RuntimeError) as error:
        await bash.fn(command="printf 'start'; sleep 2", timeout=0.1, cwd=tmp_path)

    assert str(error.value) == "start\n\nCommand timed out after 0.1 seconds"


def _text(result: ToolResult) -> str:
    """Return the single text block from a tool result."""

    assert len(result.content) == 1
    content = result.content[0]
    assert isinstance(content, ToolTextContent)
    return content.text
