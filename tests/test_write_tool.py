"""Tests for the default file write tool."""

from pathlib import Path

import pytest

import agent.tools.write as write
from ai.types.tools import ToolResult, ToolTextContent


def test_write_schema_requires_path_and_content() -> None:
    """Require the file path and replacement content."""

    assert write.tool.input_schema["required"] == ["path", "content"]


def test_write_schema_exposes_write_controls() -> None:
    """Expose only path and content inputs."""

    properties = write.tool.input_schema["properties"]

    assert write.tool.name == "write"
    assert isinstance(properties, dict)
    assert set(properties) == {"path", "content"}


@pytest.mark.asyncio
async def test_write_creates_file_and_parent_directories(tmp_path: Path) -> None:
    """Create parent directories and write UTF-8 content to a new file."""

    file_path = tmp_path / "nested" / "sample.txt"

    result = _text(await write.fn(path=str(file_path), content="hello", cwd=Path.cwd()))

    assert file_path.read_text(encoding="utf-8") == "hello"
    assert result == f"Successfully wrote 5 bytes to {file_path}"


@pytest.mark.asyncio
async def test_write_overwrites_existing_file(tmp_path: Path) -> None:
    """Overwrite existing files with the supplied content."""

    file_path = tmp_path / "sample.txt"
    file_path.write_text("old", encoding="utf-8")

    result = _text(await write.fn(path=str(file_path), content="new", cwd=Path.cwd()))

    assert file_path.read_text(encoding="utf-8") == "new"
    assert result == f"Successfully wrote 3 bytes to {file_path}"


@pytest.mark.asyncio
async def test_write_reports_utf8_byte_count(tmp_path: Path) -> None:
    """Report the real UTF-8 byte count of written content."""

    file_path = tmp_path / "sample.txt"

    result = _text(await write.fn(path=str(file_path), content="é", cwd=Path.cwd()))

    assert result == f"Successfully wrote 2 bytes to {file_path}"


@pytest.mark.asyncio
async def test_write_resolves_relative_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve relative paths against the current working directory."""

    monkeypatch.chdir(tmp_path)

    result = _text(
        await write.fn(path="relative/sample.txt", content="hello", cwd=tmp_path)
    )

    file_path = tmp_path / "relative" / "sample.txt"
    assert file_path.read_text(encoding="utf-8") == "hello"
    assert result == f"Successfully wrote 5 bytes to {file_path}"


@pytest.mark.asyncio
async def test_write_resolves_relative_path_against_supplied_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve relative write paths against the supplied tool cwd."""

    project = tmp_path / "project"
    other = tmp_path / "other"
    project.mkdir()
    other.mkdir()
    monkeypatch.chdir(other)

    result = _text(
        await write.fn(path="relative/sample.txt", content="hello", cwd=project)
    )

    file_path = project / "relative" / "sample.txt"
    assert file_path.read_text(encoding="utf-8") == "hello"
    assert not (other / "relative" / "sample.txt").exists()
    assert result == f"Successfully wrote 5 bytes to {file_path}"


@pytest.mark.asyncio
async def test_write_expands_home_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expand home-directory markers in write paths."""

    monkeypatch.setenv("HOME", str(tmp_path))

    result = _text(await write.fn(path="~/sample.txt", content="hello", cwd=Path.cwd()))

    file_path = tmp_path / "sample.txt"
    assert file_path.read_text(encoding="utf-8") == "hello"
    assert result == f"Successfully wrote 5 bytes to {file_path}"


@pytest.mark.asyncio
async def test_write_raises_when_parent_path_is_file(tmp_path: Path) -> None:
    """Raise filesystem errors so the agent can mark write failures."""

    parent = tmp_path / "parent"
    parent.write_text("not a directory", encoding="utf-8")

    with pytest.raises(FileExistsError):
        await write.fn(
            path=str(parent / "sample.txt"),
            content="hello",
            cwd=Path.cwd(),
        )


def _text(result: ToolResult) -> str:
    """Return the single text block from a tool result."""

    assert len(result.content) == 1
    content = result.content[0]
    assert isinstance(content, ToolTextContent)
    return content.text
