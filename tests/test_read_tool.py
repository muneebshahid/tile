"""Tests for the default text file read tool."""

from pathlib import Path

import pytest

import agent.tools.read as read
import agent.tools.truncation as truncation


def test_read_schema_requires_only_path() -> None:
    """Require only path so callers can omit optional offset and limit."""

    assert read.tool.input_schema["required"] == ["path"]


def test_read_schema_exposes_text_read_controls() -> None:
    """Expose the text file read inputs from the Pi-compatible schema."""

    properties = read.tool.input_schema["properties"]

    assert read.tool.name == "read"
    assert isinstance(properties, dict)
    assert set(properties) == {"path", "offset", "limit"}


@pytest.mark.asyncio
async def test_read_returns_file_contents(tmp_path: Path) -> None:
    """Return full file contents when no limit is reached."""

    file_path = _write_lines(tmp_path / "sample.txt", ["one", "two", "three"])

    result = await read.fn(path=str(file_path))

    assert result == "one\ntwo\nthree"


@pytest.mark.asyncio
async def test_read_starts_from_one_indexed_offset(tmp_path: Path) -> None:
    """Treat offset as a 1-indexed starting line number."""

    file_path = _write_lines(tmp_path / "sample.txt", ["one", "two", "three"])

    result = await read.fn(path=str(file_path), offset=2)

    assert result == "two\nthree"


@pytest.mark.asyncio
async def test_read_reports_remaining_lines_after_limit(tmp_path: Path) -> None:
    """Honor a caller limit before automatic truncation."""

    file_path = _write_numbered_lines(tmp_path / "sample.txt", count=100)

    result = await read.fn(path=str(file_path), limit=10)

    assert result == (
        "\n".join(f"line {index}" for index in range(1, 11))
        + "\n\n[90 more lines in file. Use offset=11 to continue.]"
    )


@pytest.mark.asyncio
async def test_read_handles_offset_and_limit_together(tmp_path: Path) -> None:
    """Apply offset before applying the caller line limit."""

    file_path = _write_numbered_lines(tmp_path / "sample.txt", count=100)

    result = await read.fn(path=str(file_path), offset=41, limit=20)

    assert result == (
        "\n".join(f"line {index}" for index in range(41, 61))
        + "\n\n[40 more lines in file. Use offset=61 to continue.]"
    )


@pytest.mark.asyncio
async def test_read_raises_when_offset_is_beyond_file(tmp_path: Path) -> None:
    """Raise a clear error when offset is beyond the file length."""

    file_path = _write_lines(tmp_path / "sample.txt", ["one", "two", "three"])

    with pytest.raises(RuntimeError, match="Offset 100 is beyond end of file"):
        await read.fn(path=str(file_path), offset=100)


@pytest.mark.asyncio
async def test_read_reports_line_truncation(tmp_path: Path) -> None:
    """Append a continuation notice when automatic line truncation occurs."""

    file_path = _write_numbered_lines(
        tmp_path / "sample.txt",
        count=truncation.OUTPUT_LINE_LIMIT + 1,
    )

    result = await read.fn(path=str(file_path))

    assert result.endswith(
        "\n\n[Showing lines 1-2000 of 2001. Use offset=2001 to continue.]"
    )


@pytest.mark.asyncio
async def test_read_reports_byte_truncation(tmp_path: Path) -> None:
    """Append a continuation notice when automatic byte truncation occurs."""

    file_path = _write_lines(tmp_path / "sample.txt", ["x" * 200 for _ in range(500)])

    result = await read.fn(path=str(file_path))

    assert "50.0KB limit" in result
    assert result.endswith("to continue.]")


@pytest.mark.asyncio
async def test_read_reports_first_line_exceeds_byte_limit(tmp_path: Path) -> None:
    """Return a bash fallback notice when the first selected line is too large."""

    file_path = _write_lines(tmp_path / "sample.txt", ["x" * (50 * 1024 + 1)])

    result = await read.fn(path=str(file_path))

    assert result == (
        f"[Line 1 is 50.0KB, exceeds 50.0KB limit. Use bash: "
        f"sed -n '1p' {file_path} | head -c 51200]"
    )


@pytest.mark.asyncio
async def test_read_raises_when_path_does_not_exist(tmp_path: Path) -> None:
    """Raise filesystem errors so the agent can mark tool execution as failed."""

    with pytest.raises(FileNotFoundError):
        await read.fn(path=str(tmp_path / "missing.txt"))


def _write_lines(path: Path, lines: list[str]) -> Path:
    """Write lines to a UTF-8 test file and return its path."""

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_numbered_lines(path: Path, count: int) -> Path:
    """Write numbered lines to a UTF-8 test file and return its path."""

    return _write_lines(path, [f"line {index}" for index in range(1, count + 1)])
