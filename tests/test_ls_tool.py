"""Tests for the default directory listing tool."""

from collections.abc import Callable
from pathlib import Path

import pytest

import agent.tools.ls as ls
import agent.tools.truncation as truncation


def test_ls_schema_requires_only_path() -> None:
    """Require only path so callers can omit optional limit."""

    assert ls.tool.input_schema["required"] == ["path"]


@pytest.fixture
def populated_directory(tmp_path: Path) -> Path:
    """Create a directory with representative file and directory entries."""

    _create_file(tmp_path / "README.md")
    _create_file(tmp_path / "uv.lock")
    _create_directory(tmp_path / "src")
    return tmp_path


@pytest.fixture
def unsorted_directory(tmp_path: Path) -> Path:
    """Create a directory whose entries need sorting before limiting."""

    _create_file(tmp_path / "b.txt")
    _create_file(tmp_path / "a.txt")
    _create_file(tmp_path / "c.txt")
    return tmp_path


@pytest.fixture
def long_directory(tmp_path: Path) -> Callable[[int], Path]:
    """Return a factory for directories with long file names."""

    def create(count: int) -> Path:
        """Create long file names and return the populated directory."""

        _create_long_file_names(tmp_path, count=count)
        return tmp_path

    return create


@pytest.fixture
def directory_with_child_directory(tmp_path: Path) -> Path:
    """Create a directory containing one file and one child directory."""

    _create_file(tmp_path / "file.txt")
    _create_directory(tmp_path / "folder")
    return tmp_path


@pytest.fixture
def hidden_entries_directory(tmp_path: Path) -> Path:
    """Create a directory containing hidden file and directory entries."""

    _create_file(tmp_path / ".hidden-file")
    _create_directory(tmp_path / ".hidden-dir")
    return tmp_path


@pytest.fixture
def mixed_case_directory(tmp_path: Path) -> Path:
    """Create a directory containing mixed-case file names."""

    _create_file(tmp_path / "beta.txt")
    _create_file(tmp_path / "Alpha.txt")
    _create_file(tmp_path / "charlie.txt")
    return tmp_path


@pytest.mark.asyncio
async def test_ls_returns_all_directory_entries(populated_directory: Path) -> None:
    """Return every file and directory name when the result is under the limit."""

    result = await ls.fn(path=str(populated_directory), limit=10)

    assert result.splitlines() == ["README.md", "src/", "uv.lock"]


@pytest.mark.asyncio
async def test_ls_respects_limit_after_sorting_entries(
    unsorted_directory: Path,
) -> None:
    """Return only the first sorted entries up to the requested limit."""

    result = await ls.fn(path=str(unsorted_directory), limit=2)

    assert result.splitlines() == [
        "a.txt",
        "b.txt",
        "",
        "[2 entries limit reached. Use limit=4 for more]",
    ]


@pytest.mark.asyncio
async def test_ls_reports_byte_limit(long_directory: Callable[[int], Path]) -> None:
    """Report byte truncation when the listing output exceeds 50KB."""

    path = long_directory(270)

    result = await ls.fn(path=str(path), limit=500)
    notice = "\n\n[50.0KB limit reached]"
    entries = ls._list_directory_entries(str(path))
    body = result.removesuffix(notice)

    assert result.endswith(notice)
    assert len("\n".join(entries).encode("utf-8")) > truncation.OUTPUT_BYTE_LIMIT
    assert len(body.encode("utf-8")) <= truncation.OUTPUT_BYTE_LIMIT


@pytest.mark.asyncio
async def test_ls_reports_entry_and_byte_limits(
    long_directory: Callable[[int], Path],
) -> None:
    """Report entry and byte truncation together when both limits are reached."""

    path = long_directory(300)

    result = await ls.fn(path=str(path), limit=260)

    assert result.endswith(
        "\n\n[260 entries limit reached. Use limit=520 for more. 50.0KB limit reached]"
    )


@pytest.mark.asyncio
async def test_ls_appends_slash_to_directories(
    directory_with_child_directory: Path,
) -> None:
    """Mark directory entries with a trailing slash and leave files unchanged."""

    result = await ls.fn(path=str(directory_with_child_directory), limit=10)

    assert result.splitlines() == ["file.txt", "folder/"]


@pytest.mark.asyncio
async def test_ls_includes_dotfiles_and_dot_directories(
    hidden_entries_directory: Path,
) -> None:
    """Include hidden files and hidden directories in directory listings."""

    result = await ls.fn(path=str(hidden_entries_directory), limit=10)

    assert result.splitlines() == [".hidden-dir/", ".hidden-file"]


@pytest.mark.asyncio
async def test_ls_sorts_entries_case_insensitively(
    mixed_case_directory: Path,
) -> None:
    """Sort entries alphabetically without separating upper and lower case names."""

    result = await ls.fn(path=str(mixed_case_directory), limit=10)

    assert result.splitlines() == ["Alpha.txt", "beta.txt", "charlie.txt"]


@pytest.mark.asyncio
async def test_ls_reports_empty_directory(tmp_path: Path) -> None:
    """Return an explicit marker for empty directories."""

    result = await ls.fn(path=str(tmp_path), limit=10)

    assert result == "(empty directory)"


@pytest.mark.asyncio
async def test_ls_raises_when_path_does_not_exist(tmp_path: Path) -> None:
    """Raise filesystem errors so the agent can mark tool execution as failed."""

    with pytest.raises(FileNotFoundError):
        await ls.fn(path=str(tmp_path / "missing"), limit=10)


@pytest.mark.asyncio
async def test_ls_raises_when_path_is_not_directory(tmp_path: Path) -> None:
    """Raise when callers pass a file instead of a directory."""

    file_path = tmp_path / "file.txt"
    _create_file(file_path)

    with pytest.raises(NotADirectoryError):
        await ls.fn(path=str(file_path), limit=10)


def _create_file(path: Path) -> None:
    """Create a small test file."""

    path.write_text("", encoding="utf-8")


def _create_directory(path: Path) -> None:
    """Create a test directory."""

    path.mkdir()


def _create_long_file_names(path: Path, count: int) -> None:
    """Create enough long file names to exceed listing byte limits."""

    for index in range(count):
        _create_file(path / f"{index:03d}-{'x' * 196}.txt")
