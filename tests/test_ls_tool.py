"""Tests for the default directory listing tool."""

from pathlib import Path

import pytest

from agent.tools.ls import fn


@pytest.mark.asyncio
async def test_ls_returns_all_directory_entries(tmp_path: Path) -> None:
    """Return every file and directory name when the result is under the limit."""

    _create_file(tmp_path / "README.md")
    _create_file(tmp_path / "uv.lock")
    _create_directory(tmp_path / "src")

    result = await fn(path=str(tmp_path), limit=10)

    assert result.splitlines() == ["README.md", "src/", "uv.lock"]


@pytest.mark.asyncio
async def test_ls_respects_limit_after_sorting_entries(tmp_path: Path) -> None:
    """Return only the first sorted entries up to the requested limit."""

    _create_file(tmp_path / "b.txt")
    _create_file(tmp_path / "a.txt")
    _create_file(tmp_path / "c.txt")

    result = await fn(path=str(tmp_path), limit=2)

    assert result.splitlines() == [
        "a.txt",
        "b.txt",
        "",
        "[2 entries limit reached. Use limit=4 for more]",
    ]


@pytest.mark.asyncio
async def test_ls_appends_slash_to_directories(tmp_path: Path) -> None:
    """Mark directory entries with a trailing slash and leave files unchanged."""

    _create_file(tmp_path / "file.txt")
    _create_directory(tmp_path / "folder")

    result = await fn(path=str(tmp_path), limit=10)

    assert result.splitlines() == ["file.txt", "folder/"]


@pytest.mark.asyncio
async def test_ls_includes_dotfiles_and_dot_directories(tmp_path: Path) -> None:
    """Include hidden files and hidden directories in directory listings."""

    _create_file(tmp_path / ".hidden-file")
    _create_directory(tmp_path / ".hidden-dir")

    result = await fn(path=str(tmp_path), limit=10)

    assert result.splitlines() == [".hidden-dir/", ".hidden-file"]


@pytest.mark.asyncio
async def test_ls_sorts_entries_case_insensitively(tmp_path: Path) -> None:
    """Sort entries alphabetically without separating upper and lower case names."""

    _create_file(tmp_path / "beta.txt")
    _create_file(tmp_path / "Alpha.txt")
    _create_file(tmp_path / "charlie.txt")

    result = await fn(path=str(tmp_path), limit=10)

    assert result.splitlines() == ["Alpha.txt", "beta.txt", "charlie.txt"]


@pytest.mark.asyncio
async def test_ls_reports_empty_directory(tmp_path: Path) -> None:
    """Return an explicit marker for empty directories."""

    result = await fn(path=str(tmp_path), limit=10)

    assert result == "(empty directory)"


def _create_file(path: Path) -> None:
    """Create a small test file."""

    path.write_text("", encoding="utf-8")


def _create_directory(path: Path) -> None:
    """Create a test directory."""

    path.mkdir()
