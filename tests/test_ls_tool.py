"""Tests for the default directory listing tool."""

from pathlib import Path

import pytest

import agent.tools.ls as ls


def test_ls_schema_requires_only_path() -> None:
    """Require only path so callers can omit optional limit."""

    assert ls.tool.input_schema["required"] == ["path"]


@pytest.mark.asyncio
async def test_ls_returns_all_directory_entries(tmp_path: Path) -> None:
    """Return every file and directory name when the result is under the limit."""

    _create_file(tmp_path / "README.md")
    _create_file(tmp_path / "uv.lock")
    _create_directory(tmp_path / "src")

    result = await ls.fn(path=str(tmp_path), limit=10)

    assert result.splitlines() == ["README.md", "src/", "uv.lock"]


@pytest.mark.asyncio
async def test_ls_respects_limit_after_sorting_entries(tmp_path: Path) -> None:
    """Return only the first sorted entries up to the requested limit."""

    _create_file(tmp_path / "b.txt")
    _create_file(tmp_path / "a.txt")
    _create_file(tmp_path / "c.txt")

    result = await ls.fn(path=str(tmp_path), limit=2)

    assert result.splitlines() == [
        "a.txt",
        "b.txt",
        "",
        "[2 entries limit reached. Use limit=4 for more]",
    ]


def test_truncate_to_byte_limit_keeps_complete_lines() -> None:
    """Truncate over-limit output at line boundaries instead of mid-line."""

    assert ls._truncate_to_byte_limit("a.txt\nb.txt", byte_limit=11) == (
        "a.txt\nb.txt",
        False,
    )
    assert ls._truncate_to_byte_limit("a.txt\nb.txt", byte_limit=10) == (
        "a.txt",
        True,
    )


@pytest.mark.asyncio
async def test_ls_reports_byte_limit(tmp_path: Path) -> None:
    """Report byte truncation when the listing output exceeds 50KB."""

    _create_long_file_names(tmp_path, count=270)

    result = await ls.fn(path=str(tmp_path), limit=500)
    notice = "\n\n[50.0KB limit reached]"
    entries = ls._list_directory_entries(str(tmp_path))
    body = result.removesuffix(notice)

    assert result.endswith(notice)
    assert len("\n".join(entries).encode("utf-8")) > ls.BYTE_LIMIT
    assert len(body.encode("utf-8")) <= ls.BYTE_LIMIT


@pytest.mark.asyncio
async def test_ls_reports_entry_and_byte_limits(tmp_path: Path) -> None:
    """Report entry and byte truncation together when both limits are reached."""

    _create_long_file_names(tmp_path, count=300)

    result = await ls.fn(path=str(tmp_path), limit=260)

    assert result.endswith(
        "\n\n[260 entries limit reached. Use limit=520 for more. 50.0KB limit reached]"
    )


@pytest.mark.asyncio
async def test_ls_appends_slash_to_directories(tmp_path: Path) -> None:
    """Mark directory entries with a trailing slash and leave files unchanged."""

    _create_file(tmp_path / "file.txt")
    _create_directory(tmp_path / "folder")

    result = await ls.fn(path=str(tmp_path), limit=10)

    assert result.splitlines() == ["file.txt", "folder/"]


@pytest.mark.asyncio
async def test_ls_includes_dotfiles_and_dot_directories(tmp_path: Path) -> None:
    """Include hidden files and hidden directories in directory listings."""

    _create_file(tmp_path / ".hidden-file")
    _create_directory(tmp_path / ".hidden-dir")

    result = await ls.fn(path=str(tmp_path), limit=10)

    assert result.splitlines() == [".hidden-dir/", ".hidden-file"]


@pytest.mark.asyncio
async def test_ls_sorts_entries_case_insensitively(tmp_path: Path) -> None:
    """Sort entries alphabetically without separating upper and lower case names."""

    _create_file(tmp_path / "beta.txt")
    _create_file(tmp_path / "Alpha.txt")
    _create_file(tmp_path / "charlie.txt")

    result = await ls.fn(path=str(tmp_path), limit=10)

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
