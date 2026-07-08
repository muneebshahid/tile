"""Tests for the default directory listing tool."""

from collections.abc import Callable
from pathlib import Path

import pytest

import ori.tools.ls as ls
import ori.tools.support.truncation as truncation
from ori.tools.ls import LsDetails
from ori.types.tools import ToolResult, ToolTextContent


def test_ls_schema_requires_no_arguments() -> None:
    """Allow callers to omit path and limit."""

    assert ls.tool.input_schema["required"] == []


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

    tool_result = await ls.fn(path=str(populated_directory), limit=10, cwd=Path.cwd())
    result = _text(tool_result)

    assert result.splitlines() == ["README.md", "src/", "uv.lock"]
    assert tool_result.details is None


@pytest.mark.asyncio
async def test_ls_resolves_relative_path_against_supplied_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve relative listing paths against the supplied tool cwd."""

    project = tmp_path / "project"
    other = tmp_path / "other"
    project.mkdir()
    other.mkdir()
    _create_file(project / "sample.txt")
    monkeypatch.chdir(other)

    result = _text(await ls.fn(path=".", limit=10, cwd=project))

    assert result == "sample.txt"


@pytest.mark.asyncio
async def test_ls_uses_cwd_when_path_is_omitted(tmp_path: Path) -> None:
    """List the supplied working directory when callers omit path."""

    _create_file(tmp_path / "sample.txt")

    tool_result = await ls.fn(limit=10, cwd=tmp_path)
    result = _text(tool_result)

    assert result == "sample.txt"
    assert tool_result.details is None


@pytest.mark.asyncio
async def test_ls_respects_limit_after_sorting_entries(
    unsorted_directory: Path,
) -> None:
    """Return only the first sorted entries up to the requested limit."""

    tool_result = await ls.fn(path=str(unsorted_directory), limit=2, cwd=Path.cwd())
    result = _text(tool_result)

    assert result.splitlines() == [
        "a.txt",
        "b.txt",
        "",
        "[2 entries limit reached. Use limit=4 for more]",
    ]
    details = _ls_details(tool_result)
    assert details.output.output_lines == 2
    assert details.output.total_lines == 3
    assert details.output.truncated is True
    assert details.output.truncated_by == "lines"
    assert details.output.keep == "head"
    assert details.output.max_lines == 2


@pytest.mark.asyncio
async def test_ls_clamps_limit_to_one(unsorted_directory: Path) -> None:
    """Keep entry limits positive when callers pass a low limit."""

    result = _text(await ls.fn(path=str(unsorted_directory), limit=0, cwd=Path.cwd()))

    assert result.splitlines() == [
        "a.txt",
        "",
        "[1 entries limit reached. Use limit=2 for more]",
    ]


@pytest.mark.asyncio
async def test_ls_reports_byte_limit(long_directory: Callable[[int], Path]) -> None:
    """Report byte truncation when the listing output exceeds 50KB."""

    path = long_directory(270)

    tool_result = await ls.fn(path=str(path), limit=500, cwd=Path.cwd())
    result = _text(tool_result)
    notice = "\n\n[50.0KB limit reached. Directory has 270 entries]"
    body = result.removesuffix(notice)

    assert result.endswith(notice)
    assert len(body.encode("utf-8")) <= truncation.OUTPUT_BYTE_LIMIT
    details = _ls_details(tool_result)
    assert details.output.output_lines < details.output.total_lines
    assert details.output.total_lines == 270
    assert details.output.truncated is True
    assert details.output.truncated_by == "bytes"
    assert details.output.max_bytes == truncation.OUTPUT_BYTE_LIMIT
    assert details.output.output_bytes <= truncation.OUTPUT_BYTE_LIMIT
    assert details.output.total_bytes > truncation.OUTPUT_BYTE_LIMIT


@pytest.mark.asyncio
async def test_ls_reports_first_truncation_boundary(
    long_directory: Callable[[int], Path],
) -> None:
    """Report only the first truncation boundary reached by formatted output."""

    path = long_directory(300)

    result = _text(await ls.fn(path=str(path), limit=260, cwd=Path.cwd()))

    assert result.endswith("\n\n[50.0KB limit reached. Directory has 300 entries]")


@pytest.mark.asyncio
async def test_ls_appends_slash_to_directories(
    directory_with_child_directory: Path,
) -> None:
    """Mark directory entries with a trailing slash and leave files unchanged."""

    result = _text(
        await ls.fn(
            path=str(directory_with_child_directory),
            limit=10,
            cwd=Path.cwd(),
        )
    )

    assert result.splitlines() == ["file.txt", "folder/"]


@pytest.mark.asyncio
async def test_ls_includes_dotfiles_and_dot_directories(
    hidden_entries_directory: Path,
) -> None:
    """Include hidden files and hidden directories in directory listings."""

    result = _text(
        await ls.fn(path=str(hidden_entries_directory), limit=10, cwd=Path.cwd())
    )

    assert result.splitlines() == [".hidden-dir/", ".hidden-file"]


@pytest.mark.asyncio
async def test_ls_sorts_entries_case_insensitively(
    mixed_case_directory: Path,
) -> None:
    """Sort entries alphabetically without separating upper and lower case names."""

    result = _text(
        await ls.fn(path=str(mixed_case_directory), limit=10, cwd=Path.cwd())
    )

    assert result.splitlines() == ["Alpha.txt", "beta.txt", "charlie.txt"]


@pytest.mark.asyncio
async def test_ls_reports_empty_directory(tmp_path: Path) -> None:
    """Return an explicit marker for empty directories."""

    tool_result = await ls.fn(path=str(tmp_path), limit=10, cwd=Path.cwd())
    result = _text(tool_result)

    assert result == "(empty directory)"
    assert tool_result.details is None


@pytest.mark.asyncio
async def test_ls_raises_when_path_does_not_exist(tmp_path: Path) -> None:
    """Raise filesystem errors so the agent can mark tool execution as failed."""

    with pytest.raises(FileNotFoundError):
        await ls.fn(path=str(tmp_path / "missing"), limit=10, cwd=Path.cwd())


@pytest.mark.asyncio
async def test_ls_raises_when_path_is_not_directory(tmp_path: Path) -> None:
    """Raise when callers pass a file instead of a directory."""

    file_path = tmp_path / "file.txt"
    _create_file(file_path)

    with pytest.raises(NotADirectoryError):
        await ls.fn(path=str(file_path), limit=10, cwd=Path.cwd())


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


def _text(result: ToolResult) -> str:
    """Return the single text block from a tool result."""

    assert len(result.content) == 1
    content = result.content[0]
    assert isinstance(content, ToolTextContent)
    return content.text


def _ls_details(result: ToolResult) -> LsDetails:
    """Return ls details from a tool result."""

    assert isinstance(result.details, LsDetails)
    return result.details
