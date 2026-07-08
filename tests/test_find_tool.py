"""Tests for the default file path search tool."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import ori.tools.support.executables as executables
import ori.tools.find as find
import ori.tools.support.truncation as truncation
from ori.tools.find import FindDetails
from ori.types.tools import ToolResult, ToolTextContent
from tests.support.command_mocks import (
    captured_args,
    captured_cwd,
    executable_lookup,
    no_executable,
)


def test_find_schema_requires_only_pattern() -> None:
    """Require only the glob pattern so callers can omit optional controls."""

    assert find.tool.input_schema["required"] == ["pattern"]


def test_find_schema_exposes_path_search_controls() -> None:
    """Expose the path-search inputs without execution-specific fields."""

    properties = find.tool.input_schema["properties"]

    assert find.tool.name == "find"
    assert isinstance(properties, dict)
    assert set(properties) == {"pattern", "path", "limit"}


@pytest.fixture
def fd_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the fd executable available to fn-level tests."""

    monkeypatch.setattr(
        executables.shutil,
        "which",
        executable_lookup("fd", "/usr/bin/fd"),
    )


@pytest.fixture
def fd_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the fd executable unavailable to fn-level tests."""

    monkeypatch.setattr(executables.shutil, "which", no_executable)


@pytest.fixture
def execution(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Patch command execution with an async mock for fn-level tests."""

    execution_mock = AsyncMock()
    monkeypatch.setattr(find, "execute", execution_mock)
    return execution_mock


@pytest.mark.asyncio
@pytest.mark.usefixtures("fd_available")
async def test_fn_uses_default_file_search_flags(
    execution: AsyncMock,
) -> None:
    """Build default fd arguments and return formatted file path results."""

    execution.return_value = "./ori/tools/find.py\n"

    tool_result = await find.fn(pattern="*.py", cwd=Path.cwd())
    result = _text(tool_result)

    assert result == "ori/tools/find.py"
    assert tool_result.details is None
    execution.assert_awaited_once_with(
        "/usr/bin/fd",
        [
            "--glob",
            "--color=never",
            "--hidden",
            "--no-require-git",
            "--max-results",
            "1001",
            "--",
            "*.py",
            ".",
        ],
        cwd=Path.cwd(),
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("fd_available")
async def test_fn_resolves_search_path_against_supplied_cwd(
    execution: AsyncMock,
    tmp_path: Path,
) -> None:
    """Resolve relative search roots against the supplied tool cwd."""

    execution.return_value = "./ori/tools/find.py\n"

    result = _text(await find.fn(pattern="*.py", path="src", cwd=tmp_path))

    assert result == "ori/tools/find.py"
    assert captured_args(execution)[-1] == "src"
    assert captured_cwd(execution) == tmp_path


@pytest.mark.asyncio
@pytest.mark.usefixtures("fd_available")
async def test_fn_uses_full_path_for_path_patterns(
    execution: AsyncMock,
) -> None:
    """Match path-shaped glob patterns against full candidate paths."""

    execution.return_value = "./ori/tools/find.py\n"

    result = _text(
        await find.fn(pattern="ori/**/*.py", path=".", limit=25, cwd=Path.cwd())
    )

    assert result == "ori/tools/find.py"
    assert captured_args(execution) == [
        "--glob",
        "--color=never",
        "--hidden",
        "--no-require-git",
        "--max-results",
        "26",
        "--full-path",
        "--",
        "**/ori/**/*.py",
        ".",
    ]


@pytest.mark.asyncio
@pytest.mark.usefixtures("fd_available")
async def test_fn_normalizes_root_relative_full_path_pattern(
    execution: AsyncMock,
) -> None:
    """Treat leading-slash glob patterns as search-root-relative paths."""

    execution.return_value = "./ori/tools/find.py\n"

    result = _text(
        await find.fn(pattern="/tools/*.py", path=".", limit=25, cwd=Path.cwd())
    )

    assert result == "ori/tools/find.py"
    assert captured_args(execution)[-3:] == ["--", "**/tools/*.py", "."]


@pytest.mark.asyncio
@pytest.mark.usefixtures("fd_available")
async def test_fn_preserves_prefixed_full_path_pattern(
    execution: AsyncMock,
) -> None:
    """Do not double-prefix full-path glob patterns."""

    execution.return_value = "./ori/tools/find.py\n"

    result = _text(
        await find.fn(pattern="**/tools/*.py", path=".", limit=25, cwd=Path.cwd())
    )

    assert result == "ori/tools/find.py"
    assert captured_args(execution)[-3:] == ["--", "**/tools/*.py", "."]


@pytest.mark.asyncio
@pytest.mark.usefixtures("fd_available")
async def test_fn_clamps_limit_to_one(execution: AsyncMock) -> None:
    """Keep fd max-results positive even when callers pass a low limit."""

    execution.return_value = "./a.py\n./b.py\n"

    result = _text(await find.fn(pattern="*.py", limit=0, cwd=Path.cwd()))

    assert (
        result
        == "a.py\n\n[1 results limit reached. Use limit=2 for more, or refine pattern]"
    )
    assert captured_args(execution)[4:6] == ["--max-results", "2"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("fd_available")
async def test_fn_returns_no_matches_when_fd_output_is_empty(
    execution: AsyncMock,
) -> None:
    """Return a no-match message when fd emits no paths."""

    execution.return_value = ""

    tool_result = await find.fn(pattern="*.missing", cwd=Path.cwd())
    result = _text(tool_result)

    assert result == "No files found matching pattern"
    assert tool_result.details is None


@pytest.mark.asyncio
@pytest.mark.usefixtures("fd_missing")
async def test_fn_raises_when_fd_is_missing() -> None:
    """Raise a clear exception when fd is unavailable."""

    with pytest.raises(RuntimeError, match="fd"):
        await find.fn(pattern="*.py", cwd=Path.cwd())


@pytest.mark.asyncio
@pytest.mark.usefixtures("fd_available")
async def test_fn_normalizes_paths_and_reports_result_limit(
    execution: AsyncMock,
) -> None:
    """Normalize fd output paths and report when results reach the limit."""

    execution.return_value = (
        ".\\ori\\tools\\find.py\n./tests/test_find_tool.py\n./extra.py\n"
    )

    tool_result = await find.fn(pattern="*.py", limit=2, cwd=Path.cwd())
    result = _text(tool_result)

    assert result == (
        "ori/tools/find.py\ntests/test_find_tool.py\n\n"
        "[2 results limit reached. Use limit=4 for more, or refine pattern]"
    )
    details = _find_details(tool_result)
    assert details.output.truncated is True
    assert details.output.truncated_by == "lines"
    assert details.output.output_lines == 2
    assert details.output.total_lines == 3
    assert details.output.max_lines == 2


@pytest.mark.asyncio
@pytest.mark.usefixtures("fd_available")
async def test_fn_reports_result_limit_when_result_boundary_is_first(
    execution: AsyncMock,
) -> None:
    """Report the result limit when line count is the first truncation boundary."""

    execution.return_value = "./a.py\n./b.py\n"

    tool_result = await find.fn(pattern="*.py", limit=1, cwd=Path.cwd())
    result = _text(tool_result)

    assert result == (
        "a.py\n\n[1 results limit reached. Use limit=2 for more, or refine pattern]"
    )
    details = _find_details(tool_result)
    assert details.output.truncated_by == "lines"
    assert details.output.output_lines == 1
    assert details.output.total_lines == 2


@pytest.mark.asyncio
@pytest.mark.usefixtures("fd_available")
async def test_fn_reports_byte_limit(execution: AsyncMock) -> None:
    """Append a byte-limit notice when formatted output exceeds 50KB."""

    stdout = "\n".join(f"./{index:03d}-{'x' * 196}.py" for index in range(300))
    execution.return_value = f"{stdout}\n"

    tool_result = await find.fn(pattern="*.py", limit=1000, cwd=Path.cwd())
    result = _text(tool_result)
    notice = "\n\n[50.0KB limit reached]"
    body = result.removesuffix(notice)

    assert result.endswith(notice)
    assert len(body.encode("utf-8")) <= truncation.OUTPUT_BYTE_LIMIT
    details = _find_details(tool_result)
    assert details.output.truncated_by == "bytes"
    assert details.output.output_bytes <= truncation.OUTPUT_BYTE_LIMIT
    assert details.output.total_bytes > truncation.OUTPUT_BYTE_LIMIT


@pytest.mark.asyncio
@pytest.mark.usefixtures("fd_available")
async def test_fn_reports_byte_limit_when_byte_boundary_is_first(
    execution: AsyncMock,
) -> None:
    """Report only the byte limit when bytes truncate before result count."""

    stdout = "\n".join(f"./{index:03d}-{'x' * 196}.py" for index in range(300))
    execution.return_value = f"{stdout}\n"

    result = _text(await find.fn(pattern="*.py", limit=260, cwd=Path.cwd()))

    assert result.endswith("\n\n[50.0KB limit reached]")
    assert "results limit reached" not in result


def _text(result: ToolResult) -> str:
    """Return the single text block from a tool result."""

    assert len(result.content) == 1
    content = result.content[0]
    assert isinstance(content, ToolTextContent)
    return content.text


def _find_details(result: ToolResult) -> FindDetails:
    """Return find details from a tool result."""

    assert isinstance(result.details, FindDetails)
    return result.details
