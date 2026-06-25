"""Tests for the default search tool scaffold."""

import json
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock

import pytest

import ori.tools.executables as executables
import ori.tools.grep as grep
import ori.tools.truncation as truncation
from ori.types.tools import GrepDetails, ToolResult, ToolTextContent


def test_schema_requires_only_pattern() -> None:
    """Require only the search pattern so callers can omit optional controls."""

    assert grep.tool.input_schema["required"] == ["pattern"]


def test_build_args_uses_default_search_flags() -> None:
    """Build the default argv for machine-readable search output."""

    assert grep._build_args(
        pattern="needle",
        path=".",
        glob=None,
        ignore_case=False,
        literal=False,
        context=0,
    ) == [
        "--json",
        "--line-number",
        "--color=never",
        "--hidden",
        "--",
        "needle",
        ".",
    ]


def test_build_args_adds_optional_search_flags() -> None:
    """Build argv from optional search controls."""

    assert grep._build_args(
        pattern="needle",
        path="src",
        glob="**/*.py",
        ignore_case=True,
        literal=True,
        context=2,
    ) == [
        "--json",
        "--line-number",
        "--color=never",
        "--hidden",
        "--ignore-case",
        "--fixed-strings",
        "--glob",
        "**/*.py",
        "--context",
        "2",
        "--",
        "needle",
        "src",
    ]


def test_build_args_protects_flag_like_patterns() -> None:
    """Place -- before the pattern so flag-like patterns stay search text."""

    assert grep._build_args(
        pattern="--pre=payload",
        path=".",
        glob=None,
        ignore_case=False,
        literal=True,
        context=0,
    )[-3:] == ["--", "--pre=payload", "."]


@pytest.fixture
def rg_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the rg executable available to fn-level tests."""

    monkeypatch.setattr(executables.shutil, "which", _find_command)


@pytest.fixture
def rg_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the rg executable unavailable to fn-level tests."""

    monkeypatch.setattr(executables.shutil, "which", _find_no_commands)


@pytest.fixture
def execution(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Patch command execution with an async mock for fn-level tests."""

    execution_mock = AsyncMock()
    monkeypatch.setattr(grep, "execute", execution_mock)
    return execution_mock


@pytest.mark.asyncio
@pytest.mark.usefixtures("rg_available")
async def test_fn_returns_results_when_command_is_available(
    execution: AsyncMock,
) -> None:
    """Return compact grep-style text when rg exists."""

    execution.return_value = _event("match", "example.txt", 2, "needle line\n")

    tool_result = await grep.fn(pattern="needle", cwd=Path.cwd())
    result = _text(tool_result)

    assert result == "example.txt:2: needle line"
    assert tool_result.details is None


@pytest.mark.asyncio
@pytest.mark.usefixtures("rg_available")
async def test_fn_returns_multiple_result_lines(
    execution: AsyncMock,
) -> None:
    """Return compact grep-style text for multiple command output lines."""

    execution.return_value = "\n".join(
        [
            _event("match", "one.txt", 1, "needle one\n"),
            _event("match", "two.txt", 2, "needle two\n"),
        ]
    )

    result = _text(await grep.fn(pattern="needle", cwd=Path.cwd()))

    assert result == "one.txt:1: needle one\ntwo.txt:2: needle two"


@pytest.mark.asyncio
@pytest.mark.usefixtures("rg_available")
async def test_fn_resolves_search_path_against_supplied_cwd(
    execution: AsyncMock,
    tmp_path: Path,
) -> None:
    """Resolve relative search roots against the supplied tool cwd."""

    execution.return_value = _event("match", "example.txt", 2, "needle line\n")

    result = _text(await grep.fn(pattern="needle", path="src", cwd=tmp_path))

    assert result == "example.txt:2: needle line"
    assert _captured_args(execution)[-1] == "src"
    assert _captured_cwd(execution) == tmp_path


@pytest.mark.asyncio
@pytest.mark.usefixtures("rg_available")
async def test_fn_reports_match_limit_in_details(execution: AsyncMock) -> None:
    """Return grep details when the match limit is reached."""

    execution.return_value = "\n".join(
        [
            _event("match", "one.txt", 1, "needle one\n"),
            _event("match", "two.txt", 2, "needle two\n"),
        ]
    )

    tool_result = await grep.fn(pattern="needle", limit=1, cwd=Path.cwd())

    assert _text(tool_result).endswith(
        "\n\n[1 matches limit reached. Use limit=2 for more, or refine pattern]"
    )
    details = _grep_details(tool_result)
    assert details.match_limit_reached == 1
    assert details.output is not None
    assert details.lines_truncated is False


@pytest.mark.asyncio
@pytest.mark.usefixtures("rg_available")
async def test_fn_reports_byte_truncation_in_details(execution: AsyncMock) -> None:
    """Return grep details when formatted output exceeds the byte limit."""

    execution.return_value = "\n".join(
        _event("match", f"{index:03d}.txt", 1, "x" * 196) for index in range(300)
    )

    tool_result = await grep.fn(pattern="needle", limit=500, cwd=Path.cwd())

    assert _text(tool_result).endswith("\n\n[50.0KB limit reached]")
    details = _grep_details(tool_result)
    assert details.match_limit_reached is None
    assert details.lines_truncated is False
    assert details.output is not None
    assert details.output.truncated is True
    assert details.output.truncated_by == "bytes"
    assert details.output.max_bytes == truncation.OUTPUT_BYTE_LIMIT


@pytest.mark.asyncio
@pytest.mark.usefixtures("rg_available")
async def test_fn_reports_line_truncation_in_details(execution: AsyncMock) -> None:
    """Return grep details when individual result lines are shortened."""

    execution.return_value = _event("match", "example.txt", 1, "x" * 501)

    tool_result = await grep.fn(pattern="needle", cwd=Path.cwd())

    assert _text(tool_result).endswith(
        "\n\n[Some lines truncated to 500 chars. Use read tool to see full lines]"
    )
    details = _grep_details(tool_result)
    assert details.match_limit_reached is None
    assert details.output is not None
    assert details.output.truncated is False
    assert details.lines_truncated is True


@pytest.mark.asyncio
@pytest.mark.usefixtures("rg_missing")
async def test_fn_raises_when_command_is_missing() -> None:
    """Raise a clear exception when rg is unavailable."""

    with pytest.raises(RuntimeError, match="ripgrep"):
        await grep.fn(pattern="needle", cwd=Path.cwd())


def test_parse_output_returns_internal_results() -> None:
    """Parse match and context events into internal result models."""

    result = grep._parse_output(
        "\n".join(
            [
                _event("context", "example.txt", 1, "before\n"),
                _event("match", "example.txt", 2, "needle line\n"),
                _event("context", "example.txt", 3, "after\n"),
            ]
        ),
        limit=100,
    )

    assert result == grep.Results(
        lines=[
            grep.Line(
                kind="context",
                path="example.txt",
                line_number=1,
                text="before",
            ),
            grep.Line(
                kind="match",
                path="example.txt",
                line_number=2,
                text="needle line",
            ),
            grep.Line(
                kind="context",
                path="example.txt",
                line_number=3,
                text="after",
            ),
        ],
        truncated=False,
    )


def test_parse_output_marks_truncated_after_limit() -> None:
    """Stop parsing once the global match limit is reached."""

    result = grep._parse_output(
        "\n".join(
            [
                _event("match", "example.txt", 1, "first\n"),
                _event("match", "example.txt", 2, "second\n"),
            ]
        ),
        limit=1,
    )

    assert result.truncated is True
    assert [line.text for line in result.lines] == ["first"]


def test_parse_output_ignores_non_search_events() -> None:
    """Ignore events that are not match or context rows."""

    result = grep._parse_output(
        "\n".join(
            [
                json.dumps(
                    {"type": "begin", "data": {"path": {"text": "example.txt"}}}
                ),
                _event("match", "example.txt", 1, "needle\n"),
                json.dumps({"type": "summary", "data": {"elapsed_total": {"secs": 1}}}),
            ]
        ),
        limit=100,
    )

    assert [line.text for line in result.lines] == ["needle"]


def test_build_result_returns_plain_text() -> None:
    """Build grep-style text from raw search output."""

    result = grep._build_result(
        "\n".join(
            [
                _event("context", "example.txt", 1, "before\n"),
                _event("match", "example.txt", 2, "needle line\n"),
                _event("context", "example.txt", 3, "after\n"),
            ]
        ),
        limit=100,
    )

    assert _text(result) == "\n".join(
        [
            "example.txt-1- before",
            "example.txt:2: needle line",
            "example.txt-3- after",
        ]
    )


def test_build_result_reports_truncation() -> None:
    """Append a compact truncation note when matches exceed the limit."""

    result = grep._build_result(
        "\n".join(
            [
                _event("match", "example.txt", 1, "first\n"),
                _event("match", "example.txt", 2, "second\n"),
            ]
        ),
        limit=1,
    )

    assert _text(result) == (
        "example.txt:1: first\n\n"
        "[1 matches limit reached. Use limit=2 for more, or refine pattern]"
    )


def test_build_result_reports_byte_limit() -> None:
    """Append a byte-limit notice when formatted output exceeds 50KB."""

    result = grep._build_result(
        "\n".join(
            _event("match", f"{index:03d}.txt", 1, f"{'x' * 196}\n")
            for index in range(300)
        ),
        limit=500,
    )
    notice = "\n\n[50.0KB limit reached]"
    text = _text(result)
    body = text.removesuffix(notice)

    assert text.endswith(notice)
    assert len(body.encode("utf-8")) <= truncation.OUTPUT_BYTE_LIMIT


def test_build_result_reports_line_limit() -> None:
    """Append a line-limit notice when a result line is shortened."""

    result = grep._build_result(
        _event("match", "example.txt", 1, f"{'x' * 501}\n"),
        limit=100,
    )

    assert _text(result) == (
        f"example.txt:1: {'x' * 500}... [truncated]\n\n"
        "[Some lines truncated to 500 chars. Use read tool to see full lines]"
    )


def test_build_result_combines_truncation_notices() -> None:
    """Report match, byte, and line truncation in one notice block."""

    result = grep._build_result(
        "\n".join(
            _event("match", f"{index:03d}.txt", 1, f"{'x' * 501}\n")
            for index in range(120)
        ),
        limit=100,
    )

    assert _text(result).endswith(
        "\n\n[100 matches limit reached. Use limit=200 for more, or refine pattern. "
        "50.0KB limit reached. "
        "Some lines truncated to 500 chars. Use read tool to see full lines]"
    )


def _find_command(command: str) -> str | None:
    """Return a path only for search command availability checks."""

    if command == "rg":
        return "/usr/bin/rg"
    return None


def _find_no_commands(command: str) -> None:
    """Return no command path for all availability checks."""

    _ = command
    return None


def _captured_args(execution: AsyncMock) -> list[str]:
    """Return the rg args captured by a fake execution call."""

    execution.assert_awaited_once()
    await_args = execution.await_args
    assert await_args is not None
    args = await_args.args
    assert isinstance(args[1], list)
    return args[1]


def _captured_cwd(execution: AsyncMock) -> Path | None:
    """Return the cwd captured by a fake execution call."""

    execution.assert_awaited_once()
    await_args = execution.await_args
    assert await_args is not None
    cwd = await_args.kwargs.get("cwd")
    assert isinstance(cwd, Path) or cwd is None
    return cwd


def _event(
    event_type: Literal["match", "context"],
    path: str,
    line_number: int,
    text: str,
) -> str:
    """Build one JSON event line for parser tests."""

    return json.dumps(
        {
            "type": event_type,
            "data": {
                "path": {"text": path},
                "line_number": line_number,
                "lines": {"text": text},
            },
        }
    )


def _text(result: ToolResult) -> str:
    """Return the single text block from a tool result."""

    assert len(result.content) == 1
    content = result.content[0]
    assert isinstance(content, ToolTextContent)
    return content.text


def _grep_details(result: ToolResult) -> GrepDetails:
    """Return grep details from a tool result."""

    assert isinstance(result.details, GrepDetails)
    return result.details
