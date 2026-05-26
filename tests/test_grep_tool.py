"""Tests for the default search tool scaffold."""

import json
import sys
from collections.abc import Sequence
from typing import Literal

import pytest
import agent.tools.grep as grep
import agent.tools.truncation as truncation


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
        limit=100,
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
        limit=25,
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
        limit=100,
    )[-3:] == ["--", "--pre=payload", "."]


@pytest.mark.asyncio
async def test_fn_returns_results_when_command_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return compact grep-style text when rg exists."""

    monkeypatch.setattr(grep.shutil, "which", _find_command)
    monkeypatch.setattr(grep, "_execute", _fake_execution)

    result = await grep.fn(pattern="needle")

    assert result == "example.txt:2: needle line"


@pytest.mark.asyncio
async def test_fn_returns_multiple_result_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return compact grep-style text for multiple command output lines."""

    monkeypatch.setattr(grep.shutil, "which", _find_command)
    monkeypatch.setattr(grep, "_execute", _fake_multi_line_execution)

    result = await grep.fn(pattern="needle")

    assert result == "one.txt:1: needle one\ntwo.txt:2: needle two"


@pytest.mark.asyncio
async def test_fn_raises_when_command_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise a clear exception when rg is unavailable."""

    monkeypatch.setattr(grep.shutil, "which", _find_no_commands)

    with pytest.raises(RuntimeError, match="ripgrep"):
        await grep.fn(pattern="needle")


@pytest.mark.asyncio
async def test_execute_returns_process_result() -> None:
    """Return captured stdout from a process."""

    result = await grep._execute(
        sys.executable,
        ["-c", "print('out')"],
    )

    assert result == "out\n"


@pytest.mark.asyncio
async def test_execute_raises_on_search_error() -> None:
    """Raise when the search process exits with an error code."""

    with pytest.raises(RuntimeError, match="boom"):
        await grep._execute(
            sys.executable,
            ["-c", "import sys; print('boom', file=sys.stderr); sys.exit(2)"],
        )


def test_parse_output_returns_pydantic_results() -> None:
    """Parse match and context events from JSON-lines output."""

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
        match_count=1,
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

    assert result.match_count == 1
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

    assert result.match_count == 1
    assert [line.text for line in result.lines] == ["needle"]


def test_format_results_returns_plain_text() -> None:
    """Format parsed search results using grep-style separators."""

    result = grep._format_results(
        grep.Results(
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
            match_count=1,
            truncated=False,
        ),
        limit=100,
    )

    assert result == "\n".join(
        [
            "example.txt-1- before",
            "example.txt:2: needle line",
            "example.txt-3- after",
        ]
    )


def test_format_results_reports_truncation() -> None:
    """Append a compact truncation note when matches exceed the limit."""

    result = grep._format_results(
        grep.Results(
            lines=[
                grep.Line(
                    kind="match",
                    path="example.txt",
                    line_number=1,
                    text="first",
                )
            ],
            match_count=1,
            truncated=True,
        ),
        limit=1,
    )

    assert result == (
        "example.txt:1: first\n\n"
        "[1 matches limit reached. Use limit=2 for more, or refine pattern]"
    )


def test_format_results_reports_byte_limit() -> None:
    """Append a byte-limit notice when formatted output exceeds 50KB."""

    result = grep._format_results(
        grep.Results(
            lines=[
                grep.Line(
                    kind="match",
                    path=f"{index:03d}.txt",
                    line_number=1,
                    text="x" * 196,
                )
                for index in range(300)
            ],
            match_count=300,
            truncated=False,
        ),
        limit=500,
    )
    notice = "\n\n[50.0KB limit reached]"
    body = result.removesuffix(notice)

    assert result.endswith(notice)
    assert len(body.encode("utf-8")) <= truncation.OUTPUT_BYTE_LIMIT


def test_format_results_reports_line_limit() -> None:
    """Append a line-limit notice when a result line is shortened."""

    result = grep._format_results(
        grep.Results(
            lines=[
                grep.Line(
                    kind="match",
                    path="example.txt",
                    line_number=1,
                    text="x" * 501,
                )
            ],
            match_count=1,
            truncated=False,
        ),
        limit=100,
    )

    assert result == (
        f"example.txt:1: {'x' * 500}... [truncated]\n\n"
        "[Some lines truncated to 500 chars. Use read tool to see full lines]"
    )


def test_format_results_combines_truncation_notices() -> None:
    """Report match, byte, and line truncation in one notice block."""

    result = grep._format_results(
        grep.Results(
            lines=[
                grep.Line(
                    kind="match",
                    path=f"{index:03d}.txt",
                    line_number=1,
                    text="x" * 501,
                )
                for index in range(120)
            ],
            match_count=120,
            truncated=True,
        ),
        limit=100,
    )

    assert result.endswith(
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


async def _fake_execution(
    executable: str,
    args: Sequence[str],
) -> str:
    """Return representative command output for fn tests."""

    _ = (executable, args)
    return _event("match", "example.txt", 2, "needle line\n")


async def _fake_multi_line_execution(
    executable: str,
    args: Sequence[str],
) -> str:
    """Return multiple representative command output lines for fn tests."""

    _ = (executable, args)
    return "\n".join(
        [
            _event("match", "one.txt", 1, "needle one\n"),
            _event("match", "two.txt", 2, "needle two\n"),
        ]
    )


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
