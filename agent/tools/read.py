"""Text file read tool for the default agent."""

from pathlib import Path

from pydantic import BaseModel

from ai.types.tools import ToolDefinition
from agent.tools.truncation import (
    OUTPUT_BYTE_LIMIT,
    OUTPUT_BYTE_LIMIT_LABEL,
    HeadTruncation,
    format_size,
    truncate_head,
)


class ReadSelection(BaseModel):
    """Selected file content and line metadata for formatting."""

    content: str
    start_line: int
    total_lines: int
    user_limited_lines: int | None


async def fn(
    path: str,
    offset: int | None = None,
    limit: int | None = None,
) -> str:
    """Read a UTF-8 text file with optional line offset and limit."""

    content = _execute(path)
    selection = _parse_output(content, offset, limit)
    return _format_results(selection, path)


def _execute(path: str) -> str:
    """Read a UTF-8 text file from disk."""

    return _resolve_path(path).read_text(encoding="utf-8")


def _parse_output(
    content: str,
    offset: int | None,
    limit: int | None,
) -> ReadSelection:
    """Select the requested line window from file content."""

    lines = content.split("\n")
    start_index = _start_index(offset)
    if start_index >= len(lines):
        raise RuntimeError(
            f"Offset {offset} is beyond end of file ({len(lines)} lines total)"
        )

    selected_lines = _select_lines(lines, start_index, limit)
    return ReadSelection(
        content="\n".join(selected_lines),
        start_line=start_index + 1,
        total_lines=len(lines),
        user_limited_lines=len(selected_lines) if limit is not None else None,
    )


def _format_results(selection: ReadSelection, path: str) -> str:
    """Format selected file content with Pi-compatible continuation notices."""

    truncation = truncate_head(selection.content)
    if truncation.first_line_exceeds_limit:
        return _format_first_line_too_large(selection, path)
    if truncation.truncated:
        return _format_truncated_selection(selection, truncation)
    if _user_limit_left_remaining_lines(selection):
        return _format_user_limited_selection(selection, truncation.content)
    return truncation.content


def _resolve_path(path: str) -> Path:
    """Resolve a path relative to the current working directory."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def _start_index(offset: int | None) -> int:
    """Convert a 1-indexed offset to a non-negative list index."""

    if offset is None:
        return 0
    return max(0, offset - 1)


def _select_lines(
    lines: list[str],
    start_index: int,
    limit: int | None,
) -> list[str]:
    """Return the requested line slice."""

    if limit is None:
        return lines[start_index:]
    end_index = min(start_index + limit, len(lines))
    return lines[start_index:end_index]


def _format_first_line_too_large(selection: ReadSelection, path: str) -> str:
    """Return guidance when the first selected line exceeds the byte limit."""

    first_line = selection.content.split("\n", maxsplit=1)[0]
    first_line_size = format_size(len(first_line.encode("utf-8")))
    return (
        f"[Line {selection.start_line} is {first_line_size}, exceeds "
        f"{OUTPUT_BYTE_LIMIT_LABEL} limit. Use bash: "
        f"sed -n '{selection.start_line}p' {path} | head -c {OUTPUT_BYTE_LIMIT}]"
    )


def _format_truncated_selection(
    selection: ReadSelection,
    truncation: HeadTruncation,
) -> str:
    """Return truncated content with an offset continuation notice."""

    end_line = selection.start_line + truncation.output_lines - 1
    next_offset = end_line + 1
    notice = _truncation_notice(selection, truncation, end_line, next_offset)
    return f"{truncation.content}\n\n[{notice}]"


def _format_user_limited_selection(selection: ReadSelection, content: str) -> str:
    """Return user-limited content with an offset continuation notice."""

    limited_lines = selection.user_limited_lines or 0
    remaining = selection.total_lines - (
        _start_index_from_selection(selection) + limited_lines
    )
    next_offset = selection.start_line + limited_lines
    return f"{content}\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"


def _user_limit_left_remaining_lines(selection: ReadSelection) -> bool:
    """Return whether the caller's limit stopped before end of file."""

    if selection.user_limited_lines is None:
        return False
    return (
        _start_index_from_selection(selection) + selection.user_limited_lines
        < selection.total_lines
    )


def _truncation_notice(
    selection: ReadSelection,
    truncation: HeadTruncation,
    end_line: int,
    next_offset: int,
) -> str:
    """Build the continuation notice for automatic truncation."""

    if truncation.truncated_by == "lines":
        return (
            f"Showing lines {selection.start_line}-{end_line} of "
            f"{selection.total_lines}. Use offset={next_offset} to continue."
        )
    return (
        f"Showing lines {selection.start_line}-{end_line} of {selection.total_lines} "
        f"({OUTPUT_BYTE_LIMIT_LABEL} limit). Use offset={next_offset} to continue."
    )


def _start_index_from_selection(selection: ReadSelection) -> int:
    """Return the zero-based start index for a selection."""

    return selection.start_line - 1


tool = ToolDefinition(
    name="read",
    description=(
        "Read the contents of a UTF-8 text file. Output is truncated to 2000 lines "
        "or 50KB. Use offset and limit for large files."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read, relative or absolute.",
            },
            "offset": {
                "type": "integer",
                "description": "Line number to start reading from, 1-indexed.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read.",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    fn=fn,
)
