"""Directory listing tool for the default agent."""

import asyncio
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from ai.types.tools import ToolDefinition, ToolResult

from agent.tools.paths import resolve_to_cwd
from agent.tools.truncation import OUTPUT_BYTE_LIMIT_LABEL, truncate_head


class Results(BaseModel):
    """Structured directory listing results returned by the ls tool."""

    entries: list[str]


async def fn(path: str = ".", limit: int = 500, *, cwd: Path) -> ToolResult:
    """List the contents of a directory."""

    limit = max(1, limit)
    resolved_path = _resolve_path(path, cwd)
    output = await _execute(resolved_path)
    results = _parse_output(output)
    return ToolResult.text(_format_results(results, limit))


async def _execute(path: Path) -> list[str]:
    """List directory entries asynchronously."""

    return await asyncio.to_thread(_list_directory_entries, path)


def _parse_output(output: Sequence[str]) -> Results:
    """Parse raw directory entries into structured results."""

    return Results(entries=list(output))


def _format_results(results: Results, limit: int) -> str:
    """Format directory listing results as compact plain text."""

    if not results.entries:
        return "(empty directory)"

    truncation = truncate_head("\n".join(results.entries), max_lines=limit)
    result = truncation.content

    notices: list[str] = []
    if truncation.truncated_by == "lines":
        notices.append(f"{limit} entries limit reached. Use limit={limit * 2} for more")
    if truncation.truncated_by == "bytes":
        notices.append(
            f"{OUTPUT_BYTE_LIMIT_LABEL} limit reached. "
            f"Directory has {len(results.entries)} entries"
        )
    if notices:
        result += f"\n\n[{'. '.join(notices)}]"
    return result


def _resolve_path(path: str, cwd: Path) -> Path:
    """Resolve a directory path against the tool working directory."""

    return resolve_to_cwd(path, cwd).resolve(strict=False)


def _list_directory_entries(path: Path) -> list[str]:
    """Return directory entry names for a path."""

    return sorted(
        (_format_directory_entry(entry) for entry in path.iterdir()),
        key=str.lower,
    )


def _format_directory_entry(entry: Path) -> str:
    """Return a display name with directory entries marked by a slash."""

    if entry.is_dir():
        return f"{entry.name}/"
    return entry.name


tool = ToolDefinition(
    name="ls",
    description="List the contents of a directory.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The path of the directory to list. Defaults to the current directory.",
            },
            "limit": {
                "type": "integer",
                "description": "The maximum number of entries to list. Defaults to 500.",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    fn=fn,
)
