"""File write tool for the default agent."""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel

from ai.types.tools import ToolDefinition, ToolResult
from agent.tools.paths import resolve_to_cwd


async def fn(path: str, content: str, *, cwd: Path) -> ToolResult:
    """Write content to a file."""

    resolved_path = _resolve_path(path, cwd)
    result = await _execute(resolved_path, content)
    return ToolResult.text(_format_results(result))


async def _execute(path: Path, content: str) -> Results:
    """Write file content asynchronously."""

    return await asyncio.to_thread(_write_file, path, content)


class Results(BaseModel):
    """Structured file write result."""

    path: Path
    bytes_written: int


def _format_results(result: Results) -> str:
    """Format a successful write result."""

    return f"Successfully wrote {result.bytes_written} bytes to {result.path}"


def _resolve_path(path: str, cwd: Path) -> Path:
    """Resolve a user-provided path for writing."""

    return resolve_to_cwd(path, cwd).resolve(strict=False)


def _write_file(path: Path, content: str) -> Results:
    """Create parent directories and write UTF-8 content to a file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return Results(path=path, bytes_written=len(content.encode("utf-8")))


tool = ToolDefinition(
    name="write",
    description=(
        "Write content to a file. Creates the file if it doesn't exist, "
        "overwrites if it does. Automatically creates parent directories."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write, relative or absolute.",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file.",
            },
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
    fn=fn,
)
