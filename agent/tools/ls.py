"""Directory listing tool for the default agent."""

import asyncio
from pathlib import Path

from ai.types.tools import ToolDefinition


BYTE_LIMIT = 50 * 1024


async def fn(path: str = ".", limit: int = 500) -> str:
    """List the contents of a directory."""

    entries = await asyncio.to_thread(_list_directory_entries, path)
    limited_entries = entries[:limit]
    if not limited_entries:
        return "(empty directory)"

    result = "\n".join(limited_entries)
    result, byte_limit_reached = _truncate_to_byte_limit(result)

    notices: list[str] = []
    if len(entries) > limit:
        notices.append(f"{limit} entries limit reached. Use limit={limit * 2} for more")
    if byte_limit_reached:
        notices.append("50.0KB limit reached")
    if notices:
        result += f"\n\n[{'. '.join(notices)}]"
    return result


def _truncate_to_byte_limit(
    text: str,
    byte_limit: int = BYTE_LIMIT,
) -> tuple[str, bool]:
    """Return text capped to complete lines within the UTF-8 byte limit."""

    if len(text.encode("utf-8")) <= byte_limit:
        return text, False

    output_lines: list[str] = []
    output_bytes = 0
    for line in text.split("\n"):
        separator_bytes = 1 if output_lines else 0
        line_bytes = len(line.encode("utf-8"))
        if output_bytes + separator_bytes + line_bytes > byte_limit:
            break

        output_lines.append(line)
        output_bytes += separator_bytes + line_bytes

    return "\n".join(output_lines), True


def _list_directory_entries(path: str) -> list[str]:
    """Return directory entry names for a string path."""

    return sorted(
        (_format_directory_entry(entry) for entry in Path(path).iterdir()),
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
