"""Directory listing tool for the default agent."""

import asyncio
from pathlib import Path

from ai.types.tools import ToolDefinition


async def fn(path: str = ".", limit: int = 500) -> str:
    """List the contents of a directory."""

    try:
        entries = await asyncio.to_thread(_list_directory_entries, path)
        limited_entries = entries[:limit]
        if not limited_entries:
            return "(empty directory)"

        result = "\n".join(limited_entries)
        if len(entries) > limit:
            result += (
                f"\n\n[{limit} entries limit reached. Use limit={limit * 2} for more]"
            )
        return result
    except Exception as e:
        return f"Error: {str(e)}"


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


ls = ToolDefinition(
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
        "required": ["path", "limit"],
        "additionalProperties": False,
    },
    fn=fn,
)
