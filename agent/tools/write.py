"""File write tool scaffold for the default agent."""

from ai.types.tools import ToolDefinition, ToolResult


async def fn(path: str, content: str) -> ToolResult:
    """Write content to a file."""

    _ = path, content
    raise NotImplementedError("write tool execution is not implemented yet")


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
