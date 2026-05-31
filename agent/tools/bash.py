"""Shell command tool scaffold for the default agent."""

from pathlib import Path

from ai.types.tools import ToolDefinition, ToolResult


async def fn(command: str, timeout: float | None = None, *, cwd: Path) -> ToolResult:
    """Execute a shell command from the agent working directory."""

    _ = command
    _ = timeout
    _ = cwd
    raise NotImplementedError("bash execution is not implemented yet.")


tool = ToolDefinition(
    name="bash",
    description="Execute a bash command.",
    input_schema={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Bash command to execute.",
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds. Optional, with no default timeout.",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    },
    fn=fn,
)
