"""Result tool that ends an output-contract run with a declared failure."""

from __future__ import annotations

from tile.result import FAIL_TOOL_NAME
from tile.types.tools import ToolDefinition, ToolResult


async def fail(reason: str) -> ToolResult:
    """Record the model's reason for not delivering a result."""

    if not isinstance(reason, str):
        raise ValueError("`reason` must be a string.")
    return ToolResult.text("Failure recorded.")


tool = ToolDefinition(
    name=FAIL_TOOL_NAME,
    description=(
        "Report that the task cannot be completed and end the run. "
        "Provide a clear reason naming what is missing or impossible."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Why the task cannot be completed.",
            }
        },
        "required": ["reason"],
        "additionalProperties": False,
    },
    fn=fail,
)
