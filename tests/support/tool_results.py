"""Shared helpers for inspecting tool result content in tests."""

from tile.types.conversation import ToolResultTurn
from tile.types.tools import ToolResult, ToolTextContent


def tool_text(result: ToolResult | ToolResultTurn) -> str:
    """Return the single text block from a result or replay projection."""

    assert len(result.content) == 1
    content = result.content[0]
    assert isinstance(content, ToolTextContent)
    return content.text
