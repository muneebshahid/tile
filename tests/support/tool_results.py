"""Shared helpers for inspecting tool results in tests."""

from tile.types.tools import ToolResult, ToolTextContent


def tool_text(result: ToolResult) -> str:
    """Return the single text block from a tool result."""

    assert len(result.content) == 1
    content = result.content[0]
    assert isinstance(content, ToolTextContent)
    return content.text
