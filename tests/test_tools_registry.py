"""Tests for the built-in tool registry."""

import inspect

from tile.tools import BUILTIN_TOOLS


def test_builtin_tools_preserves_default_tool_order() -> None:
    """Expose the default tools unbound, in the documented order."""

    assert [tool.name for tool in BUILTIN_TOOLS] == [
        "read",
        "bash",
        "edit",
        "grep",
        "find",
        "ls",
        "write",
    ]


def test_builtin_tools_declare_cwd_without_exposing_it() -> None:
    """Declare cwd for runtime injection on every builtin, never in schemas."""

    for tool in BUILTIN_TOOLS:
        assert "cwd" in inspect.signature(tool.fn).parameters
        properties = tool.input_schema.get("properties")
        assert isinstance(properties, dict)
        assert "cwd" not in properties
