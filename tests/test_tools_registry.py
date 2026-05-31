"""Tests for the built-in tool registry."""

from pathlib import Path

import pytest

from agent.tools import build_tools


@pytest.mark.asyncio
async def test_build_tools_binds_cwd_to_tool_functions(tmp_path: Path) -> None:
    """Bind cwd into returned tool functions without exposing it in schemas."""

    tool_by_name = {tool.name: tool for tool in build_tools(tmp_path)}
    await tool_by_name["write"].fn(path="sample.txt", content="hello")
    properties = tool_by_name["write"].input_schema["properties"]

    assert (tmp_path / "sample.txt").read_text(encoding="utf-8") == "hello"
    assert isinstance(properties, dict)
    assert "cwd" not in properties


def test_build_tools_preserves_default_tool_order(tmp_path: Path) -> None:
    """Return the same default tools as the registry list."""

    tools = build_tools(tmp_path)

    assert [tool.name for tool in tools] == ["read", "grep", "find", "ls", "write"]
