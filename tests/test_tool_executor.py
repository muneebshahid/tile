"""Tests for model-requested tool execution."""

import pytest

from agent.tool_executor import ToolExecutionRequest, ToolExecutor
from ai.types.tools import ToolDefinition, ToolResult, ToolTextContent


async def _get_weather(city: str) -> ToolResult:
    """Return deterministic weather for a city."""

    return ToolResult.text(f"{city}: sunny")


async def _raise_error(city: str) -> ToolResult:
    """Raise a deterministic tool error."""

    _ = city
    raise RuntimeError("boom")


def _sample_tool() -> ToolDefinition:
    """Build a deterministic tool definition."""

    return ToolDefinition(
        name="get_weather",
        description="Return deterministic weather.",
        input_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
        fn=_get_weather,
    )


def _failing_tool() -> ToolDefinition:
    """Build a deterministic failing tool definition."""

    return ToolDefinition(
        name="fail_weather",
        description="Raise deterministic weather failure.",
        input_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
        fn=_raise_error,
    )


def _tool_text(result: ToolResult) -> str:
    """Return the single text block from a tool result."""

    assert len(result.content) == 1
    content = result.content[0]
    assert isinstance(content, ToolTextContent)
    return content.text


@pytest.mark.asyncio
async def test_tool_executor_executes_registered_tool() -> None:
    """Execute a registered tool and return a normalized outcome."""

    executor = ToolExecutor([_sample_tool()])

    outcome = await executor.execute(
        ToolExecutionRequest(
            call_id="call_weather",
            tool_name="get_weather",
            arguments={"city": "Munich"},
        )
    )

    assert outcome.tool_result_turn.call_id == "call_weather"
    assert outcome.tool_result_turn.tool_name == "get_weather"
    assert outcome.tool_result_turn.is_error is False
    assert _tool_text(outcome.result) == "Munich: sunny"


@pytest.mark.asyncio
async def test_tool_executor_normalizes_missing_tool() -> None:
    """Return an error outcome when a requested tool is not registered."""

    executor = ToolExecutor()

    outcome = await executor.execute(
        ToolExecutionRequest(
            call_id="call_missing",
            tool_name="missing_tool",
            arguments={},
        )
    )

    assert outcome.tool_result_turn.call_id == "call_missing"
    assert outcome.tool_result_turn.tool_name == "missing_tool"
    assert outcome.tool_result_turn.is_error is True
    assert _tool_text(outcome.result) == "Tool 'missing_tool' not found"


@pytest.mark.asyncio
async def test_tool_executor_normalizes_tool_exception() -> None:
    """Return an error outcome when tool execution raises."""

    executor = ToolExecutor([_failing_tool()])

    outcome = await executor.execute(
        ToolExecutionRequest(
            call_id="call_fail",
            tool_name="fail_weather",
            arguments={"city": "Munich"},
        )
    )

    assert outcome.tool_result_turn.call_id == "call_fail"
    assert outcome.tool_result_turn.tool_name == "fail_weather"
    assert outcome.tool_result_turn.is_error is True
    assert _tool_text(outcome.result) == "boom"
