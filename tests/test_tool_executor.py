"""Tests for model-requested tool execution."""

import pytest
from pydantic import ValidationError

from ori.tool_executor import ToolExecutor
from ori.types.tools import ToolDefinition, ToolDetails, ToolResult
from tests.support.tool_definitions import city_tool
from tests.support.tool_results import tool_text


async def _get_weather(city: str) -> ToolResult:
    """Return deterministic weather for a city."""

    return ToolResult.text(f"{city}: sunny")


async def _raise_error(city: str) -> ToolResult:
    """Raise a deterministic tool error."""

    _ = city
    raise RuntimeError("boom")


async def _noop() -> ToolResult:
    """Return a fixed result for tools that take no arguments."""

    return ToolResult.text("ok")


def _sample_tool() -> ToolDefinition:
    """Build a deterministic tool definition."""

    return city_tool(
        "get_weather",
        "Return deterministic weather.",
        _get_weather,
    )


def _failing_tool() -> ToolDefinition:
    """Build a deterministic failing tool definition."""

    return city_tool(
        "fail_weather",
        "Raise deterministic weather failure.",
        _raise_error,
    )


@pytest.mark.asyncio
async def test_tool_executor_executes_registered_tool() -> None:
    """Execute a registered tool and return a normalized outcome."""

    executor = ToolExecutor([_sample_tool()])

    outcome = await executor.execute(
        call_id="call_weather",
        tool_name="get_weather",
        arguments={"city": "Munich"},
    )

    assert outcome.tool_result_turn.call_id == "call_weather"
    assert outcome.tool_result_turn.tool_name == "get_weather"
    assert outcome.tool_result_turn.is_error is False
    assert tool_text(outcome.result) == "Munich: sunny"


@pytest.mark.asyncio
async def test_tool_executor_normalizes_missing_tool() -> None:
    """Return an error outcome when a requested tool is not registered."""

    executor = ToolExecutor()

    outcome = await executor.execute(
        call_id="call_missing",
        tool_name="missing_tool",
        arguments={},
    )

    assert outcome.tool_result_turn.call_id == "call_missing"
    assert outcome.tool_result_turn.tool_name == "missing_tool"
    assert outcome.tool_result_turn.is_error is True
    assert tool_text(outcome.result) == "Tool 'missing_tool' not found"


@pytest.mark.asyncio
async def test_tool_executor_normalizes_tool_exception() -> None:
    """Return an error outcome when tool execution raises."""

    executor = ToolExecutor([_failing_tool()])

    outcome = await executor.execute(
        call_id="call_fail",
        tool_name="fail_weather",
        arguments={"city": "Munich"},
    )

    assert outcome.tool_result_turn.call_id == "call_fail"
    assert outcome.tool_result_turn.tool_name == "fail_weather"
    assert outcome.tool_result_turn.is_error is True
    assert tool_text(outcome.result) == "boom"


class _DatabaseDetails(ToolDetails):
    """User-defined details for a custom database tool."""

    type: str = "database"
    rows_scanned: int


async def _query_database() -> ToolResult:
    """Return a result carrying user-defined details."""

    return ToolResult.text("2 rows", details=_DatabaseDetails(rows_scanned=2))


@pytest.mark.asyncio
async def test_tool_executor_preserves_user_defined_details() -> None:
    """Carry and serialize user-defined tool details end to end."""

    tool = ToolDefinition(
        name="query_database",
        description="Query a database.",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        fn=_query_database,
    )
    executor = ToolExecutor([tool])

    outcome = await executor.execute(
        call_id="call_db",
        tool_name="query_database",
        arguments={},
    )

    details = outcome.result.details
    assert isinstance(details, _DatabaseDetails)
    assert details.rows_scanned == 2
    assert outcome.result.model_dump()["details"] == {
        "type": "database",
        "rows_scanned": 2,
    }


@pytest.mark.parametrize("name", ["", "   ", " read", "read ", "\tread\n"])
def test_tool_definition_rejects_empty_or_padded_names(name: str) -> None:
    """Fail tool registration for empty or whitespace-padded names."""

    with pytest.raises(ValidationError, match="non-empty without surrounding"):
        ToolDefinition(
            name=name,
            description="Read a file.",
            input_schema={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            fn=_noop,
        )


@pytest.mark.asyncio
async def test_tool_executor_finds_tool_registered_with_uppercase_name() -> None:
    """Find a tool registered with an uppercase name when the model requests lowercase."""

    tool = ToolDefinition(
        name="Read",
        description="Read a file.",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        fn=_noop,
    )
    executor = ToolExecutor([tool])

    outcome = await executor.execute(
        call_id="call_read",
        tool_name="read",
        arguments={},
    )

    assert outcome.tool_result_turn.is_error is False
    assert tool_text(outcome.result) == "ok"


@pytest.mark.asyncio
async def test_tool_executor_finds_tool_registered_with_lowercase_name() -> None:
    """Find a tool registered with a lowercase name when the model requests uppercase."""

    tool = ToolDefinition(
        name="read",
        description="Read a file.",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        fn=_noop,
    )
    executor = ToolExecutor([tool])

    outcome = await executor.execute(
        call_id="call_read",
        tool_name="Read",
        arguments={},
    )

    assert outcome.tool_result_turn.is_error is False
    assert tool_text(outcome.result) == "ok"
