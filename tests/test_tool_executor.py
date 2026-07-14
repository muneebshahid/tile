"""Tests for model-requested tool execution."""

import asyncio
from collections.abc import Callable
from typing import Literal

import pytest
from pydantic import ValidationError
from pydantic.errors import PydanticInvalidForJsonSchema

from tile.tool_executor import ToolExecutor
from tile.types.tool_execution import (
    ToolInputValidationFailure,
    ToolInvocationFailure,
)
from tile.types.tools import (
    JsonObject,
    ToolDefinition,
    ToolDetails,
    ToolInput,
    ToolResult,
)
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


class _NoInput(ToolInput):
    """Strict empty input for deterministic test tools."""


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


def test_tool_definition_generates_schema_from_input_model() -> None:
    """Use one Pydantic model for provider schema and execution validation."""

    tool = _sample_tool()

    assert tool.input_schema == tool.input_model.model_json_schema()


def test_tool_definition_rejects_input_model_without_json_schema() -> None:
    """Fail tool construction before a provider receives an invalid schema."""

    class InvalidInput(ToolInput):
        """Input containing a callable that JSON Schema cannot represent."""

        callback: Callable[[], str]

    with pytest.raises(PydanticInvalidForJsonSchema, match="CallableSchema"):
        ToolDefinition(
            name="invalid",
            description="Cannot expose this input to a provider.",
            input_model=InvalidInput,
            fn=_noop,
        )


def test_tool_executor_rejects_input_model_function_mismatch() -> None:
    """Fail runtime setup when model fields cannot reach the tool function."""

    tool = city_tool(
        "invalid",
        "Function does not accept the advertised city field.",
        _noop,
    )

    with pytest.raises(ValueError, match="function does not accept.*city"):
        ToolExecutor([tool])


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
    assert outcome.result.is_error is False
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
    assert outcome.result.is_error is True
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
    details = outcome.details
    assert isinstance(details, ToolInvocationFailure)
    assert details.tool_name == "fail_weather"
    assert details.exception_type == "RuntimeError"
    assert details.message == "boom"
    assert "exception" not in details.model_dump()


@pytest.mark.asyncio
async def test_tool_executor_does_not_normalize_cancellation() -> None:
    """Let task cancellation propagate through the tool boundary."""

    async def cancel(city: str) -> ToolResult:
        """Raise cancellation from a deterministic tool."""

        _ = city
        raise asyncio.CancelledError

    executor = ToolExecutor(
        [city_tool("cancel", "Cancel deterministic execution.", cancel)]
    )

    with pytest.raises(asyncio.CancelledError):
        await executor.execute(
            call_id="call_cancel",
            tool_name="cancel",
            arguments={"city": "Munich"},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "location", "code"),
    [
        ({}, ("city",), "missing"),
        ({"city": 5}, ("city",), "string_type"),
        (
            {"city": "Munich", "unexpected": True},
            ("unexpected",),
            "extra_forbidden",
        ),
    ],
)
async def test_tool_executor_rejects_invalid_arguments_before_invocation(
    arguments: JsonObject,
    location: tuple[str, ...],
    code: str,
) -> None:
    """Return structured correction details without invoking invalid input."""

    calls: list[str] = []

    async def capture(city: str) -> ToolResult:
        """Record valid invocations for the boundary assertion."""

        calls.append(city)
        return ToolResult.text(city)

    executor = ToolExecutor(
        [city_tool("weather", "Return deterministic weather.", capture)]
    )

    outcome = await executor.execute(
        call_id="call_invalid",
        tool_name="weather",
        arguments=arguments,
    )

    assert calls == []
    assert outcome.result.is_error is True
    assert tool_text(outcome.result).startswith("Invalid arguments for tool 'weather':")
    details = outcome.details
    assert isinstance(details, ToolInputValidationFailure)
    assert [(issue.location, issue.code) for issue in details.issues] == [
        (location, code)
    ]
    assert "input" not in details.model_dump()


class _ExpectedFailureDetails(ToolDetails):
    """Domain metadata for an intentionally returned tool failure."""

    type: Literal["expected_failure"] = "expected_failure"
    reason: str


@pytest.mark.asyncio
async def test_tool_executor_preserves_returned_error_result() -> None:
    """Let tools report handled failures without raising exceptions."""

    async def unavailable(city: str) -> ToolResult:
        """Return a deterministic handled failure."""

        return ToolResult.error(
            f"Weather unavailable for {city}",
            details=_ExpectedFailureDetails(reason="maintenance"),
        )

    executor = ToolExecutor(
        [city_tool("weather", "Return deterministic weather.", unavailable)]
    )

    outcome = await executor.execute(
        call_id="call_unavailable",
        tool_name="weather",
        arguments={"city": "Munich"},
    )

    assert outcome.result.is_error is True
    assert tool_text(outcome.result) == "Weather unavailable for Munich"
    assert isinstance(outcome.details, _ExpectedFailureDetails)


def test_tool_result_rejects_error_that_terminates_run() -> None:
    """Keep correction errors distinct from successful terminal tools."""

    with pytest.raises(ValidationError, match="cannot terminate"):
        ToolResult(
            content=[],
            is_error=True,
            terminate=True,
        )


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
        input_model=_NoInput,
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
            input_model=_NoInput,
            fn=_noop,
        )


@pytest.mark.asyncio
async def test_tool_executor_finds_tool_registered_with_uppercase_name() -> None:
    """Find a tool registered with an uppercase name when the model requests lowercase."""

    tool = ToolDefinition(
        name="Read",
        description="Read a file.",
        input_model=_NoInput,
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
        input_model=_NoInput,
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
