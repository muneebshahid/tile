"""Tests for the output contract tools and their run loop enforcement."""

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import BaseModel

from tile.agent import run_agent
from tile.history import InMemoryHistoryStore
from tile.result import (
    MAX_RESULT_FOLLOW_UPS,
    NO_RESULT_REASON,
    RESULT_ALREADY_RECORDED,
    RESULT_CONTRACT,
    RESULT_FOLLOW_UP,
    Completed,
    Failed,
)
from tile.runtime import AgentRuntime
from tile.tool_executor import ToolExecutor
from tile.tools.complete import (
    CompleteDetails,
    strict_object_schema,
    tool as complete_tool,
)
from tile.tools.fail import tool as fail_tool
from tile.events import (
    AgentEndEvent,
    AgentEvent,
    ResultFollowUpEvent,
    ToolExecutionEndEvent,
)
from tile.types.conversation import ToolResultTurn, UserMessage
from tile.types.stream_events import TextBlock
from tile.types.tools import ToolDefinition, ToolResult, ToolTextContent
from tests.support.agent_streams import (
    ProviderStreamMock,
    error_stream,
    final_text_stream,
    stream_done,
    stream_start,
    tool_call_block,
    tool_call_stream,
)
from tests.support.tool_definitions import city_tool


class WeatherReport(BaseModel):
    """Sample result schema used across output contract tests."""

    city: str
    temp_c: float


def _result_tools() -> list[ToolDefinition]:
    """Build the result tool pair for the sample schema."""

    return [complete_tool(WeatherReport), fail_tool]


async def _weather(city: str) -> ToolResult:
    """Fail loudly if a post-result tool call is ever executed."""

    raise AssertionError(f"get_weather must not execute (city={city})")


def _collect_run_events(
    history: Sequence[UserMessage],
    *,
    stream_fn,
    cwd: Path,
    enforce_output_contract: bool = False,
    tools: Sequence[ToolDefinition] = (),
) -> list[AgentEvent]:
    """Collect all events emitted by one stateless agent run."""

    async def _collect() -> list[AgentEvent]:
        """Collect run events from the async generator."""

        return [
            event
            async for event in run_agent(
                list(history),
                stream_fn=stream_fn,
                model="gpt-5.4",
                tool_executor=ToolExecutor(tools),
                instructions="Base prompt.",
                auto_mode=False,
                enforce_output_contract=enforce_output_contract,
                cwd=cwd,
            )
        ]

    return asyncio.run(_collect())


def _agent_end_event(events: Sequence[AgentEvent]) -> AgentEndEvent:
    """Return the terminal agent end event of a collected run."""

    event = events[-1]
    assert isinstance(event, AgentEndEvent)
    return event


def _complete_call_stream(
    response_id: str,
    call_id: str,
    arguments: dict,
) -> list:
    """Build a provider stream that calls the complete result tool."""

    return tool_call_stream(
        response_id=response_id,
        call_id=call_id,
        tool_name="complete",
        arguments=arguments,
    )


def test_complete_tool_validates_and_succeeds() -> None:
    """Accept schema-conforming arguments on the complete tool."""

    executor = ToolExecutor(_result_tools())

    outcome = asyncio.run(
        executor.execute(
            call_id="call_1",
            tool_name="complete",
            arguments={"city": "Munich", "temp_c": 21.0},
        )
    )

    assert not outcome.tool_result_turn.is_error


def test_complete_tool_rejects_invalid_arguments() -> None:
    """Return a tool error for arguments that violate the result schema."""

    executor = ToolExecutor(_result_tools())

    outcome = asyncio.run(
        executor.execute(
            call_id="call_1",
            tool_name="complete",
            arguments={"city": "Munich"},
        )
    )

    assert outcome.tool_result_turn.is_error


def test_fail_tool_requires_string_reason() -> None:
    """Reject non-string reasons on the fail tool."""

    executor = ToolExecutor(_result_tools())

    outcome = asyncio.run(
        executor.execute(
            call_id="call_1",
            tool_name="fail",
            arguments={"reason": 5},
        )
    )

    assert outcome.tool_result_turn.is_error


def test_strict_object_schema_closes_nested_objects() -> None:
    """Add additionalProperties: false to every object schema node."""

    class Inner(BaseModel):
        name: str

    class Outer(BaseModel):
        inner: Inner
        items: list[Inner]

    schema = strict_object_schema(Outer.model_json_schema())

    assert schema["additionalProperties"] is False
    defs = schema["$defs"]
    assert isinstance(defs, dict)
    inner = defs["Inner"]
    assert isinstance(inner, dict)
    assert inner["additionalProperties"] is False


def test_complete_tool_applies_field_defaults() -> None:
    """Fill omitted optional fields from the result model's defaults."""

    class ReportWithDefault(BaseModel):
        city: str
        note: str = "n/a"

    executor = ToolExecutor([complete_tool(ReportWithDefault), fail_tool])

    outcome = asyncio.run(
        executor.execute(
            call_id="call_1",
            tool_name="complete",
            arguments={"city": "Munich"},
        )
    )

    assert not outcome.tool_result_turn.is_error
    details = outcome.details
    assert isinstance(details, CompleteDetails)
    assert details.value == ReportWithDefault(city="Munich", note="n/a")


def test_complete_tool_returns_validated_value_in_details() -> None:
    """Carry the validated result instance on the execution details."""

    executor = ToolExecutor(_result_tools())

    outcome = asyncio.run(
        executor.execute(
            call_id="call_1",
            tool_name="complete",
            arguments={"city": "Munich", "temp_c": "21.5"},
        )
    )

    details = outcome.details
    assert isinstance(details, CompleteDetails)
    assert details.value == WeatherReport(city="Munich", temp_c=21.5)


def test_completed_round_trips_value_as_plain_data() -> None:
    """Deserialize a serialized outcome into plain data, losing nothing."""

    outcome = Completed(value=WeatherReport(city="Munich", temp_c=21.0))

    revalidated = Completed.model_validate_json(outcome.model_dump_json())

    assert revalidated.value == {"city": "Munich", "temp_c": 21.0}


def test_tool_executor_rejects_duplicate_names() -> None:
    """Refuse to register two tools with the same name."""

    with pytest.raises(ValueError, match="Duplicate tool name"):
        ToolExecutor([fail_tool, fail_tool])


def test_enforcement_requires_result_tools(tmp_path: Path) -> None:
    """Reject contract enforcement when the result tools are missing."""

    provider = ProviderStreamMock([])

    with pytest.raises(ValueError, match="complete, fail"):
        _collect_run_events(
            [UserMessage(content="Weather?")],
            stream_fn=provider.fn,
            cwd=tmp_path,
            enforce_output_contract=True,
        )


def test_agent_run_ends_immediately_on_complete(tmp_path: Path) -> None:
    """Exit the run without another provider call once complete succeeds."""

    provider = ProviderStreamMock(
        [
            _complete_call_stream(
                "resp_1", "call_1", {"city": "Munich", "temp_c": 21.0}
            ),
        ]
    )

    events = _collect_run_events(
        [UserMessage(content="Weather in Munich?")],
        stream_fn=provider.fn,
        cwd=tmp_path,
        enforce_output_contract=True,
        tools=_result_tools(),
    )

    assert provider.await_count == 1
    outcome = _agent_end_event(events).outcome
    assert outcome == Completed(
        value=WeatherReport(city="Munich", temp_c=21.0),
        output_text="",
    )


def test_agent_run_ends_on_fail(tmp_path: Path) -> None:
    """Report a model-declared failure as the terminal outcome."""

    provider = ProviderStreamMock(
        [
            tool_call_stream(
                response_id="resp_1",
                call_id="call_1",
                tool_name="fail",
                arguments={"reason": "The city is ambiguous."},
            ),
        ]
    )

    events = _collect_run_events(
        [UserMessage(content="Weather?")],
        stream_fn=provider.fn,
        cwd=tmp_path,
        enforce_output_contract=True,
        tools=_result_tools(),
    )

    assert provider.await_count == 1
    outcome = _agent_end_event(events).outcome
    assert outcome == Failed(reason="The city is ambiguous.", output_text="")


def test_agent_retries_complete_after_validation_error(tmp_path: Path) -> None:
    """Route result validation errors back to the model for correction."""

    provider = ProviderStreamMock(
        [
            _complete_call_stream("resp_1", "call_1", {"city": "Munich"}),
            _complete_call_stream(
                "resp_2", "call_2", {"city": "Munich", "temp_c": 21.0}
            ),
        ]
    )

    events = _collect_run_events(
        [UserMessage(content="Weather in Munich?")],
        stream_fn=provider.fn,
        cwd=tmp_path,
        enforce_output_contract=True,
        tools=_result_tools(),
    )

    assert provider.await_count == 2
    retry_history = provider.history(1)
    error_result = retry_history[-1]
    assert isinstance(error_result, ToolResultTurn)
    assert error_result.is_error
    outcome = _agent_end_event(events).outcome
    assert isinstance(outcome, Completed)
    assert outcome.value == WeatherReport(city="Munich", temp_c=21.0)


def test_agent_nudges_text_only_turn_toward_result(tmp_path: Path) -> None:
    """Append a follow-up reminder when an enforced run ends in plain text."""

    provider = ProviderStreamMock(
        [
            final_text_stream("resp_1", "The temperature is 21C."),
            _complete_call_stream(
                "resp_2", "call_1", {"city": "Munich", "temp_c": 21.0}
            ),
        ]
    )

    events = _collect_run_events(
        [UserMessage(content="Weather in Munich?")],
        stream_fn=provider.fn,
        cwd=tmp_path,
        enforce_output_contract=True,
        tools=_result_tools(),
    )

    follow_ups = [e for e in events if isinstance(e, ResultFollowUpEvent)]
    assert len(follow_ups) == 1
    assert follow_ups[0].message.content == RESULT_FOLLOW_UP
    nudged_history = provider.history(1)
    assert nudged_history[-1] == UserMessage(content=RESULT_FOLLOW_UP)
    outcome = _agent_end_event(events).outcome
    assert isinstance(outcome, Completed)


def test_agent_fails_after_follow_up_cap(tmp_path: Path) -> None:
    """Give up with a failure outcome when nudges never produce a result."""

    streams = [
        final_text_stream(f"resp_{index}", "Still thinking.")
        for index in range(MAX_RESULT_FOLLOW_UPS + 1)
    ]
    provider = ProviderStreamMock(streams)

    events = _collect_run_events(
        [UserMessage(content="Weather?")],
        stream_fn=provider.fn,
        cwd=tmp_path,
        enforce_output_contract=True,
        tools=_result_tools(),
    )

    assert provider.await_count == MAX_RESULT_FOLLOW_UPS + 1
    outcome = _agent_end_event(events).outcome
    assert outcome == Failed(
        reason=NO_RESULT_REASON,
        output_text="Still thinking.",
    )


def test_agent_without_contract_completes_with_text(tmp_path: Path) -> None:
    """Wrap a plain text ending in a completed outcome with no value."""

    provider = ProviderStreamMock(
        [
            final_text_stream("resp_1", "The temperature is 21C."),
        ]
    )

    events = _collect_run_events(
        [UserMessage(content="Weather in Munich?")],
        stream_fn=provider.fn,
        cwd=tmp_path,
    )

    outcome = _agent_end_event(events).outcome
    assert outcome == Completed(value=None, output_text="The temperature is 21C.")


def test_agent_outcome_is_none_on_stream_error(tmp_path: Path) -> None:
    """Leave the outcome unset when the run ends on a stream error."""

    provider = ProviderStreamMock(
        [
            error_stream("resp_1", "boom"),
        ]
    )

    events = _collect_run_events(
        [UserMessage(content="Weather?")],
        stream_fn=provider.fn,
        cwd=tmp_path,
        enforce_output_contract=True,
        tools=_result_tools(),
    )

    assert _agent_end_event(events).outcome is None


def test_agent_skips_tool_calls_after_result(tmp_path: Path) -> None:
    """Answer post-result calls in the same turn with errors, unexecuted."""

    provider = ProviderStreamMock(
        [
            [
                stream_start("resp_1"),
                stream_done(
                    "resp_1",
                    stop_reason="tool_use",
                    blocks=[
                        tool_call_block(
                            call_id="call_1",
                            name="complete",
                            arguments={"city": "Munich", "temp_c": 21.0},
                        ),
                        tool_call_block(
                            call_id="call_2",
                            name="get_weather",
                            arguments={"city": "Berlin"},
                        ),
                    ],
                ),
            ],
        ]
    )

    events = _collect_run_events(
        [UserMessage(content="Weather?")],
        stream_fn=provider.fn,
        cwd=tmp_path,
        enforce_output_contract=True,
        tools=[*_result_tools(), city_tool("get_weather", "Get weather.", _weather)],
    )

    executions = [e for e in events if isinstance(e, ToolExecutionEndEvent)]
    assert len(executions) == 2
    assert not executions[0].outcome.tool_result_turn.is_error
    skipped = executions[1].outcome.tool_result_turn
    assert skipped.is_error
    assert skipped.content == [ToolTextContent(text=RESULT_ALREADY_RECORDED)]
    outcome = _agent_end_event(events).outcome
    assert isinstance(outcome, Completed)
    assert outcome.value == WeatherReport(city="Munich", temp_c=21.0)


def test_agent_output_text_captures_ending_turn_text(tmp_path: Path) -> None:
    """Capture text streamed alongside the terminal result call."""

    provider = ProviderStreamMock(
        [
            [
                stream_start("resp_1"),
                stream_done(
                    "resp_1",
                    stop_reason="tool_use",
                    blocks=[
                        TextBlock(text="Recording the result."),
                        tool_call_block(
                            call_id="call_1",
                            name="complete",
                            arguments={"city": "Munich", "temp_c": 21.0},
                        ),
                    ],
                ),
            ],
        ]
    )

    events = _collect_run_events(
        [UserMessage(content="Weather?")],
        stream_fn=provider.fn,
        cwd=tmp_path,
        enforce_output_contract=True,
        tools=_result_tools(),
    )

    outcome = _agent_end_event(events).outcome
    assert isinstance(outcome, Completed)
    assert outcome.output_text == "Recording the result."


def test_session_prompt_composes_result_tools_and_contract() -> None:
    """Add result tools and contract for one prompt when a schema is set."""

    provider = ProviderStreamMock(
        [
            final_text_stream("resp_1", "Let me summarize instead."),
            _complete_call_stream(
                "resp_2", "call_1", {"city": "Munich", "temp_c": 21.0}
            ),
        ]
    )
    store = InMemoryHistoryStore()
    runtime = AgentRuntime(
        stream_fn=provider.fn,
        model="gpt-5.4",
        history_store=store,
        auto_mode=False,
    )

    async def _run() -> None:
        session = runtime.session(session_id="result-session")
        run = await session.prompt("Weather in Munich?", result=WeatherReport)
        assert await run.wait() == "completed"
        outcome = run.outcome
        assert isinstance(outcome, Completed)
        assert outcome.value == WeatherReport(city="Munich", temp_c=21.0)
        history = store.get_history("result-session")
        assert UserMessage(content=RESULT_FOLLOW_UP) in list(history)

    asyncio.run(_run())

    tools = provider.tools(0)
    assert tools is not None
    assert {tool.name for tool in tools} == {"complete", "fail"}
    instructions = provider.mock.await_args_list[0].kwargs["instructions"]
    assert RESULT_CONTRACT in instructions


def test_session_mixes_contract_and_plain_prompts() -> None:
    """Run contract and plain prompts back to back on one session."""

    provider = ProviderStreamMock(
        [
            _complete_call_stream(
                "resp_1", "call_1", {"city": "Munich", "temp_c": 21.0}
            ),
            final_text_stream("resp_2", "You asked about Munich."),
        ]
    )
    runtime = AgentRuntime(
        stream_fn=provider.fn,
        model="gpt-5.4",
        auto_mode=False,
    )

    async def _run() -> None:
        session = runtime.session(session_id="mixed-session")
        contract_run = await session.prompt("Weather in Munich?", result=WeatherReport)
        assert await contract_run.wait() == "completed"
        assert isinstance(contract_run.outcome, Completed)
        assert contract_run.outcome.value == WeatherReport(city="Munich", temp_c=21.0)

        plain_run = await session.prompt("Which city did I ask about?")
        assert await plain_run.wait() == "completed"
        assert plain_run.outcome == Completed(
            value=None, output_text="You asked about Munich."
        )

    asyncio.run(_run())

    contract_tools = provider.tools(0)
    assert contract_tools is not None
    assert {tool.name for tool in contract_tools} == {"complete", "fail"}
    plain_tools = provider.tools(1)
    assert plain_tools == ()
    plain_instructions = provider.mock.await_args_list[1].kwargs["instructions"]
    assert RESULT_CONTRACT not in plain_instructions


def test_runtime_rejects_reserved_tool_names() -> None:
    """Refuse caller tools named after the reserved result tools."""

    provider = ProviderStreamMock([])

    with pytest.raises(ValueError, match="reserved"):
        AgentRuntime(
            stream_fn=provider.fn,
            model="gpt-5.4",
            tools=[city_tool("complete", "Not the real complete.", _weather)],
        )
