"""Tests for the typed result contract and its run loop integration."""

import asyncio
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from tile.agent import run_agent
from tile.result import (
    MAX_RESULT_FOLLOW_UPS,
    NO_RESULT_REASON,
    RESULT_CONTRACT,
    RESULT_FOLLOW_UP,
    Completed,
    Failed,
    ResultRecorder,
)
from tile.history import InMemoryHistoryStore
from tile.runtime import AgentRuntime
from tile.tool_executor import ToolExecutor
from tile.events import (
    AgentEndEvent,
    AgentEvent,
    ResultFollowUpEvent,
    ToolExecutionEndEvent,
)
from tile.types.conversation import ToolResultTurn, UserMessage
from tile.types.stream_events import TextBlock
from tile.types.tools import ToolDefinition
from tests.support.agent_streams import (
    ProviderStreamMock,
    error_stream,
    final_text_stream,
    stream_done,
    stream_start,
    tool_call_block,
    tool_call_stream,
)


class WeatherReport(BaseModel):
    """Sample result schema used across result contract tests."""

    city: str
    temp_c: float


def _collect_run_events(
    history: Sequence[UserMessage],
    *,
    stream_fn,
    cwd: Path,
    result: type[BaseModel] | None = None,
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
                result=result,
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


def test_recorder_complete_validates_and_records() -> None:
    """Record a schema-conforming result from a complete tool call."""

    recorder = ResultRecorder(WeatherReport)
    executor = ToolExecutor(recorder.tool_definitions())

    outcome = asyncio.run(
        executor.execute(
            call_id="call_1",
            tool_name="complete",
            arguments={"city": "Munich", "temp_c": 21.0},
        )
    )

    assert not outcome.tool_result_turn.is_error
    assert recorder.value == WeatherReport(city="Munich", temp_c=21.0)


def test_recorder_complete_rejects_invalid_arguments() -> None:
    """Return a tool error and record nothing for invalid result arguments."""

    recorder = ResultRecorder(WeatherReport)
    executor = ToolExecutor(recorder.tool_definitions())

    outcome = asyncio.run(
        executor.execute(
            call_id="call_1",
            tool_name="complete",
            arguments={"city": "Munich"},
        )
    )

    assert outcome.tool_result_turn.is_error
    assert not recorder.has_outcome


def test_recorder_rejects_second_result_call() -> None:
    """Refuse to overwrite an already recorded result."""

    recorder = ResultRecorder(WeatherReport)
    executor = ToolExecutor(recorder.tool_definitions())

    async def _run() -> tuple:
        first = await executor.execute(
            call_id="call_1",
            tool_name="complete",
            arguments={"city": "Munich", "temp_c": 21.0},
        )
        second = await executor.execute(
            call_id="call_2",
            tool_name="fail",
            arguments={"reason": "Changed my mind."},
        )
        return first, second

    first, second = asyncio.run(_run())

    assert not first.tool_result_turn.is_error
    assert second.tool_result_turn.is_error
    assert recorder.value == WeatherReport(city="Munich", temp_c=21.0)
    assert recorder.reason is None


def test_recorder_fail_records_reason() -> None:
    """Record the model's failure reason from a fail tool call."""

    recorder = ResultRecorder(WeatherReport)
    executor = ToolExecutor(recorder.tool_definitions())

    outcome = asyncio.run(
        executor.execute(
            call_id="call_1",
            tool_name="fail",
            arguments={"reason": "No API key available."},
        )
    )

    assert not outcome.tool_result_turn.is_error
    assert recorder.reason == "No API key available."


def test_agent_run_ends_immediately_on_complete(tmp_path: Path) -> None:
    """Exit the run without another provider call once complete records."""

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
        result=WeatherReport,
    )

    assert provider.await_count == 1
    outcome = _agent_end_event(events).outcome
    assert isinstance(outcome, Completed)
    assert outcome.value == WeatherReport(city="Munich", temp_c=21.0)
    assert outcome.output_text == ""


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
        result=WeatherReport,
    )

    assert provider.await_count == 1
    outcome = _agent_end_event(events).outcome
    assert isinstance(outcome, Failed)
    assert outcome.reason == "The city is ambiguous."


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
        result=WeatherReport,
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
    """Append a follow-up reminder when a schema run ends in plain text."""

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
        result=WeatherReport,
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
        result=WeatherReport,
    )

    assert provider.await_count == MAX_RESULT_FOLLOW_UPS + 1
    outcome = _agent_end_event(events).outcome
    assert isinstance(outcome, Failed)
    assert outcome.reason == NO_RESULT_REASON
    assert outcome.output_text == "Still thinking."


def test_agent_without_result_completes_with_text(tmp_path: Path) -> None:
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
        result=WeatherReport,
    )

    assert _agent_end_event(events).outcome is None


def test_agent_records_first_result_in_batched_turn(tmp_path: Path) -> None:
    """Keep the first result and error later result calls in the same turn."""

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
                            name="complete",
                            arguments={"city": "Berlin", "temp_c": 18.0},
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
        result=WeatherReport,
    )

    executions = [e for e in events if isinstance(e, ToolExecutionEndEvent)]
    assert len(executions) == 2
    assert not executions[0].outcome.tool_result_turn.is_error
    assert executions[1].outcome.tool_result_turn.is_error
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
        result=WeatherReport,
    )

    outcome = _agent_end_event(events).outcome
    assert isinstance(outcome, Completed)
    assert outcome.output_text == "Recording the result."


def test_agent_offers_result_tools_and_contract(tmp_path: Path) -> None:
    """Expose the result tools and contract only when a schema is set."""

    provider = ProviderStreamMock(
        [
            _complete_call_stream(
                "resp_1", "call_1", {"city": "Munich", "temp_c": 21.0}
            ),
        ]
    )

    _collect_run_events(
        [UserMessage(content="Weather?")],
        stream_fn=provider.fn,
        cwd=tmp_path,
        result=WeatherReport,
    )

    tools = provider.tools(0)
    assert tools is not None
    assert [tool.name for tool in tools] == ["complete", "fail"]
    assert RESULT_CONTRACT in provider.instructions()


def test_session_prompt_returns_outcome_and_persists_follow_up() -> None:
    """Thread the result contract through a session run end to end."""

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
