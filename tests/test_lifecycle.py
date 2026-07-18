"""Tests for the guaranteed run lifecycle across failures and aborts."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar, cast
from unittest.mock import AsyncMock

from pydantic import BaseModel

from tile.events import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    RunEndEvent,
    RunStartEvent,
    StreamFn,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from tile.history import InMemoryHistoryStore
from tile.result import Aborted, Completed, ExecutionFailure, Failed
from tile.runs import InMemoryRunStore, RunRecord
from tile.runtime import AgentRuntime, Run
from tile.types.conversation import ConversationItem, ToolResultTurn
from tile.types.stream_events import ProviderStreamEvent
from tile.types.tools import ToolDefinition, ToolFunction, ToolResult
from tests.support.agent_streams import (
    TEST_PROVIDER,
    ProviderStreamMock,
    error_stream,
    final_text_stream,
    stream_start,
    tool_call_stream,
)
from tests.support.async_streams import async_stream
from tests.support.tool_definitions import CityInput, city_tool


def assert_run_lifecycle(events: Sequence[AgentEvent]) -> None:
    """Assert the guaranteed run shape: one run start first, one run end last.

    Inner events carry no guarantee — a failure or abort can leave inner
    starts without ends; consumers sweep them on the run end.
    """

    assert events, "empty run log"
    assert events[0].type == "run_start"
    assert events[-1].type == "run_end"
    assert sum(event.type == "run_start" for event in events) == 1
    assert sum(event.type == "run_end" for event in events) == 1


class WeatherReport(BaseModel):
    """Sample result schema for typed-result attempt tests."""

    city: str
    temp_c: float


def test_provider_raise_before_stream_fails_the_run() -> None:
    """Close the run when the provider dies before streaming."""

    failing_mock = AsyncMock(side_effect=ConnectionError("connection refused"))
    failing_mock.provider = TEST_PROVIDER
    runtime = _runtime(cast("StreamFn", failing_mock))
    session = runtime.session(session_id="raise-before-stream")

    async def _run() -> list[AgentEvent]:
        """Fail the run and collect its complete log."""

        run = await session.prompt("hello")
        assert await run.wait() == "failed"
        return [event async for event in run.events()]

    events = asyncio.run(_run())

    assert_run_lifecycle(events)
    assert [event.type for event in events] == [
        "run_start",
        "agent_start",
        "run_end",
    ]
    run_outcome = _single(events, RunEndEvent).outcome
    assert isinstance(run_outcome, Failed)
    assert isinstance(run_outcome.cause, ExecutionFailure)
    assert run_outcome.cause.message == "connection refused"


def test_provider_raise_mid_stream_leaves_message_and_turn_open() -> None:
    """Close the run with the torn-down message and turn left open."""

    interrupted_mock = AsyncMock(
        return_value=async_stream(
            [stream_start("resp_1")], error=ConnectionError("connection reset")
        )
    )
    interrupted_mock.provider = TEST_PROVIDER
    runtime = _runtime(cast("StreamFn", interrupted_mock))
    session = runtime.session(session_id="raise-mid-stream")

    async def _run() -> list[AgentEvent]:
        """Fail the run mid-message and collect its complete log."""

        run = await session.prompt("hello")
        assert await run.wait() == "failed"
        return [event async for event in run.events()]

    events = asyncio.run(_run())

    assert_run_lifecycle(events)
    assert [event.type for event in events] == [
        "run_start",
        "agent_start",
        "turn_start",
        "message_start",
        "run_end",
    ]
    assert events[-1] == RunEndEvent(
        outcome=Failed(
            cause=ExecutionFailure(
                origin="execution",
                exception_type="ConnectionError",
                message="connection reset",
            )
        )
    )


def test_stream_exhausted_without_terminal_event_fails_the_run() -> None:
    """End the agent over its abandoned message and turn, then fail."""

    quiet_mock = AsyncMock(return_value=async_stream([stream_start("resp_1")]))
    quiet_mock.provider = TEST_PROVIDER
    runtime = _runtime(cast("StreamFn", quiet_mock))
    session = runtime.session(session_id="quiet-stream-death")

    async def _run() -> tuple[list[AgentEvent], str | None]:
        """Fail the run on a stream that ends without a terminal event."""

        run = await session.prompt("hello")
        assert await run.wait() == "failed"
        return [event async for event in run.events()], run.error_message

    events, error_message = asyncio.run(_run())

    assert_run_lifecycle(events)
    assert [event.type for event in events] == [
        "run_start",
        "agent_start",
        "turn_start",
        "message_start",
        "agent_end",
        "run_end",
    ]
    run_outcome = _single(events, RunEndEvent).outcome
    assert isinstance(run_outcome, Failed)
    assert isinstance(run_outcome.cause, ExecutionFailure)
    assert run_outcome.cause.origin == "turn"
    assert error_message == "The agent run ended without an assistant turn."


def test_in_band_stream_error_keeps_producer_ends_and_fails_the_run() -> None:
    """Keep the producer's own ends for an error the provider finalized."""

    runtime = _runtime(ProviderStreamMock([error_stream("resp_1", "boom")]).fn)
    session = runtime.session(session_id="in-band-error")

    async def _run() -> list[AgentEvent]:
        """Fail the run through an in-band stream error event."""

        run = await session.prompt("hello")
        assert await run.wait() == "failed"
        return [event async for event in run.events()]

    events = asyncio.run(_run())

    assert_run_lifecycle(events)
    assert _single(events, MessageEndEvent).assistant_turn.status == "error"
    _single(events, AgentEndEvent)
    run_outcome = _single(events, RunEndEvent).outcome
    assert isinstance(run_outcome, Failed)
    assert isinstance(run_outcome.cause, ExecutionFailure)
    assert run_outcome.cause.origin == "turn"


def test_abort_during_tool_execution_leaves_the_tool_open() -> None:
    """Close the run directly over an aborted in-flight tool execution."""

    async def _blocked(params: CityInput) -> ToolResult:
        """Block forever so the abort lands inside the tool."""

        _ = params
        await asyncio.Event().wait()
        return ToolResult.text("never")

    provider = ProviderStreamMock(
        [
            tool_call_stream(
                response_id="resp_1",
                call_id="call_1",
                tool_name="get_weather",
                arguments={"city": "Munich"},
            )
        ]
    )
    runtime = _runtime(provider.fn, tools=[_weather_tool(_blocked)])
    session = runtime.session(session_id="abort-in-tool")

    async def _run() -> list[AgentEvent]:
        """Abort once the tool execution has started."""

        run = await session.prompt("check weather")
        async for event in run.events():
            if isinstance(event, ToolExecutionStartEvent):
                break
        run.abort()
        assert await run.wait() == "aborted"
        return [event async for event in run.events()]

    events = asyncio.run(_run())

    assert_run_lifecycle(events)
    assert [event.type for event in events] == [
        "run_start",
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "tool_execution_start",
        "run_end",
    ]
    assert _single(events, RunEndEvent).outcome == Aborted()


def test_abort_during_provider_stream_leaves_the_message_open() -> None:
    """Close the run directly over an aborted in-flight message."""

    async def _stalled(
        events: Sequence[ProviderStreamEvent],
    ) -> AsyncIterator[ProviderStreamEvent]:
        """Yield the given events, then stall until cancelled."""

        for event in events:
            yield event
        await asyncio.Event().wait()

    stalled_mock = AsyncMock(return_value=_stalled([stream_start("resp_1")]))
    stalled_mock.provider = TEST_PROVIDER
    runtime = _runtime(cast("StreamFn", stalled_mock))
    session = runtime.session(session_id="abort-mid-stream")

    async def _run() -> list[AgentEvent]:
        """Abort once the message has started streaming."""

        run = await session.prompt("hello")
        async for event in run.events():
            if isinstance(event, MessageStartEvent):
                break
        run.abort()
        assert await run.wait() == "aborted"
        return [event async for event in run.events()]

    events = asyncio.run(_run())

    assert_run_lifecycle(events)
    assert [event.type for event in events] == [
        "run_start",
        "agent_start",
        "turn_start",
        "message_start",
        "run_end",
    ]
    assert _single(events, RunEndEvent).outcome == Aborted()


def test_abort_before_first_tick_still_yields_a_closed_log() -> None:
    """Close the run scope for an abort landing before the pump starts."""

    provider = ProviderStreamMock([final_text_stream("resp_1", "hello back")])
    runtime = _runtime(provider.fn)
    session = runtime.session(session_id="abort-before-tick")

    async def _run() -> list[AgentEvent]:
        """Abort synchronously after submission, before the first tick."""

        run = await session.prompt("hello")
        run.abort()
        assert await run.wait() == "aborted"
        return [event async for event in run.events()]

    events = asyncio.run(_run())

    assert events == [RunStartEvent(), RunEndEvent(outcome=Aborted())]


def test_committed_run_end_stops_the_pump_and_closes_a_stalled_source() -> None:
    """Finish at the commit point without pumping a source that never returns."""

    async def _run() -> None:
        """Commit a run end from a source that would stall forever after it."""

        closed = False

        async def _events() -> AsyncGenerator[AgentEvent, None]:
            """Commit the outcome, then hold the run open forever."""

            nonlocal closed
            try:
                yield RunEndEvent(outcome=Completed(value="done"))
                await asyncio.Event().wait()
            finally:
                closed = True

        persisted: list[RunRecord] = []
        run = _bare_run(_events(), persisted=persisted)

        assert await run.wait() == "completed"
        assert closed
        assert run.outcome == Completed(value="done")
        assert persisted == [run.record]

    asyncio.run(_run())


def test_events_after_the_committed_run_end_never_reach_the_log() -> None:
    """Stop pumping at the first run end so a later one cannot rewrite it."""

    async def _run() -> None:
        """Feed a source that misbehaves after committing its outcome."""

        persisted: list[RunRecord] = []
        run = _bare_run(
            async_stream(
                [
                    RunEndEvent(outcome=Completed(value="first")),
                    AgentStartEvent(),
                    RunEndEvent(outcome=Completed(value="second")),
                ]
            ),
            persisted=persisted,
        )

        assert await run.wait() == "completed"
        events = [event async for event in run.events()]
        assert events == [
            RunStartEvent(),
            RunEndEvent(outcome=Completed(value="first")),
        ]
        assert run.outcome == Completed(value="first")

    asyncio.run(_run())


def test_history_observer_failure_fails_the_run_without_suppressing_events() -> None:
    """Keep an event visible in the log when persisting it fails."""

    history_store = _FailingHistoryStore()
    provider = ProviderStreamMock([final_text_stream("resp_1", "hello back")])
    runtime = AgentRuntime(
        stream_fn=provider.fn,
        model="gpt-5.4",
        cwd=Path("."),
        history_store=history_store,
        run_store=InMemoryRunStore(),
    )
    session = runtime.session(session_id="observer-failure")

    async def _run() -> list[AgentEvent]:
        """Fail history projection for every event after submission."""

        run = await session.prompt("hello")
        history_store.fail_appends = True
        assert await run.wait() == "failed"
        assert isinstance(run.exception, RuntimeError)
        return [event async for event in run.events()]

    events = asyncio.run(_run())

    assert_run_lifecycle(events)
    _single(events, MessageEndEvent)
    run_outcome = _single(events, RunEndEvent).outcome
    assert isinstance(run_outcome, Failed)
    assert isinstance(run_outcome.cause, ExecutionFailure)
    assert run_outcome.cause.message == "history unavailable"


def test_healing_reads_durable_history_after_observer_failure() -> None:
    """Heal a tool call whose result reached the log but not the store."""

    history_store = _FlakyToolResultStore()
    provider = ProviderStreamMock(
        [
            tool_call_stream(
                response_id="resp_1",
                call_id="call_1",
                tool_name="get_weather",
                arguments={"city": "Munich"},
            )
        ]
    )
    runtime = AgentRuntime(
        stream_fn=provider.fn,
        model="gpt-5.4",
        cwd=Path("."),
        history_store=history_store,
        run_store=InMemoryRunStore(),
        tools=[_weather_tool(_quick_weather)],
    )
    session = runtime.session(session_id="heal-durable")

    async def _run() -> None:
        """Drop one tool-result write, then verify durable history healed."""

        run = await session.prompt("check weather")
        history_store.fail_next_tool_result = True
        assert await run.wait() == "failed"

        events = [event async for event in run.events()]
        assert any(isinstance(event, ToolExecutionEndEvent) for event in events)

        history = history_store.get_history(session.id)
        healed = history[-1]
        assert isinstance(healed, ToolResultTurn)
        assert healed.call_id == "call_1"
        assert healed.is_error

    asyncio.run(_run())


def test_clean_completion_without_run_end_fails_with_a_closed_log() -> None:
    """Fail a producer that completes without committing a run end."""

    async def _run() -> None:
        """Drive a producer whose event source returns without a run end."""

        persisted: list[RunRecord] = []
        run = _bare_run(
            async_stream([AgentStartEvent(), AgentEndEvent()]),
            persisted=persisted,
        )

        assert await run.wait() == "failed"
        assert run.exception is None
        failure = run.failure
        assert failure is not None
        assert failure.message == "The run ended without a committed run end event."
        events = [event async for event in run.events()]
        assert_run_lifecycle(events)
        assert persisted == [run.record]

    asyncio.run(_run())


def test_typed_result_attempts_each_close_before_the_next_starts() -> None:
    """Pair every typed-result attempt sequentially around the follow-up."""

    provider = ProviderStreamMock(
        [
            final_text_stream("resp_1", "Still thinking."),
            tool_call_stream(
                response_id="resp_2",
                call_id="call_1",
                tool_name="complete",
                arguments={"city": "Munich", "temp_c": 21.0},
            ),
        ]
    )
    runtime = _runtime(provider.fn)
    session = runtime.session(session_id="typed-attempts")

    async def _run() -> list[AgentEvent]:
        """Complete the typed result on the nudged second attempt."""

        run = await session.prompt("Weather?", result=WeatherReport)
        assert await run.wait() == "completed"
        assert isinstance(run.outcome, Completed)
        return [event async for event in run.events()]

    events = asyncio.run(_run())

    assert_run_lifecycle(events)
    start_indices = [
        index
        for index, event in enumerate(events)
        if isinstance(event, AgentStartEvent)
    ]
    end_indices = [
        index for index, event in enumerate(events) if isinstance(event, AgentEndEvent)
    ]
    assert len(start_indices) == 2
    assert len(end_indices) == 2
    assert end_indices[0] < start_indices[1]


def test_tool_loop_prompt_yields_the_full_expected_event_order() -> None:
    """Pin the complete runtime event order for a tool-use prompt."""

    provider = ProviderStreamMock(
        [
            tool_call_stream(
                response_id="resp_1",
                call_id="call_1",
                tool_name="get_weather",
                arguments={"city": "Munich"},
            ),
            final_text_stream("resp_2", "It is sunny in Munich."),
        ]
    )
    runtime = _runtime(provider.fn, tools=[_weather_tool(_quick_weather)])
    session = runtime.session(session_id="full-order")

    async def _run() -> list[AgentEvent]:
        """Complete one tool-loop prompt and collect its full log."""

        run = await session.prompt("check weather")
        assert await run.wait() == "completed"
        return [event async for event in run.events()]

    events = asyncio.run(_run())

    assert [event.type for event in events] == [
        "run_start",
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "tool_execution_start",
        "tool_execution_end",
        "turn_end",
        "turn_start",
        "message_start",
        "message_end",
        "turn_end",
        "agent_end",
        "run_end",
    ]


def test_typed_result_prompt_yields_the_full_expected_event_order() -> None:
    """Pin the complete runtime event order across a nudged typed run."""

    provider = ProviderStreamMock(
        [
            final_text_stream("resp_1", "Still thinking."),
            tool_call_stream(
                response_id="resp_2",
                call_id="call_1",
                tool_name="complete",
                arguments={"city": "Munich", "temp_c": 21.0},
            ),
        ]
    )
    runtime = _runtime(provider.fn)
    session = runtime.session(session_id="typed-full-order")

    async def _run() -> list[AgentEvent]:
        """Complete the typed result on the nudged second attempt."""

        run = await session.prompt("Weather?", result=WeatherReport)
        assert await run.wait() == "completed"
        return [event async for event in run.events()]

    events = asyncio.run(_run())

    assert [event.type for event in events] == [
        "run_start",
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "turn_end",
        "agent_end",
        "result_follow_up",
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "tool_execution_start",
        "tool_execution_end",
        "turn_end",
        "agent_end",
        "run_end",
    ]


def _runtime(
    stream_fn: StreamFn,
    *,
    tools: Sequence[ToolDefinition] = (),
) -> AgentRuntime:
    """Build a runtime over in-memory stores for lifecycle tests."""

    return AgentRuntime(
        stream_fn=stream_fn,
        model="gpt-5.4",
        cwd=Path("."),
        history_store=InMemoryHistoryStore(),
        run_store=InMemoryRunStore(),
        tools=tools,
    )


def _bare_run(
    events: AsyncGenerator[AgentEvent, None],
    *,
    persisted: list[RunRecord],
) -> Run:
    """Start a run directly over a scripted event source."""

    return Run(
        record=RunRecord(
            run_id="bare-run",
            session_id="bare-run",
            status="running",
            started_at=datetime.now(UTC),
            model="gpt-5.4",
        ),
        events=events,
        on_done=lambda _: None,
        on_record=persisted.append,
        on_event=lambda _: None,
    )


def _weather_tool(fn: ToolFunction) -> ToolDefinition:
    """Build the deterministic weather tool around one implementation."""

    return city_tool("get_weather", "Return a deterministic weather report.", fn)


async def _quick_weather(params: CityInput) -> ToolResult:
    """Return deterministic weather text immediately."""

    return ToolResult.text(f"{params.city}: sunny")


_EventT = TypeVar("_EventT", bound=AgentEvent)


def _single(events: Sequence[AgentEvent], event_type: type[_EventT]) -> _EventT:
    """Return the only event of one type in a run log."""

    matches = [event for event in events if isinstance(event, event_type)]
    assert len(matches) == 1, f"expected one {event_type.__name__}, got {len(matches)}"
    return matches[0]


class _FailingHistoryStore(InMemoryHistoryStore):
    """History store with a switchable append failure."""

    fail_appends: bool = False

    def append_history(
        self,
        session_id: str,
        items: Sequence[ConversationItem],
    ) -> None:
        """Append history unless the test has enabled deterministic failure."""

        if self.fail_appends:
            raise RuntimeError("history unavailable")
        super().append_history(session_id, items)


class _FlakyToolResultStore(InMemoryHistoryStore):
    """History store that drops exactly one tool-result append."""

    fail_next_tool_result: bool = False

    def append_history(
        self,
        session_id: str,
        items: Sequence[ConversationItem],
    ) -> None:
        """Fail the next append carrying a tool result, then recover."""

        carries_tool_result = any(isinstance(item, ToolResultTurn) for item in items)
        if self.fail_next_tool_result and carries_tool_result:
            self.fail_next_tool_result = False
            raise RuntimeError("history unavailable")
        super().append_history(session_id, items)
