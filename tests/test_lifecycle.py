"""Tests for guaranteed lifecycle pairing across failures and aborts."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar, cast
from unittest.mock import AsyncMock

from pydantic import BaseModel

from tile.events import (
    AgentEndEvent,
    AgentEvent,
    AgentInterruptedEvent,
    AgentStartEvent,
    MessageEndEvent,
    MessageInterruptedEvent,
    MessageStartEvent,
    RunEndEvent,
    RunStartEvent,
    StreamFn,
    ToolExecutionEndEvent,
    ToolExecutionInterruptedEvent,
    ToolExecutionStartEvent,
    TurnInterruptedEvent,
    TurnStartEvent,
)
from tile.history import InMemoryHistoryStore
from tile.lifecycle import OpenScopeTracker
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

_START_SCOPES = {
    "run_start": "run",
    "agent_start": "agent",
    "turn_start": "turn",
    "message_start": "message",
    "tool_execution_start": "tool",
}
_CLOSE_SCOPES = {
    "run_end": "run",
    "agent_end": "agent",
    "agent_interrupted": "agent",
    "turn_end": "turn",
    "turn_interrupted": "turn",
    "message_end": "message",
    "message_interrupted": "message",
    "tool_execution_end": "tool",
    "tool_execution_interrupted": "tool",
}


def assert_paired_lifecycle(events: Sequence[AgentEvent]) -> None:
    """Assert every start is closed exactly once, properly nested.

    Deliberately independent of ``OpenScopeTracker`` so the invariant is
    not checked with the code under test.
    """

    open_scopes: list[str] = []
    for event in events:
        started = _START_SCOPES.get(event.type)
        if started is not None:
            open_scopes.append(started)
            continue
        closed = _CLOSE_SCOPES.get(event.type)
        if closed is not None:
            assert open_scopes, f"{event.type} closes nothing"
            assert open_scopes[-1] == closed, (
                f"{event.type} closes an open {open_scopes[-1]} scope"
            )
            open_scopes.pop()
    assert open_scopes == [], f"scopes left open: {open_scopes}"
    assert events[-1].type == "run_end"
    assert sum(event.type == "run_end" for event in events) == 1


class WeatherReport(BaseModel):
    """Sample result schema for typed-result attempt tests."""

    city: str
    temp_c: float


def test_tracker_commits_a_fully_paired_run_and_closes_nothing() -> None:
    """Track a paired attempt down to an empty stack and keep its outcome."""

    tracker = OpenScopeTracker()
    outcome = Completed(value="done")

    tracker.observe(AgentStartEvent(attempt=0))
    tracker.observe(AgentEndEvent(attempt=0))
    tracker.observe(RunEndEvent(outcome=outcome))

    assert tracker.committed_outcome == outcome
    assert tracker.close(Aborted()) == ()


def test_tracker_closes_open_scopes_innermost_first() -> None:
    """Synthesize interruptions in LIFO order, ending with the run end."""

    tracker = OpenScopeTracker()
    for event in (
        AgentStartEvent(attempt=2),
        TurnStartEvent(),
        ToolExecutionStartEvent(call_id="call_9", tool_name="read", arguments={}),
    ):
        tracker.observe(event)

    closing = tracker.close(Aborted())

    assert closing == (
        ToolExecutionInterruptedEvent(call_id="call_9"),
        TurnInterruptedEvent(),
        AgentInterruptedEvent(attempt=2),
        RunEndEvent(outcome=Aborted()),
    )
    assert tracker.committed_outcome == Aborted()


def test_tracker_tolerates_an_unmatched_end() -> None:
    """Ignore an end event whose scope was never opened."""

    tracker = OpenScopeTracker()
    tracker.observe(AgentEndEvent())

    assert tracker.close(Aborted()) == (RunEndEvent(outcome=Aborted()),)


def test_tracker_leaves_a_mismatched_tool_call_open_for_closure() -> None:
    """Sweep a tool scope whose end named a different call."""

    tracker = OpenScopeTracker()
    tracker.observe(AgentStartEvent())
    tracker.observe(TurnStartEvent())
    tracker.observe(
        ToolExecutionStartEvent(call_id="call_1", tool_name="read", arguments={})
    )
    tracker.observe(AgentEndEvent())

    closing = tracker.close(Aborted())

    assert ToolExecutionInterruptedEvent(call_id="call_1") in closing
    assert TurnInterruptedEvent() in closing


def test_tracker_close_never_duplicates_a_committed_run_end() -> None:
    """Close nothing when the producer already committed the run end."""

    tracker = OpenScopeTracker()
    outcome = Completed(value="done")
    tracker.observe(RunEndEvent(outcome=outcome))

    assert tracker.close(Aborted()) == ()
    assert tracker.committed_outcome == outcome


def test_provider_raise_before_stream_interrupts_the_attempt() -> None:
    """Pair the agent start when the provider dies before streaming."""

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

    assert_paired_lifecycle(events)
    assert _single(events, AgentInterruptedEvent) == AgentInterruptedEvent(attempt=0)
    run_outcome = _single(events, RunEndEvent).outcome
    assert isinstance(run_outcome, Failed)
    assert isinstance(run_outcome.cause, ExecutionFailure)
    assert run_outcome.cause.message == "connection refused"


def test_provider_raise_mid_stream_interrupts_message_and_turn() -> None:
    """Pair the open message and turn when the stream dies mid-message."""

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

    assert_paired_lifecycle(events)
    assert events[-4:] == [
        MessageInterruptedEvent(),
        TurnInterruptedEvent(),
        AgentInterruptedEvent(attempt=0),
        RunEndEvent(
            outcome=Failed(
                cause=ExecutionFailure(
                    origin="execution",
                    exception_type="ConnectionError",
                    message="connection reset",
                )
            )
        ),
    ]


def test_in_band_stream_error_keeps_producer_ends_and_fails_the_run() -> None:
    """Only synthesize the run end when the producer closed its own scopes."""

    runtime = _runtime(ProviderStreamMock([error_stream("resp_1", "boom")]).fn)
    session = runtime.session(session_id="in-band-error")

    async def _run() -> list[AgentEvent]:
        """Fail the run through an in-band stream error event."""

        run = await session.prompt("hello")
        assert await run.wait() == "failed"
        return [event async for event in run.events()]

    events = asyncio.run(_run())

    assert_paired_lifecycle(events)
    assert _single(events, MessageEndEvent).assistant_turn.status == "error"
    _single(events, AgentEndEvent)
    run_outcome = _single(events, RunEndEvent).outcome
    assert isinstance(run_outcome, Failed)
    assert isinstance(run_outcome.cause, ExecutionFailure)
    assert run_outcome.cause.origin == "turn"


def test_abort_during_tool_execution_interrupts_tool_turn_agent_and_run() -> None:
    """Pair the open tool execution when the run is aborted inside a tool."""

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

    assert_paired_lifecycle(events)
    assert events[-4:] == [
        ToolExecutionInterruptedEvent(call_id="call_1"),
        TurnInterruptedEvent(),
        AgentInterruptedEvent(attempt=0),
        RunEndEvent(outcome=Aborted()),
    ]


def test_abort_during_provider_stream_interrupts_the_open_message() -> None:
    """Pair the open message when the run is aborted mid-stream."""

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

    assert_paired_lifecycle(events)
    assert _single(events, MessageInterruptedEvent) == MessageInterruptedEvent()
    assert _single(events, RunEndEvent).outcome == Aborted()


def test_abort_before_first_tick_still_yields_a_paired_log() -> None:
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


def test_abort_after_committed_run_end_reclaims_a_stalled_source() -> None:
    """Cancel a stalled source after commit without relabeling the outcome."""

    async def _run() -> None:
        """Commit a run end, stall, then abort and read the terminal state."""

        async def _events() -> AsyncIterator[AgentEvent]:
            """Commit the outcome, then hold the run open forever."""

            yield RunEndEvent(outcome=Completed(value="done"))
            await asyncio.Event().wait()

        persisted: list[RunRecord] = []
        run = _bare_run(_events(), persisted=persisted)
        async for event in run.events():
            if isinstance(event, RunEndEvent):
                break

        run.abort()

        assert await run.wait() == "completed"
        assert run.outcome == Completed(value="done")
        assert persisted == [run.record]

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

    assert_paired_lifecycle(events)
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


def test_unmatched_producer_end_is_tolerated_and_the_run_still_closes() -> None:
    """Close a run whose producer ended a scope it never opened."""

    async def _run() -> None:
        """Drive a producer that ends an agent scope without starting one."""

        run = _bare_run(async_stream([AgentEndEvent()]), persisted=[])

        assert await run.wait() == "failed"
        assert run.exception is None
        events = [event async for event in run.events()]
        assert isinstance(events[0], RunStartEvent)
        assert isinstance(events[-1], RunEndEvent)

    asyncio.run(_run())


def test_clean_completion_without_run_end_fails_with_a_paired_log() -> None:
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
        assert_paired_lifecycle(events)
        assert persisted == [run.record]

    asyncio.run(_run())


def test_typed_result_attempts_each_close_before_the_next_starts() -> None:
    """Label and pair every typed-result attempt around the follow-up."""

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

    assert_paired_lifecycle(events)
    starts = [event for event in events if isinstance(event, AgentStartEvent)]
    ends = [event for event in events if isinstance(event, AgentEndEvent)]
    assert [event.attempt for event in starts] == [0, 1]
    assert [event.attempt for event in ends] == [0, 1]
    assert events.index(ends[0]) < events.index(starts[1])
    assert isinstance(events[-1], RunEndEvent)


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
    events: AsyncIterator[AgentEvent],
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
