"""Tests for guaranteed lifecycle pairing across failures and aborts."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar, cast
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from tile.events import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    LifecycleAborted,
    LifecycleFailed,
    MessageEndEvent,
    MessageStartEvent,
    RunEndEvent,
    RunStartEvent,
    StreamFn,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from tile.history import InMemoryHistoryStore
from tile.lifecycle import LifecycleLedger, LifecycleProtocolError
from tile.result import Aborted, Completed, ExecutionFailure, Failed
from tile.runs import InMemoryRunStore, RunRecord
from tile.runtime import AgentRuntime, Run
from tile.types.conversation import ConversationItem
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

FAILURE = ExecutionFailure(
    origin="execution",
    exception_type="ConnectionError",
    message="connection failed",
)

_START_SCOPES = {
    "run_start": "run",
    "agent_start": "agent",
    "turn_start": "turn",
    "message_start": "message",
    "tool_execution_start": "tool",
}
_END_SCOPES = {
    "run_end": "run",
    "agent_end": "agent",
    "turn_end": "turn",
    "message_end": "message",
    "tool_execution_end": "tool",
}


def assert_paired_lifecycle(events: Sequence[AgentEvent]) -> None:
    """Assert every start has exactly one properly nested end.

    Deliberately independent of ``LifecycleLedger`` so the invariant is not
    checked with the code under test.
    """

    open_scopes: list[str] = []
    for event in events:
        started = _START_SCOPES.get(event.type)
        if started is not None:
            open_scopes.append(started)
            continue
        ended = _END_SCOPES.get(event.type)
        if ended is not None:
            assert open_scopes, f"{event.type} closes nothing"
            assert open_scopes[-1] == ended, (
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


def test_ledger_accepts_a_complete_nested_run() -> None:
    """Accept a fully paired run and commit its outcome."""

    ledger = LifecycleLedger()
    outcome = Completed(value="done")

    for event in (
        RunStartEvent(),
        AgentStartEvent(attempt=0),
        TurnStartEvent(),
        MessageStartEvent(response_id="resp_1"),
        MessageEndEvent(),
        ToolExecutionStartEvent(call_id="call_1", tool_name="read", arguments={}),
        ToolExecutionEndEvent(call_id="call_1"),
        TurnEndEvent(),
        AgentEndEvent(attempt=0),
        RunEndEvent(outcome=outcome),
    ):
        ledger.observe(event)

    assert ledger.committed_outcome == outcome


@pytest.mark.parametrize(
    "events",
    [
        pytest.param([RunStartEvent(), RunStartEvent()], id="duplicate_run_start"),
        pytest.param([AgentStartEvent()], id="agent_start_without_run"),
        pytest.param([RunStartEvent(), TurnStartEvent()], id="turn_outside_agent"),
        pytest.param(
            [RunStartEvent(), AgentStartEvent(), TurnEndEvent()],
            id="turn_end_without_turn",
        ),
        pytest.param(
            [
                RunStartEvent(),
                AgentStartEvent(),
                TurnStartEvent(),
                MessageStartEvent(),
                ToolExecutionStartEvent(
                    call_id="call_1", tool_name="read", arguments={}
                ),
            ],
            id="tool_inside_open_message",
        ),
        pytest.param(
            [
                RunStartEvent(),
                AgentStartEvent(),
                TurnStartEvent(),
                ToolExecutionStartEvent(
                    call_id="call_1", tool_name="read", arguments={}
                ),
                ToolExecutionEndEvent(call_id="call_2"),
            ],
            id="tool_end_for_different_call",
        ),
        pytest.param(
            [
                RunStartEvent(),
                AgentStartEvent(attempt=0),
                AgentEndEvent(attempt=1),
            ],
            id="agent_end_for_different_attempt",
        ),
        pytest.param(
            [
                RunStartEvent(),
                AgentStartEvent(),
                RunEndEvent(outcome=Completed(value="done")),
            ],
            id="run_end_with_open_scopes",
        ),
        pytest.param(
            [
                RunStartEvent(),
                RunEndEvent(outcome=Completed(value="done")),
                AgentStartEvent(),
            ],
            id="event_after_committed_run_end",
        ),
    ],
)
def test_ledger_rejects_protocol_violations(events: list[AgentEvent]) -> None:
    """Reject unpaired, mismatched, or post-commit events immediately."""

    ledger = LifecycleLedger()

    with pytest.raises(LifecycleProtocolError):
        for event in events:
            ledger.observe(event)


def test_ledger_closes_open_scopes_innermost_first_on_abort() -> None:
    """Synthesize aborted ends in LIFO order, ending with the run end."""

    ledger = LifecycleLedger()
    for event in (
        RunStartEvent(),
        AgentStartEvent(attempt=2),
        TurnStartEvent(),
        ToolExecutionStartEvent(call_id="call_9", tool_name="read", arguments={}),
    ):
        ledger.observe(event)

    closing = ledger.close(Aborted())

    assert closing == (
        ToolExecutionEndEvent(call_id="call_9", termination=LifecycleAborted()),
        TurnEndEvent(termination=LifecycleAborted()),
        AgentEndEvent(attempt=2, termination=LifecycleAborted()),
        RunEndEvent(outcome=Aborted()),
    )
    assert ledger.committed_outcome == Aborted()


def test_ledger_closes_open_scopes_with_the_execution_failure() -> None:
    """Carry the run's execution failure on every synthesized end."""

    ledger = LifecycleLedger()
    outcome = Failed(cause=FAILURE)
    for event in (
        RunStartEvent(),
        AgentStartEvent(),
        TurnStartEvent(),
        MessageStartEvent(),
    ):
        ledger.observe(event)

    closing = ledger.close(outcome)

    assert closing == (
        MessageEndEvent(termination=LifecycleFailed(cause=FAILURE)),
        TurnEndEvent(termination=LifecycleFailed(cause=FAILURE)),
        AgentEndEvent(attempt=0, termination=LifecycleFailed(cause=FAILURE)),
        RunEndEvent(outcome=outcome),
    )


def test_ledger_close_never_duplicates_a_committed_run_end() -> None:
    """Close nothing when the producer already committed the run end."""

    ledger = LifecycleLedger()
    outcome = Completed(value="done")
    for event in (RunStartEvent(), RunEndEvent(outcome=outcome)):
        ledger.observe(event)

    assert ledger.close(Aborted()) == ()
    assert ledger.committed_outcome == outcome


def test_provider_raise_before_stream_closes_all_scopes() -> None:
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
    termination = _single(events, AgentEndEvent).termination
    assert isinstance(termination, LifecycleFailed)
    assert termination.cause.message == "connection refused"
    run_end = _single(events, RunEndEvent)
    assert run_end.outcome == Failed(cause=termination.cause)


def test_provider_raise_mid_stream_closes_message_and_turn() -> None:
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
    message_end = _single(events, MessageEndEvent)
    assert message_end.assistant_turn is None
    assert isinstance(message_end.termination, LifecycleFailed)
    turn_end = _single(events, TurnEndEvent)
    assert isinstance(turn_end.termination, LifecycleFailed)


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
    message_end = _single(events, MessageEndEvent)
    errored_turn = message_end.assistant_turn
    assert errored_turn is not None
    assert errored_turn.status == "error"
    assert message_end.termination.type == "lifecycle_completed"
    agent_end = _single(events, AgentEndEvent)
    assert agent_end.termination.type == "lifecycle_completed"
    run_outcome = _single(events, RunEndEvent).outcome
    assert isinstance(run_outcome, Failed)
    assert isinstance(run_outcome.cause, ExecutionFailure)
    assert run_outcome.cause.origin == "turn"


def test_abort_during_tool_execution_closes_tool_turn_agent_and_run() -> None:
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
    tool_end = _single(events, ToolExecutionEndEvent)
    assert tool_end.call_id == "call_1"
    assert tool_end.outcome is None
    assert tool_end.termination == LifecycleAborted()
    assert events[-4:] == [
        tool_end,
        TurnEndEvent(termination=LifecycleAborted()),
        AgentEndEvent(attempt=0, termination=LifecycleAborted()),
        RunEndEvent(outcome=Aborted()),
    ]


def test_abort_during_provider_stream_closes_open_message() -> None:
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
    message_end = _single(events, MessageEndEvent)
    assert message_end.assistant_turn is None
    assert message_end.termination == LifecycleAborted()
    run_end = _single(events, RunEndEvent)
    assert run_end.outcome == Aborted()


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
    message_end = _single(events, MessageEndEvent)
    assert message_end.assistant_turn is not None
    run_outcome = _single(events, RunEndEvent).outcome
    assert isinstance(run_outcome, Failed)
    assert isinstance(run_outcome.cause, ExecutionFailure)
    assert run_outcome.cause.message == "history unavailable"


def test_producer_protocol_violation_fails_the_run_and_excludes_the_event() -> None:
    """Fail the run on a protocol violation without logging the bad event."""

    async def _run() -> None:
        """Drive a producer that ends an agent scope it never opened."""

        run = _bare_run(
            [RunStartEvent(), AgentEndEvent()],
            persisted=[],
        )

        assert await run.wait() == "failed"
        assert isinstance(run.exception, LifecycleProtocolError)
        events = [event async for event in run.events()]
        assert not any(isinstance(event, AgentEndEvent) for event in events)
        assert_paired_lifecycle(events)
        run_end = _single(events, RunEndEvent)
        assert isinstance(run_end.outcome, Failed)

    asyncio.run(_run())


def test_clean_completion_without_run_end_is_a_protocol_failure() -> None:
    """Fail a producer that completes without committing a run end."""

    async def _run() -> None:
        """Drive a producer whose event source returns without a run end."""

        persisted: list[RunRecord] = []
        run = _bare_run(
            [RunStartEvent(), AgentStartEvent(), AgentEndEvent()],
            persisted=persisted,
        )

        assert await run.wait() == "failed"
        assert run.exception is None
        failure = run.failure
        assert failure is not None
        assert failure.exception_type == "LifecycleProtocolError"
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
    first_end = events.index(ends[0])
    second_start = events.index(starts[1])
    assert first_end < second_start
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


def _bare_run(events: list[AgentEvent], *, persisted: list[RunRecord]) -> Run:
    """Start a run directly over a scripted event source."""

    return Run(
        record=RunRecord(
            run_id="bare-run",
            session_id="bare-run",
            status="running",
            started_at=datetime.now(UTC),
            model="gpt-5.4",
        ),
        events=async_stream(events),
        on_done=lambda _: None,
        on_record=persisted.append,
        on_event=lambda _: None,
    )


def _weather_tool(fn: ToolFunction) -> ToolDefinition:
    """Build the deterministic weather tool around one implementation."""

    return city_tool("get_weather", "Return a deterministic weather report.", fn)


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
