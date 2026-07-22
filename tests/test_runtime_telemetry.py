"""Tests for private runtime lifecycle telemetry tracking and folding."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Literal

import pytest

from tile.events import (
    AgentEvent,
    LifecycleEventMetadata,
    MessageEndEvent,
    MessageStartEvent,
    RunEndEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from tile.result import (
    Aborted,
    AgentFailure,
    Completed,
    ExecutionFailure,
    Failed,
    RunOutcome,
)
from tile.runs import RunRecord, TerminalRunStatus
from tile.runtime.telemetry import (
    _LifecycleScopeTracker,
    _ScopeAccumulator,
    build_run_telemetry,
)
from tile.telemetry import CapturedRunException, RunTelemetryRecord
from tile.types.conversation import AssistantTurn
from tile.types.stream_events import ProviderSource
from tile.types.tool_execution import ToolExecutionOutcome
from tile.types.tools import ToolResult
from tile.types.usage import TokenUsage


def test_build_run_telemetry_folds_message_usage_and_tool_timing() -> None:
    """Build message/root usage and bounded tool data from one tool loop."""

    first_usage = _usage(10, 5, 15, cached=2, reasoning=1)
    second_usage = _usage(20, 7, 27, cached=4, reasoning=2)
    events = _stamped_events(
        _successful_tool_loop_events(first_usage, second_usage),
        ids=(
            "run-1",
            "agent-1",
            "turn-1",
            "message-1",
            "tool-1",
            "turn-2",
            "message-2",
        ),
        times=range(10, 150, 10),
    )

    telemetry = build_run_telemetry(events, _run_record(Completed(value="done")))

    assert telemetry.turn_count == 2
    assert telemetry.duration_ns == 130
    assert telemetry.token_usage == _usage(30, 12, 42, cached=6, reasoning=3)
    _assert_scope_usage(telemetry, first_usage, second_usage)
    assert telemetry.tools[0].model_dump() == {
        "tool_name": "get_weather",
        "call_count": 1,
        "completed_count": 1,
        "error_count": 0,
        "total_duration_ns": 10,
    }
    serialized = telemetry.model_dump_json()
    assert "Munich" not in serialized
    assert "sunny" not in serialized


def _assert_scope_usage(
    telemetry: RunTelemetryRecord,
    first_usage: TokenUsage,
    second_usage: TokenUsage,
) -> None:
    """Assert message/root attribution without intermediate-scope usage."""

    assert [scope.scope_type for scope in telemetry.scopes] == [
        "run",
        "agent",
        "turn",
        "message",
        "tool_execution",
        "turn",
        "message",
    ]
    assert telemetry.scopes[0].token_usage == telemetry.token_usage
    assert [
        scope.token_usage for scope in telemetry.scopes if scope.scope_type == "message"
    ] == [first_usage, second_usage]
    assert all(
        scope.token_usage is None
        for scope in telemetry.scopes
        if scope.scope_type in {"agent", "turn", "tool_execution"}
    )


def test_build_run_telemetry_aggregates_tool_errors_and_interruption() -> None:
    """Count normal and interrupted calls while retaining observed durations."""

    events = _stamped_events(
        _interrupted_tool_events(),
        ids=("run-1", "agent-1", "turn-1", "tool-1", "tool-2", "tool-3"),
        times=range(10, 100, 10),
    )

    telemetry = build_run_telemetry(
        events,
        _run_record(Aborted(), status="aborted"),
    )

    aggregate = telemetry.tools[0]
    assert aggregate.call_count == 3
    assert aggregate.completed_count == 2
    assert aggregate.error_count == 1
    assert aggregate.total_duration_ns == 30
    tool_scopes = [
        scope for scope in telemetry.scopes if scope.scope_type == "tool_execution"
    ]
    assert [scope.status for scope in tool_scopes] == [
        "completed",
        "completed",
        "interrupted",
    ]
    assert all(scope.token_usage is None for scope in tool_scopes)


def test_build_run_telemetry_uses_monotonic_run_end_to_sweep_open_scopes() -> None:
    """Ignore wall-clock duration and interrupt every scope left open at run end."""

    events = _stamped_events(
        _events("run_start", "agent_start", "turn_start", "message_start", "run_end"),
        ids=("run-1", "agent-1", "turn-1", "message-1"),
        times=range(100, 600, 100),
    )
    started_at = datetime(2026, 7, 20, tzinfo=UTC)
    record = _run_record(
        Aborted(),
        status="aborted",
        started_at=started_at,
        ended_at=started_at + timedelta(days=1),
    )

    telemetry = build_run_telemetry(events, record)

    assert telemetry.duration_ns == 400
    assert [scope.status for scope in telemetry.scopes] == [
        "completed",
        "interrupted",
        "interrupted",
        "interrupted",
    ]
    assert [scope.ended_monotonic_ns for scope in telemetry.scopes] == [
        500,
        500,
        500,
        500,
    ]


def test_build_run_telemetry_orders_primary_and_secondary_errors() -> None:
    """Derive the primary verdict once, then preserve sidecar observation order."""

    outcome = Failed(
        cause=ExecutionFailure(
            origin="execution",
            exception_type="ConnectionError",
            message="connection reset",
        )
    )
    primary = ConnectionError("connection reset")
    persistence = RuntimeError("run store unavailable")
    release = RuntimeError("release unavailable")

    telemetry = build_run_telemetry(
        _closed_run_events(outcome),
        _run_record(outcome, status="failed"),
        exceptions=(
            CapturedRunException(
                role="primary",
                stage="execution",
                error=primary,
            ),
            CapturedRunException(
                role="secondary",
                stage="run_persistence",
                error=persistence,
            ),
            CapturedRunException(
                role="secondary",
                stage="owner_release",
                error=release,
            ),
        ),
        context_receipt="ctx-123",
    )

    assert telemetry.turn_count == 0
    assert [(error.role, error.stage, error.message) for error in telemetry.errors] == [
        ("primary", "execution", "connection reset"),
        ("secondary", "run_persistence", "run store unavailable"),
        ("secondary", "owner_release", "release unavailable"),
    ]
    assert telemetry.context_receipt == "ctx-123"


def test_build_run_telemetry_maps_agent_failure_without_an_exception() -> None:
    """Represent an agent verdict without fabricating exception identity."""

    outcome = Failed(cause=AgentFailure(reason="insufficient evidence"))

    telemetry = build_run_telemetry(
        _closed_run_events(outcome),
        _run_record(outcome),
    )

    assert len(telemetry.errors) == 1
    assert telemetry.errors[0].kind == "agent_failure"
    assert telemetry.errors[0].exception_type is None
    assert telemetry.errors[0].message == "insufficient evidence"


def test_build_run_telemetry_rejects_a_tracking_disabled_event_log() -> None:
    """Refuse to emit a partial canonical record after lifecycle tracking failed."""

    tracker = _tracker(ids=("run-1",), times=range(1, 2))
    events = [
        tracker.stamp(AgentEvent(type="run_start")),
        AgentEvent(type="agent_start"),
        AgentEvent(type="run_end"),
    ]

    with pytest.raises(RuntimeError, match="lifecycle metadata"):
        build_run_telemetry(events, _run_record(Completed(value="done")))


def test_scope_accumulator_rejects_a_second_close() -> None:
    """Preserve the original lifecycle result when closure is attempted twice."""

    scope = _ScopeAccumulator(
        scope_id="message-1",
        parent_scope_id="turn-1",
        scope_type="message",
        started_monotonic_ns=10,
        operation_name="response-1",
    )
    scope.close(ended_monotonic_ns=20, status="completed")

    with pytest.raises(RuntimeError, match="already closed"):
        scope.close(ended_monotonic_ns=30, status="interrupted")

    assert scope.ended_monotonic_ns == 20
    assert scope.status == "completed"


def test_tracker_stamps_one_nested_scope_tree() -> None:
    """Reuse scope identity from each start through its matching end."""

    tracker = _tracker(
        ids=("run-1", "agent-1", "turn-1", "message-1", "tool-1"),
        times=range(10, 110, 10),
    )

    events = _stamp(
        tracker,
        "run_start",
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "tool_execution_start",
        "tool_execution_end",
        "turn_end",
        "agent_end",
        "run_end",
    )

    assert [_metadata_tuple(event) for event in events] == [
        ("run-1", None, 10),
        ("agent-1", "run-1", 20),
        ("turn-1", "agent-1", 30),
        ("message-1", "turn-1", 40),
        ("message-1", "turn-1", 50),
        ("tool-1", "turn-1", 60),
        ("tool-1", "turn-1", 70),
        ("turn-1", "agent-1", 80),
        ("agent-1", "run-1", 90),
        ("run-1", None, 100),
    ]


def test_tracker_gives_multiple_tools_distinct_sibling_scopes() -> None:
    """Parent every tool call to its turn without reusing sibling identity."""

    tracker = _tracker(
        ids=("run-1", "agent-1", "turn-1", "tool-1", "tool-2"),
        times=range(1, 10),
    )

    events = _stamp(
        tracker,
        "run_start",
        "agent_start",
        "turn_start",
        "tool_execution_start",
        "tool_execution_end",
        "tool_execution_start",
        "tool_execution_end",
    )

    first_start = _metadata(events[3])
    second_start = _metadata(events[5])
    assert first_start.scope_id == "tool-1"
    assert second_start.scope_id == "tool-2"
    assert first_start.parent_scope_id == second_start.parent_scope_id == "turn-1"


def test_tracker_gives_sequential_agent_attempts_distinct_scopes() -> None:
    """Keep typed-result attempts as sequential children of the root run."""

    tracker = _tracker(
        ids=("run-1", "agent-1", "agent-2"),
        times=range(1, 7),
    )

    events = _stamp(
        tracker,
        "run_start",
        "agent_start",
        "agent_end",
        "result_follow_up",
        "agent_start",
        "agent_end",
    )

    first_start = _metadata(events[1])
    second_start = _metadata(events[4])
    assert first_start.scope_id == "agent-1"
    assert second_start.scope_id == "agent-2"
    assert first_start.parent_scope_id == second_start.parent_scope_id == "run-1"
    assert events[3].lifecycle is None


def test_run_end_sweeps_open_scopes_without_later_duplicate_closure() -> None:
    """Close a torn-down hierarchy once at run end."""

    tracker = _tracker(
        ids=("run-1", "agent-1", "turn-1", "message-1"),
        times=range(1, 6),
    )
    events = _stamp(
        tracker,
        "run_start",
        "agent_start",
        "turn_start",
        "message_start",
        "run_end",
    )

    assert _metadata(events[-1]).scope_id == "run-1"
    with pytest.raises(RuntimeError, match="matching start"):
        tracker.stamp(AgentEvent(type="message_end"))


def test_future_interruption_events_close_existing_scope_identity() -> None:
    """Allow future producer-owned interruption events to reuse normal scopes."""

    tracker = _tracker(
        ids=("run-1", "agent-1", "turn-1", "message-1"),
        times=range(1, 9),
    )
    events = [
        tracker.stamp(AgentEvent(type="run_start")),
        tracker.stamp(AgentEvent(type="agent_start")),
        tracker.stamp(AgentEvent(type="turn_start")),
        tracker.stamp(AgentEvent(type="message_start")),
        tracker.stamp(_MessageInterruptedEvent()),
        tracker.stamp(_TurnInterruptedEvent()),
        tracker.stamp(_AgentInterruptedEvent()),
        tracker.stamp(AgentEvent(type="run_end")),
    ]

    assert _metadata(events[3]).scope_id == _metadata(events[4]).scope_id
    assert _metadata(events[2]).scope_id == _metadata(events[5]).scope_id
    assert _metadata(events[1]).scope_id == _metadata(events[6]).scope_id


def test_tracker_rejects_an_already_stamped_lifecycle_event() -> None:
    """Expose duplicate publication instead of silently accepting its metadata."""

    tracker = _tracker(ids=("run-1",), times=range(1, 3))
    run_start = tracker.stamp(AgentEvent(type="run_start"))

    with pytest.raises(RuntimeError, match="already stamped"):
        tracker.stamp(run_start)

    assert _metadata(tracker.stamp(AgentEvent(type="run_end"))).scope_id == "run-1"


def test_tracker_rejects_an_end_without_a_matching_start() -> None:
    """Expose an invalid lifecycle end instead of publishing it without metadata."""

    tracker = _tracker(ids=(), times=range(0))

    with pytest.raises(RuntimeError, match="matching start"):
        tracker.stamp(AgentEvent(type="message_end"))


def test_tracker_rejects_a_start_without_its_required_parent() -> None:
    """Expose malformed scope nesting at the event that introduces it."""

    tracker = _tracker(ids=(), times=range(0))

    with pytest.raises(RuntimeError, match="open parent"):
        tracker.stamp(AgentEvent(type="message_start"))


def test_tracker_rejects_an_unclassified_event_type() -> None:
    """Require new runtime event types to declare whether they own a scope."""

    tracker = _tracker(ids=(), times=range(0))

    with pytest.raises(RuntimeError, match="Unclassified"):
        tracker.stamp(AgentEvent(type="unknown_event"))


class _MessageInterruptedEvent(AgentEvent):
    """Representative future message interruption event."""

    type: Literal["message_interrupted"] = "message_interrupted"


class _TurnInterruptedEvent(AgentEvent):
    """Representative future turn interruption event."""

    type: Literal["turn_interrupted"] = "turn_interrupted"


class _AgentInterruptedEvent(AgentEvent):
    """Representative future agent interruption event."""

    type: Literal["agent_interrupted"] = "agent_interrupted"


def _tracker(
    *,
    ids: tuple[str, ...],
    times: range,
) -> _LifecycleScopeTracker:
    """Build a tracker over deterministic identity and time sequences."""

    id_values = iter(ids)
    time_values = iter(times)
    return _LifecycleScopeTracker(
        clock=time_values.__next__,
        scope_id_factory=id_values.__next__,
    )


def _stamp(
    tracker: _LifecycleScopeTracker,
    *event_types: str,
) -> list[AgentEvent]:
    """Stamp generic events in publication order."""

    return [tracker.stamp(AgentEvent(type=event_type)) for event_type in event_types]


def _metadata(event: AgentEvent) -> LifecycleEventMetadata:
    """Return required lifecycle metadata from a stamped event."""

    assert event.lifecycle is not None
    return event.lifecycle


def _metadata_tuple(event: AgentEvent) -> tuple[str, str | None, int]:
    """Return compact lifecycle metadata for sequence assertions."""

    metadata = _metadata(event)
    return metadata.scope_id, metadata.parent_scope_id, metadata.monotonic_ns


def _successful_tool_loop_events(
    first_usage: TokenUsage,
    second_usage: TokenUsage,
) -> list[AgentEvent]:
    """Build one successful provider-tool-provider event sequence."""

    first_turn = _assistant_turn("response-1")
    second_turn = _assistant_turn("response-2")
    outcome = ToolExecutionOutcome.from_result(
        call_id="call-1",
        tool_name="get_weather",
        result=ToolResult.text("Munich is sunny"),
    )
    return [
        *_events("run_start", "agent_start", "turn_start"),
        MessageStartEvent(response_id="response-1"),
        MessageEndEvent(assistant_turn=first_turn, token_usage=first_usage),
        ToolExecutionStartEvent(
            call_id="call-1",
            tool_name="get_weather",
            arguments={"city": "Munich"},
        ),
        ToolExecutionEndEvent(outcome=outcome),
        *_events("turn_end", "turn_start"),
        MessageStartEvent(response_id="response-2"),
        MessageEndEvent(assistant_turn=second_turn, token_usage=second_usage),
        *_events("turn_end", "agent_end"),
        RunEndEvent(outcome=Completed(value="done")),
    ]


def _interrupted_tool_events() -> list[AgentEvent]:
    """Build successful, handled-error, and interrupted calls to one tool."""

    success = ToolExecutionOutcome.from_result(
        call_id="call-1",
        tool_name="get_weather",
        result=ToolResult.text("sunny"),
    )
    error = ToolExecutionOutcome.from_error(
        call_id="call-2",
        tool_name="get_weather",
        message="weather unavailable",
    )
    return [
        *_events("run_start", "agent_start", "turn_start"),
        _tool_start("call-1"),
        ToolExecutionEndEvent(outcome=success),
        _tool_start("call-2"),
        ToolExecutionEndEvent(outcome=error),
        _tool_start("call-3"),
        RunEndEvent(outcome=Aborted()),
    ]


def _closed_run_events(outcome: RunOutcome) -> list[AgentEvent]:
    """Build a deterministically stamped root-only lifecycle."""

    return _stamped_events(
        [
            AgentEvent(type="run_start"),
            RunEndEvent(outcome=outcome),
        ],
        ids=("run-1",),
        times=range(10, 30, 10),
    )


def _stamped_events(
    events: Sequence[AgentEvent],
    *,
    ids: tuple[str, ...],
    times: range,
) -> list[AgentEvent]:
    """Stamp concrete events with deterministic lifecycle metadata."""

    tracker = _tracker(ids=ids, times=times)
    return [tracker.stamp(event) for event in events]


def _events(*event_types: str) -> list[AgentEvent]:
    """Build generic runtime events for lifecycle-only test positions."""

    return [AgentEvent(type=event_type) for event_type in event_types]


def _assistant_turn(response_id: str) -> AssistantTurn:
    """Build a completed assistant turn without replay content."""

    return AssistantTurn(
        source=ProviderSource(provider="test", model="test-model"),
        response_id=response_id,
    )


def _tool_start(call_id: str) -> ToolExecutionStartEvent:
    """Build one deterministic weather-tool start event."""

    return ToolExecutionStartEvent(
        call_id=call_id,
        tool_name="get_weather",
        arguments={"city": "Munich"},
    )


def _run_record(
    outcome: RunOutcome,
    *,
    status: TerminalRunStatus = "completed",
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
) -> RunRecord:
    """Build one finalized durable run summary."""

    start = started_at or datetime(2026, 7, 20, tzinfo=UTC)
    return RunRecord(
        run_id="run-record-1",
        session_id="session-1",
        status=status,
        started_at=start,
        ended_at=ended_at or start + timedelta(seconds=1),
        model="test-model",
        provider="test",
        outcome=outcome,
    )


def _usage(
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    *,
    cached: int,
    reasoning: int,
) -> TokenUsage:
    """Build provider-reported token usage."""

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached,
        reasoning_output_tokens=reasoning,
    )
