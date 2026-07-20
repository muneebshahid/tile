"""Tests for private runtime lifecycle telemetry tracking."""

from typing import Literal

import pytest

from tile.events import AgentEvent, LifecycleEventMetadata
from tile.runtime.telemetry import _LifecycleScopeTracker


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
