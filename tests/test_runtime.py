"""Tests for runtime-owned sessions, task-owned runs, and in-memory history."""

import asyncio
from collections.abc import Sequence
from typing import cast
from unittest.mock import AsyncMock

import pytest

from ori.history import (
    InMemoryHistoryStore,
    SessionAlreadyExistsError,
    SessionNotFoundError,
)
from ori.runtime import AgentRuntime, Run, SessionBusyError
from ori.events import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    MessageEndEvent,
    StreamFn,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from ori.types.conversation import UserMessage
from ori.types.stream_events import (
    ProviderStreamEvent,
    TextBlock,
)
from ori.types.tools import (
    ToolDefinition,
    ToolFunction,
    ToolResult,
    ToolTextContent,
)
from tests.support.agent_streams import (
    GatedProviderStreamMock,
    ProviderStreamMock,
    error_stream,
    final_text_stream,
    stream_done,
    stream_start,
    tool_call_stream,
)
from tests.support.conversation_assertions import (
    expect_assistant_turn,
    expect_tool_result_turn,
    expect_user_message,
)
from tests.support.tool_definitions import city_tool


def _collect_prompt_events(
    runtime: AgentRuntime,
    session_id: str,
    content: str,
) -> list[AgentEvent]:
    """Run one session prompt to completion and collect its events."""

    async def _collect() -> list[AgentEvent]:
        """Submit the prompt and drain its run event subscription."""

        session = runtime.get_session(session_id)
        run = await session.prompt(content)
        return [event async for event in run.events()]

    return asyncio.run(_collect())


async def _collect_run_events(run: Run) -> list[AgentEvent]:
    """Collect every event from one run subscription."""

    return [event async for event in run.events()]


async def _wait_for_invocation_count(
    provider: ProviderStreamMock,
    expected_count: int,
) -> None:
    """Wait briefly for async prompt work to reach a provider call."""

    for _ in range(20):
        if provider.await_count >= expected_count:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"Expected {expected_count} provider invocation(s).")


def _runtime_with_streams(
    streams: Sequence[Sequence[ProviderStreamEvent]],
    *,
    tools: Sequence[ToolDefinition] = (),
) -> tuple[AgentRuntime, ProviderStreamMock]:
    """Build a runtime backed by queued fake provider streams."""

    provider = ProviderStreamMock(streams)
    return AgentRuntime(stream_fn=provider.fn, model="gpt-5.4", tools=tools), provider


def _runtime_with_gated_streams(
    releases: Sequence[asyncio.Event],
) -> tuple[AgentRuntime, GatedProviderStreamMock]:
    """Build a runtime whose provider streams wait for explicit release."""

    provider = GatedProviderStreamMock(releases)
    return AgentRuntime(stream_fn=provider.fn, model="gpt-5.4"), provider


def _runtime_with_failing_provider(error: Exception) -> AgentRuntime:
    """Build a runtime whose provider call raises before streaming."""

    failing_stream_fn = cast("StreamFn", AsyncMock(side_effect=error))
    return AgentRuntime(stream_fn=failing_stream_fn, model="gpt-5.4")


class FalsyHistoryStore(InMemoryHistoryStore):
    """History store that is falsey even when injected."""

    def __bool__(self) -> bool:
        """Return false to exercise explicit None defaulting."""

        return False


def _sample_tools() -> list[ToolDefinition]:
    """Build deterministic tool definitions for runtime tests."""

    return [_weather_tool(_get_weather)]


def _weather_tool(fn: ToolFunction) -> ToolDefinition:
    """Build the deterministic weather tool around one implementation."""

    return city_tool(
        "get_weather",
        "Return a deterministic weather report.",
        fn,
    )


async def _get_weather(city: str) -> ToolResult:
    """Return deterministic weather text for runtime tests."""

    return ToolResult.text(f"{city}: sunny")


async def _raise_weather_error(city: str) -> ToolResult:
    """Raise a deterministic weather failure for runtime tests."""

    _ = city
    raise RuntimeError("weather unavailable")


def _failing_tool() -> ToolDefinition:
    """Build a deterministic failing tool definition for runtime tests."""

    return city_tool(
        "fail_weather",
        "Raise a deterministic weather failure.",
        _raise_weather_error,
    )


def test_runtime_creates_generated_and_explicit_sessions() -> None:
    """Create sessions with generated ids, explicit ids, and optional names."""

    runtime, _ = _runtime_with_streams([])

    generated = runtime.session(name="generated")
    explicit = runtime.session(session_id="known-session", name="debug")

    assert generated.id != explicit.id
    assert generated.name == "generated"
    assert explicit.id == "known-session"
    assert explicit.name == "debug"
    assert [session.id for session in runtime.sessions] == [
        generated.id,
        "known-session",
    ]


def test_runtime_get_session_rejects_unknown_id() -> None:
    """Raise a session lookup error for unknown session ids."""

    runtime, _ = _runtime_with_streams([])

    with pytest.raises(SessionNotFoundError, match="Unknown session: missing"):
        runtime.get_session("missing")


def test_session_history_is_read_only_snapshot() -> None:
    """Expose defensive history copies without leaking mutable stored items."""

    store = InMemoryHistoryStore()
    runtime = AgentRuntime(
        stream_fn=ProviderStreamMock([]).fn,
        model="gpt-5.4",
        history_store=store,
    )
    session = runtime.session(session_id="snapshot")
    user_message = UserMessage(content="hello")

    store.append_history("snapshot", [user_message])
    user_message.content = "mutated original"
    history = session.history
    first_item = expect_user_message(history[0])

    assert isinstance(history, tuple)
    assert first_item.content == "hello"
    first_item.content = "mutated snapshot"
    assert store.get_history("snapshot") == (UserMessage(content="hello"),)


def test_runtime_preserves_falsy_injected_history_store() -> None:
    """Use an injected history store even when the store is falsey."""

    store = FalsyHistoryStore()
    runtime = AgentRuntime(
        stream_fn=ProviderStreamMock([]).fn,
        model="gpt-5.4",
        history_store=store,
    )

    session = runtime.session(session_id="configured-store")

    assert session.id == "configured-store"
    assert store.get_session("configured-store").session_id == "configured-store"


def test_history_store_rejects_unknown_session_writes() -> None:
    """Require sessions to be created before history can be appended."""

    store = InMemoryHistoryStore()

    with pytest.raises(SessionNotFoundError, match="Unknown session: missing"):
        store.append_history("missing", [UserMessage(content="hello")])


def test_run_completes_and_reports_status() -> None:
    """Complete a submitted run and expose run identity and terminal status."""

    async def _run() -> None:
        """Submit one prompt and wait for its terminal status."""

        runtime, _ = _runtime_with_streams(
            [final_text_stream("resp_one", "hello back")],
        )
        session = runtime.session(session_id="run-status")

        run = await session.prompt("hello")

        assert run.session_id == "run-status"
        assert run.id
        assert await run.wait() == "completed"
        assert run.status == "completed"
        assert run.error_message is None

    asyncio.run(_run())


def test_run_events_replay_from_start_for_late_subscribers() -> None:
    """Replay the full event log to subscribers joining after completion."""

    async def _run() -> None:
        """Wait for run completion before subscribing."""

        runtime, _ = _runtime_with_streams(
            [final_text_stream("resp_one", "hello back")],
        )
        session = runtime.session(session_id="late-subscriber")

        run = await session.prompt("hello")
        await run.wait()
        events = await _collect_run_events(run)

        assert isinstance(events[0], AgentStartEvent)
        assert isinstance(events[-1], AgentEndEvent)
        assert any(isinstance(event, MessageEndEvent) for event in events)

    asyncio.run(_run())


def test_run_events_supports_multiple_subscribers() -> None:
    """Deliver the identical event sequence to concurrent subscribers."""

    async def _run() -> None:
        """Subscribe twice to the same run concurrently."""

        runtime, _ = _runtime_with_streams(
            [final_text_stream("resp_one", "hello back")],
        )
        session = runtime.session(session_id="fan-out")

        run = await session.prompt("hello")
        first, second = await asyncio.gather(
            _collect_run_events(run),
            _collect_run_events(run),
        )

        assert first == second
        assert isinstance(first[-1], AgentEndEvent)

    asyncio.run(_run())


def test_run_completes_when_subscriber_stops_early() -> None:
    """Keep executing and persisting after a subscriber stops consuming."""

    async def _run() -> None:
        """Abandon a subscription after the first event."""

        runtime, _ = _runtime_with_streams(
            [final_text_stream("resp_one", "hello back")],
        )
        session = runtime.session(session_id="early-stop")

        run = await session.prompt("hello")
        async for _ in run.events():
            break

        assert await run.wait() == "completed"
        session_history = session.history
        assert expect_user_message(session_history[0]).content == "hello"
        assert expect_assistant_turn(session_history[1]).response_id == "resp_one"

    asyncio.run(_run())


def test_run_abort_marks_run_aborted_and_frees_session() -> None:
    """Abort an active run and allow the session to accept the next prompt."""

    async def _run() -> None:
        """Abort a gated run that would otherwise never complete."""

        releases = [asyncio.Event(), asyncio.Event()]
        runtime, provider = _runtime_with_gated_streams(releases)
        session = runtime.session(session_id="abort")

        run = await session.prompt("first")
        await _wait_for_invocation_count(provider, 1)
        run.abort()

        assert await run.wait() == "aborted"
        assert run.status == "aborted"

        second = await session.prompt("second")
        releases[1].set()
        assert await second.wait() == "completed"

    asyncio.run(_run())


def test_run_abort_heals_unanswered_tool_calls() -> None:
    """Persist error results for tool calls left unanswered by an abort."""

    async def _run() -> None:
        """Abort a run while its tool call is still executing."""

        gate = asyncio.Event()

        async def _blocked_weather(city: str) -> ToolResult:
            """Wait for a release gate that never opens."""

            _ = city
            await gate.wait()
            raise AssertionError("Tool must not complete.")

        runtime, _ = _runtime_with_streams(
            [
                tool_call_stream(
                    response_id="resp_tool",
                    call_id="call_weather",
                    tool_name="get_weather",
                    arguments={"city": "Munich"},
                ),
                final_text_stream("resp_next", "answered later"),
            ],
            tools=[_weather_tool(_blocked_weather)],
        )
        session = runtime.session(session_id="abort-mid-tool")

        run = await session.prompt("check weather")
        async for event in run.events():
            if isinstance(event, ToolExecutionStartEvent):
                break
        run.abort()
        assert await run.wait() == "aborted"

        healed = expect_tool_result_turn(session.history[2])
        assert healed.call_id == "call_weather"
        assert healed.is_error is True
        content = healed.content[0]
        assert isinstance(content, ToolTextContent)
        assert content.text == "Tool execution did not complete."

        second = await session.prompt("try again")
        assert await second.wait() == "completed"

    asyncio.run(_run())


def test_run_failure_captures_error_and_frees_session() -> None:
    """Record provider call failures as run data instead of raising."""

    async def _run() -> None:
        """Submit a prompt whose provider call raises immediately."""

        runtime = _runtime_with_failing_provider(ConnectionError("connection refused"))
        session = runtime.session(session_id="provider-failure")

        run = await session.prompt("hello")

        assert await run.wait() == "failed"
        assert run.error_message == "connection refused"
        events = await _collect_run_events(run)
        assert isinstance(events[0], AgentStartEvent)

        session_history = session.history
        assert expect_user_message(session_history[0]).content == "hello"

        second = await session.prompt("again")
        assert await second.wait() == "failed"

    asyncio.run(_run())


def test_run_exposes_output_text_and_conversation_items() -> None:
    """Expose the run's produced conversation items and final message text."""

    async def _run() -> None:
        """Complete a tool-loop run and read its output from the handle."""

        runtime, _ = _runtime_with_streams(
            [
                tool_call_stream(
                    response_id="resp_tool",
                    call_id="call_weather",
                    tool_name="get_weather",
                    arguments={"city": "Munich"},
                ),
                final_text_stream("resp_final", "Munich is sunny."),
            ],
            tools=_sample_tools(),
        )
        session = runtime.session(session_id="run-output")

        run = await session.prompt("check weather")
        assert await run.wait() == "completed"

        assert run.output_text == "Munich is sunny."
        items = run.conversation_items
        assert len(items) == 3
        assert expect_assistant_turn(items[0]).response_id == "resp_tool"
        assert expect_tool_result_turn(items[1]).call_id == "call_weather"
        assert expect_assistant_turn(items[2]).response_id == "resp_final"

    asyncio.run(_run())


def test_run_output_is_empty_until_first_message_completes() -> None:
    """Report no output while the run has not completed an assistant message."""

    async def _run() -> None:
        """Inspect a gated run before and after its provider stream releases."""

        releases = [asyncio.Event()]
        runtime, provider = _runtime_with_gated_streams(releases)
        session = runtime.session(session_id="pending-output")

        run = await session.prompt("hello")
        await _wait_for_invocation_count(provider, 1)
        assert run.output_text is None
        assert run.conversation_items == ()

        releases[0].set()
        assert await run.wait() == "completed"
        assert run.output_text == "answer 0"
        assert expect_assistant_turn(run.conversation_items[0]).response_id == "resp_0"

    asyncio.run(_run())


def test_run_output_text_joins_text_blocks_with_blank_lines() -> None:
    """Join multiple text blocks of the final message with blank lines."""

    async def _run() -> None:
        """Complete a run whose final message carries two text blocks."""

        runtime, _ = _runtime_with_streams(
            [
                [
                    stream_start("resp_multi"),
                    stream_done(
                        "resp_multi",
                        blocks=[
                            TextBlock(text="part one"),
                            TextBlock(text="part two"),
                        ],
                    ),
                ]
            ],
        )
        session = runtime.session(session_id="multi-block-output")

        run = await session.prompt("hello")

        assert await run.wait() == "completed"
        assert run.output_text == "part one\n\npart two"

    asyncio.run(_run())


def test_session_prompt_persists_assistant_turn_at_message_end() -> None:
    """Persist assistant history before the message end event is published."""

    async def _run() -> None:
        """Assert persistence at the moment the subscriber observes the event."""

        runtime, provider = _runtime_with_streams(
            [final_text_stream("resp_one", "hello back")],
        )
        session = runtime.session(session_id="repo-debug", name="debug")

        run = await session.prompt("hello")
        async for event in run.events():
            if isinstance(event, MessageEndEvent):
                assert event.assistant_turn in session.history
        await run.wait()

        session_history = session.history
        assert expect_user_message(session_history[0]).content == "hello"
        assert expect_assistant_turn(session_history[1]).response_id == "resp_one"
        request_history = provider.history(0)
        assert expect_user_message(request_history[0]).content == "hello"

    asyncio.run(_run())


def test_session_prompt_persists_tool_result_at_execution_end() -> None:
    """Persist tool result history before the execution end event is published."""

    async def _run() -> None:
        """Assert persistence at the moment the subscriber observes the event."""

        runtime, _ = _runtime_with_streams(
            [
                tool_call_stream(
                    response_id="resp_tool",
                    call_id="call_weather",
                    tool_name="get_weather",
                    arguments={"city": "Munich"},
                ),
                final_text_stream("resp_final", "Munich is sunny."),
            ],
            tools=_sample_tools(),
        )
        session = runtime.session(session_id="tool-execution-history")

        run = await session.prompt("check weather")
        observed_tool_end = False
        async for event in run.events():
            if isinstance(event, ToolExecutionEndEvent):
                observed_tool_end = True
                assert event.outcome.tool_result_turn in session.history
        await run.wait()

        assert observed_tool_end
        session_history = session.history
        assert expect_user_message(session_history[0]).content == "check weather"
        assert expect_assistant_turn(session_history[1]).response_id == "resp_tool"
        assert expect_tool_result_turn(session_history[2]).call_id == "call_weather"

    asyncio.run(_run())


def test_session_prompt_replays_prior_history_on_next_prompt() -> None:
    """Include previous completed turns when prompting the same session again."""

    runtime, provider = _runtime_with_streams(
        [
            final_text_stream("resp_first", "first answer"),
            final_text_stream("resp_second", "second answer"),
        ],
    )
    session = runtime.session(session_id="multi-turn")

    _collect_prompt_events(runtime, session.id, "first")
    _collect_prompt_events(runtime, session.id, "second")

    first_invocation_history = provider.history(0)
    second_invocation_history = provider.history(1)
    assert len(first_invocation_history) == 1
    assert expect_user_message(first_invocation_history[0]).content == "first"
    assert len(second_invocation_history) == 3
    assert expect_user_message(second_invocation_history[0]).content == "first"
    assert expect_assistant_turn(second_invocation_history[1]).response_id == (
        "resp_first"
    )
    assert expect_user_message(second_invocation_history[2]).content == "second"

    session_history = session.history
    assert len(session_history) == 4
    assert expect_user_message(session_history[0]).content == "first"
    assert expect_assistant_turn(session_history[1]).response_id == "resp_first"
    assert expect_user_message(session_history[2]).content == "second"
    assert expect_assistant_turn(session_history[3]).response_id == "resp_second"


def test_session_prompt_replays_tool_history_on_later_prompt() -> None:
    """Include completed tool turns when prompting the same session later."""

    runtime, provider = _runtime_with_streams(
        [
            tool_call_stream(
                response_id="resp_tool",
                call_id="call_weather",
                tool_name="get_weather",
                arguments={"city": "Munich"},
            ),
            final_text_stream("resp_final", "Munich is sunny."),
            final_text_stream("resp_next", "I remember the tool result."),
        ],
        tools=_sample_tools(),
    )
    session = runtime.session(session_id="tool-history")

    _collect_prompt_events(runtime, session.id, "check weather")
    _collect_prompt_events(runtime, session.id, "what happened?")

    next_prompt_request_history = provider.history(2)
    assert len(next_prompt_request_history) == 5
    assert expect_user_message(next_prompt_request_history[0]).content == (
        "check weather"
    )
    assert expect_assistant_turn(next_prompt_request_history[1]).response_id == (
        "resp_tool"
    )
    assert expect_tool_result_turn(next_prompt_request_history[2]).tool_name == (
        "get_weather"
    )
    assert expect_assistant_turn(next_prompt_request_history[3]).response_id == (
        "resp_final"
    )
    assert expect_user_message(next_prompt_request_history[4]).content == (
        "what happened?"
    )


def test_session_prompt_rejects_overlapping_same_session_prompts() -> None:
    """Reject same-session prompt submission while a run is already active."""

    async def _run() -> None:
        """Submit overlapping prompts through one event loop."""

        releases = [asyncio.Event(), asyncio.Event()]
        runtime, _ = _runtime_with_gated_streams(releases)
        session = runtime.session(session_id="overlap")

        first = await session.prompt("first")
        with pytest.raises(SessionBusyError, match="overlap"):
            await session.prompt("second")

        blocked_session_history = session.history
        assert expect_user_message(blocked_session_history[0]).content == "first"
        assert len(blocked_session_history) == 1

        releases[0].set()
        assert await first.wait() == "completed"

        second = await session.prompt("second")
        releases[1].set()
        assert await second.wait() == "completed"

        completed_session_history = session.history
        assert expect_user_message(completed_session_history[0]).content == "first"
        assert expect_assistant_turn(completed_session_history[1]).response_id == (
            "resp_0"
        )
        assert expect_user_message(completed_session_history[2]).content == "second"
        assert expect_assistant_turn(completed_session_history[3]).response_id == (
            "resp_1"
        )

    asyncio.run(_run())


def test_session_prompt_persists_stream_error_history() -> None:
    """Persist provider stream failures as assistant error turns."""

    runtime, _ = _runtime_with_streams(
        [error_stream("resp_error", "Socket closed")],
    )
    session = runtime.session(session_id="stream-error")

    events = _collect_prompt_events(runtime, session.id, "hello")

    assert isinstance(events[-1], AgentEndEvent)

    session_history = session.history
    assert expect_user_message(session_history[0]).content == "hello"
    assistant_turn = expect_assistant_turn(session_history[1])
    assert assistant_turn.response_id == "resp_error"
    assert assistant_turn.status == "error"
    assert assistant_turn.stop_reason == "error"
    assert assistant_turn.error_message == "Socket closed"


def test_session_prompt_persists_tool_exception_history() -> None:
    """Persist tool exceptions as replayable error tool results."""

    runtime, _ = _runtime_with_streams(
        [
            tool_call_stream(
                response_id="resp_tool",
                call_id="call_weather",
                tool_name="fail_weather",
                arguments={"city": "Munich"},
            ),
            final_text_stream("resp_final", "Tool failed."),
        ],
        tools=[_failing_tool()],
    )
    session = runtime.session(session_id="tool-error")

    _collect_prompt_events(runtime, session.id, "check weather")

    session_history = session.history
    tool_result = expect_tool_result_turn(session_history[2])
    assert tool_result.call_id == "call_weather"
    assert tool_result.tool_name == "fail_weather"
    assert tool_result.is_error is True
    content = tool_result.content[0]
    assert isinstance(content, ToolTextContent)
    assert content.text == "weather unavailable"
    assert expect_assistant_turn(session_history[3]).response_id == "resp_final"


def test_runtime_keeps_session_histories_independent() -> None:
    """Prompt two sessions through one runtime without cross-session mutation."""

    runtime, provider = _runtime_with_streams(
        [
            final_text_stream("resp_repo", "repo answer"),
            final_text_stream("resp_docs", "docs answer"),
        ],
    )
    repo = runtime.session(session_id="repo")
    docs = runtime.session(session_id="docs")

    _collect_prompt_events(runtime, repo.id, "fix tests")
    _collect_prompt_events(runtime, docs.id, "update docs")

    repo_history = repo.history
    docs_history = docs.history
    assert [expect_user_message(repo_history[0]).content] == ["fix tests"]
    assert [expect_user_message(docs_history[0]).content] == ["update docs"]
    assert expect_assistant_turn(repo_history[1]).response_id == "resp_repo"
    assert expect_assistant_turn(docs_history[1]).response_id == "resp_docs"

    repo_request_history = provider.history(0)
    docs_request_history = provider.history(1)
    assert expect_user_message(repo_request_history[0]).content == "fix tests"
    assert expect_user_message(docs_request_history[0]).content == "update docs"


def test_session_fork_copies_history_to_new_session() -> None:
    """Fork a session into a named target with copied completed history."""

    runtime, _ = _runtime_with_streams(
        [final_text_stream("resp_first", "first answer")],
    )
    source = runtime.session(session_id="source", name="source session")

    _collect_prompt_events(runtime, source.id, "first")
    forked = source.fork(session_id="fork", name="forked session")

    assert forked.id == "fork"
    assert forked.name == "forked session"
    assert forked.history == source.history
    assert [session.id for session in runtime.sessions] == ["source", "fork"]


def test_session_fork_generates_target_session_id_by_default() -> None:
    """Generate a fork target id when one is not supplied."""

    runtime, _ = _runtime_with_streams([])
    source = runtime.session(session_id="source")

    forked = source.fork(name="generated fork")

    assert forked.id != source.id
    assert forked.name == "generated fork"
    assert forked.history == source.history
    assert {session.id for session in runtime.sessions} == {"source", forked.id}


def test_session_fork_histories_diverge_independently() -> None:
    """Allow source and fork histories to diverge after sharing a prefix."""

    runtime, _ = _runtime_with_streams(
        [
            final_text_stream("resp_first", "first answer"),
            final_text_stream("resp_source", "source answer"),
            final_text_stream("resp_fork", "fork answer"),
        ],
    )
    source = runtime.session(session_id="source")

    _collect_prompt_events(runtime, source.id, "first")
    forked = source.fork(session_id="fork")
    _collect_prompt_events(runtime, source.id, "source path")
    _collect_prompt_events(runtime, forked.id, "fork path")

    source_history = source.history
    forked_history = forked.history
    assert source_history[:2] == forked_history[:2]
    assert expect_user_message(source_history[2]).content == "source path"
    assert expect_user_message(forked_history[2]).content == "fork path"
    assert expect_assistant_turn(source_history[3]).response_id == "resp_source"
    assert expect_assistant_turn(forked_history[3]).response_id == "resp_fork"
    assert source_history != forked_history


def test_session_fork_history_copy_is_defensive() -> None:
    """Keep source and fork stored histories isolated from copied snapshots."""

    runtime, _ = _runtime_with_streams(
        [final_text_stream("resp_first", "first answer")],
    )
    source = runtime.session(session_id="source")

    _collect_prompt_events(runtime, source.id, "first")
    forked = source.fork(session_id="fork")
    source_snapshot = source.history
    fork_snapshot = forked.history
    expect_user_message(source_snapshot[0]).content = "mutated source snapshot"
    expect_user_message(fork_snapshot[0]).content = "mutated fork snapshot"

    source_history = source.history
    forked_history = forked.history
    assert expect_user_message(source_history[0]).content == "first"
    assert expect_user_message(forked_history[0]).content == "first"


def test_session_fork_rejects_duplicate_target_session_id() -> None:
    """Reject fork targets that would overwrite an existing session."""

    runtime, _ = _runtime_with_streams([])
    source = runtime.session(session_id="source")
    runtime.session(session_id="existing")

    with pytest.raises(SessionAlreadyExistsError, match="existing"):
        source.fork(session_id="existing")


def test_runtime_fork_session_rejects_missing_source_session() -> None:
    """Reject forks from unknown source sessions."""

    runtime, _ = _runtime_with_streams([])

    with pytest.raises(SessionNotFoundError, match="missing"):
        runtime.fork_session(source_session_id="missing", target_session_id="fork")


def test_tool_execution_start_precedes_persisted_result() -> None:
    """Observe tool execution start before its result lands in history."""

    async def _run() -> None:
        """Block the tool so the intermediate state is observable."""

        gate = asyncio.Event()

        async def _blocked_weather(city: str) -> ToolResult:
            """Wait for the release gate before answering."""

            await gate.wait()
            return ToolResult.text(f"{city}: sunny")

        runtime, _ = _runtime_with_streams(
            [
                tool_call_stream(
                    response_id="resp_tool",
                    call_id="call_weather",
                    tool_name="get_weather",
                    arguments={"city": "Munich"},
                ),
                final_text_stream("resp_final", "Munich is sunny."),
            ],
            tools=[_weather_tool(_blocked_weather)],
        )
        session = runtime.session(session_id="blocked-tool")

        run = await session.prompt("check weather")
        async for event in run.events():
            if isinstance(event, ToolExecutionStartEvent):
                break

        session_history = session.history
        assert len(session_history) == 2
        assert expect_assistant_turn(session_history[1]).response_id == "resp_tool"

        gate.set()
        assert await run.wait() == "completed"
        assert expect_tool_result_turn(session.history[2]).call_id == "call_weather"

    asyncio.run(_run())
