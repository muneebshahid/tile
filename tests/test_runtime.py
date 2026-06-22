"""Tests for runtime-owned sessions and in-memory history."""

import asyncio
from collections.abc import Sequence

import pytest

from agent.history import (
    InMemoryHistoryStore,
    SessionAlreadyExistsError,
    SessionNotFoundError,
)
from agent.runtime import AgentRuntime, Session
from agent.types import (
    AgentEndEvent,
    AgentEvent,
    MessageEndEvent,
    StreamFn,
    ToolExecutionEndEvent,
)
from ai.types.contracts import AsyncEventStream, Reasoning
from ai.types.conversation import ConversationItem, UserMessage
from ai.types.stream_events import (
    ProviderStreamEvent,
)
from ai.types.tools import ToolDefinition, ToolResult
from tests.support.async_streams import async_stream
from tests.support.agent_streams import (
    StreamInvocation,
    build_stream_fn,
    final_text_stream,
    tool_call_stream,
)
from tests.support.conversation_assertions import (
    expect_assistant_turn,
    expect_tool_result_turn,
    expect_user_message,
)


def _collect_prompt_events(
    runtime: AgentRuntime,
    session_id: str,
    content: str,
) -> list[AgentEvent]:
    """Collect events from one runtime session prompt."""

    async def _collect() -> list[AgentEvent]:
        """Collect prompt events from the async generator."""

        session = runtime.get_session(session_id)
        return [event async for event in session.prompt(content)]

    return asyncio.run(_collect())


def _collect_until_agent_end(
    session_id: str,
    runtime: AgentRuntime,
    content: str,
) -> list[AgentEvent]:
    """Collect prompt events and stop immediately after the terminal event."""

    async def _collect() -> list[AgentEvent]:
        """Collect events until the first agent end event is observed."""

        events: list[AgentEvent] = []
        session = runtime.get_session(session_id)
        async for event in session.prompt(content):
            events.append(event)
            if isinstance(event, AgentEndEvent):
                break
        return events

    return asyncio.run(_collect())


def _collect_until_message_end(
    session_id: str,
    runtime: AgentRuntime,
    content: str,
) -> list[AgentEvent]:
    """Collect prompt events and stop immediately after message completion."""

    async def _collect() -> list[AgentEvent]:
        """Collect events until the first message end event is observed."""

        events: list[AgentEvent] = []
        session = runtime.get_session(session_id)
        async for event in session.prompt(content):
            events.append(event)
            if isinstance(event, MessageEndEvent):
                break
        return events

    return asyncio.run(_collect())


def _collect_until_tool_execution_end(
    session_id: str,
    runtime: AgentRuntime,
    content: str,
) -> list[AgentEvent]:
    """Collect prompt events and stop after the first tool result is emitted."""

    async def _collect() -> list[AgentEvent]:
        """Collect events until the first tool execution end event is observed."""

        events: list[AgentEvent] = []
        session = runtime.get_session(session_id)
        async for event in session.prompt(content):
            events.append(event)
            if isinstance(event, ToolExecutionEndEvent):
                break
        return events

    return asyncio.run(_collect())


def _runtime_with_streams(
    streams: Sequence[Sequence[ProviderStreamEvent]],
    invocations: list[StreamInvocation],
    *,
    tools: Sequence[ToolDefinition] = (),
) -> AgentRuntime:
    """Build a runtime backed by queued fake provider streams."""

    stream_fn: StreamFn = build_stream_fn(streams, invocations)
    return AgentRuntime(stream_fn=stream_fn, model="gpt-5.4", tools=tools)


def _runtime_with_gated_streams(
    releases: Sequence[asyncio.Event],
    invocations: list[StreamInvocation],
) -> AgentRuntime:
    """Build a runtime whose provider streams wait for explicit release."""

    stream_fn: StreamFn = _build_gated_stream_fn(releases, invocations)
    return AgentRuntime(stream_fn=stream_fn, model="gpt-5.4")


def _build_gated_stream_fn(
    releases: Sequence[asyncio.Event],
    invocations: list[StreamInvocation],
) -> StreamFn:
    """Build a provider stream function that blocks each stream by index."""

    async def _stream_fn(
        history: Sequence[ConversationItem],
        model: str,
        *,
        instructions: str,
        reasoning: Reasoning | None,
        tools: Sequence[ToolDefinition] | None,
    ) -> AsyncEventStream:
        """Record one provider invocation and wait to release its stream."""

        index = len(invocations)
        invocations.append(
            StreamInvocation(
                history=tuple(history),
                model=model,
                instructions=instructions,
                reasoning=reasoning,
                tools=tuple(tools) if tools is not None else None,
            )
        )
        await releases[index].wait()
        return async_stream(final_text_stream(f"resp_{index}", f"answer {index}"))

    return _stream_fn


async def _consume_prompt(session: Session, content: str) -> list[AgentEvent]:
    """Collect every event from one session prompt."""

    return [event async for event in session.prompt(content)]


async def _wait_for_invocation_count(
    invocations: list[StreamInvocation],
    expected_count: int,
) -> None:
    """Wait briefly for async prompt work to reach a provider call."""

    for _ in range(20):
        if len(invocations) >= expected_count:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"Expected {expected_count} provider invocation(s).")


class FalsyHistoryStore(InMemoryHistoryStore):
    """History store that is falsey even when injected."""

    def __bool__(self) -> bool:
        """Return false to exercise explicit None defaulting."""

        return False


def _sample_tools() -> list[ToolDefinition]:
    """Build deterministic tool definitions for runtime tests."""

    return [
        ToolDefinition(
            name="get_weather",
            description="Return a deterministic weather report.",
            input_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
            fn=_get_weather,
        )
    ]


async def _get_weather(city: str) -> ToolResult:
    """Return deterministic weather text for runtime tests."""

    return ToolResult.text(f"{city}: sunny")


def test_runtime_creates_generated_and_explicit_sessions() -> None:
    """Create sessions with generated ids, explicit ids, and optional names."""

    invocations: list[StreamInvocation] = []
    runtime = _runtime_with_streams([], invocations)

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


def test_session_history_is_read_only_snapshot() -> None:
    """Expose defensive history copies without leaking mutable stored items."""

    store = InMemoryHistoryStore()
    runtime = AgentRuntime(
        stream_fn=build_stream_fn([], []),
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
        stream_fn=build_stream_fn([], []),
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


def test_session_prompt_persists_assistant_turn_at_message_end() -> None:
    """Persist assistant history as soon as the message is stable."""

    invocations: list[StreamInvocation] = []
    runtime = _runtime_with_streams(
        [final_text_stream("resp_one", "hello back")],
        invocations,
    )
    session = runtime.session(session_id="repo-debug", name="debug")

    events = _collect_until_message_end(
        runtime=runtime, session_id=session.id, content="hello"
    )

    agent_end = events[-1]
    assert isinstance(agent_end, MessageEndEvent)
    assert expect_user_message(session.history[0]).content == "hello"
    assert expect_assistant_turn(session.history[1]).response_id == "resp_one"
    assert expect_user_message(invocations[0].history[0]).content == "hello"


def test_session_prompt_persists_agent_end_before_consumer_stops() -> None:
    """Keep history persisted when a consumer stops at the terminal event."""

    invocations: list[StreamInvocation] = []
    runtime = _runtime_with_streams(
        [final_text_stream("resp_one", "hello back")],
        invocations,
    )
    session = runtime.session(session_id="early-stop")

    events = _collect_until_agent_end(session.id, runtime, "hello")

    assert isinstance(events[-1], AgentEndEvent)
    assert expect_user_message(session.history[0]).content == "hello"
    assert expect_assistant_turn(session.history[1]).response_id == "resp_one"


def test_session_prompt_replays_prior_history_on_next_prompt() -> None:
    """Include previous completed turns when prompting the same session again."""

    invocations: list[StreamInvocation] = []
    runtime = _runtime_with_streams(
        [
            final_text_stream("resp_first", "first answer"),
            final_text_stream("resp_second", "second answer"),
        ],
        invocations,
    )
    session = runtime.session(session_id="multi-turn")

    _collect_prompt_events(runtime, session.id, "first")
    _collect_prompt_events(runtime, session.id, "second")

    first_invocation_history = invocations[0].history
    second_invocation_history = invocations[1].history
    assert len(first_invocation_history) == 1
    assert expect_user_message(first_invocation_history[0]).content == "first"
    assert len(second_invocation_history) == 3
    assert expect_user_message(second_invocation_history[0]).content == "first"
    assert expect_assistant_turn(second_invocation_history[1]).response_id == (
        "resp_first"
    )
    assert expect_user_message(second_invocation_history[2]).content == "second"
    assert len(session.history) == 4
    assert expect_user_message(session.history[0]).content == "first"
    assert expect_assistant_turn(session.history[1]).response_id == "resp_first"
    assert expect_user_message(session.history[2]).content == "second"
    assert expect_assistant_turn(session.history[3]).response_id == "resp_second"


def test_session_prompt_serializes_overlapping_same_session_prompts() -> None:
    """Keep same-session prompts ordered when callers overlap."""

    async def _run() -> None:
        """Run overlapping prompts through one event loop."""

        releases = [asyncio.Event(), asyncio.Event()]
        invocations: list[StreamInvocation] = []
        runtime = _runtime_with_gated_streams(releases, invocations)
        session = runtime.session(session_id="overlap")

        first_task = asyncio.create_task(_consume_prompt(session, "first"))
        await _wait_for_invocation_count(invocations, 1)
        second_task = asyncio.create_task(_consume_prompt(session, "second"))
        await asyncio.sleep(0)

        assert len(invocations) == 1
        assert expect_user_message(session.history[0]).content == "first"
        assert len(session.history) == 1

        releases[0].set()
        await _wait_for_invocation_count(invocations, 2)
        second_request_history = invocations[1].history

        assert len(second_request_history) == 3
        assert expect_user_message(second_request_history[0]).content == "first"
        assert expect_assistant_turn(second_request_history[1]).response_id == "resp_0"
        assert expect_user_message(second_request_history[2]).content == "second"

        releases[1].set()
        await asyncio.gather(first_task, second_task)

        assert expect_user_message(session.history[0]).content == "first"
        assert expect_assistant_turn(session.history[1]).response_id == "resp_0"
        assert expect_user_message(session.history[2]).content == "second"
        assert expect_assistant_turn(session.history[3]).response_id == "resp_1"

    asyncio.run(_run())


def test_session_prompt_persists_tool_result_history() -> None:
    """Persist assistant tool calls, tool results, and final assistant turns."""

    invocations: list[StreamInvocation] = []
    runtime = _runtime_with_streams(
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
        invocations,
        tools=_sample_tools(),
    )
    session = runtime.session(session_id="tool-history")

    _collect_prompt_events(runtime, session.id, "check weather")
    _collect_prompt_events(runtime, session.id, "what happened?")

    assert len(session.history) == 6
    assert expect_user_message(session.history[0]).content == "check weather"
    assert expect_assistant_turn(session.history[1]).response_id == "resp_tool"
    tool_result = expect_tool_result_turn(session.history[2])
    assert tool_result.call_id == "call_weather"
    assert tool_result.tool_name == "get_weather"
    assert tool_result.is_error is False
    assert expect_assistant_turn(session.history[3]).response_id == "resp_final"
    assert expect_user_message(session.history[4]).content == "what happened?"
    assert expect_assistant_turn(session.history[5]).response_id == "resp_next"
    assert expect_tool_result_turn(invocations[1].history[2]).call_id == (
        "call_weather"
    )
    assert expect_tool_result_turn(invocations[2].history[2]).tool_name == (
        "get_weather"
    )
    assert expect_assistant_turn(invocations[2].history[3]).response_id == "resp_final"
    assert expect_user_message(invocations[2].history[4]).content == "what happened?"


def test_session_prompt_persists_tool_result_at_execution_end() -> None:
    """Persist tool result history as soon as tool execution is stable."""

    invocations: list[StreamInvocation] = []
    runtime = _runtime_with_streams(
        [
            tool_call_stream(
                response_id="resp_tool",
                call_id="call_weather",
                tool_name="get_weather",
                arguments={"city": "Munich"},
            ),
            final_text_stream("resp_final", "Munich is sunny."),
        ],
        invocations,
        tools=_sample_tools(),
    )
    session = runtime.session(session_id="tool-execution-history")

    events = _collect_until_tool_execution_end(session.id, runtime, "check weather")

    tool_execution_end = events[-1]
    assert isinstance(tool_execution_end, ToolExecutionEndEvent)
    assert len(session.history) == 3
    assert expect_user_message(session.history[0]).content == "check weather"
    assert expect_assistant_turn(session.history[1]).response_id == "resp_tool"
    assert expect_tool_result_turn(session.history[2]).call_id == "call_weather"
    assert tool_execution_end.outcome.tool_result_turn == session.history[2]


def test_runtime_keeps_session_histories_independent() -> None:
    """Prompt two sessions through one runtime without cross-session mutation."""

    invocations: list[StreamInvocation] = []
    runtime = _runtime_with_streams(
        [
            final_text_stream("resp_repo", "repo answer"),
            final_text_stream("resp_docs", "docs answer"),
        ],
        invocations,
    )
    repo = runtime.session(session_id="repo")
    docs = runtime.session(session_id="docs")

    _collect_prompt_events(runtime, repo.id, "fix tests")
    _collect_prompt_events(runtime, docs.id, "update docs")

    assert [expect_user_message(repo.history[0]).content] == ["fix tests"]
    assert [expect_user_message(docs.history[0]).content] == ["update docs"]
    assert expect_assistant_turn(repo.history[1]).response_id == "resp_repo"
    assert expect_assistant_turn(docs.history[1]).response_id == "resp_docs"
    assert expect_user_message(invocations[0].history[0]).content == "fix tests"
    assert expect_user_message(invocations[1].history[0]).content == "update docs"


def test_session_fork_copies_history_to_new_session() -> None:
    """Fork a session into a named target with copied completed history."""

    invocations: list[StreamInvocation] = []
    runtime = _runtime_with_streams(
        [final_text_stream("resp_first", "first answer")],
        invocations,
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

    invocations: list[StreamInvocation] = []
    runtime = _runtime_with_streams([], invocations)
    source = runtime.session(session_id="source")

    forked = source.fork(name="generated fork")

    assert forked.id != source.id
    assert forked.name == "generated fork"
    assert forked.history == source.history
    assert {session.id for session in runtime.sessions} == {"source", forked.id}


def test_session_fork_histories_diverge_independently() -> None:
    """Allow source and fork histories to diverge after sharing a prefix."""

    invocations: list[StreamInvocation] = []
    runtime = _runtime_with_streams(
        [
            final_text_stream("resp_first", "first answer"),
            final_text_stream("resp_source", "source answer"),
            final_text_stream("resp_fork", "fork answer"),
        ],
        invocations,
    )
    source = runtime.session(session_id="source")

    _collect_prompt_events(runtime, source.id, "first")
    forked = source.fork(session_id="fork")
    _collect_prompt_events(runtime, source.id, "source path")
    _collect_prompt_events(runtime, forked.id, "fork path")

    assert source.history[:2] == forked.history[:2]
    assert expect_user_message(source.history[2]).content == "source path"
    assert expect_user_message(forked.history[2]).content == "fork path"
    assert expect_assistant_turn(source.history[3]).response_id == "resp_source"
    assert expect_assistant_turn(forked.history[3]).response_id == "resp_fork"
    assert source.history != forked.history


def test_session_fork_history_copy_is_defensive() -> None:
    """Keep source and fork stored histories isolated from copied snapshots."""

    invocations: list[StreamInvocation] = []
    runtime = _runtime_with_streams(
        [final_text_stream("resp_first", "first answer")],
        invocations,
    )
    source = runtime.session(session_id="source")

    _collect_prompt_events(runtime, source.id, "first")
    forked = source.fork(session_id="fork")
    source_snapshot = source.history
    fork_snapshot = forked.history
    expect_user_message(source_snapshot[0]).content = "mutated source snapshot"
    expect_user_message(fork_snapshot[0]).content = "mutated fork snapshot"

    assert expect_user_message(source.history[0]).content == "first"
    assert expect_user_message(forked.history[0]).content == "first"


def test_session_fork_rejects_duplicate_target_session_id() -> None:
    """Reject fork targets that would overwrite an existing session."""

    invocations: list[StreamInvocation] = []
    runtime = _runtime_with_streams([], invocations)
    source = runtime.session(session_id="source")
    runtime.session(session_id="existing")

    with pytest.raises(SessionAlreadyExistsError, match="existing"):
        source.fork(session_id="existing")


def test_runtime_fork_session_rejects_missing_source_session() -> None:
    """Reject forks from unknown source sessions."""

    invocations: list[StreamInvocation] = []
    runtime = _runtime_with_streams([], invocations)

    with pytest.raises(SessionNotFoundError, match="missing"):
        runtime.fork_session(source_session_id="missing", target_session_id="fork")
