"""Tests for runtime-owned sessions and in-memory history."""

import asyncio
from collections.abc import Sequence

import pytest

from agent.history import InMemoryHistoryStore, SessionNotFoundError
from agent.runtime import AgentRuntime
from agent.types import AgentEndEvent, AgentEvent, StreamFn
from ai.types.conversation import (
    AssistantTurn,
    ConversationItem,
    ToolResultTurn,
    UserMessage,
)
from ai.types.stream_events import (
    ProviderSource,
    ProviderStreamEvent,
    StreamDoneEvent,
    StreamStartEvent,
    TextBlock,
    ToolCallBlock,
)
from ai.types.tools import JsonObject, ToolDefinition, ToolResult
from tests.support.agent_streams import StreamInvocation, build_stream_fn


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


def _final_text_stream(response_id: str, text: str) -> list[ProviderStreamEvent]:
    """Build a minimal provider stream that completes with text."""

    source = ProviderSource(provider="test", model="gpt-5.4")
    return [
        StreamStartEvent(source=source, response_id=response_id),
        StreamDoneEvent(
            source=source,
            response_id=response_id,
            stop_reason="stop",
            blocks=[TextBlock(text=text)],
        ),
    ]


def _tool_call_stream(
    *,
    response_id: str,
    call_id: str,
    tool_name: str,
    arguments: JsonObject,
) -> list[ProviderStreamEvent]:
    """Build a minimal provider stream that requests one tool call."""

    source = ProviderSource(provider="test", model="gpt-5.4")
    return [
        StreamStartEvent(source=source, response_id=response_id),
        StreamDoneEvent(
            source=source,
            response_id=response_id,
            stop_reason="tool_use",
            blocks=[
                ToolCallBlock(
                    call_id=call_id,
                    name=tool_name,
                    arguments=arguments,
                )
            ],
        ),
    ]


def _runtime_with_streams(
    streams: Sequence[Sequence[ProviderStreamEvent]],
    invocations: list[StreamInvocation],
    *,
    tools: Sequence[ToolDefinition] = (),
) -> AgentRuntime:
    """Build a runtime backed by queued fake provider streams."""

    stream_fn: StreamFn = build_stream_fn(streams, invocations)
    return AgentRuntime(stream_fn=stream_fn, model="gpt-5.4", tools=tools)


def _expect_user_message(item: ConversationItem) -> UserMessage:
    """Assert and return a user conversation item."""

    assert isinstance(item, UserMessage)
    return item


def _expect_assistant_turn(item: ConversationItem) -> AssistantTurn:
    """Assert and return an assistant conversation item."""

    assert isinstance(item, AssistantTurn)
    return item


def _expect_tool_result_turn(item: ConversationItem) -> ToolResultTurn:
    """Assert and return a tool result conversation item."""

    assert isinstance(item, ToolResultTurn)
    return item


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
    """Expose completed session history without exposing the mutable store list."""

    store = InMemoryHistoryStore()
    runtime = AgentRuntime(
        stream_fn=build_stream_fn([], []),
        model="gpt-5.4",
        history_store=store,
    )
    session = runtime.session(session_id="snapshot")

    store.append_history("snapshot", [UserMessage(content="hello")])
    history = session.history

    assert isinstance(history, tuple)
    assert history == (UserMessage(content="hello"),)
    assert store.get_history("snapshot") == (UserMessage(content="hello"),)


def test_history_store_rejects_unknown_session_writes() -> None:
    """Require sessions to be created before history can be appended."""

    store = InMemoryHistoryStore()

    with pytest.raises(SessionNotFoundError, match="Unknown session: missing"):
        store.append_history("missing", [UserMessage(content="hello")])


def test_session_prompt_persists_user_and_agent_items() -> None:
    """Persist the user message before the run and agent items after completion."""

    invocations: list[StreamInvocation] = []
    runtime = _runtime_with_streams(
        [_final_text_stream("resp_one", "hello back")],
        invocations,
    )
    session = runtime.session(session_id="repo-debug", name="debug")

    events = _collect_prompt_events(runtime, session.id, "hello")

    agent_end = events[-1]
    assert isinstance(agent_end, AgentEndEvent)
    assert len(agent_end.new_items) == 1
    assert _expect_user_message(session.history[0]).content == "hello"
    assert _expect_assistant_turn(session.history[1]).response_id == "resp_one"
    assert _expect_user_message(invocations[0].history[0]).content == "hello"


def test_session_prompt_persists_agent_end_before_consumer_stops() -> None:
    """Persist agent-produced items before yielding the terminal event."""

    invocations: list[StreamInvocation] = []
    runtime = _runtime_with_streams(
        [_final_text_stream("resp_one", "hello back")],
        invocations,
    )
    session = runtime.session(session_id="early-stop")

    events = _collect_until_agent_end(session.id, runtime, "hello")

    assert isinstance(events[-1], AgentEndEvent)
    assert _expect_user_message(session.history[0]).content == "hello"
    assert _expect_assistant_turn(session.history[1]).response_id == "resp_one"


def test_session_prompt_replays_prior_history_on_next_prompt() -> None:
    """Include previous completed turns when prompting the same session again."""

    invocations: list[StreamInvocation] = []
    runtime = _runtime_with_streams(
        [
            _final_text_stream("resp_first", "first answer"),
            _final_text_stream("resp_second", "second answer"),
        ],
        invocations,
    )
    session = runtime.session(session_id="multi-turn")

    _collect_prompt_events(runtime, session.id, "first")
    _collect_prompt_events(runtime, session.id, "second")

    first_invocation_history = invocations[0].history
    second_invocation_history = invocations[1].history
    assert len(first_invocation_history) == 1
    assert _expect_user_message(first_invocation_history[0]).content == "first"
    assert len(second_invocation_history) == 3
    assert _expect_user_message(second_invocation_history[0]).content == "first"
    assert _expect_assistant_turn(second_invocation_history[1]).response_id == (
        "resp_first"
    )
    assert _expect_user_message(second_invocation_history[2]).content == "second"
    assert len(session.history) == 4
    assert _expect_user_message(session.history[0]).content == "first"
    assert _expect_assistant_turn(session.history[1]).response_id == "resp_first"
    assert _expect_user_message(session.history[2]).content == "second"
    assert _expect_assistant_turn(session.history[3]).response_id == "resp_second"


def test_session_prompt_persists_tool_result_history() -> None:
    """Persist assistant tool calls, tool results, and final assistant turns."""

    invocations: list[StreamInvocation] = []
    runtime = _runtime_with_streams(
        [
            _tool_call_stream(
                response_id="resp_tool",
                call_id="call_weather",
                tool_name="get_weather",
                arguments={"city": "Munich"},
            ),
            _final_text_stream("resp_final", "Munich is sunny."),
            _final_text_stream("resp_next", "I remember the tool result."),
        ],
        invocations,
        tools=_sample_tools(),
    )
    session = runtime.session(session_id="tool-history")

    _collect_prompt_events(runtime, session.id, "check weather")
    _collect_prompt_events(runtime, session.id, "what happened?")

    assert len(session.history) == 6
    assert _expect_user_message(session.history[0]).content == "check weather"
    assert _expect_assistant_turn(session.history[1]).response_id == "resp_tool"
    tool_result = _expect_tool_result_turn(session.history[2])
    assert tool_result.call_id == "call_weather"
    assert tool_result.tool_name == "get_weather"
    assert tool_result.is_error is False
    assert _expect_assistant_turn(session.history[3]).response_id == "resp_final"
    assert _expect_user_message(session.history[4]).content == "what happened?"
    assert _expect_assistant_turn(session.history[5]).response_id == "resp_next"
    assert _expect_tool_result_turn(invocations[1].history[2]).call_id == (
        "call_weather"
    )
    assert _expect_tool_result_turn(invocations[2].history[2]).tool_name == (
        "get_weather"
    )
    assert _expect_assistant_turn(invocations[2].history[3]).response_id == "resp_final"
    assert _expect_user_message(invocations[2].history[4]).content == "what happened?"


def test_runtime_keeps_session_histories_independent() -> None:
    """Prompt two sessions through one runtime without cross-session mutation."""

    invocations: list[StreamInvocation] = []
    runtime = _runtime_with_streams(
        [
            _final_text_stream("resp_repo", "repo answer"),
            _final_text_stream("resp_docs", "docs answer"),
        ],
        invocations,
    )
    repo = runtime.session(session_id="repo")
    docs = runtime.session(session_id="docs")

    _collect_prompt_events(runtime, repo.id, "fix tests")
    _collect_prompt_events(runtime, docs.id, "update docs")

    assert [_expect_user_message(repo.history[0]).content] == ["fix tests"]
    assert [_expect_user_message(docs.history[0]).content] == ["update docs"]
    assert _expect_assistant_turn(repo.history[1]).response_id == "resp_repo"
    assert _expect_assistant_turn(docs.history[1]).response_id == "resp_docs"
    assert _expect_user_message(invocations[0].history[0]).content == "fix tests"
    assert _expect_user_message(invocations[1].history[0]).content == "update docs"
