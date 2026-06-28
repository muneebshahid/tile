"""Tests for the documented public import surface."""

import asyncio
from collections.abc import AsyncIterator, Sequence

from ori import (
    AgentRuntime,
    HistoryStore,
    InMemoryHistoryStore,
    Session,
    SessionBusyError,
    SessionNotFoundError,
)
from ori.events import AgentEndEvent, AgentEvent, MessageEndEvent, StreamFn
from ori.providers.openai import stream_api
from ori.types import (
    AsyncEventStream,
    ConversationItem,
    ProviderSource,
    ProviderStreamEvent,
    Reasoning,
    StreamDoneEvent,
    StreamStartEvent,
    TextBlock,
    ToolDefinition,
    ToolResult,
)


def test_documented_public_imports_run_fake_prompt() -> None:
    """Run one prompt using only documented public imports."""

    store: HistoryStore = InMemoryHistoryStore()
    runtime = AgentRuntime(
        stream_fn=_fake_stream_fn(),
        model="gpt-5.4",
        history_store=store,
        tools=[_fake_tool_definition()],
    )
    session: Session = runtime.session(session_id="public-imports")

    events = asyncio.run(_collect_prompt_events(session))

    assert isinstance(events[-1], AgentEndEvent)
    assert any(isinstance(event, MessageEndEvent) for event in events)
    assert len(session.history) == 2
    assert issubclass(SessionBusyError, RuntimeError)
    assert issubclass(SessionNotFoundError, KeyError)
    assert callable(stream_api)


def _fake_stream_fn() -> StreamFn:
    """Build a fake provider stream from public provider-neutral contracts."""

    async def _stream_fn(
        history: Sequence[ConversationItem],
        model: str,
        *,
        instructions: str,
        reasoning: Reasoning | None,
        tools: Sequence[ToolDefinition] | None,
    ) -> AsyncEventStream:
        """Return a deterministic assistant response."""

        _ = instructions, reasoning
        assert len(history) == 1
        assert model == "gpt-5.4"
        assert tools is not None
        return _stream_events(_assistant_response())

    return _stream_fn


def _assistant_response() -> tuple[ProviderStreamEvent, ...]:
    """Build a minimal successful provider event stream."""

    source = ProviderSource(provider="fake", model="gpt-5.4")
    return (
        StreamStartEvent(source=source, response_id="resp_public"),
        StreamDoneEvent(
            source=source,
            response_id="resp_public",
            stop_reason="stop",
            blocks=[TextBlock(text="public import response")],
        ),
    )


async def _stream_events(
    events: Sequence[ProviderStreamEvent],
) -> AsyncIterator[ProviderStreamEvent]:
    """Yield fake provider events."""

    for event in events:
        yield event


async def _collect_prompt_events(session: Session) -> list[AgentEvent]:
    """Collect all runtime events for one prompt."""

    return [event async for event in session.prompt("hello")]


async def _fake_tool() -> ToolResult:
    """Return a deterministic tool result for import smoke coverage."""

    return ToolResult.text("ok")


def _fake_tool_definition() -> ToolDefinition:
    """Build a tool definition from the public tool contract."""

    return ToolDefinition(
        name="fake_tool",
        description="Return a deterministic value.",
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        fn=_fake_tool,
    )
