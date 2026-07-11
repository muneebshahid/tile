"""Tests for the documented public import surface."""

import asyncio
from pathlib import Path
from collections.abc import AsyncIterator, Sequence

from tile import (
    AgentRuntime,
    HistoryStore,
    InMemoryHistoryStore,
    Run,
    Session,
    SessionBusyError,
    SessionNotFoundError,
)
from tile.events import AgentEndEvent, AgentEvent, MessageEndEvent, StreamFn
from tile.providers.openai import create_stream_api
from tile.types import (
    AsyncEventStream,
    ConversationItem,
    ProviderSource,
    ProviderStreamEvent,
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
        cwd=Path("."),
    )
    session: Session = runtime.session(session_id="public-imports")

    events = asyncio.run(_collect_prompt_events(session))

    assert isinstance(events[-1], AgentEndEvent)
    assert any(isinstance(event, MessageEndEvent) for event in events)
    assert len(session.history) == 2
    assert issubclass(SessionBusyError, RuntimeError)
    assert issubclass(SessionNotFoundError, KeyError)
    assert callable(create_stream_api)


def _fake_stream_fn() -> StreamFn:
    """Build a fake provider stream from public provider-neutral contracts."""

    async def _stream_fn(
        history: Sequence[ConversationItem],
        model: str,
        *,
        instructions: str,
        tools: Sequence[ToolDefinition] | None,
    ) -> AsyncEventStream:
        """Return a deterministic assistant response."""

        _ = instructions
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

    run: Run = await session.prompt("hello")
    return [event async for event in run.events()]


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
        },
        fn=_fake_tool,
    )
