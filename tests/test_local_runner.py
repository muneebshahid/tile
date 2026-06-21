"""Tests for the example local headless runner."""

import asyncio
import io
import json
from collections.abc import AsyncIterator, Sequence

import pytest

from agent.types import StreamFn
from ai.types.contracts import Reasoning
from ai.types.conversation import ConversationItem, UserMessage
from ai.types.stream_events import (
    ProviderSource,
    ProviderStreamEvent,
    StreamDoneEvent,
    StreamStartEvent,
)
from ai.types.tools import ToolDefinition
from examples import local_runner
from examples.local_runner import run_cli, run_prompt


def test_run_prompt_streams_agent_events_as_json_lines() -> None:
    """Run one prompt through the local runner with a deterministic agent."""

    invocations: list[tuple[ConversationItem, ...]] = []
    stream_fn = _build_stream_fn([_start_event(), _done_event()], invocations)
    output = io.StringIO()

    asyncio.run(
        run_prompt(
            "Hello from CLI",
            stream_fn=stream_fn,
            model="gpt-5.4",
            tools=[],
            output=output,
        )
    )

    lines = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [line["type"] for line in lines] == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "turn_end",
        "agent_end",
    ]
    assert len(invocations) == 1
    assert len(invocations[0]) == 1
    user_message = invocations[0][0]
    assert isinstance(user_message, UserMessage)
    assert user_message.content == "Hello from CLI"


def test_run_cli_rejects_empty_prompt() -> None:
    """Reject a missing prompt before constructing the default agent."""

    status = asyncio.run(run_cli(["   "]))

    assert status == 2


def test_run_cli_reads_prompt_from_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read a prompt from standard input when no prompt arguments are supplied."""

    prompts: list[str] = []

    async def _record_prompt(prompt: str) -> None:
        """Record the prompt passed by the CLI."""

        prompts.append(prompt)

    monkeypatch.setattr("sys.stdin", io.StringIO("Hello from stdin\n"))
    monkeypatch.setattr(local_runner, "run_prompt", _record_prompt)

    status = asyncio.run(run_cli([]))

    assert status == 0
    assert prompts == ["Hello from stdin"]


def _build_stream_fn(
    stream_events: Sequence[ProviderStreamEvent],
    invocations: list[tuple[ConversationItem, ...]],
) -> StreamFn:
    """Build a provider stream function that records supplied history."""

    async def _stream_fn(
        history: Sequence[ConversationItem],
        model: str,
        *,
        instructions: str,
        reasoning: Reasoning | None,
        tools: Sequence[ToolDefinition] | None,
    ) -> AsyncIterator[ProviderStreamEvent]:
        """Return the static event stream expected by ``run_agent``."""

        _ = history, model, instructions, reasoning, tools
        invocations.append(tuple(history))
        return _iter_stream_events(stream_events)

    return _stream_fn


def _iter_stream_events(
    stream_events: Sequence[ProviderStreamEvent],
) -> AsyncIterator[ProviderStreamEvent]:
    """Yield static stream events asynchronously."""

    async def _iterate() -> AsyncIterator[ProviderStreamEvent]:
        """Yield each configured stream event."""

        for event in stream_events:
            yield event

    return _iterate()


def _start_event() -> StreamStartEvent:
    """Build a deterministic stream start event."""

    return StreamStartEvent(
        source=_source(),
        response_id="resp_cli",
    )


def _done_event() -> StreamDoneEvent:
    """Build a deterministic stream completion event."""

    return StreamDoneEvent(
        source=_source(),
        response_id="resp_cli",
        stop_reason="stop",
    )


def _source() -> ProviderSource:
    """Build a deterministic provider source for runner tests."""

    return ProviderSource(provider="test", model="gpt-5.4")
