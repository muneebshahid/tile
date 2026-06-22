"""Tests for the example local headless runner."""

import asyncio
import io
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from agent.history import InMemoryHistoryStore
from ai.types.conversation import (
    AssistantTurn,
    ConversationItem,
    ToolResultTurn,
    UserMessage,
)
from ai.types.stream_events import (
    AssistantBlock,
    ProviderSource,
    ProviderStreamEvent,
    StreamDoneEvent,
    StreamStartEvent,
    StopReason,
    TextBlock,
    ToolCallBlock,
)
from ai.types.tools import JsonObject, ToolTextContent
from examples import local_runner
from examples.local_runner import run_cli, run_prompt
from tests.support.agent_streams import StreamInvocation, build_stream_fn


def test_run_prompt_streams_runtime_tool_flow_as_json_lines(tmp_path: Path) -> None:
    """Run one prompt through the local runtime with a deterministic file tool."""

    invocations, history_store, output = _run_runtime_tool_flow(tmp_path)

    _assert_runtime_event_sequence(output)
    _assert_provider_received_tool_result(invocations)
    _assert_runtime_persisted_completed_history(history_store)


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


def _run_runtime_tool_flow(
    tmp_path: Path,
) -> tuple[list[StreamInvocation], InMemoryHistoryStore, io.StringIO]:
    """Run the local runner through a fake provider and real read tool."""

    invocations: list[StreamInvocation] = []
    stream_fn = build_stream_fn(
        [
            _tool_call_stream(
                response_id="resp_read",
                call_id="call_read",
                tool_name="read",
                arguments={"path": "notes.txt"},
            ),
            _final_text_stream(response_id="resp_final", text="The note says hello."),
        ],
        invocations,
    )
    history_store = InMemoryHistoryStore()
    output = io.StringIO()
    (tmp_path / "notes.txt").write_text("hello from disk\n", encoding="utf-8")

    asyncio.run(
        run_prompt(
            "Read the note",
            stream_fn=stream_fn,
            model="gpt-5.4",
            history_store=history_store,
            cwd=tmp_path,
            output=output,
        )
    )
    return invocations, history_store, output


def _assert_runtime_event_sequence(output: io.StringIO) -> None:
    """Assert the local runner emitted the expected JSONL event sequence."""

    lines = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [line["type"] for line in lines] == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "tool_execution_start",
        "tool_execution_end",
        "turn_end",
        "turn_start",
        "message_start",
        "message_end",
        "turn_end",
        "agent_end",
    ]


def _assert_provider_received_tool_result(
    invocations: list[StreamInvocation],
) -> None:
    """Assert the second provider call received the read tool result."""

    assert len(invocations) == 2
    assert len(invocations[0].history) == 1
    assert _expect_user_message(invocations[0].history[0]).content == "Read the note"
    assert _expect_tool_text(invocations[1].history[2]) == ("hello from disk\n")


def _assert_runtime_persisted_completed_history(
    history_store: InMemoryHistoryStore,
) -> None:
    """Assert runtime history contains the completed local-runner turn."""

    sessions = history_store.list_sessions()
    assert len(sessions) == 1
    assert sessions[0].name == "local-runner"
    history = history_store.get_history(sessions[0].session_id)
    assert [_history_role(item) for item in history] == [
        "user",
        "assistant",
        "tool_result",
        "assistant",
    ]
    assert _expect_assistant_turn(history[1]).response_id == "resp_read"
    assert _expect_tool_result_turn(history[2]).call_id == "call_read"
    assert _expect_assistant_turn(history[3]).response_id == "resp_final"


def _tool_call_stream(
    *,
    response_id: str,
    call_id: str,
    tool_name: str,
    arguments: JsonObject,
) -> list[ProviderStreamEvent]:
    """Build a minimal provider stream that requests one tool call."""

    return [
        _start_event_with_response(response_id),
        _stream_done(
            response_id,
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


def _final_text_stream(
    *,
    response_id: str,
    text: str,
) -> list[ProviderStreamEvent]:
    """Build a minimal provider stream that returns final text."""

    return [
        _start_event_with_response(response_id),
        _stream_done(response_id, blocks=[TextBlock(text=text)]),
    ]


def _start_event_with_response(response_id: str) -> StreamStartEvent:
    """Build a deterministic stream start event for a response id."""

    return StreamStartEvent(source=_source(), response_id=response_id)


def _stream_done(
    response_id: str,
    *,
    stop_reason: StopReason = "stop",
    blocks: Sequence[AssistantBlock] = (),
) -> StreamDoneEvent:
    """Build a deterministic stream completion event."""

    return StreamDoneEvent(
        source=_source(),
        response_id=response_id,
        stop_reason=stop_reason,
        blocks=list(blocks),
    )


def _source() -> ProviderSource:
    """Build a deterministic provider source for runner tests."""

    return ProviderSource(provider="test", model="gpt-5.4")


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


def _expect_tool_text(item: ConversationItem) -> str:
    """Assert and return the first text block from a tool result item."""

    tool_result = _expect_tool_result_turn(item)
    content = tool_result.content[0]
    assert isinstance(content, ToolTextContent)
    return content.text


def _history_role(item: ConversationItem) -> str:
    """Return the provider-neutral conversation role for assertions."""

    return item.role
