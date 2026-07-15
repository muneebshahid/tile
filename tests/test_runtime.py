"""Tests for runtime-owned sessions, task-owned runs, and in-memory history."""

import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, cast
from unittest.mock import AsyncMock

import pytest

from tile.history import (
    InMemoryHistoryStore,
    SessionAlreadyExistsError,
    SessionNotFoundError,
)
from tile.runtime import (
    AgentRuntime,
    Run,
    RunFailure,
    SessionBusyError,
    TurnFailedError,
)
from tile.events import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    MessageEndEvent,
    StreamFn,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from tile.types.conversation import ConversationItem, UserMessage
from tile.types.stream_events import (
    ProviderStreamEvent,
    TextBlock,
)
from tile.types.tools import (
    ToolDefinition,
    ToolFunction,
    ToolInput,
    ToolResult,
    ToolTextContent,
)
from tile.result import Completed
from tests.support.agent_streams import (
    GatedProviderStreamMock,
    ProviderStreamMock,
    error_stream,
    final_text_stream,
    stream_done,
    stream_error,
    stream_start,
    tool_call_stream,
)
from tests.support.async_streams import async_stream
from tests.support.conversation_assertions import (
    expect_assistant_turn,
    expect_tool_result_turn,
    expect_user_message,
)
from tests.support.tool_definitions import CityInput, city_tool


class _NoInput(ToolInput):
    """Strict empty input for deterministic runtime tools."""


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
    cwd: Path = Path("."),
) -> tuple[AgentRuntime, ProviderStreamMock]:
    """Build a runtime backed by queued fake provider streams."""

    provider = ProviderStreamMock(streams)
    runtime = AgentRuntime(stream_fn=provider.fn, model="gpt-5.4", tools=tools, cwd=cwd)
    return runtime, provider


def _runtime_with_gated_streams(
    releases: Sequence[asyncio.Event],
) -> tuple[AgentRuntime, GatedProviderStreamMock]:
    """Build a runtime whose provider streams wait for explicit release."""

    provider = GatedProviderStreamMock(releases)
    return AgentRuntime(stream_fn=provider.fn, model="gpt-5.4", cwd=Path(".")), provider


def _runtime_with_failing_provider(error: Exception) -> AgentRuntime:
    """Build a runtime whose provider call raises before streaming."""

    failing_stream_fn = cast("StreamFn", AsyncMock(side_effect=error))
    return AgentRuntime(stream_fn=failing_stream_fn, model="gpt-5.4", cwd=Path("."))


def _runtime_with_interrupted_stream(
    events: Sequence[ProviderStreamEvent],
    error: Exception,
) -> AgentRuntime:
    """Build a runtime whose provider stream raises after partial events."""

    stream_fn = cast(
        "StreamFn",
        AsyncMock(return_value=async_stream(events, error=error)),
    )
    return AgentRuntime(stream_fn=stream_fn, model="gpt-5.4", cwd=Path("."))


class FalsyHistoryStore(InMemoryHistoryStore):
    """History store that is falsey even when injected."""

    def __bool__(self) -> bool:
        """Return false to exercise explicit None defaulting."""

        return False


class FailingHistoryStore(InMemoryHistoryStore):
    """History store with a switchable append failure for finalization tests."""

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


async def _get_weather(params: CityInput) -> ToolResult:
    """Return deterministic weather text for runtime tests."""

    return ToolResult.text(f"{params.city}: sunny")


async def _raise_weather_error(params: CityInput) -> ToolResult:
    """Raise a deterministic weather failure for runtime tests."""

    _ = params
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
        cwd=Path("."),
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
        cwd=Path("."),
    )

    session = runtime.session(session_id="configured-store")

    assert session.id == "configured-store"
    assert store.get_session("configured-store").session_id == "configured-store"


def test_runtime_binds_cwd_into_declaring_tools(tmp_path: Path) -> None:
    """Inject the resolved runtime cwd into tools that declare it, only those."""

    captured: dict[str, Path] = {}

    async def where(params: _NoInput, *, cwd: Path) -> ToolResult:
        """Capture the injected working directory."""

        _ = params
        captured["where"] = cwd
        return ToolResult.text(str(cwd))

    async def plain(params: _NoInput) -> ToolResult:
        """Run without any cwd involvement."""

        _ = params
        return ToolResult.text("plain ran")

    def _no_arg_tool(name: str, fn: ToolFunction) -> ToolDefinition:
        """Build a no-argument tool definition for the binding test."""

        return ToolDefinition(
            name=name,
            description=f"Exercise cwd binding via {name}.",
            input_model=_NoInput,
            fn=fn,
        )

    runtime, _ = _runtime_with_streams(
        [
            tool_call_stream(
                response_id="resp_where",
                call_id="call_where",
                tool_name="where",
                arguments={},
            ),
            tool_call_stream(
                response_id="resp_plain",
                call_id="call_plain",
                tool_name="plain",
                arguments={},
            ),
            final_text_stream("resp_done", "Both tools ran."),
        ],
        tools=[_no_arg_tool("where", where), _no_arg_tool("plain", plain)],
        cwd=tmp_path,
    )
    session = runtime.session(session_id="cwd-binding")

    async def _run() -> None:
        """Run one prompt that exercises both tools."""

        run = await session.prompt("run both tools")
        assert await run.wait() == "completed"
        events = [event async for event in run.events()]
        executions = [e for e in events if isinstance(e, ToolExecutionEndEvent)]
        assert [e.outcome.tool_result_turn.is_error for e in executions] == [
            False,
            False,
        ]

    asyncio.run(_run())

    assert captured["where"] == tmp_path.resolve()


def test_runtime_rejects_cwd_schema_property_on_injected_tool() -> None:
    """Reject tools that declare cwd for injection yet expose it to the model."""

    class CwdInput(ToolInput):
        """Invalidly expose the runtime-controlled cwd capability."""

        cwd: str

    async def clash(params: CwdInput, *, cwd: Path) -> ToolResult:
        """Declare cwd while the schema also exposes it."""

        _ = params
        return ToolResult.text(str(cwd))

    bad_tool = ToolDefinition(
        name="clash",
        description="Conflicting cwd declaration.",
        input_model=CwdInput,
        fn=clash,
    )

    with pytest.raises(ValueError, match="cwd"):
        AgentRuntime(
            stream_fn=ProviderStreamMock([]).fn,
            model="gpt-5.4",
            tools=[bad_tool],
            cwd=Path("."),
        )


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
        assert run.failure is None
        assert run.exception is None

    asyncio.run(_run())


def test_run_records_finalization_failure_and_releases_session() -> None:
    """Fail run finalization without stranding its session as active."""

    async def _run() -> None:
        """Abort during a tool call while history healing is unavailable."""

        gate = asyncio.Event()

        async def _blocked_weather(params: CityInput) -> ToolResult:
            """Wait until cancellation interrupts the tool execution."""

            _ = params
            await gate.wait()
            return ToolResult.text("unexpected")

        store = FailingHistoryStore()
        provider = ProviderStreamMock(
            [
                tool_call_stream(
                    response_id="resp_tool",
                    call_id="call_weather",
                    tool_name="get_weather",
                    arguments={"city": "Munich"},
                ),
                final_text_stream("resp_recovery", "Recovered."),
            ]
        )
        runtime = AgentRuntime(
            stream_fn=provider.fn,
            model="gpt-5.4",
            history_store=store,
            tools=[_weather_tool(_blocked_weather)],
            cwd=Path("."),
        )
        session = runtime.session(session_id="finalization-failure")

        run = await session.prompt("check weather")
        async for event in run.events():
            if isinstance(event, ToolExecutionStartEvent):
                break

        store.fail_appends = True
        run.abort()

        assert await run.wait() == "failed"
        assert run.failure == RunFailure(
            origin="finalization",
            exception_type="RuntimeError",
            message="history unavailable",
        )
        assert isinstance(run.exception, RuntimeError)
        assert run.error_message == "history unavailable"

        store.fail_appends = False
        recovery = await session.prompt("recover")
        assert await recovery.wait() == "completed"

    asyncio.run(_run())


def test_run_reraises_finalization_control_exception_after_finishing() -> None:
    """Preserve control flow after recording terminal run diagnostics."""

    class ControlSignal(BaseException):
        """Deterministic process-control signal for finalization testing."""

    async def _run() -> None:
        """Observe an interrupted owner callback through the event loop."""

        signal = ControlSignal("stop")
        reported: list[BaseException] = []

        def interrupt(_: Run) -> None:
            """Interrupt owner notification with a control exception."""

            raise signal

        def capture_exception(
            _: asyncio.AbstractEventLoop,
            context: dict[str, object],
        ) -> None:
            """Capture a control exception re-raised by a done callback."""

            error = context.get("exception")
            if isinstance(error, BaseException):
                reported.append(error)

        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(capture_exception)
        try:
            run = Run(
                run_id="control-signal",
                session_id="control-signal",
                events=async_stream([]),
                on_done=interrupt,
            )
            assert await run.wait() == "failed"
            await asyncio.sleep(0)
        finally:
            loop.set_exception_handler(previous_handler)

        assert reported == [signal]
        assert run.status == "failed"
        assert run.failure == RunFailure(
            origin="finalization",
            exception_type="ControlSignal",
            message="stop",
        )
        assert run.exception is signal

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
        assert run.outcome is None

        second = await session.prompt("second")
        releases[1].set()
        assert await second.wait() == "completed"

    asyncio.run(_run())


def test_run_abort_heals_unanswered_tool_calls() -> None:
    """Persist error results for tool calls left unanswered by an abort."""

    async def _run() -> None:
        """Abort a run while its tool call is still executing."""

        gate = asyncio.Event()

        async def _blocked_weather(params: CityInput) -> ToolResult:
            """Wait for a release gate that never opens."""

            _ = params
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


def _in_band_error_runtime() -> AgentRuntime:
    """Build a runtime whose provider stream ends with an in-band error event."""

    runtime, _ = _runtime_with_streams(
        [
            [
                stream_start("resp_error"),
                stream_error(
                    "resp_error",
                    "Socket closed",
                    blocks=[TextBlock(text="Munich is")],
                ),
            ]
        ],
    )
    return runtime


def _raise_before_stream_runtime() -> AgentRuntime:
    """Build a runtime whose provider call raises before streaming."""

    return _runtime_with_failing_provider(ConnectionError("connection refused"))


def _raise_mid_stream_runtime() -> AgentRuntime:
    """Build a runtime whose provider stream raises after starting."""

    return _runtime_with_interrupted_stream(
        [stream_start("resp_error")],
        ConnectionError("connection reset"),
    )


@pytest.mark.parametrize(
    ("make_runtime", "expected_error", "expected_origin", "errored_turn_streamed"),
    [
        pytest.param(
            _in_band_error_runtime,
            "Socket closed",
            "turn",
            True,
            id="in_band_stream_error",
        ),
        pytest.param(
            _raise_before_stream_runtime,
            "connection refused",
            "execution",
            False,
            id="raise_before_stream",
        ),
        pytest.param(
            _raise_mid_stream_runtime,
            "connection reset",
            "execution",
            False,
            id="raise_mid_stream",
        ),
    ],
)
def test_provider_death_fails_run_without_outcome_or_history(
    make_runtime: Callable[[], AgentRuntime],
    expected_error: str,
    expected_origin: Literal["turn", "execution"],
    errored_turn_streamed: bool,
) -> None:
    """Converge every provider death channel on the same failed-run state.

    The run fails with the provider's message and no outcome, anything
    streamed before the death stays visible on the run's event log, and
    session history keeps only the last stable state.
    """

    runtime = make_runtime()
    session = runtime.session(session_id="provider-death")

    async def _run() -> list[AgentEvent]:
        """Submit one prompt and collect its events after the run fails."""

        run = await session.prompt("hello")
        assert await run.wait() == "failed"
        assert run.error_message == expected_error
        assert run.failure == RunFailure(
            origin=expected_origin,
            exception_type=(
                "TurnFailedError" if expected_origin == "turn" else "ConnectionError"
            ),
            message=expected_error,
        )
        assert run.outcome is None
        if expected_origin == "turn":
            error = run.exception
            assert isinstance(error, TurnFailedError)
            assert error.turn is not None
            assert error.turn.error_message == expected_error
        else:
            assert isinstance(run.exception, ConnectionError)
        return [event async for event in run.events()]

    events = asyncio.run(_run())

    assert isinstance(events[0], AgentStartEvent)
    streamed_turns = [
        event.assistant_turn for event in events if isinstance(event, MessageEndEvent)
    ]
    if errored_turn_streamed:
        assert [turn.status for turn in streamed_turns] == ["error"]
        assert streamed_turns[0].blocks == [TextBlock(text="Munich is")]
    else:
        assert streamed_turns == []

    session_history = session.history
    assert len(session_history) == 1
    assert expect_user_message(session_history[0]).content == "hello"


def test_session_prompt_recovers_after_stream_error() -> None:
    """Replay clean history on the prompt after an in-band stream failure."""

    runtime, provider = _runtime_with_streams(
        [
            error_stream("resp_error", "Socket closed"),
            final_text_stream("resp_retry", "Recovered."),
        ]
    )
    session = runtime.session(session_id="stream-error-recovery")

    async def _run() -> None:
        """Fail one prompt, then complete the next on the same session."""

        failed = await session.prompt("hello")
        assert await failed.wait() == "failed"

        second = await session.prompt("try again")
        assert await second.wait() == "completed"
        assert second.outcome == Completed(value="Recovered.")

    asyncio.run(_run())

    assert provider.history(1) == (
        UserMessage(content="hello"),
        UserMessage(content="try again"),
    )


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

        async def _blocked_weather(params: CityInput) -> ToolResult:
            """Wait for the release gate before answering."""

            await gate.wait()
            return ToolResult.text(f"{params.city}: sunny")

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
