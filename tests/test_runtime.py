"""Tests for runtime-owned sessions, task-owned runs, and in-memory history."""

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal, cast
from unittest.mock import AsyncMock

import pytest

from tile.history import (
    InMemoryHistoryStore,
    SessionAlreadyExistsError,
    SessionNotFoundError,
)
from tile.runs import InMemoryRunStore, RunRecord, RunStore, SQLiteRunStore
from tile.runtime import (
    AgentRuntime,
    Run,
    SessionBusyError,
    TurnFailedError,
)
from tile.events import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    MessageEndEvent,
    RunEndEvent,
    RunStartEvent,
    StreamFn,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from tile.types.contracts import AsyncEventStream
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
from tile.prompt import AUTO_MODE
from tile.result import Aborted, Completed, ExecutionFailure, Failed
from tests.support.agent_streams import (
    TEST_PROVIDER,
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
    run_store: RunStore | None = None,
) -> tuple[AgentRuntime, ProviderStreamMock]:
    """Build a runtime backed by queued fake provider streams."""

    provider = ProviderStreamMock(streams)
    runtime = AgentRuntime(
        stream_fn=provider.fn,
        model="gpt-5.4",
        tools=tools,
        cwd=cwd,
        history_store=InMemoryHistoryStore(),
        run_store=run_store if run_store is not None else InMemoryRunStore(),
    )
    return runtime, provider


def _runtime_with_gated_streams(
    releases: Sequence[asyncio.Event],
    *,
    run_store: RunStore | None = None,
) -> tuple[AgentRuntime, GatedProviderStreamMock]:
    """Build a runtime whose provider streams wait for explicit release."""

    provider = GatedProviderStreamMock(releases)
    return (
        AgentRuntime(
            stream_fn=provider.fn,
            model="gpt-5.4",
            cwd=Path("."),
            history_store=InMemoryHistoryStore(),
            run_store=run_store if run_store is not None else InMemoryRunStore(),
        ),
        provider,
    )


def _runtime_with_failing_provider(error: Exception) -> AgentRuntime:
    """Build a runtime whose provider call raises before streaming."""

    failing_mock = AsyncMock(side_effect=error)
    failing_mock.provider = TEST_PROVIDER
    failing_stream_fn = cast("StreamFn", failing_mock)
    return AgentRuntime(
        stream_fn=failing_stream_fn,
        model="gpt-5.4",
        cwd=Path("."),
        history_store=InMemoryHistoryStore(),
        run_store=InMemoryRunStore(),
    )


def _runtime_with_interrupted_stream(
    events: Sequence[ProviderStreamEvent],
    error: Exception,
) -> AgentRuntime:
    """Build a runtime whose provider stream raises after partial events."""

    interrupted_mock = AsyncMock(return_value=async_stream(events, error=error))
    interrupted_mock.provider = TEST_PROVIDER
    stream_fn = cast("StreamFn", interrupted_mock)
    return AgentRuntime(
        stream_fn=stream_fn,
        model="gpt-5.4",
        cwd=Path("."),
        history_store=InMemoryHistoryStore(),
        run_store=InMemoryRunStore(),
    )


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


class FailingRunStore(InMemoryRunStore):
    """Run store with switchable create and update failures for submission tests."""

    fail_creates: bool = False
    fail_updates: bool = False

    def create_run(self, record: RunRecord) -> None:
        """Create the record unless the test has enabled deterministic failure."""

        if self.fail_creates:
            raise RuntimeError("run store create unavailable")
        super().create_run(record)

    def update_run(self, record: RunRecord) -> None:
        """Update the record unless the test has enabled deterministic failure."""

        if self.fail_updates:
            raise RuntimeError("run store update unavailable")
        super().update_run(record)


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
        run_store=InMemoryRunStore(),
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
            history_store=InMemoryHistoryStore(),
            run_store=InMemoryRunStore(),
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


def test_runtime_persists_running_record_before_provider_execution() -> None:
    """Create the durable running record before the provider is invoked."""

    run_store = InMemoryRunStore()
    observed_records: list[RunRecord] = []

    class _ObservingStreamFn:
        """Capture durable state at the provider execution boundary."""

        provider = TEST_PROVIDER

        async def __call__(
            self,
            history: Sequence[ConversationItem],
            model: str,
            *,
            instructions: str,
            tools: Sequence[ToolDefinition] | None,
        ) -> AsyncEventStream:
            """Record visible run records, then stream one final text."""

            _ = history, model, instructions, tools
            observed_records.extend(run_store.list_runs("persist-before-provider"))
            return async_stream(final_text_stream("resp_one", "done"))

    runtime = AgentRuntime(
        stream_fn=_ObservingStreamFn(),
        model="gpt-5.4",
        history_store=InMemoryHistoryStore(),
        run_store=run_store,
        cwd=Path("."),
    )
    session = runtime.session(session_id="persist-before-provider")

    async def _run() -> Run:
        """Submit and complete one prompt."""

        run = await session.prompt("hello")
        assert await run.wait() == "completed"
        return run

    run = asyncio.run(_run())

    assert len(observed_records) == 1
    assert observed_records[0].run_id == run.id
    assert observed_records[0].status == "running"
    assert observed_records[0].ended_at is None
    assert observed_records[0].provider == TEST_PROVIDER


def test_prompt_leaves_history_unchanged_when_run_creation_fails() -> None:
    """Reject a prompt without persisting its user message when create fails."""

    run_store = FailingRunStore()
    run_store.fail_creates = True
    runtime, _ = _runtime_with_streams(
        [final_text_stream("resp_one", "recovered")],
        run_store=run_store,
    )
    session = runtime.session(session_id="create-fails")

    async def _run() -> None:
        """Fail one submission, then recover on the same session."""

        with pytest.raises(RuntimeError, match="run store create unavailable"):
            await session.prompt("rejected")

        assert len(session.history) == 0
        assert runtime.runs_for(session.id) == ()

        run_store.fail_creates = False
        recovery = await session.prompt("recover")
        assert await recovery.wait() == "completed"

    asyncio.run(_run())


def test_prompt_abandons_run_record_when_history_append_fails() -> None:
    """Fail the persisted record as a submission failure when history fails."""

    history_store = FailingHistoryStore()
    history_store.fail_appends = True
    run_store = FailingRunStore()
    provider = ProviderStreamMock([final_text_stream("resp_one", "recovered")])
    runtime = AgentRuntime(
        stream_fn=provider.fn,
        model="gpt-5.4",
        history_store=history_store,
        run_store=run_store,
        cwd=Path("."),
    )
    session = runtime.session(session_id="append-fails")

    async def _run() -> None:
        """Fail one submission, then verify the abandoned durable record."""

        with pytest.raises(RuntimeError, match="history unavailable"):
            await session.prompt("rejected")

        records = runtime.runs_for(session.id)
        assert len(records) == 1
        assert records[0].status == "failed"
        assert records[0].ended_at is not None
        assert records[0].provider == TEST_PROVIDER
        assert records[0].outcome == Failed(
            cause=ExecutionFailure(
                origin="submission",
                exception_type="RuntimeError",
                message="history unavailable",
            )
        )

        history_store.fail_appends = False
        recovery = await session.prompt("recover")
        assert await recovery.wait() == "completed"

    asyncio.run(_run())


def test_prompt_reraises_submission_error_when_abandonment_write_fails() -> None:
    """Propagate the submission failure even when the abandonment write fails."""

    history_store = FailingHistoryStore()
    history_store.fail_appends = True
    run_store = FailingRunStore()
    run_store.fail_updates = True
    provider = ProviderStreamMock([])
    runtime = AgentRuntime(
        stream_fn=provider.fn,
        model="gpt-5.4",
        history_store=history_store,
        run_store=run_store,
        cwd=Path("."),
    )
    session = runtime.session(session_id="abandonment-fails")

    async def _run() -> None:
        """Observe the original submission failure, not the store failure."""

        with pytest.raises(RuntimeError, match="history unavailable"):
            await session.prompt("rejected")

    asyncio.run(_run())

    records = runtime.runs_for(session.id)
    assert len(records) == 1
    assert records[0].status == "running"


def test_runtime_persists_success_failure_and_abort_records(tmp_path: Path) -> None:
    """Reopen every current terminal run ending from durable SQLite state."""

    database_path = tmp_path / "runs.db"
    run_store = SQLiteRunStore(database_path)
    completed, failed, aborted = asyncio.run(_exercise_terminal_runs(run_store))
    run_store.close()

    reopened = SQLiteRunStore(database_path)
    try:
        assert reopened.get_run(completed.id) == completed.record
        assert reopened.get_run(failed.id) == failed.record
        assert reopened.get_run(aborted.id) == aborted.record
    finally:
        reopened.close()

    _assert_terminal_run_records(completed, failed, aborted)


async def _exercise_terminal_runs(run_store: RunStore) -> tuple[Run, Run, Run]:
    """Complete, fail, and abort prompts against one shared run store."""

    completed_runtime, _ = _runtime_with_streams(
        [final_text_stream("resp_completed", "done")],
        run_store=run_store,
    )
    failed_runtime, _ = _runtime_with_streams(
        [error_stream("resp_failed", "provider failed")],
        run_store=run_store,
    )
    release = asyncio.Event()
    aborted_runtime, aborted_provider = _runtime_with_gated_streams(
        [release],
        run_store=run_store,
    )

    completed = await completed_runtime.session(session_id="completed-session").prompt(
        "complete"
    )
    assert await completed.wait() == "completed"
    assert completed_runtime.runs_for("completed-session") == (completed.record,)

    failed = await failed_runtime.session(session_id="failed-session").prompt("fail")
    assert await failed.wait() == "failed"

    aborted = await aborted_runtime.session(session_id="aborted-session").prompt(
        "abort"
    )
    await _wait_for_invocation_count(aborted_provider, 1)
    aborted.abort()
    assert await aborted.wait() == "aborted"
    return completed, failed, aborted


def _assert_terminal_run_records(completed: Run, failed: Run, aborted: Run) -> None:
    """Assert persisted terminal summaries retain their distinct semantics."""

    records = (completed.record, failed.record, aborted.record)

    assert [record.status for record in records] == [
        "completed",
        "failed",
        "aborted",
    ]
    assert completed.record.outcome == Completed(value="done")
    assert completed.record.provider == TEST_PROVIDER
    assert failed.record.outcome == Failed(
        cause=ExecutionFailure(
            origin="turn",
            exception_type="TurnFailedError",
            message="provider failed",
        )
    )
    assert failed.record.provider == TEST_PROVIDER
    assert aborted.record.outcome == Aborted()
    assert aborted.record.provider == TEST_PROVIDER
    assert all(record.ended_at is not None for record in records)
    assert all(
        record.ended_at is not None and record.started_at <= record.ended_at
        for record in records
    )


def test_run_keeps_aborted_status_when_owner_release_fails() -> None:
    """Keep the execution status when post-run history healing fails."""

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
            run_store=InMemoryRunStore(),
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

        assert await run.wait() == "aborted"
        assert run.failure is None
        assert run.exception is None
        assert run.error_message is None
        assert runtime.get_run(run.id).status == "aborted"

        store.fail_appends = False
        recovery = await session.prompt("recover")
        assert await recovery.wait() == "completed"

    asyncio.run(_run())


def test_runtime_keeps_live_truth_when_terminal_store_write_fails() -> None:
    """Keep the live handle authoritative while the store retains stale state."""

    run_store = FailingRunStore()
    runtime, _ = _runtime_with_streams(
        [final_text_stream("resp_one", "done")],
        run_store=run_store,
    )
    session = runtime.session(session_id="stale-store")

    async def _run() -> Run:
        """Complete one run whose terminal store write fails."""

        run_store.fail_updates = True
        run = await session.prompt("hello")
        assert await run.wait() == "completed"
        return run

    run = asyncio.run(_run())

    assert run.outcome == Completed(value="done")
    assert run.persistence_error is not None
    assert runtime.get_run(run.id).status == "running"


def test_run_outcome_available_while_run_still_running() -> None:
    """Expose the published end-event outcome before the terminal status lands."""

    async def _run() -> None:
        """Read the outcome between the end event and run finalization."""

        gate = asyncio.Event()

        async def _events() -> AsyncIterator[AgentEvent]:
            """Commit a final outcome, then hold the run open."""

            yield RunEndEvent(outcome=Completed(value="done"))
            await gate.wait()

        run = Run(
            record=RunRecord(
                run_id="mid-stream",
                session_id="mid-stream",
                status="running",
                started_at=datetime.now(UTC),
                model="gpt-5.4",
            ),
            events=_events(),
            on_done=lambda _: None,
            on_record=lambda _: None,
            on_event=lambda _: None,
        )

        async for event in run.events():
            if isinstance(event, RunEndEvent):
                break

        assert run.status == "running"
        assert run.outcome == Completed(value="done")

        gate.set()
        assert await run.wait() == "completed"
        assert run.outcome == Completed(value="done")

    asyncio.run(_run())


def test_run_keeps_completed_state_when_terminal_persistence_fails() -> None:
    """Expose a failed terminal write without rewriting status or outcome."""

    async def _run() -> None:
        """Complete a run whose terminal record write fails."""

        error = RuntimeError("run store unavailable")
        attempted_records: list[RunRecord] = []
        released: list[Run] = []

        def reject_record(record: RunRecord) -> None:
            """Record the one write attempt before failing it."""

            attempted_records.append(record)
            raise error

        run = Run(
            record=RunRecord(
                run_id="store-failure",
                session_id="store-failure",
                status="running",
                started_at=datetime.now(UTC),
                model="gpt-5.4",
            ),
            events=async_stream([RunEndEvent(outcome=Completed(value="done"))]),
            on_done=released.append,
            on_record=reject_record,
            on_event=lambda _: None,
        )

        assert await run.wait() == "completed"
        assert run.outcome == Completed(value="done")
        assert run.failure is None
        assert run.exception is None
        assert run.persistence_error is error
        assert [record.status for record in attempted_records] == ["completed"]
        assert released == [run]

    asyncio.run(_run())


def test_run_fails_when_event_pipeline_ends_without_outcome() -> None:
    """Fail a run whose event source ends without publishing a verdict."""

    async def _run() -> None:
        """Drain an outcome-less event source and read the terminal record."""

        persisted: list[RunRecord] = []
        run = Run(
            record=RunRecord(
                run_id="missing-outcome",
                session_id="missing-outcome",
                status="running",
                started_at=datetime.now(UTC),
                model="gpt-5.4",
            ),
            events=async_stream([]),
            on_done=lambda _: None,
            on_record=persisted.append,
            on_event=lambda _: None,
        )

        assert await run.wait() == "failed"
        assert run.status == "failed"
        failure = run.failure
        assert failure is not None
        assert failure.origin == "execution"
        assert failure.message == "The run ended without a committed run end event."
        assert run.outcome == Failed(cause=failure)
        assert run.exception is None
        assert persisted == [run.record]

    asyncio.run(_run())


def test_run_reraises_owner_release_control_exception_after_finishing() -> None:
    """Preserve control flow without rewriting the recorded terminal state."""

    class ControlSignal(BaseException):
        """Deterministic process-control signal for finalization testing."""

    async def _run() -> None:
        """Observe an interrupted owner callback through the event loop."""

        signal = ControlSignal("stop")
        reported: list[BaseException] = []
        persisted: list[RunRecord] = []

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
                record=RunRecord(
                    run_id="control-signal",
                    session_id="control-signal",
                    status="running",
                    started_at=datetime.now(UTC),
                    model="gpt-5.4",
                ),
                events=async_stream([RunEndEvent(outcome=Completed(value="done"))]),
                on_done=interrupt,
                on_record=persisted.append,
                on_event=lambda _: None,
            )
            assert await run.wait() == "completed"
            await asyncio.sleep(0)
        finally:
            loop.set_exception_handler(previous_handler)

        assert reported == [signal]
        assert run.status == "completed"
        assert run.failure is None
        assert run.exception is None
        assert run.persistence_error is None
        assert persisted == [run.record]

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

        assert isinstance(events[0], RunStartEvent)
        assert isinstance(events[1], AgentStartEvent)
        assert isinstance(events[-2], AgentEndEvent)
        assert isinstance(events[-1], RunEndEvent)
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
        assert isinstance(first[-1], RunEndEvent)

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
        assert run.outcome == Aborted()

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
def test_provider_death_converges_on_failed_run_state(
    make_runtime: Callable[[], AgentRuntime],
    expected_error: str,
    expected_origin: Literal["turn", "execution"],
    errored_turn_streamed: bool,
) -> None:
    """Converge every provider death channel on the same failed-run state.

    The run fails with an execution-failure outcome carrying the provider's
    message, anything streamed before the death stays visible on the run's
    event log, and session history keeps only the last stable state.
    """

    runtime = make_runtime()
    session = runtime.session(session_id="provider-death")

    async def _run() -> list[AgentEvent]:
        """Submit one prompt and collect its events after the run fails."""

        run = await session.prompt("hello")
        assert await run.wait() == "failed"
        assert run.error_message == expected_error
        expected_failure = ExecutionFailure(
            origin=expected_origin,
            exception_type=(
                "TurnFailedError" if expected_origin == "turn" else "ConnectionError"
            ),
            message=expected_error,
        )
        assert run.failure == expected_failure
        assert run.outcome == Failed(cause=expected_failure)
        if expected_origin == "turn":
            error = run.exception
            assert isinstance(error, TurnFailedError)
            assert error.turn is not None
            assert error.turn.error_message == expected_error
        else:
            assert isinstance(run.exception, ConnectionError)
        return [event async for event in run.events()]

    events = asyncio.run(_run())

    assert isinstance(events[0], RunStartEvent)
    assert isinstance(events[1], AgentStartEvent)
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


def test_runtime_sends_the_composed_system_prompt_to_the_provider(
    tmp_path: Path,
) -> None:
    """Compose auto mode, instructions, project context, and environment lines."""

    (tmp_path / "AGENTS.md").write_text("Project rules.", encoding="utf-8")
    provider = ProviderStreamMock([final_text_stream("resp_1", "hello back")])
    runtime = AgentRuntime(
        stream_fn=provider.fn,
        model="gpt-5.4",
        cwd=tmp_path,
        history_store=InMemoryHistoryStore(),
        run_store=InMemoryRunStore(),
        instructions="Base prompt.",
    )
    session = runtime.session(session_id="composed-prompt")

    async def _run() -> None:
        """Drive one prompt to completion."""

        run = await session.prompt("hello")
        assert await run.wait() == "completed"

    asyncio.run(_run())

    assert provider.instructions() == (
        f"{AUTO_MODE}\n\n"
        f"Base prompt.\n\n"
        f"Project rules.\n\n"
        f"Current date: {date.today().isoformat()}\n"
        f"Current working directory: {tmp_path.resolve()}"
    )
