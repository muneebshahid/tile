"""Tests for run-owned mechanics: submission, finalization, and release."""

import asyncio
from collections.abc import AsyncGenerator, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from tile.events import StreamFn
from tile.history import InMemoryHistoryStore
from tile.result import Completed, Failed
from tile.runs import InMemoryRunStore, RunRecord
from tile.runtime.execution import _ExecutionDependencies
from tile.runtime.run import Run, _RunDependencies, _RunSpec
from tile.tool_executor import ToolExecutor
from tile.types.conversation import ConversationItem
from tile.types.stream_events import ProviderStreamEvent
from tests.support.agent_streams import ProviderStreamMock, final_text_stream


class _FailingHistoryStore(InMemoryHistoryStore):
    """History store with a switchable append failure."""

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


class _FailingRunStore(InMemoryRunStore):
    """Run store with a switchable update failure."""

    fail_updates: bool = False

    def update_run(self, record: RunRecord) -> None:
        """Update the record unless the test has enabled deterministic failure."""

        if self.fail_updates:
            raise RuntimeError("run store update unavailable")
        super().update_run(record)


def _deps(
    streams: Sequence[Sequence[ProviderStreamEvent]],
    *,
    session_id: str,
    history_store: InMemoryHistoryStore | None = None,
    run_store: InMemoryRunStore | None = None,
) -> _RunDependencies:
    """Build run dependencies over fake stores with the session prepared."""

    return _deps_for_stream_fn(
        ProviderStreamMock(streams).fn,
        session_id=session_id,
        history_store=history_store,
        run_store=run_store,
    )


def _deps_for_stream_fn(
    stream_fn: StreamFn,
    *,
    session_id: str,
    history_store: InMemoryHistoryStore | None = None,
    run_store: InMemoryRunStore | None = None,
) -> _RunDependencies:
    """Build run dependencies around one stream function."""

    history_store = (
        history_store if history_store is not None else InMemoryHistoryStore()
    )
    history_store.ensure_session(session_id=session_id)
    return _RunDependencies(
        execution=_ExecutionDependencies(
            stream_fn=stream_fn,
            model="gpt-5.4",
            instructions="Base prompt.",
            cwd=Path("."),
            auto_mode=False,
            tool_executor=ToolExecutor(()),
            history_store=history_store,
        ),
        history_store=history_store,
        run_store=run_store if run_store is not None else InMemoryRunStore(),
    )


class _TrackingStreamFn:
    """Stream function whose provider streams record their closure."""

    provider = "test-provider"

    def __init__(self, events: Sequence[ProviderStreamEvent]) -> None:
        """Serve one provider stream over the given events."""

        self._events = tuple(events)
        self.closed = False

    async def __call__(
        self,
        history: object,
        model: str,
        *,
        instructions: str,
        tools: object,
    ) -> AsyncGenerator[ProviderStreamEvent, None]:
        """Return a stream that flags this instance when closed."""

        _ = history, model, instructions, tools
        return self._stream()

    async def _stream(self) -> AsyncGenerator[ProviderStreamEvent, None]:
        """Yield the configured events, recording closure on every exit."""

        try:
            for event in self._events:
                yield event
        finally:
            self.closed = True


def test_run_closes_the_provider_stream_when_projection_fails() -> None:
    """Release the provider transport when a history write fails the run.

    Closure does not cascade through generator chains on its own; this
    pins the explicit forwarding from the run's execution down to the
    provider stream.
    """

    async def _run() -> None:
        """Fail projection mid-run and observe the stream's cleanup."""

        history_store = _FailingHistoryStore()
        stream_fn = _TrackingStreamFn(final_text_stream("resp_1", "hello back"))
        deps = _deps_for_stream_fn(
            stream_fn,
            session_id="stream-cleanup",
            history_store=history_store,
        )

        run = Run(
            spec=_RunSpec(session_id="stream-cleanup", content="hello", result=None),
            deps=deps,
            on_finished=lambda _: None,
        )
        history_store.fail_appends = True

        assert await run.wait() == "failed"
        assert stream_fn.closed
        failure = run.failure
        assert failure is not None
        assert failure.message == "history unavailable"

    asyncio.run(_run())


def test_run_keeps_completed_state_when_terminal_persistence_fails() -> None:
    """Expose a failed terminal write without rewriting status or outcome."""

    async def _run() -> None:
        """Complete a run whose terminal record write fails."""

        run_store = _FailingRunStore()
        released: list[Run] = []
        deps = _deps(
            [final_text_stream("resp_1", "done")],
            session_id="store-failure",
            run_store=run_store,
        )

        run = Run(
            spec=_RunSpec(session_id="store-failure", content="hello", result=None),
            deps=deps,
            on_finished=released.append,
        )
        run_store.fail_updates = True

        assert await run.wait() == "completed"
        assert run.outcome == Completed(value="done")
        assert run.failure is None
        assert run.exception is None
        assert run.persistence_error is not None
        assert run_store.get_run(run.id).status == "running"
        assert released == [run]

    asyncio.run(_run())


def test_run_continues_when_lifecycle_tracking_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Expose telemetry failure without changing the task outcome."""

    async def _run() -> None:
        """Fail lifecycle stamping after run start and complete normally."""

        deps = _deps(
            [final_text_stream("resp_1", "done")],
            session_id="telemetry-failure",
        )
        run = Run(
            spec=_RunSpec(
                session_id="telemetry-failure",
                content="hello",
                result=None,
            ),
            deps=deps,
            on_finished=lambda _: None,
        )
        tracker = run._lifecycle_scope_tracker
        tracking_error = RuntimeError("scope tracking unavailable")

        with patch.object(tracker, "stamp", side_effect=tracking_error):
            assert await run.wait() == "completed"

        events = [event async for event in run.events()]
        assert run.outcome == Completed(value="done")
        assert run.telemetry_errors == (tracking_error,)
        assert events[0].lifecycle is not None
        assert all(event.lifecycle is None for event in events[1:])

    asyncio.run(_run())

    assert "Run lifecycle telemetry disabled" in caplog.text


def test_run_abandons_its_record_when_the_user_message_write_fails() -> None:
    """Fail the created record as a submission failure and re-raise."""

    async def _run() -> None:
        """Fail submission after the running record was created."""

        history_store = _FailingHistoryStore()
        run_store = InMemoryRunStore()
        deps = _deps(
            [final_text_stream("resp_1", "never")],
            session_id="abandoned",
            history_store=history_store,
            run_store=run_store,
        )
        history_store.fail_appends = True

        with pytest.raises(RuntimeError, match="history unavailable"):
            Run(
                spec=_RunSpec(session_id="abandoned", content="hello", result=None),
                deps=deps,
                on_finished=lambda _: None,
            )

        records = run_store.list_runs("abandoned")
        assert len(records) == 1
        assert records[0].status == "failed"
        outcome = records[0].outcome
        assert isinstance(outcome, Failed)

    asyncio.run(_run())


class _ControlSignal(BaseException):
    """Deterministic process-control signal for finalization testing."""


@contextmanager
def _capturing_loop_exceptions(
    reported: list[BaseException],
) -> Iterator[None]:
    """Collect exceptions escaping event-loop callbacks into ``reported``."""

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
        yield
    finally:
        loop.set_exception_handler(previous_handler)


def test_run_reraises_owner_release_control_exception_after_finishing() -> None:
    """Preserve control flow without rewriting the recorded terminal state."""

    async def _run() -> None:
        """Observe an interrupted owner callback through the event loop."""

        signal = _ControlSignal("stop")
        reported: list[BaseException] = []

        def interrupt(_: Run) -> None:
            """Interrupt owner notification with a control exception."""

            raise signal

        run_store = InMemoryRunStore()
        deps = _deps(
            [final_text_stream("resp_1", "done")],
            session_id="control-signal",
            run_store=run_store,
        )

        with _capturing_loop_exceptions(reported):
            run = Run(
                spec=_RunSpec(
                    session_id="control-signal", content="hello", result=None
                ),
                deps=deps,
                on_finished=interrupt,
            )
            assert await run.wait() == "completed"
            await asyncio.sleep(0)

        assert reported == [signal]
        assert run.status == "completed"
        assert run.failure is None
        assert run.exception is None
        assert run.persistence_error is None
        assert run_store.get_run(run.id).status == "completed"

    asyncio.run(_run())
