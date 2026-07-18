"""Tests for run-owned mechanics: submission, finalization, and release."""

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest

from tile.history import InMemoryHistoryStore
from tile.result import Completed, Failed
from tile.runs import InMemoryRunStore, RunRecord
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

    history_store = (
        history_store if history_store is not None else InMemoryHistoryStore()
    )
    history_store.ensure_session(session_id=session_id)
    return _RunDependencies(
        stream_fn=ProviderStreamMock(streams).fn,
        model="gpt-5.4",
        instructions="Base prompt.",
        cwd=Path("."),
        auto_mode=False,
        tool_executor=ToolExecutor(()),
        history_store=history_store,
        run_store=run_store if run_store is not None else InMemoryRunStore(),
    )


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


def test_run_reraises_owner_release_control_exception_after_finishing() -> None:
    """Preserve control flow without rewriting the recorded terminal state."""

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

        run_store = InMemoryRunStore()
        deps = _deps(
            [final_text_stream("resp_1", "done")],
            session_id="control-signal",
            run_store=run_store,
        )

        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(capture_exception)
        try:
            run = Run(
                spec=_RunSpec(
                    session_id="control-signal", content="hello", result=None
                ),
                deps=deps,
                on_finished=interrupt,
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
        assert run_store.get_run(run.id).status == "completed"

    asyncio.run(_run())
