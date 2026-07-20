"""Run: one prompt execution from submission through finalization.

The run owns every run-scoped mutation: the durable record, the user
message, the event log and its projection into session history, healing,
and both run lifecycle events. Execution (what happens in between) is
delegated to the prompt programs in ``execution``, which return the
``RunOutcome``; only the run turns that outcome — or the exception or
cancellation that replaces it — into the terminal run end event.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from time import monotonic_ns
from uuid import uuid4

from pydantic import BaseModel

from tile.events import (
    AgentEvent,
    MessageEndEvent,
    ResultFollowUpEvent,
    RunEndEvent,
    RunStartEvent,
    ToolExecutionEndEvent,
)
from tile.history import HistoryStore
from tile.result import (
    Aborted,
    ExecutionFailure,
    ExecutionFailureOrigin,
    Failed,
    RunOutcome,
)
from tile.runs import RunRecord, RunStatus, RunStore
from tile.runtime.execution import (
    TurnFailedError,
    _assistant_text,
    _ExecutionDependencies,
    execute_prompt,
)
from tile.runtime.telemetry import _LifecycleScopeTracker
from tile.types.conversation import (
    AssistantTurn,
    ConversationItem,
    ToolResultTurn,
    UserMessage,
)
from tile.types.stream_events import ProviderSource, ToolCallBlock
from tile.types.tools import ToolTextContent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RunSpec:
    """What one submitted prompt asks for."""

    session_id: str
    content: str
    result: type[BaseModel] | None


@dataclass(frozen=True)
class _RunDependencies:
    """Caller-constructed dependencies shared by every run of a runtime.

    The execution contract is composed rather than flattened so the
    prompt program receives only what it may touch; the stores name the
    run's own mutation capabilities.
    """

    execution: _ExecutionDependencies
    history_store: HistoryStore
    run_store: RunStore


class Run:
    """Handle for one task-owned prompt execution.

    Construction performs submission persistence synchronously — the
    running record, then the user message — so a store failure raises to
    the caller before any task exists; a record left behind by a failed
    submission is abandoned as failed. From there the run publishes its
    own ``RunStartEvent``, drives the prompt program in an owned task,
    and finalizes exactly once. Subscribers observe events; dropping a
    subscriber never affects the run.
    """

    def __init__(
        self,
        *,
        spec: _RunSpec,
        deps: _RunDependencies,
        on_finished: Callable[[Run], None],
    ) -> None:
        """Persist the submission, open the log, and start execution.

        ``on_finished`` is the owner's narrow release hook, called during
        finalization after the terminal record and healing; it must not
        perform run work.
        """

        self._spec = spec
        self._deps = deps
        self._on_finished = on_finished
        self._events: list[AgentEvent] = []
        self._telemetry_errors: list[Exception] = []
        self._lifecycle_scope_tracker = _LifecycleScopeTracker(
            clock=monotonic_ns,
            scope_id_factory=lambda: str(uuid4()),
        )
        self._exception: BaseException | None = None
        self._persistence_error: BaseException | None = None
        self._changed = asyncio.Event()
        self._finalized = asyncio.Event()
        self._record = self._create_run_record()
        try:
            self._append_user_message()
            self._publish(RunStartEvent())
            self._task = asyncio.create_task(
                execute_prompt(
                    self._publish,
                    deps=deps.execution,
                    session_id=spec.session_id,
                    result=spec.result,
                )
            )
            self._task.add_done_callback(self._finalize)
        except BaseException as submission_error:
            self._abandon_record(submission_error)
            raise

    @property
    def id(self) -> str:
        """Return the stable run id."""

        return self._record.run_id

    @property
    def session_id(self) -> str:
        """Return the id of the session this run belongs to."""

        return self._record.session_id

    @property
    def status(self) -> RunStatus:
        """Return the current run status."""

        return self._record.status

    @property
    def record(self) -> RunRecord:
        """Return a defensive snapshot of the run's current durable summary."""

        return self._record.model_copy(deep=True)

    @property
    def error_message(self) -> str | None:
        """Return the failure message when the run execution has failed."""

        failure = self.failure
        if failure is None:
            return None
        return failure.message

    @property
    def failure(self) -> ExecutionFailure | None:
        """Return serializable execution failure diagnostics, when available."""

        outcome = self._record.outcome
        if isinstance(outcome, Failed) and isinstance(outcome.cause, ExecutionFailure):
            return outcome.cause
        return None

    @property
    def exception(self) -> BaseException | None:
        """Return the original in-process exception for a failed run."""

        return self._exception

    @property
    def persistence_error(self) -> BaseException | None:
        """Return the error that left the durable terminal record unwritten.

        A non-None value means the run's status and outcome are authoritative
        on this handle, but the run store may still report the run as running.
        """

        return self._persistence_error

    @property
    def telemetry_errors(self) -> tuple[Exception, ...]:
        """Return telemetry failures that did not alter the task outcome."""

        return tuple(self._telemetry_errors)

    @property
    def output_text(self) -> str | None:
        """Return the text of the run's latest completed assistant message.

        Text blocks are joined with a blank line. Returns None before the
        first assistant message completes.
        """

        for event in reversed(self._events):
            if isinstance(event, MessageEndEvent):
                return _assistant_text(event.assistant_turn)
        return None

    @property
    def outcome(self) -> RunOutcome | None:
        """Return the run's terminal outcome.

        None while the run is running; set atomically with the terminal
        status at finalization. Every terminal run carries an outcome,
        with execution failures and aborts appearing as
        ``Failed(cause=ExecutionFailure(...))`` and ``Aborted``.
        """

        return self._record.outcome

    @property
    def conversation_items(self) -> tuple[ConversationItem, ...]:
        """Return the conversation items this run has produced so far."""

        return tuple(
            item
            for event in self._events
            if (item := _conversation_item_for(event)) is not None
        )

    async def events(self) -> AsyncIterator[AgentEvent]:
        """Yield run events from the start, following live until the run ends."""

        index = 0
        while True:
            self._changed.clear()
            while index < len(self._events):
                yield self._events[index]
                index += 1
            if self.status != "running":
                return
            await self._changed.wait()

    async def wait(self) -> RunStatus:
        """Wait until the run is fully finalized and return its status.

        Returning only after finalization guarantees the event log is
        closed — it ends with the run end event — and that terminal
        persistence, healing, and owner release have been attempted.
        """

        await self._finalized.wait()
        return self.status

    def abort(self) -> None:
        """Request cancellation of the run task.

        Cancellation lands inside execution and surfaces as the aborted
        outcome; a run that already finished is unaffected.
        """

        if not self._task.done():
            self._task.cancel()

    def _create_run_record(self) -> RunRecord:
        """Persist one running summary before provider execution can start."""

        record = RunRecord(
            run_id=str(uuid4()),
            session_id=self._spec.session_id,
            status="running",
            started_at=datetime.now(UTC),
            model=self._deps.execution.model,
            provider=self._deps.execution.stream_fn.provider,
        )
        self._deps.run_store.create_run(record)
        return record

    def _append_user_message(self) -> None:
        """Persist the prompt's user message before execution can start."""

        self._deps.history_store.append_history(
            self._spec.session_id,
            [UserMessage(content=self._spec.content)],
        )

    def _abandon_record(self, error: BaseException) -> None:
        """Best-effort fail the running record of a submission that never ran."""

        try:
            self._deps.run_store.update_run(
                self._record.finish(
                    outcome=Failed(cause=_execution_failure(error, "submission")),
                )
            )
        except BaseException as abandonment_error:
            _log_run_bookkeeping_error(
                "Abandoned run record could not be persisted", abandonment_error
            )

    def _publish(self, event: AgentEvent) -> None:
        """Publish one event, then project its stable history.

        Publication order is the contract: the event is in the live log
        and visible to subscribers before projection, so a projection
        failure fails the run but can never suppress the event.
        """

        stamped_event = self._stamp_lifecycle(event)
        self._events.append(stamped_event)
        self._changed.set()
        item = _conversation_item_for(stamped_event)
        if item is not None:
            self._deps.history_store.append_history(self._spec.session_id, [item])

    def _stamp_lifecycle(self, event: AgentEvent) -> AgentEvent:
        """Stamp lifecycle metadata or disable telemetry after its first failure."""

        if self._telemetry_errors:
            return event

        try:
            return self._lifecycle_scope_tracker.stamp(event)
        except Exception as tracking_error:
            self._telemetry_errors.append(tracking_error)
            _log_run_bookkeeping_error(
                "Run lifecycle telemetry disabled",
                tracking_error,
            )
            return event

    def _finalize(self, task: asyncio.Task[RunOutcome]) -> None:
        """Finish the run from its task result, then release it.

        The terminal outcome is the task's returned outcome, or the
        aborted or failed outcome that replaces it. Persistence, healing,
        and owner notification are bookkeeping: their failures are logged
        and exposed, but never rewrite what the caller sees. The
        finalized flag is set unconditionally last so waiters can never
        hang on a re-raised bookkeeping error.
        """

        outcome, exception = _terminal_outcome(task)
        try:
            self._conclude(outcome, exception=exception)
        finally:
            try:
                self._release()
            except BaseException as release_error:
                _log_run_bookkeeping_error("Run release failed", release_error)
                if not isinstance(release_error, Exception):
                    raise
            finally:
                self._finalized.set()

    def _conclude(
        self, outcome: RunOutcome, *, exception: BaseException | None
    ) -> None:
        """Land the terminal state: close the log, record it, then persist.

        The run end is appended directly, not published: it carries no
        history, and finalization must not depend on a history store that
        may be the reason the run is closing abnormally. The local
        transition must not raise and happens before persistence, so a
        failed store write can only ever leave the store stale — never
        rewrite what this handle reports.
        """

        self._events.append(self._stamp_lifecycle(RunEndEvent(outcome=outcome)))
        source = _latest_provider_source(self._events)
        self._record = self._record.finish(
            outcome=outcome,
            provider=source.provider if source is not None else None,
            model=source.model if source is not None else None,
        )
        self._exception = exception
        self._changed.set()
        self._persist_record()

    def _persist_record(self) -> None:
        """Persist the terminal record without touching the live run state."""

        try:
            self._deps.run_store.update_run(self._record)
        except BaseException as persistence_error:
            self._persistence_error = persistence_error
            _log_run_bookkeeping_error(
                "Terminal run record could not be persisted", persistence_error
            )
            if not isinstance(persistence_error, Exception):
                raise

    def _release(self) -> None:
        """Heal unanswered tool calls, then notify the owner.

        The owner is notified even when healing fails; both are
        bookkeeping and cannot rewrite the run's terminal state.
        """

        try:
            self._heal_unanswered_tool_calls()
        finally:
            self._on_finished(self)

    def _heal_unanswered_tool_calls(self) -> None:
        """Persist error results for tool calls durable history left unanswered.

        Healing reads the history store, not the run's live event log: a
        failed history projection can leave a tool result in the log that
        never became durable, and it is durable history the next prompt
        replays.
        """

        results = [
            ToolResultTurn(
                call_id=call.call_id,
                tool_name=call.name,
                content=[ToolTextContent(text="Tool execution did not complete.")],
                is_error=True,
            )
            for call in _unanswered_tool_calls(
                self._deps.history_store.get_history(self._spec.session_id)
            )
        ]
        if results:
            self._deps.history_store.append_history(self._spec.session_id, results)


def _terminal_outcome(
    task: asyncio.Task[RunOutcome],
) -> tuple[RunOutcome, BaseException | None]:
    """Derive the run's terminal outcome from how the execution task ended."""

    if task.cancelled():
        return Aborted(), None
    error = task.exception()
    if error is not None:
        failure = _execution_failure(error, _execution_failure_origin(error))
        return Failed(cause=failure), error
    return task.result(), None


def _execution_failure_origin(error: BaseException) -> ExecutionFailureOrigin:
    """Classify a task failure at the most specific known runtime boundary."""

    if isinstance(error, TurnFailedError):
        return "turn"
    return "execution"


def _execution_failure(
    error: BaseException,
    origin: ExecutionFailureOrigin,
) -> ExecutionFailure:
    """Serialize one execution failure cause from an in-process exception."""

    return ExecutionFailure(
        origin=origin,
        exception_type=type(error).__name__,
        message=str(error),
    )


def _conversation_item_for(event: AgentEvent) -> ConversationItem | None:
    """Return the replayable conversation item carried by one agent event.

    Only completed assistant turns are replayable: errored and aborted turns
    are dropped so later prompts and retries see clean history.
    """

    if isinstance(event, MessageEndEvent):
        if event.assistant_turn.status != "completed":
            return None
        return event.assistant_turn
    if isinstance(event, ToolExecutionEndEvent):
        return event.outcome.tool_result_turn
    if isinstance(event, ResultFollowUpEvent):
        return event.message
    return None


def _unanswered_tool_calls(
    items: Sequence[ConversationItem],
) -> list[ToolCallBlock]:
    """Return tool calls from assistant turns that have no matching result."""

    answered = {item.call_id for item in items if isinstance(item, ToolResultTurn)}
    return [
        block
        for item in items
        if isinstance(item, AssistantTurn)
        for block in item.blocks
        if isinstance(block, ToolCallBlock) and block.call_id not in answered
    ]


def _latest_provider_source(
    events: Sequence[AgentEvent],
) -> ProviderSource | None:
    """Return the latest provider identity available from a finalized message."""

    for event in reversed(events):
        if isinstance(event, MessageEndEvent):
            return event.assistant_turn.source
    return None


def _log_run_bookkeeping_error(message: str, error: BaseException) -> None:
    """Log a bookkeeping failure that must not rewrite the run's terminal state."""

    logger.error(message, exc_info=(type(error), error, error.__traceback__))
