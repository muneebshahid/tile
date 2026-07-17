"""Runtime, session, and run facades for the stateless agent runner."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import cast
from uuid import uuid4

from pydantic import BaseModel

from tile.types.conversation import (
    AssistantTurn,
    ConversationItem,
    ToolResultTurn,
    UserMessage,
)
from tile.types.stream_events import TextBlock, ToolCallBlock
from tile.types.tools import (
    ToolDefinition,
    ToolDetails,
    ToolFunction,
    ToolTextContent,
)
from tile.tools.support.paths import normalize_cwd
from tile.agent import run_agent
from tile.history import HistoryStore, SessionRecord
from tile.prompt import DEFAULT_INSTRUCTIONS, build_system_prompt
from tile.result import (
    COMPLETE_TOOL_NAME,
    FAIL_TOOL_NAME,
    MAX_RESULT_FOLLOW_UPS,
    NO_RESULT_REASON,
    RESULT_CONTRACT,
    RESULT_FOLLOW_UP,
    Aborted,
    AgentFailure,
    Completed,
    ExecutionFailure,
    ExecutionFailureOrigin,
    Failed,
    RunOutcome,
)
from tile.runs import (
    RunRecord,
    RunStatus,
    RunStore,
)
from tile.tool_executor import ToolExecutor
from tile.tools.complete import tool as complete_tool
from tile.tools.complete import CompleteDetails
from tile.tools.fail import tool as fail_tool
from tile.tools.fail import FailDetails
from tile.events import (
    AgentEvent,
    MessageEndEvent,
    ResultFollowUpEvent,
    RunEndEvent,
    RunStartEvent,
    StreamFn,
    ToolExecutionEndEvent,
)
from tile.lifecycle import OpenScopeTracker
from tile.types.stream_events import ProviderSource

logger = logging.getLogger(__name__)


class SessionBusyError(RuntimeError):
    """Raised when a prompt is submitted while the same session is already active."""


class TurnFailedError(RuntimeError):
    """Raised when an agent run ends without a completed assistant turn."""

    def __init__(self, turn: AssistantTurn | None) -> None:
        """Preserve the failed turn while exposing a concise exception message."""

        self.turn = turn
        super().__init__(_turn_failure_message(turn))


@dataclass
class _AgentRunObservation:
    """Result-relevant facts observed during one stateless agent run."""

    last_turn: AssistantTurn | None = None
    terminal_details: ToolDetails | None = None

    def observe(self, event: AgentEvent) -> None:
        """Record the latest assistant turn and first terminating tool details."""

        if isinstance(event, MessageEndEvent):
            self.last_turn = event.assistant_turn
        if (
            self.terminal_details is None
            and isinstance(event, ToolExecutionEndEvent)
            and event.outcome.terminate
            and isinstance(event.outcome.details, CompleteDetails | FailDetails)
        ):
            self.terminal_details = event.outcome.details


class Run:
    """Handle for one task-owned prompt execution.

    The run owns the task that pumps its event source into a replayable log.
    Subscribers observe events; dropping a subscriber never affects the run.
    """

    def __init__(
        self,
        *,
        record: RunRecord,
        events: AsyncIterator[AgentEvent],
        on_done: Callable[[Run], None],
        on_record: Callable[[RunRecord], None],
        on_event: Callable[[AgentEvent], None],
    ) -> None:
        """Start a run that drives the given event source to completion.

        The run publishes its own ``RunStartEvent`` before the event source
        starts, so the log opens the run scope on every path. ``on_event``
        observes each producer event after it is published to the live log;
        its failure fails the run but can never suppress the event.
        """

        self._record = record
        self._events: list[AgentEvent] = []
        self._exception: BaseException | None = None
        self._persistence_error: BaseException | None = None
        self._changed = asyncio.Event()
        self._finalized = asyncio.Event()
        self._scopes = OpenScopeTracker()
        self._on_done = on_done
        self._on_record = on_record
        self._on_event = on_event
        self._publish(RunStartEvent())
        self._task = asyncio.create_task(self._pump(events))
        self._task.add_done_callback(self._finalize)

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
        """Return the run outcome as soon as it is committed.

        Available from the moment the run end event is committed, before
        the terminal status lands. Returns None only before then: every
        terminal run carries an outcome, with execution failures and aborts
        appearing as ``Failed(cause=ExecutionFailure(...))`` and ``Aborted``.
        """

        if self._record.status == "running":
            return self._scopes.committed_outcome
        return self._record.outcome

    @property
    def conversation_items(self) -> tuple[ConversationItem, ...]:
        """Return the conversation items this run has produced so far."""

        return tuple(
            item for event in self._events for item in _conversation_items_for(event)
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
        complete — every start paired with an end — and that terminal
        persistence and owner release have been attempted.
        """

        await self._finalized.wait()
        return self.status

    def abort(self) -> None:
        """Request cancellation of the run task.

        Cancellation after the run end is committed still stops the event
        source, but finalization keeps the committed outcome, so a late
        abort cannot relabel a concluded run as aborted.
        """

        if not self._task.done():
            self._task.cancel()

    async def _pump(self, events: AsyncIterator[AgentEvent]) -> None:
        """Drive the event source to completion, recording each event."""

        async for event in events:
            self._publish(event)

    def _finalize(self, task: asyncio.Task[None]) -> None:
        """Finish the run from its task result, then release its owner.

        This is the single abnormal-closure site: scopes a producer left
        open are closed here, innermost first, before the terminal record
        lands. The terminal outcome is derived from the committed run end
        or the task's execution result and recorded exactly once.
        Persistence and owner release are bookkeeping: their failures are
        logged and exposed, but never rewrite what the caller sees. The
        finalized flag is set unconditionally last so waiters can never
        hang on a re-raised bookkeeping error.
        """

        task_error = _task_error(task)
        try:
            if task_error is not None:
                self._fail(task_error)
            elif task.cancelled():
                self._abort()
            else:
                self._complete()
        finally:
            try:
                self._release_owner()
            finally:
                self._finalized.set()

    def _persist_record(self) -> None:
        """Persist the terminal record without touching the live run state."""

        try:
            self._on_record(self._record)
        except BaseException as persistence_error:
            self._persistence_error = persistence_error
            _log_run_bookkeeping_error(
                "Terminal run record could not be persisted", persistence_error
            )
            if not isinstance(persistence_error, Exception):
                raise

    def _release_owner(self) -> None:
        """Release owner state without touching the live run state."""

        try:
            self._on_done(self)
        except BaseException as release_error:
            _log_run_bookkeeping_error("Run owner release failed", release_error)
            if not isinstance(release_error, Exception):
                raise

    def _complete(self) -> None:
        """Finish the run with its committed outcome.

        A clean event source that never committed a run end broke the
        lifecycle contract; the run lands as failed, with the missing
        scopes closed, rather than as completed without a verdict.
        """

        self._conclude(_missing_run_end_failure())

    def _abort(self) -> None:
        """Finish the run as aborted, keeping an already committed outcome."""

        self._conclude(Aborted())

    def _fail(self, error: BaseException) -> None:
        """Finish the run with serializable and live execution diagnostics.

        A failure after the run end was committed cannot rewrite the
        concluded outcome; it is logged as bookkeeping and kept on the
        handle for local debugging.
        """

        if self._scopes.committed_outcome is not None:
            _log_run_bookkeeping_error(
                "Run event source failed after its run end was committed", error
            )
        fallback = Failed(
            cause=_execution_failure(error, _execution_failure_origin(error))
        )
        self._conclude(fallback, exception=error)

    def _conclude(
        self, fallback: RunOutcome, *, exception: BaseException | None = None
    ) -> None:
        """Close the remaining scopes, then finish with the run's outcome.

        The committed outcome always wins over the fallback; interruptions
        are synthesized exactly when scopes remain open. Synthesized events
        bypass the event observer: they carry no history, and finalization
        must not depend on an observer that may be the reason the run is
        closing abnormally.
        """

        outcome = self._scopes.committed_outcome
        if outcome is None:
            outcome = fallback
        closing_events = self._scopes.close(outcome)
        if closing_events:
            self._events.extend(closing_events)
            self._changed.set()
        self._finish(outcome, exception=exception)

    def _publish(self, event: AgentEvent) -> None:
        """Track one event, publish it, then let the observer see it.

        Interruptions the event implies for scopes abandoned inside it are
        published first, keeping the log properly nested; like all
        synthesized events, they bypass the observer. Publication order is
        the contract: an event is in the live log and visible to
        subscribers before observation, so an observer failure fails the
        run but can never suppress the event.
        """

        self._events.extend(self._scopes.observe(event))
        self._events.append(event)
        self._changed.set()
        self._on_event(event)

    def _finish(
        self,
        outcome: RunOutcome,
        *,
        exception: BaseException | None = None,
    ) -> None:
        """Record the terminal outcome, wake subscribers, then persist.

        The local transition must not raise and always happens first, so a
        failed store write can only ever leave the store stale — never
        rewrite what this handle reports.
        """

        source = _latest_provider_source(self._events)
        self._record = self._record.finish(
            outcome=outcome,
            provider=source.provider if source is not None else None,
            model=source.model if source is not None else None,
        )
        self._exception = exception
        self._changed.set()
        self._persist_record()


class AgentRuntime:
    """Configured runtime container for many sessions."""

    def __init__(
        self,
        *,
        stream_fn: StreamFn,
        model: str,
        cwd: Path | str,
        history_store: HistoryStore,
        run_store: RunStore,
        tools: Sequence[ToolDefinition] = (),
        instructions: str = DEFAULT_INSTRUCTIONS,
        auto_mode: bool = True,
    ) -> None:
        """Create a runtime with shared agent dependencies.

        ``cwd`` is the runtime's single working directory: it is announced in
        the system prompt and injected into every tool whose function declares
        a ``cwd`` parameter. Pass tools unbound; the runtime binds them. The
        stores are required so the caller decides where records live; pass
        the in-memory stores for process-lifetime state.
        """

        _reject_reserved_tool_names(tools)
        self._stream_fn = stream_fn
        self._model = model
        self._cwd = normalize_cwd(cwd)
        self._history_store = history_store
        self._run_store = run_store
        self._tool_executor = ToolExecutor(_bind_cwd_tools(tools, self._cwd))
        self._instructions = instructions
        self._auto_mode = auto_mode
        self._active_prompt_session_ids: set[str] = set()
        self._active_runs: set[Run] = set()

    @property
    def sessions(self) -> tuple[Session, ...]:
        """Return handles for known sessions."""

        return tuple(
            self._build_session(record)
            for record in self._history_store.list_sessions()
        )

    def session(
        self,
        *,
        session_id: str | None = None,
        name: str | None = None,
    ) -> Session:
        """Create or return a session handle."""

        record = self._history_store.ensure_session(
            session_id=self._resolve_session_id(session_id),
            name=name,
        )
        return self._build_session(record)

    def get_session(self, session_id: str) -> Session:
        """Return a handle for an existing session."""

        return self._build_session(self._history_store.get_session(session_id))

    def history_for(self, session_id: str) -> Sequence[ConversationItem]:
        """Return completed conversation history for a session."""

        return self._history_store.get_history(session_id)

    def get_run(self, run_id: str) -> RunRecord:
        """Return a durable run summary by its stable id."""

        return self._run_store.get_run(run_id)

    def runs_for(self, session_id: str) -> Sequence[RunRecord]:
        """Return durable run summaries for one session."""

        return self._run_store.list_runs(session_id)

    def fork_session(
        self,
        *,
        source_session_id: str,
        target_session_id: str | None = None,
        name: str | None = None,
    ) -> Session:
        """Fork an existing session into a new session handle."""

        record = self._history_store.copy_history(
            source_session_id=source_session_id,
            target_session_id=self._resolve_session_id(target_session_id),
            target_name=name,
        )
        return self._build_session(record)

    def _submit_prompt(
        self,
        session_id: str,
        content: str,
        *,
        result: type[BaseModel] | None = None,
    ) -> Run:
        """Submit one prompt for task-owned execution and return its run handle."""

        self._start_prompt(session_id)
        record: RunRecord | None = None
        try:
            record = self._create_run_record(session_id)
            self._append_user_message(session_id, content)
            run = Run(
                record=record,
                events=self._prompt_events(session_id, result=result),
                on_done=self._release_run,
                on_record=self._run_store.update_run,
                on_event=partial(self._persist_stable_event, session_id),
            )
        except BaseException as submission_error:
            if record is not None:
                self._abandon_run_record(record, submission_error)
            self._finish_prompt(session_id)
            raise
        self._active_runs.add(run)
        return run

    def _abandon_run_record(self, record: RunRecord, error: BaseException) -> None:
        """Best-effort fail a running record whose submission never started."""

        try:
            self._run_store.update_run(
                record.finish(
                    outcome=Failed(cause=_execution_failure(error, "submission")),
                )
            )
        except BaseException as abandonment_error:
            _log_run_bookkeeping_error(
                "Abandoned run record could not be persisted", abandonment_error
            )

    def _create_run_record(self, session_id: str) -> RunRecord:
        """Persist one running summary before provider execution can start."""

        record = RunRecord(
            run_id=str(uuid4()),
            session_id=session_id,
            status="running",
            started_at=datetime.now(UTC),
            model=self._model,
            provider=self._stream_fn.provider,
        )
        self._run_store.create_run(record)
        return record

    async def _prompt_events(
        self,
        session_id: str,
        *,
        result: type[BaseModel] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Yield the lifecycle-paired event stream for one prompt run."""

        events = (
            self._plain_prompt_events(session_id)
            if result is None
            else self._result_prompt_events(session_id, result)
        )
        async for event in events:
            yield event

    async def _plain_prompt_events(
        self,
        session_id: str,
    ) -> AsyncIterator[AgentEvent]:
        """Run one plain agent invocation and commit its text outcome."""

        observation = _AgentRunObservation()
        async for event in self._agent_events(
            session_id,
            tool_executor=self._tool_executor,
            instructions=self._instructions,
        ):
            observation.observe(event)
            yield event
        turn = _require_completed_turn(observation.last_turn)
        yield RunEndEvent(outcome=Completed(value=_assistant_text(turn)))

    async def _result_prompt_events(
        self,
        session_id: str,
        result: type[BaseModel],
    ) -> AsyncIterator[AgentEvent]:
        """Run agent attempts until the required result is produced or exhausted."""

        tool_executor = ToolExecutor(
            (*self._tool_executor.tools, complete_tool(result), fail_tool)
        )
        instructions = f"{self._instructions}\n\n{RESULT_CONTRACT}"
        for attempt in range(MAX_RESULT_FOLLOW_UPS + 1):
            observation = _AgentRunObservation()
            async for event in self._agent_events(
                session_id,
                tool_executor=tool_executor,
                instructions=instructions,
            ):
                observation.observe(event)
                yield event

            _require_completed_turn(observation.last_turn)
            outcome = _result_outcome(observation.terminal_details)
            if outcome is not None:
                yield RunEndEvent(outcome=outcome)
                return
            if attempt == MAX_RESULT_FOLLOW_UPS:
                yield RunEndEvent(
                    outcome=Failed(cause=AgentFailure(reason=NO_RESULT_REASON))
                )
                return
            yield ResultFollowUpEvent(message=UserMessage(content=RESULT_FOLLOW_UP))

    async def _agent_events(
        self,
        session_id: str,
        *,
        tool_executor: ToolExecutor,
        instructions: str,
    ) -> AsyncIterator[AgentEvent]:
        """Yield one stateless agent attempt over the session's current history.

        The system prompt is composed here, per attempt, so project context
        and the environment lines stay current across attempts; the agent
        receives it fully resolved.
        """

        async for event in run_agent(
            self._history_store.get_history(session_id),
            stream_fn=self._stream_fn,
            model=self._model,
            tool_executor=tool_executor,
            instructions=build_system_prompt(
                instructions,
                self._cwd,
                auto_mode=self._auto_mode,
            ),
        ):
            yield event

    def _release_run(self, run: Run) -> None:
        """Heal unanswered tool calls, then release the session and the run."""

        try:
            self._heal_unanswered_tool_calls(run)
        finally:
            self._finish_prompt(run.session_id)
            self._active_runs.discard(run)

    def _heal_unanswered_tool_calls(self, run: Run) -> None:
        """Persist error results for tool calls durable history left unanswered.

        Healing reads the history store, not the run's live event log: a
        failed history observation can leave a tool result in the log that
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
                self._history_store.get_history(run.session_id)
            )
        ]
        if results:
            self._history_store.append_history(run.session_id, results)

    def _start_prompt(self, session_id: str) -> None:
        """Mark a session prompt active or reject overlapping prompt work."""

        if session_id in self._active_prompt_session_ids:
            raise SessionBusyError(
                f"Session already has an active prompt: {session_id}"
            )
        self._active_prompt_session_ids.add(session_id)

    def _finish_prompt(self, session_id: str) -> None:
        """Clear the active prompt marker for a session."""

        self._active_prompt_session_ids.discard(session_id)

    def _append_user_message(self, session_id: str, content: str) -> None:
        """Persist a user message before provider execution starts."""

        self._history_store.append_history(session_id, [UserMessage(content=content)])

    def _persist_stable_event(self, session_id: str, event: AgentEvent) -> None:
        """Persist replayable history items from stable agent events."""

        items = _conversation_items_for(event)
        if items:
            self._history_store.append_history(session_id, list(items))

    def _build_session(self, record: SessionRecord) -> Session:
        """Build a session handle from a stored record."""

        return Session(_record=record, _runtime=self)

    def _resolve_session_id(self, session_id: str | None) -> str:
        """Return the provided session id or generate a new one."""

        if session_id is not None:
            return session_id
        return str(uuid4())


@dataclass(frozen=True)
class Session:
    """Scoped handle for one runtime session."""

    _record: SessionRecord
    _runtime: AgentRuntime

    @property
    def id(self) -> str:
        """Return the stable session id."""

        return self._record.session_id

    @property
    def name(self) -> str | None:
        """Return the optional human-readable session name."""

        return self._record.name

    @property
    def history(self) -> Sequence[ConversationItem]:
        """Return completed conversation history for this session."""

        return self._runtime.history_for(self.id)

    async def prompt(
        self,
        content: str,
        *,
        result: type[BaseModel] | None = None,
    ) -> Run:
        """Submit one prompt to this session and return its run handle.

        When ``result`` is set, the run must end through the output contract:
        the runtime adds the `complete` and `fail` tools for this run and the
        outcome carries the schema-validated result.
        """

        return self._runtime._submit_prompt(self.id, content, result=result)

    def fork(
        self,
        *,
        session_id: str | None = None,
        name: str | None = None,
    ) -> Session:
        """Fork this session into a new independently diverging session."""

        return self._runtime.fork_session(
            source_session_id=self.id,
            target_session_id=session_id,
            name=name,
        )


RESERVED_TOOL_NAMES = (COMPLETE_TOOL_NAME, FAIL_TOOL_NAME)


def _reject_reserved_tool_names(tools: Sequence[ToolDefinition]) -> None:
    """Reject caller tools whose names the output contract reserves."""

    for tool in tools:
        if tool.name.lower() in RESERVED_TOOL_NAMES:
            raise ValueError(
                f"Tool name '{tool.name}' is reserved by the runtime for "
                "output contracts; rename the tool."
            )


def _bind_cwd_tools(
    tools: Sequence[ToolDefinition],
    cwd: Path,
) -> tuple[ToolDefinition, ...]:
    """Bind the runtime cwd into every tool that declares a cwd parameter."""

    return tuple(
        _bind_cwd(tool, cwd) if _expects_cwd(tool.fn) else tool for tool in tools
    )


def _bind_cwd(tool: ToolDefinition, cwd: Path) -> ToolDefinition:
    """Return a copy of a tool whose function receives the runtime cwd."""

    _reject_cwd_schema_property(tool)
    fn = cast(ToolFunction, partial(tool.fn, cwd=cwd))
    return tool.model_copy(update={"fn": fn})


def _expects_cwd(fn: ToolFunction) -> bool:
    """Return whether a tool function declares an explicit cwd parameter."""

    parameter = inspect.signature(fn).parameters.get("cwd")
    return parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def _reject_cwd_schema_property(tool: ToolDefinition) -> None:
    """Reject tools that expose the runtime-injected cwd to the model."""

    properties = tool.input_schema.get("properties")
    if isinstance(properties, dict) and "cwd" in properties:
        raise ValueError(
            f"Tool '{tool.name}' declares a `cwd` parameter for runtime "
            "injection but also exposes 'cwd' in its input schema; remove "
            "the schema property."
        )


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


def _conversation_items_for(event: AgentEvent) -> tuple[ConversationItem, ...]:
    """Return the replayable conversation items carried by one agent event.

    Only completed assistant turns are replayable: errored and aborted turns
    are dropped so later prompts and retries see clean history.
    """

    if isinstance(event, MessageEndEvent):
        if event.assistant_turn.status != "completed":
            return ()
        return (event.assistant_turn,)
    if isinstance(event, ToolExecutionEndEvent):
        return (event.outcome.tool_result_turn,)
    if isinstance(event, ResultFollowUpEvent):
        return (event.message,)
    return ()


def _latest_provider_source(
    events: Sequence[AgentEvent],
) -> ProviderSource | None:
    """Return the latest provider identity available from a finalized message."""

    for event in reversed(events):
        if isinstance(event, MessageEndEvent):
            return event.assistant_turn.source
    return None


def _require_completed_turn(turn: AssistantTurn | None) -> AssistantTurn:
    """Return the run's final assistant turn, raising when it did not complete."""

    if turn is None:
        raise TurnFailedError(turn)
    if turn.status != "completed":
        raise TurnFailedError(turn)
    return turn


def _turn_failure_message(turn: AssistantTurn | None) -> str:
    """Return the public message for an unsuccessful assistant turn."""

    if turn is None:
        return "The agent run ended without an assistant turn."
    return turn.error_message or "The assistant turn failed."


def _task_error(task: asyncio.Task[None]) -> BaseException | None:
    """Return a task failure without treating cancellation as an exception."""

    if task.cancelled():
        return None
    return task.exception()


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


def _missing_run_end_failure() -> Failed:
    """Return the failure cause for a run that ended without a committed end.

    Reaching this means the event pipeline broke the lifecycle contract of
    committing a run end on every clean completion. There is nothing left
    to raise into — the task already finished — so the violation is
    delivered as a terminal state, the only channel that cannot be lost.
    """

    return Failed(
        cause=ExecutionFailure(
            origin="execution",
            exception_type="RuntimeError",
            message="The run ended without a committed run end event.",
        )
    )


def _log_run_bookkeeping_error(message: str, error: BaseException) -> None:
    """Log a bookkeeping failure that must not rewrite the run's terminal state."""

    logger.error(message, exc_info=(type(error), error, error.__traceback__))


def _result_outcome(terminal_details: ToolDetails | None) -> RunOutcome | None:
    """Build a terminal outcome, or return None when a result remains missing."""

    if isinstance(terminal_details, CompleteDetails):
        return Completed(value=terminal_details.value)
    if isinstance(terminal_details, FailDetails):
        return Failed(cause=AgentFailure(reason=terminal_details.reason))
    return None


def _assistant_text(turn: AssistantTurn) -> str:
    """Join one assistant turn's text blocks."""

    return "\n\n".join(
        block.text for block in turn.blocks if isinstance(block, TextBlock)
    )
