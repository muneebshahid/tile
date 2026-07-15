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
from tile.prompt import DEFAULT_INSTRUCTIONS
from tile.result import (
    COMPLETE_TOOL_NAME,
    FAIL_TOOL_NAME,
    MAX_RESULT_FOLLOW_UPS,
    NO_RESULT_REASON,
    RESULT_CONTRACT,
    RESULT_FOLLOW_UP,
    Completed,
    Failed,
    RunOutcome,
)
from tile.runs import (
    RunFailure,
    RunFailureOrigin,
    RunRecord,
    RunStatus,
    RunStore,
    TerminalRunStatus,
)
from tile.tool_executor import ToolExecutor
from tile.tools.complete import tool as complete_tool
from tile.tools.complete import CompleteDetails
from tile.tools.fail import tool as fail_tool
from tile.tools.fail import FailDetails
from tile.events import (
    AgentEndEvent,
    AgentEvent,
    MessageEndEvent,
    ResultFollowUpEvent,
    StreamFn,
    ToolExecutionEndEvent,
)
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
    ) -> None:
        """Start a run that drives the given event source to completion."""

        self._record = record
        self._events: list[AgentEvent] = []
        self._exception: BaseException | None = None
        self._persistence_error: BaseException | None = None
        self._changed = asyncio.Event()
        self._on_done = on_done
        self._on_record = on_record
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

        if self._record.failure is None:
            return None
        return self._record.failure.message

    @property
    def failure(self) -> RunFailure | None:
        """Return serializable execution failure diagnostics, when available."""

        return self._record.failure

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
        """Return the run outcome as soon as the agent run has ended.

        Available from the moment the run's end event is published, before
        the terminal status lands. Returns None before then and for runs
        that ended without a verdict (execution failure or abort); the run
        status and error message carry that story.
        """

        if self._record.status == "running":
            return _event_outcome(self._events)
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
        """Wait until the run reaches a terminal status and return it."""

        await asyncio.wait({self._task})
        return self.status

    def abort(self) -> None:
        """Request cancellation of the run task."""

        if not self._task.done():
            self._task.cancel()

    async def _pump(self, events: AsyncIterator[AgentEvent]) -> None:
        """Drive the event source to completion, recording each event."""

        async for event in events:
            self._publish(event)

    def _finalize(self, task: asyncio.Task[None]) -> None:
        """Record the terminal run state, then persist and release it.

        The terminal status and outcome are derived only from the task's
        execution result and written exactly once. Persistence and owner
        release are bookkeeping: their failures are logged and exposed, but
        never rewrite what the caller sees.
        """

        task_error = _task_error(task)
        if task_error is not None:
            self._fail(task_error, origin=_execution_failure_origin(task_error))
        elif task.cancelled():
            self._abort()
        else:
            self._complete()
        try:
            self._persist_terminal_record()
        finally:
            self._release_owner()

    def _persist_terminal_record(self) -> None:
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
        """Record successful run execution."""

        self._set_terminal_state("completed")

    def _abort(self) -> None:
        """Record an aborted run execution."""

        self._set_terminal_state("aborted")

    def _fail(self, error: BaseException, *, origin: RunFailureOrigin) -> None:
        """Record one failed execution with serializable and live diagnostics."""

        self._set_terminal_state(
            "failed",
            failure=RunFailure(
                origin=origin,
                exception_type=type(error).__name__,
                message=str(error),
            ),
            exception=error,
        )

    def _publish(self, event: AgentEvent) -> None:
        """Append one event to the run log and wake subscribers."""

        self._events.append(event)
        self._changed.set()

    def _set_terminal_state(
        self,
        status: TerminalRunStatus,
        *,
        failure: RunFailure | None = None,
        exception: BaseException | None = None,
    ) -> None:
        """Set the terminal record state and wake subscribers.

        This transition must not raise: the end timestamp is clamped so a
        backward clock step cannot trip the record's lifecycle validator and
        strand the run in a running state.
        """

        source = _latest_provider_source(self._events)
        self._record = self._record.finish(
            status=status,
            ended_at=max(datetime.now(UTC), self._record.started_at),
            provider=source.provider if source is not None else None,
            model=source.model if source is not None else None,
            outcome=_event_outcome(self._events) if status == "completed" else None,
            failure=failure,
        )
        self._exception = exception
        self._changed.set()


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
                events=self._run_events(session_id, result=result),
                on_done=self._release_run,
                on_record=self._run_store.update_run,
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
                    status="failed",
                    ended_at=max(datetime.now(UTC), record.started_at),
                    failure=RunFailure(
                        origin="submission",
                        exception_type=type(error).__name__,
                        message=str(error),
                    ),
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

    async def _run_events(
        self,
        session_id: str,
        *,
        result: type[BaseModel] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Yield agent events for one prompt run, persisting stable history."""

        events = (
            self._plain_prompt_events(session_id)
            if result is None
            else self._result_prompt_events(session_id, result)
        )
        async for event in events:
            self._persist_stable_event(session_id, event)
            yield event

    async def _plain_prompt_events(
        self,
        session_id: str,
    ) -> AsyncIterator[AgentEvent]:
        """Run one plain agent invocation and attach its text outcome."""

        observation = _AgentRunObservation()
        async for event in self._agent_events(
            session_id,
            tool_executor=self._tool_executor,
            instructions=self._instructions,
        ):
            observation.observe(event)
            if isinstance(event, AgentEndEvent):
                turn = _require_completed_turn(observation.last_turn)
                yield AgentEndEvent(outcome=Completed(value=_assistant_text(turn)))
            else:
                yield event

    async def _result_prompt_events(
        self,
        session_id: str,
        result: type[BaseModel],
    ) -> AsyncIterator[AgentEvent]:
        """Run agent invocations until the required result is produced or exhausted."""

        tool_executor = ToolExecutor(
            (*self._tool_executor.tools, complete_tool(result), fail_tool)
        )
        instructions = f"{self._instructions}\n\n{RESULT_CONTRACT}"
        for follow_ups in range(MAX_RESULT_FOLLOW_UPS + 1):
            observation = _AgentRunObservation()
            async for event in self._agent_events(
                session_id,
                tool_executor=tool_executor,
                instructions=instructions,
            ):
                observation.observe(event)
                if not isinstance(event, AgentEndEvent):
                    yield event

            _require_completed_turn(observation.last_turn)
            outcome = _result_outcome(observation.terminal_details)
            if outcome is not None:
                yield AgentEndEvent(outcome=outcome)
                return
            if follow_ups == MAX_RESULT_FOLLOW_UPS:
                yield AgentEndEvent(outcome=Failed(reason=NO_RESULT_REASON))
                return
            yield AgentEndEvent()
            yield ResultFollowUpEvent(message=UserMessage(content=RESULT_FOLLOW_UP))

    async def _agent_events(
        self,
        session_id: str,
        *,
        tool_executor: ToolExecutor,
        instructions: str,
    ) -> AsyncIterator[AgentEvent]:
        """Yield one stateless agent run over the session's current history."""

        async for event in run_agent(
            self._history_store.get_history(session_id),
            stream_fn=self._stream_fn,
            model=self._model,
            tool_executor=tool_executor,
            instructions=instructions,
            auto_mode=self._auto_mode,
            cwd=self._cwd,
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
        """Persist error results for tool calls the run left unanswered."""

        results = [
            ToolResultTurn(
                call_id=call.call_id,
                tool_name=call.name,
                content=[ToolTextContent(text="Tool execution did not complete.")],
                is_error=True,
            )
            for call in _unanswered_tool_calls(run.conversation_items)
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


def _event_outcome(events: Sequence[AgentEvent]) -> RunOutcome | None:
    """Return the last prompt-level outcome emitted by a run."""

    for event in reversed(events):
        if isinstance(event, AgentEndEvent):
            return event.outcome
    return None


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


def _execution_failure_origin(error: BaseException) -> RunFailureOrigin:
    """Classify a task failure at the most specific known runtime boundary."""

    if isinstance(error, TurnFailedError):
        return "turn"
    return "execution"


def _log_run_bookkeeping_error(message: str, error: BaseException) -> None:
    """Log a bookkeeping failure that must not rewrite the run's terminal state."""

    logger.error(message, exc_info=(type(error), error, error.__traceback__))


def _result_outcome(terminal_details: ToolDetails | None) -> RunOutcome | None:
    """Build a terminal outcome, or return None when a result remains missing."""

    if isinstance(terminal_details, CompleteDetails):
        return Completed(value=terminal_details.value)
    if isinstance(terminal_details, FailDetails):
        return Failed(reason=terminal_details.reason)
    return None


def _assistant_text(turn: AssistantTurn) -> str:
    """Join one assistant turn's text blocks."""

    return "\n\n".join(
        block.text for block in turn.blocks if isinstance(block, TextBlock)
    )
