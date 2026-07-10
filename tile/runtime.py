"""Runtime, session, and run facades for the stateless agent runner."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias
from uuid import uuid4

from pydantic import BaseModel

from tile.types.conversation import (
    AssistantTurn,
    ConversationItem,
    ToolResultTurn,
    UserMessage,
)
from tile.types.stream_events import TextBlock, ToolCallBlock
from tile.types.tools import ToolDefinition, ToolTextContent
from tile.agent import run_agent
from tile.history import HistoryStore, InMemoryHistoryStore, SessionRecord
from tile.prompt import DEFAULT_INSTRUCTIONS
from tile.result import (
    COMPLETE_TOOL_NAME,
    FAIL_TOOL_NAME,
    RESULT_CONTRACT,
    RunOutcome,
)
from tile.tool_executor import ToolExecutor
from tile.tools.complete import tool as complete_tool
from tile.tools.fail import tool as fail_tool
from tile.events import (
    AgentEndEvent,
    AgentEvent,
    MessageEndEvent,
    ResultFollowUpEvent,
    StreamFn,
    ToolExecutionEndEvent,
)

RunStatus: TypeAlias = Literal["running", "completed", "failed", "aborted"]


class SessionBusyError(RuntimeError):
    """Raised when a prompt is submitted while the same session is already active."""


class Run:
    """Handle for one task-owned prompt execution.

    The run owns the task that pumps its event source into a replayable log.
    Subscribers observe events; dropping a subscriber never affects the run.
    """

    def __init__(
        self,
        *,
        run_id: str,
        session_id: str,
        events: AsyncIterator[AgentEvent],
        on_done: Callable[[Run], None],
    ) -> None:
        """Start a run that drives the given event source to completion."""

        self._run_id = run_id
        self._session_id = session_id
        self._events: list[AgentEvent] = []
        self._status: RunStatus = "running"
        self._error_message: str | None = None
        self._changed = asyncio.Event()
        self._on_done = on_done
        self._task = asyncio.create_task(self._pump(events))
        self._task.add_done_callback(self._finalize)

    @property
    def id(self) -> str:
        """Return the stable run id."""

        return self._run_id

    @property
    def session_id(self) -> str:
        """Return the id of the session this run belongs to."""

        return self._session_id

    @property
    def status(self) -> RunStatus:
        """Return the current run status."""

        return self._status

    @property
    def error_message(self) -> str | None:
        """Return the failure message when the run has failed."""

        return self._error_message

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
        """Return the terminal run outcome once the agent run has ended.

        Returns None while the run is in flight and for runs that ended
        without a terminal turn (stream error or abort).
        """

        for event in reversed(self._events):
            if isinstance(event, AgentEndEvent):
                return event.outcome
        return None

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
            if self._status != "running":
                return
            await self._changed.wait()

    async def wait(self) -> RunStatus:
        """Wait until the run reaches a terminal status and return it."""

        await asyncio.wait({self._task})
        return self._status

    def abort(self) -> None:
        """Request cancellation of the run task."""

        if not self._task.done():
            self._task.cancel()

    async def _pump(self, events: AsyncIterator[AgentEvent]) -> None:
        """Drive the event source to completion, recording each event."""

        async for event in events:
            self._publish(event)

    def _finalize(self, task: asyncio.Task[None]) -> None:
        """Notify the owner, then record the terminal run status."""

        self._on_done(self)
        if task.cancelled():
            self._finish("aborted")
        elif (error := task.exception()) is not None:
            self._finish("failed", error_message=str(error))
        else:
            self._finish("completed")

    def _publish(self, event: AgentEvent) -> None:
        """Append one event to the run log and wake subscribers."""

        self._events.append(event)
        self._changed.set()

    def _finish(self, status: RunStatus, error_message: str | None = None) -> None:
        """Record the terminal status and wake subscribers."""

        self._status = status
        self._error_message = error_message
        self._changed.set()


class AgentRuntime:
    """Configured runtime container for many sessions."""

    def __init__(
        self,
        *,
        stream_fn: StreamFn,
        model: str,
        history_store: HistoryStore | None = None,
        tools: Sequence[ToolDefinition] = (),
        instructions: str = DEFAULT_INSTRUCTIONS,
        auto_mode: bool = True,
        cwd: Path | str | None = None,
    ) -> None:
        """Create a runtime with shared agent dependencies."""

        _reject_reserved_tool_names(tools)
        self._stream_fn = stream_fn
        self._model = model
        self._history_store = (
            history_store if history_store is not None else InMemoryHistoryStore()
        )
        self._tool_executor = ToolExecutor(tools)
        self._instructions = instructions
        self._auto_mode = auto_mode
        self._cwd = cwd
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
        try:
            self._append_user_message(session_id, content)
            run = Run(
                run_id=str(uuid4()),
                session_id=session_id,
                events=self._run_events(session_id, result=result),
                on_done=self._release_run,
            )
        except BaseException:
            self._finish_prompt(session_id)
            raise
        self._active_runs.add(run)
        return run

    async def _run_events(
        self,
        session_id: str,
        *,
        result: type[BaseModel] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Yield agent events for one prompt run, persisting stable history."""

        tool_executor = self._tool_executor
        instructions = self._instructions
        if result is not None:
            tool_executor = ToolExecutor(
                (*tool_executor.tools, complete_tool(result), fail_tool)
            )
            instructions = f"{instructions}\n\n{RESULT_CONTRACT}"

        async for event in run_agent(
            self._history_store.get_history(session_id),
            stream_fn=self._stream_fn,
            model=self._model,
            tool_executor=tool_executor,
            instructions=instructions,
            auto_mode=self._auto_mode,
            cwd=self._cwd,
        ):
            self._persist_stable_event(session_id, event)
            yield event

    def _release_run(self, run: Run) -> None:
        """Heal unanswered tool calls, then release the session and the run."""

        self._heal_unanswered_tool_calls(run)
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
    """Return the replayable conversation items carried by one agent event."""

    if isinstance(event, MessageEndEvent):
        return (event.assistant_turn,)
    if isinstance(event, ToolExecutionEndEvent):
        return (event.outcome.tool_result_turn,)
    if isinstance(event, ResultFollowUpEvent):
        return (event.message,)
    return ()


def _assistant_text(turn: AssistantTurn) -> str:
    """Join the text blocks of one assistant turn with blank lines."""

    return "\n\n".join(
        block.text for block in turn.blocks if isinstance(block, TextBlock)
    )
