"""Runtime and session facade for the stateless agent runner."""

from __future__ import annotations
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from ai.types.contracts import Reasoning
from ai.types.conversation import ConversationItem, UserMessage
from ai.types.tools import ToolDefinition
from agent.agent import run_agent
from agent.history import HistoryStore, InMemoryHistoryStore, SessionRecord
from agent.prompt import PROMPT
from agent.tool_executor import ToolExecutor
from agent.types import AgentEvent, MessageEndEvent, StreamFn, ToolExecutionEndEvent


class SessionBusyError(RuntimeError):
    """Raised when a prompt starts while the same session is already active."""


class AgentRuntime:
    """Configured runtime container for many sessions."""

    def __init__(
        self,
        *,
        stream_fn: StreamFn,
        model: str,
        history_store: HistoryStore | None = None,
        tools: Sequence[ToolDefinition] = (),
        reasoning: Reasoning | None = None,
        system_prompt: str = PROMPT,
        cwd: Path | str | None = None,
    ) -> None:
        """Create a runtime with shared agent dependencies."""

        self._stream_fn = stream_fn
        self._model = model
        self._history_store = (
            history_store if history_store is not None else InMemoryHistoryStore()
        )
        self._tool_executor = ToolExecutor(tools)
        self._reasoning = reasoning
        self._system_prompt = system_prompt
        self._cwd = cwd
        self._active_prompt_session_ids: set[str] = set()

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

    async def _prompt_session(
        self,
        session_id: str,
        content: str,
    ) -> AsyncIterator[AgentEvent]:
        """Run one prompt in a session and persist completed items."""

        self._start_prompt(session_id)
        try:
            self._append_user_message(session_id, content)

            async for event in run_agent(
                self._history_store.get_history(session_id),
                stream_fn=self._stream_fn,
                model=self._model,
                tool_executor=self._tool_executor,
                reasoning=self._reasoning,
                system_prompt=self._system_prompt,
                cwd=self._cwd,
            ):
                self._persist_stable_event(session_id, event)
                yield event
        finally:
            self._finish_prompt(session_id)

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

        if isinstance(event, MessageEndEvent):
            self._history_store.append_history(session_id, [event.assistant_turn])
        if isinstance(event, ToolExecutionEndEvent):
            self._history_store.append_history(
                session_id,
                [event.outcome.tool_result_turn],
            )

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

    async def prompt(self, content: str) -> AsyncIterator[AgentEvent]:
        """Run one prompt in this session."""

        async for event in self._runtime._prompt_session(self.id, content):
            yield event

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
