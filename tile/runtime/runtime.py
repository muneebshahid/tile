"""AgentRuntime: configuration and orchestration for many sessions.

The runtime constructs shared dependencies, manages sessions and the
per-session prompt reservation, and starts runs. Everything run-scoped —
persistence, events, healing — belongs to ``Run``; the runtime keeps
store references only for its query facade.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import cast
from uuid import uuid4

from pydantic import BaseModel

from tile.events import StreamFn
from tile.history import HistoryStore, SessionRecord
from tile.prompt import DEFAULT_INSTRUCTIONS
from tile.result import COMPLETE_TOOL_NAME, FAIL_TOOL_NAME
from tile.runs import RunRecord, RunStore
from tile.runtime.execution import _ExecutionDependencies
from tile.runtime.run import Run, _RunDependencies, _RunSpec
from tile.runtime.session import Session
from tile.tool_executor import ToolExecutor
from tile.tools.support.paths import normalize_cwd
from tile.types.conversation import ConversationItem
from tile.types.tools import ToolDefinition, ToolFunction


class SessionBusyError(RuntimeError):
    """Raised when a prompt is submitted while the same session is already active."""


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
        normalized_cwd = normalize_cwd(cwd)
        self._history_store = history_store
        self._run_store = run_store
        self._deps = _RunDependencies(
            execution=_ExecutionDependencies(
                stream_fn=stream_fn,
                model=model,
                instructions=instructions,
                cwd=normalized_cwd,
                auto_mode=auto_mode,
                tool_executor=ToolExecutor(_bind_cwd_tools(tools, normalized_cwd)),
                history_store=history_store,
            ),
            history_store=history_store,
            run_store=run_store,
        )
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
        """Reserve the session, then start one run that owns the prompt.

        The run performs its own submission persistence and unwinding;
        the runtime's only cleanup on a failed construction is releasing
        the reservation it took.
        """

        self._start_prompt(session_id)
        try:
            run = Run(
                spec=_RunSpec(session_id=session_id, content=content, result=result),
                deps=self._deps,
                on_finished=self._release_run,
            )
        except BaseException:
            self._finish_prompt(session_id)
            raise
        self._active_runs.add(run)
        return run

    def _release_run(self, run: Run) -> None:
        """Release the session reservation and forget the finished run."""

        self._finish_prompt(run.session_id)
        self._active_runs.discard(run)

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

    def _build_session(self, record: SessionRecord) -> Session:
        """Build a session handle from a stored record."""

        return Session(_record=record, _runtime=self)

    def _resolve_session_id(self, session_id: str | None) -> str:
        """Return the provided session id or generate a new one."""

        if session_id is not None:
            return session_id
        return str(uuid4())


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
