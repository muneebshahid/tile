"""Session: a scoped handle for prompting one conversation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel

from tile.history import SessionRecord
from tile.runtime.run import Run
from tile.types.conversation import ConversationItem

if TYPE_CHECKING:
    from tile.runtime.runtime import AgentRuntime


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
