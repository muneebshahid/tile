"""Durable run-summary contracts and repository boundary."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal, Protocol, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from tile.result import Aborted, AgentFailure, Completed, RunOutcome

RunStatus: TypeAlias = Literal["running", "completed", "failed", "aborted"]
TerminalRunStatus: TypeAlias = Literal["completed", "failed", "aborted"]


def _terminal_status_for(outcome: RunOutcome) -> TerminalRunStatus:
    """Return the execution status implied by one terminal outcome.

    An agent-declared failure still executed normally, so it keeps
    ``completed``; only an execution-failure cause means ``failed``.
    """

    if isinstance(outcome, Completed):
        return "completed"
    if isinstance(outcome, Aborted):
        return "aborted"
    if isinstance(outcome.cause, AgentFailure):
        return "completed"
    return "failed"


class RunRecord(BaseModel):
    """Durable summary of one prompt run and its terminal result."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    session_id: str
    status: RunStatus
    started_at: datetime
    ended_at: datetime | None = None
    model: str
    provider: str | None = None
    outcome: RunOutcome | None = None

    def finish(
        self,
        *,
        outcome: RunOutcome,
        provider: str | None = None,
        model: str | None = None,
    ) -> RunRecord:
        """Return a terminal form of this record ending now.

        The terminal status is derived from the outcome variant, so the two
        fields cannot deviate. The end timestamp is stamped internally and
        clamped to the start, so a backward clock step cannot produce a
        record that ends before it starts.
        """

        return RunRecord(
            run_id=self.run_id,
            session_id=self.session_id,
            status=_terminal_status_for(outcome),
            started_at=self.started_at,
            ended_at=max(datetime.now(UTC), self.started_at),
            model=model if model is not None else self.model,
            provider=provider if provider is not None else self.provider,
            outcome=outcome,
        )

    @field_validator("started_at", "ended_at")
    @classmethod
    def _normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        """Require timezone-aware timestamps and normalize them to UTC."""

        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Run timestamps must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _validate_lifecycle(self) -> Self:
        """Reject combinations that contradict the current run lifecycle."""

        if self.status == "running":
            if self.ended_at is not None or self.outcome is not None:
                raise ValueError("A running run cannot have terminal data.")
            return self

        if self.ended_at is None:
            raise ValueError("A terminal run must have an end timestamp.")
        if self.ended_at < self.started_at:
            raise ValueError("A run cannot end before it starts.")
        if self.outcome is None:
            raise ValueError("A terminal run must have an outcome.")
        implied_status = _terminal_status_for(self.outcome)
        if self.status != implied_status:
            raise ValueError(
                f"Status {self.status!r} contradicts the terminal outcome, "
                f"which implies {implied_status!r}."
            )
        return self


class RunNotFoundError(KeyError):
    """Raised when a run operation references an unknown run."""


class RunAlreadyExistsError(ValueError):
    """Raised when creating a run record whose id already exists."""


class RunStore(Protocol):
    """Stores durable run summaries separately from conversation history."""

    def create_run(self, record: RunRecord) -> None:
        """Persist a newly submitted running record."""
        ...

    def update_run(self, record: RunRecord) -> None:
        """Replace an existing run record with its latest state."""
        ...

    def get_run(self, run_id: str) -> RunRecord:
        """Return a run record by its stable id."""
        ...

    def list_runs(self, session_id: str) -> Sequence[RunRecord]:
        """Return run records for one session in submission order."""
        ...
