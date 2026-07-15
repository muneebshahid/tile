"""Durable run-summary contracts and repository boundary."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal, Protocol, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from tile.result import RunOutcome

RunStatus: TypeAlias = Literal["running", "completed", "failed", "aborted"]
RunFailureOrigin: TypeAlias = Literal["turn", "execution", "finalization"]
TerminalRunStatus: TypeAlias = Literal["completed", "failed", "aborted"]


class RunFailure(BaseModel):
    """Serializable diagnostics for a failed run execution."""

    type: Literal["run_failure"] = "run_failure"
    origin: RunFailureOrigin
    exception_type: str
    message: str


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
    failure: RunFailure | None = None

    def finish(
        self,
        *,
        status: TerminalRunStatus,
        ended_at: datetime,
        provider: str | None = None,
        model: str | None = None,
        outcome: RunOutcome | None = None,
        failure: RunFailure | None = None,
    ) -> RunRecord:
        """Return a terminal form of this record."""

        return RunRecord(
            run_id=self.run_id,
            session_id=self.session_id,
            status=status,
            started_at=self.started_at,
            ended_at=ended_at,
            model=model if model is not None else self.model,
            provider=provider,
            outcome=outcome,
            failure=failure,
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
            if self.failure is not None:
                raise ValueError("A running run cannot have a failure.")
            return self

        if self.ended_at is None:
            raise ValueError("A terminal run must have an end timestamp.")
        if self.ended_at < self.started_at:
            raise ValueError("A run cannot end before it starts.")
        if self.status == "failed" and self.failure is None:
            raise ValueError("A failed run must have failure diagnostics.")
        if self.status != "failed" and self.failure is not None:
            raise ValueError("Only a failed run can have failure diagnostics.")
        if self.status != "completed" and self.outcome is not None:
            raise ValueError("Only a completed run can have an outcome.")
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
