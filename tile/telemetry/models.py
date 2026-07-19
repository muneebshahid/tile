"""Serializable wide-event telemetry contracts for completed runs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from tile.result import RunOutcome
from tile.runs import TerminalRunStatus
from tile.types.usage import TokenUsage

LifecycleScopeType: TypeAlias = Literal[
    "run",
    "agent",
    "turn",
    "message",
    "tool_execution",
]
LifecycleScopeStatus: TypeAlias = Literal["completed", "interrupted"]
TelemetryErrorRole: TypeAlias = Literal["primary", "secondary"]
TelemetryErrorKind: TypeAlias = Literal["agent_failure", "exception"]
TelemetryErrorStage: TypeAlias = Literal[
    "submission",
    "turn",
    "execution",
    "run_persistence",
    "history_healing",
    "owner_release",
]
NonNegativeInt: TypeAlias = Annotated[int, Field(ge=0)]


class LifecycleScopeRecord(BaseModel):
    """One completed lifecycle scope within a run trace tree."""

    model_config = ConfigDict(frozen=True)

    scope_id: str
    parent_scope_id: str | None
    scope_type: LifecycleScopeType
    started_monotonic_ns: NonNegativeInt
    ended_monotonic_ns: NonNegativeInt
    status: LifecycleScopeStatus
    operation_name: str | None = None
    token_usage: TokenUsage | None = None

    @model_validator(mode="after")
    def _validate_scope(self) -> Self:
        """Reject backward time and token attribution to tool execution."""

        if self.ended_monotonic_ns < self.started_monotonic_ns:
            raise ValueError("A scope cannot end before it starts.")
        if self.scope_type == "tool_execution" and self.token_usage is not None:
            raise ValueError("A tool-execution scope cannot carry token usage.")
        return self


class ToolAggregate(BaseModel):
    """Run-level timing and outcome totals for one tool name."""

    model_config = ConfigDict(frozen=True)

    tool_name: str
    call_count: NonNegativeInt
    completed_count: NonNegativeInt
    error_count: NonNegativeInt
    total_duration_ns: NonNegativeInt

    @model_validator(mode="after")
    def _validate_counts(self) -> Self:
        """Keep completion and error counts within total tool calls."""

        if self.completed_count > self.call_count:
            raise ValueError("Completed count cannot exceed call count.")
        if self.error_count > self.completed_count:
            raise ValueError("Error count cannot exceed completed count.")
        return self


class RunTelemetryError(BaseModel):
    """One serializable primary or secondary run failure."""

    model_config = ConfigDict(frozen=True)

    role: TelemetryErrorRole
    stage: TelemetryErrorStage
    kind: TelemetryErrorKind
    message: str
    exception_type: str | None = None

    @model_validator(mode="after")
    def _validate_exception_type(self) -> Self:
        """Match exception identity to the error kind."""

        if self.kind == "exception" and self.exception_type is None:
            raise ValueError("An exception error requires exception_type.")
        if self.kind == "agent_failure" and self.exception_type is not None:
            raise ValueError("An agent failure cannot carry exception_type.")
        return self


class RunTelemetryRecord(BaseModel):
    """Canonical serializable wide event for one finalized run."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    session_id: str
    status: TerminalRunStatus
    outcome: RunOutcome = Field(discriminator="type")
    started_at: datetime
    started_monotonic_ns: NonNegativeInt
    ended_monotonic_ns: NonNegativeInt
    duration_ns: NonNegativeInt
    provider: str | None
    model: str | None
    turn_count: NonNegativeInt
    token_usage: TokenUsage | None
    tools: tuple[ToolAggregate, ...] = ()
    scopes: tuple[LifecycleScopeRecord, ...] = ()
    errors: tuple[RunTelemetryError, ...] = ()
    context_receipt: str | None = None

    @field_validator("started_at")
    @classmethod
    def _normalize_started_at(cls, value: datetime) -> datetime:
        """Require a timezone-aware start anchor and normalize it to UTC."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("The telemetry start time must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _validate_duration(self) -> Self:
        """Keep the stored duration equal to the monotonic run interval."""

        if self.ended_monotonic_ns < self.started_monotonic_ns:
            raise ValueError("A run cannot end before it starts.")
        interval = self.ended_monotonic_ns - self.started_monotonic_ns
        if self.duration_ns != interval:
            raise ValueError("Run duration must equal its monotonic interval.")
        return self
