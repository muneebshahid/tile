"""Contract tests for public wide-event run telemetry models and sinks."""

from collections.abc import Sequence
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tile import (
    CapturedRunException,
    Completed,
    ExecutionFailure,
    Failed,
    LifecycleScopeRecord,
    LifecycleScopeType,
    RunTelemetryError,
    RunTelemetryRecord,
    RunTelemetrySink,
    TokenUsage,
    ToolAggregate,
)
from tile.runs import TerminalRunStatus


def test_run_telemetry_record_round_trips_through_json() -> None:
    """Preserve nested discriminated contracts through JSON serialization."""

    record = _record()

    restored = RunTelemetryRecord.model_validate_json(record.model_dump_json())

    assert restored == record
    assert isinstance(restored.outcome, Completed)
    assert restored.scopes[0].token_usage == _usage()


def test_failed_outcome_and_structured_error_round_trip() -> None:
    """Retain the typed primary failure alongside structured telemetry errors."""

    failure = ExecutionFailure(
        origin="execution",
        exception_type="ConnectionError",
        message="connection reset",
    )
    error = RunTelemetryError(
        role="primary",
        stage="execution",
        kind="exception",
        exception_type="ConnectionError",
        message="connection reset",
    )
    record = _record(
        status="failed",
        outcome=Failed(cause=failure),
        errors=(error,),
    )

    restored = RunTelemetryRecord.model_validate_json(record.model_dump_json())

    assert isinstance(restored.outcome, Failed)
    assert isinstance(restored.outcome.cause, ExecutionFailure)
    assert restored.errors == (error,)


def test_telemetry_models_are_frozen() -> None:
    """Reject mutation after a telemetry contract has been constructed."""

    record = _record()

    with pytest.raises(ValidationError, match="frozen"):
        record.run_id = "replacement"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_tokens", -1),
        ("output_tokens", -1),
        ("total_tokens", -1),
        ("cached_input_tokens", -1),
        ("reasoning_output_tokens", -1),
    ],
)
def test_token_usage_rejects_negative_counts(field: str, value: int) -> None:
    """Require every provider token counter to be non-negative."""

    values = _usage().model_dump()
    values[field] = value

    with pytest.raises(ValidationError):
        TokenUsage.model_validate(values)


def test_scope_rejects_end_before_start() -> None:
    """Reject a lifecycle duration that moves backward."""

    with pytest.raises(ValidationError, match="before"):
        _scope(started_monotonic_ns=20, ended_monotonic_ns=10)


def test_tool_scope_rejects_token_usage() -> None:
    """Keep provider token attribution off tool-execution scopes."""

    with pytest.raises(ValidationError, match="tool"):
        _scope(scope_type="tool_execution", token_usage=_usage())


@pytest.mark.parametrize(
    ("call_count", "completed_count", "error_count"),
    [
        (1, 2, 0),
        (1, 1, 2),
    ],
)
def test_tool_aggregate_rejects_impossible_counts(
    call_count: int,
    completed_count: int,
    error_count: int,
) -> None:
    """Keep completed and error totals within the number of calls."""

    with pytest.raises(ValidationError, match="count"):
        ToolAggregate(
            tool_name="bash",
            call_count=call_count,
            completed_count=completed_count,
            error_count=error_count,
            total_duration_ns=10,
        )


def test_exception_error_requires_exception_type() -> None:
    """Require serializable exception identity for exception-backed failures."""

    with pytest.raises(ValidationError, match="exception_type"):
        RunTelemetryError(
            role="primary",
            stage="execution",
            kind="exception",
            message="boom",
        )


def test_record_rejects_naive_start_time() -> None:
    """Require a timezone-aware wall-clock anchor for monotonic offsets."""

    with pytest.raises(ValidationError, match="timezone-aware"):
        _record(started_at=datetime(2026, 7, 19))


def test_record_rejects_inconsistent_duration() -> None:
    """Require duration to equal the monotonic run interval."""

    with pytest.raises(ValidationError, match="duration"):
        _record(duration_ns=99)


def test_original_exception_is_a_non_serialized_sink_sidecar() -> None:
    """Give exception-aware sinks the traceback without polluting the record."""

    exception = RuntimeError("private traceback")
    captured = CapturedRunException(
        role="primary",
        stage="execution",
        error=exception,
    )
    sink: RunTelemetrySink = _CollectingSink()

    sink.emit(_record(), exceptions=(captured,))
    collecting_sink = sink
    assert isinstance(collecting_sink, _CollectingSink)
    assert collecting_sink.exceptions == (captured,)
    assert "private traceback" not in collecting_sink.record.model_dump_json()


class _CollectingSink:
    """Test sink retaining one public telemetry emission."""

    def __init__(self) -> None:
        """Create an empty collecting sink."""

        self.record = _record()
        self.exceptions: tuple[CapturedRunException, ...] = ()

    def emit(
        self,
        record: RunTelemetryRecord,
        *,
        exceptions: Sequence[CapturedRunException],
    ) -> None:
        """Retain defensive tuple forms of the supplied contracts."""

        self.record = record
        self.exceptions = tuple(exceptions)


def _record(
    *,
    status: TerminalRunStatus = "completed",
    outcome: Completed | Failed | None = None,
    started_at: datetime | None = None,
    duration_ns: int = 80,
    errors: tuple[RunTelemetryError, ...] = (),
) -> RunTelemetryRecord:
    """Build a representative telemetry record."""

    return RunTelemetryRecord(
        run_id="run-1",
        session_id="session-1",
        status=status,
        outcome=outcome if outcome is not None else Completed(value="done"),
        started_at=started_at if started_at is not None else datetime.now(UTC),
        started_monotonic_ns=20,
        ended_monotonic_ns=100,
        duration_ns=duration_ns,
        provider="test",
        model="test-model",
        turn_count=1,
        token_usage=_usage(),
        tools=(
            ToolAggregate(
                tool_name="bash",
                call_count=1,
                completed_count=1,
                error_count=0,
                total_duration_ns=10,
            ),
        ),
        scopes=(_scope(),),
        errors=errors,
    )


def _scope(
    *,
    scope_type: LifecycleScopeType = "message",
    started_monotonic_ns: int = 20,
    ended_monotonic_ns: int = 30,
    token_usage: TokenUsage | None = None,
) -> LifecycleScopeRecord:
    """Build one representative lifecycle scope."""

    return LifecycleScopeRecord(
        scope_id="message-1",
        parent_scope_id="turn-1",
        scope_type=scope_type,
        started_monotonic_ns=started_monotonic_ns,
        ended_monotonic_ns=ended_monotonic_ns,
        status="completed",
        token_usage=token_usage if token_usage is not None else _usage(),
    )


def _usage() -> TokenUsage:
    """Build representative provider token usage."""

    return TokenUsage(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        cached_input_tokens=3,
        reasoning_output_tokens=2,
    )
