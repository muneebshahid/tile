"""Delivery contracts for finalized run telemetry."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from tile.telemetry.models import (
    RunTelemetryRecord,
    TelemetryErrorRole,
    TelemetryErrorStage,
)


@dataclass(frozen=True)
class CapturedRunException:
    """Original in-process exception retained for exception-aware sinks."""

    role: TelemetryErrorRole
    stage: TelemetryErrorStage
    error: BaseException


class RunTelemetrySink(Protocol):
    """Receives one finalized telemetry record and exception sidecars."""

    def emit(
        self,
        record: RunTelemetryRecord,
        *,
        exceptions: Sequence[CapturedRunException],
    ) -> None:
        """Accept one finalized run telemetry emission."""
        ...
