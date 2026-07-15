"""In-memory durable run-summary repository implementation."""

from dataclasses import dataclass, field

from tile.runs.base import (
    RunAlreadyExistsError,
    RunNotFoundError,
    RunRecord,
)


@dataclass
class InMemoryRunStore:
    """Store defensive run-record snapshots in process memory."""

    _records_by_id: dict[str, RunRecord] = field(default_factory=dict)

    def create_run(self, record: RunRecord) -> None:
        """Persist a newly submitted running record."""

        if record.run_id in self._records_by_id:
            raise RunAlreadyExistsError(f"Run already exists: {record.run_id}")
        self._records_by_id[record.run_id] = _copy_record(record)

    def update_run(self, record: RunRecord) -> None:
        """Replace an existing run record with its latest state."""

        self._require_run(record.run_id)
        self._records_by_id[record.run_id] = _copy_record(record)

    def get_run(self, run_id: str) -> RunRecord:
        """Return a defensive snapshot of a run record by id."""

        self._require_run(run_id)
        return _copy_record(self._records_by_id[run_id])

    def list_runs(self, session_id: str) -> tuple[RunRecord, ...]:
        """Return defensive snapshots for one session in submission order."""

        return tuple(
            _copy_record(record)
            for record in self._records_by_id.values()
            if record.session_id == session_id
        )

    def _require_run(self, run_id: str) -> None:
        """Raise a domain lookup error when a run id is unknown."""

        if run_id not in self._records_by_id:
            raise RunNotFoundError(f"Unknown run: {run_id}")


def _copy_record(record: RunRecord) -> RunRecord:
    """Return a defensive deep copy of one run record."""

    return record.model_copy(deep=True)
