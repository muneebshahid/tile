"""Shared data types for tool output helpers and contracts."""

from dataclasses import dataclass
from typing import Literal

TruncationReason = Literal["lines", "bytes"]
TruncationKeep = Literal["head", "tail"]


@dataclass(frozen=True)
class Truncation:
    """Metadata returned when keeping one edge of tool output."""

    content: str
    truncated: bool
    truncated_by: TruncationReason | None
    keep: TruncationKeep
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    edge_line_exceeds_limit: bool
    max_lines: int
    max_bytes: int
