"""Shared data types for tool output helpers and contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

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


class ToolOutputDetails(BaseModel):
    """Structured metadata describing bounded tool output."""

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

    @classmethod
    def from_truncation(
        cls,
        truncation: Truncation,
    ) -> ToolOutputDetails:
        """Create output details from matching truncation metadata."""

        return cls(
            truncated=truncation.truncated,
            truncated_by=truncation.truncated_by,
            keep=truncation.keep,
            total_lines=truncation.total_lines,
            total_bytes=truncation.total_bytes,
            output_lines=truncation.output_lines,
            output_bytes=truncation.output_bytes,
            edge_line_exceeds_limit=truncation.edge_line_exceeds_limit,
            max_lines=truncation.max_lines,
            max_bytes=truncation.max_bytes,
        )
