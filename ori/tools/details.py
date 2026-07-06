"""Result detail models for the built-in tools."""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel

from ori.tool_truncation import Truncation, TruncationKeep, TruncationReason
from ori.types.tools import ToolDetails

ToolTruncationReason: TypeAlias = TruncationReason
ToolTruncationKeep: TypeAlias = TruncationKeep


class ToolOutputDetails(BaseModel):
    """Structured metadata describing bounded tool output."""

    truncated: bool
    truncated_by: ToolTruncationReason | None
    keep: ToolTruncationKeep
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


class LsDetails(ToolDetails):
    """Directory listing metadata for UI and persistence."""

    type: Literal["ls"] = "ls"
    output: ToolOutputDetails


class GrepDetails(ToolDetails):
    """Search metadata for UI and persistence."""

    type: Literal["grep"] = "grep"
    output: ToolOutputDetails
    match_limit_reached: int | None = None
    lines_truncated: bool = False


class FindDetails(ToolDetails):
    """File path search metadata for UI and persistence."""

    type: Literal["find"] = "find"
    output: ToolOutputDetails


class ReadDetails(ToolDetails):
    """File read metadata for UI and persistence."""

    type: Literal["read"] = "read"
    output: ToolOutputDetails


class BashDetails(ToolDetails):
    """Shell command metadata for UI and persistence."""

    type: Literal["bash"] = "bash"
    output: ToolOutputDetails


class EditDetails(ToolDetails):
    """File edit metadata for UI and persistence."""

    type: Literal["edit"] = "edit"
    diff: str
